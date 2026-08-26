"""Per-QoI target transforms for S2 (log and logit warps)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import math
import torch

from experiments_toa.s2_constants import (
    S2_LOG_OFFSETS,
    S2_LOG_SCALE_TASK_NAMES,
    S2_LOGIT_BOUNDS,
    S2_TASK_NAMES,
    S4_TASK_NAMES,
)
from gpplus.utils import StandardScaler

_WARP_TASK_NAMES = frozenset(S2_TASK_NAMES) | frozenset(S4_TASK_NAMES)
_LOG_Y_STD_CAP_FOR_LOGNORMAL_MEAN = 3.0
_NORMAL_Z_95 = 1.96
_LOGIT_EPS = 1e-4

# Fixed normal quantiles for deterministic logit-normal mean/std on unit interval.
_LOGIT_NORMAL_QUANTILE_GRID = torch.tensor(
    [-2.576, -1.96, -1.0, 0.0, 1.0, 1.96, 2.576],
    dtype=torch.float64,
)


@dataclass
class InverseYOutput:
    """Original-scale predictions after inverse target transform."""

    point: torch.Tensor
    std: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    point_mean: torch.Tensor | None = None
    point_mode: torch.Tensor | None = None
    log_mu: torch.Tensor | None = None
    log_sigma: torch.Tensor | None = None
    logit_mu: torch.Tensor | None = None
    logit_sigma: torch.Tensor | None = None

    def as_tuple(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.point, self.std, self.lower, self.upper


@dataclass(frozen=True)
class YWarpConfig:
    """Resolved per-QoI output warps for one S2 run."""

    log_tasks: frozenset[str] = frozenset()
    logit_tasks: frozenset[str] = frozenset()
    log_source: str = "none"
    logit_source: str = "none"
    logit_bounds: Mapping[str, tuple[float, float]] = field(
        default_factory=lambda: dict(S2_LOGIT_BOUNDS)
    )
    # Per-task additive C in log(y + C); missing → 0 (bare log).
    log_offsets: Mapping[str, float] = field(default_factory=dict)

    @property
    def log_scale(self) -> bool:
        return bool(self.log_tasks)

    def uses_log(self, task_name: str) -> bool:
        return task_name in self.log_tasks

    def uses_logit(self, task_name: str) -> bool:
        return task_name in self.logit_tasks

    def log_offset(self, task_name: str) -> float:
        return float(self.log_offsets.get(task_name, 0.0))

    def active_log_offsets(self) -> dict[str, float]:
        """Offsets C>0 for currently active log tasks."""
        return {
            name: self.log_offset(name)
            for name in sorted(self.log_tasks)
            if self.log_offset(name) > 0.0
        }


def _merge_log_offsets(
    overrides: Mapping[str, float] | None,
) -> dict[str, float]:
    offsets = {k: float(v) for k, v in S2_LOG_OFFSETS.items()}
    if overrides:
        for name, value in overrides.items():
            c = float(value)
            if not math.isfinite(c) or c < 0.0:
                raise ValueError(
                    f"log offset for {name!r} must be a finite value >= 0, got {value!r}"
                )
            offsets[str(name)] = c
    return offsets


def _attr_to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def _attr_to_bool(value: Any) -> bool | None:
    """Parse a NetCDF/HDF5 attribute as bool; return None if not interpretable."""
    if value is None:
        return None
    if isinstance(value, bool):
        return True if value else False
    if hasattr(value, "item") and not isinstance(value, (str, bytes, bool)):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value in (0, 1):
            return bool(value)
        return None
    text = _attr_to_str(value).lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n", ""}:
        return False
    return None


def parse_log_uniform_qois(raw: Any) -> frozenset[str]:
    """Parse ``log_uniform_qois`` attr into known S2 task names."""
    text = _attr_to_str(raw)
    if not text or text.lower() in {"none", "null", "n/a"}:
        return frozenset()
    names = {
        part.strip()
        for part in text.replace(";", ",").split(",")
        if part.strip()
    }
    return frozenset(n for n in names if n in _WARP_TASK_NAMES)


def _normalize_task_list(
    tasks: Sequence[str] | None,
    *,
    label: str,
) -> frozenset[str]:
    if tasks is None:
        return frozenset()
    names = [str(t).strip() for t in tasks if str(t).strip()]
    unknown = sorted({n for n in names if n not in _WARP_TASK_NAMES})
    if unknown:
        raise ValueError(f"Unknown S2 task names in {label}: {unknown}")
    return frozenset(names)


def _merge_logit_bounds(
    overrides: Mapping[str, tuple[float, float]] | None,
) -> dict[str, tuple[float, float]]:
    bounds = dict(S2_LOGIT_BOUNDS)
    if overrides:
        for name, pair in overrides.items():
            if name not in _WARP_TASK_NAMES:
                raise ValueError(f"Unknown S2 task name in logit bounds: {name!r}")
            a, b = float(pair[0]), float(pair[1])
            if not (b > a):
                raise ValueError(
                    f"Logit bounds for {name!r} must satisfy b > a, got ({a}, {b})"
                )
            bounds[name] = (a, b)
    return bounds


def infer_log_scale_from_attrs(
    attrs: Mapping[str, Any] | None,
) -> tuple[bool, frozenset[str], str]:
    """
    Infer whether QoI targets should use log scaling from NetCDF attrs.

    Priority:
    1. ``output_log_scale`` (explicit bool-like)
    2. non-empty ``log_uniform_qois`` → True for those tasks (else default log set)
    3. otherwise False (linear / physical outputs)

    Returns
    -------
    log_scale, log_scale_tasks, source
    """
    attrs = dict(attrs or {})
    explicit = _attr_to_bool(attrs.get("output_log_scale"))
    log_qois = parse_log_uniform_qois(attrs.get("log_uniform_qois"))

    if explicit is True:
        tasks = log_qois or frozenset(S2_LOG_SCALE_TASK_NAMES)
        return True, tasks, "attr:output_log_scale=true"
    if explicit is False:
        return False, frozenset(), "attr:output_log_scale=false"
    if log_qois:
        return True, log_qois, "attr:log_uniform_qois"
    return False, frozenset(), "attr:default_false"


def resolve_log_scale(
    log_scale: bool | None,
    *,
    meta: Mapping[str, Any] | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> tuple[bool, frozenset[str], str]:
    """
    Resolve master log-scale switch.

    ``None`` means auto-detect from dataset ``meta`` / NetCDF ``attrs``.
    Explicit True/False always wins.
    """
    if log_scale is not None:
        flag = bool(log_scale)
        if flag:
            tasks = frozenset(S2_LOG_SCALE_TASK_NAMES)
            if meta is not None:
                raw = meta.get("log_scale_tasks")
                if raw:
                    tasks = frozenset(str(x) for x in raw) or tasks
            return flag, tasks, "explicit"
        return False, frozenset(), "explicit"

    if meta is not None and "log_scale" in meta and meta.get("log_scale_source"):
        tasks = frozenset(str(x) for x in (meta.get("log_scale_tasks") or []))
        return bool(meta["log_scale"]), tasks, str(meta["log_scale_source"])

    src_attrs = attrs
    if src_attrs is None and meta is not None:
        src_attrs = meta.get("dataset_attrs")  # type: ignore[assignment]
    return infer_log_scale_from_attrs(src_attrs)


def resolve_y_warps(
    log_scale_qoi: Sequence[str] | None = None,
    logit_scale_qoi: Sequence[str] | None = None,
    *,
    log_scale: bool | None = None,
    meta: Mapping[str, Any] | None = None,
    attrs: Mapping[str, Any] | None = None,
    logit_bounds: Mapping[str, tuple[float, float]] | None = None,
    log_offsets: Mapping[str, float] | None = None,
) -> YWarpConfig:
    """
    Resolve per-QoI log / logit warps.

    Precedence for log tasks:
    1. Explicit ``log_scale_qoi`` list (including empty → no log warps)
    2. Legacy ``log_scale`` bool / NetCDF auto via :func:`resolve_log_scale`

    Precedence for logit tasks:
    1. Explicit ``logit_scale_qoi`` list (including empty → no logit warps)
    2. Otherwise empty (no NetCDF auto for logit)

    Log warps use ``log(y + C)`` with per-task ``C`` from ``S2_LOG_OFFSETS``
    (overridable via ``log_offsets``). Missing tasks use ``C=0`` (bare ``log(y)``).
    """
    bounds = _merge_logit_bounds(logit_bounds)
    offsets = _merge_log_offsets(log_offsets)

    if log_scale_qoi is not None:
        log_tasks = _normalize_task_list(log_scale_qoi, label="LOG_SCALE_QOI")
        log_source = "explicit_list"
    else:
        flag, log_tasks, log_source = resolve_log_scale(log_scale, meta=meta, attrs=attrs)
        if not flag:
            log_tasks = frozenset()

    if logit_scale_qoi is not None:
        logit_tasks = _normalize_task_list(logit_scale_qoi, label="LOGIT_SCALE_QOI")
        logit_source = "explicit_list"
    else:
        logit_tasks = frozenset()
        logit_source = "default_empty"

    overlap = sorted(log_tasks & logit_tasks)
    if overlap:
        raise ValueError(
            "A QoI cannot use both log and logit warps; overlap="
            f"{overlap}. Put each task in at most one of LOG_SCALE_QOI / LOGIT_SCALE_QOI."
        )

    missing_bounds = sorted(n for n in logit_tasks if n not in bounds)
    if missing_bounds:
        raise ValueError(
            "Logit warps require affine bounds [a, b] for each task. Missing bounds for "
            f"{missing_bounds}. Pass logit_bounds=... or add defaults in S2_LOGIT_BOUNDS."
        )

    return YWarpConfig(
        log_tasks=log_tasks,
        logit_tasks=logit_tasks,
        log_source=log_source,
        logit_source=logit_source,
        logit_bounds=bounds,
        log_offsets=offsets,
    )


def task_uses_log_scale(
    task_name: str,
    *,
    log_scale: bool = True,
    log_scale_tasks: frozenset[str] | set[str] | None = None,
    warps: YWarpConfig | None = None,
) -> bool:
    """Return True when ``task_name`` should be trained in log space."""
    if warps is not None:
        return warps.uses_log(task_name)
    if not log_scale:
        return False
    allowed = (
        frozenset(log_scale_tasks)
        if log_scale_tasks is not None
        else frozenset(S2_LOG_SCALE_TASK_NAMES)
    )
    return task_name in allowed


def task_uses_logit_scale(
    task_name: str,
    *,
    logit_scale_tasks: frozenset[str] | set[str] | None = None,
    warps: YWarpConfig | None = None,
) -> bool:
    """Return True when ``task_name`` should be trained in logit space."""
    if warps is not None:
        return warps.uses_logit(task_name)
    if logit_scale_tasks is None:
        return False
    return task_name in frozenset(logit_scale_tasks)


def logit_bounds_for_task(
    task_name: str,
    *,
    warps: YWarpConfig | None = None,
    logit_bounds: Mapping[str, tuple[float, float]] | None = None,
) -> tuple[float, float]:
    """Return affine [a, b] used before logit for ``task_name``."""
    if warps is not None:
        if task_name not in warps.logit_bounds:
            raise KeyError(f"No logit bounds for task {task_name!r}")
        return tuple(warps.logit_bounds[task_name])  # type: ignore[return-value]
    bounds = _merge_logit_bounds(logit_bounds)
    if task_name not in bounds:
        raise KeyError(f"No logit bounds for task {task_name!r}")
    return bounds[task_name]


def _to_unit_interval(
    y: torch.Tensor,
    *,
    a: float,
    b: float,
) -> torch.Tensor:
    return (y - a) / (b - a)


def _from_unit_interval(
    u: torch.Tensor,
    *,
    a: float,
    b: float,
) -> torch.Tensor:
    return a + (b - a) * u


def _clamp_unit_for_logit(u: torch.Tensor) -> torch.Tensor:
    return u.clamp(_LOGIT_EPS, 1.0 - _LOGIT_EPS)


def _log_offset_for_task(task_name: str, warps: YWarpConfig | None) -> float:
    if warps is not None:
        return float(warps.log_offset(task_name))
    return float(S2_LOG_OFFSETS.get(task_name, 0.0))


def forward_y_s2(
    y: torch.Tensor,
    task_name: str,
    *,
    log_scale: bool = True,
    log_scale_tasks: frozenset[str] | set[str] | None = None,
    logit_scale_tasks: frozenset[str] | set[str] | None = None,
    warps: YWarpConfig | None = None,
) -> torch.Tensor:
    """Apply forward target transform for one S2 QoI (log or logit)."""
    if task_uses_log_scale(
        task_name,
        log_scale=log_scale,
        log_scale_tasks=log_scale_tasks,
        warps=warps,
    ):
        c = _log_offset_for_task(task_name, warps)
        if c > 0.0:
            if torch.any(y <= -c):
                raise ValueError(
                    f"log(y+{c:g}) scaling for {task_name!r} requires all targets > {-c:g}."
                )
            return torch.log(y + c)
        if torch.any(y <= 0):
            raise ValueError(
                f"log scaling for {task_name!r} requires all targets to be strictly positive."
            )
        return torch.log(y)

    if task_uses_logit_scale(
        task_name, logit_scale_tasks=logit_scale_tasks, warps=warps
    ):
        a, b = logit_bounds_for_task(
            task_name,
            warps=warps,
        )
        u = _to_unit_interval(y, a=a, b=b)
        if torch.any(u < 0.0) or torch.any(u > 1.0):
            raise ValueError(
                f"logit scaling for {task_name!r} requires all targets in [{a}, {b}]."
            )
        return torch.logit(_clamp_unit_for_logit(u))

    return y


def _unstandardize(
    pred_mean: torch.Tensor,
    pred_std: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    y_scaler: StandardScaler | None,
    standardize_y: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_mean = pred_mean.clone()
    pred_std = pred_std.clone()
    lower = lower.clone()
    upper = upper.clone()
    if standardize_y and y_scaler is not None:
        y_mean = y_scaler.mean.squeeze()
        y_std = y_scaler.std.squeeze()
        pred_mean = pred_mean * y_std + y_mean
        pred_std = pred_std * y_std
        lower = lower * y_std + y_mean
        upper = upper * y_std + y_mean
    return pred_mean, pred_std, lower, upper


def _lognormal_original_scale(
    mu_log: torch.Tensor,
    sigma_log: torch.Tensor,
    *,
    z: float = _NORMAL_Z_95,
    offset: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map log-space Normal(μ, σ²) on ``log(y+C)`` to summaries on original ``y`` scale."""
    c = float(offset)
    sigma_log = torch.clamp(sigma_log, min=0.0)
    median_yc = torch.exp(mu_log)
    sigma_for_mean = torch.clamp(sigma_log, max=_LOG_Y_STD_CAP_FOR_LOGNORMAL_MEAN)
    mean_yc = median_yc * torch.exp(0.5 * sigma_for_mean**2)
    mode_yc = torch.exp(mu_log - sigma_log**2)
    std = median_yc * torch.sqrt(torch.expm1(sigma_log**2))
    lower_yc = torch.exp(mu_log - z * sigma_log)
    upper_yc = torch.exp(mu_log + z * sigma_log)
    if c != 0.0:
        # y = (y+C) - C; clamp at 0 for non-negative physical QoIs.
        median = torch.clamp(median_yc - c, min=0.0)
        mean = torch.clamp(mean_yc - c, min=0.0)
        mode = torch.clamp(mode_yc - c, min=0.0)
        lower = torch.clamp(lower_yc - c, min=0.0)
        upper = torch.clamp(upper_yc - c, min=0.0)
    else:
        median, mean, mode = median_yc, mean_yc, mode_yc
        lower, upper = lower_yc, upper_yc
    return median, mean, mode, std, lower, upper


def _logit_normal_mode_unit(
    mu_logit: torch.Tensor,
    sigma_logit: torch.Tensor,
    *,
    n_grid: int = 2048,
) -> torch.Tensor:
    """Mode of logit-normal on (0, 1) via dense grid argmax of log-density."""
    sigma = torch.clamp(sigma_logit, min=1e-12)
    # Cover ~±6σ in logit space, then evaluate density on the unit interval.
    z_lo = mu_logit - 6.0 * sigma
    z_hi = mu_logit + 6.0 * sigma
    # Build a shared unit grid and evaluate each sample's log-pdf.
    u = torch.linspace(
        _LOGIT_EPS,
        1.0 - _LOGIT_EPS,
        n_grid,
        device=mu_logit.device,
        dtype=mu_logit.dtype,
    )
    logit_u = torch.log(u / (1.0 - u))
    # log pdf ∝ -0.5*((logit(u)-μ)/σ)^2 - log(u) - log(1-u)
    # Broadcast: (..., 1) vs (n_grid,)
    mu = mu_logit.unsqueeze(-1)
    sig = sigma.unsqueeze(-1)
    log_pdf = -0.5 * ((logit_u - mu) / sig) ** 2 - torch.log(u) - torch.log(1.0 - u)
    idx = torch.argmax(log_pdf, dim=-1)
    return u[idx]


def _logit_normal_unit_interval(
    mu_logit: torch.Tensor,
    sigma_logit: torch.Tensor,
    *,
    z: float = _NORMAL_Z_95,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Map logit-space Normal(μ, σ²) to summaries on [0, 1].

    Returns ``median, mean, mode, std, lower, upper``.
    """
    sigma_logit = torch.clamp(sigma_logit, min=0.0)
    median = torch.sigmoid(mu_logit).clamp(0.0, 1.0)
    lower = torch.sigmoid(mu_logit - z * sigma_logit).clamp(0.0, 1.0)
    upper = torch.sigmoid(mu_logit + z * sigma_logit).clamp(0.0, 1.0)

    grid = _LOGIT_NORMAL_QUANTILE_GRID.to(
        device=mu_logit.device,
        dtype=mu_logit.dtype,
    )
    z_samples = mu_logit.unsqueeze(-1) + sigma_logit.unsqueeze(-1) * grid
    u_samples = torch.sigmoid(z_samples).clamp(0.0, 1.0)
    mean = u_samples.mean(dim=-1)
    std = u_samples.std(dim=-1, unbiased=False)
    mode = _logit_normal_mode_unit(mu_logit, sigma_logit)
    return median, mean, mode, std, lower, upper


def inverse_y_s2(
    pred_mean: torch.Tensor,
    pred_std: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    task_name: str,
    y_scaler: StandardScaler | None,
    standardize_y: bool,
    log_scale: bool = True,
    log_scale_tasks: frozenset[str] | set[str] | None = None,
    logit_scale_tasks: frozenset[str] | set[str] | None = None,
    warps: YWarpConfig | None = None,
    extended: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | InverseYOutput:
    """Map single-task model outputs to the original QoI scale."""
    pred_mean, pred_std, lower, upper = _unstandardize(
        pred_mean,
        pred_std,
        lower,
        upper,
        y_scaler=y_scaler,
        standardize_y=standardize_y,
    )

    point_mean = None
    point_mode = None
    log_mu = None
    log_sigma = None
    logit_mu = None
    logit_sigma = None

    if task_uses_log_scale(
        task_name,
        log_scale=log_scale,
        log_scale_tasks=log_scale_tasks,
        warps=warps,
    ):
        mu_log = pred_mean
        sigma_log = pred_std
        c = _log_offset_for_task(task_name, warps)
        median, mean, mode, ln_std, lo, hi = _lognormal_original_scale(
            mu_log, sigma_log, offset=c
        )
        pred_mean = median
        pred_std = ln_std
        lower = lo
        upper = hi
        point_mean = mean
        point_mode = mode
        log_mu = mu_log
        log_sigma = sigma_log
    elif task_uses_logit_scale(
        task_name, logit_scale_tasks=logit_scale_tasks, warps=warps
    ):
        a, b = logit_bounds_for_task(task_name, warps=warps)
        mu_logit = pred_mean
        sigma_logit = pred_std
        med_u, mean_u, mode_u, std_u, lo_u, hi_u = _logit_normal_unit_interval(
            mu_logit, sigma_logit
        )
        pred_mean = _from_unit_interval(med_u, a=a, b=b)
        pred_std = (b - a) * std_u
        lower = _from_unit_interval(lo_u, a=a, b=b)
        upper = _from_unit_interval(hi_u, a=a, b=b)
        point_mean = _from_unit_interval(mean_u, a=a, b=b)
        point_mode = _from_unit_interval(mode_u, a=a, b=b)
        logit_mu = mu_logit
        logit_sigma = sigma_logit

    out = InverseYOutput(
        point=pred_mean,
        std=pred_std,
        lower=lower,
        upper=upper,
        point_mean=point_mean,
        point_mode=point_mode,
        log_mu=log_mu,
        log_sigma=log_sigma,
        logit_mu=logit_mu,
        logit_sigma=logit_sigma,
    )
    if extended:
        return out
    return out.as_tuple()
