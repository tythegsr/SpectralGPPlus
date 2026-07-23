"""Per-QoI target transforms for S2 (log scale on algae/dust/grain/LWC)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from experiments_toa.s2_constants import S2_LOG_SCALE_TASK_NAMES, S2_TASK_NAMES
from gpplus.utils import StandardScaler

# Cap σ in log space when computing log-normal mean (matches S1 toa_y_transform).
_LOG_Y_STD_CAP_FOR_LOGNORMAL_MEAN = 3.0
_NORMAL_Z_95 = 1.96


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

    def as_tuple(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.point, self.std, self.lower, self.upper


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
        return value
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
    return frozenset(n for n in names if n in S2_TASK_NAMES)


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


def task_uses_log_scale(
    task_name: str,
    *,
    log_scale: bool = True,
    log_scale_tasks: frozenset[str] | set[str] | None = None,
) -> bool:
    """Return True when ``task_name`` should be trained in log space."""
    if not log_scale:
        return False
    allowed = (
        frozenset(log_scale_tasks)
        if log_scale_tasks is not None
        else frozenset(S2_LOG_SCALE_TASK_NAMES)
    )
    return task_name in allowed


def forward_y_s2(
    y: torch.Tensor,
    task_name: str,
    *,
    log_scale: bool = True,
    log_scale_tasks: frozenset[str] | set[str] | None = None,
) -> torch.Tensor:
    """Apply forward target transform for one S2 QoI (``log`` for selected tasks)."""
    if not task_uses_log_scale(
        task_name, log_scale=log_scale, log_scale_tasks=log_scale_tasks
    ):
        return y
    if torch.any(y <= 0):
        raise ValueError(
            f"log scaling for {task_name!r} requires all targets to be strictly positive."
        )
    return torch.log(y)


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map log-space Normal(μ, σ²) to log-normal summaries on original scale."""
    sigma_log = torch.clamp(sigma_log, min=0.0)
    median = torch.exp(mu_log)
    sigma_for_mean = torch.clamp(sigma_log, max=_LOG_Y_STD_CAP_FOR_LOGNORMAL_MEAN)
    mean = median * torch.exp(0.5 * sigma_for_mean**2)
    mode = torch.exp(mu_log - sigma_log**2)
    std = median * torch.sqrt(torch.expm1(sigma_log**2))
    lower = torch.exp(mu_log - z * sigma_log)
    upper = torch.exp(mu_log + z * sigma_log)
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

    if task_uses_log_scale(
        task_name, log_scale=log_scale, log_scale_tasks=log_scale_tasks
    ):
        mu_log = pred_mean
        sigma_log = pred_std
        median, mean, mode, ln_std, lo, hi = _lognormal_original_scale(mu_log, sigma_log)
        pred_mean = median
        pred_std = ln_std
        lower = lo
        upper = hi
        point_mean = mean
        point_mode = mode
        log_mu = mu_log
        log_sigma = sigma_log

    out = InverseYOutput(
        point=pred_mean,
        std=pred_std,
        lower=lower,
        upper=upper,
        point_mean=point_mean,
        point_mode=point_mode,
        log_mu=log_mu,
        log_sigma=log_sigma,
    )
    if extended:
        return out
    return out.as_tuple()
