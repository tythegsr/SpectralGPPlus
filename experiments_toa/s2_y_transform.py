"""Per-QoI target transforms for S2 (log scale on algae/dust/grain/LWC)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from experiments_toa.s2_constants import S2_LOG_SCALE_TASK_NAMES
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


def task_uses_log_scale(task_name: str, *, log_scale: bool = True) -> bool:
    """Return True when ``task_name`` should be trained in log space."""
    return bool(log_scale) and task_name in S2_LOG_SCALE_TASK_NAMES


def forward_y_s2(
    y: torch.Tensor,
    task_name: str,
    *,
    log_scale: bool = True,
) -> torch.Tensor:
    """Apply forward target transform for one S2 QoI (``log`` for selected tasks)."""
    if not task_uses_log_scale(task_name, log_scale=log_scale):
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

    if task_uses_log_scale(task_name, log_scale=log_scale):
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
