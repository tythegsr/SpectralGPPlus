"""Per-task target transforms for TOA multitask experiments (log grain, logit cos_i)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gpplus.utils import StandardScaler

GRAIN_TASK_IDX = 1
COS_TASK_IDX = 0
GRAIN_TASK_NAME = "y_grain"
COS_TASK_NAME = "y_cos"

COS_EPS = 1e-4

# Cap σ in log space when computing log-normal mean (matches gpplus.utils.train_eval).
_LOG_Y_STD_CAP_FOR_LOGNORMAL_MEAN = 3.0
_NORMAL_Z_95 = 1.96

# Fixed normal quantiles for deterministic logit-normal mean/std on original scale.
_LOGIT_NORMAL_QUANTILE_GRID = torch.tensor(
    [-2.576, -1.96, -1.0, 0.0, 1.0, 1.96, 2.576],
    dtype=torch.float64,
)


@dataclass
class InverseYOutput:
    """Original-scale GP predictions after inverse target transform."""

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


def _clamp_cos_for_logit(y_cos: torch.Tensor) -> torch.Tensor:
    return y_cos.clamp(COS_EPS, 1.0 - COS_EPS)


def forward_y(
    y: torch.Tensor,
    *,
    log_grain: bool,
    logit_cos: bool = False,
) -> torch.Tensor:
    """Apply per-task forward transforms: logit(cos_i), log(grain)."""
    out = y.clone()
    if logit_cos:
        cos = out[..., COS_TASK_IDX]
        if torch.any(cos < 0.0) or torch.any(cos > 1.0):
            raise ValueError("logit_cos requires cos_i in [0, 1].")
        out[..., COS_TASK_IDX] = torch.logit(_clamp_cos_for_logit(cos))
    if log_grain:
        grain = out[..., GRAIN_TASK_IDX]
        if torch.any(grain <= 0):
            raise ValueError("log_grain requires all grain sizes to be strictly positive.")
        out[..., GRAIN_TASK_IDX] = torch.log(grain)
    return out


def _unstandardize_multitask(
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
        y_mean = y_scaler.mean.squeeze(0)
        y_std = y_scaler.std.squeeze(0)
        pred_mean = pred_mean * y_std + y_mean
        pred_std = pred_std * y_std
        lower = lower * y_std + y_mean
        upper = upper * y_std + y_mean
    return pred_mean, pred_std, lower, upper


def _unstandardize_single(
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


def _logit_normal_original_scale(
    mu_logit: torch.Tensor,
    sigma_logit: torch.Tensor,
    *,
    z: float = _NORMAL_Z_95,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map logit-space Normal(μ, σ²) to summaries on [0, 1]."""
    sigma_logit = torch.clamp(sigma_logit, min=0.0)
    median = torch.sigmoid(mu_logit).clamp(0.0, 1.0)
    lower = torch.sigmoid(mu_logit - z * sigma_logit).clamp(0.0, 1.0)
    upper = torch.sigmoid(mu_logit + z * sigma_logit).clamp(0.0, 1.0)

    grid = _LOGIT_NORMAL_QUANTILE_GRID.to(
        device=mu_logit.device,
        dtype=mu_logit.dtype,
    )
    z_samples = mu_logit.unsqueeze(-1) + sigma_logit.unsqueeze(-1) * grid
    c_samples = torch.sigmoid(z_samples).clamp(0.0, 1.0)
    mean = c_samples.mean(dim=-1)
    std = c_samples.std(dim=-1, unbiased=False)

    return median, mean, std, lower, upper


def _apply_grain_lognormal_inverse(
    point: torch.Tensor,
    std: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    task_slice: int | slice | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Apply log-normal back-transform on grain task index or full 1D tensor."""
    if task_slice is None:
        mu_log = point
        sigma_log = std
        median, mean, mode, ln_std, lo, hi = _lognormal_original_scale(mu_log, sigma_log)
        return median, mean, mode, ln_std, lo, hi, mu_log, sigma_log

    idx = task_slice
    mu_log = point[..., idx]
    sigma_log = std[..., idx]
    median, mean, mode, ln_std, lo, hi = _lognormal_original_scale(mu_log, sigma_log)
    point = point.clone()
    std = std.clone()
    lower = lower.clone()
    upper = upper.clone()
    point[..., idx] = median
    std[..., idx] = ln_std
    lower[..., idx] = lo
    upper[..., idx] = hi
    log_mu = torch.full_like(point, float("nan"))
    log_sigma = torch.full_like(point, float("nan"))
    log_mu[..., idx] = mu_log
    log_sigma[..., idx] = sigma_log
    point_mean = point.clone()
    point_mean[..., idx] = mean
    point_mode = point.clone()
    point_mode[..., idx] = mode
    return point, point_mean, point_mode, std, lower, upper, log_mu, log_sigma


def _apply_cos_logit_inverse(
    point: torch.Tensor,
    std: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    task_slice: int | slice | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Apply logit-normal back-transform on cos_i task index or full 1D tensor."""
    if task_slice is None:
        mu_logit = point
        sigma_logit = std
        median, mean, ln_std, lo, hi = _logit_normal_original_scale(mu_logit, sigma_logit)
        return median, mean, ln_std, lo, hi, mu_logit, sigma_logit

    idx = task_slice
    mu_logit = point[..., idx]
    sigma_logit = std[..., idx]
    median, mean, ln_std, lo, hi = _logit_normal_original_scale(mu_logit, sigma_logit)
    point = point.clone()
    std = std.clone()
    lower = lower.clone()
    upper = upper.clone()
    point[..., idx] = median
    std[..., idx] = ln_std
    lower[..., idx] = lo
    upper[..., idx] = hi
    logit_mu = torch.full_like(point, float("nan"))
    logit_sigma = torch.full_like(point, float("nan"))
    logit_mu[..., idx] = mu_logit
    logit_sigma[..., idx] = sigma_logit
    point_mean = point.clone()
    point_mean[..., idx] = mean
    return point, point_mean, std, lower, upper, logit_mu, logit_sigma


def lognormal_summaries_from_log_params(
    mu_log: np.ndarray | torch.Tensor,
    sigma_log: np.ndarray | torch.Tensor,
    *,
    z: float = _NORMAL_Z_95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map log-space Normal parameters to log-normal summaries (numpy arrays)."""
    import numpy as np

    mu_t = torch.as_tensor(mu_log, dtype=torch.float64)
    sig_t = torch.as_tensor(sigma_log, dtype=torch.float64)
    median, mean, mode, std, lower, upper = _lognormal_original_scale(mu_t, sig_t, z=z)
    return (
        median.detach().cpu().numpy(),
        mean.detach().cpu().numpy(),
        mode.detach().cpu().numpy(),
        std.detach().cpu().numpy(),
        lower.detach().cpu().numpy(),
        upper.detach().cpu().numpy(),
    )


def logit_normal_summaries_from_logit_params(
    mu_logit: np.ndarray | torch.Tensor,
    sigma_logit: np.ndarray | torch.Tensor,
    *,
    z: float = _NORMAL_Z_95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map logit-space Normal parameters to [0, 1] summaries (numpy arrays)."""
    import numpy as np

    mu_t = torch.as_tensor(mu_logit, dtype=torch.float64)
    sig_t = torch.as_tensor(sigma_logit, dtype=torch.float64)
    median, mean, std, lower, upper = _logit_normal_original_scale(mu_t, sig_t, z=z)
    return (
        median.detach().cpu().numpy(),
        mean.detach().cpu().numpy(),
        std.detach().cpu().numpy(),
        lower.detach().cpu().numpy(),
        upper.detach().cpu().numpy(),
    )


def inverse_y_predictions(
    pred_mean: torch.Tensor,
    pred_std: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    y_scaler: StandardScaler | None,
    standardize_y: bool,
    log_grain: bool,
    logit_cos: bool = False,
    extended: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    | InverseYOutput
):
    """Map multitask GP outputs to original cos_i / grain (µm) scale."""
    pred_mean, pred_std, lower, upper = _unstandardize_multitask(
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
    cos_mean = None
    grain_mean = None
    grain_mode = None

    if logit_cos:
        pred_mean, cos_mean, pred_std, lower, upper, logit_mu, logit_sigma = (
            _apply_cos_logit_inverse(
                pred_mean, pred_std, lower, upper, task_slice=COS_TASK_IDX
            )
        )

    if log_grain:
        pred_mean, grain_mean, grain_mode, pred_std, lower, upper, log_mu, log_sigma = (
            _apply_grain_lognormal_inverse(
                pred_mean, pred_std, lower, upper, task_slice=GRAIN_TASK_IDX
            )
        )

    if logit_cos or log_grain:
        point_mean = pred_mean.clone()
        if logit_cos and cos_mean is not None:
            point_mean[..., COS_TASK_IDX] = cos_mean[..., COS_TASK_IDX]
        if log_grain and grain_mean is not None:
            point_mean[..., GRAIN_TASK_IDX] = grain_mean[..., GRAIN_TASK_IDX]
    if log_grain and grain_mode is not None:
        point_mode = pred_mean.clone()
        point_mode[..., GRAIN_TASK_IDX] = grain_mode[..., GRAIN_TASK_IDX]

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


def forward_y_single(
    y: torch.Tensor,
    task_name: str,
    *,
    log_grain: bool,
    logit_cos: bool = False,
) -> torch.Tensor:
    """Apply forward target transform for one scalar task."""
    if logit_cos and task_name == COS_TASK_NAME:
        if torch.any(y < 0.0) or torch.any(y > 1.0):
            raise ValueError("logit_cos requires cos_i in [0, 1].")
        return torch.logit(_clamp_cos_for_logit(y))
    if log_grain and task_name == GRAIN_TASK_NAME:
        if torch.any(y <= 0):
            raise ValueError("log_grain requires all grain sizes to be strictly positive.")
        return torch.log(y)
    return y


def inverse_y_single(
    pred_mean: torch.Tensor,
    pred_std: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    task_name: str,
    y_scaler: StandardScaler | None,
    standardize_y: bool,
    log_grain: bool,
    logit_cos: bool = False,
    extended: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    | InverseYOutput
):
    """Map single-task GP outputs to original cos_i / grain (µm) scale."""
    pred_mean, pred_std, lower, upper = _unstandardize_single(
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

    if logit_cos and task_name == COS_TASK_NAME:
        pred_mean, point_mean, pred_std, lower, upper, logit_mu, logit_sigma = (
            _apply_cos_logit_inverse(
                pred_mean, pred_std, lower, upper, task_slice=None
            )
        )

    if log_grain and task_name == GRAIN_TASK_NAME:
        pred_mean, point_mean, point_mode, pred_std, lower, upper, log_mu, log_sigma = (
            _apply_grain_lognormal_inverse(
                pred_mean, pred_std, lower, upper, task_slice=None
            )
        )

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
