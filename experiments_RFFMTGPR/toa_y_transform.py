"""Per-task target transforms for TOA multitask experiments (log grain)."""

from __future__ import annotations

import torch

from gpplus.utils import StandardScaler

GRAIN_TASK_IDX = 1
COS_TASK_IDX = 0


def forward_y(y: torch.Tensor, *, log_grain: bool) -> torch.Tensor:
    """Apply forward target transform: log(grain) on task 1 only; cos_i unchanged."""
    if not log_grain:
        return y
    out = y.clone()
    grain = out[..., GRAIN_TASK_IDX]
    if torch.any(grain <= 0):
        raise ValueError("log_grain requires all grain sizes to be strictly positive.")
    out[..., GRAIN_TASK_IDX] = torch.log(grain)
    return out


def inverse_y_predictions(
    pred_mean: torch.Tensor,
    pred_std: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    y_scaler: StandardScaler | None,
    standardize_y: bool,
    log_grain: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map multitask GP outputs to original cos_i / grain (µm) scale."""
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

    if log_grain:
        idx = GRAIN_TASK_IDX
        pred_mean[..., idx] = torch.exp(pred_mean[..., idx])
        lower[..., idx] = torch.exp(lower[..., idx])
        upper[..., idx] = torch.exp(upper[..., idx])
        pred_std[..., idx] = (upper[..., idx] - lower[..., idx]) / 4.0

    return pred_mean, pred_std, lower, upper
