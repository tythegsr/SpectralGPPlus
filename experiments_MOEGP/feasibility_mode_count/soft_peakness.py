"""Soft peakness / soft peak-count surrogates for mixture densities.

``soft_peakness``: raw concavity mass (can be high on a single tall bump).
``soft_peak_count``: expected number of soft local maxima (preferred diagnostic).
"""

from __future__ import annotations

import numpy as np
import torch


def soft_peakness(
    y_grid: np.ndarray,
    density: np.ndarray,
    *,
    rel_thr: float = 0.1,
    tau: float = 0.05,
) -> float:
    """Sum of soft concavity × height gates (not a count)."""
    y = np.asarray(y_grid, dtype=np.float64).ravel()
    d = np.asarray(density, dtype=np.float64).ravel()
    if d.size < 3:
        return 0.0
    d2 = d[2:] - 2.0 * d[1:-1] + d[:-2]
    mid = d[1:-1]
    thr = rel_thr * float(d.max() + 1e-12)
    concavity = np.log1p(np.exp(np.clip(-d2, -50, 50)))
    height = 1.0 / (1.0 + np.exp(-(mid - thr) / max(tau * float(d.max() + 1e-12), 1e-8)))
    dy = float(y[1] - y[0]) if y.size > 1 else 1.0
    return float(np.sum(concavity * height) * dy)


def soft_peak_count(
    y_grid: np.ndarray,
    density: np.ndarray,
    *,
    rel_thr: float = 0.05,
    temp: float = 0.02,
    min_separation_frac: float = 0.05,
) -> float:
    """Differentiable-style expected peak count of a 1D density (numpy).

    At each interior grid index, score whether it is a local max above a
    relative height threshold. Nearby peaks are suppressed with a soft NMS
    so the sum approximates an integer peak count.
    """
    y = np.asarray(y_grid, dtype=np.float64).ravel()
    d = np.asarray(density, dtype=np.float64).ravel()
    if d.size < 3:
        return float(d.size > 0 and d.max() > 0)
    dmax = float(d.max() + 1e-12)
    thr = rel_thr * dmax
    # Soft local-max: greater than both neighbors
    left = d[1:-1] - d[:-2]
    right = d[1:-1] - d[2:]
    t = max(temp * dmax, 1e-8)
    is_peak = (
        (1.0 / (1.0 + np.exp(-left / t)))
        * (1.0 / (1.0 + np.exp(-right / t)))
        * (1.0 / (1.0 + np.exp(-(d[1:-1] - thr) / t)))
    )
    y_mid = y[1:-1]
    # Soft NMS: weight peaks by how much they dominate a separation neighborhood
    sep = min_separation_frac * float(y.max() - y.min() + 1e-12)
    kept = np.zeros_like(is_peak)
    order = np.argsort(-is_peak)
    taken_y: list[float] = []
    for idx in order:
        if is_peak[idx] < 1e-6:
            break
        if any(abs(float(y_mid[idx]) - ty) < sep for ty in taken_y):
            continue
        kept[idx] = is_peak[idx]
        taken_y.append(float(y_mid[idx]))
    return float(kept.sum())


def soft_peak_count_torch(
    y_grid: torch.Tensor,
    density: torch.Tensor,
    *,
    rel_thr: float = 0.05,
    temp: float = 0.02,
) -> torch.Tensor:
    """Torch soft peak count without NMS (usable as a training regularizer).

    Returns a scalar tensor. Nearby-peak suppression is omitted so gradients
    remain simple; pair with separation loss on mixture means.
    """
    d = density.reshape(-1)
    if d.numel() < 3:
        return d.new_zeros(())
    dmax = d.max().clamp_min(1e-12)
    thr = rel_thr * dmax
    t = (temp * dmax).clamp_min(1e-8)
    left = d[1:-1] - d[:-2]
    right = d[1:-1] - d[2:]
    mid = d[1:-1]
    is_peak = torch.sigmoid(left / t) * torch.sigmoid(right / t) * torch.sigmoid((mid - thr) / t)
    return is_peak.sum()
