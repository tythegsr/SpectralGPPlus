"""Helpers for S2 independent-task runners (band select, ARD extract, metrics)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch


def select_bands(x: torch.Tensor, band_indices: Sequence[int]) -> torch.Tensor:
    """Keep only ``band_indices`` from the last dimension of ``x``."""
    idx = torch.as_tensor(list(band_indices), dtype=torch.int64, device=x.device)
    return x.index_select(-1, idx)


def apply_x_transform(x: torch.Tensor, transform: str | None) -> torch.Tensor:
    """
    Elementwise input transform applied after band select, before PCA / X scaling.

    Supported: ``None`` / ``\"none\"``, ``\"log1p\"`` (``ln(1 + max(x, 0))``).
    """
    if transform is None or transform == "none":
        return x
    if transform == "log1p":
        return torch.log1p(x.clamp_min(0.0))
    raise ValueError(f"Unknown x_transform {transform!r}; expected 'none' or 'log1p'.")


def extract_ard_lengthscales(model) -> dict[str, Any]:
    """Extract transformed lengthscales from a GP / RFFGP model if present."""
    out: dict[str, Any] = {
        "lengthscale": [],
        "raw_lengthscale": [],
        "outputscale": None,
    }
    try:
        covar = model.covar_module
    except Exception:
        return out

    try:
        if hasattr(covar, "outputscale"):
            out["outputscale"] = float(covar.outputscale.detach().cpu().reshape(-1)[0].item())
    except Exception:
        pass

    base = getattr(covar, "base_kernel", covar)
    try:
        if hasattr(base, "lengthscale"):
            ls = base.lengthscale.detach().cpu().reshape(-1)
            out["lengthscale"] = [float(v) for v in ls.tolist()]
        if hasattr(base, "raw_lengthscale"):
            raw = base.raw_lengthscale.detach().cpu().reshape(-1)
            out["raw_lengthscale"] = [float(v) for v in raw.tolist()]
    except Exception:
        pass
    return out


def map_ard_to_bands(
    lengthscales: Sequence[float],
    band_indices: Sequence[int],
    *,
    wavelengths_nm: Sequence[float] | None = None,
    ard_space: str = "bands",
) -> dict[str, Any]:
    """Attach ARD values to original band indices or PCA component ids."""
    ls = [float(v) for v in lengthscales]
    if ard_space == "bands":
        if len(ls) not in (0, 1, len(band_indices)):
            # Soft mismatch: still return what we have.
            pass
        entries = []
        for i, band in enumerate(band_indices):
            entry = {"band_index": int(band), "lengthscale": float(ls[i]) if i < len(ls) else None}
            if wavelengths_nm is not None:
                entry["wavelength_nm"] = float(wavelengths_nm[int(band)])
            entries.append(entry)
        return {"ard_space": "bands", "entries": entries, "lengthscale": ls}

    entries = [{"component": i, "lengthscale": float(ls[i]) if i < len(ls) else None} for i in range(len(ls))]
    return {
        "ard_space": "pca_components",
        "physical_band_indices": [int(b) for b in band_indices],
        "entries": entries,
        "lengthscale": ls,
    }


def compute_per_task_metrics(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
    task_names: Sequence[str],
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1, len(task_names))
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1, len(task_names))
    metrics: dict[str, float] = {}
    for t, name in enumerate(task_names):
        yt = y_true[:, t]
        yp = y_pred[:, t]
        metrics.update(_scalar_error_metrics(yt, yp, prefix=f"{name}_"))
    return metrics


def _scalar_error_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    prefix: str = "",
) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    abs_err = np.abs(yp - yt)
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
    mae = float(np.mean(abs_err))
    medae = float(np.median(abs_err))
    std = float(np.std(yt))
    rrmse = rmse / std if std > 0 else float("inf")
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        f"{prefix}RMSE": rmse,
        f"{prefix}MAE": mae,
        f"{prefix}MedAE": medae,
        f"{prefix}RRMSE": rrmse,
        f"{prefix}R2": r2,
    }


def compute_log_scale_extra_metrics(
    y_true_physical: np.ndarray | torch.Tensor,
    *,
    log_mu: np.ndarray | torch.Tensor,
    point_mean_physical: np.ndarray | torch.Tensor | None = None,
) -> dict[str, float]:
    """
    Extra test metrics for QoIs trained in ln-space.

    - ``*_log``: errors in ln-space (``ln(y_true)`` vs predictive ``log_mu``).
    - ``*_mean``: physical errors using log-normal mean as the point estimate.
    """
    yt = np.asarray(y_true_physical, dtype=np.float64).reshape(-1)
    mu = np.asarray(log_mu, dtype=np.float64).reshape(-1)
    if yt.shape != mu.shape:
        raise ValueError(f"y_true and log_mu shape mismatch: {yt.shape} vs {mu.shape}")
    if np.any(yt <= 0):
        raise ValueError("log-scale metrics require strictly positive y_true.")

    out = _scalar_error_metrics(np.log(yt), mu, prefix="")
    # Rename to *_log
    log_out = {
        "RMSE_log": out["RMSE"],
        "MAE_log": out["MAE"],
        "MedAE_log": out["MedAE"],
        "RRMSE_log": out["RRMSE"],
        "R2_log": out["R2"],
    }
    if point_mean_physical is not None:
        ym = np.asarray(point_mean_physical, dtype=np.float64).reshape(-1)
        mean_out = _scalar_error_metrics(yt, ym, prefix="")
        log_out.update(
            {
                "RMSE_mean": mean_out["RMSE"],
                "MAE_mean": mean_out["MAE"],
                "MedAE_mean": mean_out["MedAE"],
                "RRMSE_mean": mean_out["RRMSE"],
                "R2_mean": mean_out["R2"],
            }
        )
    return log_out


def macro_rrmse(per_task: Mapping[str, float], task_names: Sequence[str]) -> float:
    vals = [float(per_task[f"{name}_RRMSE"]) for name in task_names]
    return float(np.mean(vals)) if vals else float("nan")


def macro_metric(
    per_task: Mapping[str, float],
    task_names: Sequence[str],
    suffix: str,
) -> float:
    """Mean of ``{name}_{suffix}`` over tasks that define the key."""
    vals = [
        float(per_task[f"{name}_{suffix}"])
        for name in task_names
        if f"{name}_{suffix}" in per_task
    ]
    return float(np.mean(vals)) if vals else float("nan")
