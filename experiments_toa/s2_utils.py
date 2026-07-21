"""Helpers for S2 independent-task runners (band select, ARD extract, metrics)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch


def select_bands(x: torch.Tensor, band_indices: Sequence[int]) -> torch.Tensor:
    """Keep only ``band_indices`` from the last dimension of ``x``."""
    idx = torch.as_tensor(list(band_indices), dtype=torch.int64, device=x.device)
    return x.index_select(-1, idx)


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
        rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
        mae = float(np.mean(np.abs(yp - yt)))
        std = float(np.std(yt))
        rrmse = rmse / std if std > 0 else float("inf")
        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        metrics[f"{name}_RMSE"] = rmse
        metrics[f"{name}_MAE"] = mae
        metrics[f"{name}_RRMSE"] = rrmse
        metrics[f"{name}_R2"] = r2
    return metrics


def macro_rrmse(per_task: Mapping[str, float], task_names: Sequence[str]) -> float:
    vals = [float(per_task[f"{name}_RRMSE"]) for name in task_names]
    return float(np.mean(vals)) if vals else float("nan")
