"""Shared helpers for RFFMTGPR experiment runners."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch


def json_safe_optimizer_kwargs(kwargs: dict) -> dict:
    out = {}
    for key, value in kwargs.items():
        if isinstance(value, tuple):
            out[key] = list(value)
        else:
            out[key] = value
    return out


def json_default(obj: Any) -> float:
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_metrics_json(metrics: dict, save_path: str, title: str) -> str:
    import os

    os.makedirs(save_path, exist_ok=True)
    out_json = os.path.join(save_path, f"gp_{title}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=json_default)
    return out_json


def compute_relative_error_metrics(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
    *,
    rel_tolerance: float = 0.01,
    eps: float = 1e-12,
) -> dict[str, float | int]:
    """Relative error stats: max/mean |error|/|y_true| and fraction below rel_tolerance."""
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    n_total = yt.shape[0]
    valid = np.abs(yt) > eps
    n_excluded = int(n_total - valid.sum())
    if not valid.any():
        return {
            "max_rel_error": float("nan"),
            "mean_rel_error": float("nan"),
            "pct_within_1pct": 0.0,
            "n_rel_error_valid": 0,
            "n_rel_error_excluded": n_excluded,
        }
    rel_err = np.abs(yp[valid] - yt[valid]) / np.abs(yt[valid])
    return {
        "max_rel_error": float(np.max(rel_err)),
        "mean_rel_error": float(np.mean(rel_err)),
        "pct_within_1pct": float(np.mean(rel_err < rel_tolerance)),
        "n_rel_error_valid": int(valid.sum()),
        "n_rel_error_excluded": n_excluded,
    }


def format_relative_error_summary(
    task_name: str,
    rel_metrics: dict[str, float | int],
    *,
    rel_tolerance: float = 0.01,
) -> str:
    n_valid = int(rel_metrics.get("n_rel_error_valid", 0))
    n_excluded = int(rel_metrics.get("n_rel_error_excluded", 0))
    n_total = n_valid + n_excluded
    mean_rel = float(rel_metrics["mean_rel_error"])
    max_rel = float(rel_metrics["max_rel_error"])
    pct = float(rel_metrics["pct_within_1pct"]) * 100.0
    tol_pct = rel_tolerance * 100.0
    return (
        f"{task_name}: mean_rel={mean_rel * 100:.2f}%  max_rel={max_rel * 100:.2f}%  "
        f"within_{tol_pct:g}%={pct:.1f}%  ({n_valid}/{n_total} pts)"
    )


def plot_validation_curves_after_save(
    metrics: dict,
    save_path: str,
    json_path: str | None = None,
) -> list[str]:
    """Write validation curve PNGs under {save_path}/validation/ when metrics include validation data."""
    import logging
    from pathlib import Path

    logger = logging.getLogger(__name__)

    if not metrics.get("monitor_validation"):
        return []
    if not metrics.get("validation_metrics_by_init"):
        return []

    try:
        from plot_validation_curves import plot_run
    except ImportError as exc:
        logger.warning("Skipping validation plots: %s", exc)
        return []

    enriched = dict(metrics)
    if json_path:
        enriched["_source_file"] = json_path
    try:
        return [str(p) for p in plot_run(enriched, Path(save_path) / "validation")]
    except Exception as exc:
        logger.warning("Validation plot generation failed: %s", exc)
        return []


DEFAULT_ADAM_KWARGS = {
    "lr": 0.1,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": 0.0,
    "amsgrad": False,
}
DEFAULT_LBFGS_KWARGS = {
    "max_iter": 2000,
    "max_eval": 2500,
    "tolerance_grad": 1e-5,
    "tolerance_change": 1e-9,
    "history_size": 10,
}


def compute_n_val(n_train: int, val_fraction: float = 0.2) -> int:
    """Number of validation points (20% of training by default)."""
    if n_train <= 0:
        return 0
    return max(1, round(val_fraction * n_train))


def unpack_train_val_test(data: tuple) -> tuple:
    """Unpack TOA loader output into train/val/test tensors and split indices."""
    if len(data) == 9:
        return data
    if len(data) == 7:
        x_train, y_train, x_test, y_test, train_idx, test_idx = data
        empty_x = x_train.new_zeros((0, x_train.shape[-1]))
        empty_y = y_train.new_zeros((0, y_train.shape[-1]))
        empty_idx = train_idx.new_zeros((0,), dtype=torch.int64)
        return x_train, y_train, empty_x, empty_y, x_test, y_test, train_idx, empty_idx, test_idx
    if len(data) == 6:
        x_train, y_train, x_val, y_val, x_test, y_test = data
        empty_idx = x_train.new_zeros((0,), dtype=torch.int64)
        return x_train, y_train, x_val, y_val, x_test, y_test, empty_idx, empty_idx, empty_idx
    if len(data) == 4:
        x_train, y_train, x_test, y_test = data
        empty_x = x_train.new_zeros((0, x_train.shape[-1]))
        empty_y = y_train.new_zeros((0, y_train.shape[-1]))
        empty_idx = x_train.new_zeros((0,), dtype=torch.int64)
        return x_train, y_train, empty_x, empty_y, x_test, y_test, empty_idx, empty_idx, empty_idx
    raise ValueError(f"Expected 4, 6, 7, or 9 data tensors, got {len(data)}")


def make_train_loss_callback(
    num_inits: int,
    num_epochs: int,
    *,
    verbose: bool = True,
    log_every_n_epochs: int = 1,
):
    """Build TrainLossLoggingCallback for Adam-style multi-epoch training."""
    from gpplus.training.callbacks import TrainLossLoggingCallback

    return TrainLossLoggingCallback(
        verbose=verbose,
        log_every_n_epochs=log_every_n_epochs,
        num_inits=num_inits,
        total_epochs=num_epochs,
    )


def make_validation_callback(
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    num_inits: int,
    *,
    verbose: bool = True,
    log_every_n_epochs: int = 1,
    log_every_n_iters: int = 10,
    chunk_size: int = 512,
):
    """Build ValidationMetricsCallback. Set verbose=False to silence per-epoch val prints."""
    from gpplus.training.callbacks import ValidationMetricsCallback

    return ValidationMetricsCallback(
        val_x,
        val_y,
        verbose=verbose,
        log_every_n_epochs=log_every_n_epochs,
        log_every_n_iters=log_every_n_iters,
        num_inits=num_inits,
        chunk_size=chunk_size,
    )


def summarize_validation_from_runs(runs: list[dict], best_run: dict) -> dict:
    """Extract per-init validation metrics and best-init summaries from trainer runs."""
    by_init: dict[int, list[dict]] = {}
    for run in runs:
        run_index = run.get("run_index")
        if run_index is None:
            continue
        cb_data = run.get("callback_data", {}).get("ValidationMetricsCallback", {})
        records = cb_data.get("records", [])
        if records:
            by_init[int(run_index)] = records

    summary: dict[str, Any] = {"validation_metrics_by_init": by_init}
    best_index = best_run.get("run_index")
    if best_index is not None:
        summary["best_init_index"] = int(best_index)
    if best_index is not None and int(best_index) in by_init:
        last = by_init[int(best_index)][-1]
        summary["best_val_NLL"] = last.get("val_NLL")
        summary["best_val_RRMSE"] = last.get("val_RRMSE")
    return summary
