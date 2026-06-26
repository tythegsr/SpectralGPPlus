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


def plot_validation_curves_after_save(
    metrics: dict,
    save_path: str,
    json_path: str | None = None,
) -> list[str]:
    """Write validation curve PNGs under {save_path}/validation/ when metrics include validation data."""
    from pathlib import Path

    if not metrics.get("monitor_validation"):
        return []
    if not metrics.get("validation_metrics_by_init"):
        return []

    from plot_validation_curves import plot_run

    enriched = dict(metrics)
    if json_path:
        enriched["_source_file"] = json_path
    return [str(p) for p in plot_run(enriched, Path(save_path) / "validation")]


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
    """Unpack 4- or 6-tuple from data generators into train/val/test tensors."""
    if len(data) == 6:
        return data
    if len(data) != 4:
        raise ValueError(f"Expected 4 or 6 data tensors, got {len(data)}")
    x_train, y_train, x_test, y_test = data
    empty_x = x_train.new_zeros((0, x_train.shape[-1]))
    empty_y = y_train.new_zeros((0, y_train.shape[-1]))
    return x_train, y_train, empty_x, empty_y, x_test, y_test


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
