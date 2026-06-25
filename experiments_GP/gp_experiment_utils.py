"""Shared helpers for exact-GP experiment runners."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch

from gpplus import kernels
from gpplus.models import GPR


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


def format_noise_level(noise: float) -> str:
    """Compact noise label for filenames (e.g. 0.005 -> '0.005', 0.0 -> '0.0')."""
    return f"{float(noise):g}"


def format_pi_frequency_label(frequency: float) -> str:
    """Compact pi-frequency label (e.g. 2.0 -> '2pi', 4.0 -> '4pi')."""
    return f"{format_noise_level(frequency)}pi"


def sin_pi_frequency_problem_name(frequency: float) -> str:
    """Problem slug for sin(frequency*pi*x), e.g. sin_2pi_x."""
    return f"sin_{format_pi_frequency_label(frequency)}_x"


def build_1d_run_file_tag(
    *,
    train_size: int,
    noise_train: float,
    noise_test: float,
    seed: int,
    test_outside_margin: float = 0.0,
    frequency: float | None = None,
) -> str:
    """Unique per-run tag for 1D prediction artifacts (avoids overwrites across noise levels)."""
    parts = [
        f"train{train_size}",
        f"noiseTrain{format_noise_level(noise_train)}",
        f"noiseTest{format_noise_level(noise_test)}",
        f"seed{seed}",
    ]
    if test_outside_margin > 0:
        parts.insert(1, f"ood{format_noise_level(test_outside_margin)}")
    if frequency is not None:
        parts.insert(1, format_pi_frequency_label(frequency))
    return "_".join(parts)


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
        nested = metrics.get("gp_metrics")
        if not isinstance(nested, dict) or not nested.get("validation_metrics_by_init"):
            return []

    from plot_validation_curves import plot_run

    enriched = dict(metrics)
    if json_path:
        enriched["_source_file"] = json_path
    return [str(p) for p in plot_run(enriched, Path(save_path) / "validation")]


def extract_learned_likelihood_noise(
    model,
    y_std: torch.Tensor | float | None = None,
) -> dict[str, float]:
    """
    Extract learned homoscedastic noise from the model likelihood.

    Returns raw_noise (log10 scale param), noise (variance in training y scale),
    and noise_std (standard deviation in original y scale when y_std is given).
    """
    result: dict[str, float] = {
        "raw_noise": float("nan"),
        "noise": float("nan"),
        "noise_std": float("nan"),
    }

    try:
        raw_noise = model.likelihood.raw_noise.detach().cpu()
        result["raw_noise"] = (
            float(raw_noise.item())
            if raw_noise.numel() == 1
            else float(raw_noise.numpy().flatten()[0])
        )
    except Exception:
        pass

    try:
        noise_variance = model.likelihood.noise.detach().cpu()
        noise_val = (
            float(noise_variance.item())
            if noise_variance.numel() == 1
            else float(noise_variance.numpy().flatten()[0])
        )
        result["noise"] = noise_val
        noise_std = float(np.sqrt(noise_val))

        if y_std is not None:
            if isinstance(y_std, dict):
                std_to_use = y_std[0] if 0 in y_std else list(y_std.values())[0]
            else:
                std_to_use = y_std.item() if hasattr(y_std, "item") else y_std
            result["noise_std"] = float(noise_std * float(std_to_use))
        else:
            result["noise_std"] = noise_std
    except Exception:
        pass

    return result


def build_gpr_model(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    ard: bool = True,
) -> GPR:
    """Build GPR with default mean/likelihood; kernel uses ARD when ard=True."""
    input_dim = x_train.shape[-1]
    if ard:
        kernel_module = kernels.LogScaleKernel(kernels.GaussianKernel(ard_num_dims=input_dim))
    else:
        kernel_module = kernels.LogScaleKernel(kernels.GaussianKernel())
    return GPR(x_train, y_train, kernel_module=kernel_module)


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

VAL_SEED_OFFSET = 1_000_003


def compute_n_val(n_train: int, val_fraction: float = 0.2) -> int:
    """Number of independently generated validation points (20% of training by default)."""
    if n_train <= 0:
        return 0
    return max(1, round(val_fraction * n_train))


def compute_val_samples_per_source(
    train_samples_per_source: list[int],
    val_fraction: float = 0.2,
) -> list[int]:
    return [compute_n_val(n, val_fraction) if n > 0 else 0 for n in train_samples_per_source]


def unpack_train_val_test(data: tuple) -> tuple:
    """Unpack 4- or 6-tuple from data generators into train/val/test tensors."""
    if len(data) == 6:
        return data
    if len(data) != 4:
        raise ValueError(f"Expected 4 or 6 data tensors, got {len(data)}")
    x_train, y_train, x_test, y_test = data
    empty_x = x_train.new_zeros((0, x_train.shape[-1]))
    empty_y = y_train.new_zeros((0,))
    return x_train, y_train, empty_x, empty_y, x_test, y_test


def scale_validation_tensors(
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    x_scaler,
    y_scaler,
    standardize_x: bool,
    standardize_y: bool,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x_val.numel() == 0:
        return x_val.to(dtype=dtype), y_val.to(dtype=dtype)
    x_val = x_val.to(dtype=dtype)
    y_val = y_val.to(dtype=dtype)
    if standardize_x and x_scaler is not None:
        x_val = x_scaler.transform(x_val)
    if standardize_y and y_scaler is not None:
        y_val = y_scaler.transform(y_val.unsqueeze(-1)).squeeze(-1)
    return x_val, y_val


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
