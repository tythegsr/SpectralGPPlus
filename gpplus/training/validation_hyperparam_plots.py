"""Plot hyperparameter trajectories from ValidationMetricsCallback records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

SCALAR_HYPERPARAM_KEYS = (
    "outputscale",
    "noise",
    "raw_noise",
    "mean_constant",
)
# Signed / unconstrained parameters must use linear y-scale (log + >0 filter hides them).
LINEAR_SCALE_HYPERPARAM_KEYS = frozenset({"raw_noise", "mean_constant", "outputscale"})


def _step_axis(records: list[dict]) -> tuple[np.ndarray, str]:
    if not records:
        return np.array([], dtype=np.float64), "Step"
    if any("lbfgs_iter" in r for r in records):
        steps = np.array(
            [float(r.get("lbfgs_iter", i)) for i, r in enumerate(records)],
            dtype=np.float64,
        )
        return steps, "LBFGS iteration"
    if any("epoch" in r for r in records):
        steps = np.array(
            [float(r.get("epoch", i)) for i, r in enumerate(records)],
            dtype=np.float64,
        )
        return steps, "Epoch"
    return np.arange(len(records), dtype=np.float64), "Logged step"


def _lengthscale_dim_count(records: list[dict]) -> int:
    max_d = 0
    for record in records:
        lengthscales = (record.get("val_diag") or {}).get("lengthscales")
        if isinstance(lengthscales, (list, tuple)):
            max_d = max(max_d, len(lengthscales))
    return max_d


def _lengthscale_keys(records: list[dict]) -> list[str]:
    return [f"lengthscale_{i}" for i in range(_lengthscale_dim_count(records))]


def _task_noise_keys(records: list[dict]) -> list[str]:
    max_t = 0
    for record in records:
        task_noises = (record.get("val_diag") or {}).get("task_noises")
        if isinstance(task_noises, (list, tuple)):
            max_t = max(max_t, len(task_noises))
    return [f"task_noise_{i}" for i in range(max_t)]


def _series_from_records(records: list[dict], key: str) -> np.ndarray:
    values: list[float] = []
    for record in records:
        diag = record.get("val_diag") or {}
        if key.startswith("lengthscale_"):
            idx = int(key.split("_", 1)[1])
            lengthscales = diag.get("lengthscales")
            if isinstance(lengthscales, (list, tuple)) and idx < len(lengthscales):
                values.append(float(lengthscales[idx]))
            else:
                values.append(float("nan"))
            continue
        if key.startswith("task_noise_"):
            idx = int(key.split("_", 2)[2])
            task_noises = diag.get("task_noises")
            if isinstance(task_noises, (list, tuple)) and idx < len(task_noises):
                values.append(float(task_noises[idx]))
            else:
                values.append(float("nan"))
            continue
        raw = diag.get(key)
        if raw is None:
            values.append(float("nan"))
        elif isinstance(raw, (list, tuple)):
            raise ValueError(
                f"Hyperparameter key {key!r} is a list; use indexed keys "
                f"(e.g. lengthscale_i, task_noise_i), not aggregation."
            )
        else:
            values.append(float(raw))
    return np.array(values, dtype=np.float64)


def _available_scalar_keys(records: list[dict]) -> list[str]:
    keys: list[str] = []
    for key in SCALAR_HYPERPARAM_KEYS:
        series = _series_from_records(records, key)
        if np.isfinite(series).any():
            keys.append(key)
    keys.extend(_task_noise_keys(records))
    return keys


def extract_best_init_hyperparams(by_init: dict[str, list[dict]], best_init: int | None) -> dict[str, float]:
    """Final hyperparameter snapshot from the best init's last validation record."""
    if best_init is None:
        return {}
    records = by_init.get(str(best_init)) or by_init.get(best_init)
    if not records:
        return {}
    diag = records[-1].get("val_diag") or {}
    out: dict[str, float] = {}
    for key in _available_scalar_keys(records):
        value = diag.get(key)
        if value is None and key.startswith("task_noise_"):
            idx = int(key.split("_", 2)[2])
            task_noises = diag.get("task_noises")
            if isinstance(task_noises, (list, tuple)) and idx < len(task_noises):
                value = task_noises[idx]
        if value is None:
            continue
        out[key] = float(value)
    lengthscales = diag.get("lengthscales")
    if isinstance(lengthscales, (list, tuple)):
        for i, value in enumerate(lengthscales):
            out[f"lengthscale_{i}"] = float(value)
    return out


def plot_hyperparameter_curves(
    metrics: dict[str, Any],
    by_init: dict[str, list[dict]],
    best_init: int | None,
    save_path: Path,
) -> Path | None:
    """Plot scalar hyperparameters and per-dimension lengthscale trajectories."""
    if best_init is None:
        return None
    records = by_init.get(str(best_init))
    if not records:
        return None

    scalar_keys = _available_scalar_keys(records)
    lengthscale_keys = _lengthscale_keys(records)
    if not scalar_keys and not lengthscale_keys:
        return None

    steps, x_label = _step_axis(records)
    title = metrics.get("title", "run")

    n_scalar_panels = len(scalar_keys)
    n_panels = n_scalar_panels + (1 if lengthscale_keys else 0)
    n_cols = 2
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 3.2 * n_rows), dpi=120, squeeze=False)

    panel = 0
    for key in scalar_keys:
        ax = axes[panel // n_cols][panel % n_cols]
        series = _series_from_records(records, key)
        use_linear = key in LINEAR_SCALE_HYPERPARAM_KEYS
        if use_linear:
            y = np.where(np.isfinite(series), series, np.nan)
            ax.plot(steps, y, color="#7570B3", linewidth=2.0, marker="o", markersize=3)
            ax.set_yscale("linear")
            ax.grid(True, alpha=0.28)
        else:
            y = np.where(np.isfinite(series) & (series > 0), series, np.nan)
            ax.plot(steps, y, color="#7570B3", linewidth=2.0, marker="o", markersize=3)
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.28)
        ax.set_xlabel(x_label)
        ax.set_ylabel(key)
        panel += 1

    if lengthscale_keys:
        ax = axes[panel // n_cols][panel % n_cols]
        cmap = plt.get_cmap("viridis")
        for i, key in enumerate(lengthscale_keys):
            series = _series_from_records(records, key)
            positive = np.where(np.isfinite(series) & (series > 0), series, np.nan)
            color = cmap(i / max(len(lengthscale_keys) - 1, 1))
            ax.plot(
                steps,
                positive,
                color=color,
                linewidth=1.5,
                alpha=0.9,
                label=f"dim {i}",
            )
        ax.set_yscale("log")
        ax.set_xlabel(x_label)
        ax.set_ylabel("lengthscale (per input dim)")
        ax.grid(True, which="both", alpha=0.28)
        if len(lengthscale_keys) <= 12:
            ax.legend(loc="best", fontsize=7, ncol=2)
        else:
            ax.text(
                0.02,
                0.98,
                f"{len(lengthscale_keys)} dims",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
            )
        panel += 1

    for j in range(panel, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    fig.suptitle(f"{title}\nBest init {best_init + 1}: hyperparameters during training", y=1.02)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path
