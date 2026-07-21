"""Generic S2 plotting / prediction artifacts (no cos/grain assumptions)."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


def select_posterior_example_indices(
    n_test: int,
    n_examples: int,
    *,
    seed: int = 42,
    explicit_indices: Sequence[int] | None = None,
) -> list[int]:
    if explicit_indices is not None:
        out = [int(i) for i in explicit_indices]
        bad = [i for i in out if i < 0 or i >= n_test]
        if bad:
            raise ValueError(f"posterior example indices out of range for n_test={n_test}: {bad}")
        return out
    n = min(int(n_examples), int(n_test))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(n_test, size=n, replace=False).tolist())


def save_s2_predictions_npz(
    save_path: str | Path,
    *,
    title: str,
    task_names: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    x_test: np.ndarray,
    wavelengths_nm: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    bands_by_task: Mapping[str, Sequence[int]],
    y_pred_mean: np.ndarray | None = None,
    y_pred_mode: np.ndarray | None = None,
) -> Path:
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    out = save_path / f"predictions_{title}.npz"
    payload = {
        "task_names": np.asarray(list(task_names)),
        "y_true": np.asarray(y_true),
        "y_pred": np.asarray(y_pred),
        "y_std": np.asarray(y_std),
        "lower": np.asarray(lower),
        "upper": np.asarray(upper),
        "x_test": np.asarray(x_test),
        "wavelengths_nm": np.asarray(wavelengths_nm),
        "train_idx": np.asarray(train_idx, dtype=np.int64),
        "val_idx": np.asarray(val_idx, dtype=np.int64),
        "test_idx": np.asarray(test_idx, dtype=np.int64),
    }
    if y_pred_mean is not None:
        payload["y_pred_mean"] = np.asarray(y_pred_mean)
    if y_pred_mode is not None:
        payload["y_pred_mode"] = np.asarray(y_pred_mode)
    for name, bands in bands_by_task.items():
        payload[f"bands_{name}"] = np.asarray(list(bands), dtype=np.int64)
    np.savez_compressed(out, **payload)
    return out


def plot_s2_task_scatter(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray | None,
    upper: np.ndarray | None,
    task_name: str,
    out_path: str | Path,
    title: str | None = None,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(y_true, y_pred, s=8, alpha=0.35, edgecolors="none")
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="y = x")
    if lower is not None and upper is not None:
        # Show a subset of intervals for readability.
        n = len(y_true)
        idx = np.linspace(0, n - 1, num=min(80, n), dtype=int)
        ax.vlines(
            y_true[idx],
            lower[idx],
            upper[idx],
            colors="C0",
            alpha=0.15,
            lw=0.8,
        )
    ax.set_xlabel(f"True {task_name}")
    ax.set_ylabel(f"Predicted {task_name}")
    ax.set_title(title or f"{task_name}: predicted vs true")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_s2_posterior_examples(
    *,
    x_test: np.ndarray,
    wavelengths_nm: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    task_names: Sequence[str],
    example_indices: Sequence[int],
    save_dir: str | Path,
    title: str,
) -> list[str]:
    """Spectrum + selected-task posterior panels for a few test examples."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    # Cap task panels for readability.
    panel_tasks = list(task_names[:4])
    n_panels = 1 + len(panel_tasks)

    for ex in example_indices:
        fig, axes = plt.subplots(
            n_panels,
            1,
            figsize=(8.0, 2.2 * n_panels),
            constrained_layout=True,
        )
        if n_panels == 1:
            axes = [axes]
        ax0 = axes[0]
        ax0.plot(wavelengths_nm, x_test[ex], lw=1.0, color="C0")
        ax0.set_xlabel("Wavelength (nm)")
        ax0.set_ylabel("Reflectance")
        ax0.set_title(f"{title} | test idx {ex}")

        for ax, t_name in zip(axes[1:], panel_tasks):
            t = list(task_names).index(t_name)
            yt = float(y_true[ex, t])
            yp = float(y_pred[ex, t])
            lo = float(lower[ex, t])
            hi = float(upper[ex, t])
            ax.errorbar([0], [yp], yerr=[[yp - lo], [hi - yp]], fmt="o", color="C1", label="pred")
            ax.axhline(yt, color="k", ls="--", label="true")
            ax.set_xticks([])
            ax.set_ylabel(t_name)
            ax.legend(loc="best", fontsize=8)

        out = save_dir / f"example_{ex:04d}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        paths.append(str(out))
    return paths
