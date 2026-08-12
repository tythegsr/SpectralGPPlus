"""Generic S2 plotting / prediction artifacts (no cos/grain assumptions)."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from experiments_toa.s2_constants import S2_LOG_SCALE_TASK_NAMES
from gpplus.utils.fs_path import ensure_dir, ensure_parent, fs_path

_TASK_LABELS: dict[str, str] = {
    "algae": "algae",
    "aot": "aot",
    "cos_i": "cos_i",
    "cwv": "cwv",
    "dust": "dust",
    "fNPV": "fNPV",
    "fPV": "fPV",
    "fsnow": "fsnow",
    "fsoil": "fsoil",
    "grain_size": "grain size (µm)",
    "liquid_water": "liquid water",
}


def _ensure_plot_toa_posterior():
    """Import S1 posterior density helpers (MTGPR dir may already be pinned)."""
    try:
        import plot_toa_posterior as mod

        return mod
    except ImportError:
        mtgpr = Path(__file__).resolve().parents[1] / "experiments_RFFMTGPR"
        mtgpr_s = str(mtgpr)
        if mtgpr_s not in sys.path:
            sys.path.insert(0, mtgpr_s)
        import plot_toa_posterior as mod

        return mod


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
    log_mu: np.ndarray | None = None,
    log_sigma: np.ndarray | None = None,
    log_scale_tasks: Sequence[str] | None = None,
    logit_mu: np.ndarray | None = None,
    logit_sigma: np.ndarray | None = None,
    logit_scale_tasks: Sequence[str] | None = None,
) -> Path:
    save_path = Path(save_path)
    ensure_dir(save_path)
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
    if log_mu is not None:
        payload["log_mu"] = np.asarray(log_mu)
    if log_sigma is not None:
        payload["log_sigma"] = np.asarray(log_sigma)
    if log_scale_tasks is not None:
        payload["log_scale_tasks"] = np.asarray(list(log_scale_tasks))
    if logit_mu is not None:
        payload["logit_mu"] = np.asarray(logit_mu)
    if logit_sigma is not None:
        payload["logit_sigma"] = np.asarray(logit_sigma)
    if logit_scale_tasks is not None:
        payload["logit_scale_tasks"] = np.asarray(list(logit_scale_tasks))
    for name, bands in bands_by_task.items():
        payload[f"bands_{name}"] = np.asarray(list(bands), dtype=np.int64)
    np.savez_compressed(fs_path(out), **payload)
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
    ensure_parent(out_path)
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
    fig.savefig(fs_path(out_path), dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _task_label(name: str) -> str:
    return _TASK_LABELS.get(name, name)


def _resolve_y_std(
    y_std: np.ndarray | None,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    if y_std is not None:
        return np.asarray(y_std, dtype=np.float64)
    # Approximate σ from a 95% CI when callers omit it.
    return np.asarray((upper - lower) / (2.0 * 1.96), dtype=np.float64)


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
    y_std: np.ndarray | None = None,
    rel_metrics_by_task: Mapping[str, Mapping[str, float | int]] | None = None,
    rel_tolerance: float = 0.01,
    log_scale_tasks: Sequence[str] | None = None,
    y_pred_mean: np.ndarray | None = None,
    y_pred_mode: np.ndarray | None = None,
    log_mu: np.ndarray | None = None,
    log_sigma: np.ndarray | None = None,
    spectrum_ylabel: str = "Radiance",
) -> list[str]:
    """
    S1-style posterior figures: spectrum + per-task density panels.

    With 1–2 tasks the layout matches S1 (one row: spectrum | densities).
    With more tasks, the spectrum spans the top row and densities fill a grid below.
    """
    post = _ensure_plot_toa_posterior()

    save_dir = Path(save_dir)
    ensure_dir(save_dir)
    paths: list[str] = []

    names = list(task_names)
    n_tasks = len(names)
    if n_tasks == 0:
        return paths

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    y_std_arr = _resolve_y_std(y_std, lower, upper)
    wl = np.asarray(wavelengths_nm, dtype=np.float64)
    x_test = np.asarray(x_test, dtype=np.float64)

    if log_scale_tasks is None:
        log_set = {n for n in names if n in S2_LOG_SCALE_TASK_NAMES}
    else:
        log_set = set(log_scale_tasks)

    # Classic S1 row when few tasks; otherwise spectrum on top + density grid.
    use_s1_row = n_tasks <= 2
    if use_s1_row:
        n_cols = 1 + n_tasks
        n_rows = 1
    else:
        n_cols = min(3, n_tasks)
        n_rows = 1 + int(math.ceil(n_tasks / n_cols))

    for ex in example_indices:
        if use_s1_row:
            fig, axes = plt.subplots(
                1,
                n_cols,
                figsize=(4.5 * n_cols, 4.2),
                dpi=120,
                constrained_layout=True,
            )
            axes = np.atleast_1d(axes)
            ax_spec = axes[0]
            dens_axes = list(axes[1:])
        else:
            fig = plt.figure(
                figsize=(4.5 * n_cols, 3.6 * n_rows),
                dpi=120,
                constrained_layout=True,
            )
            gs = fig.add_gridspec(n_rows, n_cols)
            ax_spec = fig.add_subplot(gs[0, :])
            dens_axes = []
            for t in range(n_tasks):
                r = 1 + t // n_cols
                c = t % n_cols
                dens_axes.append(fig.add_subplot(gs[r, c]))

        spectrum = x_test[ex]
        ax_spec.plot(wl, spectrum, color="C0", linewidth=1.0)
        ax_spec.set_xlabel("Wavelength (nm)")
        ax_spec.set_ylabel(spectrum_ylabel)
        true_bits = []
        for name in names[:2]:
            t = names.index(name)
            true_bits.append(f"true {name}={y_true[ex, t]:.4g}")
        ax_spec.set_title(", ".join(true_bits) if true_bits else f"test idx {ex}")
        ax_spec.grid(True, alpha=0.3)

        for ax, name in zip(dens_axes, names):
            t = names.index(name)
            rel = None
            if rel_metrics_by_task is not None and name in rel_metrics_by_task:
                rel = dict(rel_metrics_by_task[name])
            post._plot_posterior_density_axis(
                ax,
                task_key=name,
                task_label=_task_label(name),
                y_true=float(y_true[ex, t]),
                y_pred=float(y_pred[ex, t]),
                y_std=float(y_std_arr[ex, t]),
                lower=float(lower[ex, t]),
                upper=float(upper[ex, t]),
                rel_metrics=rel,
                rel_tolerance=rel_tolerance,
                use_lognormal=name in log_set,
                y_pred_mean=(
                    float(y_pred_mean[ex, t])
                    if y_pred_mean is not None
                    else None
                ),
                y_pred_mode=(
                    float(y_pred_mode[ex, t])
                    if y_pred_mode is not None
                    else None
                ),
                log_mu=float(log_mu[ex, t]) if log_mu is not None else None,
                log_sigma=float(log_sigma[ex, t]) if log_sigma is not None else None,
            )

        fig.suptitle(f"TOA test example {ex}", fontsize=11)
        out = save_dir / f"example_{ex:04d}.png"
        fig.savefig(fs_path(out), bbox_inches="tight")
        plt.close(fig)
        paths.append(str(out))
    return paths
