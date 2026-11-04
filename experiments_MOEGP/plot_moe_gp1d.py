"""Plots for Mixture-of-Experts GP on the 1D bimodal toy."""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from load_bimodal_data import (
    BIMODAL_NOISE_STD,
    bimodal_mode_means,
)


def _safe_savefig(fig, out_path: Path, *, retries: int = 6) -> Path:
    """Save PNG with Windows-friendly retries (locks / OneDrive / antivirus)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    path_str = str(out_path)
    last_err: OSError | None = None
    for attempt in range(retries):
        try:
            if out_path.exists():
                try:
                    out_path.unlink()
                except OSError:
                    pass
            fig.savefig(path_str, bbox_inches="tight")
            return out_path
        except OSError as err:
            last_err = err
            time.sleep(0.2 * (attempt + 1))
    fallback = out_path.parent / f"{out_path.stem}_fallback.png"
    try:
        fig.savefig(str(fallback), bbox_inches="tight")
        return fallback
    except OSError:
        if last_err is not None:
            raise last_err
        raise


def _x_file_slug(x_value: float) -> str:
    rounded = round(float(x_value))
    if abs(float(x_value) - rounded) < 0.05:
        return str(int(rounded))
    return f"{float(x_value):.2f}".replace("-", "m").replace(".", "p")


def _sorted_xy(x: np.ndarray, *arrays: np.ndarray):
    order = np.argsort(np.asarray(x).ravel())
    xs = np.asarray(x).ravel()[order]
    outs = [xs]
    for a in arrays:
        outs.append(np.asarray(a)[order] if a.ndim == 1 else np.asarray(a)[order, ...])
    return outs


def save_moe_posterior_plot(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_grid: np.ndarray,
    expert_means: np.ndarray,
    expert_stds: np.ndarray,
    mixture_mean: np.ndarray,
    out_dir: str | Path,
    *,
    file_tag: str,
    interval_z: float = 1.96,
) -> Path:
    """Data scatter + per-expert mean ± band + true mode lines + mixture mean."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_grid = np.asarray(x_grid, dtype=np.float64).ravel()
    expert_means = np.asarray(expert_means, dtype=np.float64)
    expert_stds = np.asarray(expert_stds, dtype=np.float64)
    mixture_mean = np.asarray(mixture_mean, dtype=np.float64).ravel()
    k = expert_means.shape[1]

    xs, mm, em, es = _sorted_xy(x_grid, mixture_mean, expert_means, expert_stds)
    y1_true, y2_true = bimodal_mode_means(xs)

    colors = plt.cm.tab10(np.linspace(0, 1, max(k, 2)))
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=120)
    ax.scatter(
        np.asarray(x_train).ravel(),
        np.asarray(y_train).ravel(),
        s=14,
        c="0.35",
        alpha=0.45,
        label="Train",
        zorder=2,
        edgecolors="none",
    )
    ax.plot(xs, y1_true, color="#D95F02", linestyle="--", linewidth=1.5, label="True mode 1", zorder=3)
    ax.plot(xs, y2_true, color="#E7298A", linestyle="--", linewidth=1.5, label="True mode 2", zorder=3)

    for j in range(k):
        lower = em[:, j] - interval_z * es[:, j]
        upper = em[:, j] + interval_z * es[:, j]
        ax.fill_between(xs, lower, upper, color=colors[j], alpha=0.15, linewidth=0.0, zorder=1)
        ax.plot(xs, em[:, j], color=colors[j], linewidth=1.8, label=f"Expert {j}", zorder=4)

    ax.plot(xs, mm, color="0.1", linewidth=1.4, linestyle=":", label="Mixture mean", zorder=5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("MoE-GP expert posteriors (bimodal toy)")
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path = out_dir / f"{file_tag}_moe_posterior.png"
    out_path = _safe_savefig(fig, out_path)
    plt.close(fig)
    return out_path


def save_gate_weights_plot(
    x_grid: np.ndarray,
    gate_weights: np.ndarray,
    out_dir: str | Path,
    *,
    file_tag: str,
) -> Path:
    """Plot π_k(x) vs x for each expert."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_grid = np.asarray(x_grid, dtype=np.float64).ravel()
    gate_weights = np.asarray(gate_weights, dtype=np.float64)
    k = gate_weights.shape[1]
    xs, gw = _sorted_xy(x_grid, gate_weights)

    # Sanity: simplex
    row_sums = gw.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        raise ValueError(f"Gate weights must sum to 1; max |sum-1|={np.max(np.abs(row_sums - 1)):.3e}")
    if np.any(gw < -1e-8):
        raise ValueError("Gate weights must be non-negative.")

    colors = plt.cm.tab10(np.linspace(0, 1, max(k, 2)))
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=120)
    for j in range(k):
        ax.plot(xs, gw[:, j], color=colors[j], linewidth=1.8, label=f"π_{j}(x)")
    ax.axhline(1.0 / k, color="0.5", linestyle=":", linewidth=1.0, label=f"1/K={1.0 / k:.2f}")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("x")
    ax.set_ylabel("Expert weight")
    ax.set_title("Gating network weights (softmax, ∑π=1)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path = out_dir / f"{file_tag}_gate_weights.png"
    out_path = _safe_savefig(fig, out_path)
    plt.close(fig)
    return out_path


def save_conditional_pdf_plot(
    x_value: float,
    expert_means_at_x: np.ndarray,
    expert_stds_at_x: np.ndarray,
    gate_weights_at_x: np.ndarray,
    out_dir: str | Path,
    *,
    file_tag: str,
    y_min: float = -2.0,
    y_max: float = 30.0,
    noise_std: float = BIMODAL_NOISE_STD,
) -> Path:
    """Conditional PDF at a fixed x: mixture of experts vs theoretical bimodal."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mu_e = np.asarray(expert_means_at_x, dtype=np.float64).ravel()
    std_e = np.asarray(expert_stds_at_x, dtype=np.float64).ravel()
    pi = np.asarray(gate_weights_at_x, dtype=np.float64).ravel()
    pi = np.clip(pi, 0.0, None)
    pi = pi / pi.sum()

    y_support = np.linspace(y_min, y_max, 400)
    y1_true, y2_true = bimodal_mode_means(np.array([x_value]))
    theoretical = 0.5 * norm.pdf(y_support, y1_true[0], noise_std) + 0.5 * norm.pdf(
        y_support, y2_true[0], noise_std
    )
    mixture = np.zeros_like(y_support)
    for j in range(mu_e.shape[0]):
        mixture += pi[j] * norm.pdf(y_support, mu_e[j], max(std_e[j], 1e-6))

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    ax.plot(y_support, theoretical, "r-", linewidth=2.2, label="Theory (2 modes)")
    ax.plot(y_support, mixture, color="steelblue", linewidth=2.0, label="MoE-GP mixture")
    for j in range(mu_e.shape[0]):
        ax.axvline(mu_e[j], color="0.45", linestyle=":", linewidth=1.0, alpha=0.8)
    ax.set_xlabel("y")
    ax.set_ylabel("Density")
    ax.set_title(f"Conditional p(y|x={x_value:g})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path = out_dir / f"{file_tag}_conditional_pdf_x{_x_file_slug(x_value)}.png"
    out_path = _safe_savefig(fig, out_path)
    plt.close(fig)
    return out_path
