"""Plots and diagnostics for multimodal / sharp-edge MoE-GP (MOEGPwD)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from load_bimodal_data import BIMODAL_NOISE_STD, bimodal_mode_means


def _sorted_xy(x: np.ndarray, *arrays: np.ndarray):
    order = np.argsort(np.asarray(x).ravel())
    xs = np.asarray(x).ravel()[order]
    outs = [xs]
    for a in arrays:
        outs.append(np.asarray(a)[order] if a.ndim == 1 else np.asarray(a)[order, ...])
    return outs


def gate_entropy(gate_weights: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-point Shannon entropy of π(x); shape ``[N]``."""
    pi = np.clip(np.asarray(gate_weights, dtype=np.float64), eps, 1.0)
    pi = pi / pi.sum(axis=-1, keepdims=True)
    return -(pi * np.log(pi)).sum(axis=-1)


def mode_purity(
    assignments: np.ndarray,
    true_mode_mask: np.ndarray,
    num_experts: int,
) -> dict:
    """
    Match hard expert labels to true modes (bool mask: True = mode 1).

    Returns best-permutation purity and confusion-style counts.
    """
    assign = np.asarray(assignments, dtype=np.int64).ravel()
    true_mode = np.asarray(true_mode_mask, dtype=bool).ravel()
    # Map: expert -> majority true mode
    expert_to_mode: dict[int, int] = {}
    for k in range(num_experts):
        mask = assign == k
        if not mask.any():
            expert_to_mode[k] = 0
            continue
        # mode label 1 if true_mode True else 0
        frac_mode1 = float(true_mode[mask].mean())
        expert_to_mode[k] = 1 if frac_mode1 >= 0.5 else 0
    pred_mode = np.array([expert_to_mode[int(a)] for a in assign], dtype=np.int64)
    true_int = true_mode.astype(np.int64)
    purity = float((pred_mode == true_int).mean())
    return {
        "mode_purity": purity,
        "expert_to_mode": {str(k): v for k, v in expert_to_mode.items()},
        "n_agree": int((pred_mode == true_int).sum()),
        "n_total": int(assign.size),
    }


def expert_vs_true_line_rmse(
    x_grid: np.ndarray,
    expert_means: np.ndarray,
) -> dict[str, float]:
    """RMSE of each expert mean to each true bimodal line; plus best matching."""
    xs = np.asarray(x_grid, dtype=np.float64).ravel()
    em = np.asarray(expert_means, dtype=np.float64)
    y1, y2 = bimodal_mode_means(xs)
    out: dict[str, float] = {}
    k = em.shape[1]
    for j in range(k):
        out[f"expert{j}_rmse_mode1"] = float(np.sqrt(np.mean((em[:, j] - y1) ** 2)))
        out[f"expert{j}_rmse_mode2"] = float(np.sqrt(np.mean((em[:, j] - y2) ** 2)))
    # Best assignment of experts to modes (for K=2 greedy)
    if k == 2:
        # Try both permutations
        a = out["expert0_rmse_mode1"] + out["expert1_rmse_mode2"]
        b = out["expert0_rmse_mode2"] + out["expert1_rmse_mode1"]
        out["matched_expert_line_rmse_sum"] = float(min(a, b))
    return out


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
    title: str = "MoE-GP expert posteriors",
    true_curves: list[tuple[np.ndarray, str]] | None = None,
    interval_z: float = 1.96,
    show_bimodal_truth: bool = False,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_grid = np.asarray(x_grid, dtype=np.float64).ravel()
    expert_means = np.asarray(expert_means, dtype=np.float64)
    expert_stds = np.asarray(expert_stds, dtype=np.float64)
    mixture_mean = np.asarray(mixture_mean, dtype=np.float64).ravel()
    k = expert_means.shape[1]

    xs, mm, em, es = _sorted_xy(x_grid, mixture_mean, expert_means, expert_stds)
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
    if show_bimodal_truth:
        y1_true, y2_true = bimodal_mode_means(xs)
        ax.plot(xs, y1_true, color="#D95F02", linestyle="--", linewidth=1.5, label="True mode 1", zorder=3)
        ax.plot(xs, y2_true, color="#E7298A", linestyle="--", linewidth=1.5, label="True mode 2", zorder=3)
    if true_curves is not None:
        for curve, label in true_curves:
            ax.plot(xs, np.asarray(curve).ravel()[np.argsort(x_grid)], linestyle="--", linewidth=1.5, label=label)

    for j in range(k):
        lower = em[:, j] - interval_z * es[:, j]
        upper = em[:, j] + interval_z * es[:, j]
        ax.fill_between(xs, lower, upper, color=colors[j], alpha=0.15, linewidth=0.0, zorder=1)
        ax.plot(xs, em[:, j], color=colors[j], linewidth=1.8, label=f"Expert {j}", zorder=4)

    ax.plot(xs, mm, color="0.1", linewidth=1.4, linestyle=":", label="Mixture mean", zorder=5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path = out_dir / f"{file_tag}_moe_posterior.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_gate_weights_plot(
    x_grid: np.ndarray,
    gate_weights: np.ndarray,
    out_dir: str | Path,
    *,
    file_tag: str,
    title: str = "Gating weights (softmax, ∑π=1)",
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_grid = np.asarray(x_grid, dtype=np.float64).ravel()
    gate_weights = np.asarray(gate_weights, dtype=np.float64)
    k = gate_weights.shape[1]
    xs, gw = _sorted_xy(x_grid, gate_weights)

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
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path = out_dir / f"{file_tag}_gate_weights.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_assignment_scatter(
    x_train: np.ndarray,
    y_train: np.ndarray,
    assignments: np.ndarray,
    out_dir: str | Path,
    *,
    file_tag: str,
    true_mode_mask: np.ndarray | None = None,
    title: str = "Hard expert assignments",
) -> Path:
    """Scatter train points colored by expert; optional true-mode markers."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x_train, dtype=np.float64).ravel()
    y = np.asarray(y_train, dtype=np.float64).ravel()
    a = np.asarray(assignments, dtype=np.int64).ravel()
    k = int(a.max()) + 1 if a.size else 1
    colors = plt.cm.tab10(np.linspace(0, 1, max(k, 2)))

    fig, axes = plt.subplots(1, 2 if true_mode_mask is not None else 1, figsize=(12 if true_mode_mask is not None else 7, 4.5), dpi=120)
    if true_mode_mask is None:
        axes = [axes]
    ax0 = axes[0]
    for j in range(k):
        m = a == j
        ax0.scatter(x[m], y[m], s=16, color=colors[j], alpha=0.7, label=f"Expert {j}", edgecolors="none")
    ax0.set_title(title)
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    ax0.legend(fontsize=8)
    ax0.grid(True, alpha=0.25)

    if true_mode_mask is not None:
        tm = np.asarray(true_mode_mask, dtype=bool).ravel()
        ax1 = axes[1]
        ax1.scatter(x[tm], y[tm], s=16, c="#D95F02", alpha=0.7, label="True mode 1", edgecolors="none")
        ax1.scatter(x[~tm], y[~tm], s=16, c="#E7298A", alpha=0.7, label="True mode 2", edgecolors="none")
        ax1.set_title("True mode labels")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.25)

    fig.tight_layout()
    out_path = out_dir / f"{file_tag}_assignments.png"
    fig.savefig(out_path, bbox_inches="tight")
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
    show_bimodal_theory: bool = True,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mu_e = np.asarray(expert_means_at_x, dtype=np.float64).ravel()
    std_e = np.asarray(expert_stds_at_x, dtype=np.float64).ravel()
    pi = np.asarray(gate_weights_at_x, dtype=np.float64).ravel()
    pi = np.clip(pi, 0.0, None)
    pi = pi / max(pi.sum(), 1e-12)

    y_support = np.linspace(y_min, y_max, 400)
    mixture = np.zeros_like(y_support)
    for j in range(mu_e.shape[0]):
        mixture += pi[j] * norm.pdf(y_support, mu_e[j], max(std_e[j], 1e-6))

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    if show_bimodal_theory:
        y1_true, y2_true = bimodal_mode_means(np.array([x_value]))
        theoretical = 0.5 * norm.pdf(y_support, y1_true[0], noise_std) + 0.5 * norm.pdf(
            y_support, y2_true[0], noise_std
        )
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
    x_slug = f"{float(x_value):.4g}".replace("-", "m").replace(".", "p")
    out_path = out_dir / f"{file_tag}_conditional_pdf_x{x_slug}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
