"""PIT, interval coverage, and sharpness for Gaussian-mixture predictives."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from scoringbench_metrics import _mixture_cdf_on_grid, mixture_quantile


def _safe_savefig(fig, out_path: Path, *, retries: int = 6, dpi: int = 140) -> Path:
    """Save PNG with unlink/retry (Windows often locks open preview files)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    last_err: OSError | None = None
    for attempt in range(retries):
        try:
            if out_path.exists():
                try:
                    out_path.unlink()
                except OSError:
                    pass
            fig.savefig(str(out_path), bbox_inches="tight", dpi=dpi)
            return out_path
        except OSError as err:
            last_err = err
            time.sleep(0.2 * (attempt + 1))
    # Last resort: write beside the locked file so the run can finish.
    fallback = out_path.with_name(
        f"{out_path.stem}_{int(time.time())}{out_path.suffix}"
    )
    try:
        fig.savefig(str(fallback), bbox_inches="tight", dpi=dpi)
        print(f"Warning: could not write {out_path.name}; wrote {fallback.name}")
        return fallback
    except OSError:
        if last_err is not None:
            raise last_err
        raise


def mixture_cdf_at_y(
    y: np.ndarray,
    expert_means: np.ndarray,
    expert_stds: np.ndarray,
    gate_weights: np.ndarray,
) -> np.ndarray:
    """Per-point mixture CDF F(y_i | x_i)."""
    y = np.asarray(y, dtype=np.float64).ravel()
    em = np.asarray(expert_means, dtype=np.float64)
    es = np.asarray(expert_stds, dtype=np.float64)
    gw = np.asarray(gate_weights, dtype=np.float64)
    if em.ndim == 1:
        em = em.reshape(-1, 1)
        es = es.reshape(-1, 1)
        gw = gw.reshape(-1, 1)
    out = np.empty(y.shape[0], dtype=np.float64)
    for i in range(y.shape[0]):
        out[i] = float(
            _mixture_cdf_on_grid(
                np.asarray([y[i]], dtype=np.float64),
                em[i],
                es[i],
                gw[i],
            )[0]
        )
    return np.clip(out, 0.0, 1.0)


def mixture_central_interval_bands(
    expert_means: np.ndarray,
    expert_stds: np.ndarray,
    gate_weights: np.ndarray,
    *,
    levels: tuple[float, ...] = (0.5, 0.9),
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Per-row central predictive interval bounds from mixture quantiles."""
    em = np.asarray(expert_means, dtype=np.float64)
    es = np.asarray(expert_stds, dtype=np.float64)
    gw = np.asarray(gate_weights, dtype=np.float64)
    if em.ndim == 1:
        em = em.reshape(-1, 1)
        es = es.reshape(-1, 1)
        gw = gw.reshape(-1, 1)
    n = int(em.shape[0])
    out: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for lev in levels:
        alpha = 1.0 - float(lev)
        q_lo, q_hi = alpha / 2.0, 1.0 - alpha / 2.0
        lower = np.empty(n, dtype=np.float64)
        upper = np.empty(n, dtype=np.float64)
        for i in range(n):
            lower[i] = mixture_quantile(q_lo, em[i], es[i], gw[i])
            upper[i] = mixture_quantile(q_hi, em[i], es[i], gw[i])
        out[float(lev)] = (lower, upper)
    return out


def interval_coverage(
    y: np.ndarray,
    expert_means: np.ndarray,
    expert_stds: np.ndarray,
    gate_weights: np.ndarray,
    *,
    levels: tuple[float, ...] = (0.5, 0.9),
) -> dict[str, float]:
    """Empirical coverage and mean width of central predictive intervals."""
    y = np.asarray(y, dtype=np.float64).ravel()
    bands = mixture_central_interval_bands(
        expert_means, expert_stds, gate_weights, levels=levels
    )

    out: dict[str, float] = {}
    for lev in levels:
        lower, upper = bands[float(lev)]
        covered = (y >= lower) & (y <= upper)
        tag = f"{int(round(100 * lev))}"
        out[f"cov{tag}"] = float(np.mean(covered))
        out[f"width{tag}"] = float(np.mean(upper - lower))
        out[f"cov{tag}_gap"] = float(out[f"cov{tag}"] - lev)
    return out


def compute_uq_metrics(
    y: np.ndarray,
    mixture_mean: np.ndarray,
    expert_means: np.ndarray,
    expert_stds: np.ndarray,
    gate_weights: np.ndarray,
    *,
    levels: tuple[float, ...] = (0.5, 0.9),
) -> dict[str, float]:
    """Coverage + PIT summary stats (mean, std, KS-ish max |F̂−U|)."""
    from scoringbench_metrics import compute_scoringbench_style_metrics

    base = compute_scoringbench_style_metrics(
        y, mixture_mean, expert_means, expert_stds, gate_weights
    )
    cov = interval_coverage(
        y, expert_means, expert_stds, gate_weights, levels=levels
    )
    pit = mixture_cdf_at_y(y, expert_means, expert_stds, gate_weights)
    # Uniformity proxy: max abs deviation of empirical CDF of PIT from identity
    pit_s = np.sort(pit)
    n = max(len(pit_s), 1)
    u = (np.arange(1, n + 1) - 0.5) / n
    pit_ks = float(np.max(np.abs(pit_s - u))) if n else float("nan")
    out = {
        "RMSE": float(base["RMSE"]),
        "RRMSE": float(base["RRMSE"]),
        "CRPS": float(base["CRPS"]),
        "NIS": float(base.get("NIS", float("nan"))),
        "log_score": float(base["log_score"]),
        "pit_mean": float(np.mean(pit)),
        "pit_std": float(np.std(pit)),
        "pit_ks": pit_ks,
        **cov,
    }
    return out


def save_coverage_bar_chart(
    *,
    coverage_rows: dict[str, dict[str, float]],
    labels: dict[str, str],
    order: list[str],
    out_path: Path,
    title: str = "Central predictive interval coverage (50% / 90%)",
) -> Path:
    """Grouped bars of empirical cov50 / cov90 vs nominal levels."""
    keys = [k for k in order if k in coverage_rows]
    if not keys:
        raise ValueError("no coverage rows for bar chart")

    x = np.arange(len(keys))
    w = 0.36
    cov50 = [float(coverage_rows[k].get("cov50", float("nan"))) for k in keys]
    cov90 = [float(coverage_rows[k].get("cov90", float("nan"))) for k in keys]

    fig, ax = plt.subplots(figsize=(max(5.5, 1.4 * len(keys) + 2.0), 3.6), dpi=140)
    ax.bar(x - w / 2, cov50, width=w, label="50% interval cov", color="#3182ce")
    ax.bar(x + w / 2, cov90, width=w, label="90% interval cov", color="#38a169")
    ax.axhline(0.5, color="#3182ce", ls=":", lw=1.3, alpha=0.85)
    ax.axhline(0.9, color="#38a169", ls=":", lw=1.3, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(k, k) for k in keys], rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("empirical coverage")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.92)
    ax.grid(True, axis="y", alpha=0.28)
    fig.tight_layout()
    path = _safe_savefig(fig, out_path, dpi=140)
    plt.close(fig)
    return path


def save_pit_coverage_figure(
    *,
    problem_name: str,
    model_pits: dict[str, np.ndarray],
    model_labels: dict[str, str],
    model_order: list[str],
    coverage_rows: dict[str, dict[str, float]],
    out_path: Path,
) -> Path:
    """PIT histograms (top) + coverage bar comparison (bottom)."""
    keys = [k for k in model_order if k in model_pits]
    n_m = len(keys)
    if n_m == 0:
        raise ValueError("no PIT arrays")

    fig = plt.figure(figsize=(3.6 * n_m, 6.2), dpi=140)
    gs = fig.add_gridspec(2, n_m, height_ratios=[1.35, 1.0], hspace=0.38, wspace=0.28)

    for j, mk in enumerate(keys):
        ax = fig.add_subplot(gs[0, j])
        pit = np.asarray(model_pits[mk], dtype=np.float64)
        ax.hist(pit, bins=10, range=(0.0, 1.0), density=True, color="#4a5568", alpha=0.85)
        ax.axhline(1.0, color="#e53e3e", ls="--", lw=1.4, label="Uniform")
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("PIT")
        if j == 0:
            ax.set_ylabel("density")
        ax.set_title(model_labels.get(mk, mk), fontsize=10)
        ax.grid(True, alpha=0.28)
        if j == 0:
            ax.legend(fontsize=8)

    axc = fig.add_subplot(gs[1, :])
    x = np.arange(n_m)
    w = 0.35
    cov50 = [float(coverage_rows[k].get("cov50", float("nan"))) for k in keys]
    cov90 = [float(coverage_rows[k].get("cov90", float("nan"))) for k in keys]
    axc.bar(x - w / 2, cov50, width=w, label="cov50", color="#3182ce")
    axc.bar(x + w / 2, cov90, width=w, label="cov90", color="#38a169")
    axc.axhline(0.5, color="#3182ce", ls=":", lw=1.2, alpha=0.8)
    axc.axhline(0.9, color="#38a169", ls=":", lw=1.2, alpha=0.8)
    axc.set_xticks(x)
    axc.set_xticklabels([model_labels.get(k, k) for k in keys], rotation=15, ha="right")
    axc.set_ylim(0.0, 1.05)
    axc.set_ylabel("empirical coverage")
    axc.set_title("Central predictive interval coverage vs nominal")
    axc.legend(fontsize=9, loc="lower right")
    axc.grid(True, axis="y", alpha=0.28)

    fig.suptitle(f"{problem_name}: PIT + coverage (UQ diagnostics)", fontsize=12, y=1.01)
    path = _safe_savefig(fig, out_path, dpi=140)
    plt.close(fig)
    return path


def annotate_expert_components(
    ax: Any,
    y_grid: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    weights: np.ndarray,
    *,
    mass_min: float = 0.05,
) -> None:
    """Overlay per-expert Gaussians + μ markers on a marginal panel."""
    means = np.asarray(means, dtype=np.float64).ravel()
    stds = np.asarray(stds, dtype=np.float64).ravel()
    pi = np.clip(np.asarray(weights, dtype=np.float64).ravel(), 0.0, None)
    pi = pi / max(float(pi.sum()), 1e-12)
    cmap = plt.cm.tab10
    for k in range(means.shape[0]):
        if float(pi[k]) < mass_min:
            continue
        dens_k = pi[k] * norm.pdf(y_grid, means[k], max(float(stds[k]), 1e-6))
        ax.plot(
            y_grid,
            dens_k,
            color=cmap(k % 10),
            lw=1.1,
            ls=":",
            alpha=0.75,
            label=rf"$k={k}$ π={pi[k]:.2f} σ={stds[k]:.2f}",
        )
        ax.axvline(means[k], color=cmap(k % 10), ls=":", lw=0.9, alpha=0.55)
