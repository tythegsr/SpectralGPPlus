"""CDF comparison vs theoretical bimodal CDF with Kolmogorov–Smirnov distance.

ScoringBench-style diagnostic: at a fixed x, plot theoretical / empirical /
MoE-GP mixture CDFs and report KS distances to the theoretical CDF.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from load_bimodal_data import (
    BIMODAL_C1,
    BIMODAL_C2,
    BIMODAL_M1,
    BIMODAL_M2,
    BIMODAL_NOISE_STD,
)


def theoretical_bimodal_cdf(
    y_support: np.ndarray,
    x_value: float,
    *,
    noise_std: float = BIMODAL_NOISE_STD,
) -> np.ndarray:
    """True conditional CDF: 0.5 N(m1*x+c1, σ²) + 0.5 N(m2*x+c2, σ²)."""
    mean1 = BIMODAL_M1 * x_value + BIMODAL_C1
    mean2 = BIMODAL_M2 * x_value + BIMODAL_C2
    return 0.5 * norm.cdf(y_support, mean1, noise_std) + 0.5 * norm.cdf(
        y_support, mean2, noise_std
    )


def mixture_gaussian_cdf(
    y_support: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """CDF of ∑_k π_k N(μ_k, σ_k²)."""
    means = np.asarray(means, dtype=np.float64).ravel()
    stds = np.asarray(stds, dtype=np.float64).ravel()
    pi = np.asarray(weights, dtype=np.float64).ravel()
    pi = np.clip(pi, 0.0, None)
    pi = pi / max(float(pi.sum()), 1e-12)
    cdf = np.zeros_like(y_support, dtype=np.float64)
    for j in range(means.shape[0]):
        cdf += pi[j] * norm.cdf(y_support, means[j], max(float(stds[j]), 1e-6))
    return cdf


def ks_distance_to_theory(
    y_support: np.ndarray,
    model_cdf: np.ndarray,
    theoretical_cdf: np.ndarray,
) -> float:
    """KS distance = max_y |F_model(y) − F_theory(y)| on the evaluation grid."""
    return float(np.max(np.abs(np.asarray(model_cdf) - np.asarray(theoretical_cdf))))


def empirical_cdf_and_ks(
    y_nearby: np.ndarray,
    y_support: np.ndarray,
    theoretical_cdf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Empirical step CDF of nearby y's and its KS distance to theory."""
    y_sorted = np.sort(np.asarray(y_nearby, dtype=np.float64).ravel())
    if y_sorted.size == 0:
        return y_sorted, np.array([]), float("nan")
    ecdf_vals = np.arange(1, y_sorted.size + 1, dtype=np.float64) / y_sorted.size
    ecdf_y_full = np.concatenate([[y_sorted[0]], y_sorted])
    ecdf_cdf_full = np.concatenate([[0.0], ecdf_vals])
    theo_at_ecdf = np.interp(ecdf_y_full, y_support, theoretical_cdf)
    ks = float(np.max(np.abs(ecdf_cdf_full - theo_at_ecdf)))
    return y_sorted, ecdf_vals, ks


def _x_file_slug(x_value: float) -> str:
    """Windows-safe slug for embedding x in a filename (no raw '.')."""
    # Prefer short integer-ish tags (x9) to avoid long / flaky Windows paths.
    rounded = round(float(x_value))
    if abs(float(x_value) - rounded) < 0.05:
        return str(int(rounded))
    return f"{float(x_value):.2f}".replace("-", "m").replace(".", "p")


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


def save_cdf_ks_comparison_plot(
    x_test: np.ndarray,
    y_test: np.ndarray,
    expert_means_at_x: np.ndarray,
    expert_stds_at_x: np.ndarray,
    gate_weights_at_x: np.ndarray,
    out_dir: str | Path,
    *,
    file_tag: str,
    x_value: float = 9.0,
    x_tolerance: float = 0.5,
    y_min: float = -2.0,
    y_max: float = 30.0,
    noise_std: float = BIMODAL_NOISE_STD,
    model_label: str = "MoE-GP",
) -> tuple[Path, dict[str, float]]:
    """
    Save CDF comparison at ``x_value`` and return (path, ks_distances).

    KS distances are measured against the theoretical ScoringBench bimodal CDF.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir = out_dir.resolve()

    x_vals = np.asarray(x_test, dtype=np.float64).ravel()
    y_vals = np.asarray(y_test, dtype=np.float64).ravel()
    mask = np.abs(x_vals - x_value) <= x_tolerance
    y_nearby = y_vals[mask] if mask.any() else y_vals

    y_support = np.linspace(y_min, y_max, 500)
    theo_cdf = theoretical_bimodal_cdf(y_support, x_value, noise_std=noise_std)
    moe_cdf = mixture_gaussian_cdf(
        y_support, expert_means_at_x, expert_stds_at_x, gate_weights_at_x
    )
    moe_ks = ks_distance_to_theory(y_support, moe_cdf, theo_cdf)
    ecdf_y, ecdf_vals, ecdf_ks = empirical_cdf_and_ks(y_nearby, y_support, theo_cdf)

    ks_distances: dict[str, float] = {
        "Empirical": ecdf_ks,
        model_label: moe_ks,
    }

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=120)
    ax.plot(
        y_support,
        theo_cdf,
        "r-",
        linewidth=2.5,
        label="Theoretical CDF",
        alpha=0.85,
        zorder=3,
    )
    if ecdf_y.size > 0:
        ax.step(
            ecdf_y,
            ecdf_vals,
            where="post",
            color="0.45",
            linewidth=1.8,
            label=f"Empirical CDF (KS={ecdf_ks:.4f})",
            alpha=0.75,
            zorder=2,
        )
    ax.plot(
        y_support,
        moe_cdf,
        color="steelblue",
        linewidth=2.2,
        label=f"{model_label} CDF (KS={moe_ks:.4f})",
        zorder=4,
    )
    ax.set_xlabel("y")
    ax.set_ylabel("CDF")
    ax.set_title(f"CDF comparison at x={x_value:g} (KS distance to theory)")
    ax.set_xlim(y_min, y_max)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()

    out_path = out_dir / f"{file_tag}_cdf_ks_x{_x_file_slug(x_value)}.png"
    out_path = _safe_savefig(fig, out_path)
    plt.close(fig)

    print("Kolmogorov-Smirnov distances to theoretical CDF:")
    for name in sorted(ks_distances.keys()):
        print(f"  {name:20s}: {ks_distances[name]:.6f}")

    return out_path, ks_distances
