"""Calibration diagnostics for TOA log-grain predictions."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.special import erf

from gpplus.utils.metrics_functions import compute_nis

GRAIN_IDX = 1
COS_IDX = 0


def _normal_cdf(z: np.ndarray | float) -> np.ndarray | float:
    z_arr = np.asarray(z, dtype=np.float64)
    out = 0.5 * (1.0 + erf(z_arr / math.sqrt(2.0)))
    if np.isscalar(z):
        return float(out)
    return out


def _load_npz_arrays(npz_path: str | Path) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    out = {
        "y_true": np.asarray(data["y_true"], dtype=np.float64),
        "y_pred": np.asarray(data["y_pred"], dtype=np.float64),
        "lower": np.asarray(data["lower"], dtype=np.float64),
        "upper": np.asarray(data["upper"], dtype=np.float64),
        "log_grain": bool(data["log_grain"].item()) if "log_grain" in data else False,
    }
    if "log_mu" in data:
        out["log_mu"] = np.asarray(data["log_mu"], dtype=np.float64)
    if "log_sigma" in data:
        out["log_sigma"] = np.asarray(data["log_sigma"], dtype=np.float64)
    if "y_pred_mean" in data:
        out["y_pred_mean"] = np.asarray(data["y_pred_mean"], dtype=np.float64)
    return out


def compute_grain_calibration_metrics(
    y_true_grain: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    log_mu: np.ndarray,
    log_sigma: np.ndarray,
    *,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Scalar calibration metrics for grain on original and log scales."""
    y_true_grain = np.asarray(y_true_grain, dtype=np.float64).ravel()
    lower = np.asarray(lower, dtype=np.float64).ravel()
    upper = np.asarray(upper, dtype=np.float64).ravel()
    log_mu = np.asarray(log_mu, dtype=np.float64).ravel()
    log_sigma = np.maximum(np.asarray(log_sigma, dtype=np.float64).ravel(), 0.0)

    valid = (
        np.isfinite(y_true_grain)
        & (y_true_grain > 0)
        & np.isfinite(log_mu)
        & np.isfinite(log_sigma)
        & (log_sigma > 0)
    )
    if not valid.any():
        return {
            "y_grain_coverage_95": float("nan"),
            "y_grain_pit_ks_stat": float("nan"),
            "y_grain_pit_ks_pvalue": float("nan"),
            "y_grain_nis": float("nan"),
        }

    yt = y_true_grain[valid]
    lo = lower[valid]
    hi = upper[valid]
    mu = log_mu[valid]
    sig = log_sigma[valid]

    coverage = float(np.mean((yt >= lo) & (yt <= hi)))
    z = (np.log(yt) - mu) / sig
    pit = _normal_cdf(z)
    ks_stat, ks_p = stats.kstest(pit, "uniform")
    nis = compute_nis(yt, lower=lo, upper=hi, alpha=alpha)

    return {
        "y_grain_coverage_95": coverage,
        "y_grain_pit_ks_stat": float(ks_stat),
        "y_grain_pit_ks_pvalue": float(ks_p),
        "y_grain_nis": float(nis["NIS"]),
        "y_grain_nis_width": float(nis["NIS_width"]),
        "y_grain_nis_outside": float(nis["NIS_outside"]),
    }


def _plot_pit_histogram(pit: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=120)
    ax.hist(pit, bins=20, range=(0.0, 1.0), density=True, color="C0", alpha=0.7, edgecolor="white")
    ax.axhline(1.0, color="C1", linestyle="--", linewidth=1.2, label="Uniform(0,1)")
    ax.set_xlabel("PIT value")
    ax.set_ylabel("density")
    ax.set_title("Grain: PIT (log-space calibration)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_qq_residuals(z: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=120)
    stats.probplot(z, dist="norm", plot=ax)
    ax.set_title("Grain: standardized log residuals QQ")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_grain_histogram_overlay(
    y_true: np.ndarray,
    log_mu: np.ndarray,
    log_sigma: np.ndarray,
    out_path: Path,
    *,
    n_samples: int = 5000,
    seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    log_mu = np.asarray(log_mu, dtype=np.float64).ravel()
    log_sigma = np.maximum(np.asarray(log_sigma, dtype=np.float64).ravel(), 1e-12)

    valid = np.isfinite(y_true) & (y_true > 0) & np.isfinite(log_mu) & np.isfinite(log_sigma)
    yt = y_true[valid]
    mu = log_mu[valid]
    sig = log_sigma[valid]
    if yt.size == 0:
        return

    idx = rng.choice(yt.size, size=min(n_samples, yt.size), replace=True)
    samples = np.exp(rng.normal(loc=mu[idx], scale=sig[idx]))

    fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=120)
    bins = np.histogram_bin_edges(yt, bins=30)
    ax.hist(yt, bins=bins, density=True, alpha=0.55, color="C2", label="test true grain")
    ax.hist(samples, bins=bins, density=True, alpha=0.45, color="C0", label="log-normal samples")
    ax.set_xlabel("grain size (µm)")
    ax.set_ylabel("density")
    ax.set_title("Grain: true vs predictive log-normal samples")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_toa_distribution_diagnostics(
    npz_path: str | Path,
    save_dir: str | Path,
    *,
    log_grain: bool = True,
) -> dict[str, float]:
    """
    Run grain calibration diagnostics from a predictions NPZ file.

    Writes PNGs under ``save_dir`` and returns scalar metrics for JSON logging.
    """
    arrays = _load_npz_arrays(npz_path)
    if not log_grain or not arrays["log_grain"]:
        return {}

    y_true = arrays["y_true"]
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 2)
    grain_true = y_true[:, GRAIN_IDX]
    lower = arrays["lower"][:, GRAIN_IDX] if arrays["lower"].ndim > 1 else arrays["lower"]
    upper = arrays["upper"][:, GRAIN_IDX] if arrays["upper"].ndim > 1 else arrays["upper"]

    if "log_mu" not in arrays or "log_sigma" not in arrays:
        return {}

    log_mu = arrays["log_mu"][:, GRAIN_IDX] if arrays["log_mu"].ndim > 1 else arrays["log_mu"]
    log_sigma = (
        arrays["log_sigma"][:, GRAIN_IDX]
        if arrays["log_sigma"].ndim > 1
        else arrays["log_sigma"]
    )

    metrics = compute_grain_calibration_metrics(
        grain_true, lower, upper, log_mu, log_sigma
    )

    save_dir = Path(save_dir)
    valid = (
        np.isfinite(grain_true)
        & (grain_true > 0)
        & np.isfinite(log_mu)
        & np.isfinite(log_sigma)
        & (log_sigma > 0)
    )
    if valid.any():
        z = (np.log(grain_true[valid]) - log_mu[valid]) / log_sigma[valid]
        pit = _normal_cdf(z)
        _plot_pit_histogram(pit, save_dir / "grain_pit_histogram.png")
        _plot_qq_residuals(z, save_dir / "grain_log_residuals_qq.png")
        _plot_grain_histogram_overlay(
            grain_true[valid],
            log_mu[valid],
            log_sigma[valid],
            save_dir / "grain_histogram_overlay.png",
        )

    return metrics
