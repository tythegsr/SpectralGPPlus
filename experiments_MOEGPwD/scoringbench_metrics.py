"""ScoringBench-style metrics for MoE-GP and Comparison.ipynb baselines.

Point metrics: MSE, RMSE, MAE (and RRMSE).
Probabilistic: CRPS, log_score, CRLS — matching the Comparison.ipynb columns.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import norm


# From ScoringBench Comparison.ipynb evaluate_models() on the bimodal toy
# (CRPS / log_score / CRLS were NaN in that run when scoring-rule eval failed).
SCORINGBENCH_BIMODAL_EXAMPLE_BASELINES: dict[str, dict[str, float | None]] = {
    "TabPFN": {
        "MSE": 29.102036,
        "RMSE": 5.394630,
        "MAE": 4.460580,
        "CRPS": None,
        "log_score": None,
        "CRLS": None,
    },
    "TabICL": {
        "MSE": 28.742635,
        "RMSE": 5.361216,
        "MAE": 4.420503,
        "CRPS": None,
        "log_score": None,
        "CRLS": None,
    },
    "Crepes(TabICL)": {
        "MSE": 30.458145,
        "RMSE": 5.518890,
        "MAE": 4.507888,
        "CRPS": None,
        "log_score": None,
        "CRLS": None,
    },
}


def _mixture_cdf_on_grid(
    y_grid: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    means = np.asarray(means, dtype=np.float64).ravel()
    stds = np.asarray(stds, dtype=np.float64).ravel()
    pi = np.asarray(weights, dtype=np.float64).ravel()
    pi = np.clip(pi, 0.0, None)
    pi = pi / max(float(pi.sum()), 1e-12)
    cdf = np.zeros_like(y_grid, dtype=np.float64)
    for j in range(means.shape[0]):
        cdf += pi[j] * norm.cdf(y_grid, means[j], max(float(stds[j]), 1e-6))
    return cdf


def _crps_one(
    y: float,
    means: np.ndarray,
    stds: np.ndarray,
    weights: np.ndarray,
    *,
    n_grid: int = 512,
    pad_sigma: float = 6.0,
) -> float:
    """Numerical CRPS = ∫ (F(z) − 1{y≤z})² dz for a Gaussian mixture."""
    means = np.asarray(means, dtype=np.float64).ravel()
    stds = np.asarray(stds, dtype=np.float64).ravel()
    lo = float(min(y, means.min() - pad_sigma * stds.max()))
    hi = float(max(y, means.max() + pad_sigma * stds.max()))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float("nan")
    z = np.linspace(lo, hi, n_grid)
    f = _mixture_cdf_on_grid(z, means, stds, weights)
    indicator = (z >= y).astype(np.float64)
    return float(np.trapz((f - indicator) ** 2, z))


def _crls_one(
    y: float,
    means: np.ndarray,
    stds: np.ndarray,
    weights: np.ndarray,
    *,
    n_grid: int = 512,
    pad_sigma: float = 6.0,
    eps: float = 1e-12,
) -> float:
    """Numerical CRLS = −∫_{−∞}^y log F − ∫_y^∞ log(1−F)."""
    means = np.asarray(means, dtype=np.float64).ravel()
    stds = np.asarray(stds, dtype=np.float64).ravel()
    lo = float(min(y, means.min() - pad_sigma * stds.max()))
    hi = float(max(y, means.max() + pad_sigma * stds.max()))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float("nan")
    z = np.linspace(lo, hi, n_grid)
    f = np.clip(_mixture_cdf_on_grid(z, means, stds, weights), eps, 1.0 - eps)
    left = z <= y
    right = z > y
    term_left = float(np.trapz(np.log(f[left]), z[left])) if left.any() else 0.0
    term_right = float(np.trapz(np.log(1.0 - f[right]), z[right])) if right.any() else 0.0
    return float(-(term_left + term_right))


def compute_scoringbench_style_metrics(
    y_true: np.ndarray,
    mixture_mean: np.ndarray,
    expert_means: np.ndarray,
    expert_stds: np.ndarray,
    gate_weights: np.ndarray,
    log_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute Comparison.ipynb metric columns for a MoE mixture predictive.

    ``expert_means``, ``expert_stds``, ``gate_weights`` have shape ``[N, K]``.
    ``log_prob`` is per-point mixture log-density (optional; else recomputed).
    """
    y = np.asarray(y_true, dtype=np.float64).ravel()
    mu = np.asarray(mixture_mean, dtype=np.float64).ravel()
    em = np.asarray(expert_means, dtype=np.float64)
    es = np.asarray(expert_stds, dtype=np.float64)
    gw = np.asarray(gate_weights, dtype=np.float64)
    if em.ndim == 1:
        em = em.reshape(-1, 1)
        es = es.reshape(-1, 1)
        gw = gw.reshape(-1, 1)

    n = y.shape[0]
    mse = float(np.mean((mu - y) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(mu - y)))
    y_std = float(np.std(y))
    rrmse = float(rmse / max(y_std, 1e-12)) if np.isfinite(y_std) else float("nan")

    if log_prob is None:
        # Fallback: univariate mixture log-density at each point
        from math import log

        lp = np.empty(n, dtype=np.float64)
        for i in range(n):
            pi_w = np.clip(gw[i], 0.0, None)
            pi_w = pi_w / max(float(pi_w.sum()), 1e-12)
            dens = 0.0
            for k in range(em.shape[1]):
                s = max(float(es[i, k]), 1e-6)
                dens += pi_w[k] * norm.pdf(y[i], em[i, k], s)
            lp[i] = log(max(dens, 1e-300))
        log_prob_arr = lp
    else:
        log_prob_arr = np.asarray(log_prob, dtype=np.float64).ravel()

    # ScoringBench log_score is negative log predictive density (lower is better)
    log_score = float(np.mean(-log_prob_arr))

    crps_vals = np.empty(n, dtype=np.float64)
    crls_vals = np.empty(n, dtype=np.float64)
    for i in range(n):
        crps_vals[i] = _crps_one(float(y[i]), em[i], es[i], gw[i])
        crls_vals[i] = _crls_one(float(y[i]), em[i], es[i], gw[i])

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "RRMSE": rrmse,
        "CRPS": float(np.nanmean(crps_vals)),
        "log_score": log_score,
        "CRLS": float(np.nanmean(crls_vals)),
        # Keep repo-style aliases used elsewhere
        "mse_mixture_mean": mse,
        "rmse_mixture_mean": rmse,
        "mae_mixture_mean": mae,
        "rrmse_mixture_mean": rrmse,
        "mixture_nll": log_score,
    }


def metrics_table_with_baselines(
    moe_metrics: dict[str, float],
    *,
    moe_name: str = "MoE-GP",
) -> list[dict[str, Any]]:
    """Rows for Comparison.ipynb-style table: baselines + this MoE run."""
    cols = ["MSE", "RMSE", "MAE", "CRPS", "log_score", "CRLS"]
    rows: list[dict[str, Any]] = []
    for name, base in SCORINGBENCH_BIMODAL_EXAMPLE_BASELINES.items():
        row: dict[str, Any] = {"Model": name}
        for c in cols:
            v = base.get(c)
            row[c] = float(v) if v is not None and np.isfinite(v) else None
        rows.append(row)
    moe_row: dict[str, Any] = {"Model": moe_name}
    for c in cols:
        v = moe_metrics.get(c)
        moe_row[c] = float(v) if v is not None and np.isfinite(float(v)) else None
    rows.append(moe_row)
    return rows


def print_metrics_table(rows: list[dict[str, Any]]) -> None:
    cols = ["Model", "MSE", "RMSE", "MAE", "CRPS", "log_score", "CRLS"]
    header = f"{'Model':18s} " + " ".join(f"{c:>10s}" for c in cols[1:])
    print(header)
    print("-" * len(header))
    for row in rows:
        parts = [f"{str(row['Model']):18s}"]
        for c in cols[1:]:
            v = row.get(c)
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                parts.append(f"{'NaN':>10s}")
            else:
                parts.append(f"{float(v):10.4f}")
        print(" ".join(parts))
