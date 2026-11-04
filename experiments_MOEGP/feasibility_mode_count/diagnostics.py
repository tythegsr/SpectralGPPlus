"""Diagnostics for mode-count vs expert-count feasibility probes."""

from __future__ import annotations

import numpy as np
import torch


def count_density_peaks(
    y_grid: np.ndarray,
    density: np.ndarray,
    *,
    rel_height: float = 0.05,
    min_separation: float | None = None,
) -> int:
    """Count local maxima of a 1D density above ``rel_height * max``."""
    y_grid = np.asarray(y_grid, dtype=np.float64).ravel()
    d = np.asarray(density, dtype=np.float64).ravel()
    if d.size < 3:
        return int(d.size > 0)
    thr = float(rel_height) * float(d.max() + 1e-12)
    peaks_idx: list[int] = []
    for i in range(1, d.size - 1):
        if d[i] >= d[i - 1] and d[i] > d[i + 1] and d[i] >= thr:
            peaks_idx.append(i)
    if not peaks_idx:
        return 0
    if min_separation is None:
        # Default: 5% of y-range
        min_separation = 0.05 * float(y_grid.max() - y_grid.min() + 1e-12)
    kept: list[int] = []
    for i in sorted(peaks_idx, key=lambda j: d[j], reverse=True):
        if all(abs(y_grid[i] - y_grid[j]) >= min_separation for j in kept):
            kept.append(i)
    return len(kept)


def mixture_density_on_grid(
    y_grid: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """p(y) = ∑ π_k N(y; μ_k, σ_k²) on a 1D grid."""
    from scipy.stats import norm

    y_grid = np.asarray(y_grid, dtype=np.float64).ravel()
    means = np.asarray(means, dtype=np.float64).ravel()
    stds = np.asarray(stds, dtype=np.float64).ravel()
    pi = np.asarray(weights, dtype=np.float64).ravel()
    pi = np.clip(pi, 0.0, None)
    pi = pi / max(float(pi.sum()), 1e-12)
    dens = np.zeros_like(y_grid)
    for k in range(means.shape[0]):
        dens += pi[k] * norm.pdf(y_grid, means[k], max(float(stds[k]), 1e-6))
    return dens


def posthoc_k_modes_at_x(
    model,
    x_value: float,
    *,
    y_min: float = -2.0,
    y_max: float = 30.0,
    n_grid: int = 500,
) -> dict:
    """Peak-count of mixture predictive density at a single x."""
    x = torch.tensor([[x_value]], dtype=model.dtype, device=model.train_x.device)
    pred = model.predict(x)
    means = pred["expert_means"].detach().cpu().numpy().ravel()
    stds = pred["expert_stds"].detach().cpu().numpy().ravel()
    weights = pred["gate_weights"].detach().cpu().numpy().ravel()
    y_grid = np.linspace(y_min, y_max, n_grid)
    dens = mixture_density_on_grid(y_grid, means, stds, weights)
    k_modes = count_density_peaks(y_grid, dens)
    return {
        "x": float(x_value),
        "k_modes": int(k_modes),
        "means": means.tolist(),
        "stds": stds.tolist(),
        "weights": weights.tolist(),
    }


def average_posthoc_k_modes(
    model,
    x_values: list[float] | np.ndarray,
    **kwargs,
) -> dict:
    """Mean/median peak count of p(y|x) over a list of x."""
    rows = [posthoc_k_modes_at_x(model, float(x), **kwargs) for x in x_values]
    ks = np.array([r["k_modes"] for r in rows], dtype=np.float64)
    return {
        "per_x": rows,
        "k_modes_mean": float(ks.mean()),
        "k_modes_median": float(np.median(ks)),
        "k_modes_round": int(np.rint(ks.mean())),
    }


def expert_activity(
    model,
    *,
    mass_eps: float = 0.05,
) -> dict:
    """Expert usage from last soft responsibilities / gate."""
    k = model.num_experts
    if getattr(model, "_soft_responsibilities_last", None) is not None:
        r = model._soft_responsibilities_last.detach().cpu().numpy()
        mean_r = r.mean(axis=0)
    else:
        mean_r = np.full(k, 1.0 / k)
    mean_r = mean_r / max(float(mean_r.sum()), 1e-12)
    active = int((mean_r >= mass_eps).sum())
    # Diversity proxy: pairwise |corr| of expert means on train x
    with torch.no_grad():
        means = []
        for expert in model.experts:
            mu, _ = model._expert_predictive(expert, model.train_x)
            means.append(mu.detach().cpu().numpy().ravel())
    corr_abs = []
    for i in range(k):
        for j in range(i + 1, k):
            a = means[i] - means[i].mean()
            b = means[j] - means[j].mean()
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            if denom < 1e-12:
                corr_abs.append(1.0)
            else:
                corr_abs.append(abs(float(np.dot(a, b) / denom)))
    mean_abs_corr = float(np.mean(corr_abs)) if corr_abs else 0.0
    # Soft K_eff: usage * (1 - mean pairwise |corr| contribution)
    a_k = mean_r * (1.0 - mean_abs_corr)  # shared diversity factor (simple probe)
    # Better per-expert: usage only for probe
    k_eff_usage = float((mean_r / mean_r.max().clip(1e-12)).sum()) if mean_r.max() > 0 else 0.0
    return {
        "mean_r": mean_r.tolist(),
        "k_experts_active": active,
        "k_eff_usage_sum": float(mean_r.sum()),  # =1
        "k_eff_renorm_sum": float(k_eff_usage),
        "mean_abs_expert_corr": mean_abs_corr,
        "a_k_usage": mean_r.tolist(),
    }


def mixture_nll_numpy(y: np.ndarray, means: np.ndarray, stds: np.ndarray, weights: np.ndarray) -> float:
    """Mean −log ∑ π_k N(y; μ_k, σ_k) for batch rows."""
    from scipy.stats import norm

    y = np.asarray(y, dtype=np.float64).ravel()
    means = np.asarray(means, dtype=np.float64)
    stds = np.asarray(stds, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if means.ndim == 1:
        means = means.reshape(1, -1)
        stds = stds.reshape(1, -1)
        w = w.reshape(1, -1)
    n, k = means.shape
    log_comp = np.zeros((n, k))
    for j in range(k):
        log_comp[:, j] = np.log(np.clip(w[:, j], 1e-12, None)) + norm.logpdf(
            y, means[:, j], np.clip(stds[:, j], 1e-6, None)
        )
    return float(-np.mean(np.logaddexp.reduce(log_comp, axis=1)))


def _pred_components(pred: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (means, stds, weights) from MOEGPR or MidlineSparseMoE predict dict."""
    means = pred["expert_means"].detach().cpu().numpy()
    stds = pred["expert_stds"].detach().cpu().numpy()
    weights = pred["gate_weights"].detach().cpu().numpy()
    if means.ndim == 1:
        means = means.reshape(1, -1)
        stds = stds.reshape(1, -1)
        weights = weights.reshape(1, -1)
    return means, stds, weights


def dual_report(
    model,
    x_values: list[float] | np.ndarray,
    *,
    y_min: float = -2.0,
    y_max: float = 30.0,
    n_grid: int = 400,
    mass_eps: float = 0.05,
    include_soft_peak_count: bool = True,
) -> dict:
    """Dual-report ``K_modes`` (peaks) vs expert activity / ``K_eff``.

    Works for models exposing ``predict`` with ``expert_means``, ``expert_stds``,
    ``gate_weights``. If the model has ``k_eff`` / ``k_hat`` / ``alpha``, those
    are preferred for expert-side metrics.
    """
    from soft_peakness import soft_peak_count

    x_values = [float(x) for x in x_values]
    dtype = getattr(model, "dtype", torch.float64)
    device = model.train_x.device if hasattr(model, "train_x") else "cpu"

    per_x = []
    soft_counts = []
    for xv in x_values:
        x = torch.tensor([[xv]], dtype=dtype, device=device)
        pred = model.predict(x)
        means, stds, weights = _pred_components(pred)
        m0, s0, w0 = means[0], stds[0], weights[0]
        y_grid = np.linspace(y_min, y_max, n_grid)
        dens = mixture_density_on_grid(y_grid, m0, s0, w0)
        k_modes = count_density_peaks(y_grid, dens)
        row = {
            "x": xv,
            "k_modes": int(k_modes),
            "means": m0.tolist(),
            "stds": s0.tolist(),
            "weights": w0.tolist(),
        }
        if include_soft_peak_count:
            spc = soft_peak_count(y_grid, dens)
            row["soft_peak_count"] = spc
            soft_counts.append(spc)
        per_x.append(row)

    ks = np.array([r["k_modes"] for r in per_x], dtype=np.float64)
    out: dict = {
        "per_x": per_x,
        "k_modes_mean": float(ks.mean()),
        "k_modes_median": float(np.median(ks)),
        "k_modes_round": int(np.rint(ks.mean())),
    }
    if soft_counts:
        out["soft_peak_count_mean"] = float(np.mean(soft_counts))

    # Expert-side metrics
    if hasattr(model, "k_eff") and callable(model.k_eff):
        with torch.no_grad():
            ke = float(model.k_eff().detach().cpu())
        out["k_eff"] = ke
        if hasattr(model, "k_hat") and callable(model.k_hat):
            out["k_hat"] = int(model.k_hat(eps=mass_eps))
        if hasattr(model, "alpha") and callable(getattr(model, "alpha", None)):
            pass
        if hasattr(model, "get_alpha"):
            with torch.no_grad():
                a = model.get_alpha().detach().cpu().numpy().ravel()
            out["alpha"] = a.tolist()
            out["k_experts_active"] = int((a >= mass_eps).sum())
        elif "k_hat" in out:
            out["k_experts_active"] = out["k_hat"]
    else:
        act = expert_activity(model, mass_eps=mass_eps)
        out["k_experts_active"] = act["k_experts_active"]
        out["mean_r"] = act["mean_r"]
        out["mean_abs_expert_corr"] = act["mean_abs_expert_corr"]
        out["k_eff_renorm_sum"] = act["k_eff_renorm_sum"]

    return out


def recovery_certificate(
    model,
    x_values: list[float] | np.ndarray,
    *,
    y_min: float = -2.0,
    y_max: float = 30.0,
    n_grid: int = 400,
    rec_margin: float = 1.0,
    mass_eps: float = 0.05,
) -> dict:
    """Post-fit certificate: fitted mixture lies in a multimodal-separation regime.

    Does **not** compare to ground truth. Checks standardized mean gaps
    ``g_ij = |μ_i-μ_j|/(σ_i+σ_j)`` among active experts (απ ≥ mass_eps) and
    reports ``K_modes`` / ``K_hat`` at each probe ``x``.
    """
    x_values = [float(x) for x in x_values]
    dtype = getattr(model, "dtype", torch.float64)
    device = model.train_x.device if hasattr(model, "train_x") else "cpu"

    alpha = None
    if hasattr(model, "get_alpha"):
        with torch.no_grad():
            alpha = model.get_alpha(deterministic=True).detach().cpu().numpy().ravel()

    per_x: list[dict] = []
    for xv in x_values:
        x = torch.tensor([[xv]], dtype=dtype, device=device)
        pred = model.predict(x)
        means, stds, weights = _pred_components(pred)
        m0, s0, w0 = means[0], stds[0], weights[0]
        if alpha is None:
            mass = w0
        else:
            mass = alpha * w0
        active = np.where(mass >= mass_eps)[0]
        gaps: list[dict] = []
        min_gap = float("inf")
        for ii in range(len(active)):
            for jj in range(ii + 1, len(active)):
                i, j = int(active[ii]), int(active[jj])
                g = float(abs(m0[i] - m0[j]) / (float(s0[i]) + float(s0[j]) + 1e-12))
                gaps.append({"i": i, "j": j, "gap": g})
                min_gap = min(min_gap, g)
        if not gaps:
            min_gap = float("nan")
            sep_ok = len(active) <= 1
        else:
            sep_ok = bool(min_gap >= rec_margin)

        y_grid = np.linspace(y_min, y_max, n_grid)
        dens = mixture_density_on_grid(y_grid, m0, s0, w0)
        k_modes = int(count_density_peaks(y_grid, dens))
        k_hat = int(len(active))
        per_x.append(
            {
                "x": xv,
                "k_modes": k_modes,
                "k_hat": k_hat,
                "n_active": k_hat,
                "min_gap": None if np.isnan(min_gap) else float(min_gap),
                "gaps": gaps,
                "modes_identifiable_by_separation": sep_ok,
                "k_modes_matches_k_hat": bool(k_modes == k_hat),
            }
        )

    n = len(per_x)
    cert_rate = float(
        sum(1 for r in per_x if r["modes_identifiable_by_separation"]) / max(n, 1)
    )
    match_rate = float(
        sum(1 for r in per_x if r["k_modes_matches_k_hat"]) / max(n, 1)
    )
    return {
        "rec_margin": float(rec_margin),
        "mass_eps": float(mass_eps),
        "per_x": per_x,
        "certificate_pass_rate": cert_rate,
        "k_modes_matches_k_hat_rate": match_rate,
        "certificate_pass": bool(cert_rate >= 0.8),
    }
