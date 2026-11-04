"""Posterior predictive densities for foundation-model marginal plots.

TabPFN exposes a bar distribution (borders + logits). Density in bucket ``k`` is
``softmax(logits)_k / width_k`` — piecewise constant on the border grid.

KumoTabular returns 999 quantiles (``q001``–``q999``); we rebuild a density via
finite differences of that quantile CDF. (Legacy TabFM was point-only.)
"""

from __future__ import annotations

import numpy as np


def density_from_bar_distribution(
    y_grid: np.ndarray,
    borders: np.ndarray,
    logits: np.ndarray,
) -> np.ndarray:
    """Piecewise-constant PDF from TabPFN ``FullSupportBarDistribution``.

    Parameters
    ----------
    y_grid :
        Evaluation points, shape ``[G]``.
    borders :
        Sorted bucket edges, shape ``[B+1]``.
    logits :
        Per-point bar logits, shape ``[B]`` (one test location).

    Returns
    -------
    dens : ndarray, shape ``[G]``
        Density values (0 outside the support).
    """
    y = np.asarray(y_grid, dtype=np.float64).ravel()
    b = np.asarray(borders, dtype=np.float64).ravel()
    logits = np.asarray(logits, dtype=np.float64).ravel()
    if b.size < 2 or logits.size != b.size - 1:
        return np.zeros_like(y)

    # Stable softmax
    z = logits - np.max(logits)
    p = np.exp(z)
    p = p / max(float(p.sum()), 1e-300)
    widths = np.clip(np.diff(b), 1e-12, None)
    bar_dens = p / widths  # density value inside each bar

    # searchsorted: index i means b[i-1] <= y < b[i] for right side
    idx = np.searchsorted(b, y, side="right") - 1
    dens = np.zeros_like(y)
    inside = (idx >= 0) & (idx < bar_dens.size) & (y >= b[0]) & (y <= b[-1])
    dens[inside] = bar_dens[idx[inside]]
    return dens


def density_from_quantiles(
    y_grid: np.ndarray,
    quantile_levels: np.ndarray,
    quantile_values: np.ndarray,
) -> np.ndarray:
    """Approximate PDF from a quantile function via finite differences.

    For consecutive levels ``(q_i, q_{i+1})`` with values ``(y_i, y_{i+1})``,
    density on ``[y_i, y_{i+1}]`` is ``(q_{i+1}-q_i) / (y_{i+1}-y_i)``.
    """
    y = np.asarray(y_grid, dtype=np.float64).ravel()
    q = np.asarray(quantile_levels, dtype=np.float64).ravel()
    v = np.asarray(quantile_values, dtype=np.float64).ravel()
    if q.size < 2 or v.size != q.size:
        return np.zeros_like(y)

    order = np.argsort(v)
    v = v[order]
    q = q[order]
    # Enforce nondecreasing CDF after sorting by y (ties → tiny eps)
    for i in range(1, v.size):
        if v[i] <= v[i - 1]:
            v[i] = v[i - 1] + 1e-9
        if q[i] < q[i - 1]:
            q[i] = q[i - 1]

    dens = np.zeros_like(y)
    for i in range(v.size - 1):
        lo, hi = v[i], v[i + 1]
        dq = max(float(q[i + 1] - q[i]), 0.0)
        dy = max(float(hi - lo), 1e-12)
        d = dq / dy
        mask = (y >= lo) & (y < hi if i < v.size - 2 else y <= hi)
        dens[mask] = d
    return dens


def density_from_ensemble_samples(
    y_grid: np.ndarray,
    samples: np.ndarray,
    *,
    n_bins: int | None = None,
) -> np.ndarray:
    """Histogram density from per-estimator predictive samples (TabFM ensemble).

    Builds equal-width bins over the sample range (or ``y_grid`` span when the
    sample range is tiny) and evaluates the piecewise-constant density on
    ``y_grid`` — same spirit as TabPFN's bar PPD.
    """
    y = np.asarray(y_grid, dtype=np.float64).ravel()
    s = np.asarray(samples, dtype=np.float64).ravel()
    s = s[np.isfinite(s)]
    if s.size < 1:
        return np.zeros_like(y)
    if s.size == 1 or float(np.std(s)) < 1e-12:
        dens = np.zeros_like(y)
        j = int(np.argmin(np.abs(y - float(np.mean(s)))))
        dy = float(y[1] - y[0]) if y.size > 1 else 1.0
        dens[j] = 1.0 / max(dy, 1e-12)
        return dens

    # Prefer ~sqrt(E) bins, at least 4, at most 32
    nb = int(n_bins) if n_bins is not None else int(np.clip(np.sqrt(s.size) * 2, 4, 32))
    lo = float(min(s.min(), y.min()))
    hi = float(max(s.max(), y.max()))
    if hi <= lo:
        hi = lo + 1e-6
    counts, edges = np.histogram(s, bins=nb, range=(lo, hi), density=True)
    # Map y_grid → bin density (0 outside)
    idx = np.searchsorted(edges, y, side="right") - 1
    dens = np.zeros_like(y)
    inside = (idx >= 0) & (idx < counts.size)
    dens[inside] = counts[idx[inside]]
    return dens


def model_predictive_density(
    y_grid: np.ndarray,
    pred: dict,
    *,
    probe_index: int,
) -> tuple[np.ndarray | None, str]:
    """Return ``(density, source)`` for one probe row of a model prediction dict.

    Preference order:
      1. TabPFN bar distribution (``pred_borders`` + ``pred_logits``)
      2. Quantile CDF finite differences
      3. Ensemble-sample histogram (legacy TabFM path)
      4. ``None`` → caller falls back to Gaussian mixture PDF
    """
    borders = pred.get("pred_borders")
    logits = pred.get("pred_logits")
    if borders is not None and logits is not None:
        log_i = np.asarray(logits, dtype=np.float64)
        if log_i.ndim == 1:
            log_row = log_i
        else:
            log_row = log_i[probe_index]
        dens = density_from_bar_distribution(y_grid, borders, log_row)
        return dens, "bar_distribution"

    q_levels = pred.get("quantile_levels")
    q_vals = pred.get("quantile_values")
    if q_levels is not None and q_vals is not None:
        qv = np.asarray(q_vals, dtype=np.float64)
        if qv.ndim == 1:
            row = qv
        else:
            row = qv[probe_index]
        dens = density_from_quantiles(y_grid, q_levels, row)
        return dens, "quantile_diff"

    ens = pred.get("ensemble_samples")
    if ens is not None:
        e = np.asarray(ens, dtype=np.float64)
        row = e if e.ndim == 1 else e[probe_index]
        dens = density_from_ensemble_samples(y_grid, row)
        return dens, "ensemble_samples"

    return None, "mixture_gaussian"
