"""Histogram / bar utilities for multi-output and Dirichlet GP baselines.

Mirrors TabPFN's piecewise-constant bar density interface (borders + logits).
Default hist MoGP bins are **equal-width** on the padded training range so
``p=π/w`` is a faithful piecewise density (quantile bins force tiny bulk
widths and spikey ``π/w``). ``B`` is capped at ``n_train``: hard one-hots
cannot occupy more bins than observations. See FOUNDATION_COMPARE.md.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def effective_n_bins(n_train: int, n_bins: int) -> int:
    """Cap requested bin count for hard one-hot / classification targets.

    With one observation → one bin index, at most ``n_train`` bins can be
    occupied. Asking for ``B > n`` guarantees empty bins and an underdetermined
    multitask GP; cap at ``n`` (at least 2).
    """
    n = int(n_train)
    b = int(n_bins)
    if n < 1:
        raise ValueError(f"n_train must be >= 1, got {n}")
    if b < 2:
        raise ValueError(f"n_bins must be >= 2, got {b}")
    return int(max(2, min(b, n)))


def make_equal_width_bins(
    y_train: np.ndarray,
    n_bins: int = 100,
    *,
    pad_frac: float = 0.05,
) -> np.ndarray:
    """Equal-width borders on the padded training range.

    Returns
    -------
    borders : ndarray, shape ``[n_bins + 1]``
        Strictly increasing edges. Outer edges extend beyond the training
        min/max by ``pad_frac * (max - min)``.
    """
    y = np.asarray(y_train, dtype=np.float64).ravel()
    y = y[np.isfinite(y)]
    if y.size < 1:
        raise ValueError("y_train must contain at least one finite value")
    b = int(n_bins)
    if b < 2:
        raise ValueError(f"n_bins must be >= 2, got {b}")

    y_min = float(y.min())
    y_max = float(y.max())
    span = max(y_max - y_min, 1e-6)
    pad = float(pad_frac) * span
    return np.linspace(y_min - pad, y_max + pad, b + 1, dtype=np.float64)


def make_quantile_bins(
    y_train: np.ndarray,
    n_bins: int = 32,
    *,
    pad_frac: float = 0.05,
) -> np.ndarray:
    """Equal-frequency borders from training quantiles, with outer padding.

    Duplicate quantile values (ties / ``B > #unique y``) are merged, so the
    returned bin count may be ``< n_bins``. Falls back to equal-width if too
    few unique edges remain.

    Returns
    -------
    borders : ndarray, shape ``[B_eff + 1]``
        Strictly increasing edges. Outer edges extend beyond the training
        min/max by ``pad_frac * (max - min)``.
    """
    y = np.asarray(y_train, dtype=np.float64).ravel()
    y = y[np.isfinite(y)]
    if y.size < 1:
        raise ValueError("y_train must contain at least one finite value")
    b = int(n_bins)
    if b < 2:
        raise ValueError(f"n_bins must be >= 2, got {b}")

    y_min = float(y.min())
    y_max = float(y.max())
    span = max(y_max - y_min, 1e-6)
    pad = float(pad_frac) * span

    qs = np.linspace(0.0, 1.0, b + 1)
    raw = np.quantile(y, qs).astype(np.float64)
    # Merge ties instead of eps-pushing (eps-push → near-zero widths → π/w spikes).
    interior = np.unique(raw[1:-1])
    if interior.size < 1:
        return make_equal_width_bins(y, n_bins=b, pad_frac=pad_frac)
    edges = np.concatenate(
        [[y_min - pad], interior, [y_max + pad]]
    ).astype(np.float64)
    # Final strict increase guard (should already hold after unique + pad).
    for i in range(1, edges.size):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + span * 1e-9
    if edges.size - 1 < 2:
        return make_equal_width_bins(y, n_bins=b, pad_frac=pad_frac)
    return edges


def make_hist_bins(
    y_train: np.ndarray,
    n_bins: int = 200,
    *,
    pad_frac: float = 0.05,
    scheme: str = "equal_width",
    tail_frac: float = 0.0,
) -> tuple[np.ndarray, dict[str, int | str]]:
    """Build hist-MoGP borders with a proper ``B ≤ n`` cap.

    Parameters
    ----------
    scheme :
        ``"equal_width"`` (default; recommended for density ``π/w``) or
        ``"quantile"`` (equal-frequency; may shrink ``B`` after tie merges).
    tail_frac :
        If ``> 0``, append one wide outer bin on each side of width
        ``tail_frac * (max - min)`` so test responses beyond the padded range
        keep finite density (``bar_log_prob`` floors outside-border y at 1e-300).

    Returns
    -------
    borders, info
        ``info`` records ``n_bins_requested``, ``n_bins_effective``, ``scheme``.
    """
    y = np.asarray(y_train, dtype=np.float64).ravel()
    y = y[np.isfinite(y)]
    b_req = int(n_bins)
    b_eff = effective_n_bins(y.size, b_req)
    scheme_l = str(scheme).strip().lower()
    if scheme_l in ("equal_width", "width", "ew"):
        borders = make_equal_width_bins(y, n_bins=b_eff, pad_frac=pad_frac)
        used = "equal_width"
    elif scheme_l in ("quantile", "equal_freq", "eqf"):
        borders = make_quantile_bins(y, n_bins=b_eff, pad_frac=pad_frac)
        used = "quantile"
    else:
        raise ValueError(f"Unknown bin scheme {scheme!r}")
    if float(tail_frac) > 0.0:
        span = max(float(y.max() - y.min()), 1e-6)
        t = float(tail_frac) * span
        borders = np.concatenate([[borders[0] - t], borders, [borders[-1] + t]])
    b_final = int(np.asarray(borders).size - 1)
    return borders, {
        "n_bins_requested": b_req,
        "n_bins_effective": b_final,
        "n_train": int(y.size),
        "bin_scheme": used,
        "tail_frac": float(tail_frac),
    }


def smoothed_marginal(
    y: np.ndarray,
    borders: np.ndarray,
    *,
    pseudocount: float = 0.5,
    kernel: Sequence[float] = (0.25, 0.5, 0.25),
) -> np.ndarray:
    """Strictly positive training-marginal bin masses (for ε-floor mixing).

    Counts + pseudocount, convolved with a short kernel over neighbouring bins,
    renormalised. Every bin (incl. tail bins) gets positive mass.
    """
    oh = y_to_onehot(y, borders)
    c = oh.sum(axis=0).astype(np.float64) + float(pseudocount)
    k = np.asarray(kernel, dtype=np.float64)
    k = k / k.sum()
    c = np.convolve(np.pad(c, len(k) // 2, mode="edge"), k, mode="valid")
    return c / c.sum()


def floor_mix(probs: np.ndarray, marginal: np.ndarray, eps: float) -> np.ndarray:
    """``(1-ε)·π + ε·marginal`` row-wise; keeps every bin strictly positive."""
    p = np.asarray(probs, dtype=np.float64)
    m = np.asarray(marginal, dtype=np.float64).reshape(1, -1)
    out = (1.0 - float(eps)) * p + float(eps) * m
    return out / out.sum(axis=-1, keepdims=True)


def bin_centers(borders: np.ndarray) -> np.ndarray:
    b = np.asarray(borders, dtype=np.float64).ravel()
    return 0.5 * (b[:-1] + b[1:])


def bin_widths(borders: np.ndarray) -> np.ndarray:
    b = np.asarray(borders, dtype=np.float64).ravel()
    return np.clip(np.diff(b), 1e-12, None)


def y_to_bin_index(y: np.ndarray, borders: np.ndarray) -> np.ndarray:
    """Map each y to bin index in ``[0, B-1]`` (edges inclusive via clamp)."""
    y = np.asarray(y, dtype=np.float64).ravel()
    b = np.asarray(borders, dtype=np.float64).ravel()
    # searchsorted right: index i means b[i-1] < y <= b[i] for interior;
    # clamp into [0, B-1].
    idx = np.searchsorted(b, y, side="right") - 1
    return np.clip(idx, 0, b.size - 2).astype(np.int64)


def y_to_onehot(y: np.ndarray, borders: np.ndarray) -> np.ndarray:
    """One-hot encoding of bin membership, shape ``[N, B]``."""
    idx = y_to_bin_index(y, borders)
    b = int(np.asarray(borders).size - 1)
    out = np.zeros((idx.size, b), dtype=np.float64)
    out[np.arange(idx.size), idx] = 1.0
    return out


def empirical_bin_frequencies(
    y: np.ndarray,
    borders: np.ndarray,
    *,
    alpha: float = 1e-2,
) -> np.ndarray:
    """Training histogram frequencies (simplex), shape ``[B]``.

    Dirichlet smoothing ``(c + α) / (n + α B)`` keeps empty bins off exact zero
    so the multitask mean is not an all-mass-on-occupied-bins spike.
    """
    oh = y_to_onehot(y, borders)
    counts = oh.sum(axis=0).astype(np.float64)
    b = int(counts.size)
    a = max(float(alpha), 0.0)
    denom = float(counts.sum()) + a * float(b)
    if denom <= 0:
        return np.full(b, 1.0 / float(b), dtype=np.float64)
    return (counts + a) / denom


def project_onto_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the probability simplex (Wang & Carreira-Perpiñán).

    Parameters
    ----------
    v :
        Shape ``[..., B]``. Projection is applied independently on the last axis.
    Non-finite rows fall back to the uniform distribution on the simplex.
    """
    v = np.asarray(v, dtype=np.float64)
    if v.ndim == 1:
        return _project_simplex_1d(v)
    flat = v.reshape(-1, v.shape[-1])
    out = np.empty_like(flat)
    for i in range(flat.shape[0]):
        out[i] = _project_simplex_1d(flat[i])
    return out.reshape(v.shape)


def _project_simplex_1d(v: np.ndarray) -> np.ndarray:
    """O(B log B) exact projection: p_i = max(v_i - tau, 0), sum p = 1."""
    v = np.asarray(v, dtype=np.float64).ravel()
    n = v.size
    if n == 0:
        return v.copy()
    if not np.all(np.isfinite(v)):
        return np.full(n, 1.0 / float(n), dtype=np.float64)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_candidates = u * np.arange(1, n + 1) > (cssv - 1.0)
    if not np.any(rho_candidates):
        # Degenerate: return uniform
        return np.full(n, 1.0 / n, dtype=np.float64)
    rho = int(np.where(rho_candidates)[0][-1])
    tau = (cssv[rho] - 1.0) / float(rho + 1)
    w = np.maximum(v - tau, 0.0)
    s = float(w.sum())
    if s <= 0:
        return np.full(n, 1.0 / n, dtype=np.float64)
    return w / s


def project_samples_average(
    mean: np.ndarray,
    cov: np.ndarray | None,
    *,
    n_samples: int = 64,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Idea 1B: sample → project → average (optional; off by default in adapters).

    Parameters
    ----------
    mean :
        ``[B]`` or ``[N, B]`` posterior mean.
    cov :
        If ``mean`` is ``[B]``: ``[B, B]`` covariance (or ``None`` → diagonal from
        ignored). If ``mean`` is ``[N, B]``: only diagonal per-task variances
        ``[N, B]`` are supported (independent across bins).
    """
    rng = rng or np.random.default_rng(0)
    mu = np.asarray(mean, dtype=np.float64)
    if mu.ndim == 1:
        b = mu.size
        if cov is None:
            samples = rng.normal(mu, 0.05, size=(n_samples, b))
        else:
            samples = rng.multivariate_normal(mu, cov, size=n_samples)
        return project_onto_simplex(samples).mean(axis=0)

    # [N, B] with diagonal stds in cov
    n, b = mu.shape
    if cov is None:
        std = np.full_like(mu, 0.05)
    else:
        std = np.sqrt(np.clip(np.asarray(cov, dtype=np.float64), 1e-12, None))
    acc = np.zeros_like(mu)
    for _ in range(n_samples):
        samples = rng.normal(mu, std)
        acc += project_onto_simplex(samples)
    return acc / float(n_samples)


def probs_to_bar_logits(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """``log(p + eps)`` so ``predictive_density.density_from_bar_distribution`` works."""
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    p = p / np.clip(p.sum(axis=-1, keepdims=True), 1e-300, None)
    return np.log(p + float(eps))


def predictive_moments_from_bars(
    borders: np.ndarray,
    probs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std of a piecewise-uniform bar distribution.

    Interior bars: uniform on ``[b_k, b_{k+1}]`` → var = width^2 / 12.
    """
    borders = np.asarray(borders, dtype=np.float64).ravel()
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    widths = bin_widths(borders)
    centers = bin_centers(borders)
    p = np.clip(p, 0.0, None)
    p = p / np.clip(p.sum(axis=-1, keepdims=True), 1e-300, None)
    mean = (p * centers).sum(axis=-1)
    second = (p * (centers**2 + (widths**2) / 12.0)).sum(axis=-1)
    var = np.clip(second - mean**2, 1e-12, None)
    return mean, np.sqrt(var)


def bar_log_prob(
    y: np.ndarray,
    borders: np.ndarray,
    logits: np.ndarray,
) -> np.ndarray:
    """Width-scaled log-density at each ``y`` (padded edge bins; no half-normals).

    ``logits`` may be ``[B]`` (broadcast) or ``[N, B]``.
    Density in bin k: ``softmax(logits)_k / width_k``.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    borders = np.asarray(borders, dtype=np.float64).ravel()
    logits = np.asarray(logits, dtype=np.float64)
    n = y.size
    widths = bin_widths(borders)
    idx = y_to_bin_index(y, borders)

    if logits.ndim == 1:
        z = logits - np.max(logits)
        p = np.exp(z)
        p = p / max(float(p.sum()), 1e-300)
        dens = p[idx] / widths[idx]
    else:
        if logits.shape[0] != n:
            raise ValueError(
                f"logits rows {logits.shape[0]} != n_y {n}"
            )
        z = logits - logits.max(axis=-1, keepdims=True)
        p = np.exp(z)
        p = p / np.clip(p.sum(axis=-1, keepdims=True), 1e-300, None)
        dens = p[np.arange(n), idx] / widths[idx]

    # Outside the padded borders → tiny density (should be rare with pad).
    outside = (y < borders[0]) | (y > borders[-1])
    dens = np.asarray(dens, dtype=np.float64)
    dens[outside] = 1e-300
    return np.log(np.clip(dens, 1e-300, None))


def bar_crps(
    y: np.ndarray,
    borders: np.ndarray,
    logits: np.ndarray,
) -> np.ndarray:
    """Exact CRPS of the piecewise-uniform bar distribution at each ``y``.

    ``F`` is linear inside each bar, so ``∫(F − 1{z ≥ y})²`` is closed form
    per bar; mass outside the borders is zero.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    b = np.asarray(borders, dtype=np.float64).ravel()
    lg = np.asarray(logits, dtype=np.float64)
    if lg.ndim == 1:
        lg = np.broadcast_to(lg, (y.size, lg.size))
    p = np.exp(lg - lg.max(axis=-1, keepdims=True))
    p = p / p.sum(axis=-1, keepdims=True)
    cdf = np.concatenate([np.zeros((y.size, 1)), np.cumsum(p, axis=-1)], axis=-1)
    cdf[:, -1] = 1.0
    lo, hi = b[:-1][None, :], b[1:][None, :]
    fa, fb = cdf[:, :-1], cdf[:, 1:]
    w = hi - lo
    yy = y[:, None]

    def _seg(a_val, b_val, width):
        return width * (a_val**2 + a_val * b_val + b_val**2) / 3.0

    below = yy <= lo
    above = yy >= hi
    out = np.where(below, _seg(fa - 1.0, fb - 1.0, w), 0.0)
    out = out + np.where(above, _seg(fa, fb, w), 0.0)
    inside = ~(below | above)
    fy = fa + (fb - fa) * np.clip((yy - lo) / np.clip(w, 1e-300, None), 0.0, 1.0)
    split = _seg(fa, fy, yy - lo) + _seg(fy - 1.0, fb - 1.0, hi - yy)
    out = out + np.where(inside, split, 0.0)
    total = out.sum(axis=-1)
    total = total + np.clip(b[0] - y, 0.0, None) + np.clip(y - b[-1], 0.0, None)
    return total


def mixture_crps(
    y: np.ndarray, means: np.ndarray, stds: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Exact CRPS of a Gaussian mixture (Grimit et al. 2006), ``[N]``.

    ``CRPS = Σ_k w_k A(y − μ_k, σ_k²) − ½ Σ_kl w_k w_l A(μ_k − μ_l, σ_k² + σ_l²)``
    with ``A(m, v) = 2√v φ(m/√v) + m (2Φ(m/√v) − 1)``.
    """
    from scipy.stats import norm

    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    mu = np.asarray(means, dtype=np.float64).reshape(y.shape[0], -1)
    sd = np.clip(np.asarray(stds, dtype=np.float64).reshape(mu.shape), 1e-12, None)
    w = np.clip(np.asarray(weights, dtype=np.float64).reshape(mu.shape), 0.0, None)
    w = w / np.clip(w.sum(-1, keepdims=True), 1e-300, None)

    def _a(m: np.ndarray, v: np.ndarray) -> np.ndarray:
        s = np.sqrt(v)
        z = m / s
        return 2.0 * s * norm.pdf(z) + m * (2.0 * norm.cdf(z) - 1.0)

    t1 = (w * _a(y - mu, sd**2)).sum(-1)
    dm = mu[:, :, None] - mu[:, None, :]
    dv = sd[:, :, None] ** 2 + sd[:, None, :] ** 2
    t2 = (w[:, :, None] * w[:, None, :] * _a(dm, dv)).sum((-1, -2))
    return t1 - 0.5 * t2


def task_cov_from_bin_centers(
    borders: np.ndarray,
    *,
    lengthscale: float | None = None,
    outputscale: float = 1.0,
    nugget: float = 1e-3,
) -> np.ndarray:
    """RBF task covariance ``K_HH`` on bin centers (for ICM / diagnostics)."""
    c = bin_centers(borders)
    if lengthscale is None:
        span = float(c[-1] - c[0]) if c.size > 1 else 1.0
        lengthscale = max(span / max(c.size - 1, 1), 1e-3)
    d = c[:, None] - c[None, :]
    k = float(outputscale) * np.exp(-(d**2) / (2.0 * float(lengthscale) ** 2))
    k = k + float(nugget) * np.eye(c.size)
    return k
