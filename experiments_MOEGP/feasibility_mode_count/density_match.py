"""Eval-only true-density match metrics (Hellinger / TV) at probe locations.

Uses ``TRUE_PDF`` when available. Training never sees these scores — they are
for foundation-compare reporting and acceptance checks only.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from plot_foundation_marginals import TRUE_PDF, _mixture_pdf
from predictive_density import model_predictive_density


def normalize_density(dens: np.ndarray, dy: float) -> np.ndarray:
    """Normalize a grid density so ∫ p ≈ 1 under uniform spacing ``dy``."""
    p = np.asarray(dens, dtype=np.float64).ravel()
    p = np.clip(p, 0.0, None)
    mass = float(np.sum(p) * dy)
    if mass <= 1e-12:
        return np.zeros_like(p)
    return p / mass


def hellinger_distance(p: np.ndarray, q: np.ndarray, dy: float) -> float:
    """Hellinger distance H(p,q) = sqrt(1 − ∑ sqrt(p q) dy) on a grid."""
    pn = normalize_density(p, dy)
    qn = normalize_density(q, dy)
    bc = float(np.sum(np.sqrt(pn * qn)) * dy)
    return float(np.sqrt(max(0.0, 1.0 - min(bc, 1.0))))


def total_variation(p: np.ndarray, q: np.ndarray, dy: float) -> float:
    """Total variation distance TV(p,q) = ½ ∫ |p−q| on a grid."""
    pn = normalize_density(p, dy)
    qn = normalize_density(q, dy)
    return float(0.5 * np.sum(np.abs(pn - qn)) * dy)


def predictive_density_at_probe(
    y_grid: np.ndarray,
    pred: dict[str, Any],
    probe_index: int,
) -> np.ndarray:
    """Model predictive density on ``y_grid`` for one probe row."""
    dens, _src = model_predictive_density(y_grid, pred, probe_index=probe_index)
    if dens is None:
        dens = _mixture_pdf(
            y_grid,
            pred["expert_means"][probe_index],
            pred["expert_stds"][probe_index],
            pred["gate_weights"][probe_index],
        )
    return np.asarray(dens, dtype=np.float64).ravel()


def probe_density_match(
    *,
    problem_name: str,
    x_probes: np.ndarray,
    pred: dict[str, Any],
    y_min: float,
    y_max: float,
    n_grid: int = 400,
    true_pdf: Callable[[np.ndarray, float], np.ndarray] | None = None,
    true_k_modes: int | None = None,
    k_modes_mean: float | None = None,
) -> dict[str, float] | None:
    """Mean Hellinger / TV vs true p(y|x) at probes; optional |k_modes−true_k|.

    Returns ``None`` when no true PDF is registered for ``problem_name``.
    """
    fn = true_pdf if true_pdf is not None else TRUE_PDF.get(problem_name)
    if fn is None:
        return None

    x_probes = np.asarray(x_probes, dtype=np.float64).ravel()
    if x_probes.size == 0:
        return None

    y_grid = np.linspace(float(y_min), float(y_max), int(n_grid))
    dy = float(y_grid[1] - y_grid[0]) if y_grid.size > 1 else 1.0

    hells: list[float] = []
    tvs: list[float] = []
    for i, xv in enumerate(x_probes):
        true_dens = np.asarray(fn(y_grid, float(xv)), dtype=np.float64).ravel()
        model_dens = predictive_density_at_probe(y_grid, pred, i)
        hells.append(hellinger_distance(true_dens, model_dens, dy))
        tvs.append(total_variation(true_dens, model_dens, dy))

    out: dict[str, float] = {
        "hellinger_mean": float(np.mean(hells)),
        "tv_mean": float(np.mean(tvs)),
        "hellinger_max": float(np.max(hells)),
        "tv_max": float(np.max(tvs)),
        "n_probes": float(len(hells)),
    }
    if true_k_modes is not None and k_modes_mean is not None:
        out["abs_k_modes_err"] = float(abs(float(k_modes_mean) - float(true_k_modes)))
    return out
