"""Tests for TabPFN bar-distribution posterior histogram plotting."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_MTGPR_DIR = _ROOT / "experiments_RFFMTGPR"
if str(_MTGPR_DIR) not in sys.path:
    sys.path.insert(0, str(_MTGPR_DIR))

from plot_toa_posterior import _softmax_logits, _tabpfn_bar_histogram


def _density_at(edges: np.ndarray, heights: np.ndarray, x: float) -> float:
    left, right = edges[:-1], edges[1:]
    idx = np.where((left <= x) & (x < right))[0]
    if idx.size == 0:
        if x == right[-1]:
            return float(heights[-1])
        return 0.0
    return float(heights[idx[0]])


def test_tabpfn_bar_histogram_peak_aligns_with_dominant_bucket() -> None:
    """Histogram peak should lie in the highest-probability bucket."""
    n_buckets = 50
    borders = np.linspace(0.0, 1.0, n_buckets + 1)
    logits = np.full(n_buckets, -10.0)
    peak_idx = 25
    logits[peak_idx] = 5.0

    x0, x1 = 0.35, 0.55
    edges, heights = _tabpfn_bar_histogram(
        logits,
        borders,
        x0=x0,
        x1=x1,
        x_min=0.0,
        x_max=1.0,
        train_scale_log=False,
    )

    assert heights.size > 0
    peak_bucket = int(np.argmax(heights))
    peak_center = 0.5 * (edges[peak_bucket] + edges[peak_bucket + 1])
    expected_center = 0.5 * (borders[peak_idx] + borders[peak_idx + 1])
    assert abs(peak_center - expected_center) < (borders[1] - borders[0])

    mean_x = float(np.sum(_softmax_logits(logits) * (borders[:-1] + borders[1:]) / 2.0))
    assert _density_at(edges, heights, mean_x) >= 0.5 * float(np.max(heights))


def test_tabpfn_bar_histogram_log_scale_maps_to_original_units() -> None:
    """Log-scale borders map to original-scale edges via exp."""
    log_borders = np.linspace(np.log(10.0), np.log(100.0), 21)
    logits = np.zeros(20)
    logits[10] = 8.0

    edges, heights = _tabpfn_bar_histogram(
        logits,
        log_borders,
        x0=30.0,
        x1=70.0,
        x_min=0.0,
        x_max=None,
        train_scale_log=True,
    )

    assert np.all(edges >= 10.0)
    assert np.all(edges <= 100.0)
    assert heights.size > 0
    peak_idx = int(np.argmax(heights))
    peak_center = 0.5 * (edges[peak_idx] + edges[peak_idx + 1])
    expected = np.exp(0.5 * (log_borders[10] + log_borders[11]))
    assert abs(peak_center - expected) / expected < 0.05


def test_tabpfn_bar_histogram_renormalizes_visible_mass() -> None:
    """Visible bucket probabilities integrate to 1 over the plot window."""
    borders = np.linspace(0.0, 1.0, 11)
    logits = np.array([0.0, 1.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0])

    edges, heights = _tabpfn_bar_histogram(
        logits,
        borders,
        x0=0.2,
        x1=0.5,
        x_min=0.0,
        x_max=1.0,
        train_scale_log=False,
    )

    widths = edges[1:] - edges[:-1]
    integral = float(np.sum(heights * widths))
    assert integral == pytest.approx(1.0, rel=1e-6)
