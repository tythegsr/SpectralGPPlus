"""
Shared TOA benchmark runner using SORF-GP (Woodbury inference).

Trains two independent RFFGPR models on y_cos and y_grain with shared scaled inputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_RFF_DIR = _ROOT / "experiments_RFF"
_SORF_DIR = Path(__file__).resolve().parent
for p in (_ROOT, _RFF_DIR, _SORF_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from toa_stgp_base import (
    NUM_TASKS,
    TASK_NAMES,
    TOA_INPUT_DIM,
    compute_per_task_metrics,
    run_toa_stgp,
)


def run_toa_sorf(
    num_sorf: int | None = None,
    num_rff: int | None = None,
    **kwargs,
) -> dict:
    """Train independent SORF-GP models on TOA (rff_sampling='sorf')."""
    feature_count = num_sorf if num_sorf is not None else num_rff
    return run_toa_stgp(rff_sampling="sorf", num_rff=feature_count, **kwargs)


__all__ = [
    "NUM_TASKS",
    "TASK_NAMES",
    "TOA_INPUT_DIM",
    "compute_per_task_metrics",
    "run_toa_sorf",
]
