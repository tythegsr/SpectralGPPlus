"""
Shared TOA benchmark runner using RFF-GP (Woodbury inference).

Trains two independent RFFGPR models on y_cos and y_grain with shared scaled inputs.
"""

from __future__ import annotations

from toa_stgp_base import (
    NUM_TASKS,
    TASK_NAMES,
    TOA_INPUT_DIM,
    compute_per_task_metrics,
    run_toa_rff,
    run_toa_stgp,
)

__all__ = [
    "NUM_TASKS",
    "TASK_NAMES",
    "TOA_INPUT_DIM",
    "compute_per_task_metrics",
    "run_toa_rff",
    "run_toa_stgp",
]
