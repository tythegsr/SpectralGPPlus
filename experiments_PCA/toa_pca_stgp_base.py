"""PCA preprocessing + RFF/ORF/SORF STGP (Woodbury) on TOA."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import torch

_ROOT = Path(__file__).resolve().parents[1]
_PCA_DIR = Path(__file__).resolve().parent
_RFF_DIR = _ROOT / "experiments_RFF"
_MTGPR_DIR = _ROOT / "experiments_RFFMTGPR"
_DEFAULT_SAVE_DIRS = {
    "rff": "experiments_PCA/results/toa_pca_rff",
    "orf": "experiments_PCA/results/toa_pca_orf",
    "sorf": "experiments_PCA/results/toa_pca_sorf",
}

DEFAULT_DROP_COLUMNS = sorted({132, *range(195, 209)})


def _pin_experiment_paths() -> None:
    ordered = (str(_MTGPR_DIR), str(_RFF_DIR), str(_PCA_DIR), str(_ROOT))
    sys.path[:] = list(ordered) + [p for p in sys.path if p not in ordered]


if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments_toa.paths import pin_toa_import_paths

pin_toa_import_paths(_MTGPR_DIR, _RFF_DIR, _PCA_DIR)

_pin_experiment_paths()

from toa_stgp_base import RFF_SAMPLING_CHOICES, run_toa_stgp


def run_toa_pca_stgp(
    *,
    n_components: int = 30,
    rff_sampling: Literal["rff", "orf", "sorf"] = "rff",
    pca_svd_solver: str = "randomized",
    drop_columns: list[int] | None = None,
    save_path: str | None = None,
    **kwargs,
) -> dict:
    """
    TOA runner: column drop -> PCA on train -> RFFGPR (Woodbury) on PC scores.

    All remaining kwargs are forwarded to ``run_toa_stgp`` (n_train, num_rff, etc.).
    """
    if rff_sampling not in RFF_SAMPLING_CHOICES:
        raise ValueError(f"rff_sampling must be one of {RFF_SAMPLING_CHOICES}, got {rff_sampling!r}")

    if drop_columns is None:
        drop_columns = list(DEFAULT_DROP_COLUMNS)
    if save_path is None:
        save_path = _DEFAULT_SAVE_DIRS[rff_sampling]

    return run_toa_stgp(
        rff_sampling=rff_sampling,
        save_path=save_path,
        drop_columns=drop_columns,
        n_pca_components=n_components,
        pca_svd_solver=pca_svd_solver,
        **kwargs,
    )


def run_toa_pca_rff(**kwargs) -> dict:
    """PCA + RFF-GP on TOA."""
    return run_toa_pca_stgp(rff_sampling="rff", **kwargs)
