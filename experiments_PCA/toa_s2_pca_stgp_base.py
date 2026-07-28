"""S2 PCA + RFF/ORF/SORF independent GPs with per-QoI band subsets."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

_ROOT = Path(__file__).resolve().parents[1]
_PCA_DIR = Path(__file__).resolve().parent
_RFF_DIR = _ROOT / "experiments_RFF"
_MTGPR_DIR = _ROOT / "experiments_RFFMTGPR"
_DEFAULT_SAVE_DIRS = {
    "rff": "experiments_PCA/results/s2_toa_pca_rff",
    "orf": "experiments_PCA/results/s2_toa_pca_orf",
    "sorf": "experiments_PCA/results/s2_toa_pca_sorf",
}

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments_toa.paths import pin_toa_import_paths

pin_toa_import_paths(_MTGPR_DIR, _RFF_DIR, _PCA_DIR)

from experiments_toa.s2_bands import NComponentsSpec
from toa_s2_stgp_base import RFF_SAMPLING_CHOICES, run_s2_toa_stgp


def run_s2_toa_pca_stgp(
    *,
    n_components: NComponentsSpec = 100,
    rff_sampling: Literal["rff", "orf", "sorf"] = "sorf",
    pca_svd_solver: str = "randomized",
    save_path: str | None = None,
    **kwargs,
) -> dict:
    if rff_sampling not in RFF_SAMPLING_CHOICES:
        raise ValueError(f"rff_sampling must be one of {RFF_SAMPLING_CHOICES}, got {rff_sampling!r}")
    if save_path is None:
        save_path = _DEFAULT_SAVE_DIRS[rff_sampling]
    return run_s2_toa_stgp(
        rff_sampling=rff_sampling,
        save_path=save_path,
        n_pca_components=n_components,
        pca_svd_solver=pca_svd_solver,
        **kwargs,
    )


def run_s2_toa_pca_rff(**kwargs) -> dict:
    return run_s2_toa_pca_stgp(rff_sampling="rff", **kwargs)
