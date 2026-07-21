"""S2 ORF wrapper around the shared S2 STGP runner."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_RFF_DIR = _ROOT / "experiments_RFF"
_ORF_DIR = Path(__file__).resolve().parent
for p in (_ROOT, _RFF_DIR, _ORF_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from toa_s2_stgp_base import run_s2_toa_stgp


def run_s2_toa_orf(
    num_orf: int | None = None,
    num_rff: int | None = None,
    **kwargs,
) -> dict:
    feature_count = num_orf if num_orf is not None else num_rff
    return run_s2_toa_stgp(rff_sampling="orf", num_rff=feature_count, **kwargs)
