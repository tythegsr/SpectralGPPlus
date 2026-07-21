"""S2 SORF wrapper around the shared S2 STGP runner."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_RFF_DIR = _ROOT / "experiments_RFF"
_SORF_DIR = Path(__file__).resolve().parent
for p in (_ROOT, _RFF_DIR, _SORF_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from toa_s2_stgp_base import run_s2_toa_stgp


def run_s2_toa_sorf(
    num_sorf: int | None = None,
    num_rff: int | None = None,
    **kwargs,
) -> dict:
    feature_count = num_sorf if num_sorf is not None else num_rff
    return run_s2_toa_stgp(rff_sampling="sorf", num_rff=feature_count, **kwargs)
