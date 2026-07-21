"""S2 RFF wrapper around the shared S2 STGP runner."""

from __future__ import annotations

from toa_s2_stgp_base import run_s2_toa_stgp


def run_s2_toa_rff(**kwargs) -> dict:
    return run_s2_toa_stgp(rff_sampling="rff", **kwargs)
