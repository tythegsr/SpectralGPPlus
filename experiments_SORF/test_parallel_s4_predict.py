"""Parity tests for parallel multi-QoI S4 predict phase."""

from __future__ import annotations

import torch

from experiments_SORF.emit_s4_sorf_base import (
    resolve_predict_n_jobs,
    run_s4_predict_phase,
)
from experiments_toa.s2_y_transform import resolve_y_warps
from gpplus.models.rff_gpr import RFFGPR


def _tiny_pack(task_name: str, seed: int, n_train=40, n_test=24, d=3, num_rff=8):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_train, d, generator=g, dtype=torch.float64)
    y = torch.randn(n_train, generator=g, dtype=torch.float64)
    x_te = torch.randn(n_test, d, generator=g, dtype=torch.float64)
    y_te = torch.randn(n_test, generator=g, dtype=torch.float64)
    torch.manual_seed(seed)
    model = RFFGPR(
        x, y, num_rff=num_rff, ard=True, rff_sampling="rff", nigp=True
    )
    model.eval()
    return {
        "task_name": task_name,
        "model": model.cpu(),
        "x_te": x_te,
        "y_te": y_te,
        "y_scaler": None,
        "train_time": 0.1,
        "best_loss": 1.0,
        "learned_noise": {},
        "best_run": {},
        "runs": [],
        "task_epochs": 1,
    }


def _make_packs(n: int = 4):
    return [_tiny_pack(f"t{i}", seed=100 + i) for i in range(n)]


def test_resolve_predict_n_jobs_defaults():
    assert resolve_predict_n_jobs("cpu", 7, 4) == 4
    assert resolve_predict_n_jobs("cpu", 7, None) >= 1
    # Explicit override always wins.
    assert resolve_predict_n_jobs("cuda", 7, 1) == 1
    assert resolve_predict_n_jobs("cuda", 7, 3) == 3
    # cuda device string defaults to 1 when CUDA unavailable or single GPU.
    assert resolve_predict_n_jobs("cuda", 7, None) == 1


def test_parallel_predict_matches_sequential():
    packs = _make_packs(4)
    packs2 = _make_packs(4)
    warps = resolve_y_warps([], [])

    seq, wall1, sum1 = run_s4_predict_phase(
        packs,
        device="cpu",
        dtype=torch.float64,
        predict_chunk_size=0,
        warps=warps,
        standardize_y=False,
        rel_tolerance=0.01,
        predict_n_jobs=1,
    )
    par, wall2, sum2 = run_s4_predict_phase(
        packs2,
        device="cpu",
        dtype=torch.float64,
        predict_chunk_size=0,
        warps=warps,
        standardize_y=False,
        rel_tolerance=0.01,
        predict_n_jobs=4,
    )
    assert len(seq) == len(par) == 4
    for a, b in zip(seq, par):
        assert a["task_name"] == b["task_name"]
        assert torch.allclose(a["pred_mean"], b["pred_mean"], rtol=1e-10, atol=1e-10)
        assert torch.allclose(a["pred_std"], b["pred_std"], rtol=1e-10, atol=1e-10)
        assert torch.allclose(a["lower"], b["lower"], rtol=1e-10, atol=1e-10)
        assert torch.allclose(a["upper"], b["upper"], rtol=1e-10, atol=1e-10)
    assert sum1 > 0 and sum2 > 0
    # Wall should not exceed sum by much; parallel wall should be <= sum (soft).
    assert wall2 <= sum2 * 1.05 + 0.5
    assert wall1 > 0
