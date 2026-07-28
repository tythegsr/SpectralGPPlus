"""Unit checks for soft probabilistic bound penalties."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from gpplus.training.bound_penalty import BoundConfig, soft_probabilistic_bound_penalty
from gpplus.models import RFFGPR
from experiments_toa.s2_bound_penalty import (
    learned_bound_penalty_lambda,
    resolve_bound_penalty_for_tasks,
)
from experiments_toa.s2_y_transform import YWarpConfig


def test_floor_and_ceiling_penalties_independent():
    sig = torch.tensor([0.1, 0.1, 0.1])
    p_lo = soft_probabilistic_bound_penalty(
        torch.tensor([-1.0, 0.0, 3.0]), sig, a=0.2, b=None, k=2.0, alpha=10.0
    )
    p_ok = soft_probabilistic_bound_penalty(
        torch.tensor([1.0, 2.0, 3.0]), sig, a=0.2, b=None, k=2.0, alpha=10.0
    )
    assert float(p_lo) > 1.0
    assert float(p_ok) < 0.05

    p_hi = soft_probabilistic_bound_penalty(
        torch.tensor([6.0, 7.0]), torch.tensor([0.1, 0.1]), a=None, b=5.2, k=2.0, alpha=10.0
    )
    p_hi_ok = soft_probabilistic_bound_penalty(
        torch.tensor([1.0, 2.0]), torch.tensor([0.1, 0.1]), a=None, b=5.2, k=2.0, alpha=10.0
    )
    assert float(p_hi) > 1.0
    assert float(p_hi_ok) < 0.05


def test_box_penalty_sums_both_sides():
    mu = torch.tensor([0.0])
    sig = torch.tensor([0.05])
    p_both = soft_probabilistic_bound_penalty(mu, sig, a=0.2, b=5.2, k=2.0, alpha=10.0)
    p_floor = soft_probabilistic_bound_penalty(mu, sig, a=0.2, b=None, k=2.0, alpha=10.0)
    assert torch.isclose(p_both, p_floor, atol=1e-4)


def test_resolve_qoi_aligned_bounds_and_conflicts():
    qoi = ["algae", "aot", "cos_i", "cwv", "dust", "grain_size", "liquid_water"]
    warps = YWarpConfig(
        log_tasks=frozenset(["algae", "dust", "grain_size", "liquid_water"]),
        logit_tasks=frozenset(["cos_i", "aot"]),
        log_source="test",
        logit_source="test",
    )
    cfg = resolve_bound_penalty_for_tasks(
        qoi,
        [None, None, None, 0.2, None, None, None],
        [None, None, None, 5.2, None, None, None],
        warps=warps,
    )
    assert cfg["cwv"] is not None
    assert cfg["cwv"].a == 0.2 and cfg["cwv"].b == 5.2
    assert all(cfg[n] is None for n in qoi if n != "cwv")

    try:
        resolve_bound_penalty_for_tasks(qoi, [0.2], [5.2], warps=warps)
        raise AssertionError("expected length mismatch")
    except ValueError as exc:
        assert "BOUND_MIN" in str(exc)

    try:
        resolve_bound_penalty_for_tasks(
            qoi,
            [0.0, None, None, None, None, None, None],
            [None] * 7,
            warps=warps,
        )
        raise AssertionError("expected warp conflict")
    except ValueError as exc:
        assert "conflicts" in str(exc)

    try:
        BoundConfig(a=None, b=None)
        raise AssertionError("expected BoundConfig error")
    except ValueError:
        pass


def test_learnable_lambda_init_and_floor():
    lam_min = 1.0
    lam_init = 10.0
    x = torch.randn(24, 4, dtype=torch.float64)
    y = torch.randn(24, dtype=torch.float64)
    model = RFFGPR(x, y, num_rff=8)
    model.register_learnable_bound_penalty_lambda(lam_init=lam_init, lam_min=lam_min)
    assert abs(float(model.bound_penalty_lambda) - lam_init) < 1e-5
    assert float(model.bound_penalty_lambda) >= lam_min
    assert learned_bound_penalty_lambda(model) == float(model.bound_penalty_lambda)

    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    mu = torch.tensor([-0.5, 0.0, 2.0], dtype=torch.float64)
    sig = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
    penalty = soft_probabilistic_bound_penalty(mu, sig, a=0.2, b=5.2, k=2.0, alpha=10.0)
    for _ in range(8):
        opt.zero_grad()
        loss = model.bound_penalty_lambda * penalty
        loss.backward()
        opt.step()
        assert float(model.bound_penalty_lambda) >= lam_min


def test_fixed_lambda_when_not_registered():
    x = torch.randn(12, 3, dtype=torch.float64)
    y = torch.randn(12, dtype=torch.float64)
    model = RFFGPR(x, y, num_rff=8)
    assert learned_bound_penalty_lambda(model) is None
    assert not model.has_learnable_bound_penalty_lambda()


if __name__ == "__main__":
    test_floor_and_ceiling_penalties_independent()
    test_box_penalty_sums_both_sides()
    test_resolve_qoi_aligned_bounds_and_conflicts()
    test_learnable_lambda_init_and_floor()
    test_fixed_lambda_when_not_registered()
    print("test_bound_penalty: OK")
