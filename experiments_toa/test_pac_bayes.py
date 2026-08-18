"""Unit checks for Catoni PAC-Bayes MLL wrapper."""

from __future__ import annotations

import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import gpytorch
import torch

from gpplus.models import GPR, RFFGPR
from gpplus.training.pac_bayes import (
    collect_learnable_params,
    diagonal_gaussian_kl,
    pac_bayes_complexity,
    pac_bayes_mll_class,
)
from gpplus.training.rff_mll import RFFWoodburyMarginalLogLikelihood


def test_diagonal_gaussian_kl_1d_and_isotropic():
    # Q = N(0, sigma_q^2), P = N(0, sigma_p^2) => KL = 0.5*(r - 1 - log r), r=sq2/sp2
    sp, sq = 2.0, 0.5
    r = (sq / sp) ** 2
    expected = 0.5 * (r - 1.0 - math.log(r))
    got = float(diagonal_gaussian_kl(torch.tensor([0.0]), prior_std=sp, posterior_std=sq))
    assert abs(got - expected) < 1e-6

    # Isotropic d=3 with zero mean: d times the 1-D KL
    mu = torch.zeros(3)
    got3 = float(diagonal_gaussian_kl(mu, prior_std=sp, posterior_std=sq))
    assert abs(got3 - 3.0 * expected) < 1e-6

    # Nonzero mean adds 0.5 * ||mu||^2 / sp^2
    mu2 = torch.tensor([1.0, -1.0])
    base = float(diagonal_gaussian_kl(torch.zeros(2), prior_std=sp, posterior_std=sq))
    with_mean = float(diagonal_gaussian_kl(mu2, prior_std=sp, posterior_std=sq))
    assert abs(with_mean - (base + 0.5 * float((mu2 * mu2).sum()) / (sp**2))) < 1e-6


def test_pac_bayes_wraps_woodbury_mll():
    torch.manual_seed(0)
    x = torch.randn(32, 3, dtype=torch.float64)
    y = torch.randn(32, dtype=torch.float64)
    model = RFFGPR(x, y, num_rff=16)
    model.train()

    wrapped = pac_bayes_mll_class(
        RFFWoodburyMarginalLogLikelihood,
        temperature=1.0,
        prior_std=1.0,
        posterior_std=0.1,
    )
    assert issubclass(wrapped, RFFWoodburyMarginalLogLikelihood)

    mll = wrapped(model.likelihood, model, jitter=1e-6)
    val = mll(None, y)
    assert torch.isfinite(val)
    assert val.ndim == 0

    base = RFFWoodburyMarginalLogLikelihood(model.likelihood, model, jitter=1e-6)
    base_val = float(base(None, y))

    # Huge temperature => KL/(λ n) → 0, so PAC MLL ≈ base MLL.
    soft = pac_bayes_mll_class(
        RFFWoodburyMarginalLogLikelihood,
        temperature=1e8,
        prior_std=1.0,
        posterior_std=0.1,
    )
    soft_mll = soft(model.likelihood, model, jitter=1e-6)
    soft_val = float(soft_mll(None, y))
    assert abs(soft_val - base_val) < 1e-4

    # Finite temperature: PAC value is strictly below base MLL (KL > 0).
    params = collect_learnable_params(model)
    assert params
    kl = float(pac_bayes_complexity(model, prior_std=1.0, posterior_std=0.1))
    assert kl > 0.0
    assert float(val) < base_val


def test_pac_bayes_wraps_exact_mll():
    torch.manual_seed(1)
    x = torch.randn(20, 2, dtype=torch.float64)
    y = torch.randn(20, dtype=torch.float64)
    model = GPR(x, y)
    model.train()

    wrapped = pac_bayes_mll_class(
        gpytorch.mlls.ExactMarginalLogLikelihood,
        temperature=1.0,
        prior_std=1.0,
        posterior_std=0.1,
    )
    mll = wrapped(model.likelihood, model)
    output = model(x)
    val = mll(output, y)
    assert torch.isfinite(val)


if __name__ == "__main__":
    test_diagonal_gaussian_kl_1d_and_isotropic()
    test_pac_bayes_wraps_woodbury_mll()
    test_pac_bayes_wraps_exact_mll()
    print("test_pac_bayes: OK")
