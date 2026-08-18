"""Correctness tests for the inducing-point SVGP ELBO and its minibatch path.

Two properties matter. First, the homoskedastic bound must agree with GPyTorch's
own ``VariationalELBO`` -- that pins the closed-form expected log likelihood and
the KL scaling against a reference implementation. Second, the objective must
decompose over data points, so summing a partition of the data reproduces the
full-data value exactly (KL counted once). Everything else here guards the NIGP
slope path, which is the part that could silently make the estimator biased by
letting one point's ``d_i`` depend on the rest of the batch.
"""

from __future__ import annotations

import math

import gpytorch
import pytest
import torch

from gpplus.models.svgp_gpr import SVGPR
from gpplus.training.minibatch_trainer import (
    MinibatchGPTrainer,
    resolve_elbo_class,
    variational_param_groups,
)
from gpplus.training.parameter_initializer import skip_initialization
from gpplus.training.svgp_elbo import SVGPELBO
from gpplus.utils.nigp_utils import effective_noise_variance

TOL = 1e-10
RTOL = 1e-9


def _make_model(n=40, d=3, num_inducing=8, seed=0, nigp=False):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g, dtype=torch.float64)
    y = torch.randn(n, generator=g, dtype=torch.float64)
    model = SVGPR(x, y, num_inducing=num_inducing, nigp=nigp, seed=seed)
    with torch.no_grad():
        model.likelihood.raw_noise.fill_(-2.0)
        model.mean_module.raw_constant.fill_(0.3)
    model.train()
    # First call materializes q(u) from the prior.
    model(x)
    return model, x, y


def _randomize_q(model, seed=1):
    """Move ``q(u)`` off the prior so the mean has nonzero slope."""
    g = torch.Generator().manual_seed(seed)
    dist = model.variational_strategy._variational_distribution
    with torch.no_grad():
        dist.variational_mean.copy_(
            torch.randn(dist.variational_mean.shape, generator=g, dtype=torch.float64)
        )
        dist.chol_variational_covar.mul_(0.5)


# ---------------------------------------------------------------------------
# Agreement with GPyTorch and decomposition over data
# ---------------------------------------------------------------------------


def test_homoskedastic_elbo_matches_gpytorch_variational_elbo():
    model, x, y = _make_model()
    _randomize_q(model)
    ours = SVGPELBO(model.likelihood, model)(x, y, x.shape[0])

    reference = gpytorch.mlls.VariationalELBO(
        model.likelihood, model, num_data=x.shape[0]
    )
    theirs = reference(model(x), y)

    assert torch.allclose(ours, theirs, rtol=RTOL, atol=TOL), (
        float(ours),
        float(theirs),
    )


def test_expected_log_likelihood_matches_closed_form():
    model, x, y = _make_model()
    _randomize_q(model)
    with torch.no_grad():
        dist = model(x)
        d = model.likelihood.noise.reshape(-1)[0]
        manual = -0.5 * (
            math.log(2 * math.pi)
            + d.log()
            + ((y - dist.mean).pow(2) + dist.variance) / d
        )
        ours = model.expected_log_likelihood(x, y)
    assert torch.allclose(ours, manual, rtol=RTOL, atol=TOL)


@pytest.mark.parametrize("nigp", [False, True])
def test_minibatch_partition_sums_to_full_elbo(nigp):
    model, x, y = _make_model(n=36, nigp=nigp)
    _randomize_q(model)
    elbo = SVGPELBO(model.likelihood, model)
    n = x.shape[0]

    with torch.no_grad():
        full = elbo(x, y, n)
        kl = elbo.kl_divergence()
        batch_size = 9
        total = torch.zeros((), dtype=x.dtype)
        for start in range(0, n, batch_size):
            xb = x[start : start + batch_size]
            yb = y[start : start + batch_size]
            # Undo the N/B rescaling and the single KL to recover a raw sum.
            total = total + (elbo(xb, yb, n) * n + kl) * (batch_size / n)
        reconstructed = (total - kl) / n

    assert torch.allclose(full, reconstructed, rtol=RTOL, atol=TOL), (
        float(full),
        float(reconstructed),
    )


def test_full_data_elbo_is_invariant_to_chunking():
    model, x, y = _make_model(n=36)
    _randomize_q(model)
    elbo = SVGPELBO(model.likelihood, model)
    with torch.no_grad():
        assert torch.allclose(
            elbo.full_data_elbo(x, y),
            elbo.full_data_elbo(x, y, chunk_size=7),
            rtol=RTOL,
            atol=TOL,
        )


def test_kl_beta_scales_only_the_kl_term():
    model, x, y = _make_model()
    _randomize_q(model)
    n = x.shape[0]
    with torch.no_grad():
        full = SVGPELBO(model.likelihood, model)(x, y, n)
        halved = SVGPELBO(model.likelihood, model, kl_beta=0.5)(x, y, n)
        kl = model.kl_divergence()
    assert torch.allclose(halved - full, 0.5 * kl / n, rtol=RTOL, atol=TOL)


# ---------------------------------------------------------------------------
# NIGP
# ---------------------------------------------------------------------------


def test_nigp_slopes_match_autograd_of_the_predictive_mean():
    model, x, _ = _make_model(nigp=True)
    _randomize_q(model)
    slopes = model.grad_mu(x)

    def mean_at(xi):
        return model(xi.unsqueeze(0)).mean.squeeze(0)

    for i in (0, 5, 17):
        expected = torch.autograd.functional.jacobian(mean_at, x[i])
        assert torch.allclose(slopes[i], expected, rtol=1e-7, atol=1e-9), i


def test_nigp_slopes_are_batch_local():
    model, x, _ = _make_model(nigp=True)
    _randomize_q(model)
    full = model.grad_mu(x)
    head = model.grad_mu(x[:9])
    subset = model.grad_mu(x[[3, 20, 31]])
    assert torch.allclose(head, full[:9], rtol=1e-9, atol=1e-11)
    assert torch.allclose(subset, full[[3, 20, 31]], rtol=1e-9, atol=1e-11)


def test_nigp_observation_noise_uses_effective_variance():
    model, x, _ = _make_model(nigp=True)
    _randomize_q(model)
    d = model.observation_noise_var(x)
    expected = effective_noise_variance(
        model.likelihood.noise.reshape(-1), model.input_noise_var, model.grad_mu(x)
    )
    assert torch.allclose(d, expected, rtol=1e-9, atol=1e-11)
    assert (d > model.likelihood.noise.reshape(-1)[0]).all()


def test_nigp_is_inert_exactly_at_the_prior_mean():
    """
    ``m_u`` equal to the prior mean makes the predictive mean constant, so the
    slopes -- and with them the whole NIGP correction -- vanish.

    GPyTorch's ``CholeskyVariationalDistribution`` offsets ``m_u`` by a small
    random amount at initialization precisely to avoid this degeneracy, so in
    practice ``sigma_x`` receives gradient from the first step.
    """
    model, x, y = _make_model(nigp=True)
    assert model.grad_mu(x).abs().max() > 0

    with torch.no_grad():
        model.variational_strategy._variational_distribution.variational_mean.zero_()
        assert torch.allclose(model.grad_mu(x), torch.zeros_like(x))
        d = model.observation_noise_var(x)
    assert torch.allclose(d, model.likelihood.noise.reshape(-1)[0].expand_as(d))


def test_freezing_the_correction_recovers_the_standard_elbo():
    model, x, y = _make_model(nigp=True)
    _randomize_q(model)
    n = x.shape[0]
    with torch.no_grad():
        with_nigp = SVGPELBO(model.likelihood, model)(x, y, n)
        model.nigp_correction_enabled = False
        frozen = SVGPELBO(model.likelihood, model)(x, y, n)
        reference = gpytorch.mlls.VariationalELBO(
            model.likelihood, model, num_data=n
        )(model(x), y)
    assert not torch.allclose(with_nigp, frozen)
    assert torch.allclose(frozen, reference, rtol=RTOL, atol=TOL)


# ---------------------------------------------------------------------------
# Gradients and optimization
# ---------------------------------------------------------------------------


def test_gradients_flow_to_all_trainable_parameters():
    model, x, y = _make_model(nigp=True)
    # The sigma_x gradient is identically zero at m_u = 0 (see the inertness
    # test above), so q must be moved before this check is meaningful.
    _randomize_q(model)
    loss = -SVGPELBO(model.likelihood, model)(x, y, x.shape[0])
    loss.backward()

    expected = {
        "raw_input_noise",
        "variational_strategy.inducing_points",
        "variational_strategy._variational_distribution.variational_mean",
        "variational_strategy._variational_distribution.chol_variational_covar",
        "likelihood.noise_covar.raw_noise",
        "mean_module.raw_constant",
        "covar_module.raw_outputscale",
        "covar_module.base_kernel.raw_lengthscale",
    }
    names = {name for name, _ in model.named_parameters()}
    assert expected <= names, expected - names

    for name, param in model.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient for {name}"
        assert param.grad.abs().max() > 0, f"zero gradient for {name}"


def test_variational_param_groups_separate_inducing_state():
    model, _, _ = _make_model()
    groups = variational_param_groups(model, {"lr": 0.01}, 0.2)
    assert len(groups) == 2
    assert groups[0]["lr"] == 0.01
    assert groups[1]["lr"] == 0.2

    variational_numel = sum(p.numel() for p in groups[1]["params"])
    strategy = model.variational_strategy
    expected = (
        strategy.inducing_points.numel()
        + strategy._variational_distribution.variational_mean.numel()
        + strategy._variational_distribution.chol_variational_covar.numel()
    )
    assert variational_numel == expected


def test_variational_params_are_excluded_from_sobol_init():
    model, _, _ = _make_model()
    skipped = {n for n, _ in model.named_parameters() if skip_initialization(n)}
    assert skipped == {
        "variational_strategy.inducing_points",
        "variational_strategy._variational_distribution.variational_mean",
        "variational_strategy._variational_distribution.chol_variational_covar",
    }


def test_resolve_elbo_class_picks_svgp():
    model, _, _ = _make_model()
    assert resolve_elbo_class(model) is SVGPELBO


def test_adam_steps_decrease_the_loss():
    g = torch.Generator().manual_seed(3)
    x = torch.rand(200, 1, generator=g, dtype=torch.float64) * 4 - 2
    y = (x[:, 0] * 2.0).sin() + 0.05 * torch.randn(200, generator=g, dtype=torch.float64)
    model = SVGPR(x, y, num_inducing=16, seed=0)
    model.train()
    elbo = SVGPELBO(model.likelihood, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    losses = []
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        loss = -elbo(x, y, x.shape[0])
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    assert losses[-1] < losses[0] - 0.1, (losses[0], losses[-1])


# ---------------------------------------------------------------------------
# Trainer integration
# ---------------------------------------------------------------------------


def _toy_training_data(n=256, seed=5):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 2, generator=g, dtype=torch.float64) * 4 - 2
    y = (x[:, 0] * 1.5).sin() + 0.3 * x[:, 1]
    y = y + 0.05 * torch.randn(n, generator=g, dtype=torch.float64)
    return x, y


@pytest.mark.parametrize("nigp", [False, True])
def test_minibatch_trainer_end_to_end(nigp):
    x, y = _toy_training_data()
    model = SVGPR(x, y, num_inducing=24, nigp=nigp, seed=0)
    trainer = MinibatchGPTrainer(
        model,
        batch_size=64,
        num_epochs=15,
        num_inits=1,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs={"lr": 0.05},
        dtype=torch.float64,
        seed=0,
    )
    results = trainer.train()
    assert len(results) == 1
    assert not results[0].get("error")
    assert math.isfinite(results[0]["loss"])

    trained = trainer.model
    trained.eval()
    from gpplus.training.eval import evaluate_svgp_gp_model

    mean, lower, upper, stddev = evaluate_svgp_gp_model(trained, x[:32])
    assert mean.shape == (32,)
    assert (upper > lower).all()
    assert torch.isfinite(stddev).all()


def test_minibatch_trainer_rejects_non_variational_models():
    from gpplus.models.gpr import GPR

    x, y = _toy_training_data(n=32)
    with pytest.raises(TypeError, match="SVGPR or VIRFFGPR"):
        MinibatchGPTrainer(GPR(x, y), batch_size=8, num_epochs=1)


def test_minibatch_trainer_rejects_lbfgs():
    from gpplus.training.optimizers import LBFGSScipy

    x, y = _toy_training_data(n=64)
    model = SVGPR(x, y, num_inducing=8, seed=0)
    with pytest.raises(ValueError, match="stochastic optimizer"):
        MinibatchGPTrainer(
            model, batch_size=16, num_epochs=1, optimizer_class=LBFGSScipy
        )
