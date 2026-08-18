"""Tightness and minibatch-decomposition tests for the variational RFF ELBO.

The ELBO is a bound on the exact marginal log likelihood that is *tight* for a
Gaussian likelihood, so the strongest correctness check is to place ``q`` at its
analytic optimum and require the ELBO to reproduce the existing Woodbury values
bit-for-bit (to float64 tolerance).
"""

from __future__ import annotations

import math

import torch

from gpplus.models.vi_rff_gpr import VIRFFGPR
from gpplus.training.vi_rff_elbo import VIRFFELBO
from gpplus.utils.nigp_utils import effective_noise_variance
from gpplus.utils.rff_utils import (
    woodbury_marginal_log_likelihood,
    woodbury_marginal_log_likelihood_diag_noise,
)
from gpplus.utils.transforms import inv_softplus

# Values are O(1e4), so identities that should hold exactly are checked
# relatively. The dual Woodbury reference forms y^T Sigma^-1 y as a difference of
# two large terms, which costs it several digits at sigma^2 = 1e-3; the dense
# multivariate normal is used as ground truth wherever that matters.
RTOL = 1e-11
TOL = 1e-9
WOODBURY_RTOL = 1e-8


def _dense_log_prob(phi, y, d):
    """Exact ``log N(y | 0, diag(d) + Phi Phi^T)`` with no Woodbury structure."""
    n = y.shape[-1]
    if d.dim() == 0:
        d = d.expand(n)
    cov = torch.diag(d) + phi @ phi.T
    zero = torch.zeros(n, dtype=y.dtype)
    return torch.distributions.MultivariateNormal(zero, cov).log_prob(y)


def _make_model(n=48, d=3, num_rff=6, seed=0, variational_cov="chol", nigp=False):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g, dtype=torch.float64)
    y = torch.randn(n, generator=g, dtype=torch.float64)
    model = VIRFFGPR(
        x,
        y,
        variational_cov=variational_cov,
        num_rff=num_rff,
        ard=True,
        rff_sampling="rff",
        nigp=nigp,
    )
    with torch.no_grad():
        # A noise floor near 1e-6 makes the tight-bound comparison exercise the
        # ill-conditioned regime the dual Woodbury form exists to handle.
        model.likelihood.raw_noise.fill_(-3.0)
        model.mean_module.constant.fill_(0.0)
    model.train()
    return model, x, y


def _set_q(model, m_w, s_mat):
    """Force q(w) = N(m_w, S) exactly, for both covariance parameterizations."""
    with torch.no_grad():
        model.raw_variational_mean.copy_(m_w)
        if model.variational_cov == "chol":
            l_s = torch.linalg.cholesky(s_mat)
            raw = l_s.clone()
            raw.diagonal().copy_(inv_softplus(torch.diagonal(l_s)))
            model.raw_variational_tril.copy_(raw)
        else:
            model.raw_variational_logvar.copy_(torch.diagonal(s_mat).log())


def _optimal_q_homoskedastic(phi, y_centered, noise):
    """m_w = Lam^-1 Phi^T y, S = sigma^2 Lam^-1 with Lam = Phi^T Phi + sigma^2 I."""
    m = phi.shape[-1]
    lam = phi.T @ phi + noise * torch.eye(m, dtype=phi.dtype)
    chol = torch.linalg.cholesky(lam)
    m_w = torch.cholesky_solve((phi.T @ y_centered).unsqueeze(-1), chol).squeeze(-1)
    s_mat = torch.cholesky_solve(torch.eye(m, dtype=phi.dtype), chol) * noise
    return m_w, 0.5 * (s_mat + s_mat.T)


def _optimal_q_diag_noise(phi, y_centered, d):
    """m_w = Lam_d^-1 Phi^T D^-1 y, S = Lam_d^-1 with Lam_d = Phi^T D^-1 Phi + I."""
    m = phi.shape[-1]
    phi_scaled = phi / d.unsqueeze(-1)
    lam = phi.T @ phi_scaled + torch.eye(m, dtype=phi.dtype)
    chol = torch.linalg.cholesky(lam)
    m_w = torch.cholesky_solve((phi_scaled.T @ y_centered).unsqueeze(-1), chol).squeeze(-1)
    s_mat = torch.cholesky_solve(torch.eye(m, dtype=phi.dtype), chol)
    return m_w, 0.5 * (s_mat + s_mat.T)


# ---------------------------------------------------------------------------
# Tightness: at the optimal q the ELBO equals the exact Woodbury MLL
# ---------------------------------------------------------------------------


def test_elbo_at_optimal_q_matches_woodbury_mll():
    model, x, y = _make_model(seed=1)
    with torch.no_grad():
        phi = model.scaled_features(x)
        noise = model.likelihood.noise.reshape(())
        m_w, s_mat = _optimal_q_homoskedastic(phi, y, noise)
    _set_q(model, m_w, s_mat)

    elbo = VIRFFELBO(model.likelihood, model)
    with torch.no_grad():
        got = elbo.full_data_elbo(x, y)
        exact = _dense_log_prob(phi, y, noise)
        woodbury = woodbury_marginal_log_likelihood(noise, phi, y, woodbury_form="primal")
    assert torch.allclose(got, exact, rtol=RTOL, atol=0), (got.item(), exact.item())
    assert torch.allclose(got, woodbury, rtol=WOODBURY_RTOL, atol=0)


def test_elbo_at_optimal_q_matches_diag_noise_woodbury_mll():
    model, x, y = _make_model(seed=2, nigp=True)
    model.nigp_correction_enabled = True
    with torch.no_grad():
        phi = model.scaled_features(x)
        noise = model.likelihood.noise.reshape(())
        m_w, _ = _optimal_q_homoskedastic(phi, y, noise)

        # d depends on the slopes, which depend on m_w, which depends on d. Iterate
        # to the self-consistent point so the bound is evaluated at a genuine
        # optimum of q for the d that q itself induces.
        for _ in range(100):
            model.raw_variational_mean.copy_(m_w)
            d = effective_noise_variance(noise, model.input_noise_var, model.grad_mu(x))
            m_next, s_mat = _optimal_q_diag_noise(phi, y, d)
            shift = (m_next - m_w).abs().max()
            m_w = m_next
            if shift < 1e-14:
                break
        assert shift < 1e-14, f"NIGP slope fixed point did not converge ({shift:g})"
    _set_q(model, m_w, s_mat)

    elbo = VIRFFELBO(model.likelihood, model)
    with torch.no_grad():
        d_final = effective_noise_variance(noise, model.input_noise_var, model.grad_mu(x))
        assert torch.allclose(d, d_final, rtol=RTOL, atol=0)
        got = elbo.full_data_elbo(x, y)
        exact = _dense_log_prob(phi, y, d_final)
        woodbury = woodbury_marginal_log_likelihood_diag_noise(d_final, phi, y)
    assert torch.allclose(got, exact, rtol=RTOL, atol=0), (got.item(), exact.item())
    assert torch.allclose(got, woodbury, rtol=WOODBURY_RTOL, atol=0)


def test_elbo_is_a_lower_bound_away_from_the_optimum():
    model, x, y = _make_model(seed=3)
    with torch.no_grad():
        phi = model.scaled_features(x)
        noise = model.likelihood.noise.reshape(())
        elbo = VIRFFELBO(model.likelihood, model)
        # q is still the prior N(0, I) here, which is not optimal.
        got = elbo.full_data_elbo(x, y)
        exact = _dense_log_prob(phi, y, noise)
    assert got < exact


# ---------------------------------------------------------------------------
# Minibatch decomposition
# ---------------------------------------------------------------------------


def test_minibatch_partition_sums_to_full_data_elbo():
    model, x, y = _make_model(n=60, seed=4)
    elbo = VIRFFELBO(model.likelihood, model)
    with torch.no_grad():
        full = elbo.full_data_elbo(x, y)
        kl = elbo.kl_divergence()
        total = -kl
        for start in range(0, 60, 13):
            total = total + elbo.expected_log_likelihood(
                x[start : start + 13], y[start : start + 13]
            ).sum()
    assert torch.allclose(total, full, rtol=RTOL, atol=0)


def test_scaled_minibatch_elbo_is_unbiased():
    """
    Averaging the ``N/B``-rescaled estimator over equal-sized batches is exact.

    Each point appears in exactly one batch of a partition, so this is the same
    average an epoch of SGD takes -- checking it deterministically rather than by
    Monte Carlo, where the per-batch spread would swamp any useful tolerance.
    """
    n, batch = 60, 12
    model, x, y = _make_model(n=n, seed=5)
    elbo = VIRFFELBO(model.likelihood, model)
    with torch.no_grad():
        estimates = [
            elbo(x[start : start + batch], y[start : start + batch], n)
            for start in range(0, n, batch)
        ]
        got = torch.stack(estimates).mean()
        want = elbo.full_data_elbo(x, y) / n
    assert torch.allclose(got, want, rtol=RTOL, atol=0), (got.item(), want.item())


def test_full_batch_elbo_matches_per_observation_scaling():
    model, x, y = _make_model(n=40, seed=6)
    elbo = VIRFFELBO(model.likelihood, model)
    with torch.no_grad():
        got = elbo(x, y, 40)
        want = elbo.full_data_elbo(x, y) / 40.0
    assert torch.allclose(got, want, rtol=RTOL, atol=0)


# ---------------------------------------------------------------------------
# NIGP slopes
# ---------------------------------------------------------------------------


def test_analytic_grad_mu_matches_autograd():
    model, x, _y = _make_model(seed=7, nigp=True)
    with torch.no_grad():
        model.raw_variational_mean.normal_(generator=torch.Generator().manual_seed(11))
    analytic = model.grad_mu(x)

    x_req = x.detach().clone().requires_grad_(True)
    phi = model.scaled_features(x_req)
    mu = model.mean_module(x_req) + phi @ model.variational_mean.detach()
    reference = torch.autograd.grad(mu.sum(), x_req)[0]
    assert torch.allclose(analytic, reference, rtol=0, atol=1e-10)


def test_grad_mu_is_local_to_the_batch():
    """Slopes for a subset must not depend on the rest of the data."""
    model, x, _y = _make_model(n=64, seed=8, nigp=True)
    with torch.no_grad():
        model.raw_variational_mean.normal_(generator=torch.Generator().manual_seed(12))
    full = model.grad_mu(x)
    part = model.grad_mu(x[:10])
    assert torch.allclose(full[:10], part, rtol=0, atol=TOL)


# ---------------------------------------------------------------------------
# Covariance parameterizations and gradient flow
# ---------------------------------------------------------------------------


def test_diag_elbo_is_bounded_by_chol_elbo():
    """Mean-field q is a restriction of full-covariance q, so it cannot do better."""
    chol_model, x, y = _make_model(seed=9, variational_cov="chol")
    with torch.no_grad():
        phi = chol_model.scaled_features(x)
        noise = chol_model.likelihood.noise.reshape(())
        m_w, s_mat = _optimal_q_homoskedastic(phi, y, noise)
    _set_q(chol_model, m_w, s_mat)

    diag_model, _x, _y = _make_model(seed=9, variational_cov="diag")
    with torch.no_grad():
        diag_model.raw_variational_mean.copy_(m_w)
        diag_model.raw_variational_logvar.copy_(torch.diagonal(s_mat).log())

    with torch.no_grad():
        chol_elbo = VIRFFELBO(chol_model.likelihood, chol_model).full_data_elbo(x, y)
        diag_elbo = VIRFFELBO(diag_model.likelihood, diag_model).full_data_elbo(x, y)
    assert diag_elbo <= chol_elbo + TOL


def test_prior_q_has_zero_kl():
    for cov in ("chol", "diag"):
        model, _x, _y = _make_model(seed=10, variational_cov=cov)
        assert abs(float(model.kl_divergence())) < 1e-12


def test_variational_quad_matches_dense_form():
    for cov in ("chol", "diag"):
        model, x, _y = _make_model(seed=13, variational_cov=cov)
        with torch.no_grad():
            if cov == "chol":
                model.raw_variational_tril.normal_(
                    generator=torch.Generator().manual_seed(3)
                )
                l_s = model.variational_tril()
                s_mat = l_s @ l_s.T
            else:
                model.raw_variational_logvar.normal_(
                    generator=torch.Generator().manual_seed(3)
                )
                s_mat = torch.diag(model.raw_variational_logvar.exp())
            phi = model.scaled_features(x)
            got = model.variational_quad(phi)
            want = ((phi @ s_mat) * phi).sum(dim=-1)
            assert torch.allclose(got, want, rtol=0, atol=1e-11)
            assert torch.allclose(model.variational_trace(), torch.diagonal(s_mat).sum())
            assert torch.allclose(model.variational_logdet(), torch.logdet(s_mat))


def test_gradients_flow_to_all_trainable_parameters():
    for cov in ("chol", "diag"):
        model, x, y = _make_model(seed=14, variational_cov=cov, nigp=True)
        model.nigp_correction_enabled = True
        with torch.no_grad():
            # At the prior mean m_w = 0 the posterior mean is constant, so the
            # NIGP slopes -- and hence the input-noise gradient -- are exactly zero.
            model.raw_variational_mean.normal_(generator=torch.Generator().manual_seed(7))
        elbo = VIRFFELBO(model.likelihood, model)
        loss = -elbo(x[:16], y[:16], x.shape[0])
        loss.backward()

        expected = {
            "raw_variational_mean",
            "likelihood.noise_covar.raw_noise",
            "covar_module.raw_outputscale",
            "covar_module.base_kernel.raw_lengthscale",
            "raw_input_noise",
            "mean_module.raw_constant" if hasattr(model.mean_module, "raw_constant") else "mean_module.constant",
        }
        expected.add(
            "raw_variational_tril" if cov == "chol" else "raw_variational_logvar"
        )
        grads = {name: p.grad for name, p in model.named_parameters()}
        for name in expected:
            assert name in grads, f"{name} missing from {sorted(grads)}"
            g = grads[name]
            assert g is not None, f"no gradient for {name} ({cov})"
            assert torch.isfinite(g).all(), f"non-finite gradient for {name} ({cov})"
            assert g.abs().sum() > 0, f"zero gradient for {name} ({cov})"


def test_nigp_is_inert_at_the_prior_mean():
    """m_w = 0 gives a constant posterior mean, so there is no slope to amplify."""
    model, x, _y = _make_model(seed=19, nigp=True)
    model.nigp_correction_enabled = True
    assert torch.count_nonzero(model.grad_mu(x)) == 0
    with torch.no_grad():
        noise = model.likelihood.noise.reshape(())
        d = model.observation_noise_var(x)
    assert torch.allclose(d, noise.expand_as(d), rtol=0, atol=TOL)


def test_kl_beta_scales_the_regularizer():
    model, x, y = _make_model(seed=15)
    with torch.no_grad():
        model.raw_variational_mean.normal_(generator=torch.Generator().manual_seed(4))
        full = VIRFFELBO(model.likelihood, model).full_data_elbo(x, y)
        half = VIRFFELBO(model.likelihood, model, kl_beta=0.5).full_data_elbo(x, y)
        kl = model.kl_divergence()
    assert torch.allclose(half - full, 0.5 * kl, rtol=0, atol=TOL)


# ---------------------------------------------------------------------------
# Warm start and prediction
# ---------------------------------------------------------------------------


def test_warm_start_reaches_the_tight_bound():
    model, x, y = _make_model(n=50, seed=16)
    model.warm_start_from_subset(x, y, jitter=0.0)
    with torch.no_grad():
        phi = model.scaled_features(x)
        noise = model.likelihood.noise.reshape(())
        got = VIRFFELBO(model.likelihood, model).full_data_elbo(x, y)
        exact = _dense_log_prob(phi, y, noise)
    assert torch.allclose(got, exact, rtol=1e-9, atol=0), (got.item(), exact.item())


def test_predict_matches_woodbury_after_warm_start():
    from gpplus.utils.rff_utils import woodbury_predict

    model, x, y = _make_model(n=50, seed=17)
    model.warm_start_from_subset(x, y, jitter=0.0)
    model.eval()
    x_test = torch.randn(9, x.shape[-1], generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    with torch.no_grad():
        mean, lower, upper = model.predict(x_test, return_latent=True)
        phi_train = model.scaled_features(x)
        phi_test = model.scaled_features(x_test)
        noise = model.likelihood.noise.reshape(())
        ref_mean, ref_var = woodbury_predict(noise, phi_train, phi_test, y, jitter=0.0)
    assert torch.allclose(mean, ref_mean, rtol=0, atol=1e-7)
    ref_std = ref_var.clamp_min(0.0).sqrt()
    assert torch.allclose((upper - lower) / 4.0, ref_std, rtol=0, atol=1e-7)


# ---------------------------------------------------------------------------
# End-to-end minibatch training
# ---------------------------------------------------------------------------


def _train(model, *, batch_size, num_epochs=12, **kwargs):
    from gpplus.training.minibatch_trainer import MinibatchGPTrainer

    trainer = MinibatchGPTrainer(
        model,
        batch_size=batch_size,
        num_epochs=num_epochs,
        num_inits=1,
        seed=0,
        dtype=torch.float64,
        optimizer_kwargs={"lr": 0.05},
        stop_conditions=[],
        **kwargs,
    )
    trainer.train()
    return trainer


def test_minibatch_training_improves_the_elbo():
    model, x, y = _make_model(n=64, seed=20)
    before = float(VIRFFELBO(model.likelihood, model).full_data_elbo(x, y))
    trainer = _train(model, batch_size=16)
    after = float(
        VIRFFELBO(trainer.model.likelihood, trainer.model).full_data_elbo(x, y)
    )
    assert after > before


def test_minibatch_training_runs_with_nigp_freeze():
    from gpplus.training.callbacks import NIGPInputNoiseFreezeCallback

    model, x, y = _make_model(n=64, seed=21, nigp=True)
    callback = NIGPInputNoiseFreezeCallback(freeze_epochs=4, verbose=False)
    trainer = _train(
        model,
        batch_size=16,
        num_epochs=10,
        warm_start_points=32,
        callbacks=[callback],
    )
    trained = trainer.model
    # The freeze window ends mid-run, so the correction must be live at the end.
    assert trained.nigp_correction_enabled
    assert trained.raw_input_noise.requires_grad
    assert torch.isfinite(trained.raw_input_noise).all()
    assert torch.isfinite(VIRFFELBO(trained.likelihood, trained).full_data_elbo(x, y))


def test_minibatch_rejects_non_variational_models():
    from gpplus.models.rff_gpr import RFFGPR
    from gpplus.training.minibatch_trainer import MinibatchGPTrainer

    g = torch.Generator().manual_seed(22)
    x = torch.randn(20, 2, generator=g, dtype=torch.float64)
    y = torch.randn(20, generator=g, dtype=torch.float64)
    model = RFFGPR(x, y, num_rff=4)
    try:
        MinibatchGPTrainer(model, batch_size=4, num_inits=1, num_epochs=1)
    except TypeError as exc:
        assert "VIRFFGPR" in str(exc)
    else:
        raise AssertionError("expected a TypeError for a non-variational model")


def test_variational_params_are_excluded_from_sobol_init():
    """A full Cholesky factor has m^2 entries, far past any Sobol basis."""
    from gpplus.training.parameter_initializer import DefaultParameterInitializer

    model, _x, _y = _make_model(seed=23, num_rff=16)
    init = DefaultParameterInitializer(num_inits=2, seed=0)
    init.setup(model)
    assert init.num_params < model.num_features

    before = model.raw_variational_tril.detach().clone()
    init.initialize(model, 0)
    assert torch.equal(model.raw_variational_tril, before)


def test_expected_log_likelihood_matches_closed_form():
    model, x, y = _make_model(n=20, seed=18)
    with torch.no_grad():
        model.raw_variational_mean.normal_(generator=torch.Generator().manual_seed(6))
        phi = model.scaled_features(x)
        m_w = model.variational_mean
        s_mat = model.variational_tril() @ model.variational_tril().T
        d = model.likelihood.noise.reshape(())
        resid = y - model.mean_module(x) - phi @ m_w
        f_var = ((phi @ s_mat) * phi).sum(dim=-1)
        want = -0.5 * (math.log(2 * math.pi) + d.log() + (resid.pow(2) + f_var) / d)
        got = model.expected_log_likelihood(x, y)
    assert torch.allclose(got, want, rtol=0, atol=1e-12)
