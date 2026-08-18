"""Variational RFF Gaussian process regression for minibatch SGD training.

The exact marginal likelihood couples every observation through ``log|K+σ²I|``,
so it admits no unbiased minibatch gradient. Introducing an explicit variational
posterior over the RFF weights removes that coupling: with

    f(x) = m(x) + φ(x)ᵀ w,      w ~ N(0, I_m),      q(w) = N(m_w, S)

the ELBO is

    L = Σ_i E_q[log N(y_i | m(x_i) + φ(x_i)ᵀ w, d_i)] − KL(q(w) ‖ N(0, I_m))

whose likelihood term is a plain sum over data points, so ``(N/B) Σ_{i∈batch}``
is an unbiased estimator. Because :meth:`RFFGPR.scaled_features` already absorbs
the output scale (``Φ Φᵀ = K``), the prior on ``w`` is exactly ``N(0, I)`` and no
whitening transform is needed.

The bound is tight for Gaussian likelihoods: at the optimal ``q`` it recovers the
exact Woodbury marginal log likelihood, so this is the same model as
:class:`~gpplus.models.rff_gpr.RFFGPR`, only with a different (minibatchable)
training objective.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn

from ..config import logger
from ..utils.nigp_utils import effective_noise_variance, nigp_correction_enabled
from ..utils.rff_utils import woodbury_factor_dual
from ..utils.transforms import inv_softplus
from .rff_gpr import RFFGPR, _drop_singleton_batch

VariationalCov = Literal["chol", "diag"]

# Below this the softplus diagonal of L would make S numerically singular.
_MIN_CHOL_DIAG = 1e-8


class VIRFFGPR(RFFGPR):
    """
    RFF GP with an explicit Gaussian variational posterior over the feature weights.

    Train with :class:`~gpplus.training.vi_rff_elbo.VIRFFELBO` in
    :class:`~gpplus.training.minibatch_trainer.MinibatchGPTrainer`. Kernel,
    likelihood, mean and NIGP parameters are inherited unchanged from
    :class:`~gpplus.models.rff_gpr.RFFGPR`, so SORF sampling, ARD and the
    learnable per-dimension input noise all behave identically.

    Parameters
    ----------
    variational_cov :
        ``"chol"`` parameterizes the full covariance ``S = L Lᵀ`` with ``L``
        lower triangular (exact, ``O(B m²)`` per batch). ``"diag"`` uses
        ``S = diag(exp(raw))`` (``O(B m)`` per batch, but underestimates the
        predictive variance and cannot reproduce the exact Woodbury posterior).
    """

    def __init__(
        self,
        train_x: Tensor,
        train_y: Tensor,
        *,
        variational_cov: VariationalCov = "chol",
        **rff_kwargs,
    ):
        if variational_cov not in ("chol", "diag"):
            raise ValueError(
                f"variational_cov must be 'chol' or 'diag', got {variational_cov!r}."
            )
        super().__init__(train_x, train_y, **rff_kwargs)
        if len(self.batch_shape) > 0:
            raise ValueError(
                "VIRFFGPR does not support batched multi-init (batch_shape="
                f"{self.batch_shape}); minibatch training runs inits independently."
            )

        self.variational_cov = variational_cov
        num_features = 2 * int(self.num_rff)
        self.num_features = num_features

        # q(w) starts at the prior N(0, I): zero mean, S = I.
        self.register_parameter(
            "raw_variational_mean",
            nn.Parameter(torch.zeros(num_features, dtype=self.dtype)),
        )
        if variational_cov == "chol":
            raw = torch.zeros(num_features, num_features, dtype=self.dtype)
            raw.diagonal().fill_(inv_softplus(torch.tensor(1.0, dtype=self.dtype)).item())
            self.register_parameter("raw_variational_tril", nn.Parameter(raw))
        else:
            self.register_parameter(
                "raw_variational_logvar",
                nn.Parameter(torch.zeros(num_features, dtype=self.dtype)),
            )

        logger.info(
            "VIRFFGPR: variational posterior over %s RFF weights (cov=%s).",
            num_features,
            variational_cov,
        )

    # ------------------------------------------------------------------
    # Variational posterior q(w) = N(m_w, S)
    # ------------------------------------------------------------------

    @property
    def variational_mean(self) -> Tensor:
        return self.raw_variational_mean

    def variational_tril(self) -> Tensor:
        """Lower-triangular ``L`` with ``S = L Lᵀ`` (``variational_cov='chol'``)."""
        if self.variational_cov != "chol":
            raise AttributeError(
                "variational_tril requires variational_cov='chol'; "
                f"model uses {self.variational_cov!r}."
            )
        raw = self.raw_variational_tril
        diag = nn.functional.softplus(torch.diagonal(raw)).clamp_min(_MIN_CHOL_DIAG)
        return torch.tril(raw, diagonal=-1) + torch.diag(diag)

    def variational_logdet(self) -> Tensor:
        """``log|S|``."""
        if self.variational_cov == "chol":
            return 2.0 * torch.log(torch.diagonal(self.variational_tril())).sum()
        return self.raw_variational_logvar.sum()

    def variational_trace(self) -> Tensor:
        """``tr(S)``."""
        if self.variational_cov == "chol":
            return self.variational_tril().pow(2).sum()
        return self.raw_variational_logvar.exp().sum()

    def variational_quad(self, phi: Tensor) -> Tensor:
        """Row-wise ``φᵀ S φ`` for features ``phi`` of shape ``(..., n, m)``."""
        if self.variational_cov == "chol":
            # phi @ L has shape (..., n, m); the row norms are exactly phi^T L L^T phi.
            return (phi @ self.variational_tril()).pow(2).sum(dim=-1)
        return (phi.pow(2) * self.raw_variational_logvar.exp()).sum(dim=-1)

    def kl_divergence(self) -> Tensor:
        """``KL(q(w) ‖ N(0, I_m)) = ½ (tr(S) + ‖m_w‖² − m − log|S|)``."""
        m_w = self.variational_mean
        return 0.5 * (
            self.variational_trace()
            + m_w.pow(2).sum()
            - float(self.num_features)
            - self.variational_logdet()
        )

    # ------------------------------------------------------------------
    # NIGP slopes: purely local under the variational mean
    # ------------------------------------------------------------------

    def grad_mu(self, x: Tensor) -> Tensor:
        """
        Detached ``∇_x μ`` of the variational posterior mean at ``x``.

        Since ``μ(x) = m(x) + φ(x)ᵀ m_w`` with ``m_w`` a free parameter, this
        needs only the points in ``x`` -- no solve over the training set and no
        cached ``(N, D)`` slope matrix, unlike the Woodbury NIGP path in
        :func:`~gpplus.utils.nigp_utils.posterior_mean_grad_wrt_x`.

        At the prior ``m_w = 0`` the posterior mean is constant, so the slopes are
        identically zero and ``σ_x`` receives no gradient. Warm-starting ``q`` or
        using :class:`~gpplus.training.callbacks.NIGPInputNoiseFreezeCallback`
        moves ``m_w`` away from zero before the input noise starts learning.
        """
        from ..utils.woodbury_mll_autograd import (
            featurize_rbf_scaled_omega,
            rff_grad_mu_from_proj,
        )

        x_det = x.detach()
        w = self.variational_mean.detach()

        from ..kernels import LogScaleKernel, RFFKernel
        from gpytorch.means import ConstantMean, ZeroMean

        base = self.covar_module.base_kernel if isinstance(self.covar_module, LogScaleKernel) else None
        analytic = (
            isinstance(base, RFFKernel)
            and hasattr(base, "randn_weights")
            and isinstance(self.mean_module, (ConstantMean, ZeroMean))
        )
        if analytic:
            with torch.no_grad():
                _phi, proj, omega, scale_out = featurize_rbf_scaled_omega(
                    x_det,
                    base.randn_weights,
                    base.lengthscale,
                    self.covar_module.outputscale,
                    int(base.num_samples),
                )
                grad = rff_grad_mu_from_proj(proj, omega, scale_out, w, int(base.num_samples))
            return grad.to(dtype=x.dtype).detach()

        with torch.enable_grad():
            x_req = x_det.clone().requires_grad_(True)
            phi = self.scaled_features(x_req)
            mean = self.mean_module(x_req)
            if mean.dim() > 1 and mean.shape[0] == 1 and phi.dim() == 2:
                mean = mean.squeeze(0)
            mu = mean + (phi * w.to(dtype=phi.dtype)).sum(dim=-1)
            grad = torch.autograd.grad(mu.sum(), x_req)[0]
        return grad.to(dtype=x.dtype).detach()

    def observation_noise_var(self, x: Tensor) -> Tensor:
        """Per-point ``d_i``: ``σ_y²``, plus the NIGP input-noise term when active."""
        noise = self.likelihood.noise
        if noise.dim() > 1:
            noise = noise.reshape(noise.shape[0])
        if nigp_correction_enabled(self):
            return effective_noise_variance(noise, self.input_noise_var, self.grad_mu(x))
        noise_flat = noise.clamp_min(1e-12).reshape(-1)
        if noise_flat.numel() != 1:
            raise ValueError(
                f"VIRFFGPR expects scalar likelihood noise, got {tuple(noise.shape)}."
            )
        return noise_flat[0].expand(x.shape[0])

    # ------------------------------------------------------------------
    # Warm start and prediction
    # ------------------------------------------------------------------

    def warm_start_from_subset(
        self,
        x_sub: Tensor,
        y_sub: Tensor,
        *,
        jitter: float = 1e-6,
    ) -> None:
        """
        Set ``q`` to the analytically optimal posterior for ``(x_sub, y_sub)``.

        For fixed homoskedastic hyperparameters the optimal variational posterior
        is ``m_w = Λ⁻¹ Φᵀ y``, ``S = σ² Λ⁻¹`` with ``Λ = ΦᵀΦ + σ² I``, so this
        starts SGD from the exact answer on a subsample instead of the prior.
        """
        with torch.no_grad():
            x_sub = x_sub.to(device=self.raw_variational_mean.device, dtype=self.dtype)
            y_sub = y_sub.to(device=self.raw_variational_mean.device, dtype=self.dtype)
            phi = self.scaled_features(x_sub)
            noise = self.likelihood.noise
            if noise.dim() > 1:
                noise = noise.reshape(noise.shape[0])
            mean_sub = self.mean_module(x_sub)
            if mean_sub.dim() > 1 and mean_sub.shape[0] == 1:
                mean_sub = mean_sub.squeeze(0)
            y_centered = y_sub - mean_sub

            chol, noise_c, z_lin = woodbury_factor_dual(noise, phi, jitter=jitter)
            phi_ty = (z_lin.transpose(-1, -2) @ y_centered.to(chol.dtype).unsqueeze(-1))
            m_w = torch.cholesky_solve(phi_ty, chol).squeeze(-1)
            self.raw_variational_mean.copy_(m_w.to(dtype=self.dtype))

            eye = torch.eye(chol.shape[-1], dtype=chol.dtype, device=chol.device)
            lambda_inv = torch.cholesky_solve(eye, chol)
            s_mat = lambda_inv * noise_c.reshape(())
            s_mat = 0.5 * (s_mat + s_mat.transpose(-1, -2))
            s_mat.diagonal().add_(jitter)
            if self.variational_cov == "chol":
                l_s = torch.linalg.cholesky(s_mat).to(dtype=self.dtype)
                raw = l_s.clone()
                diag = torch.diagonal(l_s).clamp_min(_MIN_CHOL_DIAG)
                raw.diagonal().copy_(inv_softplus(diag))
                self.raw_variational_tril.copy_(raw)
            else:
                var_diag = torch.diagonal(s_mat).clamp_min(1e-30).to(dtype=self.dtype)
                self.raw_variational_logvar.copy_(var_diag.log())

        logger.info(
            "VIRFFGPR warm start from %s points (optimal q for the subsample).",
            int(x_sub.shape[0]),
        )

    def warm_start_from_train(
        self,
        *,
        max_points: int = 8192,
        generator: torch.Generator | None = None,
        jitter: float = 1e-6,
    ) -> None:
        """Warm start from a random subsample of at most ``max_points`` training points."""
        train_x = _drop_singleton_batch(self.train_inputs[0])
        train_y = _drop_singleton_batch(self.train_targets)
        n = int(train_x.shape[0])
        if n > max_points:
            # Keep the generator on CPU; move indices to the data device.
            idx = torch.randperm(n, generator=generator)[:max_points].to(train_x.device)
            train_x = train_x[idx]
            train_y = train_y[idx]
        self.warm_start_from_subset(train_x, train_y, jitter=jitter)

    def predict(
        self,
        test_x: Tensor,
        jitter: float = 1e-6,
        return_latent: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Posterior mean and bounds from ``q(w)``; needs no training data.

        ``μ(x*) = m(x*) + φ(x*)ᵀ m_w`` and ``Var[f(x*)] = φ(x*)ᵀ S φ(x*)``.
        Observation bounds add ``d(x*)``, including the NIGP correction when
        the input-noise term is active.
        """
        phi = self.scaled_features(test_x)
        f_mean = self.mean_module(test_x) + phi @ self.variational_mean
        f_var = self.variational_quad(phi).clamp_min(0.0)

        if return_latent:
            f_std = f_var.sqrt()
            return f_mean, f_mean - 2 * f_std, f_mean + 2 * f_std

        obs_std = (f_var + self.observation_noise_var(test_x)).clamp_min(0.0).sqrt()
        return f_mean, f_mean - 2 * obs_std, f_mean + 2 * obs_std

    def expected_log_likelihood(
        self,
        x_batch: Tensor,
        y_batch: Tensor,
    ) -> Tensor:
        """
        Per-point ``E_q[log N(y_i | m(x_i) + φ(x_i)ᵀ w, d_i)]``, shape ``(B,)``.

        The variance of ``φᵀw`` under ``q`` enters additively alongside the
        squared residual, which is what makes the bound tight at the optimum.
        """
        phi = self.scaled_features(x_batch)
        mean_b = self.mean_module(x_batch)
        if mean_b.dim() > 1 and mean_b.shape[0] == 1 and phi.dim() == 2:
            mean_b = mean_b.squeeze(0)
        resid = y_batch - mean_b - phi @ self.variational_mean
        f_var = self.variational_quad(phi)
        d = self.observation_noise_var(x_batch)
        return -0.5 * (math.log(2.0 * math.pi) + d.log() + (resid.pow(2) + f_var) / d)
