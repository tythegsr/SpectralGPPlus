"""Sparse variational GP regression over inducing points, for minibatch SGD.

The exact marginal likelihood ``log N(y | 0, K + sigma^2 I)`` couples every
observation through ``log|K + sigma^2 I|``, so it admits no unbiased minibatch
gradient. Introducing inducing variables ``u = f(Z)`` at ``M`` locations with a
Gaussian variational posterior ``q(u) = N(m, S)`` fixes that:

    ELBO = sum_i E_q[log N(y_i | f_i, d_i)] - KL(q(u) || p(u))
         ~ (N/B) sum_{i in batch} l_i - KL

The likelihood term is a plain sum over data points and the KL is
data-independent, so the minibatch estimator is unbiased.

Unlike :class:`~gpplus.models.vi_rff_gpr.VIRFFGPR`, this uses the ordinary
ARD RBF kernel -- no random Fourier features are involved anywhere.

NIGP composes cleanly here for the same reason it does in the RFF variational
model: the posterior mean ``mu(x) = m(x) + k(x, Z) K_zz^-1 (m - m(Z))`` depends
only on ``x`` and the variational parameters, so ``d mu / d x`` is a local
quantity computable per minibatch with no solve over the training set.
"""

from __future__ import annotations

import math
from typing import Optional

import gpytorch
import torch
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from torch import Tensor, nn

from ..config import logger
from ..kernels import GaussianKernel, LogScaleKernel
from ..likelihoods import LogGaussianLikelihood
from ..utils.nigp_utils import (
    effective_noise_variance,
    input_noise_softclamp,
    nigp_correction_enabled,
    raw_input_noise_init_value,
)


def _select_inducing_points(
    train_x: Tensor,
    num_inducing: int,
    seed: Optional[int],
) -> Tensor:
    """Random distinct subset of the training inputs, used to seed ``Z``."""
    n = int(train_x.shape[0])
    m = min(int(num_inducing), n)
    if m < 1:
        raise ValueError(f"num_inducing must be >= 1, got {num_inducing}.")
    generator = torch.Generator().manual_seed(0 if seed is None else int(seed))
    idx = torch.randperm(n, generator=generator)[:m].to(train_x.device)
    return train_x[idx].clone()


class SVGPR(gpytorch.models.ApproximateGP):
    """
    Sparse variational GP with learnable inducing locations.

    Mean, kernel and likelihood defaults match :class:`~gpplus.models.gpr.GPR`
    (``ConstantMean``, ``LogScaleKernel(GaussianKernel(ARD))``,
    ``LogGaussianLikelihood``) so hyperparameter initialization, the NIGP freeze
    callback and the existing metrics all behave the same way.

    Train with :class:`~gpplus.training.svgp_elbo.SVGPELBO` inside
    :class:`~gpplus.training.minibatch_trainer.MinibatchGPTrainer`.

    Parameters
    ----------
    num_inducing :
        Number of inducing locations ``M``. Cost is ``O(B M^2)`` per step.
    inducing_points :
        Explicit ``(M, D)`` starting locations; defaults to a seeded random
        subset of ``train_x``.
    learn_inducing_locations :
        Whether ``Z`` is optimized alongside the hyperparameters.
    """

    def __init__(
        self,
        train_x: Tensor,
        train_y: Tensor,
        *,
        likelihood: gpytorch.likelihoods.Likelihood | None = None,
        mean_module: gpytorch.means.Mean | None = None,
        kernel_module: gpytorch.kernels.Kernel | None = None,
        num_inducing: int = 512,
        inducing_points: Tensor | None = None,
        learn_inducing_locations: bool = True,
        nigp: bool = False,
        input_noise_init: float | None = None,
        seed: int | None = None,
    ):
        if not isinstance(train_x, torch.Tensor) or not isinstance(train_y, torch.Tensor):
            raise TypeError("train_x and train_y must be torch.Tensor instances.")

        dtype = train_x.dtype
        input_dim = int(train_x.shape[-1])

        if inducing_points is None:
            inducing_points = _select_inducing_points(train_x, num_inducing, seed)
        inducing_points = inducing_points.to(dtype=dtype).clone()
        num_inducing = int(inducing_points.shape[0])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing, dtype=dtype
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=bool(learn_inducing_locations),
        )
        super().__init__(variational_strategy)

        self.dtype = dtype
        self.num_inducing = num_inducing
        self.batch_shape = torch.Size([])

        if likelihood is None:
            likelihood = LogGaussianLikelihood()
        if not isinstance(likelihood, gpytorch.likelihoods.Likelihood):
            raise TypeError("likelihood must be a gpytorch.likelihoods.Likelihood.")
        if mean_module is None:
            mean_module = gpytorch.means.ConstantMean()
        if kernel_module is None:
            kernel_module = LogScaleKernel(GaussianKernel(ard_num_dims=input_dim))
            logger.warning(
                "No kernel_module provided. Using LogScaleKernel(GaussianKernel("
                "ard_num_dims=%s)) for SVGPR.",
                input_dim,
            )

        self.likelihood = likelihood
        self.mean_module = mean_module
        self.covar_module = kernel_module

        self.nigp = bool(nigp)
        # Set False during the freeze warm-start so the ELBO is a standard SVGP.
        self.nigp_correction_enabled = True
        if self.nigp:
            raw_init = raw_input_noise_init_value(input_noise_init, dtype=dtype)
            self.register_parameter(
                "raw_input_noise", nn.Parameter(raw_init.expand(input_dim).clone())
            )
            self.register_constraint("raw_input_noise", input_noise_softclamp(dtype=dtype))
            logger.info(
                "NIGP enabled on SVGPR: learnable per-dim input noise "
                "(D=%s, sigma_x=10^SoftClamp(raw) in (1e-6, 1)).",
                input_dim,
            )

        # CholeskyVariationalDistribution allocates in the default dtype, so
        # align the whole module rather than each submodule individually.
        self.to(dtype=dtype)

        self.set_train_data(train_x, train_y)
        logger.info(
            "SVGPR: %s inducing points over %s input dims (learned=%s).",
            num_inducing,
            input_dim,
            bool(learn_inducing_locations),
        )

    # ------------------------------------------------------------------
    # ExactGP-compatible training-data surface, so GPTrainer works unchanged
    # ------------------------------------------------------------------

    def set_train_data(self, inputs: Tensor, targets: Tensor, strict: bool = False) -> None:
        """
        Attach training data without registering it in the ``state_dict``.

        SVGP does not condition on the training set at prediction time; this
        exists so :class:`~gpplus.training.trainer.GPTrainer` and the minibatch
        loop can read ``train_inputs`` / ``train_targets`` the same way they do
        for exact and RFF models.
        """
        self.train_inputs = (inputs,)
        self.train_targets = targets

    def forward(self, x: Tensor) -> gpytorch.distributions.MultivariateNormal:
        """Prior at ``x``; the variational strategy turns this into ``q(f)``."""
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input x must be a torch.Tensor.")
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )

    # ------------------------------------------------------------------
    # Variational posterior
    # ------------------------------------------------------------------

    def kl_divergence(self) -> Tensor:
        """``KL(q(u) || p(u))``, independent of the data."""
        return self.variational_strategy.kl_divergence()

    @property
    def inducing_points(self) -> Tensor:
        return self.variational_strategy.inducing_points

    # ------------------------------------------------------------------
    # NIGP
    # ------------------------------------------------------------------

    @property
    def input_noise(self) -> Tensor:
        """Per-dimension input noise std ``sigma_x = 10^SoftClamp(raw)``."""
        if not getattr(self, "nigp", False) or not hasattr(self, "raw_input_noise"):
            raise AttributeError("Model has no NIGP input_noise (construct with nigp=True).")
        return torch.pow(
            10.0, self.raw_input_noise_constraint.transform(self.raw_input_noise)
        )

    @property
    def input_noise_var(self) -> Tensor:
        """Per-dimension input noise variance ``sigma_x^2``."""
        s = self.input_noise
        return s * s

    def grad_mu(self, x: Tensor) -> Tensor:
        """
        Detached ``d mu / d x`` of the variational predictive mean at ``x``.

        ``mu(x) = m(x) + k(x, Z) K_zz^-1 (m_u - m(Z))`` involves only ``x``, the
        inducing state and the kernel, so this is local to the batch: no solve
        over the training set and no cached ``(N, D)`` slope matrix, which is
        what makes NIGP compatible with minibatching.

        Exactly at the prior mean the predictive mean is constant and the slopes
        vanish, which would starve ``sigma_x`` of gradient; GPyTorch's default
        variational initialization offsets ``m_u`` slightly to avoid that, and
        the :class:`~gpplus.training.callbacks.NIGPInputNoiseFreezeCallback`
        warm-up window keeps the correction off until ``q`` is meaningful.
        """
        with torch.enable_grad():
            x_req = x.detach().clone().requires_grad_(True)
            mean = self(x_req).mean
            grad = torch.autograd.grad(mean.sum(), x_req)[0]
        return grad.detach().to(dtype=x.dtype)

    def observation_noise_var(self, x: Tensor) -> Tensor:
        """Per-point ``d_i``: ``sigma_y^2``, plus the NIGP term when active."""
        noise = self.likelihood.noise
        if noise.dim() > 1:
            noise = noise.reshape(noise.shape[0])
        if nigp_correction_enabled(self):
            return effective_noise_variance(noise, self.input_noise_var, self.grad_mu(x))
        noise_flat = noise.clamp_min(1e-12).reshape(-1)
        if noise_flat.numel() != 1:
            raise ValueError(
                f"SVGPR expects scalar likelihood noise, got {tuple(noise.shape)}."
            )
        return noise_flat[0].expand(x.shape[0])

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        test_x: Tensor,
        return_latent: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Posterior mean and 2-sigma bounds from ``q(f)``; needs no training data."""
        dist = self(test_x)
        f_mean = dist.mean
        f_var = dist.variance.clamp_min(0.0)
        if return_latent:
            f_std = f_var.sqrt()
            return f_mean, f_mean - 2 * f_std, f_mean + 2 * f_std
        obs_std = (f_var + self.observation_noise_var(test_x)).clamp_min(0.0).sqrt()
        return f_mean, f_mean - 2 * obs_std, f_mean + 2 * obs_std

    def expected_log_likelihood(self, x_batch: Tensor, y_batch: Tensor) -> Tensor:
        """
        Per-point ``E_q[log N(y_i | f_i, d_i)]``, shape ``(B,)``.

        The variational variance of ``f`` enters additively next to the squared
        residual, which is what makes the bound tight at the optimum.
        """
        dist = self(x_batch)
        resid = y_batch - dist.mean
        f_var = dist.variance
        d = self.observation_noise_var(x_batch)
        return -0.5 * (math.log(2.0 * math.pi) + d.log() + (resid.pow(2) + f_var) / d)
