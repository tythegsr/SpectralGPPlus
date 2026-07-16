"""Low-rank NN (Deep Basis) GP regression with Woodbury inference."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Callable

import gpytorch
import torch
import torch.nn as nn
from gpytorch.distributions import MultivariateNormal
from linear_operator.operators import LowRankRootLinearOperator

from ..config import logger
from ..kernels import LRNNKernel
from ..likelihoods import LogGaussianLikelihood
from ..utils.lrnn_utils import woodbury_predict_lrnn


def _drop_singleton_batch(t: torch.Tensor) -> torch.Tensor:
    """Woodbury helpers expect (n, m); GPyTorch may store train_x as (1, n, d)."""
    if t.dim() >= 3 and t.shape[0] == 1:
        return t.squeeze(0)
    return t


class LRNNGPR(gpytorch.models.ExactGP):
    """
    GP regression with an :class:`~gpplus.kernels.LRNNKernel` and Woodbury inference.

    Implements the Deep Basis Kernel (DBK) pathway of Zhu et al. (arxiv:2505.18526):
    ``K = Φ Φ^T`` with neural basis ``φ_θ``, exact inference via Woodbury, and optional
    variance correction (trace regularization + diagonal correction).

    Use with :class:`~gpplus.training.lrnn_mll.LRNNWoodburyMarginalLogLikelihood`
    in :class:`~gpplus.training.GPTrainer`.
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood | None = None,
        mean_module: gpytorch.means.Mean | None = None,
        kernel_module: gpytorch.kernels.Kernel | None = None,
        hidden_dims: Sequence[int] = (128, 128),
        feature_rank: int = 128,
        activation: Callable[[], nn.Module] | type[nn.Module] = nn.Tanh,
        layer_config: dict | None = None,
        variance_correction: bool = True,
    ):
        if not isinstance(train_x, torch.Tensor) or not isinstance(train_y, torch.Tensor):
            raise TypeError("train_x and train_y must be torch.Tensor instances.")

        self.dtype = train_x.dtype
        if likelihood is None:
            likelihood = LogGaussianLikelihood()
            logger.warning("No likelihood provided. Using LogGaussianLikelihood.")
        if mean_module is None:
            mean_module = gpytorch.means.ConstantMean()
        if kernel_module is None:
            input_dim = int(train_x.shape[-1])
            kernel_module = LRNNKernel(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                feature_rank=feature_rank,
                activation=activation,
                layer_config=layer_config,
            )
            logger.warning(
                "No kernel_module provided. Using LRNNKernel("
                f"input_dim={input_dim}, hidden_dims={tuple(hidden_dims)}, "
                f"feature_rank={feature_rank}, variance_correction={variance_correction})."
            )

        if not isinstance(likelihood, gpytorch.likelihoods.Likelihood):
            raise TypeError("likelihood must be a gpytorch.likelihoods.Likelihood.")

        super().__init__(train_x, train_y, likelihood)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.feature_rank = int(feature_rank)
        self.variance_correction = bool(variance_correction)
        self.mean_module = mean_module.to(dtype=self.dtype)
        self.covar_module = kernel_module.to(dtype=self.dtype)
        self.likelihood = self.likelihood.to(dtype=self.dtype)
        self._train_z_cache: torch.Tensor | None = None
        self._train_z_cache_key: tuple | None = None

    @property
    def _lrnn_kernel(self) -> LRNNKernel:
        kernel = self.covar_module
        if isinstance(kernel, LRNNKernel):
            return kernel
        base = getattr(kernel, "base_kernel", None)
        if isinstance(base, LRNNKernel):
            return base
        raise TypeError("covar_module must be LRNNKernel (or wrap one as base_kernel).")

    def _feature_cache_key(self, x: torch.Tensor) -> tuple:
        # Invalidate when any trainable NN weight changes (version bump) or x identity.
        ver = getattr(self._lrnn_kernel, "_feature_cache_version", 0)
        # Hash a few parameter norms as a cheap change detector in eval mode.
        weight_sig = 0.0
        for p in self._lrnn_kernel.parameters():
            if p.requires_grad:
                weight_sig += float(p.detach().float().sum().cpu())
                break
        return (id(x), ver, weight_sig)

    def invalidate_feature_cache(self) -> None:
        self._train_z_cache = None
        self._train_z_cache_key = None
        if hasattr(self._lrnn_kernel, "bump_feature_cache_version"):
            self._lrnn_kernel.bump_feature_cache_version()

    def unscaled_features(self, x: torch.Tensor) -> torch.Tensor:
        z = self._lrnn_kernel.featurize(x)
        return _drop_singleton_batch(z)

    def featurize(self, x: torch.Tensor) -> torch.Tensor:
        """Neural basis features ``φ(x)`` of shape ``(n, r)``."""
        return self.unscaled_features(x)

    def scaled_features(self, x: torch.Tensor) -> torch.Tensor:
        """Features for Woodbury: ``Φ`` such that ``K = Φ Φ^T`` (no extra scale)."""
        return self.featurize(x)

    def train_features(self) -> torch.Tensor:
        """Features at training inputs ``(n, r)``, with eval-mode caching."""
        train_x = _drop_singleton_batch(self.train_inputs[0])
        key = self._feature_cache_key(train_x)
        if not self.training and self._train_z_cache is not None and self._train_z_cache_key == key:
            return self._train_z_cache
        z = self.scaled_features(train_x)
        if not self.training:
            self._train_z_cache = z
            self._train_z_cache_key = key
        return z

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        """Prior/posterior hook for GPyTorch; prefer Woodbury MLL and ``predict``."""
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input x must be a torch.Tensor.")
        mean = self.mean_module(x)
        z = self.scaled_features(x)
        covar = LowRankRootLinearOperator(z)
        return MultivariateNormal(mean, covar)

    def predict(
        self,
        test_x: torch.Tensor,
        jitter: float = 1e-6,
        return_latent: bool = False,
        variance_correction: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Posterior mean and ±2σ bands at ``test_x`` using Woodbury inference.

        When variance correction is enabled (default), applies DBK diagonal
        correction at prediction time (paper §3.3).
        """
        if variance_correction is None:
            variance_correction = self.variance_correction

        train_x = _drop_singleton_batch(self.train_inputs[0])
        train_y = _drop_singleton_batch(self.train_targets)
        z_train = self.train_features()
        z_test = self.scaled_features(test_x)
        mean_train = self.mean_module(train_x)
        if mean_train.dim() > 1 and mean_train.shape[0] == 1:
            mean_train = mean_train.squeeze(0)
        y_centered = train_y - mean_train
        noise = self.likelihood.noise

        f_mean, f_var, obs_std = woodbury_predict_lrnn(
            noise,
            z_train,
            z_test,
            y_centered,
            jitter=jitter,
            variance_correction=variance_correction,
        )
        f_mean = f_mean + self.mean_module(test_x)
        f_std = f_var.clamp_min(0.0).sqrt()

        if return_latent:
            return f_mean, f_mean - 2 * f_std, f_mean + 2 * f_std
        return f_mean, f_mean - 2 * obs_std, f_mean + 2 * obs_std

    def save(self, filepath: str = "lrnn_model_weights.pth") -> None:
        logger.info("Saving LRNNGPR state dict to %s", filepath)
        torch.save(self.state_dict(), filepath)

    def load(self, filepath: str = "lrnn_model_weights.pth") -> None:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model weights found at {filepath}")
        logger.info("Loading LRNNGPR state dict from %s", filepath)
        self.load_state_dict(torch.load(filepath))
        self.invalidate_feature_cache()
