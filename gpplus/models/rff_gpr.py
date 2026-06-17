"""RFF Gaussian process regression with Woodbury inference."""

from __future__ import annotations

import os

import gpytorch
import torch
from gpytorch.distributions import MultivariateNormal
from linear_operator.operators import LowRankRootLinearOperator

from ..config import logger
from ..kernels import LogScaleKernel, RFFKernel
from ..likelihoods import LogGaussianLikelihood
from ..utils.rff_utils import woodbury_predict


def _drop_singleton_batch(t: torch.Tensor) -> torch.Tensor:
    """Woodbury helpers expect (n, m); GPyTorch may store train_x as (1, n, d)."""
    if t.dim() >= 3 and t.shape[0] == 1:
        return t.squeeze(0)
    return t


class RFFGPR(gpytorch.models.ExactGP):
    """
    GP regression with an RFF covariance and Woodbury training/inference.

    Use with :class:`~gpplus.training.rff_mll.RFFWoodburyMarginalLogLikelihood`
    in :class:`~gpplus.training.GPTrainer` instead of ``ExactMarginalLogLikelihood``.
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood | None = None,
        mean_module: gpytorch.means.Mean | None = None,
        kernel_module: gpytorch.kernels.Kernel | None = None,
        num_rff: int = 500,
        ard: bool = False,
        orthogonal: bool = False,
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
            input_dim = train_x.shape[-1]
            kernel_kwargs = {"ard_num_dims": input_dim} if ard else {}
            kernel_module = LogScaleKernel(
                RFFKernel(
                    num_samples=num_rff,
                    num_dims=input_dim,
                    orthogonal=orthogonal,
                    **kernel_kwargs,
                )
            )
            feature_kind = "ORF" if orthogonal else "RFF"
            logger.warning(
                "No kernel_module provided. Using LogScaleKernel(RFFKernel(...)) "
                f"({feature_kind}, num_rff={num_rff}, ard={ard}, input_dim={input_dim})."
            )

        if not isinstance(likelihood, gpytorch.likelihoods.Likelihood):
            raise TypeError("likelihood must be a gpytorch.likelihoods.Likelihood.")

        super().__init__(train_x, train_y, likelihood)
        self.num_rff = num_rff
        self.orthogonal = orthogonal
        self.mean_module = mean_module.to(dtype=self.dtype)
        self.covar_module = kernel_module.to(dtype=self.dtype)
        self.likelihood = self.likelihood.to(dtype=self.dtype)
        self._train_z_cache: torch.Tensor | None = None
        self._train_z_cache_key: tuple | None = None

    @property
    def _rff_kernel(self) -> RFFKernel:
        base = self.covar_module.base_kernel
        if not isinstance(base, RFFKernel):
            raise TypeError("covar_module.base_kernel must be RFFKernel.")
        return base

    def _output_scale(self) -> torch.Tensor:
        return torch.pow(10.0, self.covar_module.outputscale / 2.0)

    def _feature_cache_key(self, x: torch.Tensor) -> tuple:
        ls = self._rff_kernel.raw_lengthscale.detach().cpu().tolist()
        os_ = float(self.covar_module.raw_outputscale.detach().cpu())
        ver = getattr(self._rff_kernel, "_feature_cache_version", 0)
        return (id(x), tuple(ls), os_, ver)

    def invalidate_feature_cache(self) -> None:
        self._train_z_cache = None
        self._train_z_cache_key = None

    def unscaled_features(self, x: torch.Tensor) -> torch.Tensor:
        z = self._rff_kernel.featurize(x)
        return _drop_singleton_batch(z)

    def featurize(self, x: torch.Tensor) -> torch.Tensor:
        """Unscaled RFF features with output-scale applied (n, 2*num_rff)."""
        return self.unscaled_features(x) * self._output_scale()

    def scaled_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Features for Woodbury / low-rank covariance: kernel RFF features times output scale.

        Matches ``LogScaleKernel(RFFKernel)`` so that ``Z Z^T`` equals the kernel matrix.
        """
        return self.featurize(x)

    def train_features(self) -> torch.Tensor:
        """Scaled RFF features at training inputs (n, 2*num_rff), with eval-mode caching."""
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Posterior mean and std at test_x using Woodbury inference.

        Returns
        -------
        mean, lower, upper : tensors on test points (observation space includes noise if not latent).
        """
        train_x = self.train_inputs[0]
        train_y = self.train_targets
        z_train = self.train_features()
        z_test = self.scaled_features(test_x)
        mean_train = self.mean_module(train_x)
        y_centered = train_y - mean_train
        noise = self.likelihood.noise

        f_mean, f_var = woodbury_predict(noise, z_train, z_test, y_centered, jitter=jitter)
        f_mean = f_mean + self.mean_module(test_x)
        f_std = f_var.clamp_min(0.0).sqrt()

        if return_latent:
            return f_mean, f_mean - 2 * f_std, f_mean + 2 * f_std

        obs_var = f_var + noise
        obs_std = obs_var.clamp_min(0.0).sqrt()
        return f_mean, f_mean - 2 * obs_std, f_mean + 2 * obs_std

    def save(self, filepath: str = "rff_model_weights.pth") -> None:
        logger.info("Saving RFFGPR state dict to %s", filepath)
        torch.save(self.state_dict(), filepath)

    def load(self, filepath: str = "rff_model_weights.pth") -> None:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model weights found at {filepath}")
        logger.info("Loading RFFGPR state dict from %s", filepath)
        self.load_state_dict(torch.load(filepath))
        self.invalidate_feature_cache()
