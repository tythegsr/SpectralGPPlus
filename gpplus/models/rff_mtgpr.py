"""Multitask RFF-GP with ICM kernel and Woodbury inference."""

from __future__ import annotations

import os

import gpytorch
import torch
from gpytorch.distributions import MultitaskMultivariateNormal

from ..config import logger
from ..kernels import LogScaleKernel, RFFKernel
from ..utils.rff_utils import (
    RffSampling,
    build_icm_joint_features,
    flatten_multitask_targets,
    task_psd_factor,
    woodbury_predict_mt,
)
from .rff_gpr import _drop_singleton_batch


class RFFMTGPR(gpytorch.models.ExactGP):
    """
    Multitask GP with RFF spatial kernel, ICM task covariance, and Woodbury inference.

    Use with :class:`~gpplus.training.rff_mt_mll.RFFMTWoodburyMarginalLogLikelihood`.
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        num_tasks: int | None = None,
        likelihood: gpytorch.likelihoods.Likelihood | None = None,
        mean_module: gpytorch.means.MultitaskMean | None = None,
        kernel_module: gpytorch.kernels.MultitaskKernel | None = None,
        num_rff: int = 500,
        ard: bool = False,
        rff_sampling: RffSampling = "rff",
        rank_kernel: int = 1,
        rank_likelihood: int = 0,
    ):
        if not isinstance(train_x, torch.Tensor) or not isinstance(train_y, torch.Tensor):
            raise TypeError("train_x and train_y must be torch.Tensor instances.")
        if train_y.dim() != 2:
            raise ValueError(f"train_y must be (n, T), got shape {tuple(train_y.shape)}.")

        self.num_tasks = num_tasks if num_tasks is not None else train_y.shape[-1]
        self.rank_kernel = rank_kernel
        self.rank_likelihood = rank_likelihood
        self.num_rff = num_rff
        self.rff_sampling = rff_sampling

        if likelihood is None:
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                num_tasks=self.num_tasks,
                rank=self.rank_likelihood,
            )
            logger.warning("No likelihood provided. Using MultitaskGaussianLikelihood.")
        if mean_module is None:
            base_mean = gpytorch.means.ConstantMean()
            mean_module = gpytorch.means.MultitaskMean(base_mean, self.num_tasks)
        if kernel_module is None:
            input_dim = train_x.shape[-1]
            kernel_kwargs = {"ard_num_dims": input_dim} if ard else {}
            base = LogScaleKernel(
                RFFKernel(
                    num_samples=num_rff,
                    num_dims=input_dim,
                    rff_sampling=rff_sampling,
                    **kernel_kwargs,
                )
            )
            kernel_module = gpytorch.kernels.MultitaskKernel(
                base,
                num_tasks=self.num_tasks,
                rank=self.rank_kernel,
            )
        elif not isinstance(kernel_module, gpytorch.kernels.MultitaskKernel):
            kernel_module = gpytorch.kernels.MultitaskKernel(
                kernel_module,
                num_tasks=self.num_tasks,
                rank=self.rank_kernel,
            )

        super().__init__(train_x, train_y, likelihood)
        self.dtype = train_x.dtype
        self.mean_module = mean_module.to(dtype=self.dtype)
        self.covar_module = kernel_module.to(dtype=self.dtype)
        self.likelihood = self.likelihood.to(dtype=self.dtype)
        self._train_omega_cache: torch.Tensor | None = None
        self._train_omega_cache_key: tuple | None = None

    @property
    def _rff_kernel(self) -> RFFKernel:
        base = self.covar_module.data_covar_module.base_kernel
        if not isinstance(base, RFFKernel):
            raise TypeError("data_covar_module.base_kernel must be RFFKernel.")
        return base

    def _output_scale(self) -> torch.Tensor:
        return torch.pow(10.0, self.covar_module.data_covar_module.outputscale / 2.0)

    def _feature_cache_key(self, x: torch.Tensor) -> tuple:
        ls = self._rff_kernel.raw_lengthscale.detach().cpu().tolist()
        os_ = float(self.covar_module.data_covar_module.raw_outputscale.detach().cpu())
        task_key = self.covar_module.task_covar_module.covar_matrix.to_dense().detach().cpu().tolist()
        ver = getattr(self._rff_kernel, "_feature_cache_version", 0)
        return (id(x), tuple(ls), os_, tuple(map(tuple, task_key)), ver)

    def invalidate_feature_cache(self) -> None:
        self._train_omega_cache = None
        self._train_omega_cache_key = None

    def scaled_spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        z = self._rff_kernel.featurize(x)
        z = _drop_singleton_batch(z)
        return z * self._output_scale()

    def task_psd_factor(self) -> torch.Tensor:
        return task_psd_factor(self.covar_module.task_covar_module.covar_matrix)

    def joint_features(self, x: torch.Tensor) -> torch.Tensor:
        phi = self.scaled_spatial_features(x)
        r_b = self.task_psd_factor()
        return build_icm_joint_features(phi, r_b)

    def train_joint_features(self) -> torch.Tensor:
        train_x = _drop_singleton_batch(self.train_inputs[0])
        key = self._feature_cache_key(train_x)
        if not self.training and self._train_omega_cache is not None and self._train_omega_cache_key == key:
            return self._train_omega_cache
        omega = self.joint_features(train_x)
        if not self.training:
            self._train_omega_cache = omega
            self._train_omega_cache_key = key
        return omega

    def task_noises(self) -> torch.Tensor:
        return self.likelihood.task_noises.clamp_min(1e-12)

    def forward(self, x: torch.Tensor) -> MultitaskMultivariateNormal:
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input x must be a torch.Tensor.")
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return MultitaskMultivariateNormal(mean, covar)

    def predict(
        self,
        test_x: torch.Tensor,
        jitter: float = 1e-6,
        return_latent: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        train_x = _drop_singleton_batch(self.train_inputs[0])
        train_y = _drop_singleton_batch(self.train_targets)
        n_train = train_x.shape[0]
        omega_train = self.train_joint_features()
        omega_test = self.joint_features(test_x)
        mean_train = self.mean_module(train_x)
        y_centered = flatten_multitask_targets(train_y - mean_train)
        task_noises = self.task_noises()

        f_mean, f_var = woodbury_predict_mt(
            task_noises,
            omega_train,
            omega_test,
            n_train,
            self.num_tasks,
            y_centered,
            jitter=jitter,
        )
        f_mean = f_mean + self.mean_module(test_x)
        f_std = f_var.clamp_min(0.0).sqrt()

        if return_latent:
            return f_mean, f_mean - 2 * f_std, f_mean + 2 * f_std

        noise_rows = task_noises.view(1, -1).expand(f_mean.shape[0], -1)
        obs_std = (f_var + noise_rows).clamp_min(0.0).sqrt()
        return f_mean, f_mean - 2 * obs_std, f_mean + 2 * obs_std

    def save(self, filepath: str = "rff_mt_model_weights.pth") -> None:
        logger.info("Saving RFFMTGPR state dict to %s", filepath)
        torch.save(self.state_dict(), filepath)

    def load(self, filepath: str = "rff_mt_model_weights.pth") -> None:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model weights found at {filepath}")
        logger.info("Loading RFFMTGPR state dict from %s", filepath)
        self.load_state_dict(torch.load(filepath))
        self.invalidate_feature_cache()
