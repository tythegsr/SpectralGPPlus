"""Multitask RFF-GP with ICM kernel and Woodbury inference."""

from __future__ import annotations

import os

import gpytorch
import torch
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.priors import Prior
from torch import nn

from ..config import logger
from ..kernels import LogIndexKernel, LogScaleKernel, RFFKernel
from ..priors.response_noise import align_registered_priors, build_multitask_noise_likelihood
from ..utils.nigp_utils import (
    effective_noise_variance_mt,
    input_noise_softclamp,
    nigp_correction_enabled,
    posterior_mean_grad_wrt_x_mt,
    raw_input_noise_init_value,
)
from ..utils.rff_utils import (
    RffSampling,
    SpectralKernel,
    WoodburyMtMethod,
    build_icm_joint_features,
    flatten_multitask_targets,
    task_psd_factor,
    woodbury_predict_mt,
    woodbury_predict_mt_diag_noise,
    woodbury_predictive_obs_std,
)
from .rff_gpr import _drop_singleton_batch


class RFFMTGPR(gpytorch.models.ExactGP):
    """
    Multitask GP with RFF spatial kernel, ICM task covariance, and Woodbury inference.

    Use with :class:`~gpplus.training.rff_mt_mll.RFFMTWoodburyMarginalLogLikelihood`
    or :class:`~gpplus.training.nigp_mt_mll.NIGPMTWoodburyMarginalLogLikelihood`
    when ``nigp=True``.
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
        correct_sorf: bool = False,
        spectral_kernel: SpectralKernel = "rbf",
        rank_kernel: int = 1,
        rank_likelihood: int = 0,
        noise_prior: Prior | None = None,
        outputscale_prior: Prior | None = None,
        lengthscale_prior: Prior | None = None,
        batch_shape: torch.Size | None = None,
        nigp: bool = False,
        input_noise_init: float | None = None,
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
        self.correct_sorf = bool(correct_sorf) if rff_sampling == "sorf" else False
        self.spectral_kernel = spectral_kernel
        self.batch_shape = torch.Size([]) if batch_shape is None else torch.Size(batch_shape)
        self.nigp = bool(nigp)

        if self.nigp and int(self.rank_likelihood) > 0:
            raise ValueError(
                "NIGP on RFFMTGPR requires rank_likelihood=0 (diagonal task noise)."
            )

        if likelihood is None:
            likelihood = build_multitask_noise_likelihood(
                self.num_tasks,
                noise_prior=noise_prior,
                rank=self.rank_likelihood,
                batch_shape=self.batch_shape,
            )
            logger.warning(
                "No likelihood provided. Using LogMultitaskGaussianLikelihood "
                "(per-task log10 SoftClamp noise, Woodbury-compatible)."
            )
        if mean_module is None:
            base_mean = gpytorch.means.ConstantMean(batch_shape=self.batch_shape)
            mean_module = gpytorch.means.MultitaskMean(base_mean, self.num_tasks)
        if kernel_module is None:
            input_dim = train_x.shape[-1]
            kernel_kwargs = {"ard_num_dims": input_dim, "batch_shape": self.batch_shape} if ard else {
                "batch_shape": self.batch_shape
            }
            if lengthscale_prior is not None:
                kernel_kwargs["lengthscale_prior"] = lengthscale_prior
            base = LogScaleKernel(
                RFFKernel(
                    num_samples=num_rff,
                    num_dims=input_dim,
                    rff_sampling=rff_sampling,
                    correct_sorf=self.correct_sorf,
                    spectral_kernel=spectral_kernel,
                    **kernel_kwargs,
                ),
                outputscale_prior=outputscale_prior,
                batch_shape=self.batch_shape,
            )
            feature_kind = rff_sampling.upper()
            logger.warning(
                "No kernel_module provided. Using MultitaskKernel(LogScaleKernel(RFFKernel(...))) "
                f"({feature_kind}, spectral_kernel={spectral_kernel}, num_rff={num_rff}, "
                f"ard={ard}, input_dim={input_dim}"
                + (f", correct_sorf={self.correct_sorf}" if rff_sampling == "sorf" else "")
                + f", batch_shape={self.batch_shape})."
            )
            kernel_module = gpytorch.kernels.MultitaskKernel(
                base,
                num_tasks=self.num_tasks,
                rank=self.rank_kernel,
            )
            kernel_module.task_covar_module = LogIndexKernel(
                num_tasks=self.num_tasks,
                rank=self.rank_kernel,
                batch_shape=self.batch_shape,
            )
        elif not isinstance(kernel_module, gpytorch.kernels.MultitaskKernel):
            kernel_module = gpytorch.kernels.MultitaskKernel(
                kernel_module,
                num_tasks=self.num_tasks,
                rank=self.rank_kernel,
            )
            kernel_module.task_covar_module = LogIndexKernel(
                num_tasks=self.num_tasks,
                rank=self.rank_kernel,
            )
        elif not isinstance(kernel_module.task_covar_module, LogIndexKernel):
            kernel_module.task_covar_module = LogIndexKernel(
                num_tasks=self.num_tasks,
                rank=self.rank_kernel,
            )

        super().__init__(train_x, train_y, likelihood)
        self.dtype = train_x.dtype
        self.mean_module = mean_module.to(dtype=self.dtype)
        self.covar_module = kernel_module.to(dtype=self.dtype)
        self.likelihood = self.likelihood.to(dtype=self.dtype)
        align_registered_priors(self)
        self._train_phi_cache: torch.Tensor | None = None
        self._train_phi_cache_key: tuple | None = None
        # When False (freeze warm-start), MLL/predict ignore σ_x (standard MT GP).
        self.nigp_correction_enabled = True

        if self.nigp:
            input_dim = int(train_x.shape[-1])
            raw_init = raw_input_noise_init_value(input_noise_init, dtype=self.dtype)
            if len(self.batch_shape) > 0:
                raw_init = raw_init.expand(*self.batch_shape, input_dim).clone()
            else:
                raw_init = raw_init.expand(input_dim).clone()
            self.register_parameter("raw_input_noise", nn.Parameter(raw_init))
            self.register_constraint(
                "raw_input_noise",
                input_noise_softclamp(dtype=self.dtype),
            )
            logger.info(
                "NIGP enabled: learnable per-dim input noise "
                "(D=%s, sigma_x=10^SoftClamp(raw) in (1e-6, 1)).",
                input_dim,
            )

    def to(self, *args, **kwargs):
        out = super().to(*args, **kwargs)
        align_registered_priors(self)
        return out

    @property
    def _rff_kernel(self) -> RFFKernel:
        base = self.covar_module.data_covar_module.base_kernel
        if not isinstance(base, RFFKernel):
            raise TypeError("data_covar_module.base_kernel must be RFFKernel.")
        return base

    def _output_scale(self) -> torch.Tensor:
        return torch.pow(10.0, self.covar_module.data_covar_module.outputscale / 2.0)

    def _feature_cache_key(self, x: torch.Tensor) -> tuple:
        ls = self._rff_kernel.raw_lengthscale.detach().reshape(-1).cpu().tolist()
        os_ = self.covar_module.data_covar_module.raw_outputscale.detach().reshape(-1).cpu().tolist()
        task_key = (
            self.covar_module.task_covar_module.covar_matrix.to_dense().detach().reshape(-1).cpu().tolist()
        )
        ver = getattr(self._rff_kernel, "_feature_cache_version", 0)
        return (id(x), tuple(ls), tuple(os_), tuple(task_key), ver)

    def invalidate_feature_cache(self) -> None:
        self._train_phi_cache = None
        self._train_phi_cache_key = None

    @property
    def input_noise(self) -> torch.Tensor:
        """Per-dimension input noise std ``σ_x = 10^{SoftClamp(raw)}`` (``nigp=True``)."""
        if not getattr(self, "nigp", False) or not hasattr(self, "raw_input_noise"):
            raise AttributeError("Model has no NIGP input_noise (construct with nigp=True).")
        return torch.pow(
            10.0, self.raw_input_noise_constraint.transform(self.raw_input_noise)
        )

    @property
    def input_noise_var(self) -> torch.Tensor:
        """Per-dimension input noise variance ``σ_x²``."""
        s = self.input_noise
        return s * s

    def scaled_spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        z = self._rff_kernel.featurize(x)
        z = _drop_singleton_batch(z)
        scale = self._output_scale()
        while scale.dim() < z.dim():
            scale = scale.unsqueeze(-1)
        return z * scale

    def train_spatial_features(self) -> torch.Tensor:
        """Cached train Φ for eval-mode Woodbury (never materializes Omega).

        In train mode, hypers change every step so caching is skipped — and the
        cache key is not built (avoids per-step CPU sync / ``to_dense`` of ``B``).
        """
        train_x = _drop_singleton_batch(self.train_inputs[0])
        if self.training:
            return self.scaled_spatial_features(train_x)
        key = self._feature_cache_key(train_x)
        if self._train_phi_cache is not None and self._train_phi_cache_key == key:
            return self._train_phi_cache
        phi = self.scaled_spatial_features(train_x)
        self._train_phi_cache = phi
        self._train_phi_cache_key = key
        return phi

    def task_psd_factor(self) -> torch.Tensor:
        return task_psd_factor(self.covar_module.task_covar_module.covar_matrix)

    def joint_features(self, x: torch.Tensor) -> torch.Tensor:
        """Materialize Omega = Phi kron R_B (debug / tests only)."""
        phi = self.scaled_spatial_features(x)
        r_b = self.task_psd_factor()
        return build_icm_joint_features(phi, r_b)

    def train_joint_features(self) -> torch.Tensor:
        """Materialize train Omega (debug / tests only)."""
        return self.joint_features(_drop_singleton_batch(self.train_inputs[0]))

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
        method: WoodburyMtMethod = "eigen",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        train_x = _drop_singleton_batch(self.train_inputs[0])
        train_y = _drop_singleton_batch(self.train_targets)
        n_train = train_x.shape[0]
        phi_train = self.train_spatial_features()
        phi_test = self.scaled_spatial_features(test_x)
        r_b = self.task_psd_factor()
        mean_train = self.mean_module(train_x)
        y_centered = flatten_multitask_targets(train_y - mean_train)
        task_noises = self.task_noises()

        if nigp_correction_enabled(self):
            from ..utils.nigp_utils import _posterior_feature_weights_homoskedastic_mt

            homo_v = _posterior_feature_weights_homoskedastic_mt(
                task_noises.detach(),
                phi_train,
                r_b.detach(),
                n_train,
                y_centered.detach(),
                jitter=jitter,
            )
            with torch.enable_grad():
                grad_mu_train = posterior_mean_grad_wrt_x_mt(
                    self,
                    train_x,
                    y_centered,
                    task_noises,
                    jitter=jitter,
                    feature_weights=homo_v,
                )
            d_train = effective_noise_variance_mt(
                task_noises, self.input_noise_var, grad_mu_train
            )
            with torch.enable_grad():
                grad_mu_test = posterior_mean_grad_wrt_x_mt(
                    self,
                    test_x,
                    y_centered,
                    task_noises,
                    train_x=train_x,
                    jitter=jitter,
                    feature_weights=homo_v,
                )
            d_test = effective_noise_variance_mt(
                task_noises, self.input_noise_var, grad_mu_test
            )
            f_mean, f_var, obs_std = woodbury_predict_mt_diag_noise(
                d_train,
                phi_train,
                phi_test,
                r_b,
                n_train,
                self.num_tasks,
                y_centered,
                jitter=jitter,
                d_test=d_test,
            )
            f_mean = f_mean + self.mean_module(test_x)
            out_dtype = test_x.dtype
            if f_mean.dtype != out_dtype:
                f_mean = f_mean.to(out_dtype)
                f_var = f_var.to(out_dtype)
                obs_std = obs_std.to(out_dtype)
            if return_latent:
                f_std = f_var.clamp_min(0.0).sqrt()
                return f_mean, f_mean - 2 * f_std, f_mean + 2 * f_std
            return f_mean, f_mean - 2 * obs_std, f_mean + 2 * obs_std

        f_mean, f_var = woodbury_predict_mt(
            task_noises,
            phi_train,
            phi_test,
            r_b,
            n_train,
            self.num_tasks,
            y_centered,
            jitter=jitter,
            method=method,
        )
        f_mean = f_mean + self.mean_module(test_x)
        out_dtype = test_x.dtype
        if f_mean.dtype != out_dtype:
            f_mean = f_mean.to(out_dtype)
            f_var = f_var.to(out_dtype)
        f_std = f_var.clamp_min(0.0).sqrt()

        if return_latent:
            return f_mean, f_mean - 2 * f_std, f_mean + 2 * f_std

        noise_rows = task_noises.view(1, -1).expand(f_mean.shape[0], -1)
        obs_std = woodbury_predictive_obs_std(f_var, noise_rows)
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
