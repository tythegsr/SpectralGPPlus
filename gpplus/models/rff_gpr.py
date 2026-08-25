"""RFF Gaussian process regression with Woodbury inference."""

from __future__ import annotations

import os

import gpytorch
import torch
import torch.nn.functional as F
from gpytorch.distributions import MultivariateNormal
from torch import nn
from linear_operator.operators import LowRankRootLinearOperator

from ..config import logger
from ..kernels import LogScaleKernel, RFFKernel
from ..likelihoods import LogGaussianLikelihood
from ..priors.response_noise import align_registered_priors
from ..utils.line_profile import profile
from ..utils.rff_utils import (
    RffSampling,
    SpectralKernel,
    woodbury_predict,
    woodbury_predict_diag_noise,
    woodbury_predictive_obs_std,
)
from ..utils.nigp_utils import (
    effective_noise_variance,
    input_noise_softclamp,
    nigp_correction_enabled,
    posterior_mean_grad_wrt_x,
    raw_input_noise_init_value,
)
from ..utils.transforms import inv_softplus


def _drop_singleton_batch(t: torch.Tensor) -> torch.Tensor:
    """Woodbury helpers expect (n, ...) data layout; GPyTorch may store train_x as (1, n, d).

    Does **not** squeeze a real init-batch dimension (leading size > 1).
    """
    if t.dim() >= 3 and t.shape[0] == 1:
        return t.squeeze(0)
    return t


class RFFGPR(gpytorch.models.ExactGP):
    """
    GP regression with an RFF covariance and Woodbury training/inference.

    Use with :class:`~gpplus.training.rff_mll.RFFWoodburyMarginalLogLikelihood`
    in :class:`~gpplus.training.GPTrainer` instead of ``ExactMarginalLogLikelihood``.

    Pass ``batch_shape=torch.Size([init_batch_size])`` for batched multi-init training.
    ``init_batch_size`` is the concurrent wave size; total random starts are set on
    :class:`~gpplus.training.GPTrainer` via ``num_inits`` (must be a multiple of
    ``init_batch_size``).
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
        rff_sampling: RffSampling = "rff",
        correct_sorf: bool = False,
        spectral_kernel: SpectralKernel = "rbf",
        batch_shape: torch.Size | None = None,
        init_batch_size: int | None = None,
        nigp: bool = False,
        input_noise_init: float | None = None,
    ):
        if not isinstance(train_x, torch.Tensor) or not isinstance(train_y, torch.Tensor):
            raise TypeError("train_x and train_y must be torch.Tensor instances.")

        self.dtype = train_x.dtype
        self.batch_shape = torch.Size([]) if batch_shape is None else torch.Size(batch_shape)
        if init_batch_size is None:
            init_batch_size = int(self.batch_shape[0]) if len(self.batch_shape) > 0 else 1
        if int(init_batch_size) < 1:
            raise ValueError(f"init_batch_size must be >= 1, got {init_batch_size}.")
        self.init_batch_size = int(init_batch_size)
        if likelihood is None:
            likelihood = LogGaussianLikelihood(batch_shape=self.batch_shape)
            logger.warning("No likelihood provided. Using LogGaussianLikelihood.")
        elif len(self.batch_shape) > 0:
            raw = getattr(likelihood, "raw_noise", None)
            if raw is None and hasattr(likelihood, "noise_covar"):
                raw = getattr(likelihood.noise_covar, "raw_noise", None)
            if torch.is_tensor(raw) and (
                raw.dim() < 1 or int(raw.shape[0]) != int(self.batch_shape[0])
            ):
                raise ValueError(
                    f"RFFGPR batch_shape={self.batch_shape} requires a likelihood whose "
                    f"trainable noise has leading dim {int(self.batch_shape[0])}; got "
                    f"raw_noise shape {tuple(raw.shape)}. Pass batch_shape into "
                    f"LogGaussianLikelihood / build_rff_scalar_noise_likelihood."
                )
        if mean_module is None:
            mean_module = gpytorch.means.ConstantMean(batch_shape=self.batch_shape)
        if kernel_module is None:
            input_dim = train_x.shape[-1]
            kernel_kwargs = {"ard_num_dims": input_dim, "batch_shape": self.batch_shape} if ard else {
                "batch_shape": self.batch_shape
            }
            kernel_module = LogScaleKernel(
                RFFKernel(
                    num_samples=num_rff,
                    num_dims=input_dim,
                    rff_sampling=rff_sampling,
                    correct_sorf=correct_sorf,
                    spectral_kernel=spectral_kernel,
                    **kernel_kwargs,
                ),
                batch_shape=self.batch_shape,
            )
            feature_kind = rff_sampling.upper()
            logger.warning(
                "No kernel_module provided. Using LogScaleKernel(RFFKernel(...)) "
                f"({feature_kind}, spectral_kernel={spectral_kernel}, num_rff={num_rff}, "
                f"ard={ard}, input_dim={input_dim}"
                + (f", correct_sorf={correct_sorf}" if rff_sampling == "sorf" else "")
                + f", batch_shape={self.batch_shape})."
            )

        if not isinstance(likelihood, gpytorch.likelihoods.Likelihood):
            raise TypeError("likelihood must be a gpytorch.likelihoods.Likelihood.")

        super().__init__(train_x, train_y, likelihood)
        self.num_rff = num_rff
        self.rff_sampling = rff_sampling
        self.correct_sorf = bool(correct_sorf)
        self.spectral_kernel = spectral_kernel
        self.nigp = bool(nigp)
        self.mean_module = mean_module.to(dtype=self.dtype)
        self.covar_module = kernel_module.to(dtype=self.dtype)
        self.likelihood = self.likelihood.to(dtype=self.dtype)
        align_registered_priors(self)
        self._train_z_cache: torch.Tensor | None = None
        self._train_z_cache_key: tuple | None = None
        # When False (freeze warm-start), MLL/predict ignore σ_x (standard GP).
        self.nigp_correction_enabled = True

        if self.nigp:
            input_dim = int(train_x.shape[-1])
            # Temporary placeholder; DefaultParameterInitializer Uniform(-4, -1) overwrites.
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
        base = self.covar_module.base_kernel
        if not isinstance(base, RFFKernel):
            raise TypeError("covar_module.base_kernel must be RFFKernel.")
        return base

    def _output_scale(self) -> torch.Tensor:
        # ( ) or (B,) -> broadcast against features (..., n, m)
        scale = torch.pow(10.0, self.covar_module.outputscale / 2.0)
        return scale

    def _feature_cache_key(self, x: torch.Tensor) -> tuple:
        ls = self._rff_kernel.raw_lengthscale.detach().reshape(-1).cpu().tolist()
        os_ = self.covar_module.raw_outputscale.detach().reshape(-1).cpu().tolist()
        ver = getattr(self._rff_kernel, "_feature_cache_version", 0)
        return (id(x), tuple(ls), tuple(os_), ver)

    def invalidate_feature_cache(self) -> None:
        self._train_z_cache = None
        self._train_z_cache_key = None

    def has_learnable_bound_penalty_lambda(self) -> bool:
        return hasattr(self, "raw_bound_penalty_lambda")

    def register_learnable_bound_penalty_lambda(
        self,
        *,
        lam_init: float,
        lam_min: float = 1.0,
    ) -> None:
        """
        Register a scalar bound-penalty weight ``lambda_eff = lam_min + softplus(raw)``.

        ``lam_init`` sets the initial effective weight; ``lam_min`` prevents collapse to zero.
        """
        if lam_min < 0:
            raise ValueError(f"lam_min must be >= 0, got {lam_min}.")
        if lam_init < lam_min:
            raise ValueError(
                f"lam_init must be >= lam_min for learnable bound penalty, "
                f"got lam_init={lam_init}, lam_min={lam_min}."
            )
        if self.has_learnable_bound_penalty_lambda():
            raise RuntimeError("Learnable bound penalty lambda already registered on this model.")

        delta = max(float(lam_init) - float(lam_min), 1e-8)
        raw_init = inv_softplus(torch.tensor(delta, dtype=self.dtype))
        self.register_buffer(
            "bound_penalty_lam_min",
            torch.tensor(float(lam_min), dtype=self.dtype),
        )
        self.register_parameter("raw_bound_penalty_lambda", nn.Parameter(raw_init.clone()))

    @property
    def bound_penalty_lambda(self) -> torch.Tensor:
        if not self.has_learnable_bound_penalty_lambda():
            raise AttributeError("Model has no learnable bound_penalty_lambda parameter.")
        lam_min = self.bound_penalty_lam_min.to(
            device=self.raw_bound_penalty_lambda.device,
            dtype=self.raw_bound_penalty_lambda.dtype,
        )
        return lam_min + F.softplus(self.raw_bound_penalty_lambda)

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

    def unscaled_features(self, x: torch.Tensor) -> torch.Tensor:
        z = self._rff_kernel.featurize(x)
        return _drop_singleton_batch(z)

    @profile
    def featurize(self, x: torch.Tensor) -> torch.Tensor:
        """Unscaled RFF features with output-scale applied (..., n, 2*num_rff)."""
        z = self.unscaled_features(x)
        scale = self._output_scale()
        while scale.dim() < z.dim():
            scale = scale.unsqueeze(-1)
        return z * scale

    @profile
    def scaled_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Features for Woodbury / low-rank covariance: kernel RFF features times output scale.

        Matches ``LogScaleKernel(RFFKernel)`` so that ``Z Z^T`` equals the kernel matrix.
        """
        return self.featurize(x)

    def train_features(self) -> torch.Tensor:
        """Scaled RFF features at training inputs (..., n, 2*num_rff), with eval-mode caching."""
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

        With ``nigp=True``, training uses diag-noise Woodbury and observation
        intervals include the NIGP correction ``(∇μ_*)ᵀ Σ_x (∇μ_*)``.

        Returns
        -------
        mean, lower, upper : tensors on test points (observation space includes noise if not latent).
        """
        train_x = _drop_singleton_batch(self.train_inputs[0])
        train_y = _drop_singleton_batch(self.train_targets)
        z_train = self.train_features()
        z_test = self.scaled_features(test_x)
        mean_train = self.mean_module(train_x)
        if mean_train.dim() > 1 and mean_train.shape[0] == 1 and z_train.dim() == 2:
            mean_train = mean_train.squeeze(0)
        y_centered = train_y - mean_train
        noise = self.likelihood.noise
        if noise.dim() > 1:
            noise = noise.reshape(noise.shape[0])

        if nigp_correction_enabled(self):
            with torch.enable_grad():
                grad_mu_train = posterior_mean_grad_wrt_x(
                    self, train_x, y_centered, noise, jitter=jitter
                )
            d_train = effective_noise_variance(
                noise, self.input_noise_var, grad_mu_train
            )
            with torch.enable_grad():
                grad_mu_test = posterior_mean_grad_wrt_x(
                    self,
                    test_x,
                    y_centered,
                    noise,
                    train_x=train_x,
                    jitter=jitter,
                )
            d_test = effective_noise_variance(
                noise, self.input_noise_var, grad_mu_test
            )
            f_mean, f_var, obs_std = woodbury_predict_diag_noise(
                d_train,
                z_train,
                z_test,
                y_centered,
                jitter=jitter,
                d_test=d_test,
            )
            f_mean = f_mean + self.mean_module(test_x)
            if return_latent:
                f_std = f_var.clamp_min(0.0).sqrt()
                return f_mean, f_mean - 2 * f_std, f_mean + 2 * f_std
            return f_mean, f_mean - 2 * obs_std, f_mean + 2 * obs_std

        f_mean, f_var = woodbury_predict(noise, z_train, z_test, y_centered, jitter=jitter)
        f_mean = f_mean + self.mean_module(test_x)
        f_std = f_var.clamp_min(0.0).sqrt()

        if return_latent:
            return f_mean, f_mean - 2 * f_std, f_mean + 2 * f_std

        obs_std = woodbury_predictive_obs_std(f_var, noise)
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
