import os

import gpytorch
import torch
from torch import nn

from ..config import logger
from ..kernels import GaussianKernel, LogScaleKernel
from ..likelihoods import LogGaussianLikelihood
from ..utils.nigp_utils import input_noise_softclamp, raw_input_noise_init_value


class GPR(gpytorch.models.ExactGP):
    """Gaussian Process model for regression using GPyTorch.

    The GPR class encapsulates:
      - A mean module (defaults to ConstantMean if None).
      - A kernel module (defaults to a Scale Gaussian kernel if None).
      - A likelihood module (defaults to LogGaussianLikelihood if None).

    Attributes:
        mean_module (gpytorch.means.Mean): The mean function of the GP.
        covar_module (gpytorch.kernels.Kernel): The covariance (kernel) function.
        batch_shape (torch.Size): Leading init-batch shape (empty if unbatched).
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood = None,
        mean_module: gpytorch.means.Mean = None,
        kernel_module: gpytorch.kernels.Kernel = None,
        batch_shape: torch.Size | None = None,
        nigp: bool = False,
        input_noise_init: float | None = None,
    ):
        """Initializes GPR.

        Args:
            train_x (torch.Tensor): Training data features.
            train_y (torch.Tensor): Training data targets.
            likelihood (gpytorch.likelihoods.Likelihood, optional): The likelihood function. Defaults to \
                GaussianLikelihood if None.
            mean_module (gpytorch.means.Mean, optional): Mean function. Defaults to ConstantMean if None.
            kernel_module (gpytorch.kernels.Kernel, optional): Covariance kernel function.
                Defaults to a ScaleKernel * Gaussian combo if None.
            batch_shape: Optional leading init-batch shape, e.g. ``torch.Size([num_inits])``.
            nigp: If True, register learnable per-dimension input noise (classic NIGP).
            input_noise_init: Optional physical ``σ_x`` placeholder in ``(1e-6, 1)`` before
                the parameter initializer runs; ``None`` uses mid-range default.

        Raises:
            TypeError: If any of `train_x`, `train_y`, or `likelihood` are of incorrect types.
        """
        self.dtype = train_x.dtype
        self.batch_shape = torch.Size([]) if batch_shape is None else torch.Size(batch_shape)

        if not isinstance(train_x, torch.Tensor) or not isinstance(train_y, torch.Tensor):
            logger.error("train_x and train_y must be torch.Tensor instances.")
            raise TypeError("train_x and train_y must be torch.Tensor instances.")

        logger.debug(f"train_x shape: {train_x.shape}, train_y shape: {train_y.shape}")

        if likelihood is None:
            likelihood = LogGaussianLikelihood(batch_shape=self.batch_shape)
            logger.warning("No likelihood provided. Using LogGaussianLikelihood as default.")

        if mean_module is None:
            mean_module = gpytorch.means.ConstantMean(batch_shape=self.batch_shape)
            logger.warning("No mean_module provided. Using ConstantMean as default.")

        if kernel_module is None:
            input_dim = train_x.shape[-1]
            kernel_module = LogScaleKernel(
                GaussianKernel(ard_num_dims=input_dim, batch_shape=self.batch_shape),
                batch_shape=self.batch_shape,
            )
            logger.warning(
                "No kernel_module provided. Using LogScaleKernel(GaussianKernel(ard_num_dims=%s)) "
                f"(batch_shape={self.batch_shape}).",
                input_dim,
            )

        if not isinstance(likelihood, gpytorch.likelihoods.Likelihood):
            logger.error("likelihood must be an instance of gpytorch.likelihoods.Likelihood.")
            raise TypeError("likelihood must be an instance of gpytorch.likelihoods.Likelihood.")

        super().__init__(train_x, train_y, likelihood)

        self.mean_module = mean_module
        self.covar_module = kernel_module
        self.nigp = bool(nigp)
        # When False (freeze warm-start), MLL ignores σ_x (standard exact GP).
        self.nigp_correction_enabled = True

        # Ensure all components use the same dtype as the input data
        self.mean_module = self.mean_module.to(dtype=self.dtype)
        self.covar_module = self.covar_module.to(dtype=self.dtype)
        self.likelihood = self.likelihood.to(dtype=self.dtype)

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
                "NIGP enabled on GPR: learnable per-dim input noise "
                "(D=%s, sigma_x=10^SoftClamp(raw) in (1e-6, 1)).",
                input_dim,
            )

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

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """Runs the forward pass of the Gaussian Process model with ensembling
            if embedding or calibration is probabilistic.

        Args:
            x (torch.Tensor): Test data features for prediction.

        Returns:
            gpytorch.distributions.MultivariateNormal:
                Multivariate normal distribution containing
                the mean and covariance of the predictions.

        Raises:
            TypeError: If `x` is not a torch.Tensor.
        """
        if not isinstance(x, torch.Tensor):
            logger.error("Input x must be a torch.Tensor instance.")
            raise TypeError("Input x must be a torch.Tensor.")

        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def save(self, filepath: str = "model_weights.pth") -> None:
        """Saves this model's state dictionary to the specified file.

        Args:
            filepath (str, optional): Path to save the state dictionary file.
                Defaults to 'model_weights.pth' in the current directory.
        """
        logger.info(f"Saving model state dict to {filepath}")
        torch.save(self.state_dict(), filepath)

    def load(self, filepath: str = "model_weights.pth") -> None:
        """Loads this model's state dictionary from the specified file.

        Args:
            filepath (str, optional): Path to the file containing the saved state dict.
                Defaults to 'model_weights.pth' in the current directory.

        Raises:
            FileNotFoundError: If no file is found at `filepath`.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model weights found at {filepath}")

        logger.info(f"Loading model state dict from {filepath}")
        state_dict = torch.load(filepath)
        self.load_state_dict(state_dict)
