# This will be used as a working file until a working EnsembleMean and EnsembleCovar
# has been implemented to work with the the original .gpr.py file.

import os

import gpytorch
import torch
from linear_operator.operators import DenseLinearOperator

from gpplus.utils.set_seed import set_seed

from ..config import logger


class GPR(gpytorch.models.ExactGP):
    """Gaussian Process model for regression using GPyTorch.

    The GPR class encapsulates:
      - A mean module (defaults to ConstantMean if None).
      - A kernel module (defaults to a Scale Gaussian kernel if None).
      - A likelihood module (defaults to GaussianLikelihood if None).

    Attributes:
        mean_module (gpytorch.means.Mean): The mean function of the GP.
        covar_module (gpytorch.kernels.Kernel): The covariance (kernel) function.
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        dtype: torch.float32 = None,
        device=None,
        likelihood: gpytorch.likelihoods.Likelihood = None,
        mean_module: gpytorch.means.Mean = None,
        kernel_module: gpytorch.kernels.Kernel = None,
        seed=None,
        learnable_priors=False,
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
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            learnable_priors (bool, optional): If True, registers learnable Normal priors to the likelihood. 
                Defaults to False.

        Raises:
            TypeError: If any of `train_x`, `train_y`, or `likelihood` are of incorrect types.
        """

        if likelihood is None:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            logger.warning("No likelihood provided. Using GaussianLikelihood as default.")

        if mean_module is None:
            mean_module = gpytorch.means.ConstantMean()
            logger.warning("No mean_module provided. Using ConstantMean as default.")

        if kernel_module is None:
            kernel_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            logger.warning("No kernel_module provided. Using Gaussian Kernel as default.")

        if not isinstance(train_x, torch.Tensor) or not isinstance(train_y, torch.Tensor):
            logger.error("train_x and train_y must be torch.Tensor instances.")
            raise TypeError("train_x and train_y must be torch.Tensor instances.")

        logger.debug(f"train_x shape: {train_x.shape}, train_y shape: {train_y.shape}")

        if seed is None:
            self.seed = torch.randint(0, 2**32 - 1, (1,)).item()
        else:
            self.seed = seed

        super().__init__(train_x, train_y, likelihood)

        if dtype is None:
            self.dtype = train_x.dtype
        else:
            self.dtype = dtype

        if learnable_priors:
            self.register_parameter(
                name="mean_noise", parameter=torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
            )
            self.register_parameter(
                name="deviation_noise", parameter=torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
            )

            # Register priors on these hyperparameters (hyperpriors)
            self.register_prior("prior_noise_mean", gpytorch.priors.NormalPrior(2.0, 30.0), "mean_noise")
            self.register_prior("prior_noise_deviation", gpytorch.priors.NormalPrior(2.0, 10.0), "deviation_noise")

            # Define noise prior based on learnable parameters
            noise_prior = gpytorch.priors.NormalPrior(
                self.mean_noise.detach().exp().clone(), self.deviation_noise.detach().exp().clone()
            )

            # Attach this prior to the raw_noise parameter of the likelihood
            self.likelihood.register_prior("noise_prior", noise_prior, "raw_noise")

        self.mean_module = mean_module
        self.covar_module = kernel_module

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """Runs the forward pass of the Gaussian Process model with ensembling if probabilistic.

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
        encoder = getattr(self.covar_module, "source_encoder", None)
        embedding_is_prob = getattr(encoder, "is_probabilistic", False)
        calibration_is_prob = (
            hasattr(self, "calibration_type") and getattr(self, "calibration_type", None) == "probabilistic"
        )
        if self.training:
            k = max(
                getattr(encoder, "num_passes", 1) if embedding_is_prob else 1,
                getattr(self, "num_calibration_passes", 1) if calibration_is_prob else 1,
            )
        else:
            k = max(
                getattr(encoder, "num_passes_pred", 1) if embedding_is_prob else 1,
                getattr(self, "num_calibration_passes", 1) if calibration_is_prob else 1,
            )
        if k > 1:
            set_seed(self.seed)
        mean_sum = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
        covar_sum = torch.zeros(x.size(0), x.size(0), dtype=x.dtype, device=x.device)
        # covar_sum = torch.zeros(x.size(0), x.size(0), device=x.device)
        # print(f"x input dtype = {x.dtype}")
        # print(f"covar_sum dtype = {covar_sum.dtype}")

        for _ in range(k):
            x_pass = x.clone()
            # print(f"x_pass input dtype = {x_pass.dtype}")

            # Always apply embedding (probabilistic or deterministic)
            if encoder is not None and hasattr(self.covar_module.source_encoder, "apply_embedding"):
                x_pass = self.covar_module.source_encoder.apply_embedding(x_pass).to(
                    self.dtype
                )  # IF THIS FAILS, COMMENT THIS OUT AND UNCOMMENT LINE BELOW
                # x_pass = self.covar_module.source_encoder.apply_embedding(x_pass)
            # Always apply calibration (probabilistic or deterministic)
            if hasattr(self, "apply_calibration"):
                x_pass = self.apply_calibration(x_pass).to(
                    self.dtype
                )  # IF THIS FAILS, COMMENT THIS OUT AND UNCOMMENT LINE BELOW
                # x_pass = self.apply_calibration(x_pass)
            # Compute mean and covariance as usual
            mean = self.mean_module(x_pass).to(self.dtype)
            covar = self.covar_module(x_pass).to(self.dtype)
            mean_sum += mean
            covar_sum += covar.to_dense() + torch.outer(mean, mean)
            # print(f"mean dtype = {mean.dtype}")
            # print(f"mean_sum dtype = {mean_sum.dtype}")
            # print(f"covar dtype = {covar.dtype}")
        ensemble_mean = mean_sum / k
        ensemble_covar = covar_sum / k
        ensemble_covar -= torch.outer(ensemble_mean, ensemble_mean)
        ensemble_covar = DenseLinearOperator(ensemble_covar)
        # print(f"mean_sum dtype = {mean_sum.dtype}")
        # print(f"covar_sum dtype = {covar_sum.dtype}")
        # print(f"ensemble_mean_sum dtype = {ensemble_mean.dtype}")
        # print(f"ensemble_covar_sum dtype = {ensemble_covar.dtype}")
        return gpytorch.distributions.MultivariateNormal(ensemble_mean, ensemble_covar)

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

    def initialize(self):
        """Randomly reinitialize all parameters in the model."""
        for name, param in self.named_parameters():
            if param.requires_grad:
                if "weight" in name:
                    torch.nn.init.xavier_uniform_(param)
                elif "bias" in name or "constant" in name or "raw_" in name:
                    torch.nn.init.normal_(param, mean=0.0, std=0.1)
