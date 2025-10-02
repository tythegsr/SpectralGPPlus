import gpytorch
import torch
from gpytorch.likelihoods import GaussianLikelihood

from ..config import logger


class LearnablePriorsLikelihood(GaussianLikelihood):
    """
    Gaussian Likelihood with learnable noise priors.

    This likelihood extends the standard GaussianLikelihood by adding learnable
    hyperparameters for the noise prior distribution. The noise prior is defined
    as a Normal distribution with learnable mean and standard deviation parameters.

    Args:
        mean_prior_mean (float): Mean of the prior on the noise mean parameter. Default: 2.0
        mean_prior_std (float): Standard deviation of the prior on the noise mean parameter. Default: 30.0
        deviation_prior_mean (float): Mean of the prior on the noise deviation parameter. Default: 2.0
        deviation_prior_std (float): Standard deviation of the prior on the noise deviation parameter. Default: 10.0
        initial_noise_mean (float): Initial value for the learnable noise mean parameter. Default: 1.0
        initial_noise_deviation (float): Initial value for the learnable noise deviation parameter. Default: 1.0
        **kwargs: Additional arguments passed to GaussianLikelihood
    """

    def __init__(
        self,
        mean_prior_mean: float = 2.0,
        mean_prior_std: float = 30.0,
        deviation_prior_mean: float = 2.0,
        deviation_prior_std: float = 10.0,
        initial_noise_mean: float = 1.0,
        initial_noise_deviation: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Register learnable parameters for noise prior
        self.register_parameter(
            name="mean_noise", parameter=torch.nn.Parameter(torch.tensor(initial_noise_mean), requires_grad=True)
        )
        self.register_parameter(
            name="deviation_noise",
            parameter=torch.nn.Parameter(torch.tensor(initial_noise_deviation), requires_grad=True),
        )

        # Register priors on these hyperparameters (hyperpriors)
        self.register_prior(
            "prior_noise_mean", gpytorch.priors.NormalPrior(mean_prior_mean, mean_prior_std), "mean_noise"
        )
        self.register_prior(
            "prior_noise_deviation",
            gpytorch.priors.NormalPrior(deviation_prior_mean, deviation_prior_std),
            "deviation_noise",
        )

        # Define noise prior based on learnable parameters
        noise_prior = gpytorch.priors.NormalPrior(
            self.mean_noise.detach().exp().clone(), self.deviation_noise.detach().exp().clone()
        )

        # Attach this prior to the raw_noise parameter of the likelihood
        self.register_prior("noise_prior", noise_prior, "raw_noise")

        logger.debug(
            f"LearnablePriorsLikelihood initialized with mean_prior_mean={mean_prior_mean}, "
            f"mean_prior_std={mean_prior_std}, deviation_prior_mean={deviation_prior_mean}, "
            f"deviation_prior_std={deviation_prior_std}"
        )

    def update_noise_prior(self):
        """
        Update the noise prior based on current learnable parameter values.
        This should be called after the learnable parameters have been updated.
        """
        noise_prior = gpytorch.priors.NormalPrior(
            self.mean_noise.detach().exp().clone(), self.deviation_noise.detach().exp().clone()
        )

        # Update the registered prior
        self.register_prior("noise_prior", noise_prior, "raw_noise")

        logger.debug(
            f"Updated noise prior: mean={self.mean_noise.detach().exp().item():.4f}, "
            f"std={self.deviation_noise.detach().exp().item():.4f}"
        )

    def get_noise_prior_params(self):
        """
        Get the current noise prior parameters.

        Returns:
            tuple: (mean, std) of the current noise prior
        """
        return (self.mean_noise.detach().exp().item(), self.deviation_noise.detach().exp().item())
