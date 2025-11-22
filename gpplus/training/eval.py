import torch

from ..config import logger


def evaluate_gp_model(model, test_x: torch.Tensor, include_likelihood_noise: bool = True):
    """
    Evaluates the Gaussian Process model on test data.

    Args:
        model (GPModel):
            The Gaussian Process model to evaluate.
        test_x (torch.Tensor):
            Test data features.
        include_likelihood_noise (bool, optional):
            If True, uses model.likelihood() to include training noise in predictive variance.
            If False, uses model() directly to get latent function predictions without noise.
            Default: True (recommended for proper uncertainty quantification).

            Note: When evaluating with noisy test data, the model's predictive variance
            includes the TRAINING noise (learned from training data), but NOT any additional
            TEST noise. If you add noise to test targets, you should either:
            - Compare against clean (noise-free) test values, OR
            - Manually add the test noise variance to the predictive variance before computing metrics.

    Returns:
        tuple:
            A tuple containing:
                - **mean** (torch.Tensor): Predictive mean for each test point.
                - **lower** (torch.Tensor): Lower confidence bound for each test point.
                - **upper** (torch.Tensor): Upper confidence bound for each test point.
                - **stddev** (torch.Tensor): Standard deviation of the predictions.
    """
    # Set the model and likelihood to evaluation mode
    model.eval()

    with torch.no_grad():
        # observed_pred = model.likelihood(model(test_x))

        observed_pred = model(test_x)
        # Get the mean, lower and upper confidence bounds
        mean = observed_pred.mean
        lower, upper = observed_pred.confidence_region()
        stddev = observed_pred.stddev

        logger.info("Evaluation completed.")
        return mean, lower, upper, stddev
