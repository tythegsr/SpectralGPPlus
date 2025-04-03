import logging

import torch

logger = logging.getLogger(__name__)


def evaluate_gp_model(model, test_x: torch.Tensor):
    """
    Evaluates the Gaussian Process model on test data.

    Args:
        model (GPModel):
            The Gaussian Process model to evaluate.
        test_x (torch.Tensor):
            Test data features.

    Returns:
        tuple:
            A tuple containing:
                - **mean** (torch.Tensor): Predictive mean for each test point.
                - **lower** (torch.Tensor): Lower confidence bound for each test point.
                - **upper** (torch.Tensor): Upper confidence bound for each test point.
                - **stddev** (torch.Tensor): Standard deviation of the predictions.
    """
    # Set the model to evaluation mode
    model.eval()
    # model.likelihood.eval()   # Uncomment if you are using likelihood explicitly

    with torch.no_grad():  # gpytorch.settings.fast_pred_var():
        # Make predictions
        observed_pred = model(test_x)  # observed_pred = model.likelihood(model(test_x))

        # Get the mean, lower and upper confidence bounds
        mean = observed_pred.mean
        lower, upper = observed_pred.confidence_region()
        stddev = observed_pred.stddev

        logger.info("Evaluation completed.")
        return mean, lower, upper, stddev
