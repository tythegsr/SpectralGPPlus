import torch
import gpytorch
import logging

logger = logging.getLogger(__name__)

def evaluate_gp_model(model, test_x: torch.Tensor):
    """
    Evaluate the Gaussian Process model on test data.

    Parameters:
        model (GPModel): The Gaussian Process model to evaluate.
        test_x (torch.Tensor): Test data features.

    Returns:
        mean (torch.Tensor): Predictive mean for each test point.
        lower (torch.Tensor): Lower confidence bound for each test point.
        upper (torch.Tensor): Upper confidence bound for each test point.
    """
    # Set the model to evaluation mode
    model.eval()
    #model.likelihood.eval()   # Uncomment if you are using likelihood explicitly

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Make predictions
        observed_pred = model.likelihood(model(test_x))

        # Get the mean, lower and upper confidence bounds
        mean = observed_pred.mean
        lower, upper = observed_pred.confidence_region()

        logger.info("Evaluation completed.")
        return mean, lower, upper

# Usage example (assuming you have the GP model instance `model` and test data `test_x`):
# mean, lower, upper = evaluate_gp_model(model, test_x)