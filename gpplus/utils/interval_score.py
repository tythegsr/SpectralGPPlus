import torch

def compute_interval_score(
    y_true: torch.Tensor,
    lower_bound: torch.Tensor,
    upper_bound: torch.Tensor,
    confidence_level: float = 0.95
):
    """
    Compute the interval score:
       interval_width + penalty for points falling outside the interval.

    For reference, see Gneiting & Raftery (2007), "Strictly Proper Scoring
    Rules, Prediction, and Estimation."

    Args:
        y_true: Ground truth
        lower_bound: Predictive lower bound
        upper_bound: Predictive upper bound
        confidence_level: typically 0.95

    Returns:
        interval_score: torch.Tensor of shape matching y_true
    """
    alpha = 1.0 - confidence_level
    interval_width = upper_bound - lower_bound

    # penalty if actual y < lower_bound
    below_penalty = (2 / alpha) * (lower_bound - y_true) * (y_true < lower_bound)
    # penalty if actual y > upper_bound
    above_penalty = (2 / alpha) * (y_true - upper_bound) * (y_true > upper_bound)

    interval_score = interval_width + below_penalty + above_penalty
    return interval_score