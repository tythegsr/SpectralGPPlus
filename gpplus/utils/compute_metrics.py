import torch
import numpy as np
from torch.quasirandom import SobolEngine

# Compute MSE properly
def compute_mse(pred_mean, y_true):
    return torch.mean((pred_mean - y_true) ** 2)

def compute_mae(pred_mean, y_true):
    return torch.mean(torch.abs(pred_mean - y_true))

# Compute NRMSE properly
def compute_nrmse(pred_mean, y_true, y_std=None):
    """
        Compute Normalized Root Mean Squared Error (NRMSE)
        
        Args:
            pred_mean: predicted values (tensor)
            y_true: ground truth values (tensor)
            y_std: standard deviation for normalization (scalar or None)
            
        Returns:
            NRMSE as a Python float
        """
    mse = torch.mean((pred_mean - y_true) ** 2)
    rmse = torch.sqrt(mse).item()  # Convert to Python float
    
    if y_std is None:
        # Use range normalization if no std provided
        y_range = y_true.max() - y_true.min()
        return rmse / (y_range.item() + 1e-6)
    else:
        # Use provided standard deviation
        return rmse / (y_std.item() if torch.is_tensor(y_std) else y_std + 1e-6)
    
# Compute RRMSE properly
def compute_rrmse(y_true, pred_mean, y_std=None):
    return torch.sqrt(torch.mean((y_true - pred_mean) ** 2)) / (y_std + 1e-6)

def compute_nis(y_true, pred_lower, pred_upper, y_std=None, confidence_level=0.95):
    """
    Compute Normalized Interval Score (NIS)
    
    Args:
        y_true: Ground truth values (tensor)
        pred_lower: Lower prediction bounds (tensor)
        pred_upper: Upper prediction bounds (tensor)
        y_std: Standard deviation for normalization (scalar or None)
        confidence_level: Confidence level for interval (default: 0.95)
        
    Returns:
        NIS as a Python float
    """
    # Compute raw interval score
    alpha = 1 - confidence_level
    interval_score = (
        (pred_upper - pred_lower) + 
        (2/alpha) * torch.maximum(
            torch.zeros_like(y_true),
            torch.maximum(
                y_true - pred_upper,
                pred_lower - y_true
            )
        )
    )
    
    # Compute mean interval score
    mis = torch.mean(interval_score).item()
    
    # Normalize
    if y_std is None:
        # Use range normalization if no std provided
        y_range = y_true.max() - y_true.min()
        return mis / (y_range.item() + 1e-6)
    else:
        # Use provided standard deviation
        return mis / (y_std.item() if torch.is_tensor(y_std) else y_std + 1e-6)