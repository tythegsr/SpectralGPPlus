"""
Custom constraint classes for GPyTorch parameters.
"""

import torch
import gpytorch
from gpytorch.constraints import Interval


class CustomLog10Interval(Interval):
    """
    Custom interval constraint that uses log10 transforms.
    
    This constraint maps raw parameters in the range [lower_bound, upper_bound]
    to actual parameter values using 10^x transform.
    """
    
    def __init__(self, lower_bound=None, upper_bound=None, transform=None, inv_transform=None, initial_value=None):
        super().__init__(lower_bound, upper_bound, transform, inv_transform, initial_value)
        self._transform_func = transform
        self._inv_transform_func = inv_transform
    
    def transform(self, x):
        """Transform raw parameter to actual parameter using 10^x."""
        if self._transform_func is not None:
            return self._transform_func(x)
        else:
            # Default to 10^x if no transform function provided
            return torch.pow(10, x)
    
    def inv_transform(self, x):
        """Inverse transform actual parameter to raw parameter using log10(x)."""
        if self._inv_transform_func is not None:
            return self._inv_transform_func(x)
        else:
            # Default to log10(x) if no inverse transform function provided
            return torch.log10(x + 1e-8)  # Add small epsilon to avoid log10(0)


class CustomLog10RBFInterval(Interval):
    """
    Custom interval constraint that uses RBF-specific log10 transforms.
    
    This constraint maps raw parameters in the range [lower_bound, upper_bound]
    to actual parameter values using the RBF-specific transform:
    actual = 2^(-0.5) * 10^(-raw/2)
    """
    
    def __init__(self, lower_bound=None, upper_bound=None, transform=None, inv_transform=None, initial_value=None):
        super().__init__(lower_bound, upper_bound, transform, inv_transform, initial_value)
        self._transform_func = transform
        self._inv_transform_func = inv_transform
    
    def transform(self, x):
        """Transform raw parameter to actual parameter using RBF-specific transform."""
        if self._transform_func is not None:
            return self._transform_func(x)
        else:
            # Default RBF transform: 2^(-0.5) * 10^(-x/2)
            return 2.0**(-0.5) * torch.pow(10, -x / 2)
    
    def inv_transform(self, x):
        """Inverse transform actual parameter to raw parameter using RBF-specific inverse."""
        if self._inv_transform_func is not None:
            return self._inv_transform_func(x)
        else:
            # Default RBF inverse transform: -2 * log10(x / 2^(-0.5))
            epsilon = 1e-8
            return -2.0 * torch.log10(x / 2.0**(-0.5) + epsilon)
