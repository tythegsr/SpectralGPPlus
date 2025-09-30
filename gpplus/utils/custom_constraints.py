"""
Custom constraint classes for GPyTorch parameters.
"""

import torch
from gpytorch.constraints import Interval


class Log10Interval(Interval):
    """
    Custom interval constraint that uses log10 transforms.

    This constraint maps raw parameters in the range [lower_bound, upper_bound]
    to actual parameter values using 10^x transform.
    The bounds are enforced by clamping the raw parameter values.
    """

    def __init__(self, lower_bound=None, upper_bound=None, initial_value=None):
        # Pass None for transform/inv_transform to avoid the default sigmoid behavior
        super().__init__(lower_bound, upper_bound, transform=None, inv_transform=None, initial_value=initial_value)

    def transform(self, x):
        """Transform raw parameter to actual parameter using 10^x."""
        # Clamp raw values to bounds first
        clamped_x = torch.clamp(x, self.lower_bound, self.upper_bound)
        return torch.pow(10, clamped_x)

    def inverse_transform(self, y):
        """Inverse transform actual parameter to raw parameter using log10(x)."""
        # Add small epsilon to avoid log10(0)
        raw = torch.log10(y + 1e-8)
        # Clamp to bounds
        return torch.clamp(raw, self.lower_bound, self.upper_bound)


class Log10RBFInterval(Interval):
    """
    Custom interval constraint that uses RBF-specific log10 transforms.

    This constraint maps raw parameters in the range [lower_bound, upper_bound]
    to actual parameter values using the RBF-specific transform:
    actual = 2^(-0.5) * 10^(-raw/2)
    The bounds are enforced by clamping the raw parameter values.
    """

    def __init__(self, lower_bound=None, upper_bound=None, initial_value=None):
        # Pass None for transform/inv_transform to avoid the default sigmoid behavior
        super().__init__(lower_bound, upper_bound, transform=None, inv_transform=None, initial_value=initial_value)

    def transform(self, x):
        """Transform raw parameter to actual parameter using RBF-specific transform."""
        # Clamp raw values to bounds first
        clamped_x = torch.clamp(x, self.lower_bound, self.upper_bound)
        # RBF transform: 2^(-0.5) * 10^(-x/2)
        return 2.0 ** (-0.5) * torch.pow(10, -clamped_x / 2)

    def inverse_transform(self, y):
        """Inverse transform actual parameter to raw parameter using RBF-specific inverse."""
        # RBF inverse transform: -2 * log10(y / 2^(-0.5))
        epsilon = 1e-8
        raw = -2.0 * torch.log10(y / 2.0 ** (-0.5) + epsilon)
        # Clamp to bounds
        return torch.clamp(raw, self.lower_bound, self.upper_bound)