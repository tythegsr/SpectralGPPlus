import torch

softplus = torch.nn.Softplus()


def inv_softplus(x):
    return x + torch.log(-torch.expm1(-x))


def log10_rbf_transform(x):
    """RBF-specific log10 transform with additional factors."""
    return 2.0 ** (-0.5) * torch.pow(10, -x / 2)


def log10_rbf_inv_transform(x):
    """RBF-specific log10 inverse transform with additional factors."""
    # Add small epsilon to avoid log10(0) which is -inf
    epsilon = 1e-8
    return -2.0 * torch.log10(x / 2.0 ** (-0.5) + epsilon)


def log10_transform(x):
    """Simple log10 transform: 10^x"""
    return torch.pow(10, x)


def log10_inv_transform(x):
    """Simple log10 inverse transform: log10(x)"""
    # Add small epsilon to avoid log10(0) which is -inf
    epsilon = 1e-8
    return torch.log10(x + epsilon)
