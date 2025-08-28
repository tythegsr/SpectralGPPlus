import torch

softplus = torch.nn.Softplus()


def inv_softplus(x):
    return x + torch.log(-torch.expm1(-x))

def log10_transform(x):
    return 2.0**(-0.5) * torch.pow(10, -x / 2)

def log10_inv_transform(l):
    return -2.0 * torch.log10(l / 2.0**(-0.5))
