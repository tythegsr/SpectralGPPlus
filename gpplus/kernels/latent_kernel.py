import torch
from gpytorch.kernels import RBFKernel, ScaleKernel


class LatentKernel(ScaleKernel):
    """Scaled RBF kernel with identity metric"""

    def __init__(self, z_dim=2):
        rbf = RBFKernel(ard_num_dims=z_dim)
        rbf.raw_lengthscale.requires_grad_(False)  # Fix lengthscales
        rbf.lengthscale = torch.ones(z_dim)  # Identity

        super().__init__(rbf)
        self.initialize(outputscale=1.0)  # Optional: fix output scale
