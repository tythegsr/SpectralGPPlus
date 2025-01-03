from enum import Enum

import torch
from gpytorch.constraints import Positive

from gpytorch.kernels import Kernel, RBFKernel, MaternKernel

# Enum to define kernel types
class KernelType(Enum):
    RBFKernel = 'RBFKernel'
    MaternKernel = 'MaternKernel'

class KernelFactory():
    """
    Simple Factory pattern implementation to instantiate kernels.
    """
    @staticmethod
    def create_kernel(
        kernel_type: KernelType,
        ard_num_dims: int = None,
        active_dims = None,
        lengthscale_constraint = None,
    ) -> Kernel:
        """
        Function to instantiate a kernel based on the provided type.
        """
        if kernel_type == KernelType.RBFKernel:
            if lengthscale_constraint is None:
                lengthscale_constraint = Positive(transform = lambda x: 2.0**(-0.5) * torch.pow(10,-x/2), inv_transform= lambda x: -2.0*torch.log10(x/2.0))
            return RBFKernel(
                ard_num_dims = ard_num_dims,
                active_dims = active_dims,
                lengthscale_constraint=lengthscale_constraint
            )
        elif kernel_type == KernelType.MaternKernel:
            if lengthscale_constraint is None:
                lengthscale_constraint = Positive(transform = lambda x: 2.0**(-0.5) * torch.pow(10,-x/2), inv_transform= lambda x: -2.0*torch.log10(x/2.0))
            return MaternKernel(
                ard_num_dims = ard_num_dims,
                active_dims = active_dims,
                lengthscale_constraint = lengthscale_constraint
            )
        else:
            raise ValueError(f"Kernel type {kernel_type} provided is not supported.")