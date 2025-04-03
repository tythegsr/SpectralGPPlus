from typing import List

import torch
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import Likelihood
from gpytorch.means import Mean

from gpplus.kernels.kronecker import KroneckerKernel
from gpplus.models.gpr import GPR


class KroneckerMOGP(GPR):
    """
    Multi output Kronecker Gaussian Process
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: Likelihood,
        mean_module: Mean,
        kernels: List[Kernel],
        column_indices: List[List[int]],
        **kwargs,
    ) -> None:
        # Kernel
        kernel_module = KroneckerKernel(
            ard_num_dims=sum(len(lst) for lst in column_indices),
            kernels=kernels,
            column_indices=column_indices,
        )

        # Parent class initialization
        GPR.__init__(self, train_x, train_y, likelihood, mean_module, kernel_module, **kwargs)
