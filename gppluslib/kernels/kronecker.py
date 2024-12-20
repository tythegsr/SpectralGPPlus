from typing import List

import torch
import numpy as np
from linear_operator.operators import KroneckerProductLinearOperator
from gpytorch.kernels import Kernel

class KroneckerKernel(Kernel):
    """
    Computes the covariance matrix based on a Kronecker structure.

    :param ard_num_dims: Set this if you want a separate lengthscale for each input
        dimension. It should be `d` if :math:`\mathbf{x_1}` is a `n x d` matrix. (Default: `None`.)
    :param kernels: Ordered list of kernels to perform Kronecker product.
    :param column_indices: List with start and end indices for the input columns corresponding to each kernel features.
    """
    def __init__(
        self,
        ard_num_dims,
        kernels: List[Kernel],
        column_indices: List[List[int]],
        **kwargs
    ):
        
        super(KroneckerKernel, self).__init__(ard_num_dims=ard_num_dims, **kwargs)

        self._kernels = kernels
        self._xi = column_indices

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        """
        Compute covariance matrix.
        """
        if last_dim_is_batch:
            raise RuntimeError("KroneckerKernel does not accept the last_dim_is_batch argument.")
        covar = []
        for i, kernel in enumerate(self._kernels):
            unique_x1 = self._unique_rows_in_order(x1[:, :, self._xi[i]])
            unique_x2 = self._unique_rows_in_order(x2[:, :, self._xi[i]])
            kres = kernel(unique_x1, unique_x2)
            covar.append(kres)
        res = KroneckerProductLinearOperator(*covar)
        return res.diagonal(dim1=-1, dim2=-2) if diag else res

    def _unique_rows_in_order(self, a: torch.tensor) -> torch.tensor:
        a_np = np.array(a[0])
        _, unique_indices = np.unique(a_np, axis=0, return_index=True)
        unique_rows = a_np[np.sort(unique_indices)]
        res = torch.tensor(unique_rows).unsqueeze(0)
        return res
    
    def get_kernel(self, idx: int) -> Kernel:
        """
        Get Kronecker Kernel based on order index.
        """
        return self._kernels[idx]