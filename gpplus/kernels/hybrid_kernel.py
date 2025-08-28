import gpytorch
import torch
from gpytorch.kernels import ScaleKernel, RBFKernel, AdditiveKernel, LinearKernel, ProductKernel


class HybridKernel(gpytorch.kernels.Kernel):
    """
    A hybrid kernel that combines linear and RBF kernels.
    
    This kernel is designed for use with matrix encoders (projection matrices)
    to provide both interpretability (linear relationships) and flexibility (RBF).
    
    Args:
        z_dim (int): Dimension of the latent space
        combination_method (str): Either "product" or "additive" to specify how to combine kernels
        linear_weight (float): Weight for the linear kernel component (default: 1.0)
        rbf_weight (float): Weight for the RBF kernel component (default: 1.0)
        rbf_lengthscale (float): Fixed lengthscale for RBF kernel (default: 1.0)
        jitter (float): Small value to add to diagonal for numerical stability (default: 1e-6)
        **kwargs: Additional arguments passed to the parent class
    """
    
    def __init__(
        self,
        z_dim: int,
        combination_method: str = "additive",
        linear_weight: float = 1.0,
        rbf_weight: float = 1.0,
        rbf_lengthscale: float = 1.0,
        jitter: float = 1e-6,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.z_dim = z_dim
        self.combination_method = combination_method.lower()
        self.linear_weight = linear_weight
        self.rbf_weight = rbf_weight
        self.rbf_lengthscale = rbf_lengthscale
        self.jitter = jitter
        
        if self.combination_method not in ["product", "additive"]:
            raise ValueError("combination_method must be either 'product' or 'additive'")
        
        # Create linear kernel
        self.linear_kernel = LinearKernel()
        
        # Create RBF kernel with fixed lengthscale
        self.rbf_kernel = RBFKernel(ard_num_dims=z_dim)
        self.rbf_kernel.raw_lengthscale.requires_grad_(False)  # Fix lengthscale
        self.rbf_kernel.lengthscale = torch.ones(z_dim) * rbf_lengthscale
        
        # Scale the RBF kernel
        self.scaled_rbf_kernel = ScaleKernel(
            self.rbf_kernel,
            outputscale_constraint=gpytorch.constraints.Interval(1e-6, 1e4)
        )
        
        # Register kernels as modules
        self.register_module("linear_kernel", self.linear_kernel)
        self.register_module("scaled_rbf_kernel", self.scaled_rbf_kernel)
        
        # Create the combined kernel based on the method
        if self.combination_method == "additive":
            self.hybrid_kernel = AdditiveKernel(
                self.linear_kernel, 
                self.scaled_rbf_kernel
            )
        else:  # product
            self.hybrid_kernel = ProductKernel(
                self.linear_kernel, 
                self.scaled_rbf_kernel
            )
        
        self.register_module("hybrid_kernel", self.hybrid_kernel)
    
    def forward(self, x1, x2=None, diag=False, **kwargs):
        """
        Forward pass of the hybrid kernel.
        
        Args:
            x1: First input tensor
            x2: Second input tensor (optional, defaults to x1)
            diag: Whether to return diagonal elements only
            **kwargs: Additional arguments
            
        Returns:
            Kernel matrix or diagonal elements
        """
        result = self.hybrid_kernel(x1, x2, diag=diag, **kwargs)
        
        # Add numerical stability: ensure the kernel matrix is PSD
        if not diag:
            # Add jitter to diagonal for numerical stability
            if hasattr(result, 'add_diag'):
                result = result.add_diag(self.jitter)
            else:
                # If it's a dense tensor, add jitter directly
                if hasattr(result, 'to_dense'):
                    result_dense = result.to_dense()
                    result_dense.add_(torch.eye(result_dense.size(-1), device=x1.device, dtype=result_dense.dtype) * self.jitter)
                    result = result_dense
                else:
                    # For other types, try to add jitter if possible
                    try:
                        result = result + torch.eye(result.size(-1), device=x1.device, dtype=result.dtype) * self.jitter
                    except:
                        # If all else fails, just return the result
                        pass
        
        return result
    
    def __str__(self):
        return f"HybridKernel(z_dim={self.z_dim}, method={self.combination_method})"
    
    def __repr__(self):
        return self.__str__() 