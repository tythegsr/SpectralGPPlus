"""Standalone RFF feature extractor."""

import math

import torch

from .sampling import SUPPORTED_KERNELS, sample_rff_omega


class RFFApproximator:
    """
    Transforms input data into Random Fourier Feature space so that
    the inner product z(x)'z(y) approximates a shift-invariant kernel k(x, y).

    Supports scalar or per-dimension (ARD) lengthscales.
    """

    SUPPORTED_KERNELS = SUPPORTED_KERNELS

    def __init__(
        self,
        num_samples: int,
        base_kernel: str = "rbf",
        input_dim: int = 1,
        lengthscale: float | torch.Tensor = 1.0,
        device: torch.device | None = None,
    ):
        """
        Parameters
        ----------
        num_samples   : D — number of random frequency samples. The output
                        feature vector has dimension 2*D (cos + sin components).
        base_kernel   : which shift-invariant kernel to approximate.
        input_dim     : d — dimensionality of the input space.
        lengthscale   : ell — kernel length scale. gamma = 1/(2*ell^2) for RBF.
        device        : torch device for the random weights.
        """
        if base_kernel not in self.SUPPORTED_KERNELS:
            raise ValueError(f"base_kernel must be one of {self.SUPPORTED_KERNELS}")

        self.D = num_samples
        self.base_kernel = base_kernel
        self.d = input_dim
        self.lengthscale = lengthscale
        self.device = device or torch.device("cpu")

        self.omega = None
        self.resample()

    def resample(self) -> None:
        """Draw fresh random frequencies from p(omega) for the chosen kernel."""
        self.omega = sample_rff_omega(
            self.base_kernel,
            self.D,
            self.d,
            self.lengthscale,
            device=self.device,
        )

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Map input data X to the 2D-dimensional RFF feature space.

        Parameters
        ----------
        X : Tensor of shape (..., d)

        Returns
        -------
        Z : Tensor of shape (..., 2D)
            z(x) = (1/sqrt(D)) * [cos(omega_1'x),...,cos(omega_D'x), sin(omega_1'x),...,sin(omega_D'x)]
        """
        proj = X @ self.omega.T
        scale = 1.0 / math.sqrt(self.D)
        return scale * torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

    def kernel_approx(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute the approximate kernel matrix z(X)'z(Y).

        Parameters
        ----------
        X : Tensor (N, d)
        Y : Tensor (M, d)

        Returns
        -------
        K_approx : Tensor (N, M)
        """
        ZX = self.transform(X)
        ZY = self.transform(Y)
        return ZX @ ZY.T
