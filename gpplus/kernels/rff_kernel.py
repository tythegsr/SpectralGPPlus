"""Random Fourier Feature kernel for GPPlus."""



from __future__ import annotations



from typing import Optional



import torch

from linear_operator.operators import MatmulLinearOperator

from torch import Tensor



from ..utils.rff_utils import RFF_SAMPLING_MODES, RffSampling, featurize_rbf, init_rbf_weights

from .unconstrained_kernel import UnconstrainedKernel





class RFFKernel(UnconstrainedKernel):

    """

    RBF random Fourier feature kernel (Sutherland–Schneider cos/sin features).



    ``rff_sampling`` controls how frequency weights are drawn:



    - ``"rff"``: i.i.d. Gaussian (default).

    - ``"orf"``: full ORF (Yu et al. arXiv:1610.09072 Eq. 2), QR + chi(d) scaling.

    - ``"sorf"``: structured ORF (Yu et al. Eq. 5), Walsh–Hadamard + Rademacher signs.



    Training with :class:`~gpplus.models.RFFGPR` and

    :class:`~gpplus.training.rff_mll.RFFWoodburyMarginalLogLikelihood` uses

    Woodbury solves on a (2D)x(2D) system instead of an n x n Cholesky.

    """



    has_lengthscale = True

    is_stationary = True



    def __init__(

        self,

        num_samples: int = 500,

        ard_num_dims: Optional[int] = None,

        num_dims: Optional[int] = None,

        rff_sampling: RffSampling = "rff",

        **kwargs,

    ):

        if rff_sampling not in RFF_SAMPLING_MODES:

            raise ValueError(f"rff_sampling must be one of {sorted(RFF_SAMPLING_MODES)}, got {rff_sampling!r}.")

        super().__init__(ard_num_dims=ard_num_dims, **kwargs)

        self.num_samples = num_samples

        self.rff_sampling = rff_sampling

        self._feature_cache_version = 0

        if num_dims is not None:

            self._init_weights(num_dims, num_samples, rff_sampling=rff_sampling)



    def _init_weights(

        self,

        num_dims: int,

        num_samples: Optional[int] = None,

        randn_weights: Optional[Tensor] = None,

        *,

        spectral: bool = False,

        rff_sampling: RffSampling | None = None,

    ) -> None:

        D = num_samples if num_samples is not None else self.num_samples

        sampling = rff_sampling if rff_sampling is not None else self.rff_sampling

        if randn_weights is None:

            ls = self.lengthscale if spectral and self.has_lengthscale else None

            randn_weights = init_rbf_weights(

                num_dims,

                D,

                device=self.raw_lengthscale.device,

                dtype=self.raw_lengthscale.dtype,

                lengthscale=ls,

                rff_sampling=sampling,

            )

        self.register_buffer("randn_weights", randn_weights)



    def resample_weights(self, spectral: bool = True, rff_sampling: RffSampling | None = None) -> None:

        """Redraw omega (optionally from current lengthscale / sampling mode). Invalidates feature caches."""

        num_dims = self.randn_weights.shape[-2] if hasattr(self, "randn_weights") else self.ard_num_dims

        if num_dims is None:

            raise RuntimeError("Cannot resample RFF weights before kernel dimensions are known.")

        sampling = rff_sampling if rff_sampling is not None else self.rff_sampling

        self._init_weights(

            num_dims,

            self.num_samples,

            spectral=spectral,

            rff_sampling=sampling,

        )

        if hasattr(self, "_feature_cache_version"):

            self._feature_cache_version += 1



    def featurize(self, x: Tensor) -> Tensor:

        if not hasattr(self, "randn_weights"):

            self._init_weights(x.shape[-1], self.num_samples)

        return featurize_rbf(x, self.randn_weights, self.lengthscale, self.num_samples)



    def forward(

        self,

        x1: Tensor,

        x2: Tensor,

        diag: bool = False,

        last_dim_is_batch: bool = False,

        **params,

    ) -> Tensor:

        if last_dim_is_batch:

            x1 = x1.transpose(-1, -2).unsqueeze(-1)

            x2 = x2.transpose(-1, -2).unsqueeze(-1)



        if not hasattr(self, "randn_weights"):

            self._init_weights(x1.shape[-1], self.num_samples)



        z1 = self.featurize(x1)

        if x1 is x2 or (x1.shape == x2.shape and torch.equal(x1, x2)):

            z2 = z1

        else:

            z2 = self.featurize(x2)



        if diag:

            return (z1 * z2).sum(dim=-1)



        return MatmulLinearOperator(z1, z2.transpose(-1, -2))


