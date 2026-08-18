"""Minibatch ELBO for :class:`~gpplus.models.svgp_gpr.SVGPR`.

    L = sum_i E_q[log N(y_i | f_i, d_i)] - beta * KL(q(u) || p(u))

With homoskedastic noise (``d_i = sigma_y^2``) this is exactly GPyTorch's
``VariationalELBO``; the class below reproduces it in closed form so the NIGP
case, where ``d_i`` varies per point, shares one code path.

The likelihood term is a sum over independent points, so scaling a minibatch
sum by ``N/B`` is unbiased for the full-data term while the KL, which does not
touch the data, is evaluated exactly. The returned value is divided by ``N`` so
the loss is per observation, matching
:class:`~gpplus.training.vi_rff_elbo.VIRFFELBO` and the Woodbury MLL -- stop
conditions and logged losses stay comparable across paths.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..config import logger


class SVGPELBO(nn.Module):
    """
    Maximization objective (an ELBO, not a negative loss) for an SVGP.

    Constructed as ``SVGPELBO(likelihood, model, ...)`` to match the
    ``mll_class(likelihood, model)`` convention used elsewhere in
    :mod:`gpplus.training`, but called with an explicit minibatch.

    Parameters
    ----------
    kl_beta :
        Weight on ``KL(q(u) || p(u))``. ``1.0`` is the true ELBO; smaller values
        anneal the regularizer and give a bound that no longer holds.
    """

    def __init__(self, likelihood, model, *, kl_beta: float = 1.0):
        super().__init__()
        from ..models.svgp_gpr import SVGPR

        if not isinstance(model, SVGPR):
            raise TypeError(
                f"SVGPELBO requires an SVGPR model, got {type(model).__name__}."
            )
        if kl_beta < 0:
            raise ValueError(f"kl_beta must be >= 0, got {kl_beta}.")
        self.likelihood = likelihood
        self.model = model
        self.kl_beta = float(kl_beta)
        if kl_beta != 1.0:
            logger.info("SVGPELBO: KL weight beta=%g (annealed bound).", kl_beta)

    def expected_log_likelihood(self, x_batch: Tensor, y_batch: Tensor) -> Tensor:
        """Per-point expected log likelihood over the batch, shape ``(B,)``."""
        return self.model.expected_log_likelihood(x_batch, y_batch)

    def kl_divergence(self) -> Tensor:
        """``KL(q(u) || p(u))``, independent of the data."""
        return self.model.kl_divergence()

    def forward(self, x_batch: Tensor, y_batch: Tensor, num_data: int) -> Tensor:
        """
        Per-observation ELBO estimate from one minibatch.

        ``num_data`` is the full training set size ``N`` used to rescale the
        minibatch likelihood sum; passing ``B == N`` recovers the exact
        full-data bound.
        """
        if x_batch.shape[0] != y_batch.shape[-1]:
            raise ValueError(
                f"Batch size mismatch: x_batch has {x_batch.shape[0]} rows but "
                f"y_batch has {y_batch.shape[-1]} entries."
            )
        num_data = int(num_data)
        if num_data < 1:
            raise ValueError(f"num_data must be >= 1, got {num_data}.")

        ell = self.expected_log_likelihood(x_batch, y_batch)
        scale = float(num_data) / float(ell.shape[-1])
        total = ell.sum(dim=-1) * scale - self.kl_beta * self.kl_divergence()
        return total / float(num_data)

    def full_data_elbo(self, x: Tensor, y: Tensor, *, chunk_size: int = 0) -> Tensor:
        """
        Exact full-data ELBO (not per observation), optionally chunked.

        Chunking only splits the likelihood sum, so the result matches the
        unchunked value up to summation order.
        """
        n = int(x.shape[0])
        step = n if chunk_size <= 0 else int(chunk_size)
        total = None
        for start in range(0, n, step):
            part = self.expected_log_likelihood(
                x[start : start + step], y[start : start + step]
            ).sum(dim=-1)
            total = part if total is None else total + part
        if total is None:
            total = torch.zeros((), dtype=x.dtype, device=x.device)
        return total - self.kl_beta * self.kl_divergence()


__all__ = ["SVGPELBO"]
