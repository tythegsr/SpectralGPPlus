"""Minibatch ELBO for :class:`~gpplus.models.vi_rff_gpr.VIRFFGPR`.

    L = Σ_i E_q[log N(y_i | m(x_i) + φ(x_i)ᵀ w, d_i)] − β · KL(q(w) ‖ N(0, I_m))

The likelihood term is a sum over independent data points, so rescaling a
minibatch sum by ``N/B`` gives an unbiased estimate of the full-data term while
the KL is evaluated exactly (it does not depend on the data). The result is
divided by ``N`` so the reported loss is per observation, matching the
convention of :class:`~gpplus.training.rff_mll.RFFWoodburyMarginalLogLikelihood`.

The bound is tight: at the optimal ``q`` this equals the exact Woodbury
marginal log likelihood (see ``gpplus/utils/test_vi_rff_elbo.py``).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..config import logger


class VIRFFELBO(nn.Module):
    """
    Negative-free (maximization) ELBO objective for a variational RFF GP.

    Constructed as ``VIRFFELBO(likelihood, model, ...)`` to match the
    ``mll_class(likelihood, model)`` convention used elsewhere in
    :mod:`gpplus.training`, but called with an explicit minibatch rather than a
    GPyTorch distribution.

    Parameters
    ----------
    kl_beta :
        Weight on the KL term. ``1.0`` is the true ELBO; smaller values anneal
        the regularizer, which can speed early progress at the cost of a bound
        that is no longer valid.
    """

    def __init__(
        self,
        likelihood,
        model,
        *,
        kl_beta: float = 1.0,
    ):
        super().__init__()
        from ..models.vi_rff_gpr import VIRFFGPR

        if not isinstance(model, VIRFFGPR):
            raise TypeError(
                f"VIRFFELBO requires a VIRFFGPR model, got {type(model).__name__}."
            )
        if kl_beta < 0:
            raise ValueError(f"kl_beta must be >= 0, got {kl_beta}.")
        self.likelihood = likelihood
        self.model = model
        self.kl_beta = float(kl_beta)
        if kl_beta != 1.0:
            logger.info("VIRFFELBO: KL weight beta=%g (annealed bound).", kl_beta)

    def expected_log_likelihood(self, x_batch: Tensor, y_batch: Tensor) -> Tensor:
        """Per-point expected log likelihood over the batch, shape ``(B,)``."""
        return self.model.expected_log_likelihood(x_batch, y_batch)

    def kl_divergence(self) -> Tensor:
        """``KL(q(w) ‖ N(0, I_m))``, independent of the data."""
        return self.model.kl_divergence()

    def forward(
        self,
        x_batch: Tensor,
        y_batch: Tensor,
        num_data: int,
    ) -> Tensor:
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
        batch_size = int(ell.shape[-1])
        scale = float(num_data) / float(batch_size)
        total = ell.sum(dim=-1) * scale - self.kl_beta * self.kl_divergence()
        return total / float(num_data)

    def full_data_elbo(
        self,
        x: Tensor,
        y: Tensor,
        *,
        chunk_size: int = 0,
    ) -> Tensor:
        """
        Exact full-data ELBO (not per observation), optionally chunked.

        Chunking only splits the likelihood sum, so the result is identical to
        the unchunked value up to floating point summation order.
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


def gaussian_entropy_constant(num_features: int) -> float:
    """``m/2 · log(2πe)``; useful when comparing ELBO decompositions by hand."""
    return 0.5 * num_features * math.log(2.0 * math.pi * math.e)


__all__ = ["VIRFFELBO", "gaussian_entropy_constant"]
