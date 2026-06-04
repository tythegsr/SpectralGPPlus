"""Woodbury marginal log-likelihood for RFFGPR."""

from __future__ import annotations

import gpytorch
import torch
from typing import TYPE_CHECKING

from ..utils.rff_utils import woodbury_marginal_log_likelihood

if TYPE_CHECKING:
    from ..models.rff_gpr import RFFGPR


class RFFWoodburyMarginalLogLikelihood(gpytorch.mlls.ExactMarginalLogLikelihood):
    """
    Marginal log-likelihood for RFF-GP with homoskedastic Gaussian noise.

    Target covariance (after subtracting the mean):

        Sigma = noise * I_n + Phi Phi^T,

    with ``Phi = scaled_features(train_x)`` of shape ``(n, m)``. The Woodbury inverse is

        Sigma^{-1} = (noise I)^{-1}
                     - (noise I)^{-1} Phi (I + Phi^T (noise I)^{-1} Phi)^{-1}
                       Phi^T (noise I)^{-1}.

    Training evaluates ``log p(y | X)`` via :func:`~gpplus.utils.rff_utils.woodbury_marginal_log_likelihood`
    using an ``m x m`` Cholesky of ``M = I + Phi^T Phi / noise`` only (no ``n x n`` Cholesky of ``Sigma``).

    Subclasses :class:`~gpytorch.mlls.ExactMarginalLogLikelihood` for GPyTorch MLL API and prior terms
    (``_add_other_terms``); ``forward`` does **not** use the parent exact ``n x n`` covariance path.

    ``function_samples`` is ignored; :class:`~gpplus.training.training_single_run.GPTrainerSingleProcess`
    passes ``None`` and does not run a full ``ExactGP`` forward for this MLL.

    See ``docs/overleaf/rff_woodbury_derivation.tex`` for notation (``Phi`` vs features-as-columns ``Z``).

    Requires an :class:`~gpplus.models.RFFGPR` model.
    """

    def __init__(
        self,
        likelihood: gpytorch.likelihoods.Likelihood,
        model: RFFGPR,
        jitter: float = 1e-6,
    ):
        from ..models.rff_gpr import RFFGPR as _RFFGPR

        if not isinstance(model, _RFFGPR):
            raise TypeError("RFFWoodburyMarginalLogLikelihood requires an RFFGPR model.")
        super().__init__(likelihood, model)
        self.jitter = jitter

    def forward(
        self,
        function_samples: torch.Tensor | gpytorch.distributions.MultivariateNormal,
        target: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        from ..models.rff_gpr import _drop_singleton_batch

        model: RFFGPR = self.model
        train_x = _drop_singleton_batch(model.train_inputs[0])
        # Phi in Sigma = noise I + Phi Phi^T (n x m)
        phi_train = model.scaled_features(train_x)
        mean = model.mean_module(train_x)
        if mean.dim() > 1 and mean.shape[0] == 1:
            mean = mean.squeeze(0)
        target_1d = _drop_singleton_batch(target)
        y_centered = target_1d - mean
        noise = model.likelihood.noise
        res = woodbury_marginal_log_likelihood(
            noise,
            phi_train,
            y_centered,
            jitter=self.jitter,
        )
        res = self._add_other_terms(res, args)
        num_data = target.numel()
        return res.div(num_data)
