"""Woodbury marginal log-likelihood for RFFGPR (and duck-typed Woodbury models)."""

from __future__ import annotations

import gpytorch
import torch
from typing import TYPE_CHECKING

from ..utils.rff_utils import WoodburyForm, woodbury_marginal_log_likelihood

if TYPE_CHECKING:
    from ..models.rff_gpr import RFFGPR


def _is_woodbury_feature_model(model) -> bool:
    """True if model exposes ``scaled_features`` for Woodbury MLL."""
    return callable(getattr(model, "scaled_features", None))


class RFFWoodburyMarginalLogLikelihood(gpytorch.mlls.ExactMarginalLogLikelihood):
    """
    Marginal log-likelihood for RFF-GP with homoskedastic Gaussian noise.

    Target covariance (after subtracting the mean):

        Sigma = noise * I_n + Phi Phi^T,

    with ``Phi = scaled_features(train_x)`` of shape ``(n, m)``.

    Default ``woodbury_form="primal"`` factors ``M = I + ΦᵀΦ/σ²``.
    Set ``woodbury_form="dual"`` to use ``Λ = ΦᵀΦ + σ² I_m`` (stable for small ``σ²``).

    Subclasses :class:`~gpytorch.mlls.ExactMarginalLogLikelihood` for GPyTorch MLL API and prior terms
    (``_add_other_terms``); ``forward`` does **not** use the parent exact ``n x n`` covariance path.

    ``function_samples`` is ignored; :class:`~gpplus.training.training_single_run.GPTrainerSingleProcess`
    passes ``None`` and does not run a full ``ExactGP`` forward for this MLL.

    See ``docs/overleaf/rff_woodbury_derivation.tex`` for notation (``Phi`` vs features-as-columns ``Z``).

    Requires a model with ``scaled_features`` (typically :class:`~gpplus.models.RFFGPR`).
    For LRNN / DBK with variance correction, use
    :class:`~gpplus.training.lrnn_mll.LRNNWoodburyMarginalLogLikelihood` instead.
    """

    def __init__(
        self,
        likelihood: gpytorch.likelihoods.Likelihood,
        model: RFFGPR,
        jitter: float = 1e-6,
        woodbury_form: WoodburyForm = "primal",
    ):
        from ..models.rff_gpr import RFFGPR as _RFFGPR
        from ..models.lrnn_gpr import LRNNGPR as _LRNNGPR

        if not (_is_woodbury_feature_model(model) or isinstance(model, (_RFFGPR, _LRNNGPR))):
            raise TypeError(
                "RFFWoodburyMarginalLogLikelihood requires a model with scaled_features "
                "(e.g. RFFGPR)."
            )
        super().__init__(likelihood, model)
        self.jitter = jitter
        self.woodbury_form: WoodburyForm = woodbury_form

    def forward(
        self,
        function_samples: torch.Tensor | gpytorch.distributions.MultivariateNormal,
        target: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        from ..models.rff_gpr import _drop_singleton_batch

        model = self.model
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
            woodbury_form=self.woodbury_form,
        )
        res = self._add_other_terms(res, args)
        num_data = target.numel()
        return res.div(num_data)


# Backward-compatible alias for the generic Woodbury MLL surface.
WoodburyMarginalLogLikelihood = RFFWoodburyMarginalLogLikelihood
