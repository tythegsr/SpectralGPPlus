"""Woodbury marginal log-likelihood for RFFGPR (and duck-typed Woodbury models)."""

from __future__ import annotations

import gpytorch
import torch
from typing import TYPE_CHECKING

from ..utils.line_profile import profile
from ..utils.rff_utils import WoodburyForm, woodbury_marginal_log_likelihood
from ..utils.woodbury_mll_autograd import (
    use_fused_featurize_vjp,
    use_reference_woodbury_autograd,
    woodbury_dual_rff_mll_apply,
)

if TYPE_CHECKING:
    from ..models.rff_gpr import RFFGPR


def _is_woodbury_feature_model(model) -> bool:
    """True if model exposes ``scaled_features`` for Woodbury MLL."""
    return callable(getattr(model, "scaled_features", None))


def _try_fused_dual_rff_mll(model, train_x, y_centered, noise, jitter: float):
    """Fused dual Woodbury + RFF featurize VJP, or None if not applicable."""
    if not use_fused_featurize_vjp() or use_reference_woodbury_autograd():
        return None
    from ..kernels import LogScaleKernel, RFFKernel
    from ..models.rff_gpr import RFFGPR

    if not isinstance(model, RFFGPR):
        return None
    covar = model.covar_module
    if not isinstance(covar, LogScaleKernel):
        return None
    base = covar.base_kernel
    if not isinstance(base, RFFKernel) or not hasattr(base, "randn_weights"):
        return None
    return woodbury_dual_rff_mll_apply(
        noise,
        y_centered,
        train_x,
        base.lengthscale,
        covar.outputscale,
        base.randn_weights,
        base.num_samples,
        jitter=jitter,
    )


class RFFWoodburyMarginalLogLikelihood(gpytorch.mlls.ExactMarginalLogLikelihood):
    """
    Marginal log-likelihood for RFF-GP with homoskedastic Gaussian noise.

    Target covariance (after subtracting the mean):

        Sigma = noise * I_n + Phi Phi^T,

    with ``Phi = scaled_features(train_x)`` of shape ``(n, m)``.

    Default ``woodbury_form="primal"`` factors ``M = I + ΦᵀΦ/σ²``.
    Set ``woodbury_form="dual"`` to use ``Λ = ΦᵀΦ + σ² I_m`` (stable for small ``σ²``).

    For dual + :class:`~gpplus.models.RFFGPR`, featurize is fused into the custom
    Woodbury autograd (no cos/sin graph) unless
    ``GPPLUS_WOODBURY_FUSE_FEATURIZE=0`` or ``GPPLUS_WOODBURY_AUTOGRAD=reference``.

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

    @profile
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
        mean = model.mean_module(train_x)
        target_1d = _drop_singleton_batch(target)
        noise = model.likelihood.noise
        if noise.dim() > 1:
            noise = noise.reshape(noise.shape[0])

        # Dual + RFFGPR: fused featurize VJP (no cos/sin autograd graph).
        if self.woodbury_form == "dual":
            if mean.dim() > 1 and mean.shape[0] == 1:
                mean_for_y = mean.squeeze(0)
            else:
                mean_for_y = mean
            if target_1d.dim() == 1 and mean_for_y.dim() == 2:
                y_centered = target_1d.unsqueeze(0) - mean_for_y
            else:
                y_centered = target_1d - mean_for_y
            fused = _try_fused_dual_rff_mll(
                model, train_x, y_centered, noise, self.jitter
            )
            if fused is not None:
                res = self._add_other_terms(fused, args)
                return res.div(target_1d.shape[-1])

        phi_train = model.scaled_features(train_x)
        if mean.dim() > 1 and mean.shape[0] == 1 and phi_train.dim() == 2:
            mean = mean.squeeze(0)
        if phi_train.dim() == 3 and target_1d.dim() == 1:
            y_centered = target_1d.unsqueeze(0) - mean
        else:
            y_centered = target_1d - mean
        res = woodbury_marginal_log_likelihood(
            noise,
            phi_train,
            y_centered,
            jitter=self.jitter,
            woodbury_form=self.woodbury_form,
        )
        res = self._add_other_terms(res, args)
        num_data = target_1d.shape[-1]
        return res.div(num_data)


# Backward-compatible alias for the generic Woodbury MLL surface.
WoodburyMarginalLogLikelihood = RFFWoodburyMarginalLogLikelihood
