"""NIGP multitask Woodbury marginal log-likelihood for RFFMTGPR."""

from __future__ import annotations

import gpytorch
import torch
from typing import TYPE_CHECKING

from ..utils.nigp_utils import (
    effective_noise_variance_mt,
    nigp_correction_enabled,
    posterior_mean_grad_wrt_x_mt,
)
from ..utils.rff_utils import (
    flatten_multitask_targets,
    woodbury_marginal_log_likelihood_mt_diag_noise,
)
from .rff_mt_mll import RFFMTWoodburyMarginalLogLikelihood

if TYPE_CHECKING:
    from ..models.rff_mtgpr import RFFMTGPR


class NIGPMTWoodburyMarginalLogLikelihood(RFFMTWoodburyMarginalLogLikelihood):
    """
    McHutchon–Rasmussen NIGP MLL on top of multitask RFF Woodbury.

    Effective covariance (centered):

        Σ = diag(d) + Ω Ωᵀ,
        d_{i,t} = σ_t² + Σ_d σ_{x,d}² (∂_{x_d} μ_t(x_i))²,

    with ``Ω = Φ ⊗ R_B`` and ``μ`` the homoskedastic MT Woodbury posterior
    (``∇μ`` detached). Requires ``model.nigp``, ``rank_likelihood=0``, and
    ``raw_input_noise``.

    When ``model.nigp_correction_enabled`` is ``False`` (freeze window), falls
    back to the parent Kronecker MT MLL.
    """

    def __init__(
        self,
        likelihood: gpytorch.likelihoods.Likelihood,
        model: RFFMTGPR,
        jitter: float = 1e-6,
        method: str = "eigen",
        promote_features: bool = False,
    ):
        if not getattr(model, "nigp", False):
            raise ValueError(
                "NIGPMTWoodburyMarginalLogLikelihood requires RFFMTGPR(nigp=True)."
            )
        if not hasattr(model, "raw_input_noise"):
            raise ValueError("NIGP model is missing raw_input_noise parameter.")
        if int(getattr(model, "rank_likelihood", 0)) > 0:
            raise ValueError(
                "NIGP MT Woodbury requires rank_likelihood=0 (diagonal task noise)."
            )
        super().__init__(
            likelihood,
            model,
            jitter=jitter,
            method=method,  # type: ignore[arg-type]
            promote_features=promote_features,
        )

    def forward(
        self,
        function_samples: torch.Tensor | gpytorch.distributions.MultivariateNormal,
        target: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        from ..models.rff_gpr import _drop_singleton_batch
        from ..models.rff_mtgpr import RFFMTGPR

        model: RFFMTGPR = self.model
        if not nigp_correction_enabled(model):
            return super().forward(function_samples, target, *args, **kwargs)

        train_x = _drop_singleton_batch(model.train_inputs[0])
        n_train = train_x.shape[0]
        phi = model.train_spatial_features()
        r_b = model.task_psd_factor()
        mean = model.mean_module(train_x)
        target_mt = _drop_singleton_batch(target)
        if target_mt.dim() == 1:
            target_mt = target_mt.reshape(-1, model.num_tasks)
        y_centered_mt = target_mt - mean
        y_centered = flatten_multitask_targets(y_centered_mt)
        task_noises = model.task_noises()

        # Slopes from homoskedastic μ (σ_t only); detached — not from NIGP-corrected μ.
        grad_mu = posterior_mean_grad_wrt_x_mt(
            model, train_x, y_centered, task_noises, jitter=self.jitter
        )
        d_nt = effective_noise_variance_mt(
            task_noises, model.input_noise_var, grad_mu
        )
        res = woodbury_marginal_log_likelihood_mt_diag_noise(
            d_nt,
            phi,
            r_b,
            n_train,
            y_centered,
            jitter=self.jitter,
        )
        res = self._add_other_terms(res, args)
        return res.div(target_mt.numel())
