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


def nigp_mt_woodbury_mll_class(
    method: str = "eigen",
    *,
    nigp_slope_refreshes: int | None = None,
    nigp_active_epochs: int | None = None,
    promote_features: bool = False,
) -> type[NIGPMTWoodburyMarginalLogLikelihood]:
    """Bind MT Woodbury method + optional paper-style slope refresh for :class:`GPTrainer`."""

    class BoundNIGPMTWoodburyMLL(NIGPMTWoodburyMarginalLogLikelihood):
        def __init__(
            self,
            likelihood,
            model,
            jitter: float = 1e-6,
            method: str = method,
            promote_features: bool = promote_features,
        ):
            super().__init__(
                likelihood,
                model,
                jitter=jitter,
                method=method,
                promote_features=promote_features,
                nigp_slope_refreshes=nigp_slope_refreshes,
                nigp_active_epochs=nigp_active_epochs,
            )

    BoundNIGPMTWoodburyMLL.__name__ = f"NIGPMTWoodburyMarginalLogLikelihood_{method}"
    BoundNIGPMTWoodburyMLL.__qualname__ = BoundNIGPMTWoodburyMLL.__name__
    return BoundNIGPMTWoodburyMLL


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

    ``nigp_slope_refreshes`` (optional): paper-style outer-loop slopes. After the
    correction enables, recompute ``∇μ`` this many times (including the unlock
    step), spaced evenly over ``nigp_active_epochs`` Adam steps; reuse the cached
    slopes between refreshes. ``None`` recomputes every step (legacy).
    """

    def __init__(
        self,
        likelihood: gpytorch.likelihoods.Likelihood,
        model: RFFMTGPR,
        jitter: float = 1e-6,
        method: str = "eigen",
        promote_features: bool = False,
        *,
        nigp_slope_refreshes: int | None = None,
        nigp_active_epochs: int | None = None,
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
        if nigp_slope_refreshes is not None:
            refreshes = int(nigp_slope_refreshes)
            if refreshes < 1:
                raise ValueError(
                    f"nigp_slope_refreshes must be >= 1 or None, got {nigp_slope_refreshes}."
                )
            nigp_slope_refreshes = refreshes
            if nigp_active_epochs is None:
                raise ValueError(
                    "nigp_active_epochs is required when nigp_slope_refreshes is set."
                )
            if int(nigp_active_epochs) < 1:
                raise ValueError(
                    f"nigp_active_epochs must be >= 1, got {nigp_active_epochs}."
                )
        super().__init__(
            likelihood,
            model,
            jitter=jitter,
            method=method,  # type: ignore[arg-type]
            promote_features=promote_features,
        )
        self.nigp_slope_refreshes = nigp_slope_refreshes
        self.nigp_active_epochs = (
            None if nigp_active_epochs is None else int(nigp_active_epochs)
        )
        self._cached_grad_mu: torch.Tensor | None = None
        self._slope_phase_active = False
        self._active_step = 0
        self._refresh_count = 0

    @property
    def nigp_slope_refresh_count(self) -> int:
        """Number of ``∇μ`` recomputations in the current NIGP-active phase."""
        return int(self._refresh_count)

    def invalidate_nigp_slope_cache(self) -> None:
        """Drop cached slopes (forces refresh on the next correction forward)."""
        self._cached_grad_mu = None
        self._slope_phase_active = False
        self._active_step = 0
        self._refresh_count = 0

    def _begin_slope_phase_if_needed(self) -> None:
        if self._slope_phase_active:
            return
        self._cached_grad_mu = None
        self._active_step = 0
        self._refresh_count = 0
        self._slope_phase_active = True

    def _should_refresh_grad_mu(self) -> bool:
        if self.nigp_slope_refreshes is None:
            return True
        if self._cached_grad_mu is None:
            return True
        r = int(self.nigp_slope_refreshes)
        if self._refresh_count >= r:
            return False
        a = max(1, int(self.nigp_active_epochs or 1))
        interval = max(1, a // r)
        return (self._active_step % interval) == 0

    def _grad_mu_for_correction(
        self,
        model: RFFMTGPR,
        train_x: torch.Tensor,
        y_centered: torch.Tensor,
        task_noises: torch.Tensor,
    ) -> torch.Tensor:
        """Homoskedastic ``∇μ``, optionally paper-style cached between refreshes."""
        self._begin_slope_phase_if_needed()
        if self._should_refresh_grad_mu():
            grad_mu = posterior_mean_grad_wrt_x_mt(
                model, train_x, y_centered, task_noises, jitter=self.jitter
            )
            self._cached_grad_mu = grad_mu.detach()
            self._refresh_count += 1
        else:
            grad_mu = self._cached_grad_mu
            assert grad_mu is not None
        self._active_step += 1
        return grad_mu

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
            self._slope_phase_active = False
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
        grad_mu = self._grad_mu_for_correction(
            model, train_x, y_centered, task_noises
        )
        d_nt = effective_noise_variance_mt(
            task_noises, model.input_noise_var, grad_mu
        )
        from ..utils.rff_utils import _task_factor_is_diagonal
        from ..utils.woodbury_mll_autograd import (
            use_reference_woodbury_autograd,
            woodbury_mt_diag_mll_apply,
        )

        if (
            not use_reference_woodbury_autograd()
            and not _task_factor_is_diagonal(r_b)
        ):
            res = woodbury_mt_diag_mll_apply(
                d_nt, phi, r_b, y_centered, jitter=self.jitter
            )
        else:
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
