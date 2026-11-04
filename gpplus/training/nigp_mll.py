"""NIGP marginal log-likelihoods (Woodbury RFF and exact GPR)."""

from __future__ import annotations

import gpytorch
import torch
from gpytorch.distributions import MultivariateNormal
from linear_operator.operators import DiagLinearOperator
from typing import TYPE_CHECKING

from ..utils.line_profile import profile
from ..utils.nigp_utils import (
    effective_noise_variance,
    exact_posterior_mean_grad_wrt_x,
    nigp_correction_enabled,
    posterior_mean_grad_wrt_x,
)
from ..utils.rff_utils import WoodburyForm, woodbury_marginal_log_likelihood_diag_noise
from ..utils.woodbury_mll_autograd import (
    use_fused_featurize_vjp,
    use_reference_woodbury_autograd,
    woodbury_dual_rff_diag_mll_apply,
)
from .rff_mll import RFFWoodburyMarginalLogLikelihood

if TYPE_CHECKING:
    from ..models.gpr import GPR
    from ..models.rff_gpr import RFFGPR


def _try_fused_dual_rff_diag_mll(model, train_x, y_centered, d, jitter: float):
    """Fused diag-noise dual Woodbury + RFF featurize VJP, or None if not applicable."""
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
    return woodbury_dual_rff_diag_mll_apply(
        d,
        y_centered,
        train_x,
        base.lengthscale,
        covar.outputscale,
        base.randn_weights,
        base.num_samples,
        jitter=jitter,
    )


def nigp_woodbury_mll_class(
    woodbury_form: WoodburyForm = "dual",
    *,
    nigp_slope_refreshes: int | None = None,
    nigp_active_epochs: int | None = None,
    nigp_slope_manual: bool = False,
) -> type[NIGPWoodburyMarginalLogLikelihood]:
    """Bind Woodbury form + optional paper-style slope refresh for :class:`GPTrainer`."""

    class BoundNIGPWoodburyMLL(NIGPWoodburyMarginalLogLikelihood):
        def __init__(
            self,
            likelihood,
            model,
            jitter: float = 1e-6,
            woodbury_form: WoodburyForm = woodbury_form,
        ):
            super().__init__(
                likelihood,
                model,
                jitter=jitter,
                woodbury_form=woodbury_form,
                nigp_slope_refreshes=nigp_slope_refreshes,
                nigp_active_epochs=nigp_active_epochs,
                nigp_slope_manual=nigp_slope_manual,
            )

    BoundNIGPWoodburyMLL.__name__ = f"NIGPWoodburyMarginalLogLikelihood_{woodbury_form}"
    BoundNIGPWoodburyMLL.__qualname__ = BoundNIGPWoodburyMLL.__name__
    return BoundNIGPWoodburyMLL


class NIGPWoodburyMarginalLogLikelihood(RFFWoodburyMarginalLogLikelihood):
    """
    McHutchon–Rasmussen NIGP MLL on top of RFF Woodbury.

    Effective covariance:

        Σ = diag(d) + Φ Φᵀ,
        d_i = σ_y² + Σ_d σ_{x,d}² (∂_{x_d} μ(x_i))²

    with ``μ`` the homoskedastic Woodbury posterior mean (frozen feature weights)
    and ``∇μ`` detached when forming ``d``. Requires ``model.nigp`` and
    ``model.raw_input_noise``.

    When ``model.use_hetero_noise`` is True, ``σ_y`` is the per-point Noise-SORF
    ``σ_{y,het}²(x)`` (MAP mean). With NIGP off that is hetero-only; with NIGP on,
    ``d = σ_{y,het}² + σ_nigp²``.

    When both hetero and NIGP correction are off, falls back to the parent
    homoskedastic Woodbury MLL.

    ``nigp_slope_manual`` / ``model.hetero_nigp_manual_slopes``: refresh ``∇μ`` only
    when the cache is empty or ``model.nigp_force_slope_refresh`` is set.
    """

    def __init__(
        self,
        likelihood: gpytorch.likelihoods.Likelihood,
        model: RFFGPR,
        jitter: float = 1e-6,
        woodbury_form: str = "primal",
        *,
        nigp_slope_refreshes: int | None = None,
        nigp_active_epochs: int | None = None,
        nigp_slope_manual: bool = False,
    ):
        if not getattr(model, "nigp", False):
            raise ValueError(
                "NIGPWoodburyMarginalLogLikelihood requires RFFGPR(nigp=True)."
            )
        if not hasattr(model, "raw_input_noise"):
            raise ValueError("NIGP model is missing raw_input_noise parameter.")
        self.nigp_slope_manual = bool(nigp_slope_manual) or bool(
            getattr(model, "hetero_nigp_manual_slopes", False)
        )
        if nigp_slope_refreshes is not None and not self.nigp_slope_manual:
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
        super().__init__(likelihood, model, jitter=jitter, woodbury_form=woodbury_form)
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
        model = self.model
        if getattr(model, "nigp_force_slope_refresh", False):
            model.nigp_force_slope_refresh = False
            return True
        manual = self.nigp_slope_manual or bool(
            getattr(model, "hetero_nigp_manual_slopes", False)
        )
        if manual:
            return self._cached_grad_mu is None
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
        model,
        train_x: torch.Tensor,
        y_centered: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Homoskedastic ``∇μ``, optionally paper-style cached between refreshes."""
        self._begin_slope_phase_if_needed()
        if self._should_refresh_grad_mu():
            grad_mu = posterior_mean_grad_wrt_x(
                model, train_x, y_centered, noise, jitter=self.jitter
            )
            self._cached_grad_mu = grad_mu.detach()
            self._refresh_count += 1
        else:
            grad_mu = self._cached_grad_mu
            assert grad_mu is not None
        self._active_step += 1
        return grad_mu

    def _base_observation_noise(self, model, train_x: torch.Tensor) -> torch.Tensor:
        """Scalar ``σ_{y,hom}²`` or per-point ``σ_{y,het}²(x)``."""
        if getattr(model, "use_hetero_noise", False) and hasattr(model, "sigma_y_het"):
            return model.sigma_y_het(train_x)
        noise = model.likelihood.noise
        if noise.dim() > 1:
            noise = noise.reshape(noise.shape[0])
        return noise

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
        use_hetero = bool(getattr(model, "use_hetero_noise", False))
        nigp_on = nigp_correction_enabled(model)

        if not nigp_on and not use_hetero:
            self._slope_phase_active = False
            return super().forward(function_samples, target, *args, **kwargs)

        train_x = _drop_singleton_batch(model.train_inputs[0])
        mean = model.mean_module(train_x)
        target_1d = _drop_singleton_batch(target)
        noise = self._base_observation_noise(model, train_x)

        if mean.dim() > 1 and mean.shape[0] == 1:
            mean_for_y = mean.squeeze(0)
        else:
            mean_for_y = mean
        if target_1d.dim() == 1 and mean_for_y.dim() == 2:
            y_centered = target_1d.unsqueeze(0) - mean_for_y
        else:
            y_centered = target_1d - mean_for_y

        if nigp_on:
            slope_noise = model.likelihood.noise
            if slope_noise.dim() > 1:
                slope_noise = slope_noise.reshape(slope_noise.shape[0])
            slope_noise = slope_noise.detach()
            grad_mu = self._grad_mu_for_correction(
                model, train_x, y_centered, slope_noise
            )
            d = effective_noise_variance(noise, model.input_noise_var, grad_mu)
        else:
            d = noise.clamp_min(1e-12)
            self._slope_phase_active = False

        if self.woodbury_form == "dual":
            fused = _try_fused_dual_rff_diag_mll(
                model, train_x, y_centered, d, self.jitter
            )
            if fused is not None:
                res = self._add_other_terms(fused, args)
                return res.div(target_1d.shape[-1])

        phi_train = model.scaled_features(train_x)
        res = woodbury_marginal_log_likelihood_diag_noise(
            d, phi_train, y_centered, jitter=self.jitter
        )
        res = self._add_other_terms(res, args)
        num_data = target_1d.shape[-1]
        return res.div(num_data)


class NIGPExactMarginalLogLikelihood(gpytorch.mlls.ExactMarginalLogLikelihood):
    """
    McHutchon–Rasmussen NIGP MLL for exact :class:`~gpplus.models.gpr.GPR`.

    Observation covariance uses ``K + diag(d)`` with

        d_i = σ_y² + Σ_d σ_{x,d}² (∂_{x_d} μ(x_i))²

    and detached exact-GP ``∇μ`` (homoskedastic ``α``). Requires ``model.nigp``.

    When ``model.nigp_correction_enabled`` is ``False``, uses the parent exact MLL.
    """

    def __init__(
        self,
        likelihood: gpytorch.likelihoods.Likelihood,
        model: GPR,
        jitter: float = 1e-6,
    ):
        if not getattr(model, "nigp", False):
            raise ValueError("NIGPExactMarginalLogLikelihood requires GPR(nigp=True).")
        if not hasattr(model, "raw_input_noise"):
            raise ValueError("NIGP model is missing raw_input_noise parameter.")
        super().__init__(likelihood, model)
        self.jitter = float(jitter)

    def forward(
        self,
        function_dist: MultivariateNormal,
        target: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        model = self.model
        if not nigp_correction_enabled(model):
            return super().forward(function_dist, target, *args, **kwargs)

        train_x = model.train_inputs[0]
        if train_x.dim() == 3 and train_x.shape[0] == 1:
            train_x = train_x.squeeze(0)

        noise = model.likelihood.noise
        if noise.dim() > 1:
            noise = noise.reshape(noise.shape[0])

        grad_mu = exact_posterior_mean_grad_wrt_x(
            model, train_x, noise, jitter=self.jitter
        )
        d = effective_noise_variance(noise, model.input_noise_var, grad_mu)

        mean = function_dist.mean
        covar = function_dist.lazy_covariance_matrix + DiagLinearOperator(d)
        dist = MultivariateNormal(mean, covar)
        res = dist.log_prob(target)
        res = self._add_other_terms(res, args)
        num_data = target.size(-1)
        return res.div(num_data)


__all__ = [
    "NIGPWoodburyMarginalLogLikelihood",
    "NIGPExactMarginalLogLikelihood",
    "nigp_woodbury_mll_class",
]
