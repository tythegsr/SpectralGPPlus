"""Woodbury MLL with soft probabilistic physical-scale bound penalties."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from ..utils.rff_utils import (
    WoodburyForm,
    woodbury_predict,
    woodbury_predictive_obs_std,
)
from .bound_penalty import BoundConfig, soft_probabilistic_bound_penalty
from .rff_mll import RFFWoodburyMarginalLogLikelihood

if TYPE_CHECKING:
    import gpytorch
    from ..models.rff_gpr import RFFGPR


class BoundPenalizedRFFWoodburyMarginalLogLikelihood(RFFWoodburyMarginalLogLikelihood):
    """
    Woodbury MLL minus a soft probabilistic bound penalty in physical units.

    Trainer uses ``loss = -mll(...)``, so returning ``mll_avg - lam * penalty``
    yields ``loss = -MLL + lam * penalty``.
    """

    def __init__(
        self,
        likelihood: gpytorch.likelihoods.Likelihood,
        model: RFFGPR,
        jitter: float = 1e-6,
        woodbury_form: WoodburyForm = "primal",
        *,
        bound_config: BoundConfig,
        y_mean: float | torch.Tensor = 0.0,
        y_std: float | torch.Tensor = 1.0,
    ):
        super().__init__(likelihood, model, jitter=jitter, woodbury_form=woodbury_form)
        self.bound_config = bound_config
        self.y_mean = float(y_mean) if not torch.is_tensor(y_mean) else float(y_mean.detach().cpu())
        self.y_std = float(y_std) if not torch.is_tensor(y_std) else float(y_std.detach().cpu())
        if self.y_std <= 0:
            raise ValueError(f"y_std must be > 0 for bound penalty unstandardization, got {self.y_std}")

    def _constraint_indices(self, n: int, device: torch.device) -> torch.Tensor:
        max_pts = self.bound_config.max_points
        if max_pts is None or n <= int(max_pts):
            return torch.arange(n, device=device)
        # Fixed stride subsample (deterministic, covers the train set evenly).
        return torch.linspace(0, n - 1, steps=int(max_pts), device=device).round().long().unique()

    def forward(
        self,
        function_samples: torch.Tensor | gpytorch.distributions.MultivariateNormal,
        target: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        from ..models.rff_gpr import _drop_singleton_batch

        mll = super().forward(function_samples, target, *args, **kwargs)

        model = self.model
        train_x = _drop_singleton_batch(model.train_inputs[0])
        train_y = _drop_singleton_batch(target)
        phi_train = model.scaled_features(train_x)
        if phi_train.dim() == 3:
            # Batched inits: apply penalty on the first batch slice only for stability.
            # Full per-batch penalties can be added later if needed.
            phi_train = phi_train[0]
            if train_y.dim() > 1:
                train_y = train_y[0]
            if train_x.dim() == 3:
                train_x = train_x[0]

        mean = model.mean_module(train_x)
        if mean.dim() > 1 and mean.shape[0] == 1:
            mean = mean.squeeze(0)
        y_centered = train_y - mean
        noise = model.likelihood.noise
        if noise.dim() > 0:
            noise = noise.reshape(-1)[0]

        n = train_x.shape[-2] if train_x.dim() >= 2 else train_x.shape[0]
        idx = self._constraint_indices(int(n), device=train_x.device)
        x_c = train_x.index_select(0, idx)
        phi_c = phi_train.index_select(0, idx)

        f_mean, f_var = woodbury_predict(
            noise,
            phi_train,
            phi_c,
            y_centered,
            jitter=self.jitter,
            woodbury_form=self.woodbury_form,
        )
        f_mean = f_mean + model.mean_module(x_c).reshape_as(f_mean)
        obs_std = woodbury_predictive_obs_std(f_var, noise)

        mu_phys = f_mean * self.y_std + self.y_mean
        sigma_phys = obs_std * self.y_std
        cfg = self.bound_config
        penalty = soft_probabilistic_bound_penalty(
            mu_phys,
            sigma_phys,
            a=cfg.a,
            b=cfg.b,
            k=cfg.k,
            alpha=cfg.alpha,
        )
        raw_lam = getattr(model, "raw_bound_penalty_lambda", None)
        if raw_lam is not None:
            lam = model.bound_penalty_lambda.reshape(-1)[0]
        else:
            lam = penalty.new_tensor(float(cfg.lam))
        return mll - lam * penalty


def bound_penalized_rff_mll_class(
    bound_config: BoundConfig,
    *,
    y_mean: float | torch.Tensor = 0.0,
    y_std: float | torch.Tensor = 1.0,
    woodbury_form: WoodburyForm = "dual",
) -> type[BoundPenalizedRFFWoodburyMarginalLogLikelihood]:
    """Bind bound config + y scaler for :class:`~gpplus.training.GPTrainer`."""

    class BoundConfiguredMLL(BoundPenalizedRFFWoodburyMarginalLogLikelihood):
        def __init__(self, likelihood, model, jitter: float = 1e-6):
            super().__init__(
                likelihood,
                model,
                jitter=jitter,
                woodbury_form=woodbury_form,
                bound_config=bound_config,
                y_mean=y_mean,
                y_std=y_std,
            )

    BoundConfiguredMLL.__name__ = "BoundPenalizedRFFWoodburyMLL"
    BoundConfiguredMLL.__qualname__ = "BoundPenalizedRFFWoodburyMLL"
    return BoundConfiguredMLL
