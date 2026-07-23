"""Woodbury marginal log-likelihood for LRNNGPR (Deep Basis Kernel)."""

from __future__ import annotations

import gpytorch
import torch
from typing import TYPE_CHECKING

from ..utils.lrnn_utils import woodbury_marginal_log_likelihood_lrnn

if TYPE_CHECKING:
    from ..models.lrnn_gpr import LRNNGPR


class LRNNWoodburyMarginalLogLikelihood(gpytorch.mlls.ExactMarginalLogLikelihood):
    """
    Marginal log-likelihood for LRNN / DBK with optional variance correction.

    Target covariance (after subtracting the mean):

        Sigma = noise * I_n + Phi Phi^T,

    with ``Phi = scaled_features(train_x)`` of shape ``(n, r)``.

    When ``variance_correction`` is True (default), subtracts the DBK trace
    regularization term ``tr(C) / (2 σ²)`` from ``log p(y)`` (Zhu et al. 2025 §3.3).

    ``function_samples`` is ignored; the trainer passes ``None`` and does not run
    a full ExactGP forward for this MLL.
    """

    def __init__(
        self,
        likelihood: gpytorch.likelihoods.Likelihood,
        model: LRNNGPR,
        jitter: float = 1e-6,
        variance_correction: bool | None = None,
    ):
        from ..models.lrnn_gpr import LRNNGPR as _LRNNGPR

        if not isinstance(model, _LRNNGPR):
            raise TypeError("LRNNWoodburyMarginalLogLikelihood requires an LRNNGPR model.")
        super().__init__(likelihood, model)
        self.jitter = jitter
        if variance_correction is None:
            variance_correction = bool(getattr(model, "variance_correction", True))
        self.variance_correction = bool(variance_correction)

    def forward(
        self,
        function_samples: torch.Tensor | gpytorch.distributions.MultivariateNormal,
        target: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        from ..models.lrnn_gpr import _drop_singleton_batch

        model: LRNNGPR = self.model
        train_x = _drop_singleton_batch(model.train_inputs[0])
        phi_train = model.scaled_features(train_x)
        mean = model.mean_module(train_x)
        if mean.dim() > 1 and mean.shape[0] == 1 and phi_train.dim() == 2:
            mean = mean.squeeze(0)
        target_1d = _drop_singleton_batch(target)
        if phi_train.dim() == 3 and target_1d.dim() == 1:
            y_centered = target_1d.unsqueeze(0) - mean
        else:
            y_centered = target_1d - mean
        noise = model.likelihood.noise
        if noise.dim() > 1:
            noise = noise.reshape(noise.shape[0])
        res = woodbury_marginal_log_likelihood_lrnn(
            noise,
            phi_train,
            y_centered,
            jitter=self.jitter,
            variance_correction=self.variance_correction,
        )
        res = self._add_other_terms(res, args)
        num_data = target_1d.shape[-1]
        return res.div(num_data)
