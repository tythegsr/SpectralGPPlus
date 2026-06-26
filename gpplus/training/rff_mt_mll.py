"""Woodbury marginal log-likelihood for RFFMTGPR."""

from __future__ import annotations

import gpytorch
import torch
from typing import TYPE_CHECKING

from ..utils.rff_utils import flatten_multitask_targets, woodbury_marginal_log_likelihood_mt

if TYPE_CHECKING:
    from ..models.rff_mtgpr import RFFMTGPR


class RFFMTWoodburyMarginalLogLikelihood(gpytorch.mlls.ExactMarginalLogLikelihood):
    """
    Marginal log-likelihood for multitask RFF-GP with per-task diagonal noise.

    Target covariance (centered): Sigma = Lambda + Omega Omega^T with
    Lambda = I_n kron diag(task_noises) and Omega = Phi kron R_B.
    """

    def __init__(
        self,
        likelihood: gpytorch.likelihoods.Likelihood,
        model: RFFMTGPR,
        jitter: float = 1e-6,
    ):
        from ..models.rff_mtgpr import RFFMTGPR as _RFFMTGPR

        if not isinstance(model, _RFFMTGPR):
            raise TypeError("RFFMTWoodburyMarginalLogLikelihood requires an RFFMTGPR model.")
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
        from ..models.rff_mtgpr import RFFMTGPR

        model: RFFMTGPR = self.model
        train_x = _drop_singleton_batch(model.train_inputs[0])
        n_train = train_x.shape[0]
        omega = model.train_joint_features()
        mean = model.mean_module(train_x)
        target_mt = _drop_singleton_batch(target)
        if target_mt.dim() == 1:
            target_mt = target_mt.reshape(-1, model.num_tasks)
        y_centered = flatten_multitask_targets(target_mt - mean)
        task_noises = model.task_noises()
        res = woodbury_marginal_log_likelihood_mt(
            task_noises,
            omega,
            n_train,
            y_centered,
            jitter=self.jitter,
        )
        res = self._add_other_terms(res, args)
        return res.div(target.numel())
