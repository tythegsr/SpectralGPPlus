"""Multitask Gaussian likelihood with log10 SoftClamp task noises."""

from __future__ import annotations

from typing import Optional, Union

import torch
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.priors import Prior
from torch import Tensor

from ..constraints import SoftClamp


class LogMultitaskGaussianLikelihood(MultitaskGaussianLikelihood):
    r"""
    Multitask Gaussian likelihood with per-task noise in log10 SoftClamp space.

    Matches :class:`~gpplus.likelihoods.LogGaussianLikelihood`: learnable parameters are
    unconstrained ``raw_task_noises``; actual variances are
    :math:`\sigma^2_t = 10^{\mathrm{SoftClamp}(\mathrm{raw})}`.

    Defaults to Woodbury-compatible settings: ``rank=0``, ``has_global_noise=False``,
    ``has_task_noise=True``.
    """

    def __init__(
        self,
        num_tasks: int,
        rank: int = 0,
        batch_shape: torch.Size = torch.Size(),
        task_prior: Optional[Prior] = None,
        noise_prior: Optional[Prior] = None,
        noise_constraint: Optional[SoftClamp] = None,
        has_global_noise: bool = False,
        has_task_noise: bool = True,
    ) -> None:
        if noise_constraint is None:
            noise_constraint = SoftClamp(lower_bound=-5.0, upper_bound=3.0)
        super().__init__(
            num_tasks=num_tasks,
            rank=rank,
            batch_shape=batch_shape,
            task_prior=task_prior,
            noise_prior=noise_prior,
            noise_constraint=noise_constraint,
            has_global_noise=has_global_noise,
            has_task_noise=has_task_noise,
        )

    @property
    def task_noises(self) -> Optional[Tensor]:
        if self.rank != 0:
            raise AttributeError("task_noises is only defined for rank=0 likelihoods.")
        return torch.pow(10.0, self.raw_task_noises_constraint.transform(self.raw_task_noises))

    @task_noises.setter
    def task_noises(self, value: Union[float, Tensor]) -> None:
        self._set_task_noises(value)

    def _set_task_noises(self, value: Union[float, Tensor]) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_task_noises)
        else:
            value = value.to(device=self.raw_task_noises.device, dtype=self.raw_task_noises.dtype)
        log_value = torch.log10(value.clamp_min(1e-30))
        self.initialize(raw_task_noises=self.raw_task_noises_constraint.inverse_transform(log_value))
