"""IndexKernel with log10 SoftClamp diagonal task variances."""

from __future__ import annotations

from typing import Optional

import torch
from gpytorch.kernels import IndexKernel
from gpytorch.priors import Prior
from torch import Tensor

from ..constraints import SoftClamp


class LogIndexKernel(IndexKernel):
    r"""
    ICM task covariance with diagonal variances in log10 SoftClamp space.

    Same structure as GPyTorch :class:`~gpytorch.kernels.IndexKernel`:

    .. math::

        B = FF^\top + \mathrm{diag}(\mathbf{v}), \quad
        v_t = 10^{\mathrm{SoftClamp}(\mathrm{raw\_var}_t)}.

    ``covar_factor`` remains unconstrained (signed). Default SoftClamp bounds match
    :class:`~gpplus.kernels.LogScaleKernel` outputscale (``[-5, 4]``).
    """

    def __init__(
        self,
        num_tasks: int,
        rank: Optional[int] = 1,
        prior: Optional[Prior] = None,
        var_constraint: Optional[SoftClamp] = None,
        **kwargs,
    ):
        if var_constraint is None:
            var_constraint = SoftClamp(lower_bound=-5, upper_bound=4, margin=1e-2)
        super().__init__(
            num_tasks=num_tasks,
            rank=rank,
            prior=prior,
            var_constraint=var_constraint,
            **kwargs,
        )

    @property
    def var(self) -> Tensor:
        return torch.pow(10.0, self.raw_var_constraint.transform(self.raw_var))

    @var.setter
    def var(self, value: Tensor) -> None:
        self._set_var(value)

    def _set_var(self, value: Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_var)
        else:
            value = value.to(device=self.raw_var.device, dtype=self.raw_var.dtype)
        log_value = torch.log10(value.clamp_min(1e-30))
        self.initialize(raw_var=self.raw_var_constraint.inverse_transform(log_value))
