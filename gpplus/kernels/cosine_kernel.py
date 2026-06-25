import math
from typing import Optional

import torch
from gpytorch.priors import Prior
from torch import Tensor

from ..constraints import SoftClamp
from ..kernels import UnconstrainedKernel


class CosineKernel(UnconstrainedKernel):
    r"""
    Cosine distance kernel with log10 period parameterization.

    .. math::

        k(x, x') = \cos\!\left(\pi \,\big\| \tfrac{x}{P} - \tfrac{x'}{P} \big\|_2 \right),
        \quad P = 10^{\,\text{period}}

    Here ``period`` is the learnable *log10* period; the effective period used in the forward
    pass is ``10**period``. The raw parameter is constrained with
    :class:`~gpplus.constraints.SoftClamp` (default ``[-6, 3]``), matching lengthscale conventions.

    .. note::

        ``cos(\pi \cdot \text{distance})`` on Euclidean inputs is not positive semi-definite
        in high dimensions. Exact GP training may require extra jitter or a different kernel
        (e.g. :class:`gpytorch.kernels.PeriodicKernel`) when PSD is required.
    """

    has_lengthscale = False
    is_stationary = True

    def __init__(
        self,
        period_prior: Optional[Prior] = None,
        period_constraint: Optional[SoftClamp] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if period_constraint is None:
            period_constraint = SoftClamp(lower_bound=-3, upper_bound=3, margin=1e-2)

        self.register_parameter(
            name="raw_period",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)),
        )

        if period_prior is not None:
            if not isinstance(period_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(period_prior).__name__)
            self.register_prior("period_prior", period_prior, self._period_param, self._period_closure)

        self.register_constraint("raw_period", period_constraint)

    def _period_param(self, m):
        return m.period

    def _period_closure(self, m, v):
        m._set_period(v)

    @property
    def period(self) -> Tensor:
        return self.raw_period_constraint.transform(self.raw_period)

    @period.setter
    def period(self, value):
        self._set_period(value)

    def _set_period(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_period)
        self.initialize(raw_period=self.raw_period_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag: bool = False, **params):
        effective_period = torch.pow(10.0, self.period)
        x1_ = x1.div(effective_period)
        x2_ = x2.div(effective_period)
        diff = self.covar_dist(x1_, x2_, diag=diag, **params)
        return torch.cos(diff.mul(math.pi))
