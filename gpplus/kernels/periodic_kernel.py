import math
from typing import Optional

import torch
from gpytorch.kernels import Kernel
from gpytorch.priors import Prior
from torch import Tensor

from ..constraints import SoftClamp
from ..kernels import UnconstrainedKernel


class PeriodicKernel(UnconstrainedKernel):
    r"""
    Periodic kernel (Mackay Eq. 47) with log10 lengthscale and period parameterization.

    .. math::

        k(x, x') = \exp\!\left(
            -\sum_i 10^{\,\omega_i}\,
            \sin^2\!\left(\frac{\pi}{p}(x_i - x'_i)\right)
        \right),
        \quad p = 10^{\,\text{period}},\;
        \frac{2}{\ell_i^2} = 10^{\,\omega_i}

    Here ``lengthscale`` stores :math:`\omega_i` (log10 precision per input dimension when
    ``ard_num_dims`` is set). ``period`` is a single shared *log10* period across all
    dimensions; the effective period is ``10**period``.

    This kernel is positive semi-definite. For output scaling, wrap with
    :class:`~gpplus.kernels.LogScaleKernel`. For a non-PSD cosine-distance kernel,
    see :class:`~gpplus.kernels.CosineKernel`.
    """

    has_lengthscale = True
    is_stationary = True

    def __init__(
        self,
        period_prior: Optional[Prior] = None,
        period_constraint: Optional[SoftClamp] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if period_constraint is None:
            period_constraint = SoftClamp(lower_bound=-6, upper_bound=3, margin=1e-2)

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
        last_dim_is_batch = params.pop("last_dim_is_batch", False)

        effective_period = torch.pow(10.0, self.period)
        precision = torch.pow(10.0, self.lengthscale)

        x1_ = x1.div(effective_period / math.pi)
        x2_ = x2.div(effective_period / math.pi)
        diff = Kernel.covar_dist(self, x1_, x2_, diag=diag, last_dim_is_batch=True, **params)

        if diag:
            precision = precision[..., 0, :, None]
        else:
            precision = precision[..., 0, :, None, None]
        exp_term = diff.sin().pow(2.0).mul(precision).mul(-1.0)

        if not last_dim_is_batch:
            exp_term = exp_term.sum(dim=(-2 if diag else -3))

        return exp_term.exp()
