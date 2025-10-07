#!/usr/bin/env python3

from typing import Optional

import torch
from gpytorch.kernels import Kernel
from gpytorch.priors import Prior
from linear_operator.operators import to_dense

from ..constraints import SoftClamp


class ProcessVarianceKernel(Kernel):
    r"""
    Decorates an existing kernel object with an output scale, i.e.

    .. math::

       \begin{equation*}
          K_{\text{scaled}} = \theta_\text{outputscale} K_{\text{orig}}
       \end{equation*}

    where :math:`\theta_\text{outputscale}` is the `outputscale` parameter.

    In batch-mode (i.e. when :math:`x_1` and :math:`x_2` are batches of input matrices), each
    batch of data can have its own `outputscale` parameter by setting the `batch_shape`
    keyword argument to the appropriate number of batches.

    .. note::
        The outputscale parameter is internally parameterized on a log scale (via raw_outputscale)
        for better numerical stability and optimization properties. The outputscale property applies
        the constraint and 10^x transformation. You can set a prior on this parameter using the
        outputscale_prior argument.

    Args:
        base_kernel (Kernel):
            The base kernel to be scaled.
        batch_shape (int, optional):
            Set this if you want a separate outputscale for each batch of input data. It should be `b`
            if x1 is a `b x n x d` tensor. Default: `torch.Size([])`
        outputscale_prior (Prior, optional): Set this if you want to apply a prior to the outputscale
            parameter.  Default: `None`
        outputscale_constraint (Constraint, optional): Set this if you want to apply a constraint to the
            raw_outputscale parameter. Default: `SoftClamp(-6.0, 3.0)` (constrains log scale).

    Attributes:
        base_kernel (Kernel):
            The kernel module to be scaled.
        raw_outputscale (Parameter):
            The raw, unconstrained log-scale parameter. This is what the optimizer updates.
        outputscale (Tensor):
            The transformed outputscale parameter (10^constrained(raw_outputscale)).
            Size/shape depends on the batch_shape arguments.

    Example:
        >>> x = torch.randn(10, 5)
        >>> base_covar_module = gpytorch.kernels.RBFKernel()
        >>> scaled_covar_module = ProcessVarianceKernel(base_covar_module)
        >>> covar = scaled_covar_module(x)  # Output: LinearOperator of size (10 x 10)
    """

    @property
    def is_stationary(self) -> bool:
        """
        Kernel is stationary if base kernel is stationary.
        """
        return self.base_kernel.is_stationary

    def __init__(
        self,
        base_kernel: Kernel,
        outputscale_prior: Optional[Prior] = None,
        outputscale_constraint: Optional[SoftClamp] = None,
        **kwargs,
    ):
        if base_kernel.active_dims is not None:
            kwargs["active_dims"] = base_kernel.active_dims
        super(ProcessVarianceKernel, self).__init__(**kwargs)

        # Default constraint for log outputscale (allows scales from 0.00001 to 1000)
        if outputscale_constraint is None:
            outputscale_constraint = SoftClamp(lower_bound=-6.0, upper_bound=3.0)

        self.base_kernel = base_kernel

        # Initialize log outputscale parameter (raw parameter is in log space)
        log_outputscale = torch.zeros(*self.batch_shape) if len(self.batch_shape) else torch.tensor(0.0)
        self.register_parameter(name="raw_outputscale", parameter=torch.nn.Parameter(log_outputscale))

        if outputscale_prior is not None:
            if not isinstance(outputscale_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(outputscale_prior).__name__)
            self.register_prior(
                "outputscale_prior", outputscale_prior, self._outputscale_param, self._outputscale_closure
            )

        self.register_constraint("raw_outputscale", outputscale_constraint)

    def _outputscale_param(self, m):
        return m.outputscale

    def _outputscale_closure(self, m, v):
        m._set_outputscale(v)

    @property
    def outputscale(self):
        """
        Get the actual outputscale parameter.
        Applies constraint to raw_outputscale, then transforms via 10^x.
        """
        # Apply constraint to get log-scale value
        log_scale = self.raw_outputscale_constraint.transform(self.raw_outputscale)
        # Transform from log-scale: outputscale = 10^log_scale
        return torch.pow(10, log_scale)

    @outputscale.setter
    def outputscale(self, value):
        """Set the outputscale parameter (will be converted to raw parameter internally)."""
        self._set_outputscale(value)

    def _set_outputscale(self, value):
        """Internal method to set outputscale parameter (converts to raw parameter)."""
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_outputscale)
        # Convert to log scale: log_scale = log10(value)
        log_value = torch.log10(value)
        # Apply inverse constraint to get raw parameter
        self.initialize(raw_outputscale=self.raw_outputscale_constraint.inverse_transform(log_value))

    def forward(self, x1, x2, last_dim_is_batch=False, diag=False, **params):
        orig_output = self.base_kernel.forward(x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params)
        outputscales = self.outputscale
        if last_dim_is_batch:
            outputscales = outputscales.unsqueeze(-1)
        if diag:
            outputscales = outputscales.unsqueeze(-1)
            return to_dense(orig_output) * outputscales
        else:
            outputscales = outputscales.view(*outputscales.shape, 1, 1)
            return orig_output.mul(outputscales)

    def num_outputs_per_input(self, x1, x2):
        return self.base_kernel.num_outputs_per_input(x1, x2)

    def prediction_strategy(self, train_inputs, train_prior_dist, train_labels, likelihood):
        return self.base_kernel.prediction_strategy(train_inputs, train_prior_dist, train_labels, likelihood)
