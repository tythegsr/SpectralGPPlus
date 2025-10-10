from typing import Any, List, Optional, Union

import numpy as np
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import _GaussianLikelihoodBase
from gpytorch.likelihoods.noise_models import _HomoskedasticNoiseBase
from linear_operator.operators import ConstantDiagLinearOperator, DiagLinearOperator
from torch import Tensor


class MultiLikelihood(_GaussianLikelihoodBase):
    """
    Multifidelity likelihood that allows different noise levels for each fidelity source.

    Args:
        encoded_cols: Either:
            - Tensor, list, or numpy array of column indices for one-hot encoded fidelity levels, OR
            - Single integer indicating the column index containing fidelity indicators
        noise_prior: Prior distribution for noise parameters
        noise_constraint: Constraint for noise parameters
        batch_shape: Batch shape for the likelihood
        training_x: Training data to determine fidelity order
    """

    def __init__(
        self,
        encoded_cols: Union[int, List[int], Tensor],
        noise_prior=None,
        noise_constraint=None,
        batch_shape=torch.Size(),
        training_data: Optional[Tensor] = None,
        **kwargs,
    ):
        # Store encoded_cols and determine if it's one-hot or single column
        self.encoded_cols = encoded_cols
        self.is_onehot = isinstance(encoded_cols, (list, tuple, Tensor, np.ndarray))

        if self.is_onehot:
            self.encoded_cols = torch.tensor(encoded_cols, dtype=torch.long)
            self.num_fidelities = len(encoded_cols)
            self.all_fidelities = torch.arange(self.num_fidelities)
        else:
            self.source_col = int(encoded_cols)
            self.num_fidelities = None  # Will be determined from data
            self.all_fidelities = None  # Will be determined from data

        # Initialize noise model
        noise_covar = MultiNoise(
            noise_prior=noise_prior,
            noise_constraint=noise_constraint,
            batch_shape=batch_shape,
            num_noises=self.num_fidelities if self.is_onehot else 1,
        )

        # Initialize parent class with noise_covar FIRST (needed before registering buffers)
        super().__init__(noise_covar=noise_covar)

        # Buffers to persist fidelity/noise indices across state_dict save/load
        self.register_buffer("fidel_indices", torch.empty(0, dtype=torch.long), persistent=True)
        self.register_buffer("noise_indices", torch.empty(0, dtype=torch.long), persistent=True)

        # Optionally set fidelity indices from provided training data to fix source order
        if training_data is not None:
            try:
                self.set_fidelity_indices(training_data)
            except KeyError:
                pass

    def _extract_fidelity_indices(self, x: Tensor) -> Tensor:
        """
        Extract fidelity indices from input data based on encoded_cols.

        Args:
            x: Input tensor of shape (N, D) where N is number of samples, D is number of features

        Returns:
            Tensor of shape (N,) containing fidelity indices for each sample
        """
        if self.is_onehot:
            # One-hot encoded case: find which column has value 1
            onehot_cols = x[:, self.encoded_cols]  # Shape: (N, num_fidelities)
            fidel_indices = torch.argmax(onehot_cols, dim=1)  # Shape: (N,)
        else:
            # Single column case: use values directly as fidelity indices
            fidel_indices = x[:, self.source_col].long()  # Shape: (N,)

            # Update discovered fidelities if not set
            if self.all_fidelities is None:
                unique_fidelities = torch.unique(fidel_indices)
                self.all_fidelities = unique_fidelities
                self.num_fidelities = len(unique_fidelities)

        return fidel_indices

    def set_fidelity_indices(self, x: Tensor) -> None:
        """
        Set the fidelity indices from input data. This should be called during training.

        Args:
            x: Input tensor of shape (N, D) where N is number of samples, D is number of features
        """
        fidel = self._extract_fidelity_indices(x)
        # Ensure on same device
        fidel = fidel.to(device=self.noise_covar.noise.device)
        if self.fidel_indices.numel() == 0 or self.fidel_indices.shape != fidel.shape:
            # Resize/replace buffer
            self.register_buffer("fidel_indices", fidel.detach(), persistent=True)
        else:
            self.fidel_indices.copy_(fidel)
        if self.noise_indices.numel() == 0:
            ni = self.all_fidelities if self.all_fidelities is not None else torch.tensor([0], dtype=torch.long)
            ni = ni.to(device=self.noise_covar.noise.device, dtype=torch.long)
            self.register_buffer("noise_indices", ni.detach(), persistent=True)

    def set_training_data(self, training_data: Tensor) -> None:
        """Convenience method to pass training inputs and lock in fidelity order."""
        self.set_fidelity_indices(training_data)

    @property
    def noise(self) -> Tensor:
        return self.noise_covar.noise

    @noise.setter
    def noise(self, value: Tensor) -> None:
        self.noise_covar.initialize(noise=value)

    @property
    def raw_noise(self) -> Tensor:
        return self.noise_covar.raw_noise

    @raw_noise.setter
    def raw_noise(self, value: Tensor) -> None:
        self.noise_covar.initialize(raw_noise=value)

    def _shaped_noise_covar(self, base_shape: torch.Size, *params: Any, **kwargs: Any):
        """Get the noise covariance matrix for the given parameters.
        Simple rule: use provided fidel_indices or the stored ones. Otherwise, error.
        """
        if "fidel_indices" in kwargs and kwargs["fidel_indices"] is not None:
            pass
        elif self.fidel_indices.numel() > 0:
            kwargs["fidel_indices"] = self.fidel_indices
            if self.noise_indices.numel() > 0:
                kwargs["noise_indices"] = self.noise_indices.tolist()
        else:
            raise ValueError(
                "Fidelity indices are required to build the noise matrix. "
                "Call likelihood.set_fidelity_indices(x) beforehand or provide fidel_indices."
            )

        return self.noise_covar.forward(*params, shape=base_shape, **kwargs)

    def marginal(self, function_dist: MultivariateNormal, *params: Any, **kwargs: Any) -> MultivariateNormal:
        """Compute the marginal distribution by adding noise covariance."""
        mean, covar = function_dist.mean, function_dist.lazy_covariance_matrix
        noise_covar = self._shaped_noise_covar(mean.shape, *params, **kwargs)

        # Add noise covariance to the function covariance
        full_covar = covar + noise_covar

        return function_dist.__class__(mean, full_covar)


class MultiNoise(_HomoskedasticNoiseBase):
    """
    Multifidelity noise model that applies different noise levels based on fidelity indices.

    Args:
        noise_prior: Prior distribution for noise parameters
        noise_constraint: Constraint for noise parameters
        batch_shape: Batch shape for the noise model
        num_noises: Number of different noise levels to learn
    """

    def __init__(self, noise_prior=None, noise_constraint=None, batch_shape=torch.Size(), num_noises=1):
        super().__init__(noise_prior, noise_constraint, batch_shape, num_tasks=num_noises)

    def forward(
        self,
        *params: Any,
        shape: Optional[torch.Size] = None,
        fidel_indices: Optional[Tensor] = None,
        noise_indices: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> DiagLinearOperator:
        """
        Compute a diagonal noise covariance where each point's variance
        is selected by its fidelity index.

        Args:
            *params: Parameters from the base noise model
            shape: Unused—shape is inferred from fidel_indices
            fidel_indices: Tensor of shape (N,) mapping each point to a fidelity level
            noise_indices: Which fidelity levels correspond to each learned noise parameter
            **kwargs: Additional keyword arguments

        Returns:
            DiagLinearOperator: A (N×N) diagonal covariance tensor where entry i,i
                = noise[fidelity_indices[i]]
        """
        if fidel_indices is None or len(fidel_indices) == 0:
            raise ValueError("fidel_indices must be provided and non-empty")

        if noise_indices is None:
            noise_indices = list(range(self.num_tasks))

        # Get noise variances from parent class
        covar = super().forward(*params, shape=fidel_indices.shape, **kwargs)

        # Handle different covariance dimensions
        if covar.dim() > 2:
            if covar.shape[1] != len(noise_indices):
                raise ValueError(
                    f"Number of noise parameters ({covar.shape[1]}) does not \
                        match number of noise indices ({len(noise_indices)})"
                )

        if covar.dim() == 4:  # batch case
            covar = covar.squeeze(0)
        elif covar.dim() == 5:  # batch case
            covar = covar.squeeze(1)

        # Initialize diagonal matrix with zeros
        temp = ConstantDiagLinearOperator(torch.tensor([0.0]), len(fidel_indices))
        temp = temp.to(dtype=covar.dtype, device=covar.device)

        # For each noise level, create a diagonal mask and accumulate
        for i, noise_idx in enumerate(noise_indices):
            # Create diagonal mask for points with this fidelity level
            # Ensure the mask is on the same device as the covariance matrix
            mask_values = (fidel_indices == noise_idx).float()
            if covar.dim() >= 3:
                # Get device from the first noise parameter
                device = covar[i, ...].device if covar.dim() == 3 else covar[0, i, ...].device
                mask_values = mask_values.to(device)
            diag_mask = DiagLinearOperator(mask_values)

            # Select the appropriate noise variance and multiply by mask
            if covar.dim() == 4:  # batch case
                temp += diag_mask * covar[:, i, ...]
            elif covar.dim() == 3:  # no batch
                temp += diag_mask * covar[i, ...]
            elif covar.dim() == 2:  # single noise parameter
                temp += diag_mask * covar
            else:
                raise ValueError(f"Unexpected covariance dimension: {covar.dim()}")

        return temp
