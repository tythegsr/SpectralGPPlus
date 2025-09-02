from typing import Any, List, Optional, Union

import gpytorch
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
        learn_additional_noise: Whether to learn additional noise beyond the base noise
        batch_shape: Batch shape for the likelihood
        training_data: Optional training data tensor to determine number of fidelities for single column case
    """

    def __init__(
        self,
        encoded_cols: Union[Tensor, List[int], int],
        noise_prior=None,
        noise_constraint=None,
        learn_additional_noise: bool = False,
        batch_shape: torch.Size = torch.Size(),
        training_data: Optional[Tensor] = None,
        **kwargs,
    ) -> None:
        # Handle both single column index and one-hot encoded columns
        if isinstance(encoded_cols, int):
            # Single column index - not one-hot encoded
            self.is_onehot = False
            self.source_col = encoded_cols
            # Check training data if provided to determine number of fidelities upfront
            if training_data is not None and hasattr(training_data, "shape") and len(training_data.shape) >= 2:
                # Extract the source column and count unique values
                source_values = training_data[:, encoded_cols].long()
                unique_sources = torch.unique(source_values)
                self.num_fidelities = len(unique_sources)
                print(f"Detected {self.num_fidelities} fidelity levels from training data in column {encoded_cols}")
            else:
                # We'll need to determine num_fidelities from data during forward pass
                self.num_fidelities = None
        else:
            # One-hot encoded columns - handle numpy arrays, tensors, lists, etc.
            self.is_onehot = True
            # Convert to tensor if it's not already
            if not isinstance(encoded_cols, torch.Tensor):
                if hasattr(encoded_cols, "__len__") and not isinstance(encoded_cols, (str, bytes)):
                    # It's an array-like object (numpy array, list, etc.)
                    encoded_cols = torch.tensor(encoded_cols, dtype=torch.long)
                else:
                    # Fallback: wrap in tensor
                    encoded_cols = [encoded_cols]
            self.encoded_cols = encoded_cols
            self.num_fidelities = len(encoded_cols)

        # Create noise covariance with appropriate number of noise parameters
        if self.num_fidelities is not None:
            initial_fidelities = self.num_fidelities
        else:
            # Single column case without training data - use a reasonable default
            # This will be updated during the first forward pass
            initial_fidelities = 2

        noise_covar = MultiNoise(
            num_fidelities=initial_fidelities,
            noise_prior=noise_prior,
            noise_constraint=noise_constraint,
            batch_shape=batch_shape,
        )

        super().__init__(noise_covar=noise_covar)

        # Store the noise covariance for potential updates (but don't add it as a module)
        # self._noise_covar = noise_covar  # This was causing duplicate noise_covar objects

    def _update_noise_covar_fidelities(self, new_num_fidelities: int):
        """Update the noise covariance to handle a new number of fidelities."""
        if hasattr(self, "noise_covar") and self.noise_covar is not None:
            # Update the number of fidelities in the noise covariance
            if hasattr(self.noise_covar, "num_fidelities"):
                self.noise_covar.num_fidelities = new_num_fidelities
            # The noise parameters will be automatically resized by PyTorch during training

    @property
    def noise(self) -> Tensor:
        """Get the noise parameters for each fidelity level."""
        return self.noise_covar.noise

    @property
    def num_noise_params(self) -> int:
        """Get the number of noise parameters."""
        if hasattr(self.noise_covar, "noise"):
            return self.noise_covar.noise.shape[0]
        return 0

    @noise.setter
    def noise(self, value: Tensor) -> None:
        """Set the noise parameters for each fidelity level."""
        self.noise_covar.initialize(noise=value)

    @property
    def raw_noise(self) -> Tensor:
        """Get the raw (unconstrained) noise parameters."""
        return self.noise_covar.raw_noise

    @raw_noise.setter
    def raw_noise(self, value: Tensor) -> None:
        """Set the raw (unconstrained) noise parameters."""
        self.noise_covar.initialize(raw_noise=value)

    def _shaped_noise_covar(self, base_shape: torch.Size, *params: Any, **kwargs: Any):
        """Create the noise covariance matrix with proper shape."""
        # Get device and input data - handle different input formats
        input_data = None
        device = torch.device("cpu")

        # Try to extract input data from params or kwargs
        if params:
            # params[0] might be the input tensor or a list
            if isinstance(params[0], torch.Tensor):
                input_data = params[0]
                device = input_data.device
            elif isinstance(params[0], (list, tuple)) and len(params[0]) > 0:
                # If it's a list, try to get the first tensor from it
                for item in params[0]:
                    if isinstance(item, torch.Tensor):
                        input_data = item
                        device = input_data.device
                        break

        # If we still don't have input_data, try kwargs
        if input_data is None and "input" in kwargs:
            input_data = kwargs["input"]
            if hasattr(input_data, "device"):
                device = input_data.device

        # Extract fidelity indices from input data
        if input_data is not None and hasattr(input_data, "shape") and len(input_data.shape) >= 2:
            if self.is_onehot:
                fidel_indices = torch.argmax(input_data[:, self.encoded_cols], dim=1)
            else:
                fidel_indices = input_data[:, self.source_col].long()
        else:
            # Fallback: create default fidelity indices
            fidel_indices = torch.zeros(base_shape[0], dtype=torch.long, device=device)

        # Create noise covariance matrix
        noise_covar = self.noise_covar.forward(
            shape=base_shape,
            fidel_indices=fidel_indices,
            unique_fidelities=torch.arange(self.num_fidelities, device=device),
        )

        return noise_covar

    def marginal(self, function_dist: MultivariateNormal, *params: Any, **kwargs: Any) -> MultivariateNormal:
        """Compute the marginal likelihood."""
        mean, covar = function_dist.mean, function_dist.lazy_covariance_matrix
        noise_covar = self._shaped_noise_covar(mean.shape, *params, **kwargs)
        full_covar = covar + noise_covar
        return function_dist.__class__(mean, full_covar)

    def to(self, device_or_dtype):
        """Move the likelihood to the specified device or dtype."""
        super().to(device_or_dtype)
        # Also move encoded_cols to the same device if it exists
        if hasattr(device_or_dtype, "type") and device_or_dtype.type == "torch.device":
            if hasattr(self, "encoded_cols"):
                self.encoded_cols = self.encoded_cols.to(device=device_or_dtype)
        return self


class MultiNoise(_HomoskedasticNoiseBase):
    """
    Noise model for multifidelity data that allows different noise levels for each fidelity.

    Args:
        num_fidelities: Number of different fidelity levels
        noise_prior: Prior distribution for noise parameters
        noise_constraint: Constraint for noise parameters
        batch_shape: Batch shape for the noise model
    """

    def __init__(
        self, num_fidelities: int = 1, noise_prior=None, noise_constraint=None, batch_shape: torch.Size = torch.Size()
    ):
        # Set default constraint if none provided
        if noise_constraint is None:
            noise_constraint = gpytorch.constraints.GreaterThan(1e-6)

        # Set default prior if none provided
        if noise_prior is None:
            noise_prior = gpytorch.priors.LogNormalPrior(loc=0.0, scale=1.0)

        # Initialize with number of tasks equal to number of fidelities
        super().__init__(noise_prior, noise_constraint, batch_shape, num_tasks=num_fidelities)

        self.num_fidelities = num_fidelities

    def forward(
        self,
        shape: Optional[torch.Size] = None,
        fidel_indices: Union[Tensor, List[int]] = None,
        unique_fidelities: Union[Tensor, List[int]] = None,
        *params: Any,
        **kwargs: Any,
    ) -> Union[DiagLinearOperator, ConstantDiagLinearOperator]:
        """Forward pass for multifidelity noise computation.

        Args:
            shape: Shape parameter for the noise covariance
            fidel_indices: Tensor indicating fidelity level of each input data point
            unique_fidelities: Tensor of unique fidelity levels
            *params: Additional parameters
            **kwargs: Additional keyword arguments

        Returns:
            DiagLinearOperator: Diagonal noise covariance matrix with different noise for each fidelity
        """

        if fidel_indices is None or len(fidel_indices) == 0:
            raise ValueError("fidel_indices must be provided")

        if unique_fidelities is None:
            raise ValueError("unique_fidelities must be provided")

        # Ensure fidel_indices is a tensor
        if not isinstance(fidel_indices, torch.Tensor):
            fidel_indices = torch.tensor(fidel_indices, dtype=torch.long)

        if not isinstance(unique_fidelities, torch.Tensor):
            unique_fidelities = torch.tensor(unique_fidelities, dtype=torch.long)

        # Get the base noise covariance from parent class
        base_covar = super().forward(shape=shape, *params, **kwargs)

        # Get device and dtype
        device = base_covar.device
        dtype = base_covar.dtype

        # Ensure tensors are on the same device
        if fidel_indices.device != device:
            fidel_indices = fidel_indices.to(device=device)
        if unique_fidelities.device != device:
            unique_fidelities = unique_fidelities.to(device=device)

        # Create mapping from fidelity indices to noise parameter indices
        # This maps each unique fidelity to its corresponding noise parameter
        fidelity_to_noise_idx = {fid.item(): idx for idx, fid in enumerate(unique_fidelities)}

        # Extract noise values for each fidelity level
        if base_covar.dim() == 1:
            # Single batch, multiple tasks
            noise_values = base_covar
        elif base_covar.dim() == 2:
            # Multiple batches, multiple tasks
            noise_values = base_covar[0]  # Take first batch
        else:
            # Higher dimensional case
            noise_values = base_covar[0, 0]  # Take first batch, first dimension

        # Create a tensor of noise values for each data point based on their fidelity
        data_noise = torch.zeros(len(fidel_indices), dtype=dtype, device=device)

        for i, fidelity in enumerate(fidel_indices):
            noise_idx = fidelity_to_noise_idx[fidelity.item()]
            # Ensure we get a scalar value by using .item() if it's a single-element tensor
            noise_val = noise_values[noise_idx]
            if hasattr(noise_val, "item") and noise_val.numel() == 1:
                data_noise[i] = noise_val.item()
            else:
                # If it's still a tensor, take the first element
                data_noise[i] = noise_val.flatten()[0]

        # Return diagonal operator with the computed noise values
        return DiagLinearOperator(data_noise)


# Example usage (commented out for production)
"""
if __name__ == '__main__':
    # Example 1: One-hot encoded fidelity levels
    # Columns 10, 11, 12 contain one-hot encoded fidelity levels
    encoded_cols = [10, 11, 12]  # 3 fidelity levels
    likelihood = MultiLikelihood(encoded_cols=encoded_cols)
    
    print(f"Number of fidelities: {likelihood.num_fidelities}")
    print(f"One-hot encoded: {likelihood.is_onehot}")
    
    # Example 2: Single column with fidelity indicators
    # Column 10 contains fidelity indicators (0, 1, 2, etc.)
    source_col = 10
    # Option A: Without training data (will determine fidelities during forward pass)
    likelihood2a = MultiLikelihood(encoded_cols=source_col)
    # Option B: With training data (determines fidelities upfront)
    # likelihood2b = MultiLikelihood(encoded_cols=source_col, training_data=X_train)
    
    print(f"Single column case: {likelihood2a.is_onehot}")
    print(f"Source column: {likelihood2a.source_col}")
"""
