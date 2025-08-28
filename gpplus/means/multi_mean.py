import torch
import torch.nn as nn
from typing import Optional
from torch import Tensor

class MultipleMean(nn.Module):
    def __init__(
        self,
        means=None,
        encoded_cols=None,
        training_data: Optional[Tensor] = None
    ):
        """
        A mean function that can specify multiple mean modules for different fidelity sources
        (e.g., multi-fidelity or grouped data). Each source can have its own mean function.
        
        This class now works as a single mean_module that internally handles the multi-mean logic,
        eliminating the need for separate mean_module_0, mean_module_1, etc. in the GPR model.
        
        Args:
            means (list[gpytorch.means.Mean] or gpytorch.means.Mean, optional):
                Either a single mean module or a list of mean modules. If not provided:
                - Uses ZeroMean if num_sources = 1
                - Uses ConstantMean (with NormalPrior on constant) for multiple sources.
        
            encoded_cols (Union[list, int, numpy.ndarray, torch.Tensor], optional):
                Either:
                - List/array of column indices for one-hot encoded fidelity levels, OR
                - Single integer indicating the column index containing fidelity indicators
                If None, assumes single source (no categorical distinction).
            training_data (torch.Tensor, optional):
                Training data tensor to determine number of sources for single column case.
                If provided, will count unique values in the specified column upfront.
        
        Attributes:
            encoded_cols: Either list of one-hot encoded columns or single source column index
            num_sources (int): Number of fidelity sources. Defaults to 1 if encoded_cols is None.
            means (list): List of mean modules, one per source.
        
        Behavior:
            - If multiple sources are provided via encoded_cols, ensures there is one mean per source.
            - If fewer means are provided than sources, duplicates the last mean for remaining sources.
            - Validates number of means equals number of sources.
        
        Methods:
            forward(x: torch.Tensor) -> torch.Tensor:
                Computes the mean for each input point, selecting the appropriate mean function based on source index.
                If single source, uses the only mean module.
        
            register_to(model):
                Registers each individual mean module (e.g., "mean_module_0", "mean_module_1", ...) to the given model.
                Note: This is now optional since the class works as a single mean_module.
        
        Example:
            >>> # One-hot encoded case
            >>> encoded_cols = [10, 11, 12]  # Columns 10, 11, 12 contain one-hot encoded sources
            >>> means = [gpytorch.means.ConstantMean()] * 3
            >>> multi_mean = MultipleMean(means=means, encoded_cols=encoded_cols)
            
            >>> # Single column case
            >>> encoded_cols = 10  # Column 10 contains source indicators (0, 1, 2, etc.)
            >>> means = [gpytorch.means.ConstantMean()] * 3
            >>> # Option A: Without training data (will determine sources during forward pass)
            >>> multi_mean = MultipleMean(means=means, encoded_cols=encoded_cols)
            >>> # Option B: With training data (determines sources upfront)
            >>> # multi_mean = MultipleMean(means=means, encoded_cols=encoded_cols, training_data=X_train)
        """

        super().__init__()

        # Handle both single column index and one-hot encoded columns
        if encoded_cols is None:
            self.is_onehot = False
            self._source_col = None
            self._encoded_cols = None  # No encoded columns for single source
            self.num_sources = 1
        elif isinstance(encoded_cols, int):
            # Single column index - not one-hot encoded
            self.is_onehot = False
            self._source_col = encoded_cols
            # Check training data if provided to determine number of sources upfront
            if training_data is not None and hasattr(training_data, 'shape') and len(training_data.shape) >= 2:
                # Extract the source column and count unique values
                source_values = training_data[:, encoded_cols].long()
                unique_sources = torch.unique(source_values)
                self.num_sources = len(unique_sources)
                self._encoded_cols = [encoded_cols]  # Store as list for compatibility
                print(f"Detected {self.num_sources} sources from training data in column {encoded_cols}")
            else:
                # We'll need to determine num_sources from data during forward pass
                self.num_sources = None
                self._encoded_cols = [encoded_cols]  # Store as list for compatibility
        else:
            # One-hot encoded columns - handle numpy arrays, tensors, lists, etc.
            self.is_onehot = True
            self._source_col = None  # No single source column for one-hot case
            # Convert to list if it's not already
            if hasattr(encoded_cols, '__len__') and not isinstance(encoded_cols, (str, bytes)):
                # It's an array-like object (numpy array, tensor, list, etc.)
                encoded_cols = list(encoded_cols)
            else:
                # Fallback: wrap in list
                encoded_cols = [encoded_cols]
            self._encoded_cols = encoded_cols
            self.num_sources = len(encoded_cols)
        
        # Set default means if not provided
        if means is None:
            from gpytorch.means import ZeroMean, ConstantMean
            from gpytorch.priors import NormalPrior
            means = [ZeroMean()]
            if self.num_sources and self.num_sources > 1:
                means += [ConstantMean(constant_prior=NormalPrior(0.0, 1.0))]
        if not isinstance(means, list):
            means = [means]
            
        # For single column case, we'll need to handle this during forward pass
        # For now, create a reasonable default
        if self.num_sources is None:
            # Single column case without training data - use a reasonable default
            # This will be updated during the first forward pass
            if len(means) < 2:
                means += [means[-1]] * (2 - len(means))
            self.num_sources = len(means)
        elif len(means) < self.num_sources:
            means += [means[-1]] * (self.num_sources - len(means))
        
        assert len(means) == self.num_sources, (
            f"Expected {self.num_sources} means, but got {len(means)}"
            )
    
        # Store the means internally but don't register them as submodules
        # This allows the class to work as a single mean_module
        self.means = means
        
        # Create internal mean modules with names mean0, mean1, mean2, etc.
        for i, m in enumerate(means):
            self.add_module(f"mean{i}", m)
        
        if "mean_module" in self._modules:
            del self._modules["mean_module"]
    
    @property
    def encoded_cols(self):
        """Get the encoded columns, ensuring backward compatibility."""
        if hasattr(self, '_encoded_cols'):
            return self._encoded_cols
        return None
    
    @encoded_cols.setter
    def encoded_cols(self, value):
        """Set the encoded columns."""
        self._encoded_cols = value
    
    @property
    def source_col(self):
        """Get the source column, ensuring backward compatibility."""
        if hasattr(self, '_source_col'):
            return self._source_col
        return None
    
    @source_col.setter
    def source_col(self, value):
        """Set the source column."""
        self._source_col = value
            
    def forward(self, x):
        """
        Forward pass that internally handles the multi-mean logic.
        This allows the class to be used as a single mean_module in the GPR model.
        """
        
        if self.is_onehot:
            # One-hot encoded case
            source_ids_onehot = x[:, self.encoded_cols]
            source_ids = source_ids_onehot.argmax(dim=-1)
        else:
            # Single column case
            if self.source_col is not None:
                source_ids = x[:, self.source_col].long()
                # Update num_sources if this is the first time we've seen the data
                if self.num_sources is None:
                    unique_sources = torch.unique(source_ids)
                    self.num_sources = len(unique_sources)
                    # Ensure we have enough means
                    while len(self.means) < self.num_sources:
                        self.means.append(self.means[-1])
                        self.add_module(f"mean{len(self.means)-1}", self.means[-1])
            else:
                # Single source case
                return self.means[0](x)
        
        output = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        for i in range(self.num_sources):
            mask = (source_ids == i)
            if mask.any():
                mean_i = getattr(self, f"mean{i}")
                output[mask] = mean_i(x[mask])
        return output
 
    def register_to(self, model):
        """
        Optional method to register individual mean modules to the model.
        This is now optional since the class works as a single mean_module.
        """
        # Use the actual number of sources (may have been updated during forward pass)
        num_sources = self.num_sources if self.num_sources is not None else len(self.means)
        for i in range(num_sources):
            setattr(model, f"mean_module_{i}", getattr(self, f"mean{i}"))

        