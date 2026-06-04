import torch


class StandardScaler:
    """A utility class for standardizing data by subtracting the mean and dividing by the standard deviation.

    This scaler follows a similar API to scikit-learn's StandardScaler but is implemented
    for PyTorch tensors. It computes the mean and standard deviation of the features
    along the first dimension (assuming data shape is [N, features]).

    Attributes:
        mean (torch.Tensor or None): The per-feature mean, computed in the `fit` method.
        std (torch.Tensor or None): The per-feature standard deviation, computed in the `fit` method.
    """

    def __init__(self):
        """Initializes a new instance of the StandardScaler with empty mean and std attributes."""
        self.mean = None
        self.std = None

    def fit(self, data: torch.Tensor) -> None:
        """Computes the mean and standard deviation for each feature in the given data.

        Args:
            data (torch.Tensor): The input data of shape [N, features].

        Notes:
            - If `data` has NaN or Inf values, the computed statistics may be invalid.
            - Uses `std(dim=0, correction=0)` which mimics `unbiased=False` behavior (like
              dividing by N instead of N-1 in NumPy).
            - If there's only 1 sample, std is set to 1.0 to avoid division by zero warnings.
        """
        self.mean = data.mean(dim=0, keepdim=True)
        self.std = data.std(dim=0, correction=0, keepdim=True)

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Applies standardization to the input data using the stored mean and std.

        Args:
            data (torch.Tensor): The input data of shape [N, features] to be standardized.

        Returns:
            torch.Tensor: The standardized data, where each feature has zero mean and unit variance.

        Raises:
            ValueError: If `mean` or `std` have not been set (i.e., if `fit` has not been called).
        """
        if self.mean is None or self.std is None:
            raise ValueError("StandardScaler has not been fitted. Call `fit` first.")

        # (Optional) You may want to handle the case where self.std == 0, which can lead to NaNs.
        # For instance:
        # safe_std = torch.where(self.std == 0, torch.ones_like(self.std), self.std)
        # return (data - self.mean) / safe_std

        return (data - self.mean) / self.std

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Reverts the standardization of the input data using the stored mean and std.

        Args:
            data (torch.Tensor): The standardized data of shape [N, features].

        Returns:
            torch.Tensor: The data in its original scale.

        Raises:
            ValueError: If `mean` or `std` have not been set (i.e., if `fit` has not been called).
        """
        if self.mean is None or self.std is None:
            raise ValueError("StandardScaler has not been fitted. Call `fit` first.")
        return data * self.std + self.mean


class LogScaler:
    """A utility class for standardizing data in log space.

    This scaler first applies a log(y + C) transformation to the data, then standardizes
    the log-transformed values. This is useful for heavily right-skewed distributions
    or when the coefficient of variation (std/mean) is high.

    The transformation pipeline is:
    1. Log transform: log(data + C)
    2. Standardize: (log_data - mean) / std

    The inverse transformation pipeline is:
    1. Unstandardize: log_data = standardized * std + mean
    2. Exp transform: exp(log_data) - C

    Attributes:
        mean (torch.Tensor or None): The per-feature mean of log-transformed data, computed in `fit`.
        std (torch.Tensor or None): The per-feature standard deviation of log-transformed data, computed in `fit`.
        epsilon (float): Small value used to ensure positivity when C is auto-computed. Default is 1e-8.
        C (float): Constant added to data before log transform. Auto-computed from data minimum if None.
    """

    def __init__(self, epsilon: float = 1, C: float = None):
        """Initializes a new instance of the LogScaler.

        Args:
            epsilon (float): Small value added to data before log transform.
                Default is 1e-8. Used to ensure positivity when C is auto-computed.
            C (float, optional): Constant to add to data before log transform.
                If None, will be automatically set to ensure data + C > 0 during fit.
                Default is None.
        """
        self.mean = None
        self.std = None
        self.epsilon = epsilon
        self.C = C  # Will be set during fit if None

    def fit(self, data: torch.Tensor) -> None:
        """Computes the mean and standard deviation for log-transformed data.

        Args:
            data (torch.Tensor): The input data of shape [N, features]. Should contain
                positive values (or values that become positive after adding C).

        Notes:
            - Applies log(data + C) before computing statistics.
            - If C is None, automatically sets C to ensure data + C > 0.
            - If `data` has NaN or Inf values, the computed statistics may be invalid.
            - Uses `std(dim=0, correction=0)` which mimics `unbiased=False` behavior.
        """
        # Auto-compute C if not provided: set C such that data + C > epsilon
        if self.C is None:
            data_min = data.min().item()  # Global minimum across all features
            # Set C to ensure data + C >= epsilon (i.e., C >= epsilon - min(data))
            self.C = max(self.epsilon, self.epsilon - data_min)
        
        # Validate that data + C is positive
        data_min_shifted = (data + self.C).min().item()
        if data_min_shifted <= 0:
            raise ValueError(
                f"Data minimum with C shift is <= 0: min(data + C) = {data_min_shifted} <= 0. "
                f"Consider using a larger C value (current C = {self.C})."
            )
        if data_min_shifted < self.epsilon:
            import warnings
            warnings.warn(
                f"Data minimum with C shift is less than epsilon: min(data + C) = {data_min_shifted} < {self.epsilon}. "
                f"Consider using a larger C value (current C = {self.C})."
            )
        
        # Shift data and apply log transform
        data_shifted = data + self.C
        log_data = torch.log(data_shifted)

        # Compute statistics on log-transformed data
        self.mean = log_data.mean(dim=0, keepdim=True)
        self.std = log_data.std(dim=0, correction=0, keepdim=True)

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Applies log transformation and standardization to the input data.

        Args:
            data (torch.Tensor): The input data of shape [N, features] to be transformed.

        Returns:
            torch.Tensor: The log-transformed and standardized data, where each feature
                has zero mean and unit variance in log space.

        Raises:
            ValueError: If `mean` or `std` have not been set (i.e., if `fit` has not been called).
        """
        if self.mean is None or self.std is None:
            raise ValueError("LogScaler has not been fitted. Call `fit` first.")
        if self.C is None:
            raise ValueError("LogScaler.C has not been set. Call `fit` first.")

        # Shift data using C (same C used during fit)
        data_shifted = data + self.C

        # Apply log transform
        log_data = torch.log(data_shifted)

        # Handle the case where self.std == 0, which can lead to NaNs
        zero_std_mask = self.std == 0
        if torch.any(zero_std_mask):
            import warnings
            # Handle both single-feature and multi-feature cases
            zero_std_squeezed = zero_std_mask.squeeze()
            if zero_std_squeezed.dim() == 0:
                # Single feature case
                zero_std_indices = [0] if zero_std_squeezed.item() else []
            else:
                # Multi-feature case
                zero_std_indices = torch.where(zero_std_squeezed)[0].tolist()
            warnings.warn(
                f"LogScaler: Standard deviation is zero for feature(s) {zero_std_indices}. "
                "This indicates constant values in log space. Using std=1.0 for these features to avoid division by zero."
            )
        safe_std = torch.where(zero_std_mask, torch.ones_like(self.std), self.std)

        # Standardize
        return (log_data - self.mean) / safe_std

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Reverts the log transformation and standardization.

        Args:
            data (torch.Tensor): The standardized log-transformed data of shape [N, features].

        Returns:
            torch.Tensor: The data in its original scale.

        Raises:
            ValueError: If `mean` or `std` have not been set (i.e., if `fit` has not been called).
        """
        if self.mean is None or self.std is None:
            raise ValueError("LogScaler has not been fitted. Call `fit` first.")

        # Handle the case where self.std == 0
        zero_std_mask = self.std == 0
        if torch.any(zero_std_mask):
            import warnings
            # Handle both single-feature and multi-feature cases
            zero_std_squeezed = zero_std_mask.squeeze()
            if zero_std_squeezed.dim() == 0:
                # Single feature case
                zero_std_indices = [0] if zero_std_squeezed.item() else []
            else:
                # Multi-feature case
                zero_std_indices = torch.where(zero_std_squeezed)[0].tolist()
            warnings.warn(
                f"LogScaler: Standard deviation is zero for feature(s) {zero_std_indices} during inverse transform. "
                "This indicates constant values in log space. Using std=1.0 for these features to avoid division by zero."
            )
        safe_std = torch.where(zero_std_mask, torch.ones_like(self.std), self.std)

        # Unstandardize (reverse standardization)
        log_data = data * safe_std + self.mean

        # Apply inverse log transform: exp(log_data) - C
        # This is the inverse of log(y + C)
        exp_data = torch.exp(log_data)
        return exp_data - self.C


class UniformScaler:
    """A utility class for scaling data uniformly using min-max normalization.

    This scaler follows a similar API to scikit-learn's MinMaxScaler but is implemented
    for PyTorch tensors. It computes the minimum and maximum of the features along the
    first dimension (assuming data shape is [N, features]) and scales all values to the
    specified feature range.

    Unlike StandardScaler which uses Gaussian scaling (mean 0, std 1), UniformScaler
    performs min-max scaling, ensuring all values are uniformly distributed in the
    specified range (default [0, 1], or [-1, 1] if feature_range=(-1, 1)).

    Attributes:
        min (torch.Tensor or None): The per-feature minimum, computed in the `fit` method.
        max (torch.Tensor or None): The per-feature maximum, computed in the `fit` method.
        feature_range (tuple): The desired range of transformed data (min, max). Default is (0, 1).
    """

    def __init__(self, feature_range=(0, 1), scale_to_neg_one=True):
        """Initializes a new instance of the UniformScaler.

        Args:
            feature_range (tuple): Desired range of transformed data (min, max).
                Default is (0, 1). Use (-1, 1) to scale to [-1, 1] range.
            scale_to_neg_one (bool): If True, scales to [-1, 1] instead of [0, 1].
                Overrides feature_range if provided. Default is False.
        """
        self.min = None
        self.max = None
        
        # If scale_to_neg_one is True, override feature_range
        if scale_to_neg_one:
            self.feature_range = (-1, 1)
            self.scale_min, self.scale_max = -1, 1
        else:
            self.feature_range = feature_range
            if len(feature_range) != 2:
                raise ValueError("feature_range must be a tuple of length 2 (min, max)")
            self.scale_min, self.scale_max = feature_range
            if self.scale_min >= self.scale_max:
                raise ValueError("feature_range min must be less than max")

    def fit(self, data: torch.Tensor) -> None:
        """Computes the minimum and maximum for each feature in the given data.

        Args:
            data (torch.Tensor): The input data of shape [N, features].

        Notes:
            - If `data` has NaN or Inf values, the computed statistics may be invalid.
            - If a feature has constant values (min == max), the range will be set to 1.0
              to avoid division by zero during transform.
        """
        self.min = data.min(dim=0, keepdim=True)[0]
        self.max = data.max(dim=0, keepdim=True)[0]

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Applies min-max scaling to the input data using the stored min and max.

        Args:
            data (torch.Tensor): The input data of shape [N, features] to be scaled.

        Returns:
            torch.Tensor: The scaled data, where each feature is in the range specified
                by feature_range (default [0, 1]).

        Raises:
            ValueError: If `min` or `max` have not been set (i.e., if `fit` has not been called).
        """
        if self.min is None or self.max is None:
            raise ValueError("UniformScaler has not been fitted. Call `fit` first.")

        # Handle the case where min == max (constant feature), which would lead to division by zero
        range_vals = self.max - self.min
        zero_range_mask = range_vals == 0
        if torch.any(zero_range_mask):
            import warnings
            # Handle both single-feature and multi-feature cases
            zero_range_squeezed = zero_range_mask.squeeze()
            if zero_range_squeezed.dim() == 0:
                # Single feature case
                zero_range_indices = [0] if zero_range_squeezed.item() else []
            else:
                # Multi-feature case
                zero_range_indices = torch.where(zero_range_squeezed)[0].tolist()
            warnings.warn(
                f"UniformScaler: Range is zero for feature(s) {zero_range_indices}. "
                "This indicates constant values. Using range=1.0 for these features to avoid division by zero."
            )
        safe_range = torch.where(zero_range_mask, torch.ones_like(range_vals), range_vals)

        # Apply min-max scaling to [0, 1] first: (data - min) / (max - min)
        data_scaled_01 = (data - self.min) / safe_range
        
        # Then scale to desired feature_range: scale_min + (scale_max - scale_min) * data_scaled_01
        scale_range = self.scale_max - self.scale_min
        return self.scale_min + scale_range * data_scaled_01

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Reverts the min-max scaling of the input data using the stored min and max.

        Args:
            data (torch.Tensor): The scaled data of shape [N, features] in the range
                specified by feature_range (default [0, 1]).

        Returns:
            torch.Tensor: The data in its original scale.

        Raises:
            ValueError: If `min` or `max` have not been set (i.e., if `fit` has not been called).
        """
        if self.min is None or self.max is None:
            raise ValueError("UniformScaler has not been fitted. Call `fit` first.")

        # Handle the case where min == max (constant feature)
        range_vals = self.max - self.min
        zero_range_mask = range_vals == 0
        if torch.any(zero_range_mask):
            import warnings
            # Handle both single-feature and multi-feature cases
            zero_range_squeezed = zero_range_mask.squeeze()
            if zero_range_squeezed.dim() == 0:
                # Single feature case
                zero_range_indices = [0] if zero_range_squeezed.item() else []
            else:
                # Multi-feature case
                zero_range_indices = torch.where(zero_range_squeezed)[0].tolist()
            warnings.warn(
                f"UniformScaler: Range is zero for feature(s) {zero_range_indices} during inverse transform. "
                "This indicates constant values. Using range=1.0 for these features to avoid division by zero."
            )
        safe_range = torch.where(zero_range_mask, torch.ones_like(range_vals), range_vals)

        # First, convert from feature_range back to [0, 1]: (data - scale_min) / (scale_max - scale_min)
        scale_range = self.scale_max - self.scale_min
        data_scaled_01 = (data - self.scale_min) / scale_range
        
        # Then reverse min-max scaling: data_scaled_01 * (max - min) + min
        return data_scaled_01 * safe_range + self.min


if __name__ == "__main__":
    # Example usage
    # -------------
    # Assume X_train is your training data as a torch.Tensor of shape [N, features].
    X_train = torch.randn(100, 10)  # example data

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_std = scaler.transform(X_train)

    # Later, for test data or predictions:
    X_test = torch.randn(20, 10)  # example test data
    X_test_std = scaler.transform(X_test)

    # Suppose predictions are in standardized space:
    predictions_std = torch.randn(20, 10)  # dummy model predictions
    predictions_original = scaler.inverse_transform(predictions_std)

    print("Standardized predictions:", predictions_std)
    print("Original-scale predictions:", predictions_original)

    # Example usage of LogScaler
    # --------------------------
    # LogScaler is useful for heavily right-skewed distributions
    # (e.g., buckling load, borehole flow rate)
    y_train = torch.abs(torch.randn(100, 1)) * 1000 + 100  # positive, right-skewed data

    log_scaler = LogScaler(epsilon=1e-8)
    log_scaler.fit(y_train)
    y_train_log_std = log_scaler.transform(y_train)

    # For test data or predictions:
    y_test = torch.abs(torch.randn(20, 1)) * 1000 + 100
    y_test_log_std = log_scaler.transform(y_test)

    # Suppose predictions are in standardized log space:
    predictions_log_std = torch.randn(20, 1)  # dummy model predictions
    predictions_original = log_scaler.inverse_transform(predictions_log_std)

    print("\nLogScaler example:")
    print("Original y_train range:", y_train.min().item(), "to", y_train.max().item())
    print("Log-standardized y_train range:", y_train_log_std.min().item(), "to", y_train_log_std.max().item())
    print("Reconstructed predictions range:", predictions_original.min().item(), "to", predictions_original.max().item())

    # Example usage of UniformScaler
    # --------------------------------
    # UniformScaler scales data uniformly using min-max scaling
    # instead of Gaussian scaling (mean 0, std 1)
    X_train_uniform = torch.randn(100, 10) * 10 + 5  # example data with different range

    # Default: scale to [0, 1]
    uniform_scaler_01 = UniformScaler()  # or UniformScaler(feature_range=(0, 1))
    uniform_scaler_01.fit(X_train_uniform)
    X_train_uniform_scaled_01 = uniform_scaler_01.transform(X_train_uniform)

    # Scale to [-1, 1]
    uniform_scaler_11 = UniformScaler(feature_range=(-1, 1))
    uniform_scaler_11.fit(X_train_uniform)
    X_train_uniform_scaled_11 = uniform_scaler_11.transform(X_train_uniform)

    # For test data or predictions:
    X_test_uniform = torch.randn(20, 10) * 10 + 5
    X_test_uniform_scaled_01 = uniform_scaler_01.transform(X_test_uniform)
    X_test_uniform_scaled_11 = uniform_scaler_11.transform(X_test_uniform)

    # Suppose predictions are in uniform space:
    predictions_uniform_01 = torch.rand(20, 10)  # dummy model predictions in [0, 1]
    predictions_uniform_11 = torch.rand(20, 10) * 2 - 1  # dummy model predictions in [-1, 1]
    predictions_original_01 = uniform_scaler_01.inverse_transform(predictions_uniform_01)
    predictions_original_11 = uniform_scaler_11.inverse_transform(predictions_uniform_11)

    print("\nUniformScaler [0, 1] example:")
    print("Original X_train range:", X_train_uniform.min().item(), "to", X_train_uniform.max().item())
    print("Uniform-scaled X_train range:", X_train_uniform_scaled_01.min().item(), "to", X_train_uniform_scaled_01.max().item())
    print("Reconstructed predictions range:", predictions_original_01.min().item(), "to", predictions_original_01.max().item())
    
    print("\nUniformScaler [-1, 1] example:")
    print("Original X_train range:", X_train_uniform.min().item(), "to", X_train_uniform.max().item())
    print("Uniform-scaled X_train range:", X_train_uniform_scaled_11.min().item(), "to", X_train_uniform_scaled_11.max().item())
    print("Reconstructed predictions range:", predictions_original_11.min().item(), "to", predictions_original_11.max().item())
