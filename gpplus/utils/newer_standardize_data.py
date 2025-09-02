import torch

import gpplus


class StandardizeData:
    """
    A class to handle standardizing the input (x) and output (y) data,
    using gpplus.utils.StandardScaler internally.

    Methods:
        fit_transform(x_train, y_train, x_test, y_test):
            Fits internal scalers on x_train, y_train,
            returns standardized data plus references to scalers.

        inverse_transform_y(y_std):
            Transforms standardized y-values back to the original scale.
    """

    def __init__(self, continuous_cols: list):
        """
        Args:
            continuous_cols: List of indices for continuous columns
        """
        self.x_scaler = gpplus.utils.StandardScaler()
        self.y_scaler = gpplus.utils.StandardScaler()
        self.continuous_cols = continuous_cols
        self._fitted = False  # Flag so we can check if we've called fit

    def fit_transform(self, x_train: torch.Tensor, y_train: torch.Tensor, x_test: torch.Tensor, y_test: torch.Tensor):
        """
        Fit scalers on (x_train, y_train) and transform them,
        then also transform x_test and y_test.

        Returns:
            x_train_std, y_train_std, x_test_std, y_test_std
        """
        # Validate inputs
        assert x_train.shape[1] == x_test.shape[1] # noqa: S101

        # Check for NaNs in inputs
        if torch.isnan(x_train).any() or torch.isnan(y_train).any():
            raise ValueError("Input data contains NaN values")

        # Standardize outputs
        self.y_scaler.fit(y_train)
        y_train_std = self.y_scaler.transform(y_train)
        y_test_std = self.y_scaler.transform(y_test)

        # Standardize only continuous inputs
        x_train_cont = x_train[:, self.continuous_cols]
        x_test_cont = x_test[:, self.continuous_cols]

        # Check for constant columns
        stds = x_train_cont.std(dim=0)
        if (stds == 0).any():
            print("Warning: Constant columns detected in continuous features")
            # Add small noise to prevent NaN
            noise = torch.randn_like(x_train_cont) * 1e-6
            x_train_cont = x_train_cont + noise
            stds = x_train_cont.std(dim=0)  # Recompute stds

        # Standardize with safety checks
        self.x_scaler.fit(x_train_cont)
        x_train_cont_std = self.x_scaler.transform(x_train_cont)
        x_test_cont_std = self.x_scaler.transform(x_test_cont)

        # Verify no NaNs in standardized data
        if torch.isnan(x_train_cont_std).any():
            raise RuntimeError("NaN values detected during standardization")

        # Recombine with discrete columns
        x_train_std = x_train.clone()
        x_train_std[:, self.continuous_cols] = x_train_cont_std
        x_test_std = x_test.clone()
        x_test_std[:, self.continuous_cols] = x_test_cont_std

        self._fitted = True
        return x_train_std, y_train_std, x_test_std, y_test_std

    def inverse_predictions(
        self,
        pred_mean_std: torch.Tensor,
        pred_lower_std: torch.Tensor,
        pred_upper_std: torch.Tensor,
        pred_stddev_std: torch.Tensor,
    ):
        """
        Inverse-transform standardized predictions back to original space,
        including predicted standard deviations.

        Returns:
            pred_mean, pred_lower, pred_upper, pred_std_dev
        """
        if not self._fitted:
            raise RuntimeError("Data not fitted yet. Call .fit_transform(...) first.")

        # Inverse-transform the mean and quantiles
        pred_mean = self.y_scaler.inverse_transform(pred_mean_std)
        pred_lower = self.y_scaler.inverse_transform(pred_lower_std)
        pred_upper = self.y_scaler.inverse_transform(pred_upper_std)

        # Convert predicted std from standardized space to original space
        #   In gpplus.utils.StandardScaler, the learned standard deviation
        #   is typically stored as y_scaler.std. Make sure you confirm
        #   the attribute name in that class if needed.
        pred_stddev = pred_stddev_std * self.y_scaler.std

        return pred_mean, pred_lower, pred_upper, pred_stddev
