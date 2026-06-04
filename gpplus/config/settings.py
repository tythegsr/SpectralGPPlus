"""
Global settings for GPyTorch and linear_operator configuration.

This module provides a centralized way to configure GPyTorch and linear_operator
settings that are used throughout the library. Settings can be configured at the
beginning of a script and will be applied automatically during training.
"""


class GPSettings:
    """
    Global settings for GPyTorch and linear_operator.

    These settings control numerical behavior and performance characteristics
    of Gaussian Process operations. Configure these at the beginning of your
    script to customize behavior across all training operations.

    Attributes:
        max_cholesky_size (int): Maximum size of matrices to use Cholesky
            decomposition for. For larger matrices, conjugate gradient (CG)
            methods will be used instead. Default: 10_000.
        cg_tolerance (float): Tolerance for conjugate gradient convergence.
            Default: 5e-3.
        max_cg_iterations (int): Maximum number of conjugate gradient iterations.
            Default: 2000.
    """

    def __init__(
        self,
        max_cholesky_size: int = 10_000,
        cg_tolerance: float = 5e-3,
        max_cg_iterations: int = 2000,
    ):
        """
        Initialize GPSettings.

        Parameters:
            max_cholesky_size (int): Maximum size of matrices to use Cholesky
                decomposition for. Default: 10_000.
            cg_tolerance (float): Tolerance for conjugate gradient convergence.
                Default: 5e-3.
            max_cg_iterations (int): Maximum number of conjugate gradient iterations.
                Default: 2000.
        """
        self.max_cholesky_size = max_cholesky_size
        self.cg_tolerance = cg_tolerance
        self.max_cg_iterations = max_cg_iterations

    def apply(self):
        """
        Apply the current settings to GPyTorch and linear_operator.

        This method should be called at the beginning of worker processes
        to ensure settings are properly configured.
        """
        from gpytorch.settings import max_cholesky_size
        from linear_operator.settings import cg_tolerance, max_cg_iterations

        max_cholesky_size._global_value = self.max_cholesky_size
        cg_tolerance._global_value = self.cg_tolerance
        max_cg_iterations._global_value = self.max_cg_iterations


# Global settings instance
_global_settings = GPSettings()


def get_settings() -> GPSettings:
    """
    Get the global GPSettings instance.

    Returns:
        GPSettings: The global settings instance.
    """
    return _global_settings


def configure_settings(
    max_cholesky_size: int = None,
    cg_tolerance: float = None,
    max_cg_iterations: int = None,
):
    """
    Configure the global GPSettings.

    This is a convenience function to update the global settings without
    needing to access the settings instance directly.

    Parameters:
        max_cholesky_size (int, optional): Maximum size of matrices to use
            Cholesky decomposition for.
        cg_tolerance (float, optional): Tolerance for conjugate gradient convergence.
        max_cg_iterations (int, optional): Maximum number of conjugate gradient iterations.

    Example:
        >>> from gpplus.config import configure_settings
        >>> configure_settings(max_cholesky_size=5000, cg_tolerance=1e-3)
    """
    if max_cholesky_size is not None:
        _global_settings.max_cholesky_size = max_cholesky_size
    if cg_tolerance is not None:
        _global_settings.cg_tolerance = cg_tolerance
    if max_cg_iterations is not None:
        _global_settings.max_cg_iterations = max_cg_iterations
        