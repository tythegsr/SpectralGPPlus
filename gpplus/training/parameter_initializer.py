from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
from torch.quasirandom import SobolEngine

from ..config import logger


class ParameterInitializer(ABC):
    """Abstract base class for parameter initializers."""

    @abstractmethod
    def __init__(self, num_runs: int, seed: int = None):
        pass

    @abstractmethod
    def setup(self, model: torch.nn.Module):
        raise NotImplementedError

    @abstractmethod
    def initialize(self, model: torch.nn.Module, run_index: int):
        raise NotImplementedError


class DefaultParameterInitializer(ParameterInitializer):
    """
    Default parameter initializer with constraint-based handling and log10-scale support.

    Features:
    - Sets log10-scale constraints directly on GPyTorch kernels for optimal performance
    - Supports different initialization strategies per parameter type
    - Simple initialization mode for MLE optimization
    - Conservative initialization to avoid numerical instability
    - Simplified logic with constraint-based parameter handling
    """

    def __init__(self, num_runs: int, seed: int = None, parameter_configs: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Initialize the default parameter initializer.

        Args:
            num_runs: Total number of initialization runs.
            seed: Random seed for reproducibility.
            parameter_configs: Optional custom parameter configurations.

        Note:
            This initializer automatically sets appropriate constraints on GPyTorch parameters
            during setup, enabling direct parameter initialization in appropriate scales.
        """
        self.num_runs = num_runs
        self.seed = seed
        self.num_params = None
        self.sobol_samples = None
        self.parameter_configs = parameter_configs or {}

    def setup(self, model: torch.nn.Module):
        """Calculate the total number of learnable parameters and precompute Sobol samples."""
        self.num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        sobol_engine = SobolEngine(dimension=self.num_params, scramble=True, seed=self.seed)
        self.sobol_samples = sobol_engine.draw(self.num_runs)

        # Set constraints on GPyTorch parameters to use appropriate scales
        self._setup_gpytorch_parameter_constraints(model)

        logger.info("Using DefaultParameterInitializer")
        logger.debug(f"Sobol samples generated: {self.sobol_samples.shape}")

    def _setup_gpytorch_parameter_constraints(self, model: torch.nn.Module):
        """Set constraints on GPyTorch parameters to use appropriate scales."""
        try:
            # Import the custom constraints
            from gpplus.utils.custom_constraints import CustomLog10Interval, CustomLog10RBFInterval
            from gpplus.utils.transforms import (
                log10_inv_transform,
                log10_rbf_inv_transform,
                log10_rbf_transform,
                log10_transform,
            )

            # Set up the log10-scale constraint for lengthscales (RBF-specific)
            lengthscale_constraint = CustomLog10RBFInterval(
                lower_bound=-7.0,  # More conservative bounds
                upper_bound=5.0,
                transform=log10_rbf_transform,
                inv_transform=log10_rbf_inv_transform,
            )

            # Set up the log10-scale constraint for noise and outputscale
            # Use more conservative bounds to avoid numerical instability
            log10_scale_constraint_outputscale = CustomLog10Interval(
                lower_bound=-7.0,  # 10^(-2) = 0.01
                upper_bound=5.0,  # 10^1 = 10
                transform=log10_transform,
                inv_transform=log10_inv_transform,
            )

            log10_scale_constraint_noise = CustomLog10Interval(
                lower_bound=-7.0,  # 10^(-2) = 0.01
                upper_bound=5.0,  # 10^1 = 10
                transform=log10_transform,
                inv_transform=log10_inv_transform,
            )

            # Find all GPyTorch modules and set appropriate constraints
            for name, module in model.named_modules():
                if "gpytorch" in module.__class__.__module__:
                    # Set lengthscale constraint
                    if hasattr(module, "raw_lengthscale") and hasattr(module, "raw_lengthscale_constraint"):
                        logger.debug(f"Found GPyTorch kernel: {name}, class: {module.__class__}")
                        old_constraint = module.raw_lengthscale_constraint

                        # Use RBF-specific constraint for RBF kernels, simple log10 for Matern kernels
                        if "MaternKernel" in module.__class__.__name__:
                            # Create a simple log10 constraint for Matern kernels
                            matern_lengthscale_constraint = CustomLog10Interval(
                                lower_bound=-7.0,
                                upper_bound=5.0,
                                transform=log10_transform,
                                inv_transform=log10_inv_transform,
                            )
                            module.raw_lengthscale_constraint = matern_lengthscale_constraint
                            logger.debug(
                                f"Set log10-scale constraint on Matern lengthscale: {name} "
                                f"(old: {old_constraint}, new: {matern_lengthscale_constraint})"
                            )
                        else:
                            # Use RBF-specific constraint for other kernels (RBF, etc.)
                            module.raw_lengthscale_constraint = lengthscale_constraint
                            logger.debug(
                                f"Set log10-scale constraint on lengthscale: {name} "
                                f"(old: {old_constraint}, new: {lengthscale_constraint})"
                            )

                    # Ensure constraints are properly registered with parameters
                    if hasattr(module, "raw_lengthscale"):
                        if "MaternKernel" in module.__class__.__name__:
                            # Use the Matern-specific constraint that was created above
                            matern_lengthscale_constraint = CustomLog10Interval(
                                lower_bound=-7.0,
                                upper_bound=5.0,
                                transform=log10_transform,
                                inv_transform=log10_inv_transform,
                            )
                            module.raw_lengthscale.constraint = matern_lengthscale_constraint
                        else:
                            module.raw_lengthscale.constraint = lengthscale_constraint
                # Set outputscale constraint for any module that has it
                if hasattr(module, "raw_outputscale") and hasattr(module, "raw_outputscale_constraint"):
                    old_constraint = module.raw_outputscale_constraint
                    # Try to replace the constraint object
                    try:
                        module.raw_outputscale_constraint = log10_scale_constraint_outputscale
                        logger.debug(
                            f"Set log10-scale constraint on outputscale: {name} "
                            f"(old: {old_constraint}, new: {log10_scale_constraint_outputscale})"
                        )
                    except Exception as e:
                        # If direct replacement fails, try to modify the constraint attributes
                        logger.debug(f"Direct constraint replacement failed for {name}: {e}")
                        if hasattr(module.raw_outputscale_constraint, "lower_bound"):
                            module.raw_outputscale_constraint.lower_bound = (
                                log10_scale_constraint_outputscale.lower_bound
                            )
                        if hasattr(module.raw_outputscale_constraint, "upper_bound"):
                            module.raw_outputscale_constraint.upper_bound = (
                                log10_scale_constraint_outputscale.upper_bound
                            )
                        if hasattr(module.raw_outputscale_constraint, "transform"):
                            module.raw_outputscale_constraint.transform = log10_scale_constraint_outputscale.transform
                        if hasattr(module.raw_outputscale_constraint, "inv_transform"):
                            module.raw_outputscale_constraint.inv_transform = (
                                log10_scale_constraint_outputscale.inv_transform
                            )
                        logger.debug(f"Modified constraint attributes for outputscale: {name}")

                # Set noise constraint (for noise_covar modules)
                # Note: likelihood.raw_noise is a property that delegates to noise_covar.raw_noise
                # The actual constraint lives in noise_covar.raw_noise_constraint
                if hasattr(module, "noise_covar") and hasattr(module.noise_covar, "raw_noise_constraint"):
                    old_constraint = module.noise_covar.raw_noise_constraint
                    module.noise_covar.raw_noise_constraint = log10_scale_constraint_noise
                    logger.debug(
                        f"Set log10-scale constraint on noise_covar: {name} "
                        f"(old: {old_constraint}, new: {log10_scale_constraint_noise})"
                    )

                if hasattr(module, "raw_outputscale"):
                    module.raw_outputscale_constraint = log10_scale_constraint_outputscale
                if hasattr(module, "noise_covar") and hasattr(module.noise_covar, "raw_noise"):
                    module.noise_covar.raw_noise_constraint = log10_scale_constraint_noise

        except Exception as e:
            logger.warning(f"Could not set parameter constraints: {e}")

    def get_parameter_type(self, name: str, param: torch.Tensor) -> str:
        """Determine the parameter type based on name and context."""
        if "projection_matrix" in name:
            return "projection_matrix"
        elif "lengthscale" in name:
            return "lengthscale"
        elif "outputscale" in name:
            return "outputscale"
        elif "raw_noise" in name or "noise" in name:
            return "noise"
        elif "weight" in name and param.dim() >= 2:
            return "weight"
        elif "bias" in name and param.dim() == 1:
            return "bias"
        elif "power" in name:
            return "power"
        else:
            return "unknown"

    def is_gpytorch_kernel_parameter(self, name: str, param: torch.Tensor, model: torch.nn.Module = None) -> bool:
        """Check if this parameter belongs to a GPyTorch kernel."""
        if model is None:
            return False

        try:
            module = self._get_module_from_name(model, name)
            if module is None:
                return False

            module_class = module.__class__
            module_module = module_class.__module__

            # Check if it's from gpytorch package
            if "gpytorch" in module_module:
                # Exclude custom kernels
                custom_kernel_classes = ["GaussianKernel"]
                if any(kernel_class in module_class.__name__ for kernel_class in custom_kernel_classes):
                    return False

                # Check for known GPyTorch kernel classes
                gpytorch_kernel_classes = [
                    "RBFKernel",
                    "MaternKernel",
                    "ScaleKernel",
                    "PeriodicKernel",
                    "LinearKernel",
                    "PolynomialKernel",
                    "GaussianLikelihood",
                ]
                return any(kernel_class in module_class.__name__ for kernel_class in gpytorch_kernel_classes)

        except Exception as e:
            logger.debug(f"Could not determine kernel type for parameter {name}: {e}")

        # Additional check: if the parameter has a constraint, it's likely GPyTorch
        if hasattr(param, "constraint") and param.constraint is not None:
            return True

        # Check for GPyTorch parameter name patterns
        if "raw_lengthscale" in name or "raw_outputscale" in name or "raw_noise" in name:
            return True

        return False

    def is_scale_kernel_parameter(self, name: str, param: torch.Tensor, model: torch.nn.Module = None) -> bool:
        """Check if this parameter belongs to a ScaleKernel."""
        if model is None:
            return False

        try:
            module = self._get_module_from_name(model, name)
            if module is None:
                return False
            return "ScaleKernel" in module.__class__.__name__
        except Exception as e:
            logger.debug(f"Could not determine if parameter {name} belongs to ScaleKernel: {e}")
        return False

    def is_base_kernel_gpytorch(self, name: str, param: torch.Tensor, model: torch.nn.Module = None) -> bool:
        """Check if this parameter belongs to a GPyTorch base kernel within a ScaleKernel."""
        if model is None:
            return False

        try:
            # Check if this is a base kernel parameter
            if "base_kernel" in name:
                # Get the base kernel module
                module = self._get_module_from_name(model, name)
                if module is None:
                    return False

                # Check if the base kernel is a GPyTorch kernel
                base_kernel_class = module.__class__
                base_kernel_module = base_kernel_class.__module__

                # Check if it's from gpytorch package
                if "gpytorch" in base_kernel_module:
                    # Exclude custom kernels
                    custom_kernel_classes = ["GaussianKernel"]
                    if any(kernel_class in base_kernel_class.__name__ for kernel_class in custom_kernel_classes):
                        return False

                    # Check for known GPyTorch kernel classes
                    gpytorch_kernel_classes = [
                        "RBFKernel",
                        "MaternKernel",
                        "PeriodicKernel",
                        "LinearKernel",
                        "PolynomialKernel",
                    ]
                    return any(kernel_class in base_kernel_class.__name__ for kernel_class in gpytorch_kernel_classes)

            # Check if this is a ScaleKernel parameter (not base_kernel)
            elif "raw_outputscale" in name or "raw_lengthscale" in name:
                # Get the ScaleKernel module
                module = self._get_module_from_name(model, name)
                if module is None:
                    return False

                # Check if it's a ScaleKernel
                if "ScaleKernel" in module.__class__.__name__:
                    # Get the base kernel
                    if hasattr(module, "base_kernel"):
                        base_kernel = module.base_kernel
                        base_kernel_class = base_kernel.__class__
                        base_kernel_module = base_kernel_class.__module__

                        # Check if it's from gpytorch package
                        if "gpytorch" in base_kernel_module:
                            # Exclude custom kernels
                            custom_kernel_classes = ["GaussianKernel"]
                            if any(
                                kernel_class in base_kernel_class.__name__ for kernel_class in custom_kernel_classes
                            ):
                                return False

                            # Check for known GPyTorch kernel classes
                            gpytorch_kernel_classes = [
                                "RBFKernel",
                                "MaternKernel",
                                "PeriodicKernel",
                                "LinearKernel",
                                "PolynomialKernel",
                            ]
                            return any(
                                kernel_class in base_kernel_class.__name__ for kernel_class in gpytorch_kernel_classes
                            )

        except Exception as e:
            logger.debug(f"Could not determine if parameter {name} belongs to GPyTorch base kernel: {e}")
        return False

    def _get_module_from_name(self, model: torch.nn.Module, param_name: str):
        """Helper method to get the module from parameter name."""
        try:
            parts = param_name.split(".")
            current_module = model

            for part in parts[:-1]:  # Exclude the parameter name itself
                if hasattr(current_module, part):
                    current_module = getattr(current_module, part)
                else:
                    return None

            return current_module
        except Exception:
            return None

    def is_combined_kernel_parameter(self, name: str, param: torch.Tensor, model: torch.nn.Module = None) -> bool:
        """Check if this parameter belongs to a CombinedKernel_MVMF."""
        if model is None:
            return False

        try:
            module = self._get_module_from_name(model, name)
            if module is None:
                return False

            # Check if the parameter is within a CombinedKernel_MVMF structure
            # Look for patterns like covar_module.cont_kernel.lengthscale
            if "covar_module" in name and ("cont_kernel" in name or "cat_kernel" in name or "source_kernel" in name):
                return True

            # Check if the module is a CombinedKernel_MVMF
            module_class = module.__class__
            return "CombinedKernel" in module_class.__name__

        except Exception as e:
            logger.debug(f"Could not determine if parameter {name} is from CombinedKernel: {e}")
            return False

    def _get_combined_kernel_config(
        self, name: str, param: torch.Tensor, model: torch.nn.Module = None
    ) -> Dict[str, Any]:
        """Get initialization configuration for CombinedKernel_MVMF parameters."""
        param_type = self.get_parameter_type(name, param)

        logger.debug(f"CombinedKernel config for {name}: type={param_type}, shape={param.shape}")

        # Special handling for different sub-kernels in CombinedKernel_MVMF
        if "cont_kernel.lengthscale" in name:
            # Continuous kernel lengthscale - should be trainable
            # Use more conservative initialization to avoid numerical instability
            is_ard = param.dim() == 2 and param.shape[1] > 1
            config = {
                "method": "normal",
                "mean": -2.0 if is_ard else 0.0,  # More conservative mean
                "std": 1.5 if is_ard else 0.3,  # Smaller std for stability
                "description": f"CombinedKernel cont_kernel {'ARD' if is_ard else 'single'} lengthscale (conservative)",
                "is_gpytorch": False,
                "is_scale_kernel": False,
            }
            logger.debug(f"CombinedKernel cont_kernel config: {config}")
            return config
        elif "cat_kernel.lengthscale" in name:
            # Categorical kernel lengthscale - should be fixed (requires_grad=False)
            return {
                "method": "constant",
                "value": 0.0,
                "description": "CombinedKernel cat_kernel lengthscale (fixed)",
                "is_gpytorch": False,
                "is_scale_kernel": False,
            }
        elif "source_kernel.lengthscale" in name:
            # Source kernel lengthscale - should be fixed (requires_grad=False)
            return {
                "method": "constant",
                "value": 0.0,
                "description": "CombinedKernel source_kernel lengthscale (fixed)",
                "is_gpytorch": False,
                "is_scale_kernel": False,
            }
        elif "outputscale" in name and "covar_module" in name:
            # Main outputscale for CombinedKernel_MVMF
            return {
                "method": "normal",
                "mean": -1.0,
                "std": 0.5,
                "description": "CombinedKernel_MVMF outputscale (log scale)",
                "is_gpytorch": False,
                "is_scale_kernel": False,
            }
        else:
            # Fallback to default handling
            return {
                "method": "normal",
                "mean": 0.0,
                "std": 0.1,
                "description": f"CombinedKernel {param_type} parameter (fallback)",
                "is_gpytorch": False,
                "is_scale_kernel": False,
            }

    def get_initialization_config(
        self, name: str, param: torch.Tensor, model: torch.nn.Module = None
    ) -> Dict[str, Any]:
        """Get initialization configuration based on parameter name and constraints."""
        param_type = self.get_parameter_type(name, param)
        is_scale_kernel = self.is_scale_kernel_parameter(name, param, model)

        # Check if it's a GPyTorch kernel (either direct or base kernel in ScaleKernel)
        is_gpytorch = self.is_gpytorch_kernel_parameter(name, param, model) or self.is_base_kernel_gpytorch(
            name, param, model
        )

        # Special handling for CombinedKernel_MVMF parameters
        if self.is_combined_kernel_parameter(name, param, model):
            logger.debug(f"Using CombinedKernel config for {name}")
            config = self._get_combined_kernel_config(name, param, model)
            logger.debug(f"CombinedKernel config for {name}: {config}")
            return config

        # Check for custom configuration first
        if param_type in self.parameter_configs:
            config = self.parameter_configs[param_type].copy()
            config["description"] = f"{param_type} parameter (custom config)"
            config["is_gpytorch"] = is_gpytorch
            config["is_scale_kernel"] = is_scale_kernel
            return config

        # Default configurations for each parameter type
        if param_type == "projection_matrix":
            # Try to find the initialization type from the specific module
            init_type = "orthogonal"  # default
            init_std = 0.1  # default value

            # Look for the parameter in the model's modules
            if hasattr(self, "_current_model"):
                # Find the specific module that contains this parameter
                for module_name, module in self._current_model.named_modules():
                    if hasattr(module, "_param_init_types") and "projection_matrix" in module._param_init_types:
                        # Check if this parameter name starts with the module name followed by a dot
                        if name.startswith(module_name + "."):
                            init_type = module._param_init_types["projection_matrix"]
                            # Get init_std from the module if available
                            if (
                                hasattr(module, "_param_init_params")
                                and "projection_matrix" in module._param_init_params
                            ):
                                init_std = module._param_init_params["projection_matrix"]["init_std"]
                            else:
                                init_std = 0.1  # default
                            break

            if init_type == "orthogonal":
                return {
                    "method": "orthogonal_matrix",
                    "gain": init_std,
                    "description": "Matrix encoder projection matrix",
                    "is_gpytorch": False,
                    "is_scale_kernel": False,
                }
            elif init_type == "normal":
                return {
                    "method": "normal_matrix",
                    "mean": 0.0,
                    "std": init_std,
                    "description": "Matrix encoder projection matrix",
                    "is_gpytorch": False,
                    "is_scale_kernel": False,
                }
            elif init_type == "uniform":
                return {
                    "method": "uniform_matrix",
                    "lower": -init_std,
                    "upper": init_std,
                    "description": "Matrix encoder projection matrix",
                    "is_gpytorch": False,
                    "is_scale_kernel": False,
                }
            else:
                # Default to orthogonal if unknown type
                return {
                    "method": "orthogonal_matrix",
                    "gain": init_std,
                    "description": "Matrix encoder projection matrix",
                    "is_gpytorch": False,
                    "is_scale_kernel": False,
                }
        elif param_type == "lengthscale":
            is_ard = param.dim() == 2 and param.shape[1] > 1
            if is_gpytorch:
                # This is a GPyTorch lengthscale (constraint handles omega-scale conversion)
                return {
                    "method": "normal",
                    "mean": -2.0,  # More conservative bounds (10^(-2) = 0.01)
                    "std": 1.5,  # 10^1 = 10
                    "description": f"GPyTorch {'ARD' if is_ard else 'single'} lengthscale (omega-scale)",
                    "is_gpytorch": True,
                    "is_scale_kernel": is_scale_kernel,
                }
            else:
                # This is a custom kernel lengthscale
                return {
                    "method": "normal",
                    "mean": -2.0 if is_ard else 1.0,
                    "std": 1.5 if is_ard else 0.5,
                    "description": f"Custom {'ARD' if is_ard else 'single'} lengthscale (actual space)",
                    "is_gpytorch": False,
                    "is_scale_kernel": False,
                }
        elif param_type == "outputscale":
            return {
                "method": "normal",
                "mean": -1.0,  # More conservative bounds (10^(-1) = 0.1)
                "std": 0.5,  # 10^0.5 = 3.16
                "description": f"{'ScaleKernel' if is_scale_kernel else 'Custom'} outputscale (log scale)",
                "is_gpytorch": is_gpytorch,
                "is_scale_kernel": is_scale_kernel,
            }
        elif param_type == "noise" or "raw_noise" in name:
            return {
                "method": "normal",
                "mean": -3.0,  # More conservative mean (10^(-2) = 0.01)
                "std": 1.0,  # Smaller std for stability
                "description": f"{'GPyTorch' if is_gpytorch else 'Custom'} noise (log scale, conservative)",
                "is_gpytorch": is_gpytorch,
                "is_scale_kernel": False,
            }
        elif param_type == "weight":
            fan_in, fan_out = param.size(1), param.size(0)
            limit = torch.sqrt(torch.tensor(2.0 / (fan_in + fan_out)))
            return {
                "method": "xavier_uniform",
                "limit": limit.item(),
                "description": f"Neural network weight ({fan_in}->{fan_out})",
                "is_gpytorch": False,
                "is_scale_kernel": False,
            }
        elif param_type == "bias":
            return {
                "method": "zeros",
                "description": "Neural network bias",
                "is_gpytorch": False,
                "is_scale_kernel": False,
            }
        elif param_type == "power":
            return {
                "method": "normal",
                "mean": 1.5,
                "std": 0.3,
                "description": "Power kernel parameter",
                "is_gpytorch": False,
                "is_scale_kernel": False,
            }
        else:
            return {
                "method": "normal",
                "mean": 0.0,
                "std": 0.1,
                "description": "Unknown parameter (conservative)",
                "is_gpytorch": False,
                "is_scale_kernel": False,
            }

    def _generate_normal_samples(self, sample: torch.Tensor, mean: float, std: float) -> torch.Tensor:
        """Generate normal samples using inverse CDF from Sobol samples."""
        # Use inverse CDF of normal distribution directly
        # This is more efficient and numerically stable than Box-Muller
        z = torch.erfinv(2.0 * sample - 1.0) * torch.sqrt(torch.tensor(2.0))
        return mean + std * z

    def initialize_parameter(
        self,
        param: torch.Tensor,
        sample: torch.Tensor,
        config: Dict[str, Any],
        name: str = "",
        model: torch.nn.Module = None,
    ):
        """Initialize a single parameter based on the configuration and constraints."""
        method = config["method"]

        if method == "orthogonal_matrix":
            # Use PyTorch's orthogonal initialization
            torch.nn.init.orthogonal_(
                param, gain=config.get("gain", 1.0), generator=torch.Generator().manual_seed(self.seed)
            )

        elif method == "normal_matrix":
            # Use PyTorch's orthogonal initialization
            torch.nn.init.normal_(
                param,
                mean=config.get("mean", 0.0),
                std=config.get("std", 0.1),
                generator=torch.Generator().manual_seed(self.seed),
            )

        elif method == "uniform_matrix":
            # Use PyTorch's uniform initialization
            torch.nn.init.uniform_(
                param,
                a=config.get("lower", -0.1),
                b=config.get("upper", 0.1),
                generator=torch.Generator().manual_seed(self.seed),
            )

        if method == "orthogonal":
            torch.nn.init.orthogonal_(param, gain=config.get("gain", 1.0))

        elif method == "normal_matrix":
            # Use PyTorch's orthogonal initialization
            torch.nn.init.normal_(
                param,
                mean=config.get("mean", 0.0),
                std=config.get("std", 0.1),
                generator=torch.Generator().manual_seed(self.seed),
            )

        elif method == "uniform_matrix":
            # Use PyTorch's uniform initialization
            torch.nn.init.uniform_(
                param,
                a=config.get("lower", -0.1),
                b=config.get("upper", 0.1),
                generator=torch.Generator().manual_seed(self.seed),
            )

        elif method == "xavier_uniform":
            limit = config.get("limit", 1.0)
            param.data = (sample * 2 - 1) * limit

        elif method == "uniform":
            lower = config.get("lower", -1.0)
            upper = config.get("upper", 1.0)
            raw_value = lower + (upper - lower) * sample
            param.data = raw_value

        elif method == "normal":
            mean = config.get("mean", 0.0)
            std = config.get("std", 1.0)
            raw_value = self._generate_normal_samples(sample, mean, std)
            # All parameters use direct initialization (constraints handle conversions)
            param.data = raw_value

            # Ensure parameter is within reasonable bounds for numerical stability
            if hasattr(param, "constraint") and param.constraint is not None:
                # Clamp the raw value to be within constraint bounds
                if hasattr(param.constraint, "lower_bound") and hasattr(param.constraint, "upper_bound"):
                    param.data = torch.clamp(param.data, param.constraint.lower_bound, param.constraint.upper_bound)

            logger.debug(f"Direct initialization: {name} = {raw_value} (constraint handles conversion)")

        elif method == "zeros":
            torch.nn.init.zeros_(param)

        elif method == "constant":
            value = config.get("value", 0.0)
            param.data = torch.full_like(param, value)

        elif method == "skip":
            pass

        else:
            # Fallback to normal with conservative parameters
            raw_value = 0.1 * (sample * 2 - 1)
            param.data = raw_value

    def initialize(self, model: torch.nn.Module, run_index: int):
        """Initialize the model parameters for a specific run using improved configuration."""
        idx = 0

        with torch.no_grad():
            all_params = list(model.named_parameters())
            for i, (name, param) in enumerate(all_params):
                param_length = param.numel()

                # Skip parameters with zero elements or non-learnable parameters
                if param_length == 0 or not param.requires_grad:
                    logger.debug(f"Skipping parameter: {name}")
                    continue

                # Get initialization configuration
                config = self.get_initialization_config(name, param, model)

                # Slice the sobol_samples for the current parameter
                sample = self.sobol_samples[run_index, idx : idx + param_length]
                sample = sample.reshape(param.shape)
                sample = sample.to(param.device)

                # Initialize the parameter
                old_value = param.data.clone()
                self.initialize_parameter(param, sample, config, name, model)
                new_value = param.data.clone()

                logger.debug(
                    f"Initialized {name}: {config['description']} (shape={param.shape}, method={config['method']})"
                )
                logger.debug(f"  Old value: {old_value}")
                logger.debug(f"  New value: {new_value}")

                # Special debug for key parameters
                if "cont_kernel.lengthscale" in name or "raw_noise" in name:
                    logger.info(f"KEY PARAMETER INITIALIZED: {name}")
                    logger.info(f"  Method: {config['method']}")
                    logger.info(f"  Old: {old_value}")
                    logger.info(f"  New: {new_value}")
                    logger.info(f"  Config: {config}")

                idx += param_length

        logger.info(f"Model parameters initialized with run #{run_index} (improved method)")