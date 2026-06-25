from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
from torch.quasirandom import SobolEngine

from ..config import logger


class ParameterInitializer(ABC):
    """Abstract base class for parameter initializers."""

    @abstractmethod
    def __init__(self, num_inits: int, seed: int = None):
        pass

    @abstractmethod
    def setup(self, model: torch.nn.Module):
        raise NotImplementedError

    @abstractmethod
    def initialize(self, model: torch.nn.Module, run_index: int):
        raise NotImplementedError


class DefaultParameterInitializer(ParameterInitializer):
    """
    Parameter initializer that looks at parameter names to determine initialization strategies.

    Features:
    - Parameter type detection based on parameter names only
    - Conservative initialization values for numerical stability
    - Allows the user to specify custom initialization strategies for specific parameters.
    """

    def __init__(self, num_inits: int, seed: int = None, parameter_configs: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Initialize the default parameter initializer.

        Args:
            num_inits: Total number of initialization runs.
            seed: Random seed for reproducibility.
            parameter_configs: Optional custom parameter configurations.

        Note:
            This initializer looks at parameter names to determine initialization strategies.
            All constraints are handled by the model's kernel and likelihood classes.
        """
        self.num_inits = num_inits
        self.seed = seed
        self.num_params = None
        self.sobol_samples = None
        self.parameter_configs = parameter_configs or {}

    def setup(self, model: torch.nn.Module):
        """
        Calculate the total number of learnable parameters and precompute Sobol samples.

        Excludes '.weight' and '.bias' parameters from Sobol sampling count,
        as these are initialized separately (Xavier uniform for weights, zeros for biases).
        """
        # Count only parameters that will use Sobol samples (exclude .weight and .bias)
        self.num_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad and ".weight" not in name and ".bias" not in name:
                self.num_params += param.numel()

        sobol_engine = SobolEngine(dimension=self.num_params, scramble=True, seed=self.seed)
        self.sobol_samples = sobol_engine.draw(self.num_inits)

        logger.info("Using DefaultParameterInitializer")
        logger.debug(f"Sobol samples generated: {self.sobol_samples.shape}")
        logger.info("All constraints are now built into kernel and likelihood classes - no manual setup needed")
        logger.debug("Excluding .weight and .bias parameters from Sobol sampling (initialized separately)")

    def get_parameter_type(self, name: str, param: torch.Tensor) -> str:
        """Determine the parameter type based on parameter name only."""
        if "projection_matrix" in name:
            return "projection_matrix"
        elif "raw_lengthscale" in name:
            return "raw_lengthscale"
        elif "raw_outputscale" in name:
            return "raw_outputscale"
        elif "raw_noise" in name:
            return "raw_noise"
        elif "weight" in name and param.dim() >= 2:
            return "weight"
        elif "bias" in name and param.dim() == 1:
            return "bias"
        elif "power" in name:
            return "power"
        elif "period" in name:
            return "period"
        elif "constant" in name:
            return "constant"
        else:
            return "unknown"

    def get_initialization_config(
        self, name: str, param: torch.Tensor, model: torch.nn.Module = None
    ) -> Dict[str, Any]:
        """Get initialization configuration based on parameter name only."""
        param_type = self.get_parameter_type(name, param)

        # Check for custom configuration first
        if param_type in self.parameter_configs:
            config = self.parameter_configs[param_type].copy()
            config["description"] = f"{param_type} parameter (custom config)"
            return config

        # Default configurations based on parameter type
        if param_type == "raw_lengthscale":
            is_ard = param.dim() == 2 and param.shape[1] > 1
            return {
                "method": "normal",
                "mean": -2.0,
                "std": 2.0,
                "description": f"Lengthscale parameter {'(ARD)' if is_ard else '(single)'} - log scale",
            }
        elif param_type == "raw_outputscale":
            return {
                "method": "normal",
                "mean": -2.0,
                "std": 0.5,
                "description": "Outputscale parameter - log scale",
            }
        elif param_type == "raw_noise":
            return {
                "method": "uniform",
                "lower": -7.0,
                "upper": -1.0,
                "description": "Noise parameter - uniform scale",
            }
        elif param_type == "constant":
            return {
                "method": "normal",
                "mean": 0.0,
                "std": 1.0,
                "description": "Mean constant parameter",
            }
        elif param_type == "weight":
            fan_in, fan_out = param.size(1), param.size(0)
            limit = torch.sqrt(torch.tensor(2.0 / (fan_in + fan_out)))
            return {
                "method": "xavier_uniform",
                "limit": limit.item(),
                "description": f"Neural network weight ({fan_in}->{fan_out})",
            }
        elif param_type == "bias":
            return {
                "method": "zeros",
                "description": "Neural network bias",
            }
        elif param_type == "power":
            return {
                "method": "uniform",
                "lower": 1.0,
                "upper": 2.0,
                "description": "Power kernel parameter (uniform)",
            }
        elif param_type == "period":
            is_ard = param.dim() == 2 and param.shape[1] > 1
            return {
                "method": "normal",
                "mean": 0.0,
                "std": 1.0,
                "description": f"Period parameter {'(ARD)' if is_ard else '(single)'} - log10 scale",
            }
        elif param_type == "projection_matrix":
            # Try to find the initialization type from the specific module
            init_type = "orthogonal"
            init_std = 0.1

            # Look for the parameter in the model's modules
            if model is not None:
                # Find the specific module that contains this parameter
                for module_name, module in model.named_modules():
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
                                init_std = 0.1
                            break

            if init_type == "orthogonal":
                return {
                    "method": "orthogonal_matrix",
                    "gain": init_std,
                    "description": "Matrix encoder projection matrix (orthogonal)",
                }
            elif init_type == "normal":
                return {
                    "method": "normal_matrix",
                    "mean": 0.0,
                    "std": init_std,
                    "description": "Matrix encoder projection matrix (normal)",
                }
            elif init_type == "uniform":
                return {
                    "method": "uniform_matrix",
                    "lower": -init_std,
                    "upper": init_std,
                    "description": "Matrix encoder projection matrix (uniform)",
                }
            else:
                # Default to orthogonal if unknown type
                return {
                    "method": "orthogonal_matrix",
                    "gain": init_std,
                    "description": "Matrix encoder projection matrix (default orthogonal)",
                }
        else:
            return {
                "method": "orthogonal_matrix",
                "gain": 0.1,
                "description": "Unknown parameter",
            }

    def _generate_normal_samples(self, sample: torch.Tensor, mean: float, std: float) -> torch.Tensor:
        """Generate normal samples using inverse CDF from Sobol samples."""
        # Use inverse CDF of normal distribution directly
        # Ensure all operations maintain the same dtype as the input sample
        z = torch.erfinv(2.0 * sample - 1.0) * torch.sqrt(torch.tensor(2.0, dtype=sample.dtype, device=sample.device))
        return mean + std * z

    def initialize_parameter(
        self,
        param: torch.Tensor,
        sample: torch.Tensor,
        config: Dict[str, Any],
        name: str = "",
        model: torch.nn.Module = None,
        run_index: int = 0,
    ):
        """Initialize a single parameter based on the configuration and constraints."""
        method = config["method"]

        if method == "orthogonal_matrix":
            # Use PyTorch's orthogonal initialization
            # Create a temporary tensor with the correct dtype, then copy to param
            temp_param = torch.empty_like(param, dtype=param.dtype, device=param.device)
            try:
                # Use run_index to generate different seeds for each initialization
                generator_seed = (self.seed + run_index * 1000) if self.seed is not None else (run_index * 1000)
                torch.nn.init.orthogonal_(
                    temp_param, gain=config.get("gain", 1.0), generator=torch.Generator().manual_seed(generator_seed)
                )
                # Check for NaN after initialization
                if torch.isnan(temp_param).any():
                    logger.error(f"NaN detected in orthogonal initialization for {name}")
                    logger.error(f"temp_param: {temp_param}")
                    logger.error(f"param shape: {param.shape}, dtype: {param.dtype}")
                param.data = temp_param
                logger.debug(f"Orthogonal initialization successful for {name} with seed {generator_seed}")
            except Exception as e:
                logger.error(f"Orthogonal initialization failed for {name}: {e}")
                # Fallback to normal initialization
                torch.nn.init.normal_(temp_param, mean=0.0, std=0.1)
                param.data = temp_param

        elif method == "orthogonal":
            torch.nn.init.orthogonal_(param, gain=config.get("gain", 1.0))

        elif method == "normal_matrix":
            # Use run_index to generate different seeds for each initialization
            generator_seed = (self.seed + run_index * 1000) if self.seed is not None else (run_index * 1000)
            torch.nn.init.normal_(
                param,
                mean=config.get("mean", 0.0),
                std=config.get("std", 0.1),
                generator=torch.Generator().manual_seed(generator_seed),
            )

        elif method == "uniform_matrix":
            # Use run_index to generate different seeds for each initialization
            generator_seed = (self.seed + run_index * 1000) if self.seed is not None else (run_index * 1000)
            torch.nn.init.uniform_(
                param,
                a=config.get("lower", -0.1),
                b=config.get("upper", 0.1),
                generator=torch.Generator().manual_seed(generator_seed),
            )

        elif method == "xavier_uniform":
            # Use PyTorch's xavier_uniform initialization directly
            # Use run_index to generate different seeds for each initialization
            generator_seed = (self.seed + run_index) if self.seed is not None else run_index
            g = torch.Generator().manual_seed(generator_seed)
            torch.nn.init.xavier_uniform_(param, generator=g)
            logger.debug(f"Initialized weight parameter '{name}' with Xavier uniform (seed={generator_seed})")

        elif method == "uniform":
            lower = config.get("lower", -6.0)
            upper = config.get("upper", 3.0)
            raw_value = lower + (upper - lower) * sample
            param.data = raw_value.to(dtype=param.dtype)

        elif method == "normal":
            mean = config.get("mean", -2.0)
            std = config.get("std", 1.5)
            raw_value = self._generate_normal_samples(sample, mean, std)
            # Direct initialization - constraints are built into kernel classes
            param.data = raw_value.to(dtype=param.dtype)
            logger.debug(f"Direct initialization: {name} = {raw_value} (constraints built into kernel classes)")

        elif method == "zeros":
            torch.nn.init.zeros_(param)

        elif method == "constant":
            value = config.get("value", 0.0)
            param.data = torch.full_like(param, value, dtype=param.dtype)

        elif method == "skip":
            pass

        else:
            # Fallback to normal with conservative parameters
            raw_value = 0.1 * (sample * 2 - 1)
            param.data = raw_value.to(dtype=param.dtype)

    def initialize(self, model: torch.nn.Module, run_index: int):
        """
        Initialize the model parameters for a specific run.

        We exclude '.weight' and '.bias' parameters from Sobol sampling:
        - '.weight': Xavier uniform
        - '.bias': zeros
        Other parameters are mapped from Sobol [0,1] samples into configured ranges.

        :param model: The model whose parameters need initialization.
        :param run_index: The run index corresponding to the precomputed Sobol sample.
        """
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

                # Handle weight parameters with Xavier uniform (exclude from Sobol sampling)
                # Only apply Xavier to parameters with at least 2 dimensions (required for fan_in/fan_out calculation)
                if ".weight" in name:
                    # reproducible per-run, on CPU
                    generator_seed = (self.seed + run_index) if self.seed is not None else run_index
                    g = torch.Generator().manual_seed(generator_seed)
                    if param.dim() >= 2:
                        # Standard neural network weight matrix: use Xavier uniform
                        torch.nn.init.xavier_uniform_(param, generator=g)
                        logger.debug(f"Initialized weight parameter '{name}' with Xavier uniform (shape={param.shape})")
                    else:
                        # 1D or scalar weight: use uniform initialization instead
                        torch.nn.init.uniform_(param, -0.1, 0.1)
                        logger.debug(f"Initialized 1D/scalar weight parameter '{name}' with uniform (shape={param.shape})")
                    continue

                # Handle bias parameters with zeros (exclude from Sobol sampling)
                if ".bias" in name:
                    torch.nn.init.zeros_(param)
                    logger.debug(f"Initialized bias parameter '{name}' with zeros")
                    continue

                # Slice the sobol_samples for the current parameter
                sample = self.sobol_samples[run_index, idx : idx + param_length]
                sample = sample.reshape(param.shape)
                sample = sample.to(device=param.device, dtype=param.dtype)

                # Initialize the parameter
                old_value = param.data.clone()
                self.initialize_parameter(param, sample, config, name, model, run_index)
                new_value = param.data.clone()

                logger.debug(
                    f"Initialized {name}: {config['description']} (shape={param.shape}, method={config['method']})"
                )
                # logger.debug(f"  Old value: {old_value}")
                logger.debug(f"  New value: {new_value}")

                # Special debug for key parameters
                if "cont_kernel.lengthscale" in name or "raw_noise" in name:
                    logger.info(f"KEY PARAMETER INITIALIZED: {name}")
                    logger.info(f"  Method: {config['method']}")
                    logger.info(f"  Old: {old_value}")
                    logger.info(f"  New: {new_value}")
                    logger.info(f"  Config: {config}")

                idx += param_length

        logger.info(f"Model parameters initialized with run #{run_index}")


class RFFParameterInitializer(DefaultParameterInitializer):
    """
    Hyperparameter initialization via Sobol samples, plus a distinct RFF frequency
    draw per init.

    All ``num_inits`` weight matrices are pre-drawn in :meth:`setup` from one RNG
    stream seeded with ``seed`` (trainer seed). Init ``i`` uses draw ``i``; draws
    are not re-seeded with ``seed + i``.
    """

    def __init__(self, num_inits: int, seed: int = None, parameter_configs: Optional[Dict[str, Dict[str, Any]]] = None):
        super().__init__(num_inits=num_inits, seed=seed, parameter_configs=parameter_configs)
        self._rff_weight_draws: list[torch.Tensor] | None = None

    def setup(self, model: torch.nn.Module) -> None:
        super().setup(model)
        from ..models.rff_gpr import RFFGPR
        from ..utils.rff_utils import init_rbf_weights

        self._rff_weight_draws = None
        if not isinstance(model, RFFGPR):
            return

        train_x = model.train_inputs[0]
        num_dims = train_x.shape[-1]
        num_samples = model.num_rff
        rff_kernel = model._rff_kernel
        device = rff_kernel.raw_lengthscale.device
        dtype = rff_kernel.raw_lengthscale.dtype
        rff_sampling = rff_kernel.rff_sampling

        if self.seed is not None:
            torch.manual_seed(self.seed)

        self._rff_weight_draws = [
            init_rbf_weights(num_dims, num_samples, device=device, dtype=dtype, rff_sampling=rff_sampling)
            for _ in range(self.num_inits)
        ]
        feature_kind = rff_sampling.upper()
        logger.info(
            "Precomputed %s %s weight draws from master seed %s (sequential RNG, not seed+run_index)",
            self.num_inits,
            feature_kind,
            self.seed,
        )

    def initialize(self, model: torch.nn.Module, run_index: int) -> None:
        self._assign_rff_weights(model, run_index)
        super().initialize(model, run_index)

    def _assign_rff_weights(self, model: torch.nn.Module, run_index: int) -> None:
        from ..models.rff_gpr import RFFGPR

        if not isinstance(model, RFFGPR):
            return

        rff_kernel = model._rff_kernel
        num_dims = model.train_inputs[0].shape[-1]
        num_samples = model.num_rff
        rff_sampling = rff_kernel.rff_sampling

        if self._rff_weight_draws is not None and run_index < len(self._rff_weight_draws):
            weights = self._rff_weight_draws[run_index].to(
                device=rff_kernel.raw_lengthscale.device,
                dtype=rff_kernel.raw_lengthscale.dtype,
            )
            rff_kernel._init_weights(
                num_dims,
                num_samples,
                randn_weights=weights,
                spectral=False,
                rff_sampling=rff_sampling,
            )
        else:
            if self.seed is not None:
                torch.manual_seed(self.seed)
            rff_kernel.resample_weights(spectral=False, rff_sampling=rff_sampling)

        model.invalidate_feature_cache()
        feature_kind = rff_sampling.upper()
        logger.info(
            "Assigned %s frequencies for run #%s (master seed=%s, precomputed draw)",
            feature_kind,
            run_index,
            self.seed,
        )