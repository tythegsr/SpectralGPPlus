from abc import ABC, abstractmethod

import torch
from torch.quasirandom import SobolEngine

# Configure logging
from ..config import logger

# def initialize_parameters(self, model, num_run):


class ParameterInitializer(ABC):
    @abstractmethod
    def __init__(self, num_runs: int, seed: int = None):
        pass

    @abstractmethod
    def setup(self, model: torch.nn.Module):
        raise NotImplementedError

    @abstractmethod
    def initialize(self, model: torch.nn.Module):
        raise NotImplementedError


class DefaultParameterInitializer(ParameterInitializer):
    def __init__(self, num_runs: int, seed: int = None):
        """
        :param num_runs: Total number of initialization runs.
        :param seed: Random seed for reproducibility.
        """
        self.num_runs = num_runs
        self.seed = seed
        self.num_params = None
        self.sobol_samples = None

    def setup(self, model: torch.nn.Module):
        """
        Calculates the total number of learnable parameters in the model and
        precomputes the Sobol samples for all runs.
        """
        # Get the number of learnable parameters
        self.num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # Initialize Sobol Engine
        sobol_engine = SobolEngine(dimension=self.num_params, scramble=True, seed=self.seed)
        # Generate all initialization points at once
        self.sobol_samples = sobol_engine.draw(self.num_runs)

        logger.debug(f"Sobol samples generated: {self.sobol_samples}")
        logger.warning(f"Sobol samples shape: {self.sobol_samples.shape}")

    def get_initialization_config(self, name: str, param: torch.Tensor):
        """
        Get initialization configuration based on parameter name and shape.
        Returns a dict with initialization method and parameters.
        """
        param_shape = param.shape
        param_dim = param.dim()
        
        # Debug: Print parameter name for raw_lengthscales
        # if "raw_lengthscales" in name:
            # print(f"[DEBUG] Found raw_lengthscales parameter: {name}")
        
        # Matrix encoder parameters
        if "projection_matrix" in name:
            return {
                'method': 'orthogonal',
                'gain': 1.0,
                'description': 'Matrix encoder projection matrix'
            }
        

        

        
        # Kernel lengthscales (for all other cases)
        if "lengthscale" in name:
            if param_dim == 1:  # ARD lengthscales
                return {
                    'method': 'uniform',
                    'lower': -6.0,
                    'upper': 3.0,
                    'description': 'Kernel lengthscale'
                }
            else:  # Single lengthscale
                return {
                    'method': 'uniform',
                    'lower': -6.0,
                    'upper': 3.0,
                    'description': 'Kernel lengthscale'
                }
        
        # Kernel outputscales
        if "outputscale" in name:
            return {
                'method': 'uniform',
                'lower': 1e-6,
                'upper': 1e0,
                'description': 'Kernel outputscale'
            }
        
        # Likelihood noise
        if "raw_noise" in name or "noise" in name:
            return {
                'method': 'uniform',
                'lower': -6.0,
                'upper': -3.0,
                'description': 'Likelihood noise'
            }
        
        # Neural network weights
        if "weight" in name and param_dim >= 2:
            fan_in, fan_out = param.size(1), param.size(0)
            limit = torch.sqrt(torch.tensor(2.0 / (fan_in + fan_out)))
            return {
                'method': 'xavier_uniform',
                'limit': limit.item(),
                'description': f'Neural network weight ({fan_in}->{fan_out})'
            }
        
        # Neural network biases
        if "bias" in name and param_dim == 1:
            return {
                'method': 'zeros',
                'description': 'Neural network bias'
            }
        
        # Power parameters (for power kernels)
        if "power" in name:
            return {
                'method': 'uniform',
                'lower': 1.0,
                'upper': 3.0,
                'description': 'Power kernel parameter'
            }
        
        # Default for unknown parameters
        return {
            'method': 'uniform',
            'lower': -1.0,
            'upper': 1.0,
            'description': 'Unknown parameter (default)'
        }

    def initialize_parameter(self, param: torch.Tensor, sample: torch.Tensor, config: dict):
        """
        Initialize a single parameter based on the configuration.
        """
        method = config['method']
        
        if method == 'orthogonal':
            # Use PyTorch's orthogonal initialization
            torch.nn.init.orthogonal_(param, gain=config.get('gain', 1.0))
            
        elif method == 'xavier_uniform':
            # Xavier/Glorot uniform initialization
            limit = config.get('limit', 1.0)
            param.data = (sample * 2 - 1) * limit
            
        elif method == 'uniform':
            # Uniform initialization in specified range
            lower = config.get('lower', -1.0)
            upper = config.get('upper', 1.0)
            # Check if parameter has a constraint and respect it
            # if hasattr(param, 'constraint') and param.constraint is not None:
            #     # Use the constraint's inverse transform to get the raw value
            #     raw_value = lower + (upper - lower) * sample
            #     # Apply the constraint's transform to get the constrained value
            #     constrained_value = param.constraint.transform(raw_value)
            #     param.data = constrained_value
            # else:
            #     # No constraint, set directly
            #     param.data = lower + (upper - lower) * sample
            random_value = lower + (upper - lower) * sample
            param.data = random_value
        elif method == 'zeros':
            # Zero initialization
            torch.nn.init.zeros_(param)
            
        elif method == 'normal':
            # Normal initialization
            mean = config.get('mean', 0.0)
            std = config.get('std', 0.1)
            torch.nn.init.normal_(param, mean=mean, std=std)
            
        elif method == 'constant':
            # Constant initialization
            value = config.get('value', 0.0)
            param.data = torch.full_like(param, value)
            
        elif method == 'skip':
            # Skip initialization for this parameter
            pass

        else:
            # Fallback to uniform
            param.data = sample * 2 - 1

    def initialize(self, model: torch.nn.Module, run_index: int):
        """
        Initialize the model parameters for a specific run using adaptive configuration.
        :param model: The model whose parameters need initialization.
        :param run_index: The run index corresponding to the precomputed Sobol sample.
        """
        idx = 0
        
        # Loop over each parameter in the model and initialize based on configuration
        with torch.no_grad():
            all_params = list(model.named_parameters())
            for i, (name, param) in enumerate(all_params):
                
                param_length = param.numel()
                
                # Skip parameters with zero elements
                if param_length == 0:
                    logger.debug(f"Skipping empty parameter: {name}")
                    continue
                
                # Skip non-learnable parameters
                if not param.requires_grad:
                    logger.debug(f"Skipping non-learnable parameter: {name}")
                    continue
                
                # Get initialization configuration
                config = self.get_initialization_config(name, param)
                
                # Slice the sobol_samples for the current parameter
                sample = self.sobol_samples[run_index, idx : idx + param_length]
                sample = sample.reshape(param.shape)
                
                # Initialize the parameter
                self.initialize_parameter(param, sample, config)
                
                logger.debug(f"Initialized {name}: {config['description']} "
                           f"(shape={param.shape}, method={config['method']})")
                
                idx += param_length

        logger.info(f"Model parameters initialized with run #{run_index}")


class SimpleParameterInitializer(ParameterInitializer):
    """
    A simpler parameter initializer that uses standard PyTorch initialization methods.
    This is more robust and doesn't rely on hardcoded ranges.
    """
    
    def __init__(self, num_runs: int, seed: int = None):
        self.num_runs = num_runs
        self.seed = seed
        self.current_run = 0
    
    def setup(self, model: torch.nn.Module):
        """No setup needed for simple initializer."""
        pass
    
    def initialize(self, model: torch.nn.Module, run_index: int):
        """
        Initialize parameters using standard PyTorch methods.
        """
        torch.manual_seed(self.seed + run_index if self.seed is not None else run_index)
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                
                param_shape = param.shape
                param_dim = param.dim()
                
                # Matrix encoder parameters
                if "projection_matrix" in name:
                    torch.nn.init.orthogonal_(param, gain=1.0)
                    
                # Kernel parameters
                elif "lengthscale" in name:
                    # Check if parameter has a constraint and respect it
                    if hasattr(param, 'constraint') and param.constraint is not None:
                        # Generate raw values in constraint's domain
                        raw_values = torch.rand_like(param) * (param.constraint.upper_bound - param.constraint.lower_bound) + param.constraint.lower_bound
                        # Apply constraint transform
                        constrained_values = param.constraint.transform(raw_values)
                        param.data = constrained_values
                    else:
                        # No constraint, use standard initialization
                        torch.nn.init.uniform_(param, -6.0, 3.0)
                    
                elif "outputscale" in name:
                    torch.nn.init.uniform_(param, 1e-6, 1e1)
                    
                # Likelihood parameters
                elif "raw_noise" in name or "noise" in name:
                    torch.nn.init.uniform_(param, -6.0, -3.0)
                    
                # Neural network parameters
                elif "weight" in name and param_dim >= 2:
                    torch.nn.init.xavier_uniform_(param)
                    
                elif "bias" in name:
                    torch.nn.init.zeros_(param)
                # Default
                else:
                    torch.nn.init.uniform_(param, -1.0, 1.0)
        
        logger.info(f"Model parameters initialized with run #{run_index} (simple method)")
