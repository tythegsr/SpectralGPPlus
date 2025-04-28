from abc import ABC, abstractmethod

import torch
from torch.quasirandom import SobolEngine

# Configure logging
from ..config import logger


class ParameterInitializer(ABC):
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
        Calculates the total number of learnable parameters in the model, excluding
        weights and biases, and precomputes Sobol samples for all runs.
        """
        # Count only parameters that are not weight or bias
        self.num_params = 0
        for name, p in model.named_parameters():
            if p.requires_grad and not any(k in name for k in (".weight", ".bias")):
                self.num_params += p.numel()

        # Handle case where there are no Sobol-driven parameters
        if self.num_params == 0:
            self.sobol_samples = None
            logger.info("No Sobol-driven parameters to initialize.")
            return

        # Initialize Sobol Engine
        sobol_engine = SobolEngine(dimension=self.num_params, scramble=True, seed=self.seed)
        # Generate all initialization points at once
        self.sobol_samples = sobol_engine.draw(self.num_runs)

        logger.debug(f"Sobol samples generated: {self.sobol_samples}")
        logger.warning(f"Sobol samples shape: {self.sobol_samples.shape}")

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
        # TODO: Use registry mapping regex patterns from development branch instead of this hardcoded initialization
        idx = 0
        # Loop over each parameter in the model and initialize based on name
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue

                # Handle weight parameters with Xavier uniform
                if ".weight" in name:
                    # reproducible per‐run, on CPU
                    g = torch.Generator().manual_seed(self.seed + run_index)
                    torch.nn.init.xavier_uniform_(param, generator=g)
                    logger.debug(f"Initialized weight parameter '{name}' with Xavier uniform")
                    continue

                # Handle bias parameters with zeros
                if ".bias" in name:
                    torch.nn.init.zeros_(param)
                    logger.debug(f"Initialized bias parameter '{name}' with zeros")
                    continue

                # For other parameters, slice Sobol samples and reshape
                param_length = param.numel()
                sample = self.sobol_samples[run_index, idx : idx + param_length]
                sample = sample.reshape(param.shape)

                # Parameter-specific mapping
                if ".lengthscale" in name:
                    scale = 3
                    param.data = sample * 2 * scale - scale
                    # torch.nn.init.normal_(param, mean=1.0, std=2.0)
                elif ".outputscale" in name:
                    lower, upper = 0.1, 10
                    param.data = lower + (upper - lower) * sample
                # elif "weight" in name:
                #     # Xavier uniform initialization for weight parameters
                #     # torch.nn.init.xavier_uniform_(param)
                #     # print(f'param.shape: {name}: {param.shape[0]}, {param.shape[1]}, {param.shape}')
                #     fan_in, fan_out = param.size(1), param.size(0)
                #     # Xavier/Glorot scaling
                #     # torch.tensor(6.0 / (fan_in + fan_out))
                #     limit = torch.sqrt(torch.tensor(0.2 / (fan_in + fan_out)))
                #     param.data = (sample * 2 - 1) * limit
                # elif "bias" in name:
                #     torch.nn.init.zeros_(param)
                elif "power" in name:
                    lower, upper = -5, 10
                    param.data = lower + (upper - lower) * sample
                elif ".raw_noise" in name:
                    lower, upper = -6, -6 + 1e-2
                    param.data = lower + (upper - lower) * sample
                else:
                    # Default initialization strategy.
                    param.data = 10 * sample - 5

                idx += param_length
                logger.debug("Num Param #: {}".format(param_length))
        # Robust indexing: ensure all Sobol dims were consumed
        if idx != self.num_params:
            raise ValueError(f"Consumed {idx} Sobol samples but expected {self.num_params}")
        logger.info("Model parameters initialized for run #: {}".format(run_index))