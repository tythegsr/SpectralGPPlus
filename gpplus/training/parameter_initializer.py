import torch

from ..config import logger


class ParameterInitializer:
    @staticmethod
    def initialize_parameters(model, sobol_sample):
        idx = 0
        with torch.no_grad():
            # Loop over each parameter in the model and initialize based on name
            for name, param in model.named_parameters():
                param_length = param.numel()
                initial_values = sobol_sample[idx : idx + param_length].reshape(param.shape)

                if param.requires_grad:
                    if ".lengthscale" in name:
                        scale = 3
                        initial_values = initial_values * 2 * scale - scale
                        param.data = initial_values

                    elif ".outputscale" in name:
                        lower, upper = 0.1, 10
                        initial_values = lower + (upper - lower) * initial_values
                        param.data = initial_values

                    elif "weight" in name:
                        limit = torch.sqrt(torch.tensor(0.2 / (param.size(1) + param.size(0))))
                        initial_values = (initial_values * 2 - 1) * limit
                        param.data = initial_values

                    elif "bias" in name:
                        torch.nn.init.zeros_(param)

                    elif "power" in name:
                        lower, upper = -5, 10
                        initial_values = lower + (upper - lower) * initial_values
                        param.data = initial_values

                    elif ".raw_noise" in name:
                        lower, upper = -6, -6 + 1e-2
                        initial_values = lower + (upper - lower) * initial_values
                        param.data = initial_values

                    else:
                        param.data = 10 * initial_values - 5

                    idx += param_length

                    logger.debug("Num Param #: {}".format(param_length))
