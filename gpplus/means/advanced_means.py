from __future__ import annotations

from collections.abc import Sequence

import torch.nn as nn
from gpytorch.means import Mean

from ..utils import InputTransformNet

################################

_NEURAL_MEAN_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "identity": nn.Identity,
}


def resolve_neural_mean_activation(name: str) -> type[nn.Module]:
    key = str(name).strip().lower()
    if key not in _NEURAL_MEAN_ACTIVATIONS:
        allowed = ", ".join(sorted(_NEURAL_MEAN_ACTIVATIONS))
        raise ValueError(f"Unknown neural mean activation {name!r}. Allowed: {allowed}.")
    return _NEURAL_MEAN_ACTIVATIONS[key]


def layer_config_from_hidden(
    hidden_dims: Sequence[int],
    activation: str = "relu",
) -> dict[int, dict]:
    """Build InputTransformNet layer_config from hidden widths + activation name.

    Appends a final linear layer with dims=1 and Identity activation.
    """
    dims = [int(d) for d in hidden_dims]
    if not dims:
        raise ValueError("hidden_dims must be a non-empty sequence of positive ints.")
    if any(d < 1 for d in dims):
        raise ValueError(f"hidden_dims entries must be >= 1, got {dims}.")
    act_cls = resolve_neural_mean_activation(activation)
    layer_config: dict[int, dict] = {}
    for i, width in enumerate(dims):
        layer_config[i] = {"dims": width, "activation": act_cls}
    layer_config[len(dims)] = {"dims": 1, "activation": nn.Identity}
    return layer_config


################################


class CompositeMean(Mean):
    """
    A mean function that applies a user-provided transformation module
    to the input. The transformation is expected to yield a single-dimensional
    output (shape (batch_size, 1) or (batch_size,)) which represents the mean.

    This can be any valid callable or nn.Module, including:
      - Polynomial feature expansions
      - Small neural networks
      - Other custom transforms

    Unlike a typical 'base_mean', this class just returns the
    output of the provided transformation.
    """

    def __init__(self, input_transform):
        """
        Args:
            input_transform (callable or nn.Module):
                A user-defined transformation that takes x of shape (batch_size, input_dim)
                and returns a single-dimensional output (batch_size,).
        """
        super().__init__()
        self.input_transform = input_transform

    def forward(self, x):
        """
        Applies the user-provided transformation and returns its output
        as the mean.

        Args:
            x (torch.Tensor): Input of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Mean predictions of shape (batch_size,).
        """
        # Ensure final shape is (batch_size,) by squeezing if needed
        return self.input_transform(x).squeeze(-1)


################################


class NeuralMean(Mean):
    """
    A mean function that builds its own MLP internally using InputTransformNet.
    The final layer of InputTransformNet must have `dims=1`, so that the
    network outputs a single scalar per data point.

    For example, a valid layer_config might look like:

        layer_config = {
            0: {"dims": 64, "activation": nn.ReLU},
            1: {"dims": 32, "activation": nn.ReLU},
            2: {"dims": 1,  "activation": nn.Identity},
        }

    This ensures the network outputs shape (batch_size, 1), which we
    then squeeze to (batch_size,).

    Prefer ``NeuralMean.from_hidden`` for script-friendly hidden widths +
    activation name (final dims=1 Identity is appended automatically).
    """

    def __init__(self, input_dim, layer_config):
        """
        Args:
            input_dim (int): Number of input features.
            layer_config (dict): Configuration for each layer, where the last layer
                                 must have 'dims' == 1.
        """
        super().__init__()
        # Build the main transform network
        self.transform_net = InputTransformNet(input_dim, layer_config)

        # Check that the last layer indeed produces 1 dimension
        last_layer_idx = max(layer_config.keys())  # highest key
        last_layer_dims = layer_config[last_layer_idx]["dims"]
        if last_layer_dims != 1:
            raise ValueError(
                f"For NeuralMean, the final layer in `layer_config` must have dims=1, but got {last_layer_dims}."
            )

    @classmethod
    def from_hidden(
        cls,
        input_dim: int,
        hidden_dims: Sequence[int],
        activation: str = "relu",
    ) -> NeuralMean:
        """Build a NeuralMean MLP from hidden widths and an activation name."""
        layer_config = layer_config_from_hidden(hidden_dims, activation=activation)
        return cls(input_dim, layer_config)

    def forward(self, x):
        """
        Computes the mean by passing inputs through the internal network.

        Args:
            x (torch.Tensor): Input of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Mean predictions of shape (batch_size,).
        """
        out = self.transform_net(x)  # shape (batch_size, 1)
        return out.squeeze(-1)  # shape (batch_size,)
