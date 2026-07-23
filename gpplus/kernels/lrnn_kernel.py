"""Low-rank neural network (Deep Basis) kernel for GPPlus."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable

import torch
import torch.nn as nn
from gpytorch.kernels import Kernel
from linear_operator.operators import MatmulLinearOperator
from torch import Tensor

from ..utils.input_transform_net import InputTransformNet


def _default_layer_config(
    hidden_dims: Sequence[int],
    feature_rank: int,
    activation: Callable[[], nn.Module],
) -> dict[int, dict]:
    """Build InputTransformNet-style config: hidden layers with activation, linear output."""
    config: dict[int, dict] = {}
    idx = 0
    for width in hidden_dims:
        config[idx] = {"dims": int(width), "activation": activation}
        idx += 1
    # Final linear map to rank-r features (identity activation = no nonlinearity).
    config[idx] = {"dims": int(feature_rank), "activation": nn.Identity}
    return config


class LRNNKernel(Kernel):
    """
    Low-rank deep basis kernel (Zhu et al., arXiv:2505.18526).

    Defines

        k(x, x') = <φ_θ(x), φ_θ(x')>

    where ``φ_θ`` is an MLP with trainable parameters. The Gram matrix is
    ``K = Φ Φ^T`` with ``Φ ∈ R^{n × r}``, enabling Woodbury exact GP inference.

    Default architecture (paper §4): two hidden layers of 128 units with ``tanh``,
    then a linear map to ``feature_rank`` basis functions (default ``r=128``).

    With ``batch_shape=torch.Size([B])``, maintains ``B`` independent feature nets
    and returns features shaped ``(B, n, r)``.
    """

    has_lengthscale = False

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (128, 128),
        feature_rank: int = 128,
        activation: Callable[[], nn.Module] | type[nn.Module] = nn.Tanh,
        layer_config: dict | None = None,
        batch_shape: torch.Size | None = None,
        **kwargs,
    ):
        batch_shape = torch.Size([]) if batch_shape is None else torch.Size(batch_shape)
        super().__init__(batch_shape=batch_shape, **kwargs)
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        if feature_rank < 1:
            raise ValueError(f"feature_rank must be >= 1, got {feature_rank}")

        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.feature_rank = int(feature_rank)
        self._activation = activation

        if layer_config is not None:
            self.layer_config = layer_config
        else:
            self.layer_config = _default_layer_config(
                self.hidden_dims, self.feature_rank, activation
            )

        if len(self.batch_shape) == 0:
            self.feature_net = InputTransformNet(self.input_dim, self.layer_config)
            self.feature_nets = None
        else:
            b = int(self.batch_shape[0])
            self.feature_nets = nn.ModuleList(
                [InputTransformNet(self.input_dim, self.layer_config) for _ in range(b)]
            )
            self.feature_net = self.feature_nets[0]  # alias for single-net call sites
        self._feature_cache_version = 0

    @property
    def num_features(self) -> int:
        """Rank ``r`` of the low-rank kernel (output width of ``φ``)."""
        last = list(self.layer_config.values())[-1]
        return int(last["dims"])

    def bump_feature_cache_version(self) -> None:
        self._feature_cache_version += 1

    def featurize(self, x: Tensor) -> Tensor:
        """Map inputs to neural basis features ``φ(x)`` of shape ``(..., r)`` or ``(B, n, r)``."""
        flat = x.reshape(-1, x.shape[-1])
        if self.feature_nets is None:
            phi = self.feature_net(flat)
            phi = phi.reshape(*x.shape[:-1], phi.shape[-1])
        else:
            # Independent nets per init -> (B, n, r)
            phis = [net(flat).reshape(*x.shape[:-1], -1) for net in self.feature_nets]
            phi = torch.stack(phis, dim=0)
        # Keep typical ||φ||² and Gram entries O(1) for Woodbury stability.
        return phi * self._feature_scale

    @property
    def _feature_scale(self) -> float:
        return 1.0 / (self.num_features**0.5)

    def kernel_diag(self, x: Tensor) -> Tensor:
        """Prior variance ``k(x, x) = ||φ(x)||²``."""
        phi = self.featurize(x)
        return (phi * phi).sum(dim=-1)

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params,
    ) -> Tensor:
        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)

        z1 = self.featurize(x1)
        if x1 is x2 or (x1.shape == x2.shape and torch.equal(x1, x2)):
            z2 = z1
        else:
            z2 = self.featurize(x2)

        if diag:
            return (z1 * z2).sum(dim=-1)

        return MatmulLinearOperator(z1, z2.transpose(-1, -2))
