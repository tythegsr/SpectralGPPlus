"""Padded structured orthogonal random features (SORF) for RBF kernels.

Yu et al. (2016) SORF uses ``W = sqrt(d) H D1 H D2 H D3`` with ``d`` padded to
the next power of two. Every column then has norm exactly ``sqrt(d)``, which
for small ``d`` is far from the ``chi(d)`` radial law of a Gaussian draw: at
``d = 1`` all frequencies are ``±1`` and the feature map is one cosine.

Here each block is drawn at ``d_pad = max(pad_min, next_pow2(d))``, scaled by
``sqrt(d_pad)`` and truncated to its first ``d`` rows. The truncated entries
are close to i.i.d. ``N(0, 1)`` while columns within a block stay nearly
orthogonal. ``radial="chi"`` additionally redraws every column norm from
``chi(d)``; that matches the RBF radial law exactly but reintroduces i.i.d.
radial noise, and measured kernel error is lower without it (``"none"``).
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn

Radial = Literal["chi", "none"]


def _next_pow2(n: int) -> int:
    return 1 << max(0, int(n - 1).bit_length())


def _hadamard(n: int, *, dtype: torch.dtype, device: torch.device | None = None) -> torch.Tensor:
    h = torch.ones(1, 1, dtype=dtype, device=device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


def sample_padded_sorf(
    d: int,
    num_freq: int,
    *,
    pad_min: int = 64,
    radial: Radial = "none",
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Unit-lengthscale RBF frequencies ``W`` of shape ``(d, num_freq)``."""
    d_pad = max(int(pad_min), _next_pow2(int(d)))
    h = _hadamard(d_pad, dtype=dtype)
    cols: list[torch.Tensor] = []
    remaining = int(num_freq)
    while remaining > 0:
        signs = [
            (torch.randint(0, 2, (d_pad,), generator=generator) * 2 - 1).to(dtype)
            for _ in range(3)
        ]
        m = (h * signs[0]) @ (h * signs[1]) @ (h * signs[2])
        block = math.sqrt(d_pad) * m[: int(d)]
        take = min(d_pad, remaining)
        cols.append(block[:, torch.randperm(d_pad, generator=generator)[:take]])
        remaining -= take
    w = torch.cat(cols, dim=1)
    if radial == "chi":
        g = torch.randn(int(d), w.shape[1], generator=generator, dtype=dtype)
        w = w / w.norm(dim=0, keepdim=True).clamp_min(1e-12) * g.norm(dim=0, keepdim=True)
    return w


def auto_num_freq(
    n: int, d: int, *, lo: int = 256, hi: int = 2048, woodbury: bool = True
) -> int:
    """Frequency count ``D`` (feature dim ``2D``) grown with ``n`` and ``d``."""
    D = 2 * math.ceil(math.sqrt(max(1, int(n))) * math.log2(int(d) + 2))
    D = max(lo, min(hi, D))
    if not woodbury:
        return int(D)
    # Woodbury only pays off while the 2D x 2D system stays below n x n.
    return int(max(32, min(D, max(32, int(n) // 4))))


class SorfFeatures(nn.Module):
    """``φ(x) = sqrt(σ_f² / D) [cos(x W / ℓ), sin(x W / ℓ)]`` with learnable ARD ``ℓ``."""

    def __init__(
        self,
        d: int,
        num_freq: int,
        *,
        pad_min: int = 64,
        radial: Radial = "none",
        ard: bool = True,
        seed: int = 0,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.d = int(d)
        self.num_freq = int(num_freq)
        gen = torch.Generator().manual_seed(int(seed))
        self.register_buffer(
            "W",
            sample_padded_sorf(
                self.d, self.num_freq, pad_min=pad_min, radial=radial, generator=gen, dtype=dtype
            ),
        )
        self.log_ls = nn.Parameter(torch.zeros(self.d if ard else 1, dtype=dtype))

    @property
    def dim(self) -> int:
        return 2 * self.num_freq

    def freqs(self) -> torch.Tensor:
        """Lengthscale-scaled frequencies ``(d, D)``."""
        return self.W / torch.exp(self.log_ls).reshape(-1, 1)

    def proj(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.freqs()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.proj(x)
        return torch.cat([torch.cos(p), torch.sin(p)], dim=-1) / math.sqrt(self.num_freq)
