"""Soft probabilistic bound penalties for physical-scale GP predictions."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BoundConfig:
    """Per-QoI soft probabilistic bound settings (physical units)."""

    a: float | None = None  # floor; None = no minimum constraint
    b: float | None = None  # ceiling; None = no maximum constraint
    k: float = 2.0
    lam: float = 1.0
    alpha: float = 10.0
    max_points: int | None = 4096

    def __post_init__(self) -> None:
        if self.a is None and self.b is None:
            raise ValueError("BoundConfig requires at least one of a (floor) or b (ceiling).")
        if self.a is not None and self.b is not None and not (self.b > self.a):
            raise ValueError(f"BoundConfig requires b > a, got a={self.a}, b={self.b}.")
        if self.k < 0:
            raise ValueError(f"BoundConfig.k must be >= 0, got {self.k}.")
        if self.lam < 0:
            raise ValueError(f"BoundConfig.lam must be >= 0, got {self.lam}.")
        if self.alpha <= 0:
            raise ValueError(f"BoundConfig.alpha must be > 0, got {self.alpha}.")
        if self.max_points is not None and int(self.max_points) < 1:
            raise ValueError(f"BoundConfig.max_points must be >= 1 or None, got {self.max_points}.")


def soft_probabilistic_bound_penalty(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    *,
    a: float | None = None,
    b: float | None = None,
    k: float = 2.0,
    alpha: float = 10.0,
) -> torch.Tensor:
    """
    Mean softplus penalty for probabilistic box constraints.

    Encourages ``mu - k*sigma >= a`` and/or ``mu + k*sigma <= b`` in the same
    units as ``mu`` / ``sigma`` (typically physical QoI units after unstandardizing).

    Returns a scalar (mean over the last dimension).
    """
    if a is None and b is None:
        return mu.new_zeros(())

    sigma = sigma.clamp_min(0.0)
    # mean(L_lo + L_hi) with inactive sides omitted (Pensoneault softplus form).
    per_point = torch.zeros_like(mu)
    if a is not None:
        # Violate when mu - k*sigma < a  <=>  a - (mu - k*sigma) > 0
        lo = alpha * (float(a) - (mu - float(k) * sigma))
        per_point = per_point + torch.nn.functional.softplus(lo)
    if b is not None:
        # Violate when mu + k*sigma > b  <=>  (mu + k*sigma) - b > 0
        hi = alpha * ((mu + float(k) * sigma) - float(b))
        per_point = per_point + torch.nn.functional.softplus(hi)
    return per_point.mean()
