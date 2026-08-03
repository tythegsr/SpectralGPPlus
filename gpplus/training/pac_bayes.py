"""Catoni-style PAC-Bayes complexity term as a wrap-any-base-MLL factory.

Posterior ``Q = N(theta, sigma_q^2 I)`` is a Gaussian ball around the current
learnable parameters; prior ``P = N(0, sigma_p^2 I)``. Training minimizes

    empirical_risk + KL(Q || P) / (temperature * n)

which, with ``loss = -mll(...)``, is realized by returning

    mll_base - KL / (temperature * n).

Fixed buffers (e.g. RFF/SORF ``randn_weights``) are excluded.
"""

from __future__ import annotations

from typing import Type

import torch
import torch.nn as nn


def collect_learnable_params(model: nn.Module) -> list[torch.nn.Parameter]:
    """Return all ``requires_grad`` parameters (excludes buffers)."""
    return [p for p in model.parameters() if p.requires_grad]


def diagonal_gaussian_kl(
    mean: torch.Tensor,
    *,
    prior_std: float,
    posterior_std: float,
) -> torch.Tensor:
    """
    Closed-form KL for diagonal Gaussians ``Q = N(mean, sigma_q^2 I)``,
    ``P = N(0, sigma_p^2 I)`` (mean flattened to 1-D).

    ``KL = 0.5 * sum_i ( sigma_q^2/sigma_p^2 + mu_i^2/sigma_p^2 - 1
                          - log(sigma_q^2/sigma_p^2) )``.
    """
    if prior_std <= 0:
        raise ValueError(f"prior_std must be > 0, got {prior_std}.")
    if posterior_std <= 0:
        raise ValueError(f"posterior_std must be > 0, got {posterior_std}.")

    mu = mean.reshape(-1)
    sp2 = mean.new_tensor(float(prior_std) ** 2)
    sq2 = mean.new_tensor(float(posterior_std) ** 2)
    ratio = sq2 / sp2
    # per-dim constant terms + quadratic
    d = mu.numel()
    const = ratio - 1.0 - torch.log(ratio)
    quad = (mu * mu).sum() / sp2
    return 0.5 * (float(d) * const + quad)


def pac_bayes_complexity(
    model: nn.Module,
    *,
    prior_std: float = 1.0,
    posterior_std: float = 0.1,
) -> torch.Tensor:
    """Sum of diagonal-Gaussian KL terms over all learnable parameters."""
    params = collect_learnable_params(model)
    if not params:
        # No learnable params: complexity is zero on the model's device/dtype.
        try:
            ref = next(model.parameters())
            return ref.new_zeros(())
        except StopIteration:
            return torch.zeros((), dtype=torch.float64)

    total = params[0].new_zeros(())
    for p in params:
        total = total + diagonal_gaussian_kl(
            p, prior_std=prior_std, posterior_std=posterior_std
        )
    return total


def _num_data_from_target(target: torch.Tensor) -> int:
    if target.dim() == 0:
        return 1
    return int(target.shape[-1])


def pac_bayes_mll_class(
    base_mll_class: Type,
    *,
    temperature: float = 1.0,
    prior_std: float = 1.0,
    posterior_std: float = 0.1,
) -> Type:
    """
    Wrap any GPyTorch / GPPlus MLL class with a Catoni PAC-Bayes KL term.

    Dynamically subclasses ``base_mll_class`` so Woodbury detection via
    ``issubclass(..., RFFWoodburyMarginalLogLikelihood)`` still works.

    Defaults: ``temperature=1.0``, ``prior_std=1.0``, ``posterior_std=0.1``.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}.")
    if prior_std <= 0:
        raise ValueError(f"prior_std must be > 0, got {prior_std}.")
    if posterior_std <= 0:
        raise ValueError(f"posterior_std must be > 0, got {posterior_std}.")

    temp = float(temperature)
    sp = float(prior_std)
    sq = float(posterior_std)

    class PacBayesMLL(base_mll_class):
        pac_bayes_temperature = temp
        pac_bayes_prior_std = sp
        pac_bayes_posterior_std = sq

        def forward(self, function_samples, target, *args, **kwargs):
            mll = super().forward(function_samples, target, *args, **kwargs)
            n = _num_data_from_target(target)
            if n < 1:
                raise ValueError(f"PAC-Bayes MLL requires num_data >= 1, got {n}.")
            kl = pac_bayes_complexity(
                self.model,
                prior_std=self.pac_bayes_prior_std,
                posterior_std=self.pac_bayes_posterior_std,
            )
            # Match averaged-MLL scale: subtract KL / (lambda * n).
            return mll - kl / (self.pac_bayes_temperature * float(n))

    PacBayesMLL.__name__ = f"PacBayes_{getattr(base_mll_class, '__name__', 'MLL')}"
    PacBayesMLL.__qualname__ = PacBayesMLL.__name__
    return PacBayesMLL


__all__ = [
    "collect_learnable_params",
    "diagonal_gaussian_kl",
    "pac_bayes_complexity",
    "pac_bayes_mll_class",
]
