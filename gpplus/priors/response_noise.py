"""Per-response LogNormal priors for multitask GP observation noise.

Priors are registered on per-task noise **variances** σ²_t in model/y-scaled space.
Training adds log p(σ²) via GPyTorch ``_add_other_terms`` in Woodbury MLL (MAP).

Woodbury inference in :class:`~gpplus.models.RFFMTGPR` uses only ``task_noises``;
likelihoods built here set ``has_global_noise=False``.
"""

from __future__ import annotations

from typing import Union

import gpytorch
import torch
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.priors import LogNormalPrior, Prior
from gpytorch.priors.utils import _load_transformed_to_base_dist
from torch import Tensor
from torch.distributions import TransformedDistribution


def empirical_task_noise_variances(
    y_train: Tensor,
    *,
    fraction: float = 0.01,
    min_variance: float = 1e-6,
) -> Tensor:
    """
    Heuristic per-task noise variance targets from training responses.

    Parameters
    ----------
    y_train : (n, T) targets in model space (e.g. standardized, log-grain applied).
    fraction : Scale applied to each column's empirical variance (signal dominates full var).
    min_variance : Floor on each task variance.
    """
    if y_train.dim() != 2:
        raise ValueError(f"y_train must be (n, T), got shape {tuple(y_train.shape)}.")
    if not (0.0 < fraction):
        raise ValueError(f"fraction must be positive, got {fraction}.")
    col_var = y_train.var(dim=0, unbiased=False)
    return (fraction * col_var).clamp_min(min_variance)


def log_normal_noise_prior_from_responses(
    y_train: Tensor,
    *,
    fraction: float = 0.01,
    log_scale: Union[float, Tensor] = 0.5,
    min_variance: float = 1e-6,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> LogNormalPrior:
    """
    Build a per-task :class:`~gpytorch.priors.LogNormalPrior` on noise variances σ²_t.

    PyTorch ``LogNormal(loc, scale)`` uses log-space ``loc``; we set ``loc_t = log(v_t)``
    so the median of σ²_t is near ``v_t = fraction * var(y[:, t])``.
    """
    target_vars = empirical_task_noise_variances(
        y_train, fraction=fraction, min_variance=min_variance
    )
    ref = y_train if dtype is None and device is None else y_train.to(
        dtype=dtype or y_train.dtype, device=device or y_train.device
    )
    loc = torch.log(target_vars.to(device=ref.device, dtype=ref.dtype))
    if isinstance(log_scale, Tensor):
        scale = log_scale.to(device=ref.device, dtype=ref.dtype)
    else:
        scale = torch.full_like(loc, float(log_scale))
    return LogNormalPrior(loc=loc, scale=scale)


def _sync_prior_device(prior: Prior, device: torch.device, dtype: torch.dtype) -> None:
    """Move a GPyTorch prior and sync TransformedDistribution parameter views."""
    if hasattr(prior, "to"):
        prior.to(device=device, dtype=dtype)
    if isinstance(prior, TransformedDistribution) and any(
        key.startswith("_transformed_") for key in prior._buffers
    ):
        _load_transformed_to_base_dist(prior)


def align_multitask_noise_priors(likelihood: MultitaskGaussianLikelihood) -> None:
    """
    Move registered GPyTorch priors to the likelihood parameter device.

    ``register_prior`` does not attach priors as child modules, so ``.to(device)``
    on the likelihood leaves ``LogNormalPrior`` distribution views on CPU unless synced.
    """
    align_registered_priors(likelihood)


def align_registered_priors(module: torch.nn.Module) -> None:
    """Sync all ``named_priors()`` on a GPyTorch module tree to ``module``'s parameter device."""
    if not hasattr(module, "named_priors"):
        return
    try:
        device = next(module.parameters()).device
        dtype = next(module.parameters()).dtype
    except StopIteration:
        return
    for _name, _parent, prior, _closure, _inv in module.named_priors():
        if prior is not None:
            _sync_prior_device(prior, device, dtype)


def build_multitask_noise_likelihood(
    num_tasks: int,
    noise_prior: Prior | None = None,
    rank: int = 0,
) -> MultitaskGaussianLikelihood:
    """
    MultitaskGaussianLikelihood consistent with Woodbury MT inference.

    Uses per-task diagonal noise only (``has_global_noise=False``).
    """
    return MultitaskGaussianLikelihood(
        num_tasks=num_tasks,
        rank=rank,
        noise_prior=noise_prior,
        has_global_noise=False,
        has_task_noise=True,
    )


def task_noise_raw_init_from_variances(
    likelihood: MultitaskGaussianLikelihood,
    variances: Tensor,
) -> Tensor:
    """Map target noise variances σ²_t to ``raw_task_noises`` (constraint inverse)."""
    if not hasattr(likelihood, "raw_task_noises_constraint"):
        raise TypeError("likelihood must expose raw_task_noises_constraint (rank=0, has_task_noise=True).")
    ref = likelihood.raw_task_noises
    v = variances.reshape(-1).to(device=ref.device, dtype=ref.dtype)
    return likelihood.raw_task_noises_constraint.inverse_transform(v)
