"""
Recommended Woodbury / noise stability settings for RFF experiment runners.

Library ``gpplus`` models keep the original primal Woodbury defaults; TOA and
other RFF benchmarks import this module so training and evaluation use the
stability knobs (dual Woodbury, dual eigen MT, float32 jitter, σ² floor 1e-5).
"""

from __future__ import annotations

import torch
from gpytorch.likelihoods import Likelihood
from gpytorch.priors import Prior

from gpplus.constraints import SoftClamp
from gpplus.training import RFFMTWoodburyMarginalLogLikelihood, RFFWoodburyMarginalLogLikelihood
from gpplus.utils.rff_utils import WoodburyForm, WoodburyMtMethod

DEFAULT_WOODBURY_FORM: WoodburyForm = "dual"
DEFAULT_MT_WOODBURY_METHOD: WoodburyMtMethod = "dual_eigen"
DEFAULT_WOODBURY_JITTER_FLOAT32 = 1e-4
DEFAULT_WOODBURY_JITTER_FLOAT64 = 1e-6
DEFAULT_NOISE_FLOOR = 1e-5
DEFAULT_NOISE_LOG_LOWER = -5.0  # log10(DEFAULT_NOISE_FLOOR)
DEFAULT_NOISE_LOG_UPPER = 3.0
DEFAULT_NOISE_INIT_LOWER = -5.0
DEFAULT_NOISE_INIT_UPPER = -1.0


def woodbury_jitter_for_dtype(dtype: torch.dtype) -> float:
    """Experiment Woodbury jitter (larger for float32 feature maps)."""
    return DEFAULT_WOODBURY_JITTER_FLOAT32 if dtype == torch.float32 else DEFAULT_WOODBURY_JITTER_FLOAT64


def rff_noise_constraint() -> SoftClamp:
    return SoftClamp(lower_bound=DEFAULT_NOISE_LOG_LOWER, upper_bound=DEFAULT_NOISE_LOG_UPPER)


def rff_noise_initializer_parameter_config() -> dict:
    return {
        "method": "uniform",
        "lower": DEFAULT_NOISE_INIT_LOWER,
        "upper": DEFAULT_NOISE_INIT_UPPER,
        "description": f"Noise parameter - uniform scale (σ² >= {DEFAULT_NOISE_FLOOR:g})",
    }


def build_rff_scalar_noise_likelihood(
    noise_prior: Prior | None = None,
    *,
    batch_shape: torch.Size | None = None,
) -> Likelihood:
    from gpplus.likelihoods import LogGaussianLikelihood

    return LogGaussianLikelihood(
        noise_prior=noise_prior,
        noise_constraint=rff_noise_constraint(),
        batch_shape=torch.Size([]) if batch_shape is None else torch.Size(batch_shape),
    )


def build_rff_multitask_noise_likelihood(
    num_tasks: int,
    noise_prior: Prior | None = None,
    *,
    rank: int = 0,
    batch_shape: torch.Size | None = None,
) -> Likelihood:
    """Log10 SoftClamp per-task noise (matches single-task LogGaussianLikelihood)."""
    from gpplus.likelihoods import LogMultitaskGaussianLikelihood

    return LogMultitaskGaussianLikelihood(
        num_tasks=num_tasks,
        rank=rank,
        noise_prior=noise_prior,
        noise_constraint=rff_noise_constraint(),
        has_global_noise=False,
        has_task_noise=True,
        batch_shape=torch.Size([]) if batch_shape is None else torch.Size(batch_shape),
    )


def merge_rff_noise_initializer_kwargs(initializer_kwargs: dict | None) -> dict:
    """Apply RFF noise init bounds unless ``raw_noise`` is already configured."""
    cfg = rff_noise_initializer_parameter_config()
    if initializer_kwargs is None:
        return {"parameter_configs": {"raw_noise": cfg}}
    out = dict(initializer_kwargs)
    pcs = dict(out.get("parameter_configs") or {})
    pcs.setdefault("raw_noise", cfg)
    out["parameter_configs"] = pcs
    return out


def merge_mt_noise_initializer_kwargs(initializer_kwargs: dict | None) -> dict:
    """Apply RFF log10 noise init bounds unless ``raw_task_noises`` is already configured."""
    cfg = rff_noise_initializer_parameter_config()
    mt_cfg = {
        **cfg,
        "description": f"Per-task noise parameter - uniform log10 scale (σ² >= {DEFAULT_NOISE_FLOOR:g})",
    }
    if initializer_kwargs is None:
        return {"parameter_configs": {"raw_task_noises": mt_cfg}}
    out = dict(initializer_kwargs)
    pcs = dict(out.get("parameter_configs") or {})
    pcs.setdefault("raw_task_noises", mt_cfg)
    out["parameter_configs"] = pcs
    return out


def rff_mll_class(woodbury_form: WoodburyForm = DEFAULT_WOODBURY_FORM) -> type:
    """Bind ``woodbury_form`` for :class:`GPTrainer` (passes ``jitter`` at runtime)."""

    class BoundRFFWoodburyMLL(RFFWoodburyMarginalLogLikelihood):
        def __init__(self, likelihood, model, jitter: float = 1e-6):
            super().__init__(
                likelihood, model, jitter=jitter, woodbury_form=woodbury_form
            )

    BoundRFFWoodburyMLL.__name__ = f"RFFWoodburyMarginalLogLikelihood_{woodbury_form}"
    BoundRFFWoodburyMLL.__qualname__ = BoundRFFWoodburyMLL.__name__
    return BoundRFFWoodburyMLL


def mt_mll_class(method: WoodburyMtMethod = DEFAULT_MT_WOODBURY_METHOD) -> type:
    """Bind multitask Woodbury ``method`` for :class:`GPTrainer`."""

    class BoundRFFMTWoodburyMLL(RFFMTWoodburyMarginalLogLikelihood):
        def __init__(self, likelihood, model, jitter: float = 1e-6):
            super().__init__(likelihood, model, jitter=jitter, method=method)

    BoundRFFMTWoodburyMLL.__name__ = f"RFFMTWoodburyMarginalLogLikelihood_{method}"
    BoundRFFMTWoodburyMLL.__qualname__ = BoundRFFMTWoodburyMLL.__name__
    return BoundRFFMTWoodburyMLL


def rff_eval_kwargs(dtype: torch.dtype) -> dict:
    return {
        "jitter": woodbury_jitter_for_dtype(dtype),
        "woodbury_form": DEFAULT_WOODBURY_FORM,
    }


def mt_eval_kwargs(dtype: torch.dtype) -> dict:
    return {
        "jitter": woodbury_jitter_for_dtype(dtype),
        "method": DEFAULT_MT_WOODBURY_METHOD,
    }
