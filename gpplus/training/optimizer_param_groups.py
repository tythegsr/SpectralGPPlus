"""Helpers for building Adam parameter groups by hyperparameter role."""

from __future__ import annotations

from torch import nn

_NOISE_NAME_TOKENS = ("raw_noise", "raw_task_noises")


def _is_noise_param(name: str) -> bool:
    return any(token in name for token in _NOISE_NAME_TOKENS)


def adam_noise_vs_other_param_groups(
    model: nn.Module,
    *,
    noise_lr: float = 0.01,
    other_lr: float = 0.1,
    **adam_kwargs,
) -> list[dict]:
    """
    Split trainable params into noise vs everything else for Adam.

    Noise group matches names containing ``raw_noise`` or ``raw_task_noises``.
    Shared Adam kwargs (``betas``, ``eps``, ...) apply to both groups.
    """
    noise_params: list = []
    other_params: list = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _is_noise_param(name):
            noise_params.append(param)
        else:
            other_params.append(param)

    if not noise_params:
        raise ValueError(
            "No trainable noise parameters found "
            f"(expected names containing {_NOISE_NAME_TOKENS})."
        )
    if not other_params:
        raise ValueError("No trainable non-noise parameters found for the other LR group.")

    # Per-group lr must win over any lr accidentally passed in adam_kwargs.
    return [
        {"params": noise_params, **adam_kwargs, "lr": float(noise_lr)},
        {"params": other_params, **adam_kwargs, "lr": float(other_lr)},
    ]
