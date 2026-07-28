"""Resolve per-QoI soft probabilistic bound configs aligned with the QOI list."""

from __future__ import annotations

from typing import Mapping, Sequence

from gpplus.training.bound_penalty import BoundConfig
from experiments_toa.s2_y_transform import YWarpConfig, task_uses_log_scale, task_uses_logit_scale


def resolve_bound_penalty_for_tasks(
    task_names: Sequence[str],
    bound_min: Sequence[float | None] | None,
    bound_max: Sequence[float | None] | None,
    *,
    warps: YWarpConfig,
    k: float = 2.0,
    lam: float = 1.0,
    alpha: float = 10.0,
    max_points: int | None = 4096,
) -> dict[str, BoundConfig | None]:
    """
    Map QOI-aligned BOUND_MIN / BOUND_MAX lists to per-task BoundConfig.

    ``None`` lists disable all soft bounds. Otherwise both lists must match
    ``len(task_names)``. A task with ``(None, None)`` gets no penalty. Soft
    bounds are rejected on log/logit-warped tasks.
    """
    n = len(task_names)
    if bound_min is None and bound_max is None:
        return {name: None for name in task_names}

    if bound_min is None:
        bound_min = [None] * n
    if bound_max is None:
        bound_max = [None] * n

    if len(bound_min) != n:
        raise ValueError(
            f"BOUND_MIN length {len(bound_min)} != number of QoIs {n}. "
            "BOUND_MIN must be parallel to QOI (same order and length)."
        )
    if len(bound_max) != n:
        raise ValueError(
            f"BOUND_MAX length {len(bound_max)} != number of QoIs {n}. "
            "BOUND_MAX must be parallel to QOI (same order and length)."
        )

    out: dict[str, BoundConfig | None] = {}
    for name, a_raw, b_raw in zip(task_names, bound_min, bound_max):
        a = None if a_raw is None else float(a_raw)
        b = None if b_raw is None else float(b_raw)
        if a is None and b is None:
            out[name] = None
            continue
        if task_uses_log_scale(name, warps=warps) or task_uses_logit_scale(name, warps=warps):
            raise ValueError(
                f"Soft bound penalty for {name!r} conflicts with a log/logit warp. "
                "Use the warp OR BOUND_MIN/BOUND_MAX for that QoI, not both."
            )
        out[name] = BoundConfig(
            a=a,
            b=b,
            k=float(k),
            lam=float(lam),
            alpha=float(alpha),
            max_points=None if max_points is None else int(max_points),
        )
    return out


def learned_bound_penalty_lambda(model) -> float | None:
    """Return effective learned λ from an RFFGPR model, or None if fixed/off."""
    raw = getattr(model, "raw_bound_penalty_lambda", None)
    if raw is None:
        return None
    return float(model.bound_penalty_lambda.reshape(-1)[0].detach().cpu())


def bound_penalty_metrics(
    bound_by_task: Mapping[str, BoundConfig | None],
    *,
    bound_penalty_lambda_learnable: bool = False,
    bound_penalty_lam_min: float | None = None,
    learned_lambda_by_task: Mapping[str, float | None] | None = None,
) -> dict:
    """Flatten bound configs for metrics JSON."""
    payload: dict = {
        "bound_penalty_tasks": [
            name for name, cfg in bound_by_task.items() if cfg is not None
        ],
        "bound_penalty_lambda_learnable": bool(bound_penalty_lambda_learnable),
    }
    if bound_penalty_lam_min is not None:
        payload["bound_penalty_lam_min"] = float(bound_penalty_lam_min)
    learned = dict(learned_lambda_by_task or {})
    for name, cfg in bound_by_task.items():
        if cfg is None:
            payload[f"{name}_bound_min"] = None
            payload[f"{name}_bound_max"] = None
            continue
        payload[f"{name}_bound_min"] = cfg.a
        payload[f"{name}_bound_max"] = cfg.b
        payload[f"{name}_bound_penalty_k"] = cfg.k
        payload[f"{name}_bound_penalty_lambda"] = cfg.lam
        payload[f"{name}_bound_penalty_lambda_init"] = cfg.lam
        payload[f"{name}_bound_penalty_alpha"] = cfg.alpha
        payload[f"{name}_bound_penalty_max_points"] = cfg.max_points
        final_lam = learned.get(name)
        if final_lam is not None:
            payload[f"{name}_bound_penalty_lambda_final"] = float(final_lam)
    return payload
