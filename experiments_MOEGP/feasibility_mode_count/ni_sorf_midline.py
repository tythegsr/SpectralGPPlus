"""Shared NI-SORF (RFFGPR + SORF + NIGP) fit/predict helpers for COSI compare.

Mirrors the single-task path in ``experiments_SORF/emit_s4_sorf_base.py`` /
``S4_emit_SORF.py`` without validation monitoring or ASD post-train.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
_RFF_DIR = _ROOT / "experiments_RFF"
_MTGPR_DIR = _ROOT / "experiments_RFFMTGPR"
for p in (_ROOT, _RFF_DIR, _MTGPR_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def fit_ni_sorf_core(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    num_rff: int = 2400,
    num_epochs: int = 1000,
    num_inits: int = 1,
    lr: float = 1e-2,
    seed: int = 42,
    device: str | None = None,
    dtype: torch.dtype = torch.float64,
    ard: bool = True,
    nigp: bool = True,
    freeze_epoch_nigp: int = 200,
    nigp_slope_refreshes: int = 10,
    correct_sorf: bool = True,
    spectral_kernel: str = "rbf",
) -> tuple[Any, dict[str, Any]]:
    """Fit ``RFFGPR(rff_sampling='sorf', nigp=True)``; return (model, info)."""
    from gpplus.models import RFFGPR
    from gpplus.training import (
        GPTrainer,
        NIGPInputNoiseFreezeCallback,
        RFFParameterInitializer,
        nigp_woodbury_mll_class,
    )
    from experiments_RFF.rff_gp_defaults import (
        DEFAULT_WOODBURY_FORM,
        build_rff_scalar_noise_likelihood,
        merge_rff_noise_initializer_kwargs,
        woodbury_jitter_for_dtype,
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    x = train_x.to(device=device, dtype=dtype)
    if x.ndim == 1:
        x = x.unsqueeze(-1)
    y = train_y.to(device=device, dtype=dtype).reshape(-1)

    likelihood = build_rff_scalar_noise_likelihood()
    initializer_kwargs = merge_rff_noise_initializer_kwargs(None)
    model = RFFGPR(
        x,
        y,
        likelihood=likelihood,
        num_rff=int(num_rff),
        ard=bool(ard),
        rff_sampling="sorf",
        correct_sorf=bool(correct_sorf),
        spectral_kernel=str(spectral_kernel),
        nigp=bool(nigp),
    )

    callbacks = []
    task_epochs = int(num_epochs)
    if nigp and int(freeze_epoch_nigp) > 0 and task_epochs > 1:
        callbacks.append(
            NIGPInputNoiseFreezeCallback(
                freeze_epochs=int(freeze_epoch_nigp),
                verbose=False,
            )
        )

    if nigp:
        nigp_mll_kwargs: dict[str, Any] = {}
        slope_r = None if nigp_slope_refreshes is None else int(nigp_slope_refreshes)
        if slope_r is not None:
            nigp_mll_kwargs["nigp_slope_refreshes"] = slope_r
            nigp_mll_kwargs["nigp_active_epochs"] = max(
                1, task_epochs - max(0, int(freeze_epoch_nigp))
            )
        mll_class = nigp_woodbury_mll_class(DEFAULT_WOODBURY_FORM, **nigp_mll_kwargs)
    else:
        from experiments_RFF.rff_gp_defaults import rff_mll_class

        mll_class = rff_mll_class()

    optimizer_kwargs = {
        "lr": float(lr),
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.0,
        "amsgrad": False,
    }
    trainer = GPTrainer(
        model,
        mll_class=mll_class,
        num_epochs=task_epochs,
        num_inits=int(num_inits),
        seed=int(seed),
        device=device,
        dtype=dtype,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs=optimizer_kwargs,
        initializer_class=RFFParameterInitializer,
        initializer_kwargs=initializer_kwargs,
        n_jobs=1,
        inner_max_num_threads=1,
        cholesky_jitter=woodbury_jitter_for_dtype(dtype),
        callbacks=callbacks,
        parallel_verbose=0,
        min_epochs=(
            int(freeze_epoch_nigp)
            if nigp and int(freeze_epoch_nigp) > 0 and task_epochs > 1
            else 0
        ),
    )
    runs = trainer.train()
    model = trainer.model
    successful = [
        r for r in runs if r.get("loss") is not None and r.get("state_dict") is not None
    ]
    if not successful:
        errors = [r.get("error", "unknown") for r in runs if r.get("error")]
        raise RuntimeError(
            "All NI-SORF training runs failed. "
            + (f"First error: {errors[0]}" if errors else "")
        )
    best = min(successful, key=lambda r: r["loss"])
    model.load_state_dict(best["state_dict"])
    model.eval()
    info = {
        "best_loss": float(best["loss"]),
        "num_rff": int(num_rff),
        "num_epochs": task_epochs,
        "device": device,
        "nigp": bool(nigp),
        "model": "ni_sorf",
    }
    return model, info


@torch.no_grad()
def predict_ni_sorf(
    model: Any,
    test_x: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (mean, std) on ``test_x`` via ``evaluate_rff_gp_model``."""
    from gpplus.training import evaluate_rff_gp_model

    x = test_x.to(dtype=dtype)
    if x.ndim == 1:
        x = x.unsqueeze(-1)
    # Keep features on same device as model train_inputs when possible
    try:
        train_x = model.train_inputs[0]
        x = x.to(device=train_x.device, dtype=train_x.dtype)
    except Exception:  # noqa: BLE001
        pass
    mean, _lo, _hi, std = evaluate_rff_gp_model(model, x)
    return mean.reshape(-1).to(dtype=dtype).cpu(), std.reshape(-1).clamp_min(1e-4).to(
        dtype=dtype
    ).cpu()


def fit_predict_ni_sorf(
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_te: torch.Tensor,
    *,
    seed: int = 42,
    device: str | None = None,
    num_rff: int = 2400,
    num_epochs: int = 1000,
    num_inits: int = 1,
    lr: float = 1e-2,
    **kwargs: Any,
) -> dict[str, Any]:
    """Standalone NI-SORF → ScoringBench K=1 mixture dict."""
    import time

    t0 = time.perf_counter()
    model, info = fit_ni_sorf_core(
        x_tr,
        y_tr,
        num_rff=num_rff,
        num_epochs=num_epochs,
        num_inits=num_inits,
        lr=lr,
        seed=seed,
        device=device,
        **{k: v for k, v in kwargs.items() if k in {
            "dtype", "ard", "nigp", "freeze_epoch_nigp",
            "nigp_slope_refreshes", "correct_sorf", "spectral_kernel",
        }},
    )
    mean, std = predict_ni_sorf(model, x_te)
    wall = time.perf_counter() - t0
    mu = mean.detach().cpu().numpy().ravel()
    sd = np.clip(std.detach().cpu().numpy().ravel(), 1e-6, None)
    n = mu.shape[0]
    return {
        "mixture_mean": mu,
        "expert_means": mu.reshape(n, 1),
        "expert_stds": sd.reshape(n, 1),
        "gate_weights": np.ones((n, 1), dtype=np.float64),
        "extras": {**info, "wall_s": wall},
        "_model": model,
    }
