"""Shared runner for Mixture-of-Experts GP on ScoringBench bimodal data."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
_EXP_DIR = Path(__file__).resolve().parent
for p in (_ROOT, _EXP_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import gpplus
from gpplus.models import MOEGPR
from gpplus.training import evaluate_moe_gp_model

from load_bimodal_data import (
    BIMODAL_NOISE_STD,
    train_test_split_bimodal,
)
from plot_cdf_ks import save_cdf_ks_comparison_plot
from plot_moe_gp1d import (
    save_conditional_pdf_plot,
    save_gate_weights_plot,
    save_moe_posterior_plot,
)
from scoringbench_metrics import (
    SCORINGBENCH_BIMODAL_EXAMPLE_BASELINES,
    compute_scoringbench_style_metrics,
    metrics_table_with_baselines,
    print_metrics_table,
)


def _json_default(obj: Any):
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def run_bimodal_moe_gp(
    *,
    n_samples: int = 1000,
    test_size: float = 0.3,
    num_experts: int = 2,
    gating_hidden: Sequence[int] = (64, 64),
    assignment_init: str = "residual",
    fit_mode: str = "hard",
    gate_type: str | None = None,
    load_balance_weight: float = 0.1,
    entropy_weight: float = 0.01,
    diversity_weight: float = 0.1,
    orthogonality_weight: float = 3.0,
    orth_refit_threshold: float = 0.25,
    responsibility_temperature: float = 0.5,
    soft_init_peak: float = 0.95,
    num_em_iters: int = 5,
    num_inits: int = 8,
    num_epochs: int = 1,
    gate_epochs: int = 200,
    gate_lr: float = 1e-2,
    seed: int = 42,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    n_jobs: int | None = 1,
    save_path: str = "experiments_MOEGP/results/bimodal_moe_gp",
    plot: bool = True,
    cond_x: float = 9.0,
    grid_size: int = 200,
) -> dict:
    """Train MOEGPR on bimodal toy data, save predictions/metrics/plots."""
    gpplus.config.configure_logger()
    torch.manual_seed(seed)
    np.random.seed(seed)

    if fit_mode not in ("hard", "soft"):
        raise ValueError(f"fit_mode must be 'hard' or 'soft', got {fit_mode!r}")
    # Soft co-located modes: constant gate is the right inductive bias (π independent of x).
    resolved_gate_type = gate_type
    if resolved_gate_type is None:
        resolved_gate_type = "constant" if fit_mode == "soft" else "mlp"

    data = train_test_split_bimodal(n_samples=n_samples, test_size=test_size, seed=seed)
    x_train = torch.as_tensor(data["X_train"], dtype=dtype)
    y_train = torch.as_tensor(data["y_train"], dtype=dtype)
    x_test = torch.as_tensor(data["X_test"], dtype=dtype)
    y_test = torch.as_tensor(data["y_test"], dtype=dtype)

    model = MOEGPR(
        x_train,
        y_train,
        num_experts=num_experts,
        gating_hidden=tuple(gating_hidden),
        gate_type=resolved_gate_type,  # type: ignore[arg-type]
        assignment_init=assignment_init,  # type: ignore[arg-type]
        ard=True,
        load_balance_weight=load_balance_weight,
        entropy_weight=entropy_weight,
        diversity_weight=diversity_weight,
        orthogonality_weight=orthogonality_weight,
        orth_refit_threshold=orth_refit_threshold,
        responsibility_temperature=responsibility_temperature,
        soft_init_peak=soft_init_peak,
    )
    history = model.fit(
        fit_mode=fit_mode,  # type: ignore[arg-type]
        num_em_iters=num_em_iters,
        num_inits=num_inits,
        num_epochs=num_epochs,
        gate_epochs=gate_epochs,
        gate_lr=gate_lr,
        seed=seed,
        device=device,
        n_jobs=n_jobs,
    )

    # Dense grid for visualization
    x_lo = float(min(data["X_train"].min(), data["X_test"].min()))
    x_hi = float(max(data["X_train"].max(), data["X_test"].max()))
    x_grid = torch.linspace(x_lo, x_hi, grid_size, dtype=dtype).unsqueeze(-1)

    mean, lower, upper, stddev, pred_grid = evaluate_moe_gp_model(model, x_grid)
    pred_test = model.predict(x_test)
    log_prob_test = model.predictive_log_prob(x_test, y_test)

    mix_mean_test = pred_test["mixture_mean"].detach().cpu().numpy()
    y_test_np = y_test.detach().cpu().numpy().ravel()
    em_test = pred_test["expert_means"].detach().cpu().numpy()
    es_test = pred_test["expert_stds"].detach().cpu().numpy()
    gw_test = pred_test["gate_weights"].detach().cpu().numpy()
    log_prob_np = log_prob_test.detach().cpu().numpy()
    score_metrics = compute_scoringbench_style_metrics(
        y_test_np,
        mix_mean_test,
        em_test,
        es_test,
        gw_test,
        log_prob=log_prob_np,
    )
    comparison_table = metrics_table_with_baselines(score_metrics, moe_name="MoE-GP")

    gate_w = pred_grid["gate_weights"].detach().cpu().numpy()
    gate_sums = gate_w.sum(axis=1)
    gate_ok = bool(np.allclose(gate_sums, 1.0, atol=1e-4) and np.all(gate_w >= -1e-8))
    mean_pi = gate_w.mean(axis=0)
    gate_entropy = float(
        (-np.clip(gate_w, 1e-12, None) * np.log(np.clip(gate_w, 1e-12, None)))
        .sum(axis=1)
        .mean()
    )

    file_tag = (
        f"n{n_samples}_K{num_experts}_{fit_mode}_{resolved_gate_type}"
        f"_{assignment_init}_seed{seed}_em{num_em_iters}"
    )
    os.makedirs(save_path, exist_ok=True)

    npz_path = os.path.join(save_path, f"predictions_1d_{file_tag}.npz")
    soft_r = None
    if getattr(model, "_soft_responsibilities_last", None) is not None:
        soft_r = model._soft_responsibilities_last.detach().cpu().numpy()
    np.savez(
        npz_path,
        x_train=data["X_train"],
        y_train=data["y_train"],
        x_test=data["X_test"],
        y_test=data["y_test"],
        mode_train=data["mode_train"],
        mode_test=data["mode_test"],
        x_grid=x_grid.detach().cpu().numpy(),
        mixture_mean=pred_grid["mixture_mean"].detach().cpu().numpy(),
        mixture_std=pred_grid["mixture_std"].detach().cpu().numpy(),
        expert_means=pred_grid["expert_means"].detach().cpu().numpy(),
        expert_stds=pred_grid["expert_stds"].detach().cpu().numpy(),
        gate_weights=gate_w,
        mix_mean_test=mix_mean_test,
        log_prob_test=log_prob_np,
        soft_responsibilities=soft_r if soft_r is not None else np.array([]),
    )

    metrics = {
        "model_class": "MOEGPR",
        "num_experts": num_experts,
        "fit_mode": fit_mode,
        "gate_type": resolved_gate_type,
        "assignment_init": assignment_init,
        "load_balance_weight": load_balance_weight,
        "entropy_weight": entropy_weight,
        "diversity_weight": diversity_weight,
        "orthogonality_weight": orthogonality_weight,
        "orth_refit_threshold": orth_refit_threshold,
        "responsibility_temperature": responsibility_temperature,
        "soft_init_peak": soft_init_peak,
        "n_samples": n_samples,
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
        "seed": seed,
        "num_em_iters": num_em_iters,
        "num_inits": num_inits,
        "num_epochs": num_epochs,
        "gate_epochs": gate_epochs,
        "gate_lr": gate_lr,
        "gating_hidden": list(gating_hidden),
        "mean_pi": [float(v) for v in mean_pi.tolist()],
        "mean_gate_entropy": gate_entropy,
        **score_metrics,
        "scoringbench_bimodal_baselines": SCORINGBENCH_BIMODAL_EXAMPLE_BASELINES,
        "comparison_table": comparison_table,
        "gate_weights_simplex_ok": gate_ok,
        "gate_weight_sum_max_abs_err": float(np.max(np.abs(gate_sums - 1.0))),
        "em_history": history,
        "bimodal_noise_std": BIMODAL_NOISE_STD,
    }
    title = f"bimodal_moe_K{num_experts}_{fit_mode}_seed{seed}"
    json_path = os.path.join(save_path, f"gp_{title}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=_json_default)

    plot_paths: list[str] = []
    if plot:
        plot_dir = os.path.join(save_path, "plots", "prediction_runs")
        em = pred_grid["expert_means"].detach().cpu().numpy()
        es = pred_grid["expert_stds"].detach().cpu().numpy()
        mm = pred_grid["mixture_mean"].detach().cpu().numpy()
        xg = x_grid.detach().cpu().numpy()

        p1 = save_moe_posterior_plot(
            data["X_train"],
            data["y_train"],
            xg,
            em,
            es,
            mm,
            plot_dir,
            file_tag=file_tag,
        )
        p2 = save_gate_weights_plot(xg, gate_w, plot_dir, file_tag=file_tag)

        # Conditional PDF at cond_x: interpolate nearest grid point
        idx = int(np.argmin(np.abs(xg.ravel() - cond_x)))
        p3 = save_conditional_pdf_plot(
            float(xg.ravel()[idx]),
            em[idx],
            es[idx],
            gate_w[idx],
            plot_dir,
            file_tag=file_tag,
        )
        p4, ks_distances = save_cdf_ks_comparison_plot(
            data["X_test"],
            data["y_test"],
            em[idx],
            es[idx],
            gate_w[idx],
            plot_dir,
            file_tag=file_tag,
            x_value=float(xg.ravel()[idx]),
            model_label="MoE-GP",
        )
        plot_paths = [str(p1), str(p2), str(p3), str(p4)]
        metrics["plot_paths"] = plot_paths
        metrics["ks_distances_to_theory"] = ks_distances
        metrics["ks_eval_x"] = float(xg.ravel()[idx])
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=_json_default)

    metrics["npz_path"] = npz_path
    metrics["json_path"] = json_path
    print(f"Saved metrics -> {json_path}")
    print_metrics_table(comparison_table)
    print(
        f"fit_mode={fit_mode} gate_type={resolved_gate_type} "
        f"RRMSE={score_metrics['RRMSE']:.4f}  gate_simplex_ok={gate_ok}  "
        f"mean_pi={metrics['mean_pi']}  H(pi)={gate_entropy:.4f}"
    )
    return metrics
