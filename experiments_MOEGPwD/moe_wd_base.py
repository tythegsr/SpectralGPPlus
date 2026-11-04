"""Shared runner for multimodal / sharp-edge MoE-GP experiments (MOEGPwD)."""

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

from load_bimodal_data import BIMODAL_NOISE_STD, train_test_split_bimodal
from load_step_data import DEFAULT_STEP_VALUES, generate_step_data, step_function
from plot_cdf_ks import save_cdf_ks_comparison_plot
from plot_moe_wd import (
    expert_vs_true_line_rmse,
    gate_entropy,
    mode_purity,
    save_assignment_scatter,
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
    if isinstance(obj, dict):
        return {str(k): v for k, v in obj.items()}
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _fit_and_predict(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    num_experts: int,
    gate_type: str,
    temperature: float,
    assignment_init: str,
    gating_hidden: Sequence[int],
    num_em_iters: int,
    num_inits: int,
    num_epochs: int,
    gate_epochs: int,
    gate_lr: float,
    seed: int,
    device: str,
    dtype: torch.dtype,
    n_jobs: int | None,
    grid_size: int,
) -> tuple[MOEGPR, dict, dict, torch.Tensor, dict]:
    model = MOEGPR(
        x_train,
        y_train,
        num_experts=num_experts,
        gating_hidden=tuple(gating_hidden),
        gate_type=gate_type,  # type: ignore[arg-type]
        temperature=temperature,
        assignment_init=assignment_init,  # type: ignore[arg-type]
        ard=True,
    )
    history = model.fit(
        num_em_iters=num_em_iters,
        num_inits=num_inits,
        num_epochs=num_epochs,
        gate_epochs=gate_epochs,
        gate_lr=gate_lr,
        seed=seed,
        device=device,
        n_jobs=n_jobs,
    )

    x_lo = float(min(x_train.min().item(), x_test.min().item()))
    x_hi = float(max(x_train.max().item(), x_test.max().item()))
    x_grid = torch.linspace(x_lo, x_hi, grid_size, dtype=dtype).unsqueeze(-1)
    _mean, _lo, _hi, _std, pred_grid = evaluate_moe_gp_model(model, x_grid)
    pred_test = model.predict(x_test)
    log_prob_test = model.predictive_log_prob(x_test, y_test)
    return model, history, pred_grid, x_grid, {
        "pred_test": pred_test,
        "log_prob_test": log_prob_test,
    }


def _point_metrics(mix_mean: np.ndarray, y_true: np.ndarray, log_prob: np.ndarray) -> dict[str, float]:
    """Lightweight point metrics (used by step toy without mixture scoring suite)."""
    mse = float(np.mean((mix_mean - y_true) ** 2))
    rmse = float(np.sqrt(mse))
    y_std = float(np.std(y_true))
    rrmse = float(rmse / max(y_std, 1e-12)) if np.isfinite(y_std) else float("nan")
    mae = float(np.mean(np.abs(mix_mean - y_true)))
    nll = float(-np.mean(log_prob))
    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "RRMSE": rrmse,
        "mse_mixture_mean": mse,
        "rmse_mixture_mean": rmse,
        "mae_mixture_mean": mae,
        "rrmse_mixture_mean": rrmse,
        "mixture_nll": nll,
        "log_score": nll,
    }


def run_bimodal_constant_gate(
    *,
    n_samples: int = 1000,
    test_size: float = 0.3,
    num_experts: int = 2,
    temperature: float = 1.0,
    gating_hidden: Sequence[int] = (64, 64),
    assignment_init: str = "residual",
    num_em_iters: int = 5,
    num_inits: int = 8,
    num_epochs: int = 1,
    gate_epochs: int = 200,
    gate_lr: float = 1e-2,
    seed: int = 42,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    n_jobs: int | None = 1,
    save_path: str = "experiments_MOEGPwD/results/bimodal_constant_gate",
    plot: bool = True,
    cond_x: float = 9.0,
    grid_size: int = 200,
) -> dict:
    """A01: ScoringBench bimodal with constant gate (co-located modes)."""
    gpplus.config.configure_logger()
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = train_test_split_bimodal(n_samples=n_samples, test_size=test_size, seed=seed)
    x_train = torch.as_tensor(data["X_train"], dtype=dtype)
    y_train = torch.as_tensor(data["y_train"], dtype=dtype)
    x_test = torch.as_tensor(data["X_test"], dtype=dtype)
    y_test = torch.as_tensor(data["y_test"], dtype=dtype)

    model, history, pred_grid, x_grid, extra = _fit_and_predict(
        x_train,
        y_train,
        x_test,
        y_test,
        num_experts=num_experts,
        gate_type="constant",
        temperature=temperature,
        assignment_init=assignment_init,
        gating_hidden=gating_hidden,
        num_em_iters=num_em_iters,
        num_inits=num_inits,
        num_epochs=num_epochs,
        gate_epochs=gate_epochs,
        gate_lr=gate_lr,
        seed=seed,
        device=device,
        dtype=dtype,
        n_jobs=n_jobs,
        grid_size=grid_size,
    )

    pred_test = extra["pred_test"]
    log_prob_test = extra["log_prob_test"].detach().cpu().numpy()
    mix_mean_test = pred_test["mixture_mean"].detach().cpu().numpy()
    y_test_np = y_test.detach().cpu().numpy().ravel()
    score_metrics = compute_scoringbench_style_metrics(
        y_test_np,
        mix_mean_test,
        pred_test["expert_means"].detach().cpu().numpy(),
        pred_test["expert_stds"].detach().cpu().numpy(),
        pred_test["gate_weights"].detach().cpu().numpy(),
        log_prob=log_prob_test,
    )
    comparison_table = metrics_table_with_baselines(score_metrics, moe_name="MoE-GP-const")

    gate_w = pred_grid["gate_weights"].detach().cpu().numpy()
    ent = gate_entropy(gate_w)
    assignments = model._assignments.detach().cpu().numpy() if model._assignments is not None else None
    purity = mode_purity(assignments, data["mode_train"], num_experts) if assignments is not None else {}
    line_rmse = expert_vs_true_line_rmse(
        x_grid.detach().cpu().numpy(),
        pred_grid["expert_means"].detach().cpu().numpy(),
    )

    gate_sums = gate_w.sum(axis=1)
    gate_ok = bool(np.allclose(gate_sums, 1.0, atol=1e-4) and np.all(gate_w >= -1e-8))

    file_tag = (
        f"n{n_samples}_K{num_experts}_const_{assignment_init}"
        f"_T{temperature:g}_seed{seed}_em{num_em_iters}"
    )
    os.makedirs(save_path, exist_ok=True)
    npz_path = os.path.join(save_path, f"predictions_1d_{file_tag}.npz")
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
        assignments=assignments,
        gate_entropy=ent,
        mix_mean_test=mix_mean_test,
        log_prob_test=log_prob_test,
    )

    metrics: dict[str, Any] = {
        "model_class": "MOEGPR",
        "experiment": "bimodal_constant_gate",
        "gate_type": "constant",
        "assignment_init": assignment_init,
        "temperature": temperature,
        "num_experts": num_experts,
        "n_samples": n_samples,
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
        "seed": seed,
        "num_em_iters": num_em_iters,
        "num_inits": num_inits,
        "gate_epochs": gate_epochs,
        **score_metrics,
        "scoringbench_bimodal_baselines": SCORINGBENCH_BIMODAL_EXAMPLE_BASELINES,
        "comparison_table": comparison_table,
        "mean_gate_entropy": float(ent.mean()),
        "min_gate_entropy": float(ent.min()),
        "max_gate_entropy": float(ent.max()),
        "gate_weights_simplex_ok": gate_ok,
        "gate_weight_sum_max_abs_err": float(np.max(np.abs(gate_sums - 1.0))),
        "mean_gate_weights": gate_w.mean(axis=0).tolist(),
        **purity,
        **line_rmse,
        "em_history": history,
        "bimodal_noise_std": BIMODAL_NOISE_STD,
    }

    title = f"bimodal_const_K{num_experts}_seed{seed}"
    json_path = os.path.join(save_path, f"gp_{title}.json")
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
            title="MoE-GP (constant gate) — bimodal toy",
            show_bimodal_truth=True,
        )
        p2 = save_gate_weights_plot(
            xg,
            gate_w,
            plot_dir,
            file_tag=file_tag,
            title="Constant gate weights (∑π=1)",
        )
        idx = int(np.argmin(np.abs(xg.ravel() - cond_x)))
        p3 = save_conditional_pdf_plot(
            float(xg.ravel()[idx]),
            em[idx],
            es[idx],
            gate_w[idx],
            plot_dir,
            file_tag=file_tag,
            show_bimodal_theory=True,
        )
        p_cdf, ks_distances = save_cdf_ks_comparison_plot(
            data["X_test"],
            data["y_test"],
            em[idx],
            es[idx],
            gate_w[idx],
            plot_dir,
            file_tag=file_tag,
            x_value=float(xg.ravel()[idx]),
            model_label="MoE-GP-const",
        )
        plot_paths = [str(p1), str(p2), str(p3), str(p_cdf)]
        metrics["ks_distances_to_theory"] = ks_distances
        metrics["ks_eval_x"] = float(xg.ravel()[idx])
        if assignments is not None:
            p4 = save_assignment_scatter(
                data["X_train"],
                data["y_train"],
                assignments,
                plot_dir,
                file_tag=file_tag,
                true_mode_mask=data["mode_train"],
                title="Hard assignments vs true modes",
            )
            plot_paths.append(str(p4))
        metrics["plot_paths"] = plot_paths

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=_json_default)

    metrics["npz_path"] = npz_path
    metrics["json_path"] = json_path
    print(f"Saved metrics -> {json_path}")
    print_metrics_table(comparison_table)
    print(
        f"RRMSE={score_metrics['RRMSE']:.4f}  "
        f"purity={purity.get('mode_purity', float('nan')):.4f}  "
        f"H(pi)={float(ent.mean()):.4f}  gate_ok={gate_ok}"
    )
    return metrics


def run_bimodal_quadrant_experts(
    *,
    n_samples: int = 1000,
    test_size: float = 0.3,
    temperature: float = 0.5,
    gating_hidden: Sequence[int] = (64, 64),
    num_em_iters: int = 5,
    num_inits: int = 8,
    num_epochs: int = 1,
    gate_epochs: int = 300,
    gate_lr: float = 1e-2,
    seed: int = 42,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    n_jobs: int | None = 1,
    save_path: str = "experiments_MOEGPwD/results/bimodal_quadrant_experts",
    plot: bool = True,
    cond_x: float = 9.0,
    grid_size: int = 200,
) -> dict:
    """A03: bimodal X as four line segments (TL/BL/TR/BR) with MLP gate.

    At each x, two experts should be active (upper and lower mode in that
    half of the domain). The gate must be x-dependent so left experts are
    suppressed on the right and vice versa.
    """
    gpplus.config.configure_logger()
    torch.manual_seed(seed)
    np.random.seed(seed)
    num_experts = 4

    data = train_test_split_bimodal(n_samples=n_samples, test_size=test_size, seed=seed)
    x_train = torch.as_tensor(data["X_train"], dtype=dtype)
    y_train = torch.as_tensor(data["y_train"], dtype=dtype)
    x_test = torch.as_tensor(data["X_test"], dtype=dtype)
    y_test = torch.as_tensor(data["y_test"], dtype=dtype)

    model, history, pred_grid, x_grid, extra = _fit_and_predict(
        x_train,
        y_train,
        x_test,
        y_test,
        num_experts=num_experts,
        gate_type="mlp",
        temperature=temperature,
        assignment_init="quadrant",
        gating_hidden=gating_hidden,
        num_em_iters=num_em_iters,
        num_inits=num_inits,
        num_epochs=num_epochs,
        gate_epochs=gate_epochs,
        gate_lr=gate_lr,
        seed=seed,
        device=device,
        dtype=dtype,
        n_jobs=n_jobs,
        grid_size=grid_size,
    )

    pred_test = extra["pred_test"]
    log_prob_test = extra["log_prob_test"].detach().cpu().numpy()
    mix_mean_test = pred_test["mixture_mean"].detach().cpu().numpy()
    y_test_np = y_test.detach().cpu().numpy().ravel()
    score_metrics = compute_scoringbench_style_metrics(
        y_test_np,
        mix_mean_test,
        pred_test["expert_means"].detach().cpu().numpy(),
        pred_test["expert_stds"].detach().cpu().numpy(),
        pred_test["gate_weights"].detach().cpu().numpy(),
        log_prob=log_prob_test,
    )
    comparison_table = metrics_table_with_baselines(score_metrics, moe_name="MoE-GP-quad")

    gate_w = pred_grid["gate_weights"].detach().cpu().numpy()
    ent = gate_entropy(gate_w)
    assignments = model._assignments.detach().cpu().numpy() if model._assignments is not None else None
    purity = mode_purity(assignments, data["mode_train"], num_experts) if assignments is not None else {}
    line_rmse = expert_vs_true_line_rmse(
        x_grid.detach().cpu().numpy(),
        pred_grid["expert_means"].detach().cpu().numpy(),
    )

    xg = x_grid.detach().cpu().numpy().ravel()
    x_med = float(np.median(data["X_train"].ravel()))
    left_mask = xg < x_med
    right_mask = ~left_mask
    # Encoding: 0=TL, 1=BL, 2=TR, 3=BR
    left_pair_mass = float(gate_w[left_mask][:, [0, 1]].sum(axis=1).mean()) if left_mask.any() else float("nan")
    right_pair_mass = float(gate_w[right_mask][:, [2, 3]].sum(axis=1).mean()) if right_mask.any() else float("nan")

    gate_sums = gate_w.sum(axis=1)
    gate_ok = bool(np.allclose(gate_sums, 1.0, atol=1e-4) and np.all(gate_w >= -1e-8))

    file_tag = f"n{n_samples}_K4_quad_T{temperature:g}_seed{seed}_em{num_em_iters}"
    os.makedirs(save_path, exist_ok=True)
    npz_path = os.path.join(save_path, f"predictions_1d_{file_tag}.npz")
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
        assignments=assignments,
        gate_entropy=ent,
        mix_mean_test=mix_mean_test,
        log_prob_test=log_prob_test,
        x_median_split=x_med,
    )

    metrics: dict[str, Any] = {
        "model_class": "MOEGPR",
        "experiment": "bimodal_quadrant_experts",
        "gate_type": "mlp",
        "assignment_init": "quadrant",
        "temperature": temperature,
        "num_experts": num_experts,
        "quadrant_labels": {"0": "TL", "1": "BL", "2": "TR", "3": "BR"},
        "n_samples": n_samples,
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
        "seed": seed,
        "num_em_iters": num_em_iters,
        "num_inits": num_inits,
        "gate_epochs": gate_epochs,
        **score_metrics,
        "scoringbench_bimodal_baselines": SCORINGBENCH_BIMODAL_EXAMPLE_BASELINES,
        "comparison_table": comparison_table,
        "mean_gate_entropy": float(ent.mean()),
        "min_gate_entropy": float(ent.min()),
        "max_gate_entropy": float(ent.max()),
        "left_pair_gate_mass": left_pair_mass,
        "right_pair_gate_mass": right_pair_mass,
        "x_median_split": x_med,
        "gate_weights_simplex_ok": gate_ok,
        "gate_weight_sum_max_abs_err": float(np.max(np.abs(gate_sums - 1.0))),
        "mean_gate_weights": gate_w.mean(axis=0).tolist(),
        **purity,
        **line_rmse,
        "em_history": history,
        "bimodal_noise_std": BIMODAL_NOISE_STD,
    }

    title = f"bimodal_quad_K4_T{temperature:g}_seed{seed}"
    json_path = os.path.join(save_path, f"gp_{title}.json")
    plot_paths: list[str] = []
    if plot:
        plot_dir = os.path.join(save_path, "plots", "prediction_runs")
        em = pred_grid["expert_means"].detach().cpu().numpy()
        es = pred_grid["expert_stds"].detach().cpu().numpy()
        mm = pred_grid["mixture_mean"].detach().cpu().numpy()
        xg2 = x_grid.detach().cpu().numpy()
        p1 = save_moe_posterior_plot(
            data["X_train"],
            data["y_train"],
            xg2,
            em,
            es,
            mm,
            plot_dir,
            file_tag=file_tag,
            title="MoE-GP (4 quadrant experts) — bimodal X",
            show_bimodal_truth=True,
        )
        p2 = save_gate_weights_plot(
            xg2,
            gate_w,
            plot_dir,
            file_tag=file_tag,
            title=f"MLP gate (quadrants, T={temperature:g})",
        )
        idx = int(np.argmin(np.abs(xg2.ravel() - cond_x)))
        p3 = save_conditional_pdf_plot(
            float(xg2.ravel()[idx]),
            em[idx],
            es[idx],
            gate_w[idx],
            plot_dir,
            file_tag=file_tag,
            show_bimodal_theory=True,
        )
        p_cdf, ks_distances = save_cdf_ks_comparison_plot(
            data["X_test"],
            data["y_test"],
            em[idx],
            es[idx],
            gate_w[idx],
            plot_dir,
            file_tag=file_tag,
            x_value=float(xg2.ravel()[idx]),
            model_label="MoE-GP-quad",
        )
        plot_paths = [str(p1), str(p2), str(p3), str(p_cdf)]
        metrics["ks_distances_to_theory"] = ks_distances
        metrics["ks_eval_x"] = float(xg2.ravel()[idx])
        if assignments is not None:
            p4 = save_assignment_scatter(
                data["X_train"],
                data["y_train"],
                assignments,
                plot_dir,
                file_tag=file_tag,
                true_mode_mask=data["mode_train"],
                title="Quadrant assignments (0=TL,1=BL,2=TR,3=BR)",
            )
            plot_paths.append(str(p4))
        metrics["plot_paths"] = plot_paths

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=_json_default)

    metrics["npz_path"] = npz_path
    metrics["json_path"] = json_path
    print(f"Saved metrics -> {json_path}")
    print_metrics_table(comparison_table)
    print(
        f"RRMSE={score_metrics['RRMSE']:.4f}  "
        f"purity={purity.get('mode_purity', float('nan')):.4f}  "
        f"left_pair={left_pair_mass:.3f}  right_pair={right_pair_mass:.3f}  "
        f"H(pi)={float(ent.mean()):.4f}"
    )
    return metrics


def run_step_sharp_gate(
    *,
    n_train: int = 200,
    n_test: int = 500,
    num_experts: int = 4,
    temperature: float = 0.1,
    gating_hidden: Sequence[int] = (64, 64),
    step_values: tuple[float, ...] = DEFAULT_STEP_VALUES,
    x_bounds: tuple[float, float] = (-0.5, 0.5),
    noise_train: float = 0.02,
    noise_test: float = 0.02,
    num_em_iters: int = 5,
    num_inits: int = 8,
    num_epochs: int = 1,
    gate_epochs: int = 300,
    gate_lr: float = 1e-2,
    seed: int = 42,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    n_jobs: int | None = 1,
    save_path: str = "experiments_MOEGPwD/results/step_sharp_gate",
    plot: bool = True,
    grid_size: int = 300,
) -> dict:
    """A02: 1D step function with sharp MLP gate (low temperature)."""
    gpplus.config.configure_logger()
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = generate_step_data(
        n_train=n_train,
        n_test=n_test,
        seed=seed,
        x_bounds=x_bounds,
        step_values=step_values,
        noise_train=noise_train,
        noise_test=noise_test,
    )
    x_train = torch.as_tensor(data["X_train"], dtype=dtype)
    y_train = torch.as_tensor(data["y_train"], dtype=dtype)
    x_test = torch.as_tensor(data["X_test"], dtype=dtype)
    y_test = torch.as_tensor(data["y_test"], dtype=dtype)

    model, history, pred_grid, x_grid, extra = _fit_and_predict(
        x_train,
        y_train,
        x_test,
        y_test,
        num_experts=num_experts,
        gate_type="mlp",
        temperature=temperature,
        assignment_init="x",
        gating_hidden=gating_hidden,
        num_em_iters=num_em_iters,
        num_inits=num_inits,
        num_epochs=num_epochs,
        gate_epochs=gate_epochs,
        gate_lr=gate_lr,
        seed=seed,
        device=device,
        dtype=dtype,
        n_jobs=n_jobs,
        grid_size=grid_size,
    )

    pred_test = extra["pred_test"]
    log_prob_test = extra["log_prob_test"].detach().cpu().numpy()
    mix_mean_test = pred_test["mixture_mean"].detach().cpu().numpy()
    # Point metrics vs noisy test y and vs true step
    point = _point_metrics(mix_mean_test, y_test.numpy().ravel(), log_prob_test)
    point_true = _point_metrics(
        mix_mean_test,
        data["y_test_true"],
        log_prob_test,
    )

    gate_w = pred_grid["gate_weights"].detach().cpu().numpy()
    ent = gate_entropy(gate_w)
    assignments = model._assignments.detach().cpu().numpy() if model._assignments is not None else None

    # Region purity: majority-vote map expert -> true step bin
    region_purity: dict[str, Any] = {}
    if assignments is not None:
        bin_train = data["bin_train"]
        expert_to_bin: dict[int, int] = {}
        for k in range(num_experts):
            mask = assignments == k
            if mask.any():
                vals, counts = np.unique(bin_train[mask], return_counts=True)
                expert_to_bin[k] = int(vals[counts.argmax()])
            else:
                expert_to_bin[k] = 0
        pred_bin = np.array([expert_to_bin[int(a)] for a in assignments])
        region_purity = {
            "region_purity": float((pred_bin == bin_train).mean()),
            "expert_to_bin": {str(k): v for k, v in expert_to_bin.items()},
        }

    gate_sums = gate_w.sum(axis=1)
    gate_ok = bool(np.allclose(gate_sums, 1.0, atol=1e-4) and np.all(gate_w >= -1e-8))

    file_tag = f"n{n_train}_K{num_experts}_mlp_T{temperature:g}_seed{seed}_em{num_em_iters}"
    os.makedirs(save_path, exist_ok=True)
    npz_path = os.path.join(save_path, f"predictions_1d_{file_tag}.npz")
    np.savez(
        npz_path,
        x_train=data["X_train"],
        y_train=data["y_train"],
        x_test=data["X_test"],
        y_test=data["y_test"],
        y_test_true=data["y_test_true"],
        bin_train=data["bin_train"],
        x_grid=x_grid.detach().cpu().numpy(),
        mixture_mean=pred_grid["mixture_mean"].detach().cpu().numpy(),
        mixture_std=pred_grid["mixture_std"].detach().cpu().numpy(),
        expert_means=pred_grid["expert_means"].detach().cpu().numpy(),
        expert_stds=pred_grid["expert_stds"].detach().cpu().numpy(),
        gate_weights=gate_w,
        assignments=assignments,
        gate_entropy=ent,
    )

    xg = x_grid.detach().cpu().numpy().ravel()
    true_step = step_function(xg, x_bounds=x_bounds, step_values=step_values)

    metrics: dict[str, Any] = {
        "model_class": "MOEGPR",
        "experiment": "step_sharp_gate",
        "gate_type": "mlp",
        "assignment_init": "x",
        "temperature": temperature,
        "num_experts": num_experts,
        "n_train": n_train,
        "n_test": n_test,
        "seed": seed,
        "num_em_iters": num_em_iters,
        "step_values": list(step_values),
        "x_bounds": list(x_bounds),
        **{f"test_{k}": v for k, v in point.items()},
        **{f"test_true_{k}": v for k, v in point_true.items()},
        "mean_gate_entropy": float(ent.mean()),
        "min_gate_entropy": float(ent.min()),
        "gate_weights_simplex_ok": gate_ok,
        **region_purity,
        "em_history": history,
    }

    title = f"step_mlp_K{num_experts}_T{temperature:g}_seed{seed}"
    json_path = os.path.join(save_path, f"gp_{title}.json")
    plot_paths: list[str] = []
    if plot:
        plot_dir = os.path.join(save_path, "plots", "prediction_runs")
        em = pred_grid["expert_means"].detach().cpu().numpy()
        es = pred_grid["expert_stds"].detach().cpu().numpy()
        mm = pred_grid["mixture_mean"].detach().cpu().numpy()
        xg2 = x_grid.detach().cpu().numpy()
        p1 = save_moe_posterior_plot(
            data["X_train"],
            data["y_train"],
            xg2,
            em,
            es,
            mm,
            plot_dir,
            file_tag=file_tag,
            title=f"MoE-GP sharp gate (T={temperature:g}) — step toy",
            true_curves=[(true_step, "True step")],
            show_bimodal_truth=False,
        )
        p2 = save_gate_weights_plot(
            xg2,
            gate_w,
            plot_dir,
            file_tag=file_tag,
            title=f"MLP gate weights (T={temperature:g})",
        )
        plot_paths = [str(p1), str(p2)]
        if assignments is not None:
            p3 = save_assignment_scatter(
                data["X_train"],
                data["y_train"],
                assignments,
                plot_dir,
                file_tag=file_tag,
                true_mode_mask=None,
                title="Hard expert assignments (step)",
            )
            plot_paths.append(str(p3))
        metrics["plot_paths"] = plot_paths

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=_json_default)

    metrics["npz_path"] = npz_path
    metrics["json_path"] = json_path
    print(f"Saved metrics -> {json_path}")
    print(
        f"RRMSE(true)={point_true['rrmse_mixture_mean']:.4f}  "
        f"NLL={point['mixture_nll']:.4f}  "
        f"region_purity={region_purity.get('region_purity', float('nan')):.4f}  "
        f"H(pi)={float(ent.mean()):.4f}"
    )
    return metrics
