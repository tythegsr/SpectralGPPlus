"""Partitioned exact GPR on PCA-reduced TOA inputs."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
_PCA_DIR = Path(__file__).resolve().parent
_GP_DIR = _ROOT / "experiments_GP"
_MTGPR_DIR = _ROOT / "experiments_RFFMTGPR"
_DEFAULT_SAVE_DIR = "experiments_PCA/results/toa_pca"

DEFAULT_DROP_COLUMNS = sorted({132, *range(195, 209)})

ENSEMBLE_MODE_NAMES = (
    "full",
    "mean_var_only",
    "single_partition",
    "nearest_partition",
    "top_m",
)


def _pin_experiment_paths() -> None:
    ordered = (str(_MTGPR_DIR), str(_GP_DIR), str(_PCA_DIR), str(_ROOT))
    sys.path[:] = list(ordered) + [p for p in sys.path if p not in ordered]


_pin_experiment_paths()

from gpplus.training import GPTrainer, evaluate_gp_model
from gpplus.training.optimizers import LBFGSScipy
from gpplus.utils import StandardScaler, UniformScaler, compute_metrics, set_seed
from gp_experiment_utils import (
    DEFAULT_ADAM_KWARGS,
    DEFAULT_LBFGS_KWARGS,
    build_gpr_model,
    extract_learned_likelihood_noise,
    json_safe_optimizer_kwargs,
    save_metrics_json,
)
from load_experimental_data import load_toa_data
from mtgpr_experiment_utils import (
    compute_relative_error_metrics,
    format_relative_error_summary,
    unpack_train_val_test,
)
from plot_toa_posterior import (
    plot_toa_posterior_figures,
    select_posterior_example_indices,
    wavelength_axis,
)
from toa_mtgpr_base import (
    TASK_NAMES,
    TOA_INPUT_DIM,
    compute_per_task_metrics,
    drop_input_columns,
    normalize_columns_to_drop,
)
from toa_pca_utils import fit_pca_on_train, make_train_partitions, transform_pca
from toa_y_transform import forward_y_single, inverse_y_single


def _extract_kernel_hyperparams(model) -> dict[str, Any]:
    out: dict[str, Any] = {"outputscale": float("nan"), "lengthscale": []}
    try:
        out["outputscale"] = float(model.covar_module.outputscale.detach().cpu().item())
    except Exception:
        pass
    try:
        ls = model.covar_module.base_kernel.lengthscale.detach().cpu().reshape(-1)
        out["lengthscale"] = ls.tolist()
    except Exception:
        pass
    return out


def _task_rrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    std = float(np.std(y_true))
    return rmse / std if std > 0 else float("inf")


def _aggregate_ensemble(
    mu_by_part: np.ndarray,
    std_by_part: np.ndarray,
    centroids: np.ndarray,
    z_test: np.ndarray,
    *,
    mode: str,
    single_partition_index: int,
    top_m: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """
    Aggregate partition GP predictions.

    Parameters
    ----------
    mu_by_part, std_by_part : (K, n_test)
    centroids : (K, p)
    z_test : (n_test, p)
    """
    k, n_test = mu_by_part.shape
    meta: dict[str, float] = {}

    if mode == "single_partition":
        k0 = int(np.clip(single_partition_index, 0, k - 1))
        meta["partition_index"] = float(k0)
        return mu_by_part[k0], std_by_part[k0], meta

    if mode == "nearest_partition":
        # (n_test, K) distances
        dists = np.linalg.norm(z_test[:, None, :] - centroids[None, :, :], axis=2)
        sel = np.argmin(dists, axis=1)
        meta["mean_selected_partition"] = float(np.mean(sel))
        mu = mu_by_part[sel, np.arange(n_test)]
        std = std_by_part[sel, np.arange(n_test)]
        return mu, std, meta

    if mode == "top_m":
        m = min(int(top_m), k)
        dists = np.linalg.norm(z_test[:, None, :] - centroids[None, :, :], axis=2)
        nearest = np.argsort(dists, axis=1)[:, :m]
        mu_out = np.empty(n_test, dtype=np.float64)
        std_out = np.empty(n_test, dtype=np.float64)
        for i in range(n_test):
            idx = nearest[i]
            mu_sub = mu_by_part[idx, i]
            std_sub = std_by_part[idx, i]
            mu_out[i] = float(np.mean(mu_sub))
            within = float(np.mean(std_sub**2))
            between = float(np.var(mu_sub))
            std_out[i] = float(np.sqrt(max(within + between, 0.0)))
        meta["m"] = float(m)
        return mu_out, std_out, meta

    # full or mean_var_only: equal weight over all K
    mu = np.mean(mu_by_part, axis=0)
    if mode == "mean_var_only":
        std = np.sqrt(np.mean(std_by_part**2, axis=0))
        return mu, std, meta

    within = np.mean(std_by_part**2, axis=0)
    between = np.var(mu_by_part, axis=0)
    std = np.sqrt(np.maximum(within + between, 0.0))
    return mu, std, meta


def _ensemble_diagnostics(
    mu_by_part: np.ndarray,
    std_by_part: np.ndarray,
) -> dict[str, float]:
    within = np.mean(std_by_part**2, axis=0)
    mu_mean = np.mean(mu_by_part, axis=0)
    between = np.mean((mu_by_part - mu_mean[None, :]) ** 2, axis=0)
    spread = np.max(mu_by_part, axis=0) - np.min(mu_by_part, axis=0)
    disagreement_std = np.std(mu_by_part, axis=0)
    return {
        "within_partition_var_mean": float(np.mean(within)),
        "between_partition_var_mean": float(np.mean(between)),
        "partition_disagreement_std_mean": float(np.mean(disagreement_std)),
        "max_partition_spread": float(np.max(spread)),
    }


def _print_ensemble_comparison(
    mode_metrics: dict[str, dict[str, dict[str, float]]],
    *,
    top_m: int,
) -> None:
    print("\nEnsemble mode comparison (y_cos RRMSE / y_grain RRMSE):")
    for mode in ENSEMBLE_MODE_NAMES:
        cos_rr = mode_metrics[mode]["y_cos"].get("RRMSE", float("nan"))
        grain_rr = mode_metrics[mode]["y_grain"].get("RRMSE", float("nan"))
        label = mode if mode != "top_m" else f"top_m (M={top_m})"
        print(f"  {label:<22} {cos_rr:.6f} / {grain_rr:.6f}")

    print("\nPer-task RMSE (secondary):")
    for mode in ENSEMBLE_MODE_NAMES:
        cos_rm = mode_metrics[mode]["y_cos"].get("RMSE", float("nan"))
        grain_rm = mode_metrics[mode]["y_grain"].get("RMSE", float("nan"))
        label = mode if mode != "top_m" else f"top_m (M={top_m})"
        print(f"  {label:<22} {cos_rm:.6f} / {grain_rm:.6f}")


def _train_partition_gpr(
    z_part: torch.Tensor,
    y_part_fit: torch.Tensor,
    *,
    seed: int,
    num_inits: int,
    num_epochs: int,
    device: str,
    dtype: torch.dtype,
    ard: bool,
    n_jobs: int | None,
    optimizer_kwargs: dict | None,
) -> tuple[Any, dict, float]:
    if num_epochs <= 1:
        optimizer_class = LBFGSScipy
        default_optimizer_kwargs = DEFAULT_LBFGS_KWARGS
    else:
        optimizer_class = torch.optim.Adam
        default_optimizer_kwargs = DEFAULT_ADAM_KWARGS
    if optimizer_kwargs is None:
        optimizer_kwargs = dict(default_optimizer_kwargs)

    model = build_gpr_model(z_part, y_part_fit, ard=ard)
    trainer = GPTrainer(
        model,
        num_epochs=num_epochs,
        num_inits=num_inits,
        seed=seed,
        device=device,
        dtype=dtype,
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
        n_jobs=n_jobs,
        inner_max_num_threads=1,
        cholesky_jitter=1e-6,
        callbacks=[],
    )
    t0 = time.time()
    runs = trainer.train()
    train_time = time.time() - t0

    successful = [r for r in runs if r.get("loss") is not None and r.get("state_dict") is not None]
    if not successful:
        errors = [r.get("error", "unknown") for r in runs if r.get("error")]
        raise RuntimeError(f"All partition GP inits failed: {errors[:3]}")

    best_run = min(successful, key=lambda r: r["loss"])
    model.load_state_dict(best_run["state_dict"])
    noise_meta = extract_learned_likelihood_noise(model)
    run_meta = {
        "best_train_loss": float(best_run["loss"]),
        "train_time_s": train_time,
        **_extract_kernel_hyperparams(model),
        **noise_meta,
    }
    return model, run_meta, train_time


def run_toa_pca_partition_gpr(
    n_train: int = 49000,
    n_test: int = 5000,
    n_components: int = 30,
    partition_size: int = 1500,
    seed: int = 42,
    num_inits: int = 4,
    num_epochs: int = 1,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    save_path: str | None = None,
    standardize_x: bool = True,
    x_standardize_method: int = 2,
    standardize_y: bool = True,
    ard: bool = True,
    predict_chunk_size: int = 512,
    n_jobs: int | None = 1,
    optimizer_kwargs: dict | None = None,
    plot_posterior: bool = True,
    rel_tolerance: float = 0.01,
    posterior_n_examples: int = 8,
    posterior_example_indices: list[int] | None = None,
    data_path: str | None = None,
    log_grain: bool = True,
    drop_columns: list[int] | None = None,
    partition_shuffle: bool = True,
    top_m_partitions: int = 5,
    single_partition_index: int = 0,
    compare_input_dim: bool = False,
    pca_svd_solver: str = "randomized",
) -> dict:
    """PCA + partitioned exact GPR on TOA (independent models per task)."""
    if save_path is None:
        save_path = _DEFAULT_SAVE_DIR

    if drop_columns is None:
        drop_columns = list(DEFAULT_DROP_COLUMNS)

    input_configs: list[tuple[str, list[int] | None]] = [("270", drop_columns)]
    if compare_input_dim:
        input_configs.append(("285", None))

    all_results: dict[str, Any] = {}
    primary_metrics: dict | None = None

    for dim_label, cols in input_configs:
        metrics = _run_single_input_dim(
            n_train=n_train,
            n_test=n_test,
            n_components=n_components,
            partition_size=partition_size,
            seed=seed,
            num_inits=num_inits,
            num_epochs=num_epochs,
            device=device,
            dtype=dtype,
            save_path=save_path if dim_label == "270" or not compare_input_dim else None,
            standardize_x=standardize_x,
            x_standardize_method=x_standardize_method,
            standardize_y=standardize_y,
            ard=ard,
            predict_chunk_size=predict_chunk_size,
            n_jobs=n_jobs,
            optimizer_kwargs=optimizer_kwargs,
            plot_posterior=plot_posterior and dim_label == "270",
            rel_tolerance=rel_tolerance,
            posterior_n_examples=posterior_n_examples,
            posterior_example_indices=posterior_example_indices,
            data_path=data_path,
            log_grain=log_grain,
            drop_columns=cols,
            partition_shuffle=partition_shuffle,
            top_m_partitions=top_m_partitions,
            single_partition_index=single_partition_index,
            pca_svd_solver=pca_svd_solver,
            dim_label=dim_label,
        )
        all_results[dim_label] = metrics
        if dim_label == "270":
            primary_metrics = metrics

    if compare_input_dim and primary_metrics is not None:
        ablation = {
            "270": {
                "input_dim_after_drop": all_results["270"].get("input_dim"),
                "aggregate_RRMSE": all_results["270"].get("aggregate_RRMSE"),
                "y_cos_RRMSE": all_results["270"].get("y_cos_RRMSE"),
                "y_grain_RRMSE": all_results["270"].get("y_grain_RRMSE"),
            },
            "285": {
                "input_dim_after_drop": all_results["285"].get("input_dim"),
                "aggregate_RRMSE": all_results["285"].get("aggregate_RRMSE"),
                "y_cos_RRMSE": all_results["285"].get("y_cos_RRMSE"),
                "y_grain_RRMSE": all_results["285"].get("y_grain_RRMSE"),
            },
        }
        primary_metrics["input_dim_ablation"] = ablation
        if save_path:
            title = primary_metrics["title"]
            out_json = save_metrics_json(primary_metrics, save_path, title)
            print(f"Updated metrics with input_dim_ablation: {out_json}")

    assert primary_metrics is not None
    return primary_metrics


def _run_single_input_dim(
    *,
    n_train: int,
    n_test: int,
    n_components: int,
    partition_size: int,
    seed: int,
    num_inits: int,
    num_epochs: int,
    device: str,
    dtype: torch.dtype,
    save_path: str | None,
    standardize_x: bool,
    x_standardize_method: int,
    standardize_y: bool,
    ard: bool,
    predict_chunk_size: int,
    n_jobs: int | None,
    optimizer_kwargs: dict | None,
    plot_posterior: bool,
    rel_tolerance: float,
    posterior_n_examples: int,
    posterior_example_indices: list[int] | None,
    data_path: str | None,
    log_grain: bool,
    drop_columns: list[int] | None,
    partition_shuffle: bool,
    top_m_partitions: int,
    single_partition_index: int,
    pca_svd_solver: str,
    dim_label: str,
) -> dict:
    set_seed(seed)

    n_partitions_est = int(np.ceil(n_train / partition_size))
    title = (
        f"TOA_nTrain{n_train}_nTest{n_test}_pcaP{n_components}_"
        f"part{partition_size}_K{n_partitions_est}"
    )
    if dim_label == "285":
        title += "_full285"

    print("=" * 60)
    print(title)
    print(
        f"PCA + partitioned exact GPR, p={n_components}, partition_size={partition_size}, "
        f"ARD={ard}, dtype={dtype}, inits={num_inits}, tasks={TASK_NAMES}"
    )
    print("=" * 60)

    data = load_toa_data(
        n_train=n_train,
        n_test=n_test,
        n_val=0,
        seed=seed,
        data_path=data_path,
    )
    x_train, y_train, _x_val, _y_val, x_test, y_test, train_idx, _val_idx, test_idx = (
        unpack_train_val_test(data)
    )

    x_test_orig = x_test.clone()
    input_dim_original = TOA_INPUT_DIM
    dropped_columns = normalize_columns_to_drop(drop_columns, input_dim=input_dim_original)
    x_train, kept_column_indices_t = drop_input_columns(
        x_train, dropped_columns, input_dim=input_dim_original
    )
    x_test, _ = drop_input_columns(x_test, dropped_columns, input_dim=input_dim_original)
    input_dim = int(x_train.shape[-1])
    kept_column_indices = kept_column_indices_t.tolist()
    if dropped_columns:
        print(f"Input columns: dropped {len(dropped_columns)} / {input_dim_original} -> {input_dim}")

    x_train_np = x_train.detach().cpu().numpy()
    pca_fit = fit_pca_on_train(
        x_train_np,
        n_components=n_components,
        svd_solver=pca_svd_solver,
        random_state=seed,
    )
    print(
        f"PCA: {input_dim} -> {pca_fit.n_components} components, "
        f"variance explained={pca_fit.total_variance_explained:.4f}"
    )

    z_train = transform_pca(pca_fit, x_train, dtype=dtype)
    z_test = transform_pca(pca_fit, x_test, dtype=dtype)
    y_train = y_train.to(dtype=dtype)
    y_test = y_test.to(dtype=dtype)

    x_scaler = None
    x_scaling_type = "None"
    if standardize_x:
        if x_standardize_method == 2:
            x_scaler = UniformScaler(scale_to_neg_one=True)
            x_scaling_type = "UniformScaler [-1, 1]"
        elif x_standardize_method == 1:
            x_scaler = UniformScaler(scale_to_neg_one=False)
            x_scaling_type = "UniformScaler [0, 1]"
        elif x_standardize_method == 0:
            x_scaler = StandardScaler()
            x_scaling_type = "StandardScaler (Gaussian)"
        else:
            raise ValueError(f"x_standardize_method must be 0, 1, or 2, got {x_standardize_method}")
        x_scaler.fit(z_train)
        z_train = x_scaler.transform(z_train)
        z_test = x_scaler.transform(z_test)
        print(f"Z scaling: {x_scaling_type}")

    partitions = make_train_partitions(
        z_train,
        y_train,
        train_idx,
        partition_size=partition_size,
        seed=seed + 17,
        shuffle=partition_shuffle,
    )
    n_partitions = len(partitions)
    print(f"Partitions: K={n_partitions}, shuffle={partition_shuffle}")

    total_train_time = 0.0
    total_prediction_time = 0.0
    partition_runs: list[dict] = []

    # Per-task storage
    task_y_scalers: dict[str, StandardScaler | None] = {}
    task_mu_by_part: dict[str, list[np.ndarray]] = {n: [] for n in TASK_NAMES}
    task_std_by_part: dict[str, list[np.ndarray]] = {n: [] for n in TASK_NAMES}
    task_lower_by_part: dict[str, list[np.ndarray]] = {n: [] for n in TASK_NAMES}
    task_upper_by_part: dict[str, list[np.ndarray]] = {n: [] for n in TASK_NAMES}
    centroids = np.stack(
        [p["z"].detach().cpu().numpy().mean(axis=0) for p in partitions],
        axis=0,
    )
    z_test_np = z_test.detach().cpu().numpy()

    for task_idx, task_name in enumerate(TASK_NAMES):
        print(f"\n--- Task: {task_name} ---")
        y_scaler = None
        y_tr_raw = y_train[:, task_idx]
        y_te = y_test[:, task_idx]

        y_tr_model = forward_y_single(y_tr_raw, task_name, log_grain=log_grain)
        if standardize_y:
            y_scaler = StandardScaler()
            y_scaler.fit(y_tr_model.unsqueeze(-1))
            y_tr_fit_global = y_scaler.transform(y_tr_model.unsqueeze(-1)).squeeze(-1)
        else:
            y_tr_fit_global = y_tr_model
        task_y_scalers[task_name] = y_scaler

        for part in partitions:
            k = int(part["partition_index"])
            z_part = part["z"]
            y_part_raw = part["y"][:, task_idx]
            y_part_model = forward_y_single(y_part_raw, task_name, log_grain=log_grain)
            if standardize_y and y_scaler is not None:
                y_part_fit = y_scaler.transform(y_part_model.unsqueeze(-1)).squeeze(-1)
            else:
                y_part_fit = y_part_model

            part_seed = seed + 1000 * (task_idx + 1) + k
            model, run_meta, train_time = _train_partition_gpr(
                z_part,
                y_part_fit,
                seed=part_seed,
                num_inits=num_inits,
                num_epochs=num_epochs,
                device=device,
                dtype=dtype,
                ard=ard,
                n_jobs=n_jobs,
                optimizer_kwargs=optimizer_kwargs,
            )
            total_train_time += train_time

            model.eval()
            t_pred = time.time()
            pred_mean, lower, upper, pred_std = evaluate_gp_model(model, z_test)
            pred_time = time.time() - t_pred
            total_prediction_time += pred_time

            pred_mean, pred_std, lower, upper = inverse_y_single(
                pred_mean.detach().cpu(),
                pred_std.detach().cpu(),
                lower.detach().cpu(),
                upper.detach().cpu(),
                task_name=task_name,
                y_scaler=y_scaler,
                standardize_y=standardize_y,
                log_grain=log_grain,
            )
            y_te_np = y_te.detach().cpu().numpy()
            pred_np = pred_mean.numpy()
            part_rrmse = _task_rrmse(y_te_np, pred_np)
            part_rmse = float(np.sqrt(np.mean((pred_np - y_te_np) ** 2)))

            task_mu_by_part[task_name].append(pred_np)
            task_std_by_part[task_name].append(pred_std.numpy())
            task_lower_by_part[task_name].append(lower.numpy())
            task_upper_by_part[task_name].append(upper.numpy())

            partition_runs.append(
                {
                    "partition_index": k,
                    "task_name": task_name,
                    "n_partition": int(part["n_partition"]),
                    "train_indices": part["train_indices"].detach().cpu().tolist(),
                    "centroid_z": centroids[k].tolist(),
                    "train_time_s": run_meta["train_time_s"],
                    "best_train_loss": run_meta["best_train_loss"],
                    "raw_noise": run_meta.get("raw_noise"),
                    "noise": run_meta.get("noise"),
                    "noise_std": run_meta.get("noise_std"),
                    "lengthscale": run_meta.get("lengthscale"),
                    "outputscale": run_meta.get("outputscale"),
                    "test_RRMSE": part_rrmse,
                    "test_RMSE": part_rmse,
                }
            )
            print(
                f"  partition {k:2d}: RRMSE={part_rrmse:.6f}  RMSE={part_rmse:.6f}  "
                f"train={train_time:.1f}s"
            )

    # Ensemble all modes per task
    mode_metrics: dict[str, dict[str, dict[str, float]]] = {
        m: {} for m in ENSEMBLE_MODE_NAMES
    }
    ensemble_diagnostics: dict[str, dict[str, float]] = {}
    y_pred_by_mode: dict[str, np.ndarray] = {}
    y_std_by_mode: dict[str, np.ndarray] = {}
    y_pred_by_partition: dict[str, np.ndarray] = {}
    y_std_by_partition: dict[str, np.ndarray] = {}

    y_pred_primary_list: list[np.ndarray] = []
    y_std_primary_list: list[np.ndarray] = []
    lower_primary_list: list[np.ndarray] = []
    upper_primary_list: list[np.ndarray] = []

    y_test_np = y_test.detach().cpu().numpy()

    for task_idx, task_name in enumerate(TASK_NAMES):
        mu_stack = np.stack(task_mu_by_part[task_name], axis=0)
        std_stack = np.stack(task_std_by_part[task_name], axis=0)
        y_pred_by_partition[task_name] = mu_stack
        y_std_by_partition[task_name] = std_stack
        ensemble_diagnostics[task_name] = _ensemble_diagnostics(mu_stack, std_stack)

        y_true_task = y_test_np[:, task_idx]

        for mode in ENSEMBLE_MODE_NAMES:
            mu, std, mode_meta = _aggregate_ensemble(
                mu_stack,
                std_stack,
                centroids,
                z_test_np,
                mode=mode,
                single_partition_index=single_partition_index,
                top_m=top_m_partitions,
            )
            lower = mu - 1.96 * std
            upper = mu + 1.96 * std
            computed = compute_metrics(
                torch.as_tensor(y_true_task),
                torch.as_tensor(mu),
                output_std=torch.as_tensor(std),
                lower_95=torch.as_tensor(lower),
                upper_95=torch.as_tensor(upper),
            )
            task_rrmse = _task_rrmse(y_true_task, mu)
            mode_metrics[mode][task_name] = {
                **{k: float(v) for k, v in computed.items() if isinstance(v, (int, float, np.floating))},
                "RRMSE": task_rrmse,
            }
            mode_metrics[mode][task_name].update(
                {k: float(v) for k, v in mode_meta.items()}
            )
            y_pred_by_mode[f"{mode}_{task_name}"] = mu
            y_std_by_mode[f"{mode}_{task_name}"] = std

        # Primary full mode for stacked outputs
        mu_full, std_full, _ = _aggregate_ensemble(
            mu_stack,
            std_stack,
            centroids,
            z_test_np,
            mode="full",
            single_partition_index=single_partition_index,
            top_m=top_m_partitions,
        )
        lower_full = mu_full - 1.96 * std_full
        upper_full = mu_full + 1.96 * std_full
        y_pred_primary_list.append(mu_full)
        y_std_primary_list.append(std_full)
        lower_primary_list.append(lower_full)
        upper_primary_list.append(upper_full)

    y_pred_stacked = np.stack(y_pred_primary_list, axis=1)
    y_std_stacked = np.stack(y_std_primary_list, axis=1)
    lower_stacked = np.stack(lower_primary_list, axis=1)
    upper_stacked = np.stack(upper_primary_list, axis=1)

    per_task = compute_per_task_metrics(y_test_np, y_pred_stacked)
    aggregate_rrmse = float(np.mean([per_task[f"{n}_RRMSE"] for n in TASK_NAMES]))
    aggregate_rmse = float(np.sqrt(np.mean((y_pred_stacked - y_test_np) ** 2)))

    rel_metrics_by_task: dict[str, dict] = {}
    for t, name in enumerate(TASK_NAMES):
        rel_m = compute_relative_error_metrics(
            y_test_np[:, t],
            y_pred_stacked[:, t],
            rel_tolerance=rel_tolerance,
        )
        rel_metrics_by_task[name] = rel_m
        per_task[f"{name}_max_rel_error"] = float(rel_m["max_rel_error"])
        per_task[f"{name}_mean_rel_error"] = float(rel_m["mean_rel_error"])
        per_task[f"{name}_pct_within_1pct"] = float(rel_m["pct_within_1pct"])

    opt_name = "LBFGSScipy" if num_epochs <= 1 else "Adam"
    metrics: dict[str, Any] = {
        "title": title,
        "model_class": "GPR_partitioned",
        "input_dim": input_dim,
        "input_dim_original": input_dim_original,
        "dropped_columns": dropped_columns,
        "kept_column_indices": kept_column_indices,
        "n_train": n_train,
        "n_test": n_test,
        "num_tasks": len(TASK_NAMES),
        "task_names": list(TASK_NAMES),
        "n_components": pca_fit.n_components,
        "partition_size": partition_size,
        "n_partitions": n_partitions,
        "partition_shuffle": partition_shuffle,
        "top_m_partitions": top_m_partitions,
        "single_partition_index": single_partition_index,
        "primary_ensemble_mode": "full",
        "ard": ard,
        "num_epochs": num_epochs,
        "num_inits": num_inits,
        "optimizer": opt_name,
        "optimizer_kwargs": json_safe_optimizer_kwargs(optimizer_kwargs or DEFAULT_LBFGS_KWARGS),
        "standardize_x": standardize_x,
        "x_standardize_method": x_standardize_method,
        "x_scaling_type": x_scaling_type,
        "standardize_y": standardize_y,
        "log_grain": log_grain,
        "rel_tolerance": rel_tolerance,
        "Training_Time": total_train_time,
        "Prediction_Time": total_prediction_time,
        "Total_Time": total_train_time + total_prediction_time,
        "aggregate_RRMSE": aggregate_rrmse,
        "RMSE": aggregate_rmse,
        "pca": pca_fit.to_dict(),
        "ensemble_modes": mode_metrics,
        "ensemble_diagnostics": ensemble_diagnostics,
        "partition_runs": partition_runs,
        **per_task,
    }

    _print_ensemble_comparison(mode_metrics, top_m=top_m_partitions)

    print(f"\nTest aggregate RRMSE: {aggregate_rrmse:.6f}  RMSE: {aggregate_rmse:.6f}")
    for name in TASK_NAMES:
        print(
            f"{name} RRMSE: {per_task[f'{name}_RRMSE']:.6f}  "
            f"RMSE: {per_task[f'{name}_RMSE']:.6f}"
        )
        print(format_relative_error_summary(name, rel_metrics_by_task[name], rel_tolerance=rel_tolerance))
    print(f"Total training time: {total_train_time:.1f}s")

    if save_path:
        example_indices = select_posterior_example_indices(
            y_test_np.shape[0],
            posterior_n_examples,
            seed=seed,
            explicit_indices=posterior_example_indices,
        )
        import os

        os.makedirs(save_path, exist_ok=True)
        out_npz = os.path.join(str(save_path), f"predictions_{title}.npz")
        wl = wavelength_axis(x_test_orig.shape[-1])
        np.savez_compressed(
            out_npz,
            y_true=y_test_np,
            y_pred=y_pred_stacked,
            y_std=y_std_stacked,
            lower=lower_stacked,
            upper=upper_stacked,
            x_test_orig=x_test_orig.detach().cpu().numpy(),
            test_idx=test_idx.detach().cpu().numpy(),
            task_names=np.array(TASK_NAMES),
            title=title,
            seed=seed,
            rel_tolerance=rel_tolerance,
            wavelength_nm=wl,
            example_indices=np.array(example_indices, dtype=np.int64),
            log_grain=np.array(log_grain),
            y_pred_by_mode=np.stack(
                [y_pred_by_mode[f"{m}_{n}"] for m in ENSEMBLE_MODE_NAMES for n in TASK_NAMES]
            ).reshape(len(ENSEMBLE_MODE_NAMES), len(TASK_NAMES), -1),
            y_std_by_mode=np.stack(
                [y_std_by_mode[f"{m}_{n}"] for m in ENSEMBLE_MODE_NAMES for n in TASK_NAMES]
            ).reshape(len(ENSEMBLE_MODE_NAMES), len(TASK_NAMES), -1),
            ensemble_mode_names=np.array(ENSEMBLE_MODE_NAMES),
            y_pred_by_partition=np.stack(
                [y_pred_by_partition[n] for n in TASK_NAMES], axis=1
            ),
            y_std_by_partition=np.stack(
                [y_std_by_partition[n] for n in TASK_NAMES], axis=1
            ),
        )
        metrics["predictions_npz"] = out_npz

        if plot_posterior and example_indices:
            import logging

            from plot_multid_slice_predictions import sanitize_plot_subdir

            post_dir = Path(save_path) / "plots" / "posterior" / sanitize_plot_subdir(title)
            try:
                post_paths = plot_toa_posterior_figures(
                    x_test_orig.detach().cpu().numpy(),
                    y_test_np,
                    y_pred_stacked,
                    y_std_stacked,
                    lower_stacked,
                    upper_stacked,
                    post_dir,
                    title=title,
                    example_indices=example_indices,
                    rel_metrics_by_task=rel_metrics_by_task,
                    rel_tolerance=rel_tolerance,
                    wavelength_nm=wl,
                    log_grain=log_grain,
                )
                for plot_path in post_paths:
                    print(f"Saved posterior plot to {plot_path}")
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "Posterior plot generation failed: %s", exc
                )

        out_json = save_metrics_json(metrics, save_path, title)
        print(f"Saved metrics to {out_json}")

    return metrics
