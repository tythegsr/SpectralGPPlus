"""S2 PCA + partitioned exact GPR with per-QoI band subsets."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
_PCA_DIR = Path(__file__).resolve().parent
_GP_DIR = _ROOT / "experiments_GP"
_MTGPR_DIR = _ROOT / "experiments_RFFMTGPR"
_RFF_DIR = _ROOT / "experiments_RFF"
_DEFAULT_SAVE_DIR = "experiments_PCA/results/s2_toa_pca_gpr"
_EXACT_GP_PARTITION_WARN = 5000

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments_toa.paths import pin_toa_import_paths

pin_toa_import_paths(_MTGPR_DIR, _RFF_DIR, _GP_DIR, _PCA_DIR)

from experiments_toa.data import TOA_TEST_POOL_SIZE, TOA_TRAIN_POOL_SIZE, TOA_VAL_POOL_SIZE
from experiments_toa.s2_bands import (
    NComponentsSpec,
    band_config_metadata,
    load_task_band_config,
    resolve_task_pca_components,
)
from experiments_toa.s2_constants import S2_DEFAULT_BAND_CONFIG_PATH, S2_INPUT_DIM, S2_TASK_NAMES
from experiments_toa.s2_data import load_s2_toa_data
from experiments_toa.s2_reporting import (
    apply_log_scale_extra_metrics,
    attach_log_scale_aggregate_fields,
    attach_relative_error_fields,
    collect_log_scale_prediction_arrays,
    compute_aggregate_rmse,
    compute_aggregate_rrmse,
    print_end_of_run_summary,
    print_task_test_metrics,
    save_s2_summary_artifacts,
    stack_or_none,
)
from experiments_toa.s2_utils import (
    apply_x_transform,
    compute_per_task_metrics,
    extract_ard_lengthscales,
    macro_metric,
    map_ard_to_bands,
    select_bands,
)
from experiments_toa.s2_y_transform import (
    InverseYOutput,
    forward_y_s2,
    inverse_y_s2,
    logit_bounds_for_task,
    resolve_y_warps,
    task_uses_log_scale,
    task_uses_logit_scale,
)
from gpplus.training import evaluate_gp_model
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
from mtgpr_experiment_utils import compute_relative_error_metrics, format_relative_error_summary
from toa_pca_partition_gpr_base import (
    ENSEMBLE_MODE_NAMES,
    _aggregate_ensemble,
    _ensemble_diagnostics,
    _task_rrmse,
)
from toa_pca_utils import fit_pca_on_train, make_train_partitions, transform_pca


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
    response_noise_prior: bool = False,
    noise_var_fraction: float = 0.25,
    noise_prior_log_scale: float = 0.5,
) -> tuple[Any, dict, float]:
    from gpplus.training import GPTrainer

    if num_epochs <= 1:
        optimizer_class = LBFGSScipy
        default_optimizer_kwargs = DEFAULT_LBFGS_KWARGS
    else:
        optimizer_class = torch.optim.Adam
        default_optimizer_kwargs = DEFAULT_ADAM_KWARGS
    if optimizer_kwargs is None:
        optimizer_kwargs = dict(default_optimizer_kwargs)

    likelihood = None
    initializer_kwargs = None
    noise_prior_meta: dict[str, float] | None = None
    if response_noise_prior:
        from gpplus.priors.response_noise import (
            build_scalar_noise_likelihood,
            empirical_scalar_noise_variance,
            log_normal_scalar_noise_prior_from_responses,
            scalar_noise_raw_init_from_variance,
        )

        target_var = empirical_scalar_noise_variance(y_part_fit, fraction=noise_var_fraction)
        noise_prior = log_normal_scalar_noise_prior_from_responses(
            y_part_fit,
            fraction=noise_var_fraction,
            log_scale=noise_prior_log_scale,
            dtype=dtype,
            device=y_part_fit.device,
        )
        likelihood = build_scalar_noise_likelihood(noise_prior=noise_prior)
        raw_init = scalar_noise_raw_init_from_variance(likelihood, target_var.to(dtype=dtype))
        initializer_kwargs = {
            "parameter_configs": {"raw_noise": {"method": "constant", "value": raw_init}}
        }
        noise_prior_meta = {
            "noise_prior_target_var": float(target_var.detach().cpu()),
            "noise_prior_loc": float(noise_prior.loc.detach().cpu()),
        }

    model = build_gpr_model(z_part, y_part_fit, ard=ard)
    if likelihood is not None:
        model.likelihood = likelihood

    trainer = GPTrainer(
        model,
        num_epochs=num_epochs,
        num_inits=num_inits,
        seed=seed,
        device=device,
        dtype=dtype,
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
        initializer_kwargs=initializer_kwargs,
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
    if noise_prior_meta:
        run_meta.update(noise_prior_meta)
    return model, run_meta, train_time


def _print_ensemble_comparison(
    mode_metrics: dict[str, dict[str, dict[str, float]]],
    task_names: Sequence[str],
    *,
    top_m: int,
) -> None:
    print("\nEnsemble mode comparison (macro RRMSE):")
    for mode in ENSEMBLE_MODE_NAMES:
        vals = [mode_metrics[mode][n].get("RRMSE", float("nan")) for n in task_names]
        macro = float(np.mean(vals)) if vals else float("nan")
        label = mode if mode != "top_m" else f"top_m (M={top_m})"
        detail = " / ".join(f"{n}={mode_metrics[mode][n].get('RRMSE', float('nan')):.4f}" for n in task_names)
        print(f"  {label:<22} macro={macro:.6f}  ({detail})")


def run_s2_toa_pca_gpr(
    n_train: int = 16000,
    n_test: int = 5000,
    n_components: NComponentsSpec = 30,
    partition_size: int = 2000,
    seed: int = 42,
    num_inits: int = 4,
    num_epochs: int = 1,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    save_path: str | None = None,
    standardize_x: bool = True,
    x_standardize_method: int = 2,
    standardize_y: bool = True,
    log_scale: bool | None = None,
    log_scale_qoi: Sequence[str] | None = None,
    logit_scale_qoi: Sequence[str] | None = None,
    logit_bounds: dict[str, tuple[float, float]] | None = None,
    ard: bool = True,
    predict_chunk_size: int = 512,
    n_jobs: int | None = 1,
    optimizer_kwargs: dict | None = None,
    monitor_validation: bool = True,
    plot_validation: bool = True,
    plot_posterior: bool = True,
    rel_tolerance: float = 0.01,
    posterior_n_examples: int = 20,
    posterior_example_indices: list[int] | None = None,
    data_path: str | None = None,
    pca_svd_solver: str = "randomized",
    response_noise_prior: bool = False,
    noise_var_fraction: float = 0.25,
    noise_prior_log_scale: float = 0.5,
    input_variable: str = "toa_reflectance",
    task_names: Sequence[str] | None = None,
    task_band_config: str | None = None,
    partition_shuffle: bool = True,
    top_m_partitions: int = 5,
    single_partition_index: int = 0,
    x_transform: str | None = None,
    **_unused_kwargs,
) -> dict:
    """
    S2 PCA + partitioned exact GPR.

    ``n_train`` is drawn from the fixed 49,000 train pool. For each QoI: apply that
    task's original-band subset, fit PCA on all ``n_train``, scale scores, then split
    into partitions of size at most ``partition_size`` (S1 ``make_train_partitions``).
    Train one independent exact GPR per (task, partition) with ARD over PCA components,
    predict all test points from every partition, and aggregate with the same ensemble
    modes / primary ``full`` default as S1 PCA partition GPR.

    Identity targets with optional y-standardization (no log-grain transform).
    """
    del predict_chunk_size  # API parity; exact GP uses full test batch
    if save_path is None:
        save_path = _DEFAULT_SAVE_DIR

    names = list(task_names) if task_names is not None else list(S2_TASK_NAMES)
    band_cfg_path = task_band_config or str(S2_DEFAULT_BAND_CONFIG_PATH)
    bands_by_task = load_task_band_config(band_cfg_path, task_names=names, input_dim=S2_INPUT_DIM)
    n_components_by_task, pca_components_meta = resolve_task_pca_components(
        n_components, task_names=names
    )
    unique_ps = sorted(set(n_components_by_task.values()))
    pca_title_token = (
        str(unique_ps[0]) if len(unique_ps) == 1 else "perQoI"
    )

    set_seed(seed)
    if partition_size < 1:
        raise ValueError(f"partition_size must be >= 1, got {partition_size}")

    n_partitions_est = int(np.ceil(n_train / partition_size))
    title = (
        f"S2_TOA_nTrain{n_train}_nTest{n_test}_pcaP{pca_title_token}_"
        f"part{partition_size}_K{n_partitions_est}"
    )
    print("=" * 60)
    print(title)
    print(
        f"S2 PCA + partitioned exact GPR, p={n_components_by_task}, "
        f"partition_size={partition_size}, "
        f"ARD={ard}, dtype={dtype}, inits={num_inits}, tasks={names}, input={input_variable}"
    )
    print(f"Band config: {band_cfg_path}")
    if pca_components_meta.get("config_path"):
        print(f"PCA components config: {pca_components_meta['config_path']}")
    if partition_size > _EXACT_GP_PARTITION_WARN:
        print(
            f"WARNING: exact GP is O(n^3) per partition; "
            f"partition_size={partition_size} may be slow/OOM."
        )
    if monitor_validation:
        print(
            "Note: partitioned PCA-GPR does not run per-partition validation callbacks "
            "(monitor_validation accepted for CLI parity)."
        )
    print("=" * 60)

    (
        x_train_full,
        y_train,
        _x_val_full,
        _y_val,
        x_test_full,
        y_test,
        train_idx,
        val_idx,
        test_idx,
        wavelengths,
        data_meta,
    ) = load_s2_toa_data(
        n_train=n_train,
        n_test=n_test,
        n_val=0,
        seed=seed,
        data_path=data_path,
        input_variable=input_variable,  # type: ignore[arg-type]
        task_names=names,
    )
    warps = resolve_y_warps(
        log_scale_qoi,
        logit_scale_qoi,
        log_scale=log_scale,
        meta=data_meta,
        logit_bounds=logit_bounds,
    )
    log_scale = warps.log_scale
    log_scale_task_set = warps.log_tasks
    log_scale_source = warps.log_source
    logit_scale_task_set = warps.logit_tasks
    print(
        f"Output warps: log={sorted(log_scale_task_set) or '[]'} "
        f"(source={log_scale_source}); "
        f"logit={sorted(logit_scale_task_set) or '[]'} "
        f"(source={warps.logit_source})"
    )
    wavelengths_np = wavelengths.detach().cpu().numpy()
    x_test_orig = x_test_full.clone()
    y_train = y_train.to(dtype=dtype)
    y_test = y_test.to(dtype=dtype)

    total_train_time = 0.0
    total_prediction_time = 0.0
    partition_runs: list[dict] = []
    input_bands_by_task: dict[str, dict] = {}
    pca_by_task: dict[str, dict] = {}
    n_partitions_by_task: dict[str, int] = {}
    task_metrics: dict[str, dict] = {}

    mode_metrics: dict[str, dict[str, dict[str, float]]] = {m: {} for m in ENSEMBLE_MODE_NAMES}
    ensemble_diagnostics: dict[str, dict[str, float]] = {}
    y_pred_by_mode: dict[str, np.ndarray] = {}
    y_std_by_mode: dict[str, np.ndarray] = {}
    y_pred_by_partition: dict[str, np.ndarray] = {}
    y_std_by_partition: dict[str, np.ndarray] = {}

    y_pred_primary_list: list[np.ndarray] = []
    y_std_primary_list: list[np.ndarray] = []
    lower_primary_list: list[np.ndarray] = []
    upper_primary_list: list[np.ndarray] = []
    y_pred_mean_all: list[np.ndarray] = []
    y_pred_mode_all: list[np.ndarray] = []
    log_mu_all: list[np.ndarray] = []
    log_sigma_all: list[np.ndarray] = []
    logit_mu_all: list[np.ndarray] = []
    logit_sigma_all: list[np.ndarray] = []

    y_test_np = y_test.detach().cpu().numpy()

    for task_idx, task_name in enumerate(names):
        print(f"\n--- Task: {task_name} ---")
        band_indices = bands_by_task[task_name]
        x_tr = select_bands(x_train_full, band_indices).to(dtype=dtype)
        x_te = select_bands(x_test_full, band_indices).to(dtype=dtype)
        x_tr = apply_x_transform(x_tr, x_transform)
        x_te = apply_x_transform(x_te, x_transform)
        if x_transform and x_transform != "none" and task_idx == 0:
            print(f"X transform: {x_transform} (before PCA / scaling)")
        input_dim_before_pca = int(x_tr.shape[-1])
        task_n_components = int(n_components_by_task[task_name])

        pca_fit = fit_pca_on_train(
            x_tr,
            n_components=task_n_components,
            svd_solver=pca_svd_solver,
            random_state=seed,
        )
        z_train = transform_pca(pca_fit, x_tr, dtype=dtype)
        z_test = transform_pca(pca_fit, x_te, dtype=dtype)
        pca_meta = pca_fit.to_dict()
        pca_by_task[task_name] = pca_meta
        print(
            f"PCA ({task_name}): {input_dim_before_pca} -> {pca_fit.n_components}, "
            f"var={pca_fit.total_variance_explained:.4f}"
        )

        x_scaling_type = "None"
        x_scaler = None
        if standardize_x:
            if x_standardize_method == 0:
                x_scaler = StandardScaler()
                x_scaling_type = "StandardScaler (Gaussian)"
            elif x_standardize_method == 1:
                x_scaler = UniformScaler(scale_to_neg_one=False)
                x_scaling_type = "UniformScaler [0, 1]"
            elif x_standardize_method == 2:
                x_scaler = UniformScaler(scale_to_neg_one=True)
                x_scaling_type = "UniformScaler [-1, 1]"
            else:
                raise ValueError(
                    f"x_standardize_method must be 0, 1, or 2, got {x_standardize_method}"
                )
            x_scaler.fit(z_train)
            z_train = x_scaler.transform(z_train)
            z_test = x_scaler.transform(z_test)
            print(f"Z scaling: {x_scaling_type}")

        y_tr = y_train[:, task_idx]
        y_te = y_test[:, task_idx]
        y_tr_model = forward_y_s2(y_tr, task_name, warps=warps)
        y_scaler = None
        if standardize_y:
            y_scaler = StandardScaler()
            y_scaler.fit(y_tr_model.unsqueeze(-1))

        partitions = make_train_partitions(
            z_train,
            y_tr_model.unsqueeze(-1),
            train_idx,
            partition_size=partition_size,
            seed=seed + 17,
            shuffle=partition_shuffle,
        )
        n_partitions = len(partitions)
        n_partitions_by_task[task_name] = n_partitions
        print(f"Partitions: K={n_partitions}, shuffle={partition_shuffle}")

        centroids = np.stack(
            [p["z"].detach().cpu().numpy().mean(axis=0) for p in partitions],
            axis=0,
        )
        z_test_np = z_test.detach().cpu().numpy()

        mu_by_part: list[np.ndarray] = []
        std_by_part: list[np.ndarray] = []
        last_ard_mapped: dict[str, Any] = {}

        for part in partitions:
            k = int(part["partition_index"])
            z_part = part["z"]
            y_part_raw = part["y"].squeeze(-1)
            if standardize_y and y_scaler is not None:
                y_part_fit = y_scaler.transform(y_part_raw.unsqueeze(-1)).squeeze(-1)
            else:
                y_part_fit = y_part_raw

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
                response_noise_prior=response_noise_prior,
                noise_var_fraction=noise_var_fraction,
                noise_prior_log_scale=noise_prior_log_scale,
            )
            total_train_time += train_time

            ard_info = extract_ard_lengthscales(model)
            last_ard_mapped = map_ard_to_bands(
                ard_info["lengthscale"],
                band_indices,
                wavelengths_nm=wavelengths_np,
                ard_space="pca_components",
            )
            if ard_info["outputscale"] is not None:
                last_ard_mapped["outputscale"] = ard_info["outputscale"]
            last_ard_mapped["raw_lengthscale"] = ard_info["raw_lengthscale"]
            last_ard_mapped["model_input_dim"] = int(z_part.shape[-1])
            last_ard_mapped["pca"] = pca_meta

            model.eval()
            t_pred = time.time()
            pred_mean, lower, upper, pred_std = evaluate_gp_model(model, z_test)
            pred_time = time.time() - t_pred
            total_prediction_time += pred_time

            pred_mean = pred_mean.detach().cpu()
            pred_std = pred_std.detach().cpu()
            lower = lower.detach().cpu()
            upper = upper.detach().cpu()
            inv = inverse_y_s2(
                pred_mean,
                pred_std,
                lower,
                upper,
                task_name=task_name,
                y_scaler=y_scaler,
                standardize_y=standardize_y,
                warps=warps,
                extended=True,
            )
            pred_mean, pred_std, lower, upper = inv.as_tuple()

            pred_np = pred_mean.numpy()
            std_np = pred_std.numpy()
            y_te_np = y_te.detach().cpu().numpy()
            part_rrmse = _task_rrmse(y_te_np, pred_np)
            part_rmse = float(np.sqrt(np.mean((pred_np - y_te_np) ** 2)))

            mu_by_part.append(pred_np)
            std_by_part.append(std_np)

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

        input_bands_by_task[task_name] = {
            "indices": list(band_indices),
            "wavelength_nm": [float(wavelengths_np[i]) for i in band_indices],
            "model_input_dim": int(z_train.shape[-1]),
            "x_scaling_type": x_scaling_type,
            "n_partitions": n_partitions,
            **last_ard_mapped,
        }

        mu_stack = np.stack(mu_by_part, axis=0)
        std_stack = np.stack(std_by_part, axis=0)
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
                **{
                    k: float(v)
                    for k, v in computed.items()
                    if isinstance(v, (int, float, np.floating))
                },
                "RRMSE": task_rrmse,
            }
            mode_metrics[mode][task_name].update({k: float(v) for k, v in mode_meta.items()})
            y_pred_by_mode[f"{mode}_{task_name}"] = mu
            y_std_by_mode[f"{mode}_{task_name}"] = std

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

        # Primary-mode extras: for warped QoIs, recover warped-space arrays from physical medians.
        primary_tm = dict(mode_metrics["full"][task_name])
        if task_uses_log_scale(task_name, warps=warps):
            log_mu_t = torch.as_tensor(np.log(np.clip(mu_full, 1e-300, None)))
            # Invert physical std ≈ median * sqrt(expm1(σ²)) for a rough σ.
            ratio = np.clip(std_full / np.clip(mu_full, 1e-300, None), 0.0, None)
            log_sigma_np = np.sqrt(np.log1p(ratio**2))
            log_sigma_t = torch.as_tensor(log_sigma_np)
            sigma_for_mean = np.clip(log_sigma_np, 0.0, 3.0)
            point_mean_np = mu_full * np.exp(0.5 * sigma_for_mean**2)
            point_mode_np = mu_full * np.exp(-(log_sigma_np**2))
            inv_primary = InverseYOutput(
                point=torch.as_tensor(mu_full),
                std=torch.as_tensor(std_full),
                lower=torch.as_tensor(lower_full),
                upper=torch.as_tensor(upper_full),
                point_mean=torch.as_tensor(point_mean_np),
                point_mode=torch.as_tensor(point_mode_np),
                log_mu=log_mu_t,
                log_sigma=log_sigma_t,
            )
            apply_log_scale_extra_metrics(
                primary_tm,
                y_true=torch.as_tensor(y_true_task),
                inv=inv_primary,
                task_name=task_name,
                warps=warps,
            )
            mode_metrics["full"][task_name] = primary_tm
            collect_log_scale_prediction_arrays(
                inv_primary,
                torch.as_tensor(mu_full),
                y_pred_mean_all=y_pred_mean_all,
                y_pred_mode_all=y_pred_mode_all,
                log_mu_all=log_mu_all,
                log_sigma_all=log_sigma_all,
                logit_mu_all=logit_mu_all,
                logit_sigma_all=logit_sigma_all,
            )
        elif task_uses_logit_scale(task_name, warps=warps):
            a, b = logit_bounds_for_task(task_name, warps=warps)
            u = np.clip((mu_full - a) / (b - a), 1e-4, 1.0 - 1e-4)
            logit_mu_np = np.log(u / (1.0 - u))
            u_factor = u * (1.0 - u)
            logit_sigma_np = np.clip(
                std_full / np.clip((b - a) * u_factor, 1e-12, None), 0.0, None
            )
            inv_primary = InverseYOutput(
                point=torch.as_tensor(mu_full),
                std=torch.as_tensor(std_full),
                lower=torch.as_tensor(lower_full),
                upper=torch.as_tensor(upper_full),
                point_mean=torch.as_tensor(mu_full),
                logit_mu=torch.as_tensor(logit_mu_np),
                logit_sigma=torch.as_tensor(logit_sigma_np),
            )
            apply_log_scale_extra_metrics(
                primary_tm,
                y_true=torch.as_tensor(y_true_task),
                inv=inv_primary,
                task_name=task_name,
                warps=warps,
            )
            mode_metrics["full"][task_name] = primary_tm
            collect_log_scale_prediction_arrays(
                inv_primary,
                torch.as_tensor(mu_full),
                y_pred_mean_all=y_pred_mean_all,
                y_pred_mode_all=y_pred_mode_all,
                log_mu_all=log_mu_all,
                log_sigma_all=log_sigma_all,
                logit_mu_all=logit_mu_all,
                logit_sigma_all=logit_sigma_all,
            )
        else:
            collect_log_scale_prediction_arrays(
                InverseYOutput(
                    point=torch.as_tensor(mu_full),
                    std=torch.as_tensor(std_full),
                    lower=torch.as_tensor(lower_full),
                    upper=torch.as_tensor(upper_full),
                ),
                torch.as_tensor(mu_full),
                y_pred_mean_all=y_pred_mean_all,
                y_pred_mode_all=y_pred_mode_all,
                log_mu_all=log_mu_all,
                log_sigma_all=log_sigma_all,
                logit_mu_all=logit_mu_all,
                logit_sigma_all=logit_sigma_all,
            )
        task_metrics[task_name] = primary_tm
        print_task_test_metrics(
            task_name,
            primary_tm,
            warps=warps,
        )

    y_pred_stacked = np.stack(y_pred_primary_list, axis=1)
    y_std_stacked = np.stack(y_std_primary_list, axis=1)
    lower_stacked = np.stack(lower_primary_list, axis=1)
    upper_stacked = np.stack(upper_primary_list, axis=1)
    y_pred_mean_stacked = stack_or_none(y_pred_mean_all)
    y_pred_mode_stacked = stack_or_none(y_pred_mode_all)
    log_mu_stacked = stack_or_none(log_mu_all)
    log_sigma_stacked = stack_or_none(log_sigma_all)
    logit_mu_stacked = stack_or_none(logit_mu_all)
    logit_sigma_stacked = stack_or_none(logit_sigma_all)

    per_task = compute_per_task_metrics(y_test_np, y_pred_stacked, names)
    rel_metrics_by_task = attach_relative_error_fields(
        per_task,
        names=names,
        y_true=y_test_np,
        y_pred=y_pred_stacked,
        rel_tolerance=rel_tolerance,
        compute_relative_error_metrics=compute_relative_error_metrics,
    )
    aggregate_rmse = compute_aggregate_rmse(y_pred_stacked, y_test_np)
    aggregate_rrmse = compute_aggregate_rrmse(per_task, names)
    log_task_names, aggregate_rrmse_log, aggregate_rrmse_mean = attach_log_scale_aggregate_fields(
        per_task,
        task_metrics,
        names=names,
        warps=warps,
    )
    logit_task_names = [n for n in names if task_uses_logit_scale(n, warps=warps)]
    aggregate_medae = macro_metric(per_task, names, "MedAE")
    n_partitions = int(next(iter(n_partitions_by_task.values()))) if n_partitions_by_task else 0
    opt_name = "LBFGSScipy" if num_epochs <= 1 else "Adam"

    metrics: dict[str, Any] = {
        "title": title,
        "dataset": "s2",
        "model_class": "GPR_partitioned",
        "input_dim_original": S2_INPUT_DIM,
        "n_train": n_train,
        "n_test": n_test,
        "num_tasks": len(names),
        "task_names": list(names),
        "n_components": n_components_by_task if len(unique_ps) > 1 else unique_ps[0],
        "n_pca_components": n_components_by_task if len(unique_ps) > 1 else unique_ps[0],
        "n_components_by_task": dict(n_components_by_task),
        "pca_components_meta": pca_components_meta,
        "pca_svd_solver": pca_svd_solver,
        "partition_size": partition_size,
        "n_partitions": n_partitions,
        "n_partitions_by_task": n_partitions_by_task,
        "partition_shuffle": partition_shuffle,
        "top_m_partitions": top_m_partitions,
        "single_partition_index": single_partition_index,
        "primary_ensemble_mode": "full",
        "pca_partition_mode": "independent_gpr_per_task_partition",
        "ard": ard,
        "num_epochs": num_epochs,
        "num_inits": num_inits,
        "optimizer": opt_name,
        "optimizer_kwargs": json_safe_optimizer_kwargs(
            optimizer_kwargs or (DEFAULT_LBFGS_KWARGS if num_epochs <= 1 else DEFAULT_ADAM_KWARGS)
        ),
        "standardize_x": standardize_x,
        "x_standardize_method": x_standardize_method,
        "x_transform": x_transform or "none",
        "standardize_y": standardize_y,
        "log_scale": bool(log_scale),
        "log_scale_tasks": [n for n in names if task_uses_log_scale(n, warps=warps)],
        "log_scale_source": log_scale_source,
        "logit_scale_tasks": list(logit_task_names),
        "logit_scale_source": warps.logit_source,
        "logit_bounds": {
            k: list(v) for k, v in warps.logit_bounds.items() if k in logit_scale_task_set
        },
        "response_noise_prior": bool(response_noise_prior),
        "noise_var_fraction": float(noise_var_fraction),
        "noise_prior_log_scale": float(noise_prior_log_scale),
        "rel_tolerance": rel_tolerance,
        "task_band_config": str(band_cfg_path),
        "bands_by_task": band_config_metadata(bands_by_task, wavelengths_nm=wavelengths_np),
        "input_bands_by_task": input_bands_by_task,
        "pca_by_task": pca_by_task,
        "data_meta": data_meta,
        "train_pool_size": TOA_TRAIN_POOL_SIZE,
        "val_pool_size": TOA_VAL_POOL_SIZE,
        "test_pool_size": TOA_TEST_POOL_SIZE,
        "monitor_validation": False,
        "Training_Time": total_train_time,
        "Prediction_Time": total_prediction_time,
        "Total_Time": total_train_time + total_prediction_time,
        "aggregate_RRMSE": aggregate_rrmse,
        "aggregate_RRMSE_mean": aggregate_rrmse,
        "aggregate_MedAE": aggregate_medae,
        "aggregate_RRMSE_log_tasks": aggregate_rrmse_log,
        "aggregate_RRMSE_lognormal_mean_tasks": aggregate_rrmse_mean,
        "RMSE": aggregate_rmse,
        "ensemble_modes": mode_metrics,
        "ensemble_diagnostics": ensemble_diagnostics,
        "partition_runs": partition_runs,
        **per_task,
    }

    # Flatten primary-mode task metrics for S2 JSON parity with independent GP.
    for task_name in names:
        tm = mode_metrics["full"][task_name]
        for key, value in tm.items():
            metrics[f"{task_name}_{key}"] = value

    _print_ensemble_comparison(mode_metrics, names, top_m=top_m_partitions)
    print_end_of_run_summary(
        names=names,
        per_task=per_task,
        rel_metrics_by_task=rel_metrics_by_task,
        aggregate_rrmse=aggregate_rrmse,
        aggregate_rmse=aggregate_rmse,
        log_task_names=log_task_names,
        aggregate_rrmse_log=aggregate_rrmse_log,
        aggregate_rrmse_mean=aggregate_rrmse_mean,
        rel_tolerance=rel_tolerance,
        total_train_time=total_train_time,
        format_relative_error_summary=format_relative_error_summary,
        aggregate_medae=aggregate_medae,
    )

    if save_path:
        save_s2_summary_artifacts(
            save_path=save_path,
            title=title,
            metrics=metrics,
            names=names,
            y_test_np=y_test_np,
            y_pred_stacked=y_pred_stacked,
            y_std_stacked=y_std_stacked,
            lower_stacked=lower_stacked,
            upper_stacked=upper_stacked,
            x_test_orig=x_test_orig.numpy(),
            wavelengths_np=wavelengths_np,
            train_idx=train_idx.cpu().numpy(),
            val_idx=val_idx.cpu().numpy(),
            test_idx=test_idx.cpu().numpy(),
            bands_by_task=bands_by_task,
            rel_metrics_by_task=rel_metrics_by_task,
            rel_tolerance=rel_tolerance,
            log_scale=log_scale,
            log_scale_tasks=log_scale_task_set,
            data_meta=data_meta,
            seed=seed,
            posterior_n_examples=posterior_n_examples,
            posterior_example_indices=posterior_example_indices,
            plot_validation=plot_validation,
            plot_posterior=plot_posterior,
            monitor_validation=False,
            n_val=0,
            save_metrics_json=save_metrics_json,
            y_pred_mean_stacked=y_pred_mean_stacked,
            y_pred_mode_stacked=y_pred_mode_stacked,
            log_mu_stacked=log_mu_stacked,
            log_sigma_stacked=log_sigma_stacked,
            warps=warps,
            logit_mu_stacked=logit_mu_stacked,
            logit_sigma_stacked=logit_sigma_stacked,
        )
        # Augment with ensemble / partition arrays (S1 parity).
        out_npz = metrics.get("predictions_npz")
        if out_npz:
            with np.load(out_npz) as existing:
                payload = {k: existing[k] for k in existing.files}
            payload["y_pred_by_mode"] = np.stack(
                [y_pred_by_mode[f"{m}_{n}"] for m in ENSEMBLE_MODE_NAMES for n in names]
            ).reshape(len(ENSEMBLE_MODE_NAMES), len(names), -1)
            payload["y_std_by_mode"] = np.stack(
                [y_std_by_mode[f"{m}_{n}"] for m in ENSEMBLE_MODE_NAMES for n in names]
            ).reshape(len(ENSEMBLE_MODE_NAMES), len(names), -1)
            payload["ensemble_mode_names"] = np.array(ENSEMBLE_MODE_NAMES)
            payload["y_pred_by_partition"] = np.stack(
                [y_pred_by_partition[n] for n in names], axis=1
            )
            payload["y_std_by_partition"] = np.stack(
                [y_std_by_partition[n] for n in names], axis=1
            )
            np.savez_compressed(out_npz, **payload)

    return metrics
