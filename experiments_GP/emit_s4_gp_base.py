"""S4 PCA + partitioned exact GPR (ensemble) on EMIT / snow-TOA NetCDF.

Mirrors ``experiments_PCA/toa_s2_pca_gpr_base.py`` with S4 data loading (radiance +
geometry aux). Per QoI: band select → PCA → scale → train-partition ensemble of
exact GPRs; primary aggregation mode is ``full``. ASD field validation runs
in-process (no ``.pt`` checkpoints) and writes the usual ASD plots + neighbors.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
_GP_DIR = Path(__file__).resolve().parent
_PCA_DIR = _ROOT / "experiments_PCA"
_MTGPR_DIR = _ROOT / "experiments_RFFMTGPR"
_RFF_DIR = _ROOT / "experiments_RFF"
_DEFAULT_SAVE_DIR = "experiments_GP/results/s4_emit_gp"
_EXACT_GP_PARTITION_WARN = 5000
_DEFAULT_ASD_PATH = _ROOT / "experiments_toa" / "data 11 QoI" / "asd_validation_set.nc"

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments_toa.paths import pin_toa_import_paths

pin_toa_import_paths(_MTGPR_DIR, _RFF_DIR, _GP_DIR, _PCA_DIR)

from experiments_toa.s2_bands import (
    NComponentsSpec,
    band_config_metadata,
    load_task_band_config,
    resolve_task_pca_components,
)
from experiments_toa.s2_constants import (
    S4_BAND_CONFIG_ALIASES,
    S4_LOGIT_BOUNDS,
    S4_SPECTRAL_DIM,
)
from experiments_toa.s2_plotting import (
    plot_prediction_coverage,
    plot_s2_posterior_examples,
    plot_s2_task_scatter,
    save_s2_predictions_npz,
    select_posterior_example_indices,
)
from experiments_toa.s2_utils import (
    _scalar_error_metrics,
    compute_per_task_metrics,
    extract_ard_lengthscales,
    macro_metric,
    map_ard_to_bands,
    select_bands,
)
from experiments_toa.s2_y_transform import (
    forward_y_s2,
    inverse_y_s2,
    resolve_y_warps,
    task_uses_log_scale,
    task_uses_logit_scale,
)
from experiments_toa.s4_asd_posttrain import (
    ASD_TASK_NAMES,
    write_asd_eval_artifacts,
)
from emit_s4_mtgpr_base import (
    S4_INPUT_DIM,
    S4_TASK_NAMES,
    _DEFAULT_EMIT_PATH,
    _DEFAULT_WL_SRC,
    _load_wavelengths_nm,
    _subsample_scatter,
    force_train_local_from_asd_appended,
    load_emit_s4_xy,
    parse_s4_task_names,
    split_emit_s4,
)
from gpplus.training import evaluate_gp_model
from gpplus.utils import StandardScaler, UniformScaler, compute_metrics, set_seed
from gp_experiment_utils import (
    DEFAULT_ADAM_KWARGS,
    DEFAULT_LBFGS_KWARGS,
    json_safe_optimizer_kwargs,
    save_metrics_json,
)
from mtgpr_experiment_utils import (
    compute_prediction_coverage_metrics,
    compute_relative_error_metrics,
    format_coverage_summary,
)
from experiments_SORF._eval_s4_asd_validation import TASK_LABEL_MAP, _load_asd_xy
from toa_pca_partition_gpr_base import (
    ENSEMBLE_MODE_NAMES,
    _aggregate_ensemble,
    _ensemble_diagnostics,
    _task_rrmse,
)
from toa_pca_utils import fit_pca_on_train, make_train_partitions, transform_pca
from toa_s2_pca_gpr_base import _train_partition_gpr


def _task_input_columns(
    band_indices: Sequence[int],
    *,
    aux_indices: Sequence[int],
    n_spectral: int = S4_SPECTRAL_DIM,
) -> list[int]:
    cols = [int(i) for i in band_indices]
    for idx in aux_indices:
        if int(idx) < n_spectral:
            raise ValueError(f"aux index {idx} must be >= n_spectral={n_spectral}")
        cols.append(int(idx))
    return cols


def _prepare_model_inputs(
    x: torch.Tensor,
    *,
    task_cols: Sequence[int],
    pca_fit,
    x_scaler,
    dtype: torch.dtype,
) -> torch.Tensor:
    z = select_bands(x, task_cols).to(dtype=dtype)
    z = transform_pca(pca_fit, z, dtype=dtype)
    if x_scaler is not None:
        z = x_scaler.transform(z)
    return z


def run_s4_emit_gp(
    n_train: int = 10000,
    n_val: int = 0,
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
    log_offsets: Mapping[str, float] | None = None,
    ard: bool = True,
    n_jobs: int | None = 1,
    optimizer_kwargs: dict | None = None,
    plot_posterior: bool = True,
    rel_tolerance: float = 0.01,
    posterior_n_examples: int = 20,
    posterior_example_indices: list[int] | None = None,
    data_path: str | None = None,
    wl_src: str | None = None,
    n_pca_components: NComponentsSpec | None = None,
    pca_svd_solver: str = "randomized",
    partition_size: int = 2000,
    partition_shuffle: bool = True,
    top_m_partitions: int = 5,
    single_partition_index: int = 0,
    response_noise_prior: bool = False,
    noise_var_fraction: float = 0.25,
    noise_prior_log_scale: float = 0.5,
    task_names: Sequence[str] | None = None,
    task_band_config: str | None = None,
    filter_valid_labels: bool = False,
    nigp: bool = False,
    freeze_epoch_nigp: int = 0,
    eval_asd: bool = True,
    asd_path: str | Path | None = None,
    plot_asd: bool = True,
    neighbor_thr: float = 0.04,
    **_unused_kwargs,
) -> dict:
    """S4 PCA + partitioned exact GPR with optional in-process ASD eval/plots."""
    if nigp and n_pca_components is not None:
        # Allowed on PCA scores (S2 PCA-GPR does this); keep flag for parity.
        pass
    if n_pca_components is None:
        raise ValueError(
            "S4_emit_GP requires n_pca_components (int or per-QoI JSON path)."
        )
    if partition_size < 1:
        raise ValueError(f"partition_size must be >= 1, got {partition_size}")

    names = parse_s4_task_names(task_names)
    num_tasks = len(names)
    emit_path = Path(data_path) if data_path is not None else _DEFAULT_EMIT_PATH
    wl_path = Path(wl_src) if wl_src is not None else _DEFAULT_WL_SRC
    if save_path is None:
        save_path = _DEFAULT_SAVE_DIR
    asd_file = Path(asd_path) if asd_path is not None else _DEFAULT_ASD_PATH

    set_seed(seed)
    if num_epochs <= 1:
        default_optimizer_kwargs = dict(DEFAULT_LBFGS_KWARGS)
    else:
        default_optimizer_kwargs = dict(DEFAULT_ADAM_KWARGS)
    if optimizer_kwargs is None:
        optimizer_kwargs = dict(default_optimizer_kwargs)

    X_np, Y_np, source_idx, data_meta = load_emit_s4_xy(
        emit_path,
        task_names=names,
        filter_valid_labels=filter_valid_labels,
    )
    n_filtered = int(X_np.shape[0])
    input_dim = int(X_np.shape[1])
    aux_indices = list(data_meta.get("aux_indices", []))
    force_train_local = force_train_local_from_asd_appended(emit_path, source_idx)
    if force_train_local.size:
        print(
            f"Forcing {force_train_local.size} ASD-appended row(s) into train "
            f"(source indices {source_idx[force_train_local].tolist()})"
        )
    train_local, val_local, test_local = split_emit_s4(
        n_filtered,
        n_train=n_train,
        n_val=int(n_val),
        seed=seed,
        force_train_local=force_train_local,
    )
    n_test = int(test_local.size)
    train_idx = torch.as_tensor(source_idx[train_local], dtype=torch.int64)
    val_idx = torch.as_tensor(source_idx[val_local], dtype=torch.int64)
    test_idx = torch.as_tensor(source_idx[test_local], dtype=torch.int64)

    n_components_by_task, pca_components_meta = resolve_task_pca_components(
        n_pca_components, task_names=names
    )
    unique_ps = sorted(set(n_components_by_task.values()))
    pca_title_token = str(unique_ps[0]) if len(unique_ps) == 1 else "perQoI"

    if task_band_config is not None:
        from experiments_toa.s2_constants import S2_TASK_NAMES

        band_lookup_names = [S4_BAND_CONFIG_ALIASES.get(n, n) for n in names]
        skipped = [n for n in band_lookup_names if n not in S2_TASK_NAMES]
        if skipped:
            raise ValueError(
                f"task_band_config cannot be used with non-S2 QoI names {skipped}."
            )
        bands_by_lookup = load_task_band_config(
            task_band_config,
            task_names=band_lookup_names,
            input_dim=S4_SPECTRAL_DIM,
        )
        bands_by_task = {
            n: bands_by_lookup[S4_BAND_CONFIG_ALIASES.get(n, n)] for n in names
        }
    else:
        bands_by_task = {name: list(range(S4_SPECTRAL_DIM)) for name in names}

    n_partitions_est = int(np.ceil(n_train / partition_size))
    title = (
        f"S4_EMIT_ST_nTrain{n_train}_nTest{n_test}_pcaP{pca_title_token}_"
        f"part{partition_size}_K{n_partitions_est}_T{num_tasks}"
    )
    if nigp:
        title = f"{title}_nigp"

    print("=" * 60)
    print(title)
    print(
        f"S4 PCA + partitioned exact GPR  p={n_components_by_task}  "
        f"partition_size={partition_size}  ARD={ard}  dtype={dtype}  "
        f"inits={num_inits}  epochs={num_epochs}  tasks={names}  "
        f"input_dim={input_dim}  aux={aux_indices}  nigp={bool(nigp)}"
    )
    if pca_components_meta.get("config_path"):
        print(f"PCA components config: {pca_components_meta['config_path']}")
    if partition_size > _EXACT_GP_PARTITION_WARN:
        print(
            f"WARNING: exact GP is O(n^3) per partition; "
            f"partition_size={partition_size} may be slow/OOM."
        )
    print(f"Device: {device}")
    print(
        f"Split: filtered={n_filtered}  train={n_train}  val={int(val_local.size)}  "
        f"test={n_test}  (data {emit_path})"
    )
    print("=" * 60)

    x_train_full = torch.as_tensor(X_np[train_local], dtype=dtype)
    y_train = torch.as_tensor(Y_np[train_local], dtype=dtype)
    x_test_full = torch.as_tensor(X_np[test_local], dtype=dtype)
    y_test = torch.as_tensor(Y_np[test_local], dtype=dtype)
    del X_np, Y_np

    warps = resolve_y_warps(
        log_scale_qoi,
        logit_scale_qoi,
        log_scale=log_scale,
        meta=data_meta,
        logit_bounds=logit_bounds if logit_bounds is not None else dict(S4_LOGIT_BOUNDS),
        log_offsets=log_offsets,
    )
    print(
        f"Output warps: log={sorted(warps.log_tasks) or '[]'} "
        f"(source={warps.log_source}); "
        f"logit={sorted(warps.logit_tasks) or '[]'} "
        f"(source={warps.logit_source})"
    )
    wavelengths_np = _load_wavelengths_nm(emit_path, wl_path)
    y_test_np = y_test.detach().cpu().numpy()

    asd_tasks = [n for n in names if n in ASD_TASK_NAMES and n in TASK_LABEL_MAP]
    asd_x_np: np.ndarray | None = None
    asd_y_by_task: dict[str, np.ndarray] = {}
    asd_aux_meta: dict | None = None
    coszen_asd_original = None
    if eval_asd:
        if not asd_file.is_file():
            print(f"ASD eval skipped: ASD NetCDF missing: {asd_file}")
            eval_asd = False
        elif not asd_tasks:
            print(
                "ASD eval skipped: none of the trained QoIs have ASD labels "
                f"(trained={names}, asd={sorted(ASD_TASK_NAMES)})"
            )
            eval_asd = False
        else:
            asd_x_np, asd_y_by_task, asd_aux_meta, coszen_asd_original = _load_asd_xy(
                asd_file,
                asd_tasks,
            )
            print(f"Loaded ASD validation set: n={asd_x_np.shape[0]} from {asd_file}")
            print(f"  ASD tasks: {asd_tasks}")

    total_train_time = 0.0
    total_prediction_time = 0.0
    partition_runs: list[dict] = []
    n_partitions_by_task: dict[str, int] = {}
    pca_by_task: dict[str, dict] = {}
    input_bands_by_task: dict[str, dict] = {}
    mode_metrics: dict[str, dict[str, dict[str, float]]] = {
        m: {} for m in ENSEMBLE_MODE_NAMES
    }
    ensemble_diagnostics: dict[str, dict[str, float]] = {}

    y_pred_primary_list: list[np.ndarray] = []
    y_std_primary_list: list[np.ndarray] = []
    lower_primary_list: list[np.ndarray] = []
    upper_primary_list: list[np.ndarray] = []
    y_pred_mean_all: list[np.ndarray] = []
    y_pred_mode_all: list[np.ndarray] = []
    log_mu_all: list[np.ndarray] = []
    log_sigma_all: list[np.ndarray] = []

    asd_per_task: dict[str, dict] = {}
    asd_y_true_all: list[np.ndarray] = []
    asd_y_pred_all: list[np.ndarray] = []
    asd_y_pred_mean_all: list[np.ndarray] = []
    asd_y_pred_mode_all: list[np.ndarray] = []
    asd_log_mu_all: list[np.ndarray] = []
    asd_log_sigma_all: list[np.ndarray] = []
    asd_y_std_all: list[np.ndarray] = []
    asd_lower_all: list[np.ndarray] = []
    asd_upper_all: list[np.ndarray] = []
    asd_coverage_by_task: dict[str, dict] = {}
    asd_task_order: list[str] = []

    for task_idx, task_name in enumerate(names):
        print(f"\n--- Task: {task_name} ---")
        band_indices = bands_by_task[task_name]
        task_cols = _task_input_columns(band_indices, aux_indices=aux_indices)
        x_tr = select_bands(x_train_full, task_cols).to(dtype=dtype)
        x_te = select_bands(x_test_full, task_cols).to(dtype=dtype)
        input_dim_before_pca = int(x_tr.shape[-1])
        task_n_components = int(n_components_by_task[task_name])
        if task_n_components > input_dim_before_pca:
            raise ValueError(
                f"{task_name}: n_pca_components={task_n_components} exceeds "
                f"input_dim={input_dim_before_pca}"
            )

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

        x_scaler = None
        x_scaling_type = "None"
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

        z_asd = None
        z_asd_np = None
        if eval_asd and task_name in asd_tasks and asd_x_np is not None:
            z_asd = _prepare_model_inputs(
                torch.as_tensor(asd_x_np, dtype=dtype),
                task_cols=task_cols,
                pca_fit=pca_fit,
                x_scaler=x_scaler,
                dtype=dtype,
            )
            z_asd_np = z_asd.detach().cpu().numpy()

        mu_by_part: list[np.ndarray] = []
        std_by_part: list[np.ndarray] = []
        asd_mu_by_part: list[np.ndarray] = []
        asd_std_by_part: list[np.ndarray] = []
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
                nigp=nigp,
                freeze_epoch_nigp=freeze_epoch_nigp,
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

            inv = inverse_y_s2(
                pred_mean.detach().cpu(),
                pred_std.detach().cpu(),
                lower.detach().cpu(),
                upper.detach().cpu(),
                task_name=task_name,
                y_scaler=y_scaler,
                standardize_y=standardize_y,
                warps=warps,
                extended=True,
            )
            pred_mean_c, pred_std_c, _lo_c, _up_c = inv.as_tuple()
            pred_np = pred_mean_c.numpy()
            std_np = pred_std_c.numpy()
            y_te_np = y_te.detach().cpu().numpy()
            part_rrmse = _task_rrmse(y_te_np, pred_np)
            part_rmse = float(np.sqrt(np.mean((pred_np - y_te_np) ** 2)))
            mu_by_part.append(pred_np)
            std_by_part.append(std_np)

            if z_asd is not None:
                t_asd = time.time()
                asd_mean, asd_lo, asd_hi, asd_std = evaluate_gp_model(model, z_asd)
                total_prediction_time += time.time() - t_asd
                inv_asd = inverse_y_s2(
                    asd_mean.detach().cpu(),
                    asd_std.detach().cpu(),
                    asd_lo.detach().cpu(),
                    asd_hi.detach().cpu(),
                    task_name=task_name,
                    y_scaler=y_scaler,
                    standardize_y=standardize_y,
                    warps=warps,
                    extended=True,
                )
                asd_mu, asd_s, _, _ = inv_asd.as_tuple()
                asd_mu_by_part.append(asd_mu.numpy())
                asd_std_by_part.append(asd_s.numpy())

            partition_runs.append(
                {
                    "partition_index": k,
                    "task_name": task_name,
                    "n_partition": int(part["n_partition"]),
                    "train_time_s": run_meta["train_time_s"],
                    "best_train_loss": run_meta["best_train_loss"],
                    "test_RRMSE": part_rrmse,
                    "test_RMSE": part_rmse,
                }
            )
            print(
                f"  partition {k:2d}: RRMSE={part_rrmse:.6f}  RMSE={part_rmse:.6f}  "
                f"train={train_time:.1f}s"
            )
            del model
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

        input_bands_by_task[task_name] = {
            "indices": list(band_indices),
            "model_input_dim": int(z_train.shape[-1]),
            "x_scaling_type": x_scaling_type,
            "n_partitions": n_partitions,
            **last_ard_mapped,
        }

        mu_stack = np.stack(mu_by_part, axis=0)
        std_stack = np.stack(std_by_part, axis=0)
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
            mode_metrics[mode][task_name] = {
                **{
                    k: float(v)
                    for k, v in computed.items()
                    if isinstance(v, (int, float, np.floating))
                },
                "RRMSE": _task_rrmse(y_true_task, mu),
                **{k: float(v) for k, v in mode_meta.items()},
            }

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
        nan_col = np.full(mu_full.shape[0], np.nan, dtype=np.float64)
        y_pred_primary_list.append(mu_full)
        y_std_primary_list.append(std_full)
        lower_primary_list.append(lower_full)
        upper_primary_list.append(upper_full)
        y_pred_mean_all.append(nan_col)
        y_pred_mode_all.append(nan_col)
        log_mu_all.append(nan_col)
        log_sigma_all.append(nan_col)

        if asd_mu_by_part and z_asd_np is not None and task_name in asd_tasks:
            asd_mu_stack = np.stack(asd_mu_by_part, axis=0)
            asd_std_stack = np.stack(asd_std_by_part, axis=0)
            asd_mu, asd_std, _ = _aggregate_ensemble(
                asd_mu_stack,
                asd_std_stack,
                centroids,
                z_asd_np,
                mode="full",
                single_partition_index=single_partition_index,
                top_m=top_m_partitions,
            )
            asd_lo = asd_mu - 1.96 * asd_std
            asd_hi = asd_mu + 1.96 * asd_std
            y_true = torch.as_tensor(asd_y_by_task[task_name], dtype=dtype)
            n_eval = int(y_true.numel())
            nan_asd = np.full(n_eval, np.nan, dtype=np.float64)
            computed_asd = compute_metrics(
                y_true,
                torch.as_tensor(asd_mu),
                output_std=torch.as_tensor(asd_std),
                lower_95=torch.as_tensor(asd_lo),
                upper_95=torch.as_tensor(asd_hi),
            )
            scalar = _scalar_error_metrics(
                y_true.numpy(), asd_mu, prefix=f"{task_name}_"
            )
            cov_m = compute_prediction_coverage_metrics(y_true.numpy(), asd_mu, asd_std)
            asd_coverage_by_task[task_name] = cov_m
            row = {
                "asd_label": TASK_LABEL_MAP[task_name],
                "n_eval": n_eval,
                "y_true": y_true.numpy().tolist(),
                "y_pred": asd_mu.tolist(),
                "y_pred_mean": nan_asd.tolist(),
                "y_pred_mode": nan_asd.tolist(),
                "y_std": asd_std.tolist(),
                "lower_95": asd_lo.tolist(),
                "upper_95": asd_hi.tolist(),
                "ensemble_mode": "full",
                **cov_m,
                **computed_asd,
                **scalar,
            }
            asd_per_task[task_name] = row
            print(
                f"  ASD RMSE={computed_asd['RMSE']:.6g}  "
                f"RRMSE={computed_asd['RRMSE']:.6g}  "
                f"R2={scalar[f'{task_name}_R2']:.4f}  "
                f"cov95={cov_m['coverage_95']:.2f}"
            )
            asd_task_order.append(task_name)
            asd_y_true_all.append(y_true.numpy())
            asd_y_pred_all.append(asd_mu)
            asd_y_pred_mean_all.append(nan_asd)
            asd_y_pred_mode_all.append(nan_asd)
            asd_log_mu_all.append(nan_asd)
            asd_log_sigma_all.append(nan_asd)
            asd_y_std_all.append(asd_std)
            asd_lower_all.append(asd_lo)
            asd_upper_all.append(asd_hi)

    y_pred = np.column_stack(y_pred_primary_list)
    y_std = np.column_stack(y_std_primary_list)
    lower_95 = np.column_stack(lower_primary_list)
    upper_95 = np.column_stack(upper_primary_list)
    per_task = compute_per_task_metrics(y_test_np, y_pred, names)
    rel_metrics_by_task: dict[str, dict[str, float | int]] = {}
    for t, name in enumerate(names):
        rel = compute_relative_error_metrics(
            y_test_np[:, t], y_pred[:, t], rel_tolerance=rel_tolerance
        )
        rel_metrics_by_task[name] = rel
        for key, value in rel.items():
            per_task[f"{name}_{key}"] = value
    aggregate_rrmse = float(
        np.mean([mode_metrics["full"][n]["RRMSE"] for n in names])
    )
    coverage_by_task = {
        name: compute_prediction_coverage_metrics(
            y_test_np[:, i], y_pred[:, i], y_std[:, i]
        )
        for i, name in enumerate(names)
    }

    opt_name = "LBFGSScipy" if num_epochs <= 1 else "Adam"
    log_task_names = [n for n in names if task_uses_log_scale(n, warps=warps)]
    logit_task_names = [n for n in names if task_uses_logit_scale(n, warps=warps)]
    metrics: dict[str, Any] = {
        "title": title,
        "dataset": "s4_emit",
        "model_class": "GPR_partitioned",
        "input_dim_original": S4_INPUT_DIM,
        "n_train": n_train,
        "n_test": n_test,
        "num_tasks": num_tasks,
        "task_names": list(names),
        "n_pca_components": (
            n_components_by_task if len(unique_ps) > 1 else unique_ps[0]
        ),
        "n_components_by_task": dict(n_components_by_task),
        "pca_components_meta": pca_components_meta,
        "pca_svd_solver": pca_svd_solver,
        "partition_size": partition_size,
        "n_partitions_by_task": n_partitions_by_task,
        "partition_shuffle": partition_shuffle,
        "top_m_partitions": top_m_partitions,
        "single_partition_index": single_partition_index,
        "primary_ensemble_mode": "full",
        "pca_partition_mode": "independent_gpr_per_task_partition",
        "ard": ard,
        "nigp": bool(nigp),
        "freeze_epoch_nigp": int(freeze_epoch_nigp),
        "num_epochs": num_epochs,
        "num_inits": num_inits,
        "optimizer": opt_name,
        "optimizer_kwargs": json_safe_optimizer_kwargs(optimizer_kwargs),
        "standardize_x": standardize_x,
        "x_standardize_method": x_standardize_method,
        "standardize_y": standardize_y,
        "log_scale_tasks": log_task_names,
        "logit_scale_tasks": logit_task_names,
        "log_offsets": dict(warps.log_offsets),
        "task_band_config": task_band_config,
        "bands_by_task": band_config_metadata(
            bands_by_task, wavelengths_nm=wavelengths_np
        ),
        "input_bands_by_task": input_bands_by_task,
        "pca_by_task": pca_by_task,
        "data_meta": {
            **data_meta,
            "emit_path": str(emit_path.resolve()),
        },
        "Training_Time": total_train_time,
        "Prediction_Time": total_prediction_time,
        "Total_Time": total_train_time + total_prediction_time,
        "aggregate_RRMSE": aggregate_rrmse,
        "ensemble_modes": mode_metrics,
        "ensemble_diagnostics": ensemble_diagnostics,
        "partition_runs": partition_runs,
        "coverage_by_task": coverage_by_task,
        **per_task,
    }
    for task_name in names:
        tm = mode_metrics["full"][task_name]
        for key, value in tm.items():
            metrics[f"{task_name}_{key}"] = value

    print(
        f"\nPrimary ensemble (full) aggregate RRMSE={aggregate_rrmse:.6f}  "
        f"train={total_train_time:.1f}s  pred={total_prediction_time:.1f}s"
    )
    for name in names:
        print(
            f"  {name}: RRMSE={metrics[f'{name}_RRMSE']:.6f}  "
            f"RMSE={metrics[f'{name}_RMSE']:.6f}"
        )
    for mode in ENSEMBLE_MODE_NAMES:
        vals = [mode_metrics[mode][n]["RRMSE"] for n in names]
        print(f"  ensemble[{mode:18s}] macro RRMSE={float(np.mean(vals)):.6f}")
    for name in names:
        print(format_coverage_summary(name, coverage_by_task[name]))

    Path(save_path).mkdir(parents=True, exist_ok=True)
    out_json = save_metrics_json(metrics, save_path, title)
    print(f"Saved metrics to {out_json}")
    out_npz = save_s2_predictions_npz(
        save_path,
        title=title,
        task_names=names,
        y_true=y_test_np,
        y_pred=y_pred,
        y_std=y_std,
        lower=lower_95,
        upper=upper_95,
        x_test=x_test_full.detach().cpu().numpy()[:, :S4_SPECTRAL_DIM],
        wavelengths_nm=wavelengths_np,
        train_idx=train_idx.cpu().numpy(),
        val_idx=val_idx.cpu().numpy(),
        test_idx=test_idx.cpu().numpy(),
        bands_by_task=bands_by_task,
        y_pred_mean=np.column_stack(y_pred_mean_all),
        y_pred_mode=np.column_stack(y_pred_mode_all),
        log_mu=np.column_stack(log_mu_all),
        log_sigma=np.column_stack(log_sigma_all),
        log_scale_tasks=log_task_names,
        logit_scale_tasks=logit_task_names,
    )
    print(f"Saved predictions to {out_npz}")
    metrics["predictions_npz"] = str(out_npz)

    if plot_posterior:
        scatter_dir = Path(save_path) / "plots" / "scatter" / title
        for t, name in enumerate(names):
            yt_s, yp_s, lo_s, hi_s = _subsample_scatter(
                y_test_np[:, t],
                y_pred[:, t],
                lower_95[:, t],
                upper_95[:, t],
                seed=seed + t,
            )
            plot_s2_task_scatter(
                y_true=yt_s,
                y_pred=yp_s,
                lower=lo_s,
                upper=hi_s,
                task_name=name,
                out_path=scatter_dir / f"{name}_scatter.png",
                title=f"{title} | {name} (ensemble=full)",
                test_rrmse=float(metrics[f"{name}_RRMSE"]),
            )
        example_indices = select_posterior_example_indices(
            n_test,
            posterior_n_examples,
            seed=seed,
            explicit_indices=posterior_example_indices,
        )
        if example_indices:
            post_dir = Path(save_path) / "plots" / "posterior" / title
            post_paths = plot_s2_posterior_examples(
                x_test=x_test_full.detach().cpu().numpy()[:, :S4_SPECTRAL_DIM],
                wavelengths_nm=wavelengths_np,
                y_true=y_test_np,
                y_pred=y_pred,
                lower=lower_95,
                upper=upper_95,
                task_names=names,
                example_indices=example_indices,
                save_dir=post_dir,
                title=title,
                y_std=y_std,
                rel_metrics_by_task=rel_metrics_by_task,
                rel_tolerance=rel_tolerance,
                log_scale_tasks=log_task_names,
                logit_scale_tasks=logit_task_names,
                spectrum_ylabel="Radiance",
            )
            for p in post_paths:
                print(f"Saved posterior plot to {p}")
        cov_dir = Path(save_path) / "plots" / "coverage" / title
        for cov_path in plot_prediction_coverage(
            task_names=names,
            coverage_by_task=coverage_by_task,
            save_dir=cov_dir,
            title=title,
        ):
            print(f"Saved coverage plot to {cov_path}")

    if eval_asd and asd_task_order and asd_x_np is not None:
        flat_metrics: dict[str, float] = {}
        for name, row in asd_per_task.items():
            for key, value in row.items():
                if key.startswith(f"{name}_") or key in {
                    "RMSE",
                    "MAE",
                    "MedAE",
                    "RRMSE",
                    "R2",
                    "NLPD",
                    "NIS",
                    "PICP_95",
                    "MPIW_95",
                }:
                    if key.startswith(f"{name}_"):
                        flat_metrics[key] = value
                    else:
                        flat_metrics[f"{name}_{key}"] = value
        aggregate_asd_rrmse = macro_metric(flat_metrics, asd_task_order, "RRMSE")
        aggregate_asd_r2 = macro_metric(flat_metrics, asd_task_order, "R2")
        asd_results = {
            "title": f"asd_validation_eval_{Path(save_path).name}",
            "ckpt_dir": str(Path(save_path).resolve()),
            "asd_path": str(asd_file.resolve()),
            "model_class": "GPR_partitioned",
            "task_names": asd_task_order,
            "missing_tasks": sorted(set(TASK_LABEL_MAP) - set(asd_task_order)),
            "checkpoint_title": title,
            "ensemble_mode": "full",
            "log_scale_tasks": sorted(warps.log_tasks),
            "logit_scale_tasks": sorted(warps.logit_tasks),
            "log_offsets": dict(warps.log_offsets),
            "label_map": {t: TASK_LABEL_MAP[t] for t in asd_task_order},
            "aux_meta": asd_aux_meta,
            "coszen_override": None,
            "coszen_asd_original": (
                coszen_asd_original.tolist()
                if coszen_asd_original is not None
                else None
            ),
            "aggregate_RRMSE": aggregate_asd_rrmse,
            "aggregate_R2": aggregate_asd_r2,
            "per_task": asd_per_task,
            "coverage_by_task": asd_coverage_by_task,
            **flat_metrics,
        }
        print(
            f"\nASD Aggregate RRMSE={aggregate_asd_rrmse:.6f}  "
            f"R2={aggregate_asd_r2:.6f}"
        )
        eval_dir = write_asd_eval_artifacts(
            asd_results,
            eval_dir=Path(save_path) / "asd_validation_eval",
            asd_path=asd_file,
            prediction_arrays={
                "task_names": np.array(asd_task_order),
                "y_true": np.column_stack(asd_y_true_all),
                "y_pred": np.column_stack(asd_y_pred_all),
                "y_pred_mean": np.column_stack(asd_y_pred_mean_all),
                "y_pred_mode": np.column_stack(asd_y_pred_mode_all),
                "log_mu": np.column_stack(asd_log_mu_all),
                "log_sigma": np.column_stack(asd_log_sigma_all),
                "y_std": np.column_stack(asd_y_std_all),
                "lower_95": np.column_stack(asd_lower_all),
                "upper_95": np.column_stack(asd_upper_all),
                "x_spectral": asd_x_np[:, :S4_SPECTRAL_DIM],
            },
            plot=plot_asd,
            sim_path=emit_path,
            neighbor_thr=neighbor_thr,
            run_neighbors=True,
        )
        metrics["asd_validation"] = {
            "aggregate_RRMSE": aggregate_asd_rrmse,
            "aggregate_R2": aggregate_asd_r2,
            "task_names": asd_task_order,
            "eval_dir": str(eval_dir),
        }

    return metrics


__all__ = [
    "S4_TASK_NAMES",
    "parse_s4_task_names",
    "run_s4_emit_gp",
]
