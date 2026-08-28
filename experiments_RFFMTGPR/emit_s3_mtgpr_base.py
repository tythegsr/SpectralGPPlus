"""EMIT snow70 joint SORF+NIGP MTGPR runner (Woodbury, no SVGP/VIRFF).

Trains one ``RFFMTGPR`` with ``rff_sampling='sorf'`` on EMIT reflectance and
ISOFIT labels. Train/val/test are a seeded permutation of filtered rows:
10k / 5k / remaining (not the reserved TOA pool scheme).

Also exports the shared EMIT loader/split used by independent S3 SORF.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
_RFF_DIR = _ROOT / "experiments_RFF"
_MTGPR_DIR = Path(__file__).resolve().parent
_DEFAULT_SAVE_DIR = "experiments_RFFMTGPR/results/s3_emit_mtgpr"
_DEFAULT_EMIT_PATH = _ROOT / "split_files" / "emit_data_snow_70to100.nc"
_DEFAULT_WL_SRC = (
    _ROOT / "experiments_toa" / "data 11 QoI" / "snow_toa_fsnow_70to100_20261208.nc"
)
EMIT_INPUT_DIM = 285
# All QoIs available from ISOFIT state via ``mapped_qois`` (export_emit_as_s2).
S3_TASK_NAMES: tuple[str, ...] = (
    "algae",
    "aot",
    "cosA",
    "cwv",
    "dust",
    "fNPV",
    "fPV",
    "fsnow",
    "fsoil",
    "grain_size",
    "liquid_water",
    "sinA",
)
TASK_VALID_Y_RANGE: dict[str, tuple[float, float]] = {
    "algae": (1e-2, 6e5),
    "grain_size": (30.0, 1500.0),
    "fsnow": (0.7, 1.0),
    "fPV": (0.0, 1.0),
    "fNPV": (0.0, 1.0),
    "fsoil": (0.0, 1.0),
}
SCATTER_MAX_POINTS = 20_000

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments_toa.paths import pin_toa_import_paths

pin_toa_import_paths(_MTGPR_DIR, _RFF_DIR)

from experiments_toa.export_emit_as_s2 import mapped_qois
from experiments_toa.merge_emit_chunks import EMIT_STATE_FEATURE_NAMES
from experiments_toa.s2_data import append_elevation, read_elevation_array
from experiments_toa.s2_plotting import (
    plot_prediction_coverage,
    plot_s2_posterior_examples,
    plot_s2_task_scatter,
    save_s2_predictions_npz,
    select_posterior_example_indices,
)
from experiments_toa.s2_utils import (
    compute_floor_bound_slice_metrics,
    compute_log_scale_extra_metrics,
    compute_per_task_metrics,
    format_floor_bound_summary,
    macro_metric,
    macro_rrmse,
)
from experiments_toa.s2_y_transform import (
    resolve_y_warps,
    task_uses_log_scale,
    task_uses_logit_scale,
)
from gpplus.models import RFFMTGPR
from gpplus.training import (
    ConvergencePatienceStopCondition,
    GPTrainer,
    MinLossChangeStopCondition,
    NIGPInputNoiseFreezeCallback,
    NIGPMTWoodburyMarginalLogLikelihood,
    RFFMTParameterInitializer,
    evaluate_rff_mt_gp_model,
)
from gpplus.training.optimizers import LBFGSScipy
from gpplus.utils import StandardScaler, UniformScaler, compute_metrics, set_seed
from experiments_RFF.rff_gp_defaults import (
    DEFAULT_MT_WOODBURY_METHOD,
    build_rff_multitask_noise_likelihood,
    merge_mt_noise_initializer_kwargs,
    mt_eval_kwargs,
    mt_mll_class,
    woodbury_jitter_for_dtype,
)
from mtgpr_experiment_utils import (
    DEFAULT_ADAM_KWARGS,
    DEFAULT_LBFGS_KWARGS,
    DEFAULT_TOA_ADAM_LR,
    DEFAULT_TOA_ADAM_STOP_PATIENCE,
    compute_relative_error_metrics,
    compute_prediction_coverage_metrics,
    format_coverage_summary,
    format_relative_error_summary,
    json_safe_optimizer_kwargs,
    make_train_loss_callback,
    make_validation_callback,
    plot_validation_curves_after_save,
    save_metrics_json,
    summarize_validation_from_runs,
)
from toa_mtgpr_base import extract_mt_learned_noise
from toa_mtgpr_checkpoint import checkpoint_path_for_run, save_toa_mtgpr_checkpoint
from toa_s2_mtgpr_base import (
    forward_y_s2_matrix,
    inverse_y_s2_matrix,
)

RFF_SAMPLING_CHOICES = ("rff", "orf", "sorf")


def parse_s3_task_names(raw: str | Sequence[str] | None) -> list[str]:
    """Normalize a QoI list; ``None`` means all S3 tasks."""
    if raw is None:
        return list(S3_TASK_NAMES)
    values = [raw] if isinstance(raw, str) else list(raw)
    names = [
        name
        for value in values
        for name in (x.strip() for x in str(value).split(","))
        if name
    ]
    unknown = [n for n in names if n not in S3_TASK_NAMES]
    if unknown:
        raise ValueError(f"Unknown S3 task names: {unknown}. Valid: {list(S3_TASK_NAMES)}")
    if not names:
        raise ValueError("QoI selection produced an empty list")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate QoI names are not allowed: {duplicates}")
    return names


def _load_wavelengths_nm(emit_path: Path, wl_src: Path | None) -> np.ndarray:
    for path in (emit_path, wl_src):
        if path is None or not Path(path).is_file():
            continue
        import h5py

        with h5py.File(path, "r") as f:
            if "wl" in f:
                return np.asarray(f["wl"][:], dtype=np.float64)
    raise FileNotFoundError(
        f"No 'wl' dataset in {emit_path} and fallback wl_src={wl_src} is missing."
    )


def _valid_label_mask(y: np.ndarray, task_names: Sequence[str]) -> np.ndarray:
    mask = np.ones(y.shape[0], dtype=bool)
    for t, name in enumerate(task_names):
        bounds = TASK_VALID_Y_RANGE.get(name)
        if bounds is None:
            continue
        lo, hi = bounds
        mask &= (y[:, t] >= lo) & (y[:, t] <= hi)
    return mask


def load_emit_s3_xy(
    emit_path: str | Path,
    *,
    task_names: Sequence[str],
    nondefault_only: bool = True,
    filter_valid_labels: bool = True,
    include_elevation: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load EMIT reflectance and mapped QoIs; return filtered X, Y, source indices.

    When ``include_elevation`` is True, elevation (m) is appended as the last
    column of X (spectral bands first). Raises if elevation is requested but
    missing from the NetCDF.
    """
    import h5py

    emit_path = Path(emit_path)
    if not emit_path.is_file():
        raise FileNotFoundError(f"EMIT NetCDF not found: {emit_path}")
    names = list(task_names)
    unknown = [n for n in names if n not in S3_TASK_NAMES]
    if unknown:
        raise ValueError(f"Unknown S3 task names: {unknown}")

    with h5py.File(emit_path, "r") as f:
        n_total = int(f["reflectance"].shape[0])
        n_bands = int(f["reflectance"].shape[1])
        if n_bands != EMIT_INPUT_DIM:
            raise ValueError(f"Expected {EMIT_INPUT_DIM} bands, got {n_bands}")
        state = np.asarray(f["state"][:], dtype=np.float64)
        elev_all = read_elevation_array(f)
        if "state_feature_names" in f.attrs:
            names_attr = f.attrs["state_feature_names"]
            if isinstance(names_attr, bytes):
                names_attr = names_attr.decode("utf-8")
            state_names = [s.strip() for s in str(names_attr).split(",")]
        else:
            state_names = list(EMIT_STATE_FEATURE_NAMES)

        if nondefault_only:
            grain_col = EMIT_STATE_FEATURE_NAMES.index("grain_radius")
            keep = np.abs(state[:, grain_col] - 500.0) > 0.1
        else:
            keep = np.ones(n_total, dtype=bool)

        idx = np.flatnonzero(keep)
        X = np.asarray(f["reflectance"][idx], dtype=np.float64)
        elev = elev_all[idx] if elev_all is not None else None

    qoi = mapped_qois(state[idx])
    missing = [n for n in names if n not in qoi]
    if missing:
        raise KeyError(f"mapped_qois missing {missing}")
    Y = np.column_stack([qoi[n] for n in names])
    if filter_valid_labels:
        valid = _valid_label_mask(Y, names)
        X = X[valid]
        Y = Y[valid]
        idx = idx[valid]
        if elev is not None:
            elev = elev[valid]

    if include_elevation and elev is None:
        raise KeyError(f"include_elevation=True but 'elevation' missing in {emit_path}")
    X, elev_meta = append_elevation(X, elev if include_elevation else None)

    if not np.isfinite(X).all():
        raise ValueError("Non-finite values found in EMIT reflectance")
    if not np.isfinite(Y).all():
        raise ValueError("Non-finite values found in EMIT labels")

    meta = {
        "emit_path": str(emit_path.resolve()),
        "n_total": n_total,
        "n_filtered": int(idx.size),
        "nondefault_only": bool(nondefault_only),
        "filter_valid_labels": bool(filter_valid_labels),
        "state_feature_names": state_names,
        "task_names": names,
        "fsnow_label": "softmax_fraction(z_snow,z_pv,z_npv,z_soil)",
        "task_valid_y_range": {
            k: list(TASK_VALID_Y_RANGE[k]) for k in names if k in TASK_VALID_Y_RANGE
        },
        "input_variable": "reflectance",
        "has_elevation": bool(elev_meta["has_elevation"]),
        "n_spectral_bands": int(elev_meta["n_spectral_bands"]),
        "elevation_index": elev_meta["elevation_index"],
        "aux_inputs": list(elev_meta["aux_inputs"]),
        "input_dim": int(elev_meta["input_dim"]),
    }
    return X, Y, idx, meta


def split_emit_s3(
    n_filtered: int,
    *,
    n_train: int,
    n_val: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seeded permutation: train, val, remaining test (no reserved TOA pools)."""
    if n_train < 0 or n_val < 0:
        raise ValueError(f"n_train and n_val must be >= 0, got {n_train}, {n_val}")
    if n_train + n_val >= n_filtered:
        raise ValueError(
            f"Need leftover test rows: n_train={n_train} + n_val={n_val} "
            f">= n_filtered={n_filtered}"
        )
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_filtered)
    train_local = perm[:n_train]
    val_local = perm[n_train : n_train + n_val]
    test_local = perm[n_train + n_val :]
    return train_local, val_local, test_local


def _subsample_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    seed: int,
    max_points: int = SCATTER_MAX_POINTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = int(y_true.shape[0])
    if max_points <= 0 or n <= max_points:
        return y_true, y_pred, lower, upper
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=max_points, replace=False))
    return y_true[idx], y_pred[idx], lower[idx], upper[idx]


def run_s3_emit_mtgpr(
    n_train: int = 10000,
    n_val: int = 5000,
    num_rff: int | None = None,
    rff_sampling: Literal["rff", "orf", "sorf"] = "sorf",
    seed: int = 42,
    num_inits: int = 1,
    num_epochs: int = 2000,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    save_path: str | None = None,
    standardize_x: bool = True,
    x_standardize_method: int = 2,
    standardize_y: bool = True,
    ard: bool = True,
    predict_chunk_size: int = 2048,
    n_jobs: int | None = None,
    optimizer_kwargs: dict | None = None,
    monitor_validation: bool = True,
    validation_verbose: bool = True,
    plot_validation: bool = True,
    plot_posterior: bool = True,
    rel_tolerance: float = 0.01,
    posterior_n_examples: int = 20,
    posterior_example_indices: list[int] | None = None,
    data_path: str | None = None,
    wl_src: str | None = None,
    rank_kernel: int = 1,
    parallel_verbose: int = 10,
    training_verbose: bool = True,
    log_every_n_epochs: int = 50,
    save_checkpoint: bool = True,
    log_scale: bool | None = True,
    log_scale_qoi: Sequence[str] | None = None,
    logit_scale_qoi: Sequence[str] | None = None,
    logit_bounds: dict[str, tuple[float, float]] | None = None,
    response_noise_prior: bool = False,
    noise_var_fraction: float = 0.01,
    noise_prior_log_scale: float = 0.5,
    correct_sorf: bool = True,
    task_names: Sequence[str] | None = None,
    nigp: bool = True,
    freeze_epoch_nigp: int = 100,
    nondefault_only: bool = True,
    filter_valid_labels: bool = True,
    include_elevation: bool = False,
) -> dict:
    """Train joint SORF RFFMTGPR + NIGP on EMIT snow70 (Woodbury)."""
    if rff_sampling not in RFF_SAMPLING_CHOICES:
        raise ValueError(
            f"rff_sampling must be one of {RFF_SAMPLING_CHOICES}, got {rff_sampling!r}"
        )
    correct_sorf = bool(correct_sorf) if rff_sampling == "sorf" else False
    names = parse_s3_task_names(task_names)
    num_tasks = len(names)
    emit_path = Path(data_path) if data_path is not None else _DEFAULT_EMIT_PATH
    wl_path = Path(wl_src) if wl_src is not None else _DEFAULT_WL_SRC

    if save_path is None:
        save_path = _DEFAULT_SAVE_DIR

    set_seed(seed)
    if num_rff is None:
        num_rff = min(512, max(64, n_train // 3))

    if num_epochs <= 1:
        optimizer_class = LBFGSScipy
        default_optimizer_kwargs = DEFAULT_LBFGS_KWARGS
        stop_conditions = [
            ConvergencePatienceStopCondition(patience=10),
            MinLossChangeStopCondition(min_loss_change=1e-7),
        ]
    else:
        optimizer_class = torch.optim.Adam
        default_optimizer_kwargs = {**DEFAULT_ADAM_KWARGS, "lr": DEFAULT_TOA_ADAM_LR}
        stop_conditions = [
            ConvergencePatienceStopCondition(patience=DEFAULT_TOA_ADAM_STOP_PATIENCE),
        ]
    if optimizer_kwargs is None:
        optimizer_kwargs = dict(default_optimizer_kwargs)

    n_val_eff = int(n_val) if monitor_validation else 0
    X_np, Y_np, source_idx, data_meta = load_emit_s3_xy(
        emit_path,
        task_names=names,
        nondefault_only=nondefault_only,
        filter_valid_labels=filter_valid_labels,
        include_elevation=include_elevation,
    )
    n_filtered = int(X_np.shape[0])
    input_dim = int(X_np.shape[1])
    has_elevation = bool(data_meta.get("has_elevation", False))
    elevation_index = data_meta.get("elevation_index")
    train_local, val_local, test_local = split_emit_s3(
        n_filtered,
        n_train=n_train,
        n_val=n_val_eff,
        seed=seed,
    )
    n_test = int(test_local.size)
    train_idx = torch.as_tensor(source_idx[train_local], dtype=torch.int64)
    val_idx = torch.as_tensor(source_idx[val_local], dtype=torch.int64)
    test_idx = torch.as_tensor(source_idx[test_local], dtype=torch.int64)

    wavelengths = torch.as_tensor(_load_wavelengths_nm(emit_path, wl_path), dtype=torch.float64)
    wl_np = wavelengths.detach().cpu().numpy()
    shared_bands = list(range(EMIT_INPUT_DIM))
    bands_by_task = {name: list(shared_bands) for name in names}
    input_column_indices = list(range(input_dim))

    title = (
        f"S3_EMIT_MT_nTrain{n_train}_nVal{n_val_eff}_nTest{n_test}_"
        f"{rff_sampling}D{num_rff}_T{num_tasks}"
    )
    if rff_sampling == "sorf":
        title = f"{title}_correctSorf{correct_sorf}"
    if has_elevation:
        title = f"{title}_elev"
    feature_dim = 2 * num_rff
    joint_width = feature_dim * num_tasks
    sampling_label = rff_sampling.upper()

    print("=" * 60)
    print(title)
    print(
        f"Joint {sampling_label}-MTGP (Woodbury), D={num_rff}, m={feature_dim}, m*T={joint_width}, "
        f"ARD={ard}, dtype={dtype}, inits={num_inits}, epochs={num_epochs}, tasks={names}, "
        f"n_bands={EMIT_INPUT_DIM}, input_dim={input_dim}, elev={has_elevation}, "
        f"nigp={bool(nigp)}"
        + (f", correct_sorf={correct_sorf}" if rff_sampling == "sorf" else "")
    )
    print(f"Optimizer: {getattr(optimizer_class, '__name__', optimizer_class)}, kwargs={optimizer_kwargs}")
    print(f"Device: {device}")
    print(
        f"Split: filtered={n_filtered}  train={n_train}  val={n_val_eff}  test={n_test}  "
        f"(EMIT {emit_path})"
    )
    print("=" * 60)

    x_train = torch.as_tensor(X_np[train_local], dtype=dtype)
    y_train = torch.as_tensor(Y_np[train_local], dtype=dtype)
    x_test = torch.as_tensor(X_np[test_local], dtype=dtype)
    y_test = torch.as_tensor(Y_np[test_local], dtype=dtype)
    if n_val_eff > 0:
        x_val = torch.as_tensor(X_np[val_local], dtype=dtype)
        y_val = torch.as_tensor(Y_np[val_local], dtype=dtype)
    else:
        x_val = x_train.new_zeros((0, x_train.shape[-1]))
        y_val = y_train.new_zeros((0, y_train.shape[-1]))
    del X_np, Y_np

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
        + (
            f"; log_offsets={warps.active_log_offsets()}"
            if warps.active_log_offsets()
            else ""
        )
    )

    x_test_orig = x_test.clone()

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
            raise ValueError(f"x_standardize_method must be 0, 1, or 2, got {x_standardize_method}")
        x_scaler.fit(x_train)
        x_train = x_scaler.transform(x_train)
        x_test = x_scaler.transform(x_test)
        if x_val.numel() > 0:
            x_val = x_scaler.transform(x_val)
        print(f"X scaling: {x_scaling_type}")

    y_train_model = forward_y_s2_matrix(y_train, names, warps=warps)
    y_scaler = None
    if standardize_y:
        y_scaler = StandardScaler()
        y_scaler.fit(y_train_model)
        y_train_fit = y_scaler.transform(y_train_model)
    else:
        y_train_fit = y_train_model

    x_val_scaled = x_val
    y_val_scaled = y_val
    if y_val.numel() > 0:
        y_val_model = forward_y_s2_matrix(y_val, names, warps=warps)
        y_val_scaled = (
            y_scaler.transform(y_val_model) if standardize_y and y_scaler is not None else y_val_model
        )

    callbacks = []
    if nigp and int(freeze_epoch_nigp) > 0 and num_epochs > 1:
        callbacks.append(
            NIGPInputNoiseFreezeCallback(
                freeze_epochs=int(freeze_epoch_nigp),
                verbose=training_verbose,
            )
        )
    if num_epochs > 1 and training_verbose:
        callbacks.append(
            make_train_loss_callback(
                num_inits,
                num_epochs,
                verbose=training_verbose,
                log_every_n_epochs=log_every_n_epochs,
            )
        )
    if monitor_validation and n_val_eff > 0:
        callbacks.append(
            make_validation_callback(
                x_val_scaled,
                y_val_scaled,
                num_inits,
                chunk_size=predict_chunk_size,
                verbose=validation_verbose,
                log_every_n_epochs=log_every_n_epochs,
                woodbury_mt_method=DEFAULT_MT_WOODBURY_METHOD,
            )
        )

    noise_prior = None
    initializer_kwargs: dict | None = None
    parameter_configs: dict = {}
    noise_prior_meta: dict | None = None
    if response_noise_prior:
        from gpplus.priors.response_noise import (
            empirical_task_noise_variances,
            log_normal_noise_prior_from_responses,
            task_noise_raw_init_from_variances,
        )

        target_vars = empirical_task_noise_variances(y_train_fit, fraction=noise_var_fraction)
        noise_prior = log_normal_noise_prior_from_responses(
            y_train_fit,
            fraction=noise_var_fraction,
            log_scale=noise_prior_log_scale,
            dtype=dtype,
            device=y_train_fit.device,
        )
        temp_lik = build_rff_multitask_noise_likelihood(num_tasks, rank=0)
        raw_init = task_noise_raw_init_from_variances(temp_lik, target_vars.to(dtype=dtype))
        parameter_configs["raw_task_noises"] = {"method": "constant", "value": raw_init}
        noise_prior_meta = {
            "response_noise_prior": True,
            "noise_var_fraction": float(noise_var_fraction),
            "noise_prior_log_scale": float(noise_prior_log_scale),
            "noise_prior_loc": noise_prior.loc.detach().cpu().tolist(),
            "noise_prior_target_var": target_vars.detach().cpu().tolist(),
        }
        print(
            f"Response noise prior: LogNormal per task, fraction={noise_var_fraction}, "
            f"log_scale={noise_prior_log_scale}"
        )

    if parameter_configs:
        initializer_kwargs = {"parameter_configs": parameter_configs}
    initializer_kwargs = merge_mt_noise_initializer_kwargs(initializer_kwargs)

    likelihood = build_rff_multitask_noise_likelihood(num_tasks, noise_prior=noise_prior)
    model = RFFMTGPR(
        x_train,
        y_train_fit,
        num_tasks=num_tasks,
        num_rff=num_rff,
        ard=ard,
        rff_sampling=rff_sampling,
        correct_sorf=correct_sorf,
        rank_kernel=rank_kernel,
        rank_likelihood=0,
        likelihood=likelihood,
        nigp=bool(nigp),
    )
    if nigp:
        print(
            "NIGP: on (independent input noise, sigma_x=10^SoftClamp(raw), "
            f"freeze_epochs={int(freeze_epoch_nigp)})"
        )
        _mt_method = DEFAULT_MT_WOODBURY_METHOD

        class _NIGPMTMLL(NIGPMTWoodburyMarginalLogLikelihood):
            def __init__(self, likelihood, model, jitter: float = 1e-6):
                super().__init__(likelihood, model, jitter=jitter, method=_mt_method)

        _NIGPMTMLL.__name__ = f"NIGPMTWoodburyMarginalLogLikelihood_{_mt_method}"
        mll_cls = _NIGPMTMLL
    else:
        mll_cls = mt_mll_class()

    trainer = GPTrainer(
        model,
        mll_class=mll_cls,
        num_epochs=num_epochs,
        num_inits=num_inits,
        seed=seed,
        device=device,
        dtype=dtype,
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
        initializer_class=RFFMTParameterInitializer,
        initializer_kwargs=initializer_kwargs,
        n_jobs=n_jobs,
        inner_max_num_threads=1,
        cholesky_jitter=woodbury_jitter_for_dtype(dtype),
        callbacks=callbacks,
        stop_conditions=stop_conditions,
        parallel_verbose=parallel_verbose,
        min_epochs=(
            int(freeze_epoch_nigp)
            if nigp and int(freeze_epoch_nigp) > 0 and num_epochs > 1
            else 0
        ),
    )
    t_train = time.time()
    runs = trainer.train()
    train_time = time.time() - t_train

    successful = [r for r in runs if r.get("loss") is not None and r.get("state_dict") is not None]
    if not successful:
        errors = [r.get("error", "unknown") for r in runs if r.get("error")]
        raise RuntimeError(
            "All training runs failed for joint S3 EMIT MT model. "
            + (f"First error: {errors[0]}" if errors else "Check optimizer kwargs.")
        )
    best_run = min(successful, key=lambda r: r["loss"])
    model.load_state_dict(best_run["state_dict"])
    best_loss = float(best_run["loss"])
    learned_noise = extract_mt_learned_noise(model)
    final_lr = float(best_run["final_lr"]) if best_run.get("final_lr") is not None else None

    model.eval()
    model.invalidate_feature_cache()
    t_pred = time.time()
    pred_mean, lower, upper, pred_std = evaluate_rff_mt_gp_model(
        model, x_test, chunk_size=predict_chunk_size, **mt_eval_kwargs(dtype)
    )
    prediction_time = time.time() - t_pred

    inv = inverse_y_s2_matrix(
        pred_mean.detach().cpu(),
        pred_std.detach().cpu(),
        lower.detach().cpu(),
        upper.detach().cpu(),
        task_names=names,
        y_scaler=y_scaler,
        standardize_y=standardize_y,
        warps=warps,
    )
    y_true_np = y_test.cpu().numpy()
    y_pred_np = inv.point.numpy()
    y_pred_mean_np = inv.point_mean.numpy() if inv.point_mean is not None else y_pred_np
    y_pred_mode_np = inv.point_mode.numpy() if inv.point_mode is not None else None
    pred_std_np = inv.std.numpy()
    lower_np = inv.lower.numpy()
    upper_np = inv.upper.numpy()

    per_task = compute_per_task_metrics(y_true_np, y_pred_np, task_names=names)

    rel_metrics_by_task: dict[str, dict] = {}
    cov_metrics_by_task: dict[str, dict] = {}
    for t, name in enumerate(names):
        rel_m = compute_relative_error_metrics(
            y_true_np[:, t],
            y_pred_np[:, t],
            rel_tolerance=rel_tolerance,
        )
        cov_m = compute_prediction_coverage_metrics(
            y_true_np[:, t],
            y_pred_np[:, t],
            pred_std_np[:, t],
        )
        rel_metrics_by_task[name] = rel_m
        cov_metrics_by_task[name] = cov_m
        per_task[f"{name}_max_rel_error"] = float(rel_m["max_rel_error"])
        per_task[f"{name}_mean_rel_error"] = float(rel_m["mean_rel_error"])
        per_task[f"{name}_median_rel_error"] = float(rel_m["median_rel_error"])
        per_task[f"{name}_pct_within_1pct"] = float(rel_m["pct_within_1pct"])
        per_task[f"{name}_coverage_50"] = float(cov_m["coverage_50"])
        per_task[f"{name}_coverage_90"] = float(cov_m["coverage_90"])
        per_task[f"{name}_coverage_95"] = float(cov_m["coverage_95"])
        if task_uses_logit_scale(name, warps=warps):
            per_task[f"{name}_logit_scale"] = True
        if task_uses_log_scale(name, warps=warps) and inv.log_mu is not None:
            c = float(warps.log_offset(name))
            extra = compute_log_scale_extra_metrics(
                y_true_np[:, t],
                log_mu=inv.log_mu[:, t].numpy(),
                point_mean_physical=(
                    y_pred_mean_np[:, t] if y_pred_mean_np is not None else None
                ),
                log_offset=c,
            )
            for key, value in extra.items():
                per_task[f"{name}_{key}"] = float(value)
            if c > 0.0:
                per_task[f"{name}_log_offset"] = c

    per_task.update(
        compute_floor_bound_slice_metrics(y_true_np, y_pred_np, task_names=names)
    )

    aggregate_rmse = float(np.sqrt(np.mean((y_pred_np - y_true_np) ** 2)))
    aggregate_rrmse = float(macro_rrmse(per_task, names))
    aggregate_medae = float(macro_metric(per_task, names, "MedAE"))
    log_task_names = [n for n in names if task_uses_log_scale(n, warps=warps)]
    logit_task_names = [n for n in names if task_uses_logit_scale(n, warps=warps)]
    aggregate_rrmse_log = macro_metric(per_task, log_task_names, "RRMSE_log")
    aggregate_rrmse_lnorm_mean = macro_metric(per_task, log_task_names, "RRMSE_mean")
    off_floor_names = [n for n in names if f"{n}_off_floor_RRMSE" in per_task]
    aggregate_rrmse_off_floor = macro_metric(per_task, off_floor_names, "off_floor_RRMSE")

    prob_metrics: dict[str, float] = {}
    for t, name in enumerate(names):
        kwargs: dict = {
            "output_std": torch.as_tensor(pred_std_np[:, t]),
            "lower_95": torch.as_tensor(lower_np[:, t]),
            "upper_95": torch.as_tensor(upper_np[:, t]),
            "training_time": train_time / num_tasks,
            "prediction_time": prediction_time / num_tasks,
        }
        if (
            task_uses_log_scale(name, warps=warps)
            and inv.log_mu is not None
            and inv.log_sigma is not None
        ):
            kwargs["log_mu"] = inv.log_mu[:, t]
            kwargs["log_sigma"] = inv.log_sigma[:, t]
        computed = compute_metrics(
            torch.as_tensor(y_true_np[:, t]),
            torch.as_tensor(y_pred_np[:, t]),
            **kwargs,
        )
        for key, value in computed.items():
            prob_metrics[f"{name}_{key}"] = value

    print(
        f"\nTest macro RRMSE (physical median): {aggregate_rrmse:.6f}  "
        f"RMSE: {aggregate_rmse:.6f}  MedAE: {aggregate_medae:.6f}"
    )
    if off_floor_names:
        print(f"Test macro RRMSE (off-floor subsets): {aggregate_rrmse_off_floor:.6f}")
    for name in names:
        print(
            f"{name} RRMSE: {per_task[f'{name}_RRMSE']:.6f}  "
            f"RMSE: {per_task[f'{name}_RMSE']:.6f}  "
            f"MAE: {per_task[f'{name}_MAE']:.6f}  "
            f"MedAE: {per_task[f'{name}_MedAE']:.6f}"
        )
        summary = format_floor_bound_summary(name, per_task)
        if summary:
            print(summary)
        print(format_relative_error_summary(name, rel_metrics_by_task[name], rel_tolerance=rel_tolerance))
        print(format_coverage_summary(name, cov_metrics_by_task[name]))
    print(f"Total training time: {train_time:.1f}s")

    warped_tasks = bool(log_task_names or logit_task_names)
    metrics: dict = {
        "title": title,
        "dataset": "emit_snow70",
        "input_dim": input_dim,
        "input_dim_original": EMIT_INPUT_DIM,
        "n_spectral_bands": EMIT_INPUT_DIM,
        "has_elevation": has_elevation,
        "elevation_index": elevation_index,
        "aux_inputs": list(data_meta.get("aux_inputs", [])),
        "shared_band_indices": list(shared_bands),
        "input_column_indices": list(input_column_indices),
        "n_train": n_train,
        "n_val": n_val_eff,
        "n_test": n_test,
        "n_filtered": n_filtered,
        "split": "permutation_train_val_remaining_test",
        "num_tasks": num_tasks,
        "task_names": list(names),
        "num_rff": num_rff,
        "rff_sampling": rff_sampling,
        "correct_sorf": correct_sorf,
        "feature_dim": feature_dim,
        "joint_feature_dim": joint_width,
        "rank_kernel": rank_kernel,
        "nigp": bool(nigp),
        "freeze_epoch_nigp": int(freeze_epoch_nigp),
        "ard": ard,
        "model_class": "RFFMTGPR",
        "num_epochs": num_epochs,
        "optimizer": getattr(optimizer_class, "__name__", str(optimizer_class)),
        "optimizer_kwargs": json_safe_optimizer_kwargs(optimizer_kwargs),
        "initial_lr": float(optimizer_kwargs.get("lr")) if "lr" in optimizer_kwargs else None,
        "final_lr": final_lr,
        "standardize_x": standardize_x,
        "x_standardize_method": x_standardize_method,
        "x_scaling_type": x_scaling_type,
        "standardize_y": standardize_y,
        "log_scale": bool(log_scale),
        "log_scale_tasks": log_task_names,
        "log_scale_source": log_scale_source,
        "log_offsets": warps.active_log_offsets(),
        "logit_scale_tasks": list(logit_task_names),
        "logit_scale_source": warps.logit_source,
        "response_noise_prior": bool(response_noise_prior),
        "best_train_loss": best_loss,
        "rel_tolerance": rel_tolerance,
        "data_meta": data_meta,
        **learned_noise,
        "Training_Time": train_time,
        "Prediction_Time": prediction_time,
        "Total_Time": train_time + prediction_time,
        "RMSE": aggregate_rmse,
        "aggregate_RRMSE": aggregate_rrmse,
        "aggregate_MedAE": aggregate_medae,
        "aggregate_RRMSE_log_tasks": aggregate_rrmse_log,
        "aggregate_RRMSE_lognormal_mean_tasks": aggregate_rrmse_lnorm_mean,
        "aggregate_RRMSE_off_floor": aggregate_rrmse_off_floor,
        **per_task,
        **prob_metrics,
    }
    if noise_prior_meta is not None:
        metrics.update(noise_prior_meta)

    if monitor_validation and n_val_eff > 0:
        metrics["monitor_validation"] = True
        metrics.update(summarize_validation_from_runs(runs, best_run))

    if save_path:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        if save_checkpoint:
            ckpt_path = save_toa_mtgpr_checkpoint(
                checkpoint_path_for_run(save_path, title),
                model=model,
                train_x=x_train.cpu(),
                train_y=y_train_fit.cpu(),
                x_scaler=x_scaler,
                y_scaler=y_scaler,
                standardize_x=standardize_x,
                standardize_y=standardize_y,
                x_standardize_method=x_standardize_method,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                title=title,
                seed=seed,
                best_train_loss=best_loss,
                n_train=n_train,
                n_test=n_test,
                n_val=n_val_eff,
                data_path=str(emit_path),
                rel_tolerance=rel_tolerance,
                dtype=dtype,
                log_grain=bool(log_task_names),
                logit_cos=bool(logit_task_names),
                input_column_indices=torch.as_tensor(input_column_indices, dtype=torch.int64),
                model_config={
                    "num_tasks": num_tasks,
                    "num_rff": num_rff,
                    "ard": ard,
                    "rff_sampling": rff_sampling,
                    "correct_sorf": correct_sorf,
                    "rank_kernel": rank_kernel,
                    "rank_likelihood": 0,
                    "nigp": bool(nigp),
                    "include_elevation": bool(has_elevation),
                },
            )
            metrics["checkpoint_path"] = str(ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

        example_indices = select_posterior_example_indices(
            y_true_np.shape[0],
            posterior_n_examples,
            seed=seed,
            explicit_indices=posterior_example_indices,
        )
        out_npz = save_s2_predictions_npz(
            save_path,
            title=title,
            task_names=names,
            y_true=y_true_np,
            y_pred=y_pred_np,
            y_std=pred_std_np,
            lower=lower_np,
            upper=upper_np,
            x_test=x_test_orig.numpy(),
            wavelengths_nm=wl_np,
            train_idx=train_idx.cpu().numpy(),
            val_idx=val_idx.cpu().numpy(),
            test_idx=test_idx.cpu().numpy(),
            bands_by_task=bands_by_task,
            y_pred_mean=y_pred_mean_np if warped_tasks else None,
            y_pred_mode=y_pred_mode_np if log_task_names else None,
            log_mu=inv.log_mu.numpy() if log_task_names and inv.log_mu is not None else None,
            log_sigma=inv.log_sigma.numpy() if log_task_names and inv.log_sigma is not None else None,
            log_scale_tasks=log_task_names,
            logit_mu=(
                inv.logit_mu.numpy()
                if logit_task_names and inv.logit_mu is not None
                else None
            ),
            logit_sigma=(
                inv.logit_sigma.numpy()
                if logit_task_names and inv.logit_sigma is not None
                else None
            ),
            logit_scale_tasks=logit_task_names,
        )
        print(f"Saved predictions to {out_npz}")
        metrics["predictions_npz"] = str(out_npz)

        out_json = save_metrics_json(metrics, save_path, title)
        print(f"Saved metrics to {out_json}")

        if plot_validation and monitor_validation and n_val_eff > 0:
            for plot_path in plot_validation_curves_after_save(metrics, save_path, out_json):
                print(f"Saved validation plot to {plot_path}")

        if plot_posterior:
            scatter_dir = Path(save_path) / "plots" / "scatter" / title
            for t, name in enumerate(names):
                yt_s, yp_s, lo_s, hi_s = _subsample_scatter(
                    y_true_np[:, t],
                    y_pred_np[:, t],
                    lower_np[:, t],
                    upper_np[:, t],
                    seed=seed + t,
                )
                plot_s2_task_scatter(
                    y_true=yt_s,
                    y_pred=yp_s,
                    lower=lo_s,
                    upper=hi_s,
                    task_name=name,
                    out_path=scatter_dir / f"{name}_scatter.png",
                    title=f"{title} | {name}",
                    test_rrmse=float(per_task[f"{name}_RRMSE"]),
                )
            if example_indices:
                post_dir = Path(save_path) / "plots" / "posterior" / title
                post_paths = plot_s2_posterior_examples(
                    x_test=x_test_orig.numpy()[:, :EMIT_INPUT_DIM],
                    wavelengths_nm=wl_np,
                    y_true=y_true_np,
                    y_pred=y_pred_np,
                    lower=lower_np,
                    upper=upper_np,
                    task_names=names,
                    example_indices=example_indices,
                    save_dir=post_dir,
                    title=title,
                    y_std=pred_std_np,
                    rel_metrics_by_task=rel_metrics_by_task,
                    rel_tolerance=rel_tolerance,
                    log_scale_tasks=log_task_names,
                    y_pred_mean=y_pred_mean_np if warped_tasks else None,
                    y_pred_mode=y_pred_mode_np if log_task_names else None,
                    log_mu=inv.log_mu.numpy() if log_task_names and inv.log_mu is not None else None,
                    log_sigma=(
                        inv.log_sigma.numpy()
                        if log_task_names and inv.log_sigma is not None
                        else None
                    ),
                    spectrum_ylabel="Reflectance",
                )
                for p in post_paths:
                    print(f"Saved posterior plot to {p}")

            cov_dir = Path(save_path) / "plots" / "coverage" / title
            for cov_path in plot_prediction_coverage(
                task_names=names,
                coverage_by_task=cov_metrics_by_task,
                save_dir=cov_dir,
                title=title,
            ):
                print(f"Saved coverage plot to {cov_path}")

    return metrics


__all__ = [
    "EMIT_INPUT_DIM",
    "S3_TASK_NAMES",
    "TASK_VALID_Y_RANGE",
    "load_emit_s3_xy",
    "parse_s3_task_names",
    "run_s3_emit_mtgpr",
    "split_emit_s3",
    "_DEFAULT_EMIT_PATH",
    "_DEFAULT_WL_SRC",
    "_load_wavelengths_nm",
    "_subsample_scatter",
]
