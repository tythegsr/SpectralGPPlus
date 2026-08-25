"""
S2 independent RFF/ORF/SORF runner with per-QoI original-band subsets.

Each task gets its own band selection, X scaler, optional PCA, ARD vector,
metrics, checkpoint, and prediction column.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
_RFF_DIR = Path(__file__).resolve().parent
_MTGPR_DIR = _ROOT / "experiments_RFFMTGPR"
_DEFAULT_SAVE_DIRS = {
    "rff": "experiments_RFF/results/s2_toa_rff",
    "orf": "experiments_ORF/results/s2_toa_orf",
    "sorf": "experiments_SORF/results/s2_toa_sorf",
}

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments_toa.paths import pin_toa_import_paths

pin_toa_import_paths(_MTGPR_DIR, _RFF_DIR)

from experiments_toa.data import (
    TOA_TEST_POOL_SIZE,
    TOA_TRAIN_POOL_SIZE,
    TOA_VAL_POOL_SIZE,
)
from experiments_toa.s2_bands import (
    NComponentsSpec,
    band_config_metadata,
    load_task_band_config,
    resolve_task_pca_components,
)
from experiments_toa.s2_constants import S2_DEFAULT_BAND_CONFIG_PATH, S2_INPUT_DIM, S2_TASK_NAMES
from experiments_toa.s2_data import load_s2_toa_data
from experiments_toa.s2_plotting import (
    plot_s2_posterior_examples,
    plot_s2_task_scatter,
    save_s2_predictions_npz,
    select_posterior_example_indices,
)
from experiments_toa.s2_reporting import (
    plot_per_task_validation_curves,
    print_end_of_run_summary,
    print_task_test_metrics,
)
from experiments_toa.s2_utils import (
    apply_x_transform,
    compute_log_scale_extra_metrics,
    compute_nigp_grad_diagnostics,
    compute_per_task_metrics,
    extract_ard_lengthscales,
    extract_nigp_input_noise,
    macro_metric,
    macro_rrmse,
    map_ard_to_bands,
    map_input_noise_to_bands,
    select_bands,
)
from experiments_toa.s2_bound_penalty import (
    bound_penalty_metrics,
    learned_bound_penalty_lambda,
    resolve_bound_penalty_for_tasks,
)
from experiments_toa.s2_y_transform import (
    forward_y_s2,
    inverse_y_s2,
    resolve_y_warps,
    task_uses_log_scale,
    task_uses_logit_scale,
)
from gpplus.means import NeuralMean
from gpplus.models import RFFGPR
from gpplus.training import (
    ConvergencePatienceStopCondition,
    GPTrainer,
    MinLossChangeStopCondition,
    RFFParameterInitializer,
    evaluate_rff_gp_model,
)
from gpplus.training.optimizers import LBFGSScipy
from gpplus.utils import StandardScaler, UniformScaler, compute_metrics, set_seed
from experiments_RFF.rff_gp_defaults import (
    DEFAULT_WOODBURY_FORM,
    build_rff_scalar_noise_likelihood,
    merge_rff_noise_initializer_kwargs,
    rff_eval_kwargs,
    rff_mll_class,
    woodbury_jitter_for_dtype,
)
from gpplus.training import (
    bound_penalized_rff_mll_class,
    nigp_woodbury_mll_class,
    pac_bayes_mll_class,
)
from mtgpr_experiment_utils import (
    DEFAULT_ADAM_KWARGS,
    DEFAULT_LBFGS_KWARGS,
    DEFAULT_TOA_ADAM_LR,
    DEFAULT_TOA_ADAM_STOP_PATIENCE,
    compute_relative_error_metrics,
    format_relative_error_summary,
    json_safe_optimizer_kwargs,
    make_train_loss_callback,
    make_validation_callback,
    save_metrics_json,
    summarize_validation_from_runs,
)
from rff_experiment_utils import extract_learned_likelihood_noise
from toa_stgp_checkpoint import checkpoint_path_for_run, save_toa_stgp_checkpoint

RFF_SAMPLING_CHOICES = ("rff", "orf", "sorf")
SPECTRAL_KERNEL_CHOICES = ("rbf", "matern32")


def run_s2_toa_stgp(
    n_train: int = 16000,
    n_test: int = 5000,
    num_rff: int | None = None,
    rff_sampling: Literal["rff", "orf", "sorf"] = "rff",
    spectral_kernel: Literal["rbf", "matern32"] = "rbf",
    seed: int = 42,
    num_inits: int = 1,
    num_epochs: int = 1000,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
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
    parallel_verbose: int = 10,
    training_verbose: bool = True,
    log_every_n_epochs: int = 1,
    save_checkpoint: bool = True,
    response_noise_prior: bool = True,
    noise_var_fraction: float = 0.001,
    noise_prior_log_scale: float = 0.5,
    initializer_parameter_configs: dict | None = None,
    n_pca_components: NComponentsSpec | None = None,
    pca_svd_solver: str = "randomized",
    correct_sorf: bool = False,
    param_groups_fn=None,
    input_variable: str = "toa_reflectance",
    task_names: Sequence[str] | None = None,
    task_band_config: str | None = None,
    x_transform: str | None = None,
    train_mode: str = "independent",
    init_batch_size: int | None = None,
    bound_min: Sequence[float | None] | None = None,
    bound_max: Sequence[float | None] | None = None,
    bound_penalty_k: float = 2.0,
    bound_penalty_lambda: float = 1.0,
    bound_penalty_alpha: float = 10.0,
    bound_penalty_max_points: int | None = 4096,
    bound_penalty_lambda_learnable: bool = False,
    bound_penalty_lam_min: float = 1.0,
    pac_bayes: bool = False,
    pac_bayes_temperature: float = 1.0,
    pac_bayes_prior_std: float = 1.0,
    pac_bayes_posterior_std: float = 0.1,
    nigp: bool = False,
    freeze_epoch_nigp: int = 100,
    nigp_slope_refreshes: int | None = None,
    freeze_epoch_noise: int = 0,
    adam_stop_patience: int | None = None,
    mean_type: Literal["constant", "neural"] = "constant",
    neural_mean_hidden: Sequence[int] | None = (64, 32),
    neural_mean_activation: str = "relu",
    minibatch: bool = False,
    batch_size: int = 1024,
    variational_cov: Literal["chol", "diag"] = "chol",
    variational_lr: float | None = None,
    kl_beta: float = 1.0,
    warm_start_points: int = 0,
) -> dict:
    """
    Train independent single-task RFF GPs on the S2 11-QoI TOA dataset.

    The ``minibatch`` / variational arguments are retired. Stochastic training
    now lives on the inducing-point SVGP path
    (``experiments_GP/S2_toa_SVGP.py``), which uses an ordinary RBF kernel
    rather than random features; the plumbing below is kept only so old call
    sites fail loudly instead of silently taking a different route.
    """
    if init_batch_size is None:
        init_batch_size = num_inits
    if int(init_batch_size) < 1:
        raise ValueError(f"init_batch_size must be >= 1, got {init_batch_size}.")
    init_batch_size = int(init_batch_size)
    if train_mode == "batched" and num_inits % init_batch_size != 0:
        raise ValueError(
            f"num_inits ({num_inits}) must be divisible by init_batch_size ({init_batch_size})."
        )
    if mean_type not in ("constant", "neural"):
        raise ValueError(f"mean_type must be 'constant' or 'neural', got {mean_type!r}")
    if mean_type == "neural" and train_mode == "batched":
        raise ValueError(
            "NeuralMean does not support train_mode='batched' (no batch_shape). "
            "Use train_mode='independent'."
        )
    if mean_type == "neural":
        if neural_mean_hidden is None:
            neural_mean_hidden = (64, 32)
        neural_mean_hidden = tuple(int(d) for d in neural_mean_hidden)
    if rff_sampling not in RFF_SAMPLING_CHOICES:
        raise ValueError(f"rff_sampling must be one of {RFF_SAMPLING_CHOICES}, got {rff_sampling!r}")
    if spectral_kernel not in SPECTRAL_KERNEL_CHOICES:
        raise ValueError(
            f"spectral_kernel must be one of {SPECTRAL_KERNEL_CHOICES}, got {spectral_kernel!r}"
        )
    if bound_penalty_lambda_learnable and float(bound_penalty_lambda) < float(bound_penalty_lam_min):
        raise ValueError(
            f"bound_penalty_lambda ({bound_penalty_lambda}) must be >= "
            f"bound_penalty_lam_min ({bound_penalty_lam_min}) when learnable."
        )
    minibatch = bool(minibatch)
    if minibatch:
        raise NotImplementedError(
            "Minibatch training was moved off the RFF/SORF path: random features "
            "were replaced by inducing-point SVGP. Run "
            "experiments_GP/S2_toa_SVGP.py instead."
        )
    correct_sorf = bool(correct_sorf) if rff_sampling == "sorf" else False

    names = list(task_names) if task_names is not None else list(S2_TASK_NAMES)
    band_cfg_path = task_band_config or str(S2_DEFAULT_BAND_CONFIG_PATH)
    bands_by_task = load_task_band_config(band_cfg_path, task_names=names, input_dim=S2_INPUT_DIM)
    n_components_by_task: dict[str, int] | None = None
    pca_components_meta: dict = {}
    pca_title_token: str | None = None
    if n_pca_components is not None:
        n_components_by_task, pca_components_meta = resolve_task_pca_components(
            n_pca_components, task_names=names
        )
        unique_ps = sorted(set(n_components_by_task.values()))
        pca_title_token = str(unique_ps[0]) if len(unique_ps) == 1 else "perQoI"

    if save_path is None:
        save_path = _DEFAULT_SAVE_DIRS[rff_sampling]

    set_seed(seed)
    if num_rff is None:
        num_rff = min(512, max(64, n_train // 3))

    if adam_stop_patience is None:
        adam_stop_patience = int(DEFAULT_TOA_ADAM_STOP_PATIENCE)
    else:
        adam_stop_patience = int(adam_stop_patience)
    if adam_stop_patience < 1:
        raise ValueError(f"adam_stop_patience must be >= 1, got {adam_stop_patience}.")

    if num_epochs <= 1:
        optimizer_class = LBFGSScipy
        default_optimizer_kwargs = DEFAULT_LBFGS_KWARGS
        stop_conditions = [
            ConvergencePatienceStopCondition(patience=adam_stop_patience),
            MinLossChangeStopCondition(min_loss_change=1e-7),
        ]
    else:
        optimizer_class = torch.optim.Adam
        default_optimizer_kwargs = {**DEFAULT_ADAM_KWARGS, "lr": DEFAULT_TOA_ADAM_LR}
        stop_conditions = [
            ConvergencePatienceStopCondition(patience=adam_stop_patience),
        ]
    if optimizer_kwargs is None:
        optimizer_kwargs = dict(default_optimizer_kwargs)

    title = f"S2_TOA_nTrain{n_train}_nTest{n_test}_{rff_sampling}D{num_rff}"
    if pca_title_token is not None:
        title = (
            f"S2_TOA_nTrain{n_train}_nTest{n_test}_pcaP{pca_title_token}_"
            f"{rff_sampling}D{num_rff}"
        )
    if spectral_kernel != "rbf":
        title = f"{title}_{spectral_kernel}"
    feature_dim = 2 * num_rff
    sampling_label = rff_sampling.upper()
    print("=" * 60)
    print(title)
    print(
        f"S2 Independent {sampling_label}-GP (Woodbury), D={num_rff}, m={feature_dim}, "
        f"spectral_kernel={spectral_kernel}, ARD={ard}, dtype={dtype}, inits={num_inits}, "
        f"init_batch_size={init_batch_size}, epochs={num_epochs}, tasks={names}, "
        f"input={input_variable}"
        + (f", correct_sorf={correct_sorf}" if rff_sampling == "sorf" else "")
        + (f", pca={n_components_by_task}" if n_components_by_task is not None else "")
        + (f", train_mode={train_mode}" if train_mode != "independent" else "")
    )
    opt_name = getattr(optimizer_class, "__name__", str(optimizer_class))
    print(f"Optimizer: {opt_name}, kwargs={optimizer_kwargs}")
    print(f"Device: {device}")
    print(f"Band config: {band_cfg_path}")
    if pca_components_meta.get("config_path"):
        print(f"PCA components config: {pca_components_meta['config_path']}")
    print("=" * 60)

    n_val = TOA_VAL_POOL_SIZE if monitor_validation else 0
    (
        x_train_full,
        y_train,
        x_val_full,
        y_val,
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
        n_val=n_val,
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
    bound_by_task = resolve_bound_penalty_for_tasks(
        names,
        bound_min,
        bound_max,
        warps=warps,
        k=bound_penalty_k,
        lam=bound_penalty_lambda,
        alpha=bound_penalty_alpha,
        max_points=bound_penalty_max_points,
    )
    bound_active = [n for n, cfg in bound_by_task.items() if cfg is not None]
    if bound_active:
        lam_note = (
            f"lambda=learnable(init={bound_penalty_lambda}, min={bound_penalty_lam_min})"
            if bound_penalty_lambda_learnable
            else f"lambda={bound_penalty_lambda}"
        )
        print(
            "Soft probabilistic bounds: "
            + ", ".join(
                f"{n}[{bound_by_task[n].a}, {bound_by_task[n].b}]"
                for n in bound_active
            )
            + f" (k={bound_penalty_k}, {lam_note}, "
            f"alpha={bound_penalty_alpha}, max_points={bound_penalty_max_points})"
        )
    else:
        print("Soft probabilistic bounds: off")
    if pac_bayes:
        print(
            f"PAC-Bayes MLL: on (temperature={pac_bayes_temperature}, "
            f"prior_std={pac_bayes_prior_std}, posterior_std={pac_bayes_posterior_std})"
        )
    if nigp:
        print(
            f"NIGP: on (independent input noise, "
            f"sigma_x=10^SoftClamp(raw), freeze_epochs={int(freeze_epoch_nigp)}, "
            f"slope_refreshes={nigp_slope_refreshes})"
        )
    else:
        print("NIGP: off")
    learned_lambda_by_task: dict[str, float | None] = {}
    wavelengths_np = wavelengths.detach().cpu().numpy()
    x_test_orig = x_test_full.clone()
    y_train = y_train.to(dtype=dtype)
    y_test = y_test.to(dtype=dtype)
    y_val = y_val.to(dtype=dtype)

    total_train_time = 0.0
    total_prediction_time = 0.0
    task_metrics: dict[str, dict] = {}
    task_runs: dict[str, list] = {}
    task_best_runs: dict[str, dict] = {}
    input_bands_by_task: dict[str, dict] = {}
    y_pred_all: list[np.ndarray] = []
    y_std_all: list[np.ndarray] = []
    lower_all: list[np.ndarray] = []
    upper_all: list[np.ndarray] = []
    y_pred_mean_all: list[np.ndarray] = []
    y_pred_mode_all: list[np.ndarray] = []
    log_mu_all: list[np.ndarray] = []
    log_sigma_all: list[np.ndarray] = []
    logit_mu_all: list[np.ndarray] = []
    logit_sigma_all: list[np.ndarray] = []
    rel_metrics_by_task: dict[str, dict[str, float | int]] = {}

    for task_idx, task_name in enumerate(names):
        print(f"\n--- Task: {task_name} ---")
        band_indices = bands_by_task[task_name]
        print(f"Selected bands: n={len(band_indices)} first={band_indices[:5]} ...")

        x_tr = select_bands(x_train_full, band_indices).to(dtype=dtype)
        x_te = select_bands(x_test_full, band_indices).to(dtype=dtype)
        x_va = (
            select_bands(x_val_full, band_indices).to(dtype=dtype)
            if x_val_full.numel() > 0
            else x_val_full.to(dtype=dtype)
        )
        x_tr = apply_x_transform(x_tr, x_transform)
        x_te = apply_x_transform(x_te, x_transform)
        if x_va.numel() > 0:
            x_va = apply_x_transform(x_va, x_transform)
        if x_transform and x_transform != "none" and task_idx == 0:
            print(f"X transform: {x_transform} (before PCA / scaling)")

        pca_meta = None
        input_dim_before_pca = int(x_tr.shape[-1])
        if n_components_by_task is not None:
            _pca_dir = _ROOT / "experiments_PCA"
            if str(_pca_dir) not in sys.path:
                sys.path.insert(0, str(_pca_dir))
            from toa_pca_utils import fit_pca_on_train, transform_pca

            task_n_components = int(n_components_by_task[task_name])
            pca_fit = fit_pca_on_train(
                x_tr,
                n_components=task_n_components,
                svd_solver=pca_svd_solver,
                random_state=seed,
            )
            x_tr = transform_pca(pca_fit, x_tr, dtype=dtype)
            x_te = transform_pca(pca_fit, x_te, dtype=dtype)
            if x_va.numel() > 0:
                x_va = transform_pca(pca_fit, x_va, dtype=dtype)
            pca_meta = pca_fit.to_dict()
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
            x_scaler.fit(x_tr)
            x_tr = x_scaler.transform(x_tr)
            x_te = x_scaler.transform(x_te)
            if x_va.numel() > 0:
                x_va = x_scaler.transform(x_va)
            print(f"X scaling ({task_name}): {x_scaling_type}")

        y_tr = y_train[:, task_idx]
        y_te = y_test[:, task_idx]
        y_va = y_val[:, task_idx] if y_val.numel() > 0 else y_val
        y_tr_model = forward_y_s2(y_tr, task_name, warps=warps)

        y_scaler = None
        if standardize_y:
            y_scaler = StandardScaler()
            y_scaler.fit(y_tr_model.unsqueeze(-1))
            y_tr_fit = y_scaler.transform(y_tr_model.unsqueeze(-1)).squeeze(-1)
        else:
            y_tr_fit = y_tr_model

        y_val_scaled = y_va
        if y_va.numel() > 0:
            y_va_model = forward_y_s2(y_va, task_name, warps=warps)
            if standardize_y and y_scaler is not None:
                y_val_scaled = y_scaler.transform(y_va_model.unsqueeze(-1)).squeeze(-1)
            else:
                y_val_scaled = y_va_model

        callbacks = []
        if nigp and int(freeze_epoch_nigp) > 0 and num_epochs > 1:
            from gpplus.training import NIGPInputNoiseFreezeCallback

            callbacks.append(
                NIGPInputNoiseFreezeCallback(
                    freeze_epochs=int(freeze_epoch_nigp),
                    verbose=training_verbose,
                )
            )
        if int(freeze_epoch_noise) > 0 and num_epochs > 1:
            from gpplus.training import LikelihoodNoiseFreezeCallback

            callbacks.append(
                LikelihoodNoiseFreezeCallback(
                    freeze_epochs=int(freeze_epoch_noise),
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
        if monitor_validation and n_val > 0:
            callbacks.append(
                make_validation_callback(
                    x_va,
                    y_val_scaled,
                    num_inits,
                    chunk_size=predict_chunk_size,
                    verbose=validation_verbose,
                    log_every_n_epochs=log_every_n_epochs,
                    woodbury_form=DEFAULT_WOODBURY_FORM,
                )
            )

        likelihood = None
        initializer_kwargs: dict | None = None
        noise_prior_meta: dict | None = None
        override_pcs = dict(initializer_parameter_configs or {})
        concurrent = init_batch_size if train_mode == "batched" else num_inits
        lik_batch_shape = (
            torch.Size([concurrent])
            if train_mode == "batched" and concurrent > 1
            else torch.Size([])
        )
        if response_noise_prior:
            from gpplus.priors.response_noise import (
                empirical_scalar_noise_variance,
                log_normal_scalar_noise_prior_from_responses,
                scalar_noise_raw_init_from_variance,
            )

            target_var = empirical_scalar_noise_variance(y_tr_fit, fraction=noise_var_fraction)
            noise_prior = log_normal_scalar_noise_prior_from_responses(
                y_tr_fit,
                fraction=noise_var_fraction,
                log_scale=noise_prior_log_scale,
                dtype=dtype,
                device=y_tr_fit.device,
            )
            likelihood = build_rff_scalar_noise_likelihood(
                noise_prior=noise_prior, batch_shape=lik_batch_shape
            )
            raw_init = scalar_noise_raw_init_from_variance(
                likelihood, target_var.to(dtype=dtype)
            )
            # Constant-from-prior noise init unless the caller overrides raw_noise.
            pcs: dict = {}
            if "raw_noise" not in override_pcs:
                pcs["raw_noise"] = {"method": "constant", "value": raw_init}
            initializer_kwargs = {"parameter_configs": pcs}
            noise_prior_meta = {
                "noise_prior_target_var": float(target_var.detach().cpu()),
                "noise_prior_loc": float(noise_prior.loc.detach().cpu()),
            }
        else:
            likelihood = build_rff_scalar_noise_likelihood(batch_shape=lik_batch_shape)

        initializer_kwargs = merge_rff_noise_initializer_kwargs(initializer_kwargs)
        if override_pcs:
            pcs_merged = dict(initializer_kwargs.get("parameter_configs") or {})
            pcs_merged.update(override_pcs)
            initializer_kwargs = {**initializer_kwargs, "parameter_configs": pcs_merged}
        mean_module = None
        if mean_type == "neural":
            mean_module = NeuralMean.from_hidden(
                int(x_tr.shape[-1]),
                neural_mean_hidden,
                activation=neural_mean_activation,
            )
        model_kwargs = dict(
            likelihood=likelihood,
            mean_module=mean_module,
            num_rff=num_rff,
            ard=ard,
            rff_sampling=rff_sampling,
            correct_sorf=correct_sorf,
            spectral_kernel=spectral_kernel,
            nigp=bool(nigp),
        )
        model = RFFGPR(
            x_tr,
            y_tr_fit,
            batch_shape=lik_batch_shape if len(lik_batch_shape) > 0 else None,
            init_batch_size=concurrent if train_mode == "batched" else None,
            **model_kwargs,
        )
        bound_cfg = bound_by_task.get(task_name)
        task_mll_class = None
        nigp_mll_kwargs: dict = {}
        if nigp:
            slope_r = (
                None
                if nigp_slope_refreshes is None
                else int(nigp_slope_refreshes)
            )
            if slope_r is not None:
                nigp_mll_kwargs["nigp_slope_refreshes"] = slope_r
                nigp_mll_kwargs["nigp_active_epochs"] = max(
                    1, int(num_epochs) - max(0, int(freeze_epoch_nigp))
                )
            nigp_base_mll = nigp_woodbury_mll_class(
                DEFAULT_WOODBURY_FORM, **nigp_mll_kwargs
            )
        else:
            nigp_base_mll = None
        if bound_cfg is not None and bound_penalty_lambda_learnable:
            model.register_learnable_bound_penalty_lambda(
                lam_init=bound_penalty_lambda,
                lam_min=bound_penalty_lam_min,
            )
        if bound_cfg is not None:
            y_mean_b = float(y_scaler.mean.squeeze()) if y_scaler is not None else 0.0
            y_std_b = float(y_scaler.std.squeeze()) if y_scaler is not None else 1.0
            task_mll_class = bound_penalized_rff_mll_class(
                bound_cfg,
                y_mean=y_mean_b,
                y_std=y_std_b,
                woodbury_form=DEFAULT_WOODBURY_FORM,
                base_mll_class=nigp_base_mll,
            )
        elif nigp:
            task_mll_class = nigp_base_mll
        else:
            task_mll_class = rff_mll_class()
        if pac_bayes:
            task_mll_class = pac_bayes_mll_class(
                task_mll_class,
                temperature=pac_bayes_temperature,
                prior_std=pac_bayes_prior_std,
                posterior_std=pac_bayes_posterior_std,
            )

        shared_trainer_kwargs = dict(
            num_epochs=num_epochs,
            num_inits=num_inits,
            seed=seed,
            device=device,
            dtype=dtype,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            initializer_class=RFFParameterInitializer,
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
        trainer = GPTrainer(
            model,
            mll_class=task_mll_class,
            param_groups_fn=param_groups_fn,
            train_mode=train_mode,
            init_batch_size=concurrent if train_mode == "batched" else None,
            **shared_trainer_kwargs,
        )
        t_train = time.time()
        runs = trainer.train()
        train_time = time.time() - t_train
        total_train_time += train_time
        # Batched mode replaces trainer.model with an unbatched winner clone.
        model = trainer.model

        successful = [r for r in runs if r.get("loss") is not None and r.get("state_dict") is not None]
        if not successful:
            errors = [r.get("error", "unknown") for r in runs if r.get("error")]
            raise RuntimeError(
                f"All training runs failed for {task_name}. "
                + (f"First error: {errors[0]}" if errors else "Check optimizer kwargs.")
            )
        best_run = min(successful, key=lambda r: r["loss"])
        model.load_state_dict(best_run["state_dict"])
        best_loss = float(best_run["loss"])
        if best_run.get("aborted"):
            print(
                f"WARNING: {task_name} training aborted mid-run; using salvaged best "
                f"epoch (loss={best_loss:.6f}"
                + (
                    f", abort_epoch={int(best_run['aborted_epoch']) + 1}"
                    if best_run.get("aborted_epoch") is not None
                    else ""
                )
                + f"). Error: {best_run.get('error', 'unknown')}"
            )
        final_bound_lam = learned_bound_penalty_lambda(model)
        if final_bound_lam is not None:
            learned_lambda_by_task[task_name] = final_bound_lam
            print(f"{task_name} learned bound penalty lambda: {final_bound_lam:.6f}")
        y_std_for_noise = y_scaler.std.squeeze() if y_scaler is not None else None
        learned_noise = extract_learned_likelihood_noise(model, y_std=y_std_for_noise)
        ard_info = extract_ard_lengthscales(model)
        ard_space = "pca_components" if n_components_by_task is not None else "bands"
        ard_mapped = map_ard_to_bands(
            ard_info["lengthscale"],
            band_indices,
            wavelengths_nm=wavelengths_np,
            ard_space=ard_space,
        )
        if ard_info["outputscale"] is not None:
            ard_mapped["outputscale"] = ard_info["outputscale"]
        ard_mapped["raw_lengthscale"] = ard_info["raw_lengthscale"]
        ard_mapped["model_input_dim"] = int(x_tr.shape[-1])
        if pca_meta is not None:
            ard_mapped["pca"] = pca_meta
        nigp_info = extract_nigp_input_noise(model)
        nigp_mapped = None
        if nigp_info is not None:
            nigp_mapped = map_input_noise_to_bands(
                nigp_info["input_noise"],
                band_indices,
                input_noise_var=nigp_info["input_noise_var"],
                wavelengths_nm=wavelengths_np,
                noise_space=ard_space,
            )
            nigp_mapped["raw_input_noise"] = nigp_info["raw_input_noise"]
            try:
                nigp_grad = compute_nigp_grad_diagnostics(
                    model,
                    x_tr,
                    band_indices=band_indices,
                    wavelengths_nm=wavelengths_np,
                    noise_space=ard_space,
                )
            except Exception as exc:
                print(f"{task_name} NIGP grad diagnostics failed: {exc}")
                nigp_grad = None
            if nigp_grad is not None:
                for key in (
                    "grad_mu_abs_mean",
                    "grad_mu_abs_median",
                    "grad_mu_sq_mean",
                    "input_noise_contrib_mean",
                    "input_noise_contrib_frac",
                    "effective_input_term_mean",
                    "n_points_total",
                    "n_points_used",
                    "max_points",
                    "jitter",
                ):
                    if key in nigp_grad:
                        nigp_mapped[key] = nigp_grad[key]
                nigp_mapped["entries"] = nigp_grad["entries"]
        input_bands_by_task[task_name] = {
            "indices": list(band_indices),
            "wavelength_nm": [float(wavelengths_np[i]) for i in band_indices],
            "model_input_dim": int(x_tr.shape[-1]),
            "x_scaling_type": x_scaling_type,
            **ard_mapped,
        }
        if nigp_mapped is not None:
            input_bands_by_task[task_name]["nigp"] = nigp_mapped
            input_bands_by_task[task_name]["input_noise"] = nigp_mapped["input_noise"]
            input_bands_by_task[task_name]["input_noise_var"] = nigp_mapped[
                "input_noise_var"
            ]
            input_bands_by_task[task_name]["raw_input_noise"] = nigp_mapped[
                "raw_input_noise"
            ]

        if save_checkpoint and save_path:
            try:
                ckpt_path = save_toa_stgp_checkpoint(
                    checkpoint_path_for_run(save_path, title, task_name),
                    model=model,
                    task_name=task_name,
                    train_x=x_tr.cpu(),
                    train_y=y_tr_fit.cpu(),
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
                    n_val=n_val,
                    data_path=data_path,
                    rel_tolerance=rel_tolerance,
                    dtype=dtype,
                    log_grain=task_uses_log_scale(task_name, warps=warps),
                    logit_cos=task_uses_logit_scale(task_name, warps=warps),
                    input_column_indices=torch.as_tensor(band_indices, dtype=torch.int64),
                    model_config={
                        "num_rff": num_rff,
                        "ard": ard,
                        "rff_sampling": rff_sampling,
                        "correct_sorf": correct_sorf,
                        "spectral_kernel": spectral_kernel,
                        "n_pca_components": (
                            int(n_components_by_task[task_name])
                            if n_components_by_task is not None
                            else None
                        ),
                        "input_variable": input_variable,
                        "x_transform": x_transform or "none",
                        "dataset": "s2",
                        "band_indices": list(band_indices),
                        "ard_mapping": ard_mapped,
                        "nigp": bool(nigp),
                        "freeze_epoch_nigp": int(freeze_epoch_nigp),
                        "nigp_slope_refreshes": (
                            None
                            if nigp_slope_refreshes is None
                            else int(nigp_slope_refreshes)
                        ),
                        "nigp_mapping": nigp_mapped,
                        "data_meta": data_meta,
                        "log_scale": task_uses_log_scale(task_name, warps=warps),
                        "logit_scale": task_uses_logit_scale(task_name, warps=warps),
                        "log_scale_source": log_scale_source,
                        "log_scale_tasks": sorted(log_scale_task_set),
                        "logit_scale_tasks": sorted(logit_scale_task_set),
                        "logit_bounds": {
                            k: list(v)
                            for k, v in warps.logit_bounds.items()
                            if k in logit_scale_task_set
                        },
                    },
                )
                learned_noise["checkpoint_path"] = str(ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")
            except (OSError, RuntimeError) as exc:
                print(
                    f"WARNING: checkpoint save failed ({exc}); continuing without checkpoint."
                )

        model.eval()
        model.invalidate_feature_cache()
        t_pred = time.time()
        pred_mean, lower, upper, pred_std = evaluate_rff_gp_model(
            model, x_te, chunk_size=predict_chunk_size, **rff_eval_kwargs(dtype)
        )
        prediction_time = time.time() - t_pred
        total_prediction_time += prediction_time

        # Inverse Y standardization (+ log-/logit-normal for warped QoIs).
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
        pred_mean, pred_std, lower, upper = inv.as_tuple()

        computed = compute_metrics(
            y_te.cpu(),
            pred_mean,
            output_std=pred_std,
            lower_95=lower,
            upper_95=upper,
            training_time=train_time,
            prediction_time=prediction_time,
        )
        if task_uses_log_scale(task_name, warps=warps):
            computed["log_scale"] = True
            if inv.log_mu is None:
                raise RuntimeError(f"log_mu missing after inverse for log-scale task {task_name}")
            extra = compute_log_scale_extra_metrics(
                y_te.cpu(),
                log_mu=inv.log_mu,
                point_mean_physical=inv.point_mean,
                log_offset=warps.log_offset(task_name),
            )
            computed.update(extra)
            c = warps.log_offset(task_name)
            if c > 0.0:
                computed["log_offset"] = float(c)
        elif task_uses_logit_scale(task_name, warps=warps):
            computed["logit_scale"] = True
        tm: dict = {
            "best_train_loss": best_loss,
            **learned_noise,
            **{k: v for k, v in computed.items()},
            "n_bands": len(band_indices),
            "model_input_dim": int(x_tr.shape[-1]),
        }
        if best_run.get("aborted"):
            tm["training_aborted"] = True
            tm["abort_error"] = str(best_run.get("error", "unknown"))
            if best_run.get("aborted_epoch") is not None:
                tm["aborted_epoch"] = int(best_run["aborted_epoch"]) + 1
        if nigp_info is not None:
            tm["input_noise"] = list(nigp_info["input_noise"])
            tm["input_noise_var"] = list(nigp_info["input_noise_var"])
            tm["raw_input_noise"] = list(nigp_info["raw_input_noise"])
            if nigp_mapped is not None:
                tm["nigp_band_mapping"] = nigp_mapped
                for key in (
                    "grad_mu_abs_mean",
                    "grad_mu_abs_median",
                    "grad_mu_sq_mean",
                    "input_noise_contrib_mean",
                    "input_noise_contrib_frac",
                    "effective_input_term_mean",
                ):
                    if key in nigp_mapped:
                        tm[key] = nigp_mapped[key]
        if best_run.get("final_lr") is not None:
            tm["final_lr"] = float(best_run["final_lr"])
        if noise_prior_meta is not None:
            tm.update(noise_prior_meta)
        if final_bound_lam is not None:
            tm["bound_penalty_lambda_final"] = final_bound_lam
            tm["bound_penalty_lambda_init"] = float(bound_penalty_lambda)
            tm["bound_penalty_lam_min"] = float(bound_penalty_lam_min)
        task_metrics[task_name] = tm
        task_runs[task_name] = runs
        task_best_runs[task_name] = best_run
        y_pred_all.append(pred_mean.numpy())
        y_std_all.append(pred_std.numpy())
        lower_all.append(lower.numpy())
        upper_all.append(upper.numpy())
        if inv.point_mean is not None:
            y_pred_mean_all.append(inv.point_mean.numpy())
        else:
            y_pred_mean_all.append(pred_mean.numpy())
        if inv.point_mode is not None:
            y_pred_mode_all.append(inv.point_mode.numpy())
        else:
            y_pred_mode_all.append(np.full_like(pred_mean.numpy(), np.nan))
        if inv.log_mu is not None:
            log_mu_all.append(inv.log_mu.numpy())
        else:
            log_mu_all.append(np.full_like(pred_mean.numpy(), np.nan))
        if inv.log_sigma is not None:
            log_sigma_all.append(inv.log_sigma.numpy())
        else:
            log_sigma_all.append(np.full_like(pred_mean.numpy(), np.nan))
        if inv.logit_mu is not None:
            logit_mu_all.append(inv.logit_mu.numpy())
        else:
            logit_mu_all.append(np.full_like(pred_mean.numpy(), np.nan))
        if inv.logit_sigma is not None:
            logit_sigma_all.append(inv.logit_sigma.numpy())
        else:
            logit_sigma_all.append(np.full_like(pred_mean.numpy(), np.nan))
        print_task_test_metrics(
            task_name,
            computed,
            warps=warps,
        )

    y_pred_stacked = np.stack(y_pred_all, axis=1)
    y_std_stacked = np.stack(y_std_all, axis=1)
    lower_stacked = np.stack(lower_all, axis=1)
    upper_stacked = np.stack(upper_all, axis=1)
    y_pred_mean_stacked = np.stack(y_pred_mean_all, axis=1)
    y_pred_mode_stacked = np.stack(y_pred_mode_all, axis=1)
    log_mu_stacked = np.stack(log_mu_all, axis=1)
    log_sigma_stacked = np.stack(log_sigma_all, axis=1)
    logit_mu_stacked = np.stack(logit_mu_all, axis=1)
    logit_sigma_stacked = np.stack(logit_sigma_all, axis=1)
    y_test_np = y_test.detach().cpu().numpy()
    per_task = compute_per_task_metrics(y_test_np, y_pred_stacked, names)

    for t, name in enumerate(names):
        rel_m = compute_relative_error_metrics(
            y_test_np[:, t],
            y_pred_stacked[:, t],
            rel_tolerance=rel_tolerance,
        )
        rel_metrics_by_task[name] = rel_m
        per_task[f"{name}_max_rel_error"] = float(rel_m["max_rel_error"])
        per_task[f"{name}_mean_rel_error"] = float(rel_m["mean_rel_error"])
        per_task[f"{name}_median_rel_error"] = float(rel_m["median_rel_error"])
        per_task[f"{name}_pct_within_1pct"] = float(rel_m["pct_within_1pct"])
        per_task[f"{name}_n_rel_error_valid"] = int(rel_m["n_rel_error_valid"])
        per_task[f"{name}_n_rel_error_excluded"] = int(rel_m["n_rel_error_excluded"])

    aggregate_rmse = float(np.sqrt(np.mean((y_pred_stacked - y_test_np) ** 2)))
    aggregate_rrmse = macro_rrmse(per_task, names)
    aggregate_medae = macro_metric(per_task, names, "MedAE")

    # Flatten per-task log extras into per_task for aggregates / JSON.
    log_task_names = [n for n in names if task_uses_log_scale(n, warps=warps)]
    logit_task_names = [n for n in names if task_uses_logit_scale(n, warps=warps)]
    for name in log_task_names:
        tm = task_metrics[name]
        for key in (
            "RMSE_log",
            "MAE_log",
            "MedAE_log",
            "RRMSE_log",
            "R2_log",
            "RMSE_mean",
            "MAE_mean",
            "MedAE_mean",
            "RRMSE_mean",
            "R2_mean",
        ):
            if key in tm:
                per_task[f"{name}_{key}"] = float(tm[key])
    aggregate_rrmse_log = macro_metric(per_task, log_task_names, "RRMSE_log")
    aggregate_rrmse_mean = macro_metric(per_task, log_task_names, "RRMSE_mean")

    metrics: dict = {
        "title": title,
        "dataset": "s2",
        "input_dim_original": S2_INPUT_DIM,
        "n_train": n_train,
        "n_test": n_test,
        "num_tasks": len(names),
        "task_names": list(names),
        "num_rff": num_rff,
        "rff_sampling": rff_sampling,
        "correct_sorf": correct_sorf,
        "spectral_kernel": spectral_kernel,
        "feature_dim": feature_dim,
        "ard": ard,
        "model_class": "RFFGPR",
        "train_mode": train_mode,
        "num_inits": num_inits,
        "init_batch_size": init_batch_size if train_mode == "batched" else num_inits,
        "num_batches": (
            num_inits // init_batch_size if train_mode == "batched" else 1
        ),
        "num_epochs": num_epochs,
        "optimizer": getattr(optimizer_class, "__name__", str(optimizer_class)),
        "optimizer_kwargs": json_safe_optimizer_kwargs(optimizer_kwargs),
        "initial_lr": float(optimizer_kwargs.get("lr")) if "lr" in optimizer_kwargs else None,
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
        "bound_penalty_k": float(bound_penalty_k),
        "bound_penalty_lambda": float(bound_penalty_lambda),
        "bound_penalty_alpha": float(bound_penalty_alpha),
        "bound_penalty_max_points": (
            None if bound_penalty_max_points is None else int(bound_penalty_max_points)
        ),
        "bound_penalty_lambda_learnable": bool(bound_penalty_lambda_learnable),
        "bound_penalty_lam_min": float(bound_penalty_lam_min),
        **bound_penalty_metrics(
            bound_by_task,
            bound_penalty_lambda_learnable=bound_penalty_lambda_learnable,
            bound_penalty_lam_min=bound_penalty_lam_min,
            learned_lambda_by_task=learned_lambda_by_task,
        ),
        "pac_bayes": bool(pac_bayes),
        "pac_bayes_temperature": float(pac_bayes_temperature),
        "pac_bayes_prior_std": float(pac_bayes_prior_std),
        "pac_bayes_posterior_std": float(pac_bayes_posterior_std),
        "nigp": bool(nigp),
        "freeze_epoch_nigp": int(freeze_epoch_nigp),
        "nigp_slope_refreshes": (
            None if nigp_slope_refreshes is None else int(nigp_slope_refreshes)
        ),
        "adam_stop_patience": int(adam_stop_patience),
        "mean_type": mean_type,
        "neural_mean_hidden": (
            list(neural_mean_hidden) if mean_type == "neural" and neural_mean_hidden is not None else None
        ),
        "neural_mean_activation": (
            str(neural_mean_activation) if mean_type == "neural" else None
        ),
        "response_noise_prior": bool(response_noise_prior),
        "noise_var_fraction": float(noise_var_fraction),
        "noise_prior_log_scale": float(noise_prior_log_scale),
        "initializer_parameter_configs": dict(initializer_parameter_configs or {}),
        "rel_tolerance": rel_tolerance,
        "task_band_config": str(band_cfg_path),
        "bands_by_task": band_config_metadata(bands_by_task, wavelengths_nm=wavelengths_np),
        "input_bands_by_task": input_bands_by_task,
        "data_meta": data_meta,
        "Training_Time": total_train_time,
        "Prediction_Time": total_prediction_time,
        "Total_Time": total_train_time + total_prediction_time,
        "aggregate_RRMSE": aggregate_rrmse,
        "aggregate_RRMSE_mean": aggregate_rrmse,
        "aggregate_MedAE": aggregate_medae,
        "aggregate_RRMSE_log_tasks": aggregate_rrmse_log,
        "aggregate_RRMSE_lognormal_mean_tasks": aggregate_rrmse_mean,
        "RMSE": aggregate_rmse,
        **per_task,
    }
    if n_components_by_task is not None:
        unique_ps = sorted(set(n_components_by_task.values()))
        metrics["n_pca_components"] = (
            n_components_by_task if len(unique_ps) > 1 else unique_ps[0]
        )
        metrics["n_components_by_task"] = dict(n_components_by_task)
        metrics["pca_components_meta"] = pca_components_meta
        metrics["pca_svd_solver"] = pca_svd_solver

    for task_name in names:
        tm = task_metrics[task_name]
        metrics[f"{task_name}_best_train_loss"] = tm["best_train_loss"]
        metrics[f"{task_name}_raw_noise"] = tm["raw_noise"]
        metrics[f"{task_name}_noise"] = tm["noise"]
        metrics[f"{task_name}_noise_std"] = tm["noise_std"]
        if "checkpoint_path" in tm:
            metrics[f"{task_name}_checkpoint_path"] = tm["checkpoint_path"]
        for key, value in tm.items():
            if key in ("best_train_loss", "raw_noise", "noise", "noise_std", "checkpoint_path"):
                continue
            metrics[f"{task_name}_{key}"] = value

    if monitor_validation and n_val > 0:
        metrics["monitor_validation"] = True
        metrics["n_val"] = n_val
        metrics["train_pool_size"] = int(data_meta.get("train_pool_size", TOA_TRAIN_POOL_SIZE))
        metrics["val_pool_size"] = int(data_meta.get("val_pool_size", TOA_VAL_POOL_SIZE))
        metrics["test_pool_size"] = int(data_meta.get("test_pool_size", TOA_TEST_POOL_SIZE))
        for task_name in names:
            val_summary = summarize_validation_from_runs(
                task_runs[task_name], task_best_runs[task_name]
            )
            for key, value in val_summary.items():
                metrics[f"{task_name}_{key}"] = value

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
        example_indices = select_posterior_example_indices(
            y_test_np.shape[0],
            posterior_n_examples,
            seed=seed,
            explicit_indices=posterior_example_indices,
        )
        log_task_names_plot = [n for n in names if task_uses_log_scale(n, warps=warps)]
        logit_task_names_plot = [
            n for n in names if task_uses_logit_scale(n, warps=warps)
        ]
        warped_plot = bool(log_task_names_plot or logit_task_names_plot)
        out_npz = save_s2_predictions_npz(
            save_path,
            title=title,
            task_names=names,
            y_true=y_test_np,
            y_pred=y_pred_stacked,
            y_std=y_std_stacked,
            lower=lower_stacked,
            upper=upper_stacked,
            x_test=x_test_orig.numpy(),
            wavelengths_nm=wavelengths_np,
            train_idx=train_idx.cpu().numpy(),
            val_idx=val_idx.cpu().numpy(),
            test_idx=test_idx.cpu().numpy(),
            bands_by_task=bands_by_task,
            y_pred_mean=y_pred_mean_stacked if warped_plot else None,
            y_pred_mode=y_pred_mode_stacked if log_task_names_plot else None,
            log_mu=log_mu_stacked if log_task_names_plot else None,
            log_sigma=log_sigma_stacked if log_task_names_plot else None,
            log_scale_tasks=log_task_names_plot,
            logit_mu=logit_mu_stacked if logit_task_names_plot else None,
            logit_sigma=logit_sigma_stacked if logit_task_names_plot else None,
            logit_scale_tasks=logit_task_names_plot,
        )
        print(f"Saved predictions to {out_npz}")
        metrics["predictions_npz"] = str(out_npz)

        out_json = save_metrics_json(metrics, save_path, title)
        print(f"Saved metrics to {out_json}")

        if plot_validation and monitor_validation and n_val > 0:
            for task_name in names:
                for plot_path in plot_per_task_validation_curves(
                    metrics, save_path, task_name, json_path=out_json
                ):
                    print(f"Saved validation plot to {plot_path}")

        if plot_posterior:
            scatter_dir = Path(save_path) / "plots" / "scatter" / title
            for t, name in enumerate(names):
                plot_s2_task_scatter(
                    y_true=y_test_np[:, t],
                    y_pred=y_pred_stacked[:, t],
                    lower=lower_stacked[:, t],
                    upper=upper_stacked[:, t],
                    task_name=name,
                    out_path=scatter_dir / f"{name}_scatter.png",
                    title=f"{title} | {name}",
                )
            post_dir = Path(save_path) / "plots" / "posterior" / title
            spectrum_ylabel = (
                "Radiance"
                if str(data_meta.get("input_variable", "")).endswith("radiance")
                else "Reflectance"
            )
            post_paths = plot_s2_posterior_examples(
                x_test=x_test_orig.numpy(),
                wavelengths_nm=wavelengths_np,
                y_true=y_test_np,
                y_pred=y_pred_stacked,
                lower=lower_stacked,
                upper=upper_stacked,
                task_names=names,
                example_indices=example_indices,
                save_dir=post_dir,
                title=title,
                y_std=y_std_stacked,
                rel_metrics_by_task=rel_metrics_by_task,
                rel_tolerance=rel_tolerance,
                log_scale_tasks=log_task_names_plot,
                y_pred_mean=y_pred_mean_stacked if warped_plot else None,
                y_pred_mode=y_pred_mode_stacked if log_task_names_plot else None,
                log_mu=log_mu_stacked if log_task_names_plot else None,
                log_sigma=log_sigma_stacked if log_task_names_plot else None,
                spectrum_ylabel=spectrum_ylabel,
            )
            for p in post_paths[:3]:
                print(f"Saved posterior plot to {p}")

    return metrics
