"""S2 independent GPR runner with per-QoI original-band subsets.

Two inference paths share this pipeline: dense exact GPR (default) and
inducing-point SVGP trained with minibatch SGD (``svgp=True``), which is the
only one that scales past a few thousand training points. Everything else --
band selection, PCA, output warps, metrics, plots -- is identical.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
_GP_DIR = Path(__file__).resolve().parent
_MTGPR_DIR = _ROOT / "experiments_RFFMTGPR"
_RFF_DIR = _ROOT / "experiments_RFF"
_DEFAULT_SAVE_DIR = "experiments_GP/results/s2_toa_gp"
_DEFAULT_SVGP_SAVE_DIR = "experiments_GP/results/s2_toa_svgp"
_EXACT_GP_N_TRAIN_WARN = 5000

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments_toa.paths import pin_toa_import_paths

pin_toa_import_paths(_MTGPR_DIR, _RFF_DIR, _GP_DIR)

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
    forward_y_s2,
    inverse_y_s2,
    resolve_y_warps,
    task_uses_log_scale,
    task_uses_logit_scale,
)
from gpplus.models import SVGPR
from gpplus.training import (
    ConvergencePatienceStopCondition,
    GPTrainer,
    MinibatchGPTrainer,
    MinLossChangeStopCondition,
    NIGPInputNoiseFreezeCallback,
    evaluate_gp_model,
    evaluate_svgp_gp_model,
    pac_bayes_mll_class,
)
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
from mtgpr_experiment_utils import (
    compute_relative_error_metrics,
    format_relative_error_summary,
    make_train_loss_callback,
    make_validation_callback,
    summarize_validation_from_runs,
)


def run_s2_toa_gp(
    n_train: int = 1600,
    n_test: int = 5000,
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
    response_noise_prior: bool = False,
    noise_var_fraction: float = 0.01,
    noise_prior_log_scale: float = 0.5,
    n_pca_components: NComponentsSpec | None = None,
    pca_svd_solver: str = "randomized",
    input_variable: str = "toa_reflectance",
    task_names: Sequence[str] | None = None,
    task_band_config: str | None = None,
    x_transform: str | None = None,
    pac_bayes: bool = False,
    pac_bayes_temperature: float = 1.0,
    pac_bayes_prior_std: float = 1.0,
    pac_bayes_posterior_std: float = 0.1,
    svgp: bool = False,
    num_inducing: int = 512,
    learn_inducing_locations: bool = True,
    batch_size: int = 1024,
    variational_lr: float | None = None,
    kl_beta: float = 1.0,
    nigp: bool = False,
    freeze_epoch_nigp: int = 100,
    adam_stop_patience: int = 50,
    save_checkpoint: bool = False,
) -> dict:
    """
    Train independent GPR models on the S2 11-QoI dataset.

    With ``svgp=True`` each task is an :class:`~gpplus.models.svgp_gpr.SVGPR`
    trained by minibatch SGD on the ELBO; otherwise it is a dense exact GPR.
    """
    if save_path is None:
        save_path = _DEFAULT_SVGP_SAVE_DIR if svgp else _DEFAULT_SAVE_DIR

    if svgp:
        if nigp and n_pca_components is not None:
            raise ValueError(
                "SVGP supports NIGP or PCA, not both: NIGP learns a per-input-"
                "dimension sigma_x, which has no interpretation in a rotated, "
                "truncated PCA basis. Set nigp=False or n_pca_components=None."
            )
        if num_epochs <= 1:
            raise ValueError(
                f"SVGP needs many SGD epochs, got num_epochs={num_epochs}."
            )
        if pac_bayes:
            raise ValueError("pac_bayes is not supported on the SVGP path.")
    elif nigp:
        raise ValueError("nigp on the exact-GP path is not wired up here; use svgp=True.")

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

    set_seed(seed)
    if num_epochs <= 1:
        optimizer_class = LBFGSScipy
        default_optimizer_kwargs = DEFAULT_LBFGS_KWARGS
        stop_conditions = [
            ConvergencePatienceStopCondition(patience=10),
            MinLossChangeStopCondition(min_loss_change=1e-7),
        ]
    else:
        optimizer_class = torch.optim.Adam
        default_optimizer_kwargs = dict(DEFAULT_ADAM_KWARGS)
        stop_conditions = [
            ConvergencePatienceStopCondition(patience=int(adam_stop_patience))
        ]
    if optimizer_kwargs is None:
        optimizer_kwargs = dict(default_optimizer_kwargs)

    model_token = f"svgpM{num_inducing}" if svgp else "exactGP"
    title = f"S2_TOA_nTrain{n_train}_nTest{n_test}_{model_token}"
    if pca_title_token is not None:
        title = (
            f"S2_TOA_nTrain{n_train}_nTest{n_test}_pcaP{pca_title_token}_{model_token}"
        )
    print("=" * 60)
    print(title)
    print(
        f"S2 {'SVGP' if svgp else 'Exact GP'}, ARD={ard}, dtype={dtype}, "
        f"inits={num_inits}, epochs={num_epochs}, tasks={names}, input={input_variable}"
        + (f", pca={n_components_by_task}" if n_components_by_task is not None else "")
    )
    if svgp:
        print(
            f"SVGP: M={num_inducing} (learned={learn_inducing_locations})  "
            f"batch_size={batch_size}  kl_beta={kl_beta}  "
            f"variational_lr={variational_lr}  nigp={nigp}"
            + (f"  freeze_epoch_nigp={freeze_epoch_nigp}" if nigp else "")
        )
    print(f"Band config: {band_cfg_path}")
    if pca_components_meta.get("config_path"):
        print(f"PCA components config: {pca_components_meta['config_path']}")
    if not svgp and n_train > _EXACT_GP_N_TRAIN_WARN:
        print(f"WARNING: exact GP is O(n^3); n_train={n_train} may be slow/OOM.")
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

    for task_idx, task_name in enumerate(names):
        print(f"\n--- Task: {task_name} ---")
        band_indices = bands_by_task[task_name]
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
                raise ValueError(f"x_standardize_method must be 0, 1, or 2, got {x_standardize_method}")
            x_scaler.fit(x_tr)
            x_tr = x_scaler.transform(x_tr)
            x_te = x_scaler.transform(x_te)
            if x_va.numel() > 0:
                x_va = x_scaler.transform(x_va)

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
            y_val_scaled = (
                y_scaler.transform(y_va_model.unsqueeze(-1)).squeeze(-1)
                if standardize_y and y_scaler is not None
                else y_va_model
            )

        callbacks = []
        if svgp and nigp and freeze_epoch_nigp > 0:
            callbacks.append(
                NIGPInputNoiseFreezeCallback(
                    freeze_epochs=freeze_epoch_nigp, verbose=training_verbose
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
                )
            )

        likelihood = None
        initializer_kwargs = None
        noise_prior_meta = None
        if response_noise_prior:
            from gpplus.priors.response_noise import (
                build_scalar_noise_likelihood,
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
            likelihood = build_scalar_noise_likelihood(noise_prior=noise_prior)
            raw_init = scalar_noise_raw_init_from_variance(likelihood, target_var.to(dtype=dtype))
            initializer_kwargs = {
                "parameter_configs": {"raw_noise": {"method": "constant", "value": raw_init}}
            }
            noise_prior_meta = {
                "noise_prior_target_var": float(target_var.detach().cpu()),
                "noise_prior_loc": float(noise_prior.loc.detach().cpu()),
            }

        if svgp:
            kernel_dims = int(x_tr.shape[-1]) if ard else None
            from gpplus import kernels as _kernels

            model = SVGPR(
                x_tr,
                y_tr_fit,
                kernel_module=_kernels.LogScaleKernel(
                    _kernels.GaussianKernel(ard_num_dims=kernel_dims)
                ),
                likelihood=likelihood,
                num_inducing=num_inducing,
                learn_inducing_locations=learn_inducing_locations,
                nigp=nigp,
                seed=seed,
            )
        else:
            model = build_gpr_model(x_tr, y_tr_fit, ard=ard)
            if likelihood is not None:
                model.likelihood = likelihood

        task_mll_class = None
        if pac_bayes:
            import gpytorch

            task_mll_class = pac_bayes_mll_class(
                gpytorch.mlls.ExactMarginalLogLikelihood,
                temperature=pac_bayes_temperature,
                prior_std=pac_bayes_prior_std,
                posterior_std=pac_bayes_posterior_std,
            )

        trainer_kwargs = dict(
            num_epochs=num_epochs,
            num_inits=num_inits,
            seed=seed,
            device=device,
            dtype=dtype,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            initializer_kwargs=initializer_kwargs,
            n_jobs=n_jobs,
            callbacks=callbacks,
            stop_conditions=stop_conditions,
            parallel_verbose=parallel_verbose,
        )
        if svgp:
            trainer = MinibatchGPTrainer(
                model,
                batch_size=batch_size,
                variational_lr=variational_lr,
                kl_beta=kl_beta,
                **trainer_kwargs,
            )
        else:
            trainer = GPTrainer(model, mll_class=task_mll_class, **trainer_kwargs)
        t0 = time.time()
        runs = trainer.train()
        train_time = time.time() - t0
        total_train_time += train_time
        successful = [r for r in runs if r.get("loss") is not None and r.get("state_dict") is not None]
        if not successful:
            raise RuntimeError(f"All training runs failed for {task_name}")
        best_run = min(successful, key=lambda r: r["loss"])
        model.load_state_dict(best_run["state_dict"])
        best_loss = float(best_run["loss"])
        y_std_for_noise = y_scaler.std.squeeze() if y_scaler is not None else None
        learned_noise = extract_learned_likelihood_noise(model, y_std=y_std_for_noise)
        ard_info = extract_ard_lengthscales(model)
        ard_mapped = map_ard_to_bands(
            ard_info["lengthscale"],
            band_indices,
            wavelengths_nm=wavelengths_np,
            ard_space="pca_components" if n_components_by_task is not None else "bands",
        )
        if ard_info["outputscale"] is not None:
            ard_mapped["outputscale"] = ard_info["outputscale"]
        ard_mapped["raw_lengthscale"] = ard_info["raw_lengthscale"]
        ard_mapped["model_input_dim"] = int(x_tr.shape[-1])
        if svgp and nigp:
            ard_mapped["input_noise"] = [
                float(v) for v in model.input_noise.detach().cpu().reshape(-1).tolist()
            ]
        if pca_meta is not None:
            ard_mapped["pca"] = pca_meta
        input_bands_by_task[task_name] = {
            "indices": list(band_indices),
            "wavelength_nm": [float(wavelengths_np[i]) for i in band_indices],
            "model_input_dim": int(x_tr.shape[-1]),
            "x_scaling_type": x_scaling_type,
            **ard_mapped,
        }

        if save_checkpoint and svgp and save_path:
            _svgp_dir = _ROOT / "experiments_SVGP"
            if str(_svgp_dir) not in sys.path:
                sys.path.insert(0, str(_svgp_dir))
            from toa_svgp_checkpoint import checkpoint_path_for_run, save_toa_svgp_checkpoint

            ckpt_path = checkpoint_path_for_run(save_path, title, task_name)
            save_toa_svgp_checkpoint(
                ckpt_path,
                model=model,
                task_name=task_name,
                train_x=x_tr,
                train_y=y_tr_fit,
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
                model_config={
                    "num_inducing": int(num_inducing),
                    "learn_inducing_locations": bool(learn_inducing_locations),
                    "nigp": bool(nigp),
                    "seed": int(seed),
                    "ard": bool(ard),
                    "batch_size": int(batch_size),
                    "kl_beta": float(kl_beta),
                    "variational_lr": variational_lr,
                },
                input_column_indices=torch.as_tensor(band_indices, dtype=torch.int64),
                pca_meta=pca_meta,
                x_transform=x_transform or "none",
                log_scale_qoi=list(log_scale_qoi) if log_scale_qoi is not None else None,
                logit_scale_qoi=list(logit_scale_qoi) if logit_scale_qoi is not None else None,
                input_variable=input_variable,
            )
            print(f"Saved checkpoint: {ckpt_path}")

        model.eval()
        t1 = time.time()
        if svgp:
            pred_mean, lower, upper, pred_std = evaluate_svgp_gp_model(
                model, x_te, chunk_size=predict_chunk_size
            )
        else:
            pred_mean, lower, upper, pred_std = evaluate_gp_model(model, x_te)
        prediction_time = time.time() - t1
        total_prediction_time += prediction_time
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
        apply_log_scale_extra_metrics(
            computed,
            y_true=y_te,
            inv=inv,
            task_name=task_name,
            warps=warps,
        )
        tm = {"best_train_loss": best_loss, **learned_noise, **computed}
        if noise_prior_meta:
            tm.update(noise_prior_meta)
        task_metrics[task_name] = tm
        task_runs[task_name] = runs
        task_best_runs[task_name] = best_run
        y_pred_all.append(pred_mean.numpy())
        y_std_all.append(pred_std.numpy())
        lower_all.append(lower.numpy())
        upper_all.append(upper.numpy())
        collect_log_scale_prediction_arrays(
            inv,
            pred_mean,
            y_pred_mean_all=y_pred_mean_all,
            y_pred_mode_all=y_pred_mode_all,
            log_mu_all=log_mu_all,
            log_sigma_all=log_sigma_all,
            logit_mu_all=logit_mu_all,
            logit_sigma_all=logit_sigma_all,
        )
        print_task_test_metrics(
            task_name,
            computed,
            warps=warps,
        )

    y_pred_stacked = np.stack(y_pred_all, axis=1)
    y_std_stacked = np.stack(y_std_all, axis=1)
    lower_stacked = np.stack(lower_all, axis=1)
    upper_stacked = np.stack(upper_all, axis=1)
    y_pred_mean_stacked = stack_or_none(y_pred_mean_all)
    y_pred_mode_stacked = stack_or_none(y_pred_mode_all)
    log_mu_stacked = stack_or_none(log_mu_all)
    log_sigma_stacked = stack_or_none(log_sigma_all)
    logit_mu_stacked = stack_or_none(logit_mu_all)
    logit_sigma_stacked = stack_or_none(logit_sigma_all)
    y_test_np = y_test.detach().cpu().numpy()
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

    metrics: dict = {
        "title": title,
        "dataset": "s2",
        "input_dim_original": S2_INPUT_DIM,
        "n_train": n_train,
        "n_test": n_test,
        "num_tasks": len(names),
        "task_names": list(names),
        "ard": ard,
        "model_class": "SVGPR" if svgp else "GPR",
        "num_epochs": num_epochs,
        "optimizer": getattr(optimizer_class, "__name__", str(optimizer_class)),
        "optimizer_kwargs": json_safe_optimizer_kwargs(optimizer_kwargs),
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
        "pac_bayes": bool(pac_bayes),
        "pac_bayes_temperature": float(pac_bayes_temperature),
        "pac_bayes_prior_std": float(pac_bayes_prior_std),
        "pac_bayes_posterior_std": float(pac_bayes_posterior_std),
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
    if svgp:
        metrics.update(
            {
                "svgp": True,
                "num_inducing": int(num_inducing),
                "learn_inducing_locations": bool(learn_inducing_locations),
                "batch_size": int(batch_size),
                "variational_lr": variational_lr,
                "kl_beta": float(kl_beta),
                "nigp": bool(nigp),
                "freeze_epoch_nigp": int(freeze_epoch_nigp) if nigp else 0,
            }
        )
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
        for key, value in tm.items():
            metrics[f"{task_name}_{key}"] = value
    if monitor_validation and n_val > 0:
        metrics["monitor_validation"] = True
        metrics["n_val"] = n_val
        metrics["train_pool_size"] = TOA_TRAIN_POOL_SIZE
        metrics["val_pool_size"] = TOA_VAL_POOL_SIZE
        metrics["test_pool_size"] = TOA_TEST_POOL_SIZE
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
            monitor_validation=monitor_validation,
            n_val=n_val,
            save_metrics_json=save_metrics_json,
            y_pred_mean_stacked=y_pred_mean_stacked,
            y_pred_mode_stacked=y_pred_mode_stacked,
            log_mu_stacked=log_mu_stacked,
            log_sigma_stacked=log_sigma_stacked,
            warps=warps,
            logit_mu_stacked=logit_mu_stacked,
            logit_sigma_stacked=logit_sigma_stacked,
        )
    return metrics
