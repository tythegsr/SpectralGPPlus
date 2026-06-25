"""
Shared 1D TabPFN-style toy benchmark runner using SORF-GP (Woodbury inference).
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
_SORF_DIR = Path(__file__).resolve().parent
for p in (_ROOT, _SORF_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gpplus.models import RFFGPR
from gpplus.training import (
    GPTrainer,
    RFFParameterInitializer,
    RFFWoodburyMarginalLogLikelihood,
    evaluate_rff_gp_model,
)
from gpplus.training.optimizers import LBFGSScipy
from gpplus.utils import StandardScaler, UniformScaler, compute_metrics, set_seed
from plot_sorf1d_predictions import save_sorf1d_prediction_plot
from tabpfn1d_eval import eval_tabpfn_1d
from sorf_experiment_utils import (
    DEFAULT_ADAM_KWARGS,
    DEFAULT_LBFGS_KWARGS,
    VAL_SEED_OFFSET,
    build_1d_run_file_tag,
    compute_n_val,
    extract_learned_likelihood_noise,
    json_safe_optimizer_kwargs,
    make_validation_callback,
    save_metrics_json,
    scale_validation_tensors,
    summarize_validation_from_runs,
    unpack_train_val_test,
)


def run_tabpfn1d_sorf(
    *,
    problem_name: str,
    generate_data_fn: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    true_fn: Callable[[torch.Tensor], torch.Tensor],
    train_size: int = 10,
    num_sorf: int | None = None,
    num_test: int = 5000,
    x_bounds: list[float] | None = None,
    test_x_bounds: list[float] | None = None,
    test_outside_margin: float = 0.0,
    generate_data_kwargs: dict[str, Any] | None = None,
    noise_train: float = 0.0,
    noise_test: float = 0.0,
    noise_type: str = "gaussian",
    seed: int = 42,
    num_inits: int = 16,
    num_epochs: int = 1,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    save_path: str | None = None,
    standardize_x: bool = True,
    x_standardize_method: int = 2,
    standardize_y: bool = True,
    ard: bool = True,
    predict_chunk_size: int = 512,
    n_jobs: int | None = None,
    optimizer_kwargs: dict | None = None,
    plot_1d: bool = True,
    monitor_validation: bool = False,
    val_fraction: float = 0.2,
    validation_verbose: bool = False,
    run_tabpfn: bool = True,
    pfn_device: str = "cpu",
    run_models: str | None = None,
) -> dict:
    """Train SORF-GP on a 1D toy function and evaluate on a (possibly extended) test grid."""
    if x_bounds is None:
        x_bounds = [-0.5, 0.5]
    if test_x_bounds is None:
        test_x_bounds = [x_bounds[0] - test_outside_margin, x_bounds[1] + test_outside_margin]

    set_seed(seed)
    dimensions = 1
    n_train = train_size
    if num_sorf is None:
        num_sorf = min(512, max(64, n_train // 3))

    if num_epochs <= 1:
        optimizer_class = LBFGSScipy
        default_optimizer_kwargs = DEFAULT_LBFGS_KWARGS
    else:
        optimizer_class = torch.optim.Adam
        default_optimizer_kwargs = DEFAULT_ADAM_KWARGS
    if optimizer_kwargs is None:
        optimizer_kwargs = dict(default_optimizer_kwargs)

    gen_kwargs = dict(generate_data_kwargs or {})
    frequency = gen_kwargs.get("frequency")

    title = (
        f"TabPFN1D_{problem_name}_{train_size}Dn_{x_bounds}_"
        f"sorfD{num_sorf}_ood{test_outside_margin}_"
        f"noiseTest{noise_test}_noiseTrain{noise_train}"
    )
    print("=" * 60)
    print(title)
    feature_dim = 2 * num_sorf
    print(
        f"SORF kernel (Woodbury), D={num_sorf}, m={feature_dim}, ARD={ard}, "
        f"dtype={dtype}, inits={num_inits}, epochs={num_epochs}"
    )
    print(f"Train x in {x_bounds}, test x in {test_x_bounds}")
    if frequency is not None:
        print(f"Frequency: {float(frequency):g}  (f(x) = sin({float(frequency):g} * pi * x))")
    n_val = compute_n_val(n_train, val_fraction) if monitor_validation else 0
    if monitor_validation and validation_verbose:
        print(f"Validation monitoring: n_val={n_val} ({val_fraction:.0%} of n_train={n_train})")
    print("=" * 60)
    data = generate_data_fn(
        n_train=n_train,
        n_test=num_test,
        dimensions=dimensions,
        x_bounds=x_bounds,
        test_x_bounds=test_x_bounds,
        train_noise=noise_train,
        test_noise=noise_test,
        noise_type=noise_type,
        seed=seed,
        n_val=n_val,
        val_seed=seed + VAL_SEED_OFFSET if n_val > 0 else None,
        **gen_kwargs,
    )
    x_train, y_train, x_val, y_val, x_test, y_test = unpack_train_val_test(data)

    x_train_raw = x_train[:, 0].detach().cpu().to(dtype=torch.float64).numpy().ravel()
    y_train_raw = y_train.detach().cpu().to(dtype=torch.float64).numpy().ravel()
    x_test_raw = x_test[:, 0].detach().cpu().to(dtype=torch.float64).numpy().ravel()
    y_test_raw = y_test.detach().cpu().to(dtype=torch.float64).numpy().ravel()
    y_true_test = true_fn(x_test.to(dtype=torch.float64)).detach().cpu().numpy().ravel()

    x_train = x_train.to(dtype=dtype)
    x_test = x_test.to(dtype=dtype)
    y_train = y_train.to(dtype=dtype)
    y_test = y_test.to(dtype=dtype)

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
        print(f"X scaling: {x_scaling_type}")

    y_mean, y_std = None, None
    y_scaler = None
    if standardize_y:
        y_scaler = StandardScaler()
        y_scaler.fit(y_train.unsqueeze(-1))
        y_mean, y_std = y_scaler.mean.squeeze(), y_scaler.std.squeeze()
        y_train = y_scaler.transform(y_train.unsqueeze(-1)).squeeze(-1)
        y_test_scaled = y_scaler.transform(y_test.unsqueeze(-1)).squeeze(-1)
    else:
        y_test_scaled = y_test

    x_val_scaled, y_val_scaled = scale_validation_tensors(
        x_val,
        y_val,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        standardize_x=standardize_x,
        standardize_y=standardize_y,
        dtype=dtype,
    )
    callbacks = []
    if monitor_validation and n_val > 0:
        callbacks.append(
            make_validation_callback(
                x_val_scaled,
                y_val_scaled,
                num_inits,
                chunk_size=predict_chunk_size,
                verbose=validation_verbose,
            )
        )

    model = RFFGPR(x_train, y_train, num_rff=num_sorf, ard=ard, rff_sampling="sorf")

    trainer = GPTrainer(
        model,
        mll_class=RFFWoodburyMarginalLogLikelihood,
        num_epochs=num_epochs,
        num_inits=num_inits,
        seed=seed,
        device=device,
        dtype=dtype,
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
        initializer_class=RFFParameterInitializer,
        n_jobs=n_jobs,
        inner_max_num_threads=1,
        cholesky_jitter=1e-6,
        callbacks=callbacks,
    )
    t_train = time.time()
    runs = trainer.train()
    train_time = time.time() - t_train

    successful = [r for r in runs if r.get("loss") is not None and r.get("state_dict") is not None]
    if not successful:
        errors = [r.get("error", "unknown") for r in runs if r.get("error")]
        raise RuntimeError(
            "All training runs failed. "
            + (f"First error: {errors[0]}" if errors else "Check optimizer kwargs.")
        )
    best_run = min(successful, key=lambda r: r["loss"])
    model.load_state_dict(best_run["state_dict"])
    best_loss = float(best_run["loss"])
    learned_noise = extract_learned_likelihood_noise(model, y_std=y_std)

    model.eval()
    model.invalidate_feature_cache()
    t_pred = time.time()
    pred_mean, lower, upper, pred_std = evaluate_rff_gp_model(
        model, x_test, chunk_size=predict_chunk_size
    )
    prediction_time = time.time() - t_pred
    pred_mean = pred_mean.detach().cpu()
    pred_std = pred_std.detach().cpu()
    lower = lower.detach().cpu()
    upper = upper.detach().cpu()

    if standardize_y:
        pred_mean = pred_mean * y_std.cpu() + y_mean.cpu()
        pred_std = pred_std * y_std.cpu()
        lower = lower * y_std.cpu() + y_mean.cpu()
        upper = upper * y_std.cpu() + y_mean.cpu()
        y_test_eval = y_test.cpu()
    else:
        y_test_eval = y_test_scaled.cpu()

    computed = compute_metrics(
        y_test_eval,
        pred_mean,
        output_std=pred_std,
        lower_95=lower,
        upper_95=upper,
        training_time=train_time,
        prediction_time=prediction_time,
    )

    tabpfn_metrics: dict | None = None
    pred_tabpfn_np: np.ndarray | None = None
    std_tabpfn_np: np.ndarray | None = None
    if run_tabpfn and run_models in (None, "pfn"):
        print("\n--- TabPFN training/eval (raw x/y, no GP preprocessing) ---")
        tabpfn_metrics, pred_tabpfn_np, std_tabpfn_np = eval_tabpfn_1d(
            x_train_raw=x_train_raw,
            x_test_raw=x_test_raw,
            y_train_raw=y_train_raw,
            y_test=y_test_raw,
            seed=seed,
            pfn_device=pfn_device,
        )
        print(
            f"TabPFN test RMSE: {tabpfn_metrics.get('RMSE', float('nan')):.6f}  "
            f"RRMSE: {tabpfn_metrics.get('RRMSE', float('nan')):.6f}"
        )

    metrics = {
        "title": title,
        "problem_name": problem_name,
        "dimensions": dimensions,
        "n_train": n_train,
        "train_size": train_size,
        "n_test": num_test,
        "num_sorf": num_sorf,
        "rff_sampling": "sorf",
        "feature_dim": 2 * num_sorf,
        "ard": ard,
        "num_epochs": num_epochs,
        "optimizer": getattr(optimizer_class, "__name__", str(optimizer_class)),
        "optimizer_kwargs": json_safe_optimizer_kwargs(optimizer_kwargs),
        "best_train_loss": best_loss,
        "noise_train": noise_train,
        "noise_test": noise_test,
        "noise_type": noise_type,
        **learned_noise,
        "x_bounds": list(x_bounds),
        "test_x_bounds": list(test_x_bounds),
        "test_outside_margin": test_outside_margin,
        "standardize_x": standardize_x,
        "x_standardize_method": x_standardize_method,
        "x_scaling_type": x_scaling_type,
        **gen_kwargs,
        **computed,
    }
    if monitor_validation and n_val > 0:
        metrics["monitor_validation"] = True
        metrics["val_fraction"] = val_fraction
        metrics["n_val"] = n_val
        metrics.update(summarize_validation_from_runs(runs, best_run))
    if tabpfn_metrics is not None:
        metrics["tabpfn_metrics"] = tabpfn_metrics

    print(
        f"\nTest RMSE: {computed['RMSE']:.6f}  RRMSE: {computed['RRMSE']:.6f}  "
        f"MAE: {computed['MAE']:.6f}"
    )
    if "NIS" in computed:
        print(f"NIS: {computed['NIS']:.4f}")
    print(
        f"Learned noise_std: {learned_noise.get('noise_std', float('nan')):.6f}  "
        f"(injected noise_test: {noise_test})"
    )
    print(f"Best training loss: {best_loss:.4f}  Time: {train_time:.1f}s")

    pred_np = pred_mean.numpy().ravel()
    std_np = pred_std.numpy().ravel()
    lower_np = lower.numpy().ravel()
    upper_np = upper.numpy().ravel()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        file_tag = build_1d_run_file_tag(
            train_size=train_size,
            noise_train=noise_train,
            noise_test=noise_test,
            seed=seed,
            test_outside_margin=test_outside_margin,
            frequency=float(frequency) if frequency is not None else None,
        )
        npz_path = os.path.join(save_path, f"predictions_1d_{file_tag}.npz")
        npz_kwargs: dict[str, Any] = dict(
            x_train=x_train_raw,
            y_train=y_train_raw,
            x_test=x_test_raw,
            y_true=y_true_test,
            y_pred=pred_np,
            y_std=std_np,
            lower_95=lower_np,
            upper_95=upper_np,
            x_bounds=np.array(x_bounds, dtype=np.float64),
            test_x_bounds=np.array(test_x_bounds, dtype=np.float64),
            title=title,
            seed=seed,
            train_size=train_size,
            noise_train=noise_train,
            noise_test=noise_test,
            test_outside_margin=test_outside_margin,
            file_tag=file_tag,
            frequency=float(frequency) if frequency is not None else np.nan,
        )
        if pred_tabpfn_np is not None:
            npz_kwargs["y_pred_tabpfn"] = pred_tabpfn_np
            npz_kwargs["y_std_tabpfn"] = std_tabpfn_np
            npz_kwargs["run_tabpfn"] = True
        np.savez(npz_path, **npz_kwargs)
        print(f"Saved 1D predictions to {npz_path}")

        out_json = save_metrics_json(metrics, save_path, title)
        print(f"Saved metrics to {out_json}")

        if plot_1d:
            plot_dir = os.path.join(save_path, "plots", "prediction_runs")
            fp = save_sorf1d_prediction_plot(
                x_train_raw,
                y_train_raw,
                x_test_raw,
                pred_np,
                std_np,
                plot_dir,
                title=title,
                run_index=seed,
                y_true_test=y_true_test,
                x_bounds=x_bounds,
                test_x_bounds=test_x_bounds,
                file_tag=file_tag,
                y_pred_tabpfn=pred_tabpfn_np,
                y_std_tabpfn=std_tabpfn_np,
            )
            print(f"Saved 1D plot to {fp}")

    return metrics
