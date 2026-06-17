"""
Single-fidelity Wing (s0): train ORF-GP and TabPFN, then plot marginal slice UQ comparisons.

Fixes all inputs except one dimension x_i (default: other dims at train median),
sweeps x_i over physical bounds, and overlays true wing function vs both models.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
_ORF_DIR = Path(__file__).resolve().parent
for p in (_ROOT, _ORF_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import gpplus
from A1_wing_ORF import (
    WING_CONT_DIM,
    WING_SOURCE_NAMES,
    _expand_per_source,
    _prepare_wing_inputs,
)
from gpplus.models import RFFGPR
from gpplus.training import (
    GPTrainer,
    RFFParameterInitializer,
    RFFWoodburyMarginalLogLikelihood,
    evaluate_rff_gp_model,
)
from gpplus.training.optimizers import LBFGSScipy
from gpplus.utils import StandardScaler, UniformScaler, set_seed
from gpplus.utils.train_eval import train_eval_PFN
from load_experimental_data import generate_mf_wing_data, wing_mixed_variables
from plot_multid_slice_predictions import (
    sanitize_plot_subdir,
    save_gp_tabpfn_marginal_slices,
)
from orf_experiment_utils import (
    DEFAULT_ADAM_KWARGS,
    DEFAULT_LBFGS_KWARGS,
    compute_val_samples_per_source,
    extract_learned_likelihood_noise,
    json_default,
    json_safe_optimizer_kwargs,
    make_validation_callback,
    scale_validation_tensors,
    summarize_validation_from_runs,
    unpack_train_val_test,
)
from tabpfn import TabPFNRegressor


def run_wing_orf_vs_tabpfn(
    train_samples_per_source: list[int] | None = None,
    test_samples_per_source: list[int] | None = None,
    num_orf: int | None = None,
    noise_train: float | list[float] = 0.005,
    noise_test: float | list[float] = 0.005,
    noise_type: str = "gaussian",
    seed: int = 42,
    num_inits: int = 16,
    num_epochs: int = 1,
    device: str = "cpu",
    pfn_device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    save_path: str | None = "experiments_ORF/results/wing_s0_ORF_vs_TabPFN",
    standardize_x: bool = True,
    x_standardize_method: int = 2,
    standardize_y: bool = True,
    ard: bool = True,
    predict_chunk_size: int = 512,
    n_jobs: int | None = None,
    optimizer_kwargs: dict | None = None,
    plot_slices: bool = True,
    max_slice_dims: int | None = None,
    slice_dims: list[int] | None = None,
    fixed_point: str = "median",
    n_grid: int = 200,
    run_models: str | None = None,
    monitor_validation: bool = False,
    val_fraction: float = 0.2,
    validation_verbose: bool = True,
) -> dict:
    """
    Train ORF-GP and TabPFN on wing s0, evaluate on test set, and save slice UQ plots.

    run_models: None=both, 'orf'=ORF only, 'pfn'=TabPFN only (slice plots need both).
    """
    if train_samples_per_source is None:
        train_samples_per_source = [100, 0, 0, 0]
    if test_samples_per_source is None:
        test_samples_per_source = [5000, 0, 0, 0]

    set_seed(seed)
    n_train = sum(train_samples_per_source)
    n_test = sum(test_samples_per_source)
    if num_orf is None:
        num_orf = min(512, max(64, n_train // 3))

    train_noise = _expand_per_source(noise_train)
    test_noise = _expand_per_source(noise_test)

    if num_epochs <= 1:
        optimizer_class = LBFGSScipy
        default_optimizer_kwargs = DEFAULT_LBFGS_KWARGS
    else:
        optimizer_class = torch.optim.Adam
        default_optimizer_kwargs = DEFAULT_ADAM_KWARGS
    if optimizer_kwargs is None:
        optimizer_kwargs = dict(default_optimizer_kwargs)

    title = (
        f"wing_s0_ORFvsTabPFN_tr{train_samples_per_source}_te{test_samples_per_source}_"
        f"orfD{num_orf}_noiseTest{noise_test}_noiseTrain{noise_train}"
    )

    print("=" * 60)
    print(title)
    print(
        f"ORF-GP D={num_orf}, ARD={ard}, inits={num_inits}, epochs={num_epochs}, "
        f"TabPFN device={pfn_device}"
    )
    print("=" * 60)

    val_samples_per_source = (
        compute_val_samples_per_source(train_samples_per_source, val_fraction)
        if monitor_validation
        else [0, 0, 0, 0]
    )
    n_val = sum(val_samples_per_source)
    if monitor_validation and validation_verbose:
        print(f"Validation monitoring: n_val={n_val} ({val_fraction:.0%} of n_train={n_train})")

    data = generate_mf_wing_data(
        train_samples_per_source=train_samples_per_source,
        test_samples_per_source=test_samples_per_source,
        val_samples_per_source=val_samples_per_source,
        seed=seed,
        train_noise=train_noise,
        test_noise=test_noise,
        noise_type=noise_type,
    )
    x_train_mf, y_train, x_val_mf, y_val, x_test_mf, y_test = unpack_train_val_test(data)

    x_train_raw = x_train_mf[:, :WING_CONT_DIM].to(dtype=dtype).clone()
    x_test_raw = x_test_mf[:, :WING_CONT_DIM].to(dtype=dtype).clone()
    y_train = y_train.to(dtype=dtype)
    y_val = y_val.to(dtype=dtype)
    y_test = y_test.to(dtype=dtype)

    x_train, x_test, input_dim, fidelity_source, dropped_source_column = _prepare_wing_inputs(
        x_train_mf.to(dtype=dtype),
        x_test_mf.to(dtype=dtype),
        train_samples_per_source,
        test_samples_per_source,
        drop_source_column=True,
    )
    x_val = (
        x_val_mf[:, :WING_CONT_DIM].contiguous().to(dtype=dtype)
        if x_val_mf.numel() > 0
        else x_val_mf.to(dtype=dtype)
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
            raise ValueError(f"x_standardize_method must be 0, 1, or 2, got {x_standardize_method}")
        x_scaler.fit(x_train)
        x_train = x_scaler.transform(x_train)
        x_test = x_scaler.transform(x_test)
        print(f"X scaling: {x_scaling_type}")

    y_mean, y_std = None, None
    y_scaler = None
    y_train_orf = y_train.clone()
    y_test_eval = y_test.clone()
    if standardize_y:
        y_scaler = StandardScaler()
        y_scaler.fit(y_train.unsqueeze(-1))
        y_mean, y_std = y_scaler.mean.squeeze(), y_scaler.std.squeeze()
        y_train_orf = y_scaler.transform(y_train.unsqueeze(-1)).squeeze(-1)

    results: dict = {
        "title": title,
        "input_dim": input_dim,
        "fidelity_source": WING_SOURCE_NAMES[fidelity_source] if fidelity_source is not None else None,
        "drop_source_column": dropped_source_column,
        "train_samples_per_source": train_samples_per_source,
        "test_samples_per_source": test_samples_per_source,
        "n_train": n_train,
        "n_test": n_test,
        "num_orf": num_orf,
        "orthogonal_orf": True,
        "noise_train": train_noise,
        "noise_test": test_noise,
        "seed": seed,
        "standardize_x": standardize_x,
        "x_scaling_type": x_scaling_type,
        "standardize_y": standardize_y,
        "fixed_point": fixed_point,
        "n_grid": n_grid,
    }

    orf_model = None
    orf_metrics: dict | None = None
    if run_models in (None, "orf"):
        print("\n--- ORF-GP training ---")
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
        model = RFFGPR(x_train, y_train_orf, num_rff=num_orf, ard=ard, orthogonal=True)
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
        t0 = time.time()
        runs = trainer.train()
        orf_train_time = time.time() - t0

        successful = [r for r in runs if r.get("loss") is not None and r.get("state_dict") is not None]
        if not successful:
            errors = [r.get("error", "unknown") for r in runs if r.get("error")]
            raise RuntimeError(
                "All ORF training runs failed. "
                + (f"First error: {errors[0]}" if errors else "")
            )
        best_run = min(successful, key=lambda r: r["loss"])
        model.load_state_dict(best_run["state_dict"])
        orf_model = model
        learned_noise = extract_learned_likelihood_noise(model, y_std=y_std)

        t1 = time.time()
        model.eval()
        model.invalidate_feature_cache()
        pred_mean, lower, upper, pred_std = evaluate_rff_gp_model(
            model, x_test, chunk_size=predict_chunk_size
        )
        orf_pred_time = time.time() - t1
        pred_mean = pred_mean.detach().cpu()
        pred_std = pred_std.detach().cpu()
        lower = lower.detach().cpu()
        upper = upper.detach().cpu()
        if standardize_y:
            pred_mean = pred_mean * y_std.cpu() + y_mean.cpu()
            pred_std = pred_std * y_std.cpu()
            lower = lower * y_std.cpu() + y_mean.cpu()
            upper = upper * y_std.cpu() + y_mean.cpu()

        from gpplus.utils import compute_metrics

        orf_computed = compute_metrics(
            y_test_eval.cpu(),
            pred_mean,
            output_std=pred_std,
            lower_95=lower,
            upper_95=upper,
            training_time=orf_train_time,
            prediction_time=orf_pred_time,
        )
        orf_metrics = {
            "num_orf": num_orf,
            "orthogonal_orf": True,
            "best_train_loss": float(best_run["loss"]),
            "optimizer": getattr(optimizer_class, "__name__", str(optimizer_class)),
            "optimizer_kwargs": json_safe_optimizer_kwargs(optimizer_kwargs),
            **learned_noise,
            **orf_computed,
        }
        if monitor_validation and n_val > 0:
            orf_metrics["monitor_validation"] = True
            orf_metrics["val_fraction"] = val_fraction
            orf_metrics["n_val"] = n_val
            orf_metrics.update(summarize_validation_from_runs(runs, best_run))
        results["orf_metrics"] = orf_metrics
        print(
            f"ORF test RMSE: {orf_computed['RMSE']:.6f}  "
            f"RRMSE: {orf_computed['RRMSE']:.6f}"
        )

    tabpfn_regressor = None
    tabpfn_metrics: dict | None = None
    if run_models in (None, "pfn"):
        print("\n--- TabPFN training/eval ---")
        regressor = TabPFNRegressor(device=pfn_device, random_state=seed)
        tabpfn_metric, y_pred_tabpfn, output_std_tabpfn = train_eval_PFN(
            x_train_raw,
            x_test_raw,
            y_train,
            y_test,
            amp_device=pfn_device if pfn_device in ("cuda", "cpu") else "cpu",
            amp_dtype=torch.float32,
            regressor=regressor,
        )
        tabpfn_regressor = regressor
        tabpfn_metrics = tabpfn_metric
        results["tabpfn_metrics"] = tabpfn_metrics
        print(
            f"TabPFN test RMSE: {tabpfn_metric.get('RMSE', float('nan')):.6f}  "
            f"RRMSE: {tabpfn_metric.get('RRMSE', float('nan')):.6f}"
        )

    slice_paths: list[str] = []
    if plot_slices:
        if orf_model is None or tabpfn_regressor is None:
            print("Slice plots skipped: need both ORF-GP and TabPFN models.")
        else:
            if not hasattr(tabpfn_regressor, "feature_names_in_"):
                tabpfn_regressor.fit(
                    x_train_raw.detach().cpu().numpy(),
                    y_train.detach().cpu().numpy().ravel(),
                )

            def truth_fn(X: torch.Tensor) -> torch.Tensor:
                return wing_mixed_variables(X, source="s0")

            dims = slice_dims
            if dims is None and max_slice_dims is not None:
                dims = list(range(min(max_slice_dims, WING_CONT_DIM)))

            slice_out = (
                Path(save_path) / "plots" / "slice_predictions" / sanitize_plot_subdir(title)
                if save_path
                else Path("plots/slice_predictions") / sanitize_plot_subdir(title)
            )
            saved = save_gp_tabpfn_marginal_slices(
                orf_model=orf_model,
                tabpfn_regressor=tabpfn_regressor,
                X_train_orig=x_train_raw,
                x_scaler=x_scaler,
                standardize_x=standardize_x,
                y_mean=y_mean,
                y_std=y_std,
                truth_fn=truth_fn,
                out_dir=slice_out,
                title=title,
                slice_dims=dims,
                fixed_point=fixed_point,
                n_grid=n_grid,
                predict_chunk_size=predict_chunk_size,
            )
            slice_paths = [str(p) for p in saved]
            results["slice_plot_paths"] = slice_paths
            print(f"Saved {len(slice_paths)} slice plots to {slice_out}")

    if save_path:
        out_dir = Path(save_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_json = out_dir / f"{title}.json"
        out_json.write_text(json.dumps(results, indent=2, default=json_default), encoding="utf-8")
        print(f"Saved results to {out_json}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Wing s0 ORF-GP vs TabPFN with slice UQ plots")
    parser.add_argument(
        "--train-per-source",
        type=int,
        nargs=4,
        default=[100, 0, 0, 0],
        metavar=("S0", "S1", "S2", "S3"),
    )
    parser.add_argument(
        "--test-per-source",
        type=int,
        nargs=4,
        default=[5000, 0, 0, 0],
        metavar=("S0", "S1", "S2", "S3"),
    )
    parser.add_argument("--num-orf", type=int, default=None)
    parser.add_argument("--noise-train", type=float, default=0.005)
    parser.add_argument("--noise-test", type=float, default=0.005)
    parser.add_argument("--num-inits", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--pfn-device", type=str, default="cpu")
    parser.add_argument(
        "--save-path",
        type=str,
        default="experiments_ORF/results/wing_s0_ORF_vs_TabPFN",
    )
    parser.add_argument("--max-slice-dims", type=int, default=None, help="Plot first K dims only")
    parser.add_argument(
        "--slice-dims",
        type=int,
        nargs="*",
        default=None,
        help="Explicit dimension indices to plot (overrides --max-slice-dims)",
    )
    parser.add_argument("--fixed-point", type=str, default="median", choices=("median", "mean"))
    parser.add_argument("--n-grid", type=int, default=200)
    parser.add_argument("--no-plot-slices", action="store_true")
    parser.add_argument("--run-models", type=str, default=None, choices=("orf", "pfn"))
    args = parser.parse_args()

    gpplus.config.configure_logger()
    run_wing_orf_vs_tabpfn(
        train_samples_per_source=list(args.train_per_source),
        test_samples_per_source=list(args.test_per_source),
        num_orf=args.num_orf,
        noise_train=args.noise_train,
        noise_test=args.noise_test,
        num_inits=args.num_inits,
        seed=args.seed,
        device=args.device,
        pfn_device=args.pfn_device,
        save_path=args.save_path,
        plot_slices=not args.no_plot_slices,
        max_slice_dims=args.max_slice_dims,
        slice_dims=args.slice_dims,
        fixed_point=args.fixed_point,
        n_grid=args.n_grid,
        run_models=args.run_models,
    )
