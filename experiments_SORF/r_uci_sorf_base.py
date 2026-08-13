"""Shared SORF train/eval pipeline for UCI / Kaggle regression (R1–R4) scripts."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
_SORF_DIR = Path(__file__).resolve().parent
for p in (_ROOT, _SORF_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import gpplus
from gpplus.models import RFFGPR
from gpplus.training import (
    GPTrainer,
    NIGPInputNoiseFreezeCallback,
    RFFParameterInitializer,
    RFFWoodburyMarginalLogLikelihood,
    evaluate_rff_gp_model,
    nigp_woodbury_mll_class,
    pac_bayes_mll_class,
)
from gpplus.training.optimizers import LBFGSScipy
from gpplus.utils import StandardScaler, UniformScaler, compute_metrics, set_seed
from gpplus.utils.fs_path import ensure_parent, fs_path
from load_uci_regression_data import DATASET_DIRS, DATASET_TARGETS, load_train_test
from sorf_experiment_utils import (
    DEFAULT_ADAM_KWARGS,
    DEFAULT_LBFGS_KWARGS,
    VAL_SEED_OFFSET,
    compute_n_val,
    extract_learned_likelihood_noise,
    json_safe_optimizer_kwargs,
    make_validation_callback,
    save_metrics_json,
    plot_validation_curves_after_save,
    scale_validation_tensors,
    summarize_validation_from_runs,
)

DEFAULT_WOODBURY_FORM = "dual"

DATASET_SLUGS = {
    "fish": "qsar_fish_toxicity",
    "concrete": "concrete_compressive_strength",
    "fiat": "used_fiat_500",
    "airfoil": "airfoil_self_noise",
}


def _extract_nigp_input_noise(model) -> dict | None:
    """Learned per-dimension NIGP σ_x if enabled; else None."""
    if not bool(getattr(model, "nigp", False)):
        return None
    if not hasattr(model, "raw_input_noise"):
        return None
    try:
        raw = model.raw_input_noise.detach().cpu().reshape(-1)
        std = model.input_noise.detach().cpu().reshape(-1)
        var = model.input_noise_var.detach().cpu().reshape(-1)
    except Exception:
        return None
    return {
        "raw_input_noise": [float(v) for v in raw.tolist()],
        "input_noise": [float(v) for v in std.tolist()],
        "input_noise_var": [float(v) for v in var.tolist()],
    }


def save_true_vs_pred_plot(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray | None,
    upper: np.ndarray | None,
    target_name: str,
    out_path: str | Path,
    title: str | None = None,
    rmse: float | None = None,
    rrmse: float | None = None,
) -> Path:
    """Scatter of true vs predicted with y=x and optional 95% interval vlines."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    out_path = Path(out_path)
    ensure_parent(out_path)

    if rmse is None:
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    if rrmse is None:
        y_std = float(np.std(y_true))
        rrmse = float(rmse / y_std) if y_std > 0 else float("nan")

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(y_true, y_pred, s=8, alpha=0.35, edgecolors="none")
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="y = x")
    if lower is not None and upper is not None:
        lower = np.asarray(lower, dtype=np.float64).ravel()
        upper = np.asarray(upper, dtype=np.float64).ravel()
        n = len(y_true)
        idx = np.linspace(0, n - 1, num=min(80, n), dtype=int)
        ax.vlines(
            y_true[idx],
            lower[idx],
            upper[idx],
            colors="C0",
            alpha=0.15,
            lw=0.8,
        )
    ax.set_xlabel(f"True {target_name}")
    ax.set_ylabel(f"Predicted {target_name}")
    base_title = title or f"{target_name}: predicted vs true"
    ax.set_title(f"{base_title}\nRMSE {rmse:.4f}   RRMSE {rrmse:.4f}")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(fs_path(out_path), dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _carve_validation_from_train(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    n_val: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hold out n_val points from the training split for validation monitoring."""
    n = int(x_train.shape[0])
    if n_val <= 0 or n_val >= n:
        empty_x = x_train.new_zeros((0, x_train.shape[-1]))
        empty_y = y_train.new_zeros((0,))
        return x_train, y_train, empty_x, empty_y
    rng = np.random.default_rng(seed + VAL_SEED_OFFSET)
    perm = rng.permutation(n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return (
        x_train[train_idx],
        y_train[train_idx],
        x_train[val_idx],
        y_train[val_idx],
    )


def run_uci_sorf(
    dataset: str,
    *,
    train_frac: float = 2.0 / 3.0,
    num_sorf: int | None = None,
    seed: int = 42,
    num_inits: int = 8,
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
    monitor_validation: bool = True,
    val_fraction: float = 0.2,
    validation_verbose: bool = True,
    log_every_n_epochs: int = 50,
    plot_validation: bool = True,
    plot_true_vs_pred: bool = True,
    correct_sorf: bool = True,
    spectral_kernel: str = "rbf",
    nigp: bool = False,
    freeze_epoch_nigp: int = 100,
    nigp_slope_refreshes: int | None = None,
    pac_bayes: bool = False,
    pac_bayes_temperature: float = 2.55,
    pac_bayes_prior_std: float = 0.75,
    pac_bayes_posterior_std: float = 0.5,
) -> dict:
    """
    Train SORF-GP on a UCI/Kaggle regression dataset and evaluate on held-out test points.

    Split is a single seeded shuffle with ``train_frac`` for train (default 2/3) and the
    remainder for test. Optional validation is carved from the train split only.
    """
    key = dataset.strip().lower()
    if key not in DATASET_SLUGS:
        raise ValueError(f"Unknown dataset {dataset!r}. Choose from {sorted(DATASET_SLUGS)}")

    set_seed(seed)
    slug = DATASET_SLUGS[key]
    target_name = DATASET_TARGETS[key]
    if save_path is None:
        save_path = f"experiments_SORF/results/{slug}_sorf"

    x_train, y_train, x_test, y_test = load_train_test(
        key, train_frac=train_frac, seed=seed, print_info=True
    )
    n_train_full = int(x_train.shape[0])
    n_test = int(x_test.shape[0])

    n_val = compute_n_val(n_train_full, val_fraction) if monitor_validation else 0
    if monitor_validation and validation_verbose:
        print(
            f"Validation monitoring: carve n_val={n_val} "
            f"({val_fraction:.0%} of n_train={n_train_full}) from train split"
        )
    x_train, y_train, x_val, y_val = _carve_validation_from_train(
        x_train, y_train, n_val=n_val, seed=seed
    )
    n_train = int(x_train.shape[0])

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

    sk = str(spectral_kernel).strip().lower()
    if sk not in ("rbf", "matern32"):
        raise ValueError(f"spectral_kernel must be 'rbf' or 'matern32', got {spectral_kernel!r}")

    nigp_tag = "_nigp" if nigp else ""
    freeze_tag = (
        f"_freezeepochnigp{int(freeze_epoch_nigp)}"
        if nigp and int(freeze_epoch_nigp) > 0
        else ""
    )
    slope_tag = (
        f"_sloperefreshes{int(nigp_slope_refreshes)}"
        if nigp and nigp_slope_refreshes is not None
        else ""
    )
    sk_tag = f"_{sk}" if sk != "rbf" else ""
    pac_tag = "_pacbayes" if pac_bayes else ""
    title = (
        f"{slug}_trainFrac{train_frac:.4f}_sorfD{num_sorf}{sk_tag}{nigp_tag}{freeze_tag}{slope_tag}{pac_tag}_"
        f"seed{seed}_nTrain{n_train}_nTest{n_test}"
    )
    print("=" * 60)
    print(title)
    feature_dim = 2 * num_sorf
    print(
        f"SORF kernel (Woodbury), D={num_sorf}, m={feature_dim}, ARD={ard}, "
        f"spectral_kernel={sk}, correct_sorf={correct_sorf}, dtype={dtype}, "
        f"inits={num_inits}, epochs={num_epochs}"
    )
    if nigp:
        print(
            f"NIGP: on (independent input noise, "
            f"sigma_x=10^SoftClamp(raw), freeze_epochs={int(freeze_epoch_nigp)}, "
            f"slope_refreshes={nigp_slope_refreshes})"
        )
    else:
        print("NIGP: off")
    if pac_bayes:
        print(
            f"PAC-Bayes MLL: on (temperature={pac_bayes_temperature}, "
            f"prior_std={pac_bayes_prior_std}, posterior_std={pac_bayes_posterior_std})"
        )
    else:
        print("PAC-Bayes MLL: off")
    opt_name = getattr(optimizer_class, "__name__", str(optimizer_class))
    print(f"Optimizer: {opt_name}, kwargs={optimizer_kwargs}")
    print(f"Woodbury: n_train={n_train}, m/n={feature_dim / max(n_train, 1):.4f}")
    if feature_dim >= n_train:
        print(
            f"WARNING: m={feature_dim} >= n_train={n_train}; Woodbury may not beat dense GP. "
            f"Consider num_sorf <= {max(1, n_train // 2 - 1)}."
        )
    print("=" * 60)

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
    if monitor_validation and n_val > 0 and x_val_scaled.numel() > 0:
        callbacks.append(
            make_validation_callback(
                x_val_scaled,
                y_val_scaled,
                num_inits,
                chunk_size=predict_chunk_size,
                verbose=validation_verbose,
                log_every_n_epochs=log_every_n_epochs,
            )
        )
    if nigp and int(freeze_epoch_nigp) > 0 and num_epochs > 1:
        callbacks.append(
            NIGPInputNoiseFreezeCallback(
                freeze_epochs=int(freeze_epoch_nigp),
                verbose=True,
            )
        )

    model = RFFGPR(
        x_train,
        y_train,
        num_rff=num_sorf,
        ard=ard,
        rff_sampling="sorf",
        correct_sorf=correct_sorf,
        spectral_kernel=sk,
        nigp=bool(nigp),
    )

    nigp_mll_kwargs: dict = {}
    if nigp:
        slope_r = None if nigp_slope_refreshes is None else int(nigp_slope_refreshes)
        if slope_r is not None:
            nigp_mll_kwargs["nigp_slope_refreshes"] = slope_r
            nigp_mll_kwargs["nigp_active_epochs"] = max(
                1, int(num_epochs) - max(0, int(freeze_epoch_nigp))
            )
        mll_class = nigp_woodbury_mll_class(DEFAULT_WOODBURY_FORM, **nigp_mll_kwargs)
    else:
        mll_class = RFFWoodburyMarginalLogLikelihood

    if pac_bayes:
        mll_class = pac_bayes_mll_class(
            mll_class,
            temperature=pac_bayes_temperature,
            prior_std=pac_bayes_prior_std,
            posterior_std=pac_bayes_posterior_std,
        )

    trainer = GPTrainer(
        model,
        mll_class=mll_class,
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
    if nigp:
        # Ensure NIGP correction is active for evaluation after optional freeze warm-start.
        model.nigp_correction_enabled = True
    best_loss = float(best_run["loss"])
    learned_noise = extract_learned_likelihood_noise(model, y_std=y_std)
    nigp_info = _extract_nigp_input_noise(model) if nigp else None

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

    metrics = {
        "title": title,
        "dataset": key,
        "dataset_dir": DATASET_DIRS[key],
        "target_name": target_name,
        "n_train": n_train,
        "n_train_full": n_train_full,
        "n_test": n_test,
        "train_frac": train_frac,
        "num_sorf": num_sorf,
        "rff_sampling": "sorf",
        "correct_sorf": correct_sorf,
        "spectral_kernel": sk,
        "nigp": bool(nigp),
        "freeze_epoch_nigp": int(freeze_epoch_nigp),
        "nigp_slope_refreshes": (
            None if nigp_slope_refreshes is None else int(nigp_slope_refreshes)
        ),
        "pac_bayes": bool(pac_bayes),
        "pac_bayes_temperature": float(pac_bayes_temperature),
        "pac_bayes_prior_std": float(pac_bayes_prior_std),
        "pac_bayes_posterior_std": float(pac_bayes_posterior_std),
        "feature_dim": 2 * num_sorf,
        "ard": ard,
        "num_epochs": num_epochs,
        "optimizer": getattr(optimizer_class, "__name__", str(optimizer_class)),
        "optimizer_kwargs": json_safe_optimizer_kwargs(optimizer_kwargs),
        "best_train_loss": best_loss,
        "seed": seed,
        "standardize_x": standardize_x,
        "x_standardize_method": x_standardize_method,
        "x_scaling_type": x_scaling_type,
        **learned_noise,
        **computed,
    }
    if nigp_info is not None:
        metrics["input_noise"] = nigp_info["input_noise"]
        metrics["input_noise_var"] = nigp_info["input_noise_var"]
        metrics["raw_input_noise"] = nigp_info["raw_input_noise"]
        metrics["input_noise_mean"] = float(sum(nigp_info["input_noise"]) / max(len(nigp_info["input_noise"]), 1))
    if monitor_validation and n_val > 0:
        metrics["monitor_validation"] = True
        metrics["val_fraction"] = val_fraction
        metrics["n_val"] = n_val
        metrics.update(summarize_validation_from_runs(runs, best_run))

    print(
        f"\nTest RMSE: {computed['RMSE']:.6f}  RRMSE: {computed['RRMSE']:.6f}  "
        f"MAE: {computed['MAE']:.6f}"
    )
    if "NIS" in computed:
        print(f"NIS: {computed['NIS']:.4f}")
    print(f"Best training loss: {best_loss:.4f}  Time: {train_time:.1f}s")

    if save_path:
        out_json = save_metrics_json(metrics, save_path, title)
        print(f"Saved metrics to {out_json}")
        if plot_validation and monitor_validation and n_val > 0:
            for plot_path in plot_validation_curves_after_save(metrics, save_path, out_json):
                print(f"Saved validation plot to {plot_path}")
        if plot_true_vs_pred:
            scatter_path = (
                Path(save_path) / "plots" / "true_vs_pred" / f"{title}_true_vs_pred.png"
            )
            saved = save_true_vs_pred_plot(
                y_true=y_test_eval.numpy(),
                y_pred=pred_mean.numpy(),
                lower=lower.numpy(),
                upper=upper.numpy(),
                target_name=target_name,
                out_path=scatter_path,
                title=title,
                rmse=float(computed["RMSE"]),
                rrmse=float(computed["RRMSE"]),
            )
            print(f"Saved true vs pred plot to {saved}")
            metrics["true_vs_pred_plot"] = str(saved)

    return metrics
