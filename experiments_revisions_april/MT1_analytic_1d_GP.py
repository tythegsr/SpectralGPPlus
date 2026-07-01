import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import gpytorch

import gpplus
import defaults
from gp_prediction_diagnostics import (
    predict_gp_denormalized,
    prediction_diagnostics_plot_subdir,
    save_true_vs_pred_scatter,
)
from gpplus.utils import set_seed
from gpplus.utils.metrics_functions import analyze_metrics
from gpplus.utils.onehot_encode_data import encode_qual_data, learn_encodings
from gpplus.utils.train_eval import train_eval_gp
from load_experimental_data import generate_mt_analytic_1d_data, mt_analytic_1d_function

NUM_TASKS = 2
TASK_NAMES = ("x_squared", "sin_pi_x_plus_4x")


def compute_per_task_metrics(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
    num_tasks: int = NUM_TASKS,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1, num_tasks)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1, num_tasks)
    metrics: dict[str, float] = {}
    for t in range(num_tasks):
        yt = y_true[:, t]
        yp = y_pred[:, t]
        rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
        std = float(np.std(yt))
        rrmse = rmse / std if std > 0 else float("inf")
        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        metrics[f"task{t}_RMSE"] = rmse
        metrics[f"task{t}_RRMSE"] = rrmse
        metrics[f"task{t}_R2"] = r2
    return metrics


def save_mt_task_fit_plot(
    ax: plt.Axes,
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_grid: np.ndarray,
    y_true_grid: np.ndarray,
    y_mean_grid: np.ndarray,
    y_std_grid: np.ndarray,
    task_label: str,
    interval_z: float = 1.96,
) -> None:
    """Draw one task: thin-circle train points, red truth, GP mean + PI."""
    ax.scatter(
        x_train,
        y_train,
        s=22,
        facecolors="none",
        edgecolors="0.45",
        linewidths=0.55,
        label="Train",
        zorder=5,
    )
    ax.plot(x_grid, y_true_grid, color="red", linewidth=2.0, label="True f(x)", zorder=3)
    lower = y_mean_grid - interval_z * y_std_grid
    upper = y_mean_grid + interval_z * y_std_grid
    ax.fill_between(
        x_grid,
        lower,
        upper,
        color="C0",
        alpha=0.22,
        linewidth=0.0,
        label="GP 95% PI",
        zorder=1,
    )
    ax.plot(x_grid, y_mean_grid, color="C0", linewidth=1.8, label="GP mean", zorder=4)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(task_label)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.28)


def save_mt_analytic_fit_plots(
    *,
    out_dir: Path,
    run_index: int,
    experiment_title: str,
    x_bounds: list[float],
    X_train_orig: torch.Tensor,
    y_train: torch.Tensor,
    model,
    Xscaler,
    cont_cols: list[int],
    standardize_X: bool,
    y_train_mean: torch.Tensor | None,
    y_train_std: torch.Tensor | None,
    standardize_y: bool,
    n_grid: int = 400,
    interval_z: float = 1.96,
) -> list[Path]:
    """Per-task fit plots plus a combined 1x2 panel."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = X_train_orig.dtype
    device = next(model.parameters()).device
    x_train_np = X_train_orig[:, 0].detach().cpu().numpy()
    y_train_np = y_train.detach().cpu().numpy()

    x_lin = torch.linspace(x_bounds[0], x_bounds[1], n_grid, dtype=dtype, device=device).reshape(-1, 1)
    X_grid = x_lin.clone()
    if standardize_X and Xscaler is not None:
        X_grid[:, cont_cols] = Xscaler.transform(X_grid[:, cont_cols])

    with torch.no_grad():
        y_true_all = mt_analytic_1d_function(x_lin.cpu()).numpy()
        mean_all, std_all = predict_gp_denormalized(
            model,
            X_grid,
            y_train_mean=y_train_mean,
            y_train_std=y_train_std,
            standardize_y=standardize_y,
            standardize_y_log_scale=False,
            log_scale_C=None,
        )
    mean_np = mean_all.detach().cpu().numpy()
    std_np = std_all.detach().cpu().numpy()
    x_grid_np = x_lin[:, 0].cpu().numpy()

    saved: list[Path] = []
    fig_combined, axes = plt.subplots(1, NUM_TASKS, figsize=(12, 4.5), dpi=120, sharex=True)
    if NUM_TASKS == 1:
        axes = [axes]

    for t in range(NUM_TASKS):
        task_label = TASK_NAMES[t]
        fig_single, ax_single = plt.subplots(figsize=(8, 4.5), dpi=120)
        save_mt_task_fit_plot(
            ax_single,
            x_train=x_train_np,
            y_train=y_train_np[:, t],
            x_grid=x_grid_np,
            y_true_grid=y_true_all[:, t],
            y_mean_grid=mean_np[:, t],
            y_std_grid=std_np[:, t],
            task_label=f"{task_label} (run {run_index + 1})",
            interval_z=interval_z,
        )
        fig_single.suptitle(experiment_title, fontsize=10, y=1.02)
        fig_single.tight_layout()
        fp = out_dir / f"task_{t}_{TASK_NAMES[t]}_fit.png"
        fig_single.savefig(fp, bbox_inches="tight")
        plt.close(fig_single)
        saved.append(fp)

        save_mt_task_fit_plot(
            axes[t],
            x_train=x_train_np,
            y_train=y_train_np[:, t],
            x_grid=x_grid_np,
            y_true_grid=y_true_all[:, t],
            y_mean_grid=mean_np[:, t],
            y_std_grid=std_np[:, t],
            task_label=task_label,
            interval_z=interval_z,
        )

    fig_combined.suptitle(f"{experiment_title}\nrun {run_index + 1}", fontsize=11)
    fig_combined.tight_layout()
    fp_combined = out_dir / "both_tasks_fit.png"
    fig_combined.savefig(fp_combined, bbox_inches="tight")
    plt.close(fig_combined)
    saved.append(fp_combined)
    return saved


def run_mt_prediction_diagnostics(
    *,
    save_path: str,
    run_index: int,
    experiment_title: str,
    x_bounds: list[float],
    X_train_orig: torch.Tensor,
    y_train: torch.Tensor,
    y_test: torch.Tensor,
    y_pred_gp: np.ndarray,
    model,
    Xscaler,
    cont_cols: list[int],
    standardize_X: bool,
    y_train_mean: torch.Tensor | None,
    y_train_std: torch.Tensor | None,
    standardize_y: bool,
    plot_prediction_diagnostics: bool,
    diagnostic_run_indices: tuple[int, ...],
) -> None:
    if not plot_prediction_diagnostics or run_index not in diagnostic_run_indices:
        return
    if save_path is None:
        return

    y_pred = np.asarray(y_pred_gp, dtype=np.float64).reshape(-1, NUM_TASKS)
    y_test_np = y_test.detach().cpu().numpy().reshape(-1, NUM_TASKS)

    base_dir = (
        Path(save_path)
        / "plots"
        / prediction_diagnostics_plot_subdir(experiment_title)
        / f"run_{run_index:03d}"
    )

    saved = save_mt_analytic_fit_plots(
        out_dir=base_dir,
        run_index=run_index,
        experiment_title=experiment_title,
        x_bounds=x_bounds,
        X_train_orig=X_train_orig,
        y_train=y_train,
        model=model,
        Xscaler=Xscaler,
        cont_cols=cont_cols,
        standardize_X=standardize_X,
        y_train_mean=y_train_mean,
        y_train_std=y_train_std,
        standardize_y=standardize_y,
    )
    for fp in saved:
        print(f"  Saved fit plot: {fp}")

    for t in range(NUM_TASKS):
        panel_title = f"{experiment_title} - {TASK_NAMES[t]} (run {run_index + 1})"
        save_true_vs_pred_scatter(
            y_test_np[:, t],
            y_pred[:, t],
            base_dir / f"true_vs_pred_task{t}.png",
            title=panel_title,
            subtitle=TASK_NAMES[t],
        )


def mt_analytic_1d_GP(
    num_runs: int = defaults.NUM_RUNS,
    n_train: int = 30,
    n_test: int = 500,
    x_bounds: list[float] | None = None,
    num_inits: int = defaults.TRAINER_NUM_INITS,
    num_epochs: int = defaults.TRAINER_NUM_EPOCHS,
    lr: float = defaults.TRAINER_LR,
    convergence_patience: int = defaults.TRAINER_CONVERGENCE_PATIENCE,
    min_epochs: int = defaults.TRAINER_MIN_EPOCHS,
    min_loss_change: float = defaults.TRAINER_MIN_LOSS_CHANGE,
    optimizer_class=defaults.TRAINER_OPTIMIZER_CLASS,
    optimizer_kwargs=defaults.TRAINER_OPTIMIZER_KWARGS,
    initializer_class=defaults.TRAINER_INITIALIZER_CLASS,
    gp_device: str = defaults.TRAINER_GP_DEVICE,
    save_path: str = "./results/MT1_analytic_1d",
    title: str | None = None,
    standardize_X: bool = defaults.STANDARDIZE_X,
    standardize_y: bool = defaults.STANDARDIZE_Y,
    x_standardize_method: int = defaults.X_STANDARDIZE_METHOD,
    noise_train: float = 0.0,
    noise_test: float = 0.0,
    noise_type: str = defaults.NOISE_TYPE,
    seed: int = defaults.SEED,
    seed_trainer: int | None = defaults.SEED_TRAINER,
    gp_dtype: torch.dtype = defaults.DTYPE_GP,
    trainer_info: bool = True,
    log_lbfgs_inner: bool = defaults.TRAINER_LOG_LBFGS_INNER,
    single_dataset: bool = True,
    plot_prediction_diagnostics: bool = defaults.PLOT_PREDICTION_DIAGNOSTICS,
    diagnostic_run_indices: tuple[int, ...] = defaults.PREDICTION_DIAGNOSTIC_RUN_INDICES,
):
    if x_bounds is None:
        x_bounds = [-1.0, 1.0]

    if title is None:
        title = (
            f"MT1_analytic_1d_2Dy_{n_train}Dn_[{x_bounds[0]},{x_bounds[1]}]_"
            f"{num_inits}inits_noiseTest{noise_test}_noiseTrain{noise_train}_x{num_runs}"
        )

    print(f" GP Device: {gp_device}")
    callback_save_path = f"{save_path}/trainer_analysis/plots" if save_path else None

    set_seed(seed)

    if single_dataset:
        total_train = n_train
        print(
            f"Generating {n_test + total_train} Sobol samples for 1D multi-task analytic\n\t"
            f"Test: {n_test} / Train: {n_train} "
            f"(single_dataset=True: same train data for all {num_runs} runs)"
        )
    else:
        num_runs_gen = max(num_runs, 20)
        total_train = num_runs_gen * n_train
        print(
            f"Generating {n_test + total_train} Sobol samples for 1D multi-task analytic\n\t"
            f"Test: {n_test} / Train pool: {total_train} "
            f"(single_dataset=False: disjoint train slices, {n_train} points per run)"
        )

    X_train_all, y_train_all, X_test_all, y_test_all = generate_mt_analytic_1d_data(
        n_train=total_train,
        n_test=n_test,
        x_bounds=x_bounds,
        train_noise=noise_train,
        test_noise=noise_test,
        noise_type=noise_type,
        seed=seed,
    )
    X = torch.cat([X_test_all, X_train_all], dim=0)

    print("=" * 10)
    print(f"{title}: MTGPR on analytic 1D -> 2D")
    print("=" * 10)

    qual_dict = learn_encodings(X)
    _, cont_cols, cat_cols, source_cols = encode_qual_data(
        X_train_all, qual_dict=qual_dict, source_col=None
    )

    GPPlus_metrics: list[dict] = []
    GPTrainer_info: list[dict] = []

    if not single_dataset:
        all_indices = torch.randperm(total_train)
        train_indices_2d = all_indices.reshape(num_runs_gen, n_train)

    total_start_time = time.time()
    for i in range(num_runs):
        run_seed = seed_trainer if seed_trainer is not None else (seed + i)
        print(f"\n{'=' * 20} {title} RUN {i + 1}/{num_runs}: {run_seed} {'=' * 20}")

        if single_dataset:
            X_train = X_train_all
            y_train = y_train_all
        else:
            run_train_indices = train_indices_2d[i]
            X_train = X_train_all[run_train_indices]
            y_train = y_train_all[run_train_indices]

        X_train = X_train.detach().clone().to(dtype=gp_dtype)
        X_test = X_test_all.detach().clone().to(dtype=gp_dtype)
        y_train = y_train.detach().clone().to(dtype=gp_dtype)
        y_test = y_test_all.detach().clone().to(dtype=gp_dtype)
        X_train_orig = X_train.detach().clone()
        X_test_orig = X_test.detach().clone()

        Xscaler = None
        if standardize_X:
            if x_standardize_method == 0:
                Xscaler = gpplus.utils.StandardScaler()
                X_scaling_type = "StandardScaler (Gaussian)"
            elif x_standardize_method == 1:
                Xscaler = gpplus.utils.UniformScaler(scale_to_neg_one=False)
                X_scaling_type = "UniformScaler [0, 1]"
            elif x_standardize_method == 2:
                Xscaler = gpplus.utils.UniformScaler(scale_to_neg_one=True)
                X_scaling_type = "UniformScaler [-1, 1]"
            else:
                raise ValueError(
                    f"x_standardize_method must be 0, 1, or 2, got {x_standardize_method}"
                )
            Xscaler.fit(X_train[:, cont_cols])
            X_train[:, cont_cols] = Xscaler.transform(X_train[:, cont_cols])
            X_test[:, cont_cols] = Xscaler.transform(X_test[:, cont_cols])
        else:
            X_scaling_type = "None"

        Yscaler = gpplus.utils.StandardScaler()
        Yscaler.fit(y_train)
        y_train_mean = Yscaler.mean
        y_train_std = Yscaler.std
        y_train_normal = Yscaler.transform(y_train)

        print(f"\n--- {title} MTGPR Training ---")
        base_kernel = gpplus.kernels.LogScaleKernel(
            gpplus.kernels.GaussianKernel() * gpplus.kernels.PeriodicKernel()
        )
        kernel_module = gpytorch.kernels.MultitaskKernel(
            base_kernel,
            num_tasks=NUM_TASKS,
            rank=1,
        )
        model = gpplus.models.MTGPR(
            X_train,
            y_train_normal if standardize_y else y_train,
            num_tasks=NUM_TASKS,
            kernel_module=kernel_module,
        )
        if i == 0 or i == num_runs - 1:
            print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
            print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
            print(model)

        gp_metric, y_pred_gp, output_std_gp, gp_trainer_info = train_eval_gp(
            model,
            X_test,
            y_test,
            num_epochs=num_epochs,
            seed=run_seed,
            num_inits=num_inits,
            lr=lr,
            convergence_patience=convergence_patience,
            min_epochs=min_epochs,
            min_loss_change=min_loss_change,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            initializer_class=initializer_class,
            device=gp_device,
            y_train_mean=y_train_mean if standardize_y else None,
            y_train_std=y_train_std if standardize_y else None,
            source_cols=source_cols,
            trainer_info=trainer_info,
            callbacks=defaults.get_default_gp_callbacks(
                optimizer_class,
                callback_save_path=callback_save_path,
                log_lbfgs_inner=log_lbfgs_inner,
            ),
            callback_save_path=callback_save_path,
            log_lbfgs_inner=log_lbfgs_inner,
            cholesky_jitter=defaults.TRAINER_CHOLESKY_JITTER,
            n_jobs=defaults.TRAINER_N_JOBS,
            inner_max_num_threads=defaults.TRAINER_INNER_MAX_NUM_THREADS,
        )

        per_task = compute_per_task_metrics(y_test, y_pred_gp)
        gp_metric.update(per_task)
        GPPlus_metrics.append(gp_metric)

        if gp_trainer_info:
            gp_trainer_info["run"] = i + 1
            gp_trainer_info["metrics"] = gp_metric
            GPTrainer_info.append(gp_trainer_info)

        print(f"\nGP Results (Run {i + 1}/{num_runs})")
        for k, v in gp_metric.items():
            if isinstance(v, (int, float)) and v is not None:
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v}")

        run_mt_prediction_diagnostics(
            save_path=save_path,
            run_index=i,
            experiment_title=title,
            x_bounds=x_bounds,
            X_train_orig=X_train_orig,
            y_train=y_train,
            y_test=y_test,
            y_pred_gp=y_pred_gp,
            model=model,
            Xscaler=Xscaler,
            cont_cols=cont_cols,
            standardize_X=standardize_X,
            y_train_mean=y_train_mean if standardize_y else None,
            y_train_std=y_train_std if standardize_y else None,
            standardize_y=standardize_y,
            plot_prediction_diagnostics=plot_prediction_diagnostics,
            diagnostic_run_indices=diagnostic_run_indices,
        )

        if i == 0:
            gp_model_info = {
                "model_str": str(model),
                "model_class": "MTGPR",
                "num_tasks": NUM_TASKS,
                "task_names": list(TASK_NAMES),
                "cat_cols": cat_cols,
                "cont_cols": cont_cols,
                "source_cols": source_cols,
                "qual_dict": qual_dict,
                "input_dim": X_train.shape[1],
                "train_samples": X_train.shape[0],
                "test_samples": n_test,
                "y_train_mean": y_train_mean.detach().cpu().tolist(),
                "y_train_std": y_train_std.detach().cpu().tolist(),
                "standardize_X": standardize_X,
                "standardize_y": standardize_y,
                "X_scaling_type": X_scaling_type,
                "x_standardize_method": x_standardize_method,
                "dtype": str(gp_dtype),
                "device": str(gp_device),
                "num_epochs": num_epochs,
                "num_inits": num_inits,
                "lr": lr,
                "optimizer": optimizer_class.__name__,
                "convergence_patience": convergence_patience,
                "initializer": initializer_class.__name__ if initializer_class else None,
                "num_runs": num_runs,
                "seed": seed,
                "seed_trainer": seed_trainer,
                "x_bounds": x_bounds,
            }

    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    GPPlus_summary = analyze_metrics(
        GPPlus_metrics, print_summary=True, label="MTGPR", title=title
    )

    if save_path is not None:
        out_dir = Path(save_path)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            combined_data = {
                "gp_data": {
                    "summary": GPPlus_summary,
                    "metrics": GPPlus_metrics,
                    "gp_model_info": gp_model_info,
                },
            }
            _defaults_path = Path(__file__).resolve().parent / "defaults.py"
            if _defaults_path.is_file():
                combined_data["defaults_py"] = _defaults_path.read_text(encoding="utf-8")
            (out_dir / f"gp_{title}.json").write_text(json.dumps(combined_data, indent=2))
        except Exception:
            pass

        if trainer_info and GPTrainer_info:
            try:
                trainer_analysis_dir = Path(save_path) / "trainer_analysis"
                trainer_analysis_dir.mkdir(parents=True, exist_ok=True)
                trainer_info_by_run = {
                    f"run_{entry.get('run', j + 1)}": entry
                    for j, entry in enumerate(GPTrainer_info)
                }
                trainer_info_data = {
                    "title": title,
                    "num_runs": num_runs,
                    "num_inits_per_run": num_inits,
                    "trainer_info": trainer_info_by_run,
                }
                trainer_info_file = (
                    trainer_analysis_dir / f"gp_{title}_GP_Trainer_Analysis.json"
                )
                trainer_info_file.write_text(json.dumps(trainer_info_data, indent=2))
                print(f"\nTrainer info saved to: {trainer_info_file}")
            except Exception as e:
                print(f"Error saving trainer info: {e}")

    print(f"\nTotal experiment time for {num_runs} runs: {time.time() - total_start_time:.2f}s")
    print("=" * 60)

    return GPPlus_metrics


if __name__ == "__main__":
    mt_analytic_1d_GP(
        num_runs=2,
        n_train=10,
        n_test=500,
        x_bounds=[-4.0, 4.0],
        num_inits=16,
        num_epochs=1,
        save_path="results/MT1_analytic_1d",
        single_dataset=True,
        noise_train=0.005,
        noise_test=0.005,
        x_standardize_method=defaults.X_STANDARDIZE_METHOD,
        seed=42,
        plot_prediction_diagnostics=True,
        diagnostic_run_indices=(0,1),
    )
