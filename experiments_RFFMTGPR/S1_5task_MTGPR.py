"""Train and evaluate RFFMTGPR on the 20-D / 5-task synthetic problem.

Reports the usual per-task metrics (RMSE, RRMSE, R2, MAE, NLPD, CRPS, NIS, …)
on ``n_test`` points (default 5000) and saves ``gp_*.json`` under results/.

By default runs **both** ``chol`` and ``eigen`` and prints a side-by-side
cost/accuracy comparison (use ``--no-compare --method …`` for a single run).

Example::

  set PYTHONPATH=%CD%
  python experiments_RFFMTGPR/S1_5task_MTGPR.py --device cuda --n-test 5000
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
_MT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_MT) not in sys.path:
    sys.path.insert(0, str(_MT))

from gpplus.models.rff_mtgpr import RFFMTGPR
from gpplus.training import (
    ConvergencePatienceStopCondition,
    GPTrainer,
    MinLossChangeStopCondition,
    RFFMTWoodburyMarginalLogLikelihood,
    evaluate_rff_mt_gp_model,
)
from gpplus.utils import StandardScaler, UniformScaler, compute_metrics, set_seed
from gpplus.utils.rff_utils import WoodburyMtMethod
from experiments_RFF.rff_gp_defaults import woodbury_jitter_for_dtype

from mtgpr_experiment_utils import (
    DEFAULT_ADAM_KWARGS,
    DEFAULT_LBFGS_KWARGS,
    compute_relative_error_metrics,
    format_relative_error_summary,
    json_safe_optimizer_kwargs,
    save_metrics_json,
)
from synthetic_5task_data import (
    INPUT_DIM,
    NUM_TASKS,
    TASK_NAMES,
    generate_5task_20d_data,
)

logger = logging.getLogger(__name__)


def _mll_class_for_method(method: WoodburyMtMethod):
    """Bind Woodbury factorization method into the MLL class GPTrainer constructs."""

    class BoundRFFMTWoodburyMLL(RFFMTWoodburyMarginalLogLikelihood):
        def __init__(self, likelihood, model, jitter: float = 1e-6):
            super().__init__(likelihood, model, jitter=jitter, method=method)

    BoundRFFMTWoodburyMLL.__name__ = f"RFFMTWoodburyMarginalLogLikelihood_{method}"
    BoundRFFMTWoodburyMLL.__qualname__ = BoundRFFMTWoodburyMLL.__name__
    return BoundRFFMTWoodburyMLL


def _inverse_standardize_predictions(
    y_scaler: StandardScaler,
    mean: torch.Tensor,
    std: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map scaled predictive mean/std/bounds back to original Y scale (affine)."""
    mean_np = y_scaler.inverse_transform(mean.detach().cpu()).numpy()
    # std scales by per-task sigma (no shift); StandardScaler stores ``std`` as (1, T)
    scale = y_scaler.std.detach().cpu().numpy().reshape(1, -1)
    std_np = std.detach().cpu().numpy() * scale
    lower_np = y_scaler.inverse_transform(lower.detach().cpu()).numpy()
    upper_np = y_scaler.inverse_transform(upper.detach().cpu()).numpy()
    return (
        torch.as_tensor(mean_np, dtype=torch.float64),
        torch.as_tensor(std_np, dtype=torch.float64),
        torch.as_tensor(lower_np, dtype=torch.float64),
        torch.as_tensor(upper_np, dtype=torch.float64),
    )


def print_chol_vs_eigen_comparison(chol_m: dict, eigen_m: dict) -> dict:
    """Print and return a summary table comparing cost and accuracy."""
    train_chol = float(chol_m["Training_Time"])
    train_eigen = float(eigen_m["Training_Time"])
    pred_chol = float(chol_m["Prediction_Time"])
    pred_eigen = float(eigen_m["Prediction_Time"])
    total_chol = float(chol_m["Total_Time"])
    total_eigen = float(eigen_m["Total_Time"])
    rrmse_chol = float(chol_m["aggregate_RRMSE"])
    rrmse_eigen = float(eigen_m["aggregate_RRMSE"])
    loss_chol = float(chol_m["best_train_loss"])
    loss_eigen = float(eigen_m["best_train_loss"])

    print("\n" + "=" * 72)
    print("Chol vs Eigen comparison (same seed / data / hyperparameters)")
    print("=" * 72)
    print(f"{'metric':24} {'chol':>14} {'eigen':>14} {'eigen/chol':>12}")
    print("-" * 72)

    def row(label: str, a: float, b: float, *, ratio_kind: str = "speed"):
        if ratio_kind == "speed":
            ratio = a / max(b, 1e-12)
            ratio_s = f"{ratio:.2f}x"
        else:
            ratio = b - a
            ratio_s = f"{ratio:+.4g}"
        print(f"{label:24} {a:14.6g} {b:14.6g} {ratio_s:>12}")

    row("Training_Time (s)", train_chol, train_eigen)
    row("Prediction_Time (s)", pred_chol, pred_eigen)
    row("Total_Time (s)", total_chol, total_eigen)
    row("best_train_loss", loss_chol, loss_eigen, ratio_kind="diff")
    row("aggregate_RRMSE", rrmse_chol, rrmse_eigen, ratio_kind="diff")
    print("-" * 72)
    for name in TASK_NAMES:
        c = float(chol_m[f"{name}_RRMSE"])
        e = float(eigen_m[f"{name}_RRMSE"])
        row(f"{name}_RRMSE", c, e, ratio_kind="diff")
        c_r = float(chol_m[f"{name}_RMSE"])
        e_r = float(eigen_m[f"{name}_RMSE"])
        row(f"{name}_RMSE", c_r, e_r, ratio_kind="diff")
    print("=" * 72)
    print(
        f"Train speedup (chol/eigen): {train_chol / max(train_eigen, 1e-12):.2f}x   "
        f"|aggregate_RRMSE(chol-eigen)|={abs(rrmse_chol - rrmse_eigen):.3e}"
    )

    return {
        "Training_Time_chol": train_chol,
        "Training_Time_eigen": train_eigen,
        "Prediction_Time_chol": pred_chol,
        "Prediction_Time_eigen": pred_eigen,
        "Total_Time_chol": total_chol,
        "Total_Time_eigen": total_eigen,
        "train_speedup_chol_over_eigen": train_chol / max(train_eigen, 1e-12),
        "pred_speedup_chol_over_eigen": pred_chol / max(pred_eigen, 1e-12),
        "aggregate_RRMSE_chol": rrmse_chol,
        "aggregate_RRMSE_eigen": rrmse_eigen,
        "aggregate_RRMSE_abs_diff": abs(rrmse_chol - rrmse_eigen),
        "best_train_loss_chol": loss_chol,
        "best_train_loss_eigen": loss_eigen,
        "per_task_RRMSE_chol": {n: float(chol_m[f"{n}_RRMSE"]) for n in TASK_NAMES},
        "per_task_RRMSE_eigen": {n: float(eigen_m[f"{n}_RRMSE"]) for n in TASK_NAMES},
    }


def compare_chol_vs_eigen(**kwargs) -> dict:
    """Train+eval both Woodbury methods and write a comparison JSON."""
    save_path = kwargs.get("save_path")
    if save_path is None:
        rff_sampling = kwargs.get("rff_sampling", "sorf")
        save_path = str(_MT / "results" / f"synthetic_5task_{rff_sampling}")
        kwargs["save_path"] = save_path

    print("\n>>> Running method=chol")
    chol_m = run_5task_mtgpr(method="chol", **kwargs)
    print("\n>>> Running method=eigen")
    eigen_m = run_5task_mtgpr(method="eigen", **kwargs)

    summary = print_chol_vs_eigen_comparison(chol_m, eigen_m)
    summary.update(
        {
            "n_train": chol_m["n_train"],
            "n_test": chol_m["n_test"],
            "num_rff": chol_m["num_rff"],
            "rff_sampling": chol_m["rff_sampling"],
            "seed": kwargs.get("seed", 42),
            "chol_title": chol_m["title"],
            "eigen_title": eigen_m["title"],
        }
    )
    out = save_metrics_json(
        summary,
        save_path,
        f"compare_chol_vs_eigen_nTrain{chol_m['n_train']}_nTest{chol_m['n_test']}_"
        f"{chol_m['rff_sampling']}D{chol_m['num_rff']}",
    )
    print(f"Saved comparison summary to {out}")
    return {"chol": chol_m, "eigen": eigen_m, "comparison": summary}


def run_5task_mtgpr(
    *,
    n_train: int = 4000,
    n_test: int = 5000,
    num_rff: int = 400,
    rff_sampling: str = "sorf",
    method: WoodburyMtMethod = "eigen",
    num_epochs: int = 2000,
    num_inits: int = 1,
    lr: float = 0.1,
    seed: int = 42,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    predict_chunk_size: int = 512,
    ard: bool = True,
    train_noise: float = 0.01,
    save_path: str | None = None,
    rel_tolerance: float = 0.01,
) -> dict:
    set_seed(seed)
    device_t = torch.device(device)
    if save_path is None:
        save_path = str(_MT / "results" / f"synthetic_5task_{rff_sampling}")

    X_tr, Y_tr, X_te, Y_te = generate_5task_20d_data(
        n_train,
        n_test,
        seed=seed,
        train_noise=train_noise,
        test_noise=0.0,
    )

    x_scaler = UniformScaler(scale_to_neg_one=True)
    x_scaler.fit(X_tr)
    x_train = x_scaler.transform(X_tr).to(device=device_t, dtype=dtype)
    x_test = x_scaler.transform(X_te).to(device=device_t, dtype=dtype)

    y_scaler = StandardScaler()
    y_scaler.fit(Y_tr)
    y_train = y_scaler.transform(Y_tr).to(device=device_t, dtype=dtype)
    y_test_raw = Y_te.to(dtype=torch.float64)

    title = (
        f"5task20d_nTrain{n_train}_nTest{n_test}_"
        f"{rff_sampling}D{num_rff}_{method}_mt"
    )
    print(
        f"Training RFFMTGPR on 20-D / 5-task synthetic: "
        f"n_train={n_train} n_test={n_test} D={num_rff} method={method}"
    )
    print(f"tasks={TASK_NAMES}")
    print(f"raw Y train std={[round(float(s), 3) for s in Y_tr.std(0)]}")

    model = RFFMTGPR(
        x_train,
        y_train,
        num_tasks=NUM_TASKS,
        num_rff=num_rff,
        ard=ard,
        rff_sampling=rff_sampling,  # type: ignore[arg-type]
        rank_kernel=1,
        rank_likelihood=0,
    )

    if num_epochs <= 1:
        from gpplus.training.optimizers import LBFGSScipy

        optimizer_class = LBFGSScipy
        optimizer_kwargs = dict(DEFAULT_LBFGS_KWARGS)
        epochs = 1
    else:
        optimizer_class = torch.optim.Adam
        optimizer_kwargs = dict(DEFAULT_ADAM_KWARGS)
        optimizer_kwargs["lr"] = lr
        epochs = num_epochs

    trainer = GPTrainer(
        model,
        mll_class=_mll_class_for_method(method),
        num_epochs=epochs,
        num_inits=num_inits,
        seed=seed,
        device=device,
        dtype=dtype,
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
        n_jobs=1,
        inner_max_num_threads=1,
        cholesky_jitter=woodbury_jitter_for_dtype(dtype),
        stop_conditions=[
            ConvergencePatienceStopCondition(patience=10),
            MinLossChangeStopCondition(min_loss_change=1e-7),
        ],
    )

    t_train = time.time()
    runs = trainer.train()
    train_time = time.time() - t_train

    successful = [r for r in runs if r.get("loss") is not None and r.get("state_dict") is not None]
    if not successful:
        raise RuntimeError("All training runs failed for 5-task model.")
    best_run = min(successful, key=lambda r: r["loss"])
    model.load_state_dict(best_run["state_dict"])
    best_loss = float(best_run["loss"])

    model.eval()
    model.invalidate_feature_cache()
    t_pred = time.time()
    pred_mean_s, lower_s, upper_s, pred_std_s = evaluate_rff_mt_gp_model(
        model,
        x_test,
        chunk_size=predict_chunk_size,
        method=method,
    )
    prediction_time = time.time() - t_pred

    pred_mean, pred_std, lower, upper = _inverse_standardize_predictions(
        y_scaler, pred_mean_s, pred_std_s, lower_s, upper_s
    )
    y_true = y_test_raw.numpy()
    y_pred = pred_mean.numpy()
    pred_std_np = pred_std.numpy()
    lower_np = lower.numpy()
    upper_np = upper.numpy()

    per_task: dict[str, float | int] = {}
    rel_by_task: dict[str, dict] = {}
    prob_metrics: dict[str, float] = {}
    time_per_task = train_time / NUM_TASKS
    pred_time_per_task = prediction_time / NUM_TASKS

    for t, name in enumerate(TASK_NAMES):
        yt = y_true[:, t]
        yp = y_pred[:, t]
        rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
        std = float(np.std(yt))
        rrmse = rmse / std if std > 0 else float("inf")
        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        per_task[f"{name}_RMSE"] = rmse
        per_task[f"{name}_RRMSE"] = rrmse
        per_task[f"{name}_R2"] = r2

        rel_m = compute_relative_error_metrics(yt, yp, rel_tolerance=rel_tolerance)
        rel_by_task[name] = rel_m
        per_task[f"{name}_max_rel_error"] = float(rel_m["max_rel_error"])
        per_task[f"{name}_mean_rel_error"] = float(rel_m["mean_rel_error"])
        per_task[f"{name}_pct_within_1pct"] = float(rel_m["pct_within_1pct"])
        per_task[f"{name}_n_rel_error_valid"] = int(rel_m["n_rel_error_valid"])
        per_task[f"{name}_n_rel_error_excluded"] = int(rel_m["n_rel_error_excluded"])

        computed = compute_metrics(
            torch.as_tensor(yt),
            torch.as_tensor(yp),
            output_std=torch.as_tensor(pred_std_np[:, t]),
            lower_95=torch.as_tensor(lower_np[:, t]),
            upper_95=torch.as_tensor(upper_np[:, t]),
            training_time=time_per_task,
            prediction_time=pred_time_per_task,
        )
        for key, value in computed.items():
            prob_metrics[f"{name}_{key}"] = value

    aggregate_rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    aggregate_rrmse = float(np.mean([per_task[f"{name}_RRMSE"] for name in TASK_NAMES]))

    print(f"\nTest aggregate RMSE: {aggregate_rmse:.6f}")
    print(f"Test aggregate RRMSE: {aggregate_rrmse:.6f}")
    print(f"Predict time: {prediction_time:.1f}s")
    for name in TASK_NAMES:
        print(
            f"{name} RMSE: {per_task[f'{name}_RMSE']:.6f}  "
            f"RRMSE: {per_task[f'{name}_RRMSE']:.6f}  "
            f"R2: {per_task[f'{name}_R2']:.6f}"
        )
        for key in ("NLPD", "CRPS", "NCRPS", "NIS"):
            k = f"{name}_{key}"
            if k in prob_metrics:
                print(f"  {key}: {prob_metrics[k]:.6f}")
        print(format_relative_error_summary(name, rel_by_task[name], rel_tolerance=rel_tolerance))
    print(f"best loss: {best_loss:.4f}  train time: {train_time:.1f}s  method={method}")

    metrics: dict = {
        "title": title,
        "input_dim": INPUT_DIM,
        "n_train": n_train,
        "n_test": n_test,
        "num_tasks": NUM_TASKS,
        "task_names": list(TASK_NAMES),
        "num_rff": num_rff,
        "rff_sampling": rff_sampling,
        "feature_dim": 2 * num_rff,
        "joint_feature_dim": 2 * num_rff * NUM_TASKS,
        "woodbury_method": method,
        "ard": ard,
        "model_class": "RFFMTGPR",
        "num_epochs": epochs,
        "optimizer": getattr(optimizer_class, "__name__", str(optimizer_class)),
        "optimizer_kwargs": json_safe_optimizer_kwargs(optimizer_kwargs),
        "standardize_x": True,
        "x_scaling_type": "UniformScaler [-1, 1]",
        "standardize_y": True,
        "train_noise": train_noise,
        "best_train_loss": best_loss,
        "Training_Time": train_time,
        "Prediction_Time": prediction_time,
        "Total_Time": train_time + prediction_time,
        "RMSE": aggregate_rmse,
        "aggregate_RRMSE": aggregate_rrmse,
        **per_task,
        **prob_metrics,
    }
    task_noises = model.task_noises().detach().cpu().tolist()
    metrics["task_noises"] = task_noises
    metrics["task_noise_stds"] = [float(np.sqrt(max(v, 0.0))) for v in task_noises]

    json_path = save_metrics_json(metrics, save_path, title)
    print(f"Saved metrics to {json_path}")

    pred_path = Path(save_path) / f"predictions_{title}.npz"
    np.savez_compressed(
        pred_path,
        y_true=y_true,
        y_pred=y_pred,
        pred_std=pred_std_np,
        lower=lower_np,
        upper=upper_np,
        task_names=np.array(TASK_NAMES),
    )
    print(f"Saved predictions to {pred_path}")
    return metrics


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(name)s - %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=10000)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--num-rff", type=int, default=2000)
    parser.add_argument(
        "--rff-sampling",
        type=str,
        default="sorf",
        choices=("rff", "orf", "sorf"),
    )
    parser.add_argument(
        "--method",
        type=str,
        default="eigen",
        choices=("chol", "eigen"),
        help="Used only with --no-compare (single-method run)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        default=False,
        help="Train+eval both chol and eigen and print comparison (default)",
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        dest="compare",
        help="Run only --method",
    )
    parser.add_argument("--num-epochs", type=int, default=2000)
    parser.add_argument("--num-inits", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32", choices=("float32", "float64"))
    parser.add_argument("--predict-chunk-size", type=int, default=512)
    parser.add_argument("--train-noise", type=float, default=0.01)
    parser.add_argument("--ard", action="store_true", default=True)
    parser.add_argument("--no-ard", action="store_false", dest="ard")
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--rel-tolerance", type=float, default=0.01)
    args = parser.parse_args()

    common = dict(
        n_train=args.n_train,
        n_test=args.n_test,
        num_rff=args.num_rff,
        rff_sampling=args.rff_sampling,
        num_epochs=args.num_epochs,
        num_inits=args.num_inits,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        dtype=torch.float32 if args.dtype == "float32" else torch.float64,
        predict_chunk_size=args.predict_chunk_size,
        ard=args.ard,
        train_noise=args.train_noise,
        save_path=args.save_path,
        rel_tolerance=args.rel_tolerance,
    )
    if args.compare:
        compare_chol_vs_eigen(**common)
    else:
        run_5task_mtgpr(method=args.method, **common)  # type: ignore[arg-type]
