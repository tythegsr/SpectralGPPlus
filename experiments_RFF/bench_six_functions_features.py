"""
Single-task RFF / ORF / SORF sweep across six benchmark functions.

Grid (full defaults):
  - Functions: wing, ackley, rosenbrock, griewank, borehole, dixon_price
  - Dims: 10 and 20 for ackley/rosenbrock/griewank/dixon_price;
          wing fixed at 10; borehole fixed at 8
  - Methods: rff, orf, sorf
  - D (frequencies): 50, 100, 200, 400, 800
  - Train sizes: points per input dimension (n_train = train_size * d)

Wing and Borehole use single-fidelity s0 only (source column dropped).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
_RFF_DIR = Path(__file__).resolve().parent
for p in (_ROOT, _RFF_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import gpplus
from gpplus.models import RFFGPR
from gpplus.training import (
    GPTrainer,
    RFFParameterInitializer,
    RFFWoodburyMarginalLogLikelihood,
    evaluate_rff_gp_model,
)
from gpplus.training.optimizers import LBFGSScipy
from gpplus.utils import StandardScaler, UniformScaler, compute_metrics, set_seed
from load_experimental_data import (
    generate_ackley_data,
    generate_dixon_price_data,
    generate_griewank_data,
    generate_mf_borehole_data,
    generate_mf_wing_data,
    generate_rosenbrock_data,
)
from rff_experiment_utils import (
    DEFAULT_ADAM_KWARGS,
    DEFAULT_LBFGS_KWARGS,
    extract_learned_likelihood_noise,
    json_default,
    json_safe_optimizer_kwargs,
    save_metrics_json,
    unpack_train_val_test,
)

ANALYTIC_FUNCTIONS = ("ackley", "rosenbrock", "griewank", "dixon_price")
ALL_FUNCTIONS = ("wing", "ackley", "rosenbrock", "griewank", "borehole", "dixon_price")
ALL_METHODS = ("rff", "orf", "sorf")
DEFAULT_DIMS = (10, 20)
DEFAULT_NUM_FEATURES = (50, 100, 200, 400, 800)
DEFAULT_TRAIN_SIZES = (10,40)  # points per input dim; n_train = train_size * d

FUNCTION_BOUNDS = {
    "ackley": (-5.0, 10.0),
    "rosenbrock": (-5.0, 10.0),
    "griewank": (-600.0, 600.0),
    "dixon_price": (-10.0, 10.0),
}

WING_CONT_DIM = 10
BOREHOLE_CONT_DIM = 8
BOREHOLE_NUM_SOURCES = 5

FIXED_DIMS = {
    "wing": WING_CONT_DIM,
    "borehole": BOREHOLE_CONT_DIM,
}


def _allowed_dims(function: str, requested_dims: list[int]) -> list[int]:
    """Fixed-dim problems always use their native dim; --dims only filters analytic problems."""
    fixed = FIXED_DIMS.get(function)
    if fixed is not None:
        return [fixed]
    return list(requested_dims)


def _load_analytic_data(
    function: str,
    *,
    n_train: int,
    n_test: int,
    dimensions: int,
    noise_train: float,
    noise_test: float,
    noise_type: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bounds = list(FUNCTION_BOUNDS[function])
    common = dict(
        n_train=n_train,
        n_test=n_test,
        dimensions=dimensions,
        x_bounds=bounds,
        train_noise=noise_train,
        test_noise=noise_test,
        noise_type=noise_type,
        seed=seed,
    )
    if function == "ackley":
        data = generate_ackley_data(**common, n_val=0)
    elif function == "rosenbrock":
        data = generate_rosenbrock_data(**common)
    elif function == "griewank":
        data = generate_griewank_data(**common)
    elif function == "dixon_price":
        data = generate_dixon_price_data(**common)
    else:
        raise ValueError(f"Unknown analytic function: {function}")
    x_train, y_train, _, _, x_test, y_test = unpack_train_val_test(data)
    return x_train, y_train, x_test, y_test


def _load_wing_s0(
    *,
    n_train: int,
    n_test: int,
    noise_train: float,
    noise_test: float,
    noise_type: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    train_per = [n_train, 0, 0, 0]
    test_per = [n_test, 0, 0, 0]
    data = generate_mf_wing_data(
        train_samples_per_source=train_per,
        test_samples_per_source=test_per,
        seed=seed,
        train_noise=noise_train,
        test_noise=noise_test,
        noise_type=noise_type,
    )
    x_train, y_train, _, _, x_test, y_test = unpack_train_val_test(data)
    x_train = x_train[:, :WING_CONT_DIM].contiguous()
    x_test = x_test[:, :WING_CONT_DIM].contiguous()
    return x_train, y_train, x_test, y_test, WING_CONT_DIM


def _load_borehole_s0(
    *,
    n_train: int,
    n_test: int,
    noise_train: float,
    noise_test: float,
    noise_type: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    train_per = [n_train] + [0] * (BOREHOLE_NUM_SOURCES - 1)
    test_per = [n_test] + [0] * (BOREHOLE_NUM_SOURCES - 1)
    data = generate_mf_borehole_data(
        train_samples_per_source=train_per,
        test_samples_per_source=test_per,
        seed=seed,
        train_noise=noise_train,
        test_noise=noise_test,
        noise_type=noise_type,
    )
    x_train, y_train, _, _, x_test, y_test = unpack_train_val_test(data)
    x_train = x_train[:, :BOREHOLE_CONT_DIM].contiguous()
    x_test = x_test[:, :BOREHOLE_CONT_DIM].contiguous()
    return x_train, y_train, x_test, y_test, BOREHOLE_CONT_DIM


def load_function_data(
    function: str,
    *,
    dimensions: int,
    train_size: int,
    num_test: int,
    noise_train: float,
    noise_test: float,
    noise_type: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Return x_train, y_train, x_test, y_test, input_dim, n_train."""
    n_train = train_size * dimensions
    if function in ANALYTIC_FUNCTIONS:
        x_train, y_train, x_test, y_test = _load_analytic_data(
            function,
            n_train=n_train,
            n_test=num_test,
            dimensions=dimensions,
            noise_train=noise_train,
            noise_test=noise_test,
            noise_type=noise_type,
            seed=seed,
        )
        return x_train, y_train, x_test, y_test, dimensions, n_train
    if function == "wing":
        if dimensions != WING_CONT_DIM:
            raise ValueError(f"Wing is fixed at {WING_CONT_DIM}D, got dimensions={dimensions}")
        x_train, y_train, x_test, y_test, dim = _load_wing_s0(
            n_train=n_train,
            n_test=num_test,
            noise_train=noise_train,
            noise_test=noise_test,
            noise_type=noise_type,
            seed=seed,
        )
        return x_train, y_train, x_test, y_test, dim, n_train
    if function == "borehole":
        if dimensions != BOREHOLE_CONT_DIM:
            raise ValueError(
                f"Borehole is fixed at {BOREHOLE_CONT_DIM}D, got dimensions={dimensions}"
            )
        x_train, y_train, x_test, y_test, dim = _load_borehole_s0(
            n_train=n_train,
            n_test=num_test,
            noise_train=noise_train,
            noise_test=noise_test,
            noise_type=noise_type,
            seed=seed,
        )
        return x_train, y_train, x_test, y_test, dim, n_train
    raise ValueError(f"Unknown function: {function}")


def run_case(
    function: str,
    *,
    dimensions: int,
    method: str,
    num_rff: int,
    train_size: int = 40,
    num_test: int = 5000,
    noise_train: float = 0.005,
    noise_test: float = 0.005,
    noise_type: str = "gaussian",
    seed: int = 42,
    num_inits: int = 8,
    num_epochs: int = 1,
    lr: float = 1.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    ard: bool = True,
    correct_sorf: bool = True,
    standardize_x: bool = True,
    x_standardize_method: int = 2,
    standardize_y: bool = True,
    predict_chunk_size: int = 512,
    n_jobs: int | None = None,
    save_path: str | None = None,
) -> dict:
    if method not in ALL_METHODS:
        raise ValueError(f"method must be one of {ALL_METHODS}, got {method!r}")

    set_seed(seed)

    if num_epochs <= 1:
        optimizer_class = LBFGSScipy
        optimizer_kwargs = dict(DEFAULT_LBFGS_KWARGS)
    else:
        optimizer_class = torch.optim.Adam
        optimizer_kwargs = {**DEFAULT_ADAM_KWARGS, "lr": lr}

    x_train, y_train, x_test, y_test, input_dim, n_train = load_function_data(
        function,
        dimensions=dimensions,
        train_size=train_size,
        num_test=num_test,
        noise_train=noise_train,
        noise_test=noise_test,
        noise_type=noise_type,
        seed=seed,
    )

    x_train = x_train.to(dtype=dtype)
    x_test = x_test.to(dtype=dtype)
    y_train = y_train.to(dtype=dtype)
    y_test = y_test.to(dtype=dtype)

    sorf_tag = f"_correctSorf{correct_sorf}" if method == "sorf" else ""
    title = (
        f"{function}_{input_dim}Dx_{train_size}Dn_{method}D{num_rff}"
        f"_noiseTest{noise_test}_noiseTrain{noise_train}{sorf_tag}"
    )
    feature_dim = 2 * num_rff
    print("=" * 60)
    print(title)
    print(
        f"{method.upper()} (Woodbury), D={num_rff}, m={feature_dim}, ARD={ard}, "
        f"input_dim={input_dim}, dtype={dtype}, inits={num_inits}, epochs={num_epochs}"
    )
    opt_name = getattr(optimizer_class, "__name__", str(optimizer_class))
    print(f"Optimizer: {opt_name}, kwargs={optimizer_kwargs}")
    print(f"Woodbury: n_train={n_train}, m/n={feature_dim / n_train:.4f}")
    if feature_dim >= n_train:
        print(
            f"WARNING: m={feature_dim} >= n_train={n_train}; Woodbury may not beat dense GP. "
            f"Consider num_rff <= {max(1, n_train // 2 - 1)}."
        )
    print("=" * 60)

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
    if standardize_y:
        y_scaler = StandardScaler()
        y_scaler.fit(y_train.unsqueeze(-1))
        y_mean, y_std = y_scaler.mean.squeeze(), y_scaler.std.squeeze()
        y_train = y_scaler.transform(y_train.unsqueeze(-1)).squeeze(-1)
    else:
        y_mean = y_std = None

    model_kwargs = dict(
        num_rff=num_rff,
        ard=ard,
        rff_sampling=method,
    )
    if method == "sorf":
        model_kwargs["correct_sorf"] = correct_sorf

    model = RFFGPR(x_train, y_train, **model_kwargs)

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
        callbacks=[],
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
        y_test_eval = y_test.cpu()

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
        "function": function,
        "dimensions": input_dim,
        "train_size": train_size,
        "n_train": n_train,
        "n_test": num_test,
        "method": method,
        "rff_sampling": method,
        "num_rff": num_rff,
        "feature_dim": feature_dim,
        "ard": ard,
        "num_epochs": num_epochs,
        "optimizer": opt_name,
        "optimizer_kwargs": json_safe_optimizer_kwargs(optimizer_kwargs),
        "best_train_loss": best_loss,
        "noise_train": noise_train,
        "noise_test": noise_test,
        "noise_type": noise_type,
        "seed": seed,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "standardize_x": standardize_x,
        "x_standardize_method": x_standardize_method,
        "x_scaling_type": x_scaling_type,
        "standardize_y": standardize_y,
        **learned_noise,
        **computed,
    }
    if method == "sorf":
        metrics["correct_sorf"] = correct_sorf

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

    return metrics


def _summary_row(metrics: dict) -> dict:
    return {
        "function": metrics.get("function"),
        "dimensions": metrics.get("dimensions"),
        "method": metrics.get("method"),
        "num_rff": metrics.get("num_rff"),
        "train_size": metrics.get("train_size"),
        "n_train": metrics.get("n_train"),
        "RRMSE": metrics.get("RRMSE"),
        "RMSE": metrics.get("RMSE"),
        "MAE": metrics.get("MAE"),
        "NIS": metrics.get("NIS"),
        "CRPS": metrics.get("CRPS"),
        "NCRPS": metrics.get("NCRPS"),
        "NLPD": metrics.get("NLPD"),
        "best_train_loss": metrics.get("best_train_loss"),
        "Training_Time": metrics.get("Training_Time"),
        "Prediction_Time": metrics.get("Prediction_Time"),
        "Total_Time": metrics.get("Total_Time"),
        "seed": metrics.get("seed"),
        "device": metrics.get("device"),
        "num_epochs": metrics.get("num_epochs"),
        "correct_sorf": metrics.get("correct_sorf"),
    }


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_csv_strs(raw: str) -> list[str]:
    return [x.strip().lower() for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Six-function single-task RFF/ORF/SORF feature sweep",
    )
    parser.add_argument(
        "--functions",
        type=str,
        default=",".join(ALL_FUNCTIONS),
        help=f"Comma-separated subset of {','.join(ALL_FUNCTIONS)}",
    )
    parser.add_argument(
        "--dims",
        type=str,
        default=",".join(str(d) for d in DEFAULT_DIMS),
        help="Comma-separated dims for modifiable problems (wing=10, borehole=8 always)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(ALL_METHODS),
        help="Comma-separated subset of rff,orf,sorf",
    )
    parser.add_argument(
        "--num-features",
        type=str,
        default=",".join(str(d) for d in DEFAULT_NUM_FEATURES),
        help="Comma-separated D values (frequencies)",
    )
    parser.add_argument(
        "--train-sizes",
        type=str,
        default=",".join(str(n) for n in DEFAULT_TRAIN_SIZES),
        help="Comma-separated train points per input dimension (n_train = train_size * d)",
    )
    parser.add_argument("--num-test", type=int, default=5000)
    parser.add_argument("--noise", type=float, default=0.005, help="Train and test noise fraction")
    parser.add_argument("--noise-type", type=str, default="gaussian", choices=("gaussian", "uniform"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inits", type=int, default=4)
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=2000,
        help="1 -> LBFGSScipy; >1 -> Adam",
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Adam learning rate when num-epochs > 1")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=("float32", "float64"),
        help="Default: float32 on cuda, float64 on cpu",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel inits (default 1; use -1 for all cores on CPU)",
    )
    parser.add_argument(
        "--ard",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--correct-sorf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="SORF only: true FWHT (default) vs legacy aliased FWHT",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default="experiments_RFF/results/July14/six_fn_feature_sweep",
    )
    parser.add_argument(
        "--predict-chunk-size",
        type=int,
        default=512,
    )
    args = parser.parse_args()

    functions = _parse_csv_strs(args.functions)
    for fn in functions:
        if fn not in ALL_FUNCTIONS:
            raise ValueError(f"Unknown function {fn!r}; choose from {ALL_FUNCTIONS}")
    methods = _parse_csv_strs(args.methods)
    for m in methods:
        if m not in ALL_METHODS:
            raise ValueError(f"Unknown method {m!r}; choose from {ALL_METHODS}")
    requested_dims = _parse_csv_ints(args.dims)
    feature_counts = _parse_csv_ints(args.num_features)
    train_sizes = _parse_csv_ints(args.train_sizes)

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but torch.cuda.is_available() is False. "
            "Pass --device cpu or use a CUDA-enabled env."
        )

    if args.dtype is None:
        dtype = torch.float32 if device.startswith("cuda") else torch.float64
    else:
        dtype = torch.float32 if args.dtype == "float32" else torch.float64

    n_jobs = None if args.n_jobs < 0 else args.n_jobs
    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    gpplus.config.configure_logger()

    cases: list[tuple[str, int, str, int, int]] = []
    for function in functions:
        dims = _allowed_dims(function, requested_dims)
        for dim in dims:
            for train_size in train_sizes:
                for method in methods:
                    for D in feature_counts:
                        cases.append((function, dim, train_size, method, D))

    print(f"Planned runs: {len(cases)}  device={device}  dtype={dtype}")
    rows: list[dict] = []
    for idx, (function, dim, train_size, method, D) in enumerate(cases, start=1):
        print("\n" + "#" * 72)
        print(
            f"# [{idx}/{len(cases)}] {function} | d={dim} | {method.upper()} | D={D} | "
            f"train_size={train_size} (n={train_size * dim}) | {device}"
        )
        print("#" * 72)
        case_save = str(save_root / f"train_size_{train_size}" / function / method)
        metrics = run_case(
            function,
            dimensions=dim,
            method=method,
            num_rff=D,
            train_size=train_size,
            num_test=args.num_test,
            noise_train=args.noise,
            noise_test=args.noise,
            noise_type=args.noise_type,
            seed=args.seed,
            num_inits=args.num_inits,
            num_epochs=args.num_epochs,
            lr=args.lr,
            device=device,
            dtype=dtype,
            ard=args.ard,
            correct_sorf=args.correct_sorf,
            n_jobs=n_jobs,
            predict_chunk_size=args.predict_chunk_size,
            save_path=case_save,
        )
        rows.append(_summary_row(metrics))

        # Checkpoint summary after each run so partial grids are usable.
        summary_path = save_root / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, default=json_default)

    summary_path = save_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=json_default)

    print("\n" + "=" * 72)
    print(f"Wrote {summary_path} ({len(rows)} rows)")
    print(
        f"{'function':<12} {'d':>3} {'n/d':>4} {'meth':<5} {'D':>4} "
        f"{'RRMSE':>10} {'RMSE':>10} {'NIS':>8} {'Time':>8}"
    )
    for r in rows:
        nis = r.get("NIS")
        nis_s = f"{float(nis):8.4f}" if nis is not None else f"{'nan':>8}"
        total = r.get("Total_Time")
        time_s = f"{float(total):8.1f}" if total is not None else f"{'nan':>8}"
        print(
            f"{r['function']:<12} {int(r['dimensions']):>3} {int(r['train_size']):>4} "
            f"{r['method']:<5} {int(r['num_rff']):>4} {float(r['RRMSE']):>10.6f} "
            f"{float(r['RMSE']):>10.6f} {nis_s} {time_s}"
        )


if __name__ == "__main__":
    main()
