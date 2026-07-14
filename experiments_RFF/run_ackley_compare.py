"""
Matched RFF / ORF / SORF Ackley comparison (same n, D, seed, noise, inits, ARD).

Defaults: Adam on CUDA (float32), matching TOA-style training rather than LBFGS/CPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_RFF = _ROOT / "experiments_RFF"
_ORF = _ROOT / "experiments_ORF"
_SORF = _ROOT / "experiments_SORF"
for p in (_ROOT, _RFF, _ORF, _SORF):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import gpplus
import torch
from A4_ackley_RFF import run_ackley_40d_rff
from A4_ackley_ORF import run_ackley_40d_orf
from A4_ackley_SORF import run_ackley_40d_sorf
from rff_experiment_utils import DEFAULT_ADAM_KWARGS


def _summary_row(method: str, metrics: dict, *, train_size: int, num_features: int, seed: int) -> dict:
    d = metrics.get("num_rff") or metrics.get("num_orf") or metrics.get("num_sorf")
    if d is None and metrics.get("feature_dim") is not None:
        d = int(metrics["feature_dim"]) // 2
    return {
        "method": method,
        "dimensions": metrics.get("dimensions"),
        "train_size": train_size,
        "n_train": metrics.get("n_train"),
        "num_features_D": d if d is not None else num_features,
        "RRMSE": metrics.get("RRMSE"),
        "RMSE": metrics.get("RMSE"),
        "NIS": metrics.get("NIS"),
        "best_train_loss": metrics.get("best_train_loss"),
        "Total_Time": metrics.get("Total_Time"),
        "noise": metrics.get("likelihood_noise")
        or metrics.get("learned_noise")
        or metrics.get("noise"),
        "ard": metrics.get("ard"),
        "seed": seed,
        "optimizer": metrics.get("optimizer"),
        "num_epochs": metrics.get("num_epochs"),
        "device": metrics.get("device"),
    }


def run_one(
    method: str,
    *,
    dimensions: int,
    train_size: int,
    num_features: int,
    noise: float,
    seed: int,
    num_inits: int,
    num_epochs: int,
    lr: float,
    device: str,
    dtype,
    n_jobs,
    save_root: Path,
    log_every_n_epochs: int = 1,
    correct_sorf: bool = True,
) -> dict:
    optimizer_kwargs = None
    if num_epochs > 1:
        optimizer_kwargs = {**DEFAULT_ADAM_KWARGS, "lr": lr}
    common = dict(
        dimensions=dimensions,
        train_size=train_size,
        num_test=5000,
        noise_train=noise,
        noise_test=noise,
        seed=seed,
        num_inits=num_inits,
        num_epochs=num_epochs,
        optimizer_kwargs=optimizer_kwargs,
        device=device,
        dtype=dtype,
        ard=True,
        n_jobs=n_jobs,
        plot_validation=False,
        monitor_validation=True,
        log_every_n_epochs=log_every_n_epochs,
    )
    sorf_tag = f"_correctSorf{correct_sorf}" if method == "sorf" else ""
    tag = (
        f"ackley_{dimensions}D_n{train_size}_D{num_features}_seed{seed}"
        f"_adam{num_epochs}ep_lr{lr}{sorf_tag}"
    )
    if method == "rff":
        return run_ackley_40d_rff(
            num_rff=num_features,
            save_path=str(save_root / "rff" / tag),
            **common,
        )
    if method == "orf":
        return run_ackley_40d_orf(
            num_orf=num_features,
            save_path=str(save_root / "orf" / tag),
            **common,
        )
    if method == "sorf":
        return run_ackley_40d_sorf(
            num_sorf=num_features,
            save_path=str(save_root / "sorf" / tag),
            correct_sorf=correct_sorf,
            **common,
        )
    raise ValueError(method)


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched Ackley RFF/ORF/SORF compare (Adam+GPU)")
    parser.add_argument("--dimensions", type=int, default=10)
    parser.add_argument(
        "--train-sizes",
        type=str,
        default="40",
        help="Comma-separated train points per dimension",
    )
    parser.add_argument(
        "--num-features",
        type=str,
        default="50,100,200,400,800,1600",
        help="Comma-separated D values (frequencies)",
    )
    parser.add_argument("--noise", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inits", type=int, default=16)
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=5000,
        help="Adam epochs when >1 (default 200). Use 1 for LBFGSScipy.",
    )
    parser.add_argument("--lr", type=float, default=1.0, help="Adam learning rate")
    parser.add_argument(
        "--log-every-n-epochs",
        type=int,
        default=50,
        help="Print validation metrics every N Adam epochs (default: 1)",
    )
    parser.add_argument(
        "--correct-sorf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "SORF only: use true FWHT (default) vs legacy aliased FWHT "
            "(--no-correct-sorf, TOA-compatible)"
        ),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32", choices=("float32", "float64"))
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel inits (default 1 for CUDA; use -1 only on CPU)",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default="experiments_RFF/results/ackley_compare_adam_gpu_40Dn",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="rff,orf,sorf",
        help="Comma-separated subset of rff,orf,sorf",
    )
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but torch.cuda.is_available() is False. "
            "Use a CUDA-enabled env (e.g. gpplus_tydev / gpplus2026)."
        )

    gpplus.config.configure_logger()

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    n_jobs = None if args.n_jobs < 0 else args.n_jobs
    train_sizes = [int(x) for x in args.train_sizes.split(",") if x.strip()]
    feature_counts = [int(x) for x in args.num_features.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for train_size in train_sizes:
        for D in feature_counts:
            for method in methods:
                print("\n" + "#" * 72)
                print(
                    f"# {method.upper()} | d={args.dimensions} | n/dim={train_size} | "
                    f"D={D} | Adam ep={args.num_epochs} lr={args.lr} | {args.device}"
                    + (f" | correct_sorf={args.correct_sorf}" if method == "sorf" else "")
                )
                print("#" * 72)
                metrics = run_one(
                    method,
                    dimensions=args.dimensions,
                    train_size=train_size,
                    num_features=D,
                    noise=args.noise,
                    seed=args.seed,
                    num_inits=args.num_inits,
                    num_epochs=args.num_epochs,
                    lr=args.lr,
                    device=args.device,
                    dtype=dtype,
                    n_jobs=n_jobs,
                    save_root=save_root,
                    log_every_n_epochs=args.log_every_n_epochs,
                    correct_sorf=args.correct_sorf,
                )
                row = _summary_row(
                    method,
                    metrics,
                    train_size=train_size,
                    num_features=D,
                    seed=args.seed,
                )
                row["device"] = args.device
                row["num_epochs"] = args.num_epochs
                row["lr"] = args.lr
                if method == "sorf":
                    row["correct_sorf"] = args.correct_sorf
                rows.append(row)

    summary_path = save_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print("\n" + "=" * 72)
    print(f"Wrote {summary_path}")
    print(f"{'method':<6} {'n/dim':>5} {'D':>4} {'RRMSE':>10} {'RMSE':>10} {'NIS':>10}")
    for r in rows:
        print(
            f"{r['method']:<6} {r['train_size']:>5} {r['num_features_D']:>4} "
            f"{float(r['RRMSE']):>10.6f} {float(r['RMSE']):>10.6f} {float(r['NIS']):>10.6f}"
        )


if __name__ == "__main__":
    main()
