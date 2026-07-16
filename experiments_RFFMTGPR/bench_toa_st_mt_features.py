"""
Matched TOA single-task vs multitask RFF / ORF / SORF feature sweep.

Locked defaults (CLI-overridable):
  n_train=16000, seed=42, lr=0.01, epochs=2000, num_inits=4,
  logit_cos=False, log_grain=True, correct_sorf=True,
  train_subset=maximin, D in {100,200,400,800,1600}.

Adam runs use a fixed LR (no scheduler) via the TOA bases.
Early stop: ConvergencePatienceStopCondition(patience=10).

ST uses run_toa_stgp (two independent RFFGPRs).
MT uses run_toa_mtgpr (joint RFFMTGPR) with correct_sorf wired through.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_MT_DIR = Path(__file__).resolve().parent
_RFF_DIR = _ROOT / "experiments_RFF"
for p in (_ROOT, _MT_DIR, _RFF_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import gpplus
import torch
from mtgpr_experiment_utils import DEFAULT_ADAM_KWARGS, json_default
from toa_mtgpr_base import run_toa_mtgpr
from toa_stgp_base import run_toa_stgp

ALL_MODEL_KINDS = ("st", "mt")
ALL_METHODS = ("rff", "orf", "sorf")
DEFAULT_NUM_FEATURES = (100, 200, 400, 800, 1600)


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_csv_strs(raw: str) -> list[str]:
    return [x.strip().lower() for x in raw.split(",") if x.strip()]


def _summary_row(model_kind: str, metrics: dict) -> dict:
    return {
        "model_kind": model_kind,
        "rff_sampling": metrics.get("rff_sampling"),
        "num_rff": metrics.get("num_rff"),
        "n_train": metrics.get("n_train"),
        "n_test": metrics.get("n_test"),
        "seed": metrics.get("seed"),
        "num_epochs": metrics.get("num_epochs"),
        "num_inits": metrics.get("num_inits"),
        "logit_cos": metrics.get("logit_cos"),
        "log_grain": metrics.get("log_grain"),
        "correct_sorf": metrics.get("correct_sorf"),
        "train_subset": metrics.get("train_subset"),
        "y_cos_RRMSE": metrics.get("y_cos_RRMSE"),
        "y_grain_RRMSE": metrics.get("y_grain_RRMSE"),
        "y_cos_RMSE": metrics.get("y_cos_RMSE"),
        "y_grain_RMSE": metrics.get("y_grain_RMSE"),
        "y_cos_NIS": metrics.get("y_cos_NIS"),
        "y_grain_NIS": metrics.get("y_grain_NIS"),
        "y_cos_noise": metrics.get("y_cos_noise"),
        "y_grain_noise": metrics.get("y_grain_noise"),
        "aggregate_RRMSE": metrics.get("aggregate_RRMSE"),
        "aggregate_RRMSE_mean": metrics.get("aggregate_RRMSE_mean"),
        "RMSE": metrics.get("RMSE"),
        "Training_Time": metrics.get("Training_Time"),
        "Prediction_Time": metrics.get("Prediction_Time"),
        "Total_Time": metrics.get("Total_Time"),
        "initial_lr": metrics.get("initial_lr"),
        "final_lr": metrics.get("final_lr"),
        "scheduler": metrics.get("scheduler"),
        "device": metrics.get("device"),
        "title": metrics.get("title"),
    }


def run_one(
    model_kind: str,
    *,
    method: str,
    num_rff: int,
    n_train: int,
    n_test: int,
    seed: int,
    num_inits: int,
    num_epochs: int,
    lr: float,
    logit_cos: bool,
    log_grain: bool,
    correct_sorf: bool,
    train_subset: str,
    device: str,
    dtype: torch.dtype,
    ard: bool,
    n_jobs: int | None,
    save_root: Path,
    log_every_n_epochs: int,
) -> dict:
    optimizer_kwargs = {**DEFAULT_ADAM_KWARGS, "lr": lr}
    case_save = str(save_root / model_kind / method)
    common = dict(
        n_train=n_train,
        n_test=n_test,
        num_rff=num_rff,
        rff_sampling=method,
        seed=seed,
        num_inits=num_inits,
        num_epochs=num_epochs,
        device=device,
        dtype=dtype,
        ard=ard,
        n_jobs=n_jobs,
        optimizer_kwargs=optimizer_kwargs,
        logit_cos=logit_cos,
        log_grain=log_grain,
        correct_sorf=correct_sorf,
        train_subset=train_subset,
        monitor_validation=True,
        validation_verbose=True,
        plot_validation=False,
        plot_posterior=False,
        save_checkpoint=False,
        training_verbose=True,
        log_every_n_epochs=log_every_n_epochs,
        save_path=case_save,
    )
    if model_kind == "st":
        return run_toa_stgp(**common)
    if model_kind == "mt":
        return run_toa_mtgpr(**common)
    raise ValueError(f"Unknown model_kind: {model_kind!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Matched TOA ST/MT RFF/ORF/SORF feature sweep",
    )
    parser.add_argument(
        "--model-kinds",
        type=str,
        default=",".join(ALL_MODEL_KINDS),
        help="Comma-separated subset of st,mt",
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
    parser.add_argument("--n-train", type=int, default=16000)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Adam LR (fixed; no scheduler)",
    )
    parser.add_argument("--num-epochs", type=int, default=2000)
    parser.add_argument("--num-inits", type=int, default=4)
    parser.add_argument(
        "--logit-cos",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--log-grain",
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
        "--train-subset",
        type=str,
        default="maximin",
        choices=("random", "maximin"),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=("float32", "float64"),
    )
    parser.add_argument(
        "--ard",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel inits (default 1 for CUDA)",
    )
    parser.add_argument(
        "--log-every-n-epochs",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default="experiments_RFFMTGPR/results/July14/toa_st_mt_feature_sweep",
    )
    args = parser.parse_args()

    model_kinds = _parse_csv_strs(args.model_kinds)
    for kind in model_kinds:
        if kind not in ALL_MODEL_KINDS:
            raise ValueError(f"Unknown model_kind {kind!r}; choose from {ALL_MODEL_KINDS}")
    methods = _parse_csv_strs(args.methods)
    for method in methods:
        if method not in ALL_METHODS:
            raise ValueError(f"Unknown method {method!r}; choose from {ALL_METHODS}")
    feature_counts = _parse_csv_ints(args.num_features)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but torch.cuda.is_available() is False. "
            "Pass --device cpu or use a CUDA-enabled env."
        )

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    n_jobs = None if args.n_jobs < 0 else args.n_jobs
    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    gpplus.config.configure_logger()

    cases: list[tuple[str, str, int]] = [
        (kind, method, D)
        for kind in model_kinds
        for method in methods
        for D in feature_counts
    ]
    print(
        f"Planned runs: {len(cases)}  device={args.device}  dtype={dtype}  "
        f"n_train={args.n_train}  epochs={args.num_epochs}  lr={args.lr}  "
        f"inits={args.num_inits}  correct_sorf={args.correct_sorf}"
    )

    rows: list[dict] = []
    for idx, (model_kind, method, D) in enumerate(cases, start=1):
        print("\n" + "#" * 72)
        print(
            f"# [{idx}/{len(cases)}] {model_kind.upper()} | {method.upper()} | D={D} | "
            f"n={args.n_train} | ep={args.num_epochs} lr={args.lr} | {args.device}"
            + (f" | correct_sorf={args.correct_sorf}" if method == "sorf" else "")
        )
        print("#" * 72)
        metrics = run_one(
            model_kind,
            method=method,
            num_rff=D,
            n_train=args.n_train,
            n_test=args.n_test,
            seed=args.seed,
            num_inits=args.num_inits,
            num_epochs=args.num_epochs,
            lr=args.lr,
            logit_cos=args.logit_cos,
            log_grain=args.log_grain,
            correct_sorf=args.correct_sorf,
            train_subset=args.train_subset,
            device=args.device,
            dtype=dtype,
            ard=args.ard,
            n_jobs=n_jobs,
            save_root=save_root,
            log_every_n_epochs=args.log_every_n_epochs,
        )
        # Bases do not always stamp these; keep summary self-describing.
        metrics.setdefault("seed", args.seed)
        metrics.setdefault("num_inits", args.num_inits)
        metrics.setdefault("device", args.device)
        rows.append(_summary_row(model_kind, metrics))

        summary_path = save_root / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, default=json_default)

    summary_path = save_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=json_default)

    print("\n" + "=" * 72)
    print(f"Wrote {summary_path} ({len(rows)} rows)")
    print(
        f"{'kind':<3} {'meth':<5} {'D':>4} "
        f"{'cosRRMSE':>12} {'grainRRMSE':>12} {'aggRRMSE':>10} {'Time':>8}"
    )
    for r in rows:
        cos = r.get("y_cos_RRMSE")
        grain = r.get("y_grain_RRMSE")
        agg = r.get("aggregate_RRMSE")
        total = r.get("Total_Time")
        cos_s = f"{float(cos):12.6f}" if cos is not None else f"{'nan':>12}"
        grain_s = f"{float(grain):12.6f}" if grain is not None else f"{'nan':>12}"
        agg_s = f"{float(agg):10.6f}" if agg is not None else f"{'nan':>10}"
        time_s = f"{float(total):8.1f}" if total is not None else f"{'nan':>8}"
        print(
            f"{r['model_kind']:<3} {r['rff_sampling']:<5} {int(r['num_rff']):>4} "
            f"{cos_s} {grain_s} {agg_s} {time_s}"
        )


if __name__ == "__main__":
    main()
