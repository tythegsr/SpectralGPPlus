"""Compare algae SVGP+PCA and SVGP+NIGP against the Aug12 SORF+NIGP baseline.

Runs both SVGP configurations on the same seed / band config / data, then
prints test RRMSE (and a few companions) next to the Aug12 reference from

    experiments_SORF/results/Aug12/.../gp_S2_TOA_nTrain100000_nTest10000_sorfD800.json

Aug12 used N=100k, SORF D=800, float64, ~2000 epochs on GPU. The SVGP arms
default to a smaller N / M / epoch budget for a same-day diagnostic; pass
``--n-train 100000 --num-inducing 512 --num-epochs 1000`` for a closer match.

PCA and NIGP are run as separate arms (they do not compose).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
for _p in (
    _ROOT,
    _ROOT / "experiments_GP",
    _ROOT / "experiments_RFF",
    _ROOT / "experiments_RFFMTGPR",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import gpplus
from gp_experiment_utils import DEFAULT_ADAM_KWARGS
from toa_s2_gp_base import run_s2_toa_gp

TASK = "algae"
DATA_PATH = "experiments_toa/data 11 QoI/snow_toa_fsnow_70to100_20261208.nc"
BAND_CONFIG = "experiments_toa/configs/s2_task_bands_all.json"
AUG12_METRICS = (
    _ROOT
    / "experiments_SORF/results/Aug12"
    / "s2_toa_sorf_1inits_numrff800_lr0.1_taskbandconfig_rbf_nigp"
    "_freezeepochnigp100_sloperefreshes20_dtypefloat64"
    / "gp_S2_TOA_nTrain100000_nTest10000_sorfD800.json"
)

METRIC_KEYS = ("RRMSE", "RMSE", "MAE", "MedAE", "NLPD", "best_train_loss")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-train", type=int, default=100000)
    p.add_argument("--n-test", type=int, default=10000)
    p.add_argument("--num-inducing", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--num-epochs", type=int, default=1000)
    p.add_argument(
        "--adam-stop-patience",
        type=int,
        default=50,
        help="Epochs without train-loss improvement before early stop (Adam).",
    )
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--variational-lr", type=float, default=None)
    p.add_argument("--kl-beta", type=float, default=1.0)
    p.add_argument("--freeze-epoch-nigp", type=int, default=50)
    p.add_argument("--n-pca-components", type=int, default=12)
    p.add_argument(
        "--arms",
        default="pca,nigp",
        help="Comma-separated: pca and/or nigp (default: both).",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Requires a CUDA-enabled torch build (use the gpplus2026 env).",
    )
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    p.add_argument("--out", default="experiments_SVGP/results/svgp_vs_aug12")
    p.add_argument("--aug12", type=Path, default=AUG12_METRICS)
    p.add_argument("--skip-train", action="store_true", help="Only print Aug12.")
    return p.parse_args()


def load_aug12(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Aug12 metrics not found: {path}")
    raw = json.loads(path.read_text())
    return {
        "label": "aug12_sorf_nigp",
        "source": str(path),
        "n_train": raw.get("n_train"),
        "n_test": raw.get("n_test"),
        "num_rff": raw.get("num_rff"),
        "nigp": raw.get("nigp"),
        "n_pca_components": raw.get("n_pca_components"),
        "metrics": {k: raw.get(f"{TASK}_{k}") for k in METRIC_KEYS},
        "aggregate_RRMSE": raw.get("aggregate_RRMSE"),
        "training_time_s": raw.get(f"{TASK}_Training_Time") or raw.get("Training_Time"),
    }


def summarize_svgp(label: str, metrics: dict, *, n_train: int, wall_s: float) -> dict:
    return {
        "label": label,
        "n_train": n_train,
        "n_test": metrics.get("n_test"),
        "num_inducing": metrics.get("num_inducing"),
        "nigp": metrics.get("nigp"),
        "n_pca_components": metrics.get("n_pca_components"),
        "metrics": {k: metrics.get(f"{TASK}_{k}") for k in METRIC_KEYS},
        "aggregate_RRMSE": metrics.get("aggregate_RRMSE"),
        "training_time_s": metrics.get("Training_Time"),
        "wall_time_s": wall_s,
    }


def fmt(value) -> str:
    if value is None:
        return "     n/a"
    try:
        return f"{float(value):8.5f}"
    except (TypeError, ValueError):
        return f"{value!s:>8}"


def print_table(rows: list[dict]) -> None:
    labels = [r["label"] for r in rows]
    width = max(14, *(len(lab) + 2 for lab in labels))
    print("\n" + "=" * (12 + width * len(labels)))
    print("algae vs Aug12  (lower RRMSE / NLPD is better)")
    header = f"{'metric':<12}" + "".join(f"{lab:>{width}}" for lab in labels)
    print(header)
    print("-" * (12 + width * len(labels)))
    for key in METRIC_KEYS:
        row = f"{key:<12}" + "".join(
            f"{fmt(r['metrics'].get(key)):>{width}}" for r in rows
        )
        print(row)
    print("-" * (12 + width * len(labels)))
    print(
        f"{'n_train':<12}"
        + "".join(f"{str(r.get('n_train')):>{width}}" for r in rows)
    )
    print(
        f"{'train (s)':<12}"
        + "".join(
            f"{(r.get('training_time_s') if r.get('training_time_s') is not None else float('nan')):>{width}.1f}"
            for r in rows
        )
    )
    print("=" * (12 + width * len(labels)))


def run_arm(arm: str, args: argparse.Namespace, out_dir: Path) -> dict:
    use_nigp = arm == "nigp"
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"device={args.device!r} requested but this torch build has no CUDA "
            f"({torch.__version__}). Use the gpplus2026 env."
        )

    print("\n" + "#" * 70)
    print(
        f"# ARM: svgp_{arm}   TASK: {TASK}   N_TRAIN: {args.n_train}   "
        f"M: {args.num_inducing}   DTYPE: {args.dtype}"
    )
    print("#" * 70)

    started = time.time()
    metrics = run_s2_toa_gp(
        n_train=args.n_train,
        n_test=args.n_test,
        num_inits=1,
        num_epochs=args.num_epochs,
        seed=42,
        device=args.device,
        dtype=dtype,
        ard=True,
        task_names=[TASK],
        task_band_config=BAND_CONFIG,
        data_path=DATA_PATH,
        response_noise_prior=False,
        monitor_validation=True,
        plot_validation=False,
        plot_posterior=False,
        predict_chunk_size=4096,
        log_every_n_epochs=25,
        n_jobs=1,
        parallel_verbose=0,
        log_scale_qoi=[],
        logit_scale_qoi=[],
        svgp=True,
        num_inducing=args.num_inducing,
        batch_size=args.batch_size,
        variational_lr=args.variational_lr,
        kl_beta=args.kl_beta,
        nigp=use_nigp,
        freeze_epoch_nigp=args.freeze_epoch_nigp if use_nigp else 0,
        n_pca_components=None if use_nigp else args.n_pca_components,
        adam_stop_patience=args.adam_stop_patience,
        optimizer_kwargs={**DEFAULT_ADAM_KWARGS, "lr": args.lr},
        save_path=str(out_dir / f"svgp_{arm}"),
    )
    return summarize_svgp(
        f"svgp_{arm}",
        metrics,
        n_train=args.n_train,
        wall_s=time.time() - started,
    )


def main() -> None:
    args = parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ("pca", "nigp")]
    if unknown:
        raise ValueError(f"Unknown arms {unknown}; choose from pca,nigp")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    gpplus.config.configure_logger(
        level=logging.INFO, log_to_file=str(out_dir / "comparison.log")
    )

    rows: list[dict] = []
    if not args.skip_train:
        for arm in arms:
            rows.append(run_arm(arm, args, out_dir))

    aug12 = load_aug12(args.aug12)
    rows.append(aug12)

    print_table(rows)
    print(
        f"\nAug12 reference: SORF D={aug12.get('num_rff')}, "
        f"N={aug12.get('n_train')}, nigp={aug12.get('nigp')}, "
        f"no PCA (algae_RRMSE={fmt(aug12['metrics'].get('RRMSE')).strip()})."
    )
    print(
        f"SVGP arms: M={args.num_inducing}, batch={args.batch_size}, "
        f"epochs={args.num_epochs}, n_train={args.n_train}, "
        f"device={args.device}, dtype={args.dtype}."
    )
    if not args.skip_train and args.n_train < 100_000:
        print(
            "NOTE: Aug12 used n_train=100000; this run is smaller. "
            "Scale --n-train / --num-epochs / --num-inducing before claiming parity."
        )

    payload = {
        "task": TASK,
        "config": vars(args) | {"arms": arms, "data_path": DATA_PATH},
        "rows": rows,
    }
    out_json = out_dir / "comparison.json"
    out_json.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
