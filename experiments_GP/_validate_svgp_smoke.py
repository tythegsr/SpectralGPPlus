"""Smoke-check the inducing-point SVGP path on S2 TOA against an exact GP.

Two configurations are checked separately rather than stacked, because PCA and
NIGP do not compose: NIGP learns a per-input-dimension sigma_x, and after a PCA
rotation those dimensions are principal components rather than bands, so the
learned input noise no longer means per-band radiometric noise.

    --config pca    PCA on, NIGP off
    --config nigp   NIGP on, PCA off

The ``exact`` arm is only meaningful at small ``--n-train`` (dense GP is
O(n^3)); it is there to tell "SVGP is learning" apart from "the data at this
scale is not learnable". Run both arms on the same data, seed and band config
and compare per-QoI test RRMSE.
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

KNOWN_TASKS = ["algae", "aot", "cos_i", "cwv", "dust", "fNPV", "fPV"]
DATA_PATH = "experiments_toa/data 11 QoI/snow_toa_fsnow_70to100_20261208.nc"
BAND_CONFIG = "experiments_toa/configs/s2_task_bands_all.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="pca", choices=["pca", "nigp"])
    p.add_argument("--tasks", default="algae", help="Comma-separated QoI names.")
    p.add_argument("--arms", default="svgp", help="svgp and/or exact, comma-separated.")
    p.add_argument("--n-train", type=int, default=20000)
    p.add_argument("--n-test", type=int, default=5000)
    p.add_argument("--num-inducing", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--num-epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--variational-lr", type=float, default=None)
    p.add_argument("--kl-beta", type=float, default=1.0)
    p.add_argument("--freeze-epoch-nigp", type=int, default=50)
    p.add_argument("--n-pca-components", type=int, default=12)
    p.add_argument("--exact-n-train", type=int, default=4000)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Requires a CUDA-enabled torch build (use the gpplus2026 env).",
    )
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    # Dense Cholesky on a near-degenerate Gram matrix routinely fails in float32,
    # so the baseline arm gets its own dtype.
    p.add_argument("--exact-dtype", default="float64", choices=["float32", "float64"])
    p.add_argument("--out", default="experiments_GP/results/svgp_validation")
    return p.parse_args()


def resolve_tasks(arg: str) -> list[str]:
    tasks = [t.strip() for t in arg.split(",") if t.strip()]
    if not tasks:
        raise ValueError("--tasks must list at least one QoI")
    unknown = [t for t in tasks if t not in KNOWN_TASKS]
    if unknown:
        raise ValueError(f"Unknown tasks {unknown}; choose from {KNOWN_TASKS}")
    return tasks


def common_kwargs(args: argparse.Namespace, tasks: list[str]) -> dict:
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"device={args.device!r} requested but this torch build has no CUDA "
            f"({torch.__version__}). Use the gpplus2026 env."
        )
    return dict(
        n_test=args.n_test,
        num_inits=1,
        seed=42,
        device=args.device,
        dtype=dtype,
        ard=True,
        task_names=tasks,
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
        # [] disables NetCDF auto log/logit warps (None would re-enable them).
        log_scale_qoi=[],
        logit_scale_qoi=[],
    )


def summarize(name: str, metrics: dict, tasks: list[str]) -> dict:
    return {
        "arm": name,
        "aggregate_RRMSE": metrics.get("aggregate_RRMSE"),
        "training_time_s": metrics.get("Training_Time"),
        "per_task": {t: metrics.get(f"{t}_RRMSE") for t in tasks},
        "per_task_nlpd": {t: metrics.get(f"{t}_NLPD") for t in tasks},
        "per_task_train_loss": {t: metrics.get(f"{t}_best_train_loss") for t in tasks},
    }


def fmt(value) -> str:
    return "     n/a" if value is None else f"{value:8.5f}"


def main() -> None:
    args = parse_args()
    tasks = resolve_tasks(args.tasks)
    use_nigp = args.config == "nigp"
    pca_components = None if use_nigp else args.n_pca_components
    out_dir = Path(args.out) / args.config
    out_dir.mkdir(parents=True, exist_ok=True)
    gpplus.config.configure_logger(
        level=logging.INFO, log_to_file=str(out_dir / "validation.log")
    )
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    results: dict[str, dict] = {}
    for arm in arms:
        n_train = args.n_train if arm == "svgp" else args.exact_n_train
        print("\n" + "#" * 70)
        print(
            f"# CONFIG: {args.config}   ARM: {arm}   TASKS: {tasks}   "
            f"N_TRAIN: {n_train}   DTYPE: {args.dtype}"
        )
        print("#" * 70)
        started = time.time()
        if arm == "svgp":
            extra = dict(
                svgp=True,
                num_inducing=args.num_inducing,
                batch_size=args.batch_size,
                num_epochs=args.num_epochs,
                variational_lr=args.variational_lr,
                kl_beta=args.kl_beta,
                nigp=use_nigp,
                freeze_epoch_nigp=args.freeze_epoch_nigp if use_nigp else 0,
                n_pca_components=pca_components,
                optimizer_kwargs={**DEFAULT_ADAM_KWARGS, "lr": args.lr},
            )
        else:
            # LBFGS on the exact marginal likelihood; num_epochs<=1 selects it.
            extra = dict(
                num_epochs=1,
                n_pca_components=pca_components,
                dtype=torch.float32 if args.exact_dtype == "float32" else torch.float64,
            )
        shared = common_kwargs(args, tasks)
        shared.update(extra)
        metrics = run_s2_toa_gp(
            n_train=n_train, save_path=str(out_dir / arm), **shared
        )
        results[arm] = summarize(arm, metrics, tasks)
        results[arm]["n_train"] = n_train
        results[arm]["wall_time_s"] = time.time() - started

    print("\n" + "=" * 78)
    print(f"Config '{args.config}' dtype={args.dtype}: per-QoI test RRMSE (lower is better)")
    header = f"{'QoI':<12}" + "".join(f"{arm:>14}" for arm in arms)
    print(header)
    print("-" * 78)
    for task in tasks:
        row = f"{task:<12}" + "".join(
            f"{fmt(results[arm]['per_task'][task]):>14}" for arm in arms
        )
        print(row)
    print("-" * 78)
    print(
        f"{'aggregate':<12}"
        + "".join(f"{fmt(results[arm]['aggregate_RRMSE']):>14}" for arm in arms)
    )
    print(
        f"{'train (s)':<12}"
        + "".join(f"{results[arm]['training_time_s']:>14.1f}" for arm in arms)
    )
    print(
        f"{'n_train':<12}" + "".join(f"{results[arm]['n_train']:>14d}" for arm in arms)
    )
    print("=" * 78)
    print(
        f"SVGP: M={args.num_inducing}, batch={args.batch_size}, "
        f"epochs={args.num_epochs}, "
        + ("nigp=True, no PCA" if use_nigp else f"pca={args.n_pca_components}, nigp=False")
        + f", device={args.device}."
    )

    payload = {
        "config": vars(args) | {"tasks": tasks, "data_path": DATA_PATH},
        "arms": results,
    }
    (out_dir / "comparison.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nWrote {out_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
