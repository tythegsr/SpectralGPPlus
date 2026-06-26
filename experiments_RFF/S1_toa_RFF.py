"""TOA benchmark with GPPlus RFF kernel (Woodbury inference)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
_RFF_DIR = Path(__file__).resolve().parent
for p in (_ROOT, _RFF_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import gpplus
from rff_experiment_utils import DEFAULT_ADAM_KWARGS
from toa_rff_base import run_toa_rff


def run_toa_rff_entry(**kwargs) -> dict:
    return run_toa_rff(**kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TOA dataset with GPPlus RFF (Woodbury)")
    parser.add_argument("--n-train", type=int, default=10000)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument(
        "--num-rff",
        type=int,
        default=None,
        help="D (RFF frequencies); default min(512, n_train//3)",
    )
    parser.add_argument("--num-inits", type=int, default=8)
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Epochs per init: 1 uses LBFGSScipy; >1 uses torch.optim.Adam",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Adam learning rate (only when --num-epochs > 1)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float64",
        choices=("float32", "float64"),
    )
    parser.add_argument(
        "--predict-chunk-size",
        type=int,
        default=512,
        help="Test points per Woodbury predict chunk (0 = single batch)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel hyperparameter inits (-1 = all cores)",
    )
    parser.add_argument("--ard", action="store_true", default=True)
    parser.add_argument("--no-ard", action="store_false", dest="ard")
    parser.add_argument("--save-path", type=str, default="experiments_RFF/results/toa_rff")
    parser.add_argument(
        "--monitor-validation",
        action="store_true",
        help="Hold out val_fraction of training for validation callbacks",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip validation curve plots after saving JSON",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to toa_data_flattened.npz (default: repo root)",
    )
    args = parser.parse_args()

    gpplus.config.configure_logger()

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    n_jobs = None if args.n_jobs < 0 else args.n_jobs

    optimizer_kwargs = None
    if args.num_epochs > 1 and args.lr is not None:
        optimizer_kwargs = {**DEFAULT_ADAM_KWARGS, "lr": args.lr}

    run_toa_rff(
        n_train=args.n_train,
        n_test=args.n_test,
        num_rff=args.num_rff,
        num_inits=args.num_inits,
        num_epochs=args.num_epochs,
        optimizer_kwargs=optimizer_kwargs,
        seed=args.seed,
        device=args.device,
        dtype=dtype,
        ard=args.ard,
        save_path=args.save_path,
        n_jobs=n_jobs,
        predict_chunk_size=args.predict_chunk_size,
        monitor_validation=args.monitor_validation,
        val_fraction=args.val_fraction,
        plot_validation=not args.no_plot,
        data_path=args.data_path,
    )
