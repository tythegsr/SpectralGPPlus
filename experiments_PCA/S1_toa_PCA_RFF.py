"""TOA benchmark with PCA + GPPlus RFF/ORF/SORF kernel (Woodbury inference)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
_PCA_DIR = Path(__file__).resolve().parent
_RFF_DIR = _ROOT / "experiments_RFF"
_MTGPR_DIR = _ROOT / "experiments_RFFMTGPR"
for p in (_ROOT, _PCA_DIR, _RFF_DIR, _MTGPR_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import gpplus
from mtgpr_experiment_utils import DEFAULT_ADAM_KWARGS
from toa_pca_stgp_base import run_toa_pca_stgp
from toa_stgp_base import RFF_SAMPLING_CHOICES


def run_toa_pca_rff_entry(**kwargs) -> dict:
    return run_toa_pca_stgp(**kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="TOA dataset with PCA + GPPlus RFF/ORF/SORF (Woodbury)"
    )
    parser.add_argument("--n-train", type=int, default=49000)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--n-components", type=int, default=30, help="PCA dimension p")
    parser.add_argument(
        "--num-rff",
        type=int,
        default=1600,
        help="D (RFF frequencies); default: 1600",
    )
    parser.add_argument(
        "--rff-sampling",
        type=str,
        default="rff",
        choices=RFF_SAMPLING_CHOICES,
        help="Spectral sampling: rff, orf, or sorf",
    )
    parser.add_argument("--num-inits", type=int, default=4)
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=200,
        help="Epochs per init: 1 uses LBFGSScipy; >1 uses torch.optim.Adam",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        help="Adam learning rate (only when --num-epochs > 1)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=("float32", "float64"),
    )
    parser.add_argument(
        "--predict-chunk-size",
        type=int,
        default=512,
        help="Test points per Woodbury predict chunk (0 = single batch)",
    )
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--ard", action="store_true", default=True)
    parser.add_argument("--no-ard", action="store_false", dest="ard")
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Results directory (default: experiments_PCA/results/toa_pca_{rff|orf|sorf})",
    )
    parser.add_argument(
        "--monitor-validation",
        action="store_true",
        default=True,
        help="Hold out val_fraction of training for validation callbacks (default: on)",
    )
    parser.add_argument(
        "--no-monitor-validation",
        action="store_false",
        dest="monitor_validation",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--plot-posterior",
        action="store_true",
        default=True,
        dest="plot_posterior",
    )
    parser.add_argument("--rel-tolerance", type=float, default=0.01)
    parser.add_argument("--posterior-n-examples", type=int, default=20)
    parser.add_argument("--posterior-example-indices", type=str, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--no-log-grain", action="store_true")
    parser.add_argument("--no-save-checkpoint", action="store_true")
    parser.add_argument(
        "--drop-columns",
        type=str,
        default="132,195,196,197,198,199,200,201,202,203,204,205,206,207,208",
        help="Columns to remove before PCA; empty string = 285-dim",
    )
    parser.add_argument(
        "--pca-svd-solver",
        type=str,
        default="randomized",
        choices=("auto", "full", "randomized", "arpack"),
    )
    parser.add_argument("--response-noise-prior", action="store_true", default=False)
    parser.add_argument("--noise-var-fraction", type=float, default=0.25)
    parser.add_argument("--noise-prior-log-scale", type=float, default=0.5)
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--parallel-verbose", type=int, default=10)
    parser.add_argument("--log-every-n-epochs", type=int, default=10)
    parser.add_argument("--no-training-log", action="store_true")
    args = parser.parse_args()

    save_path = args.save_path
    log_file = args.log_file
    if log_file is None and args.device.startswith("cuda") and save_path:
        import os

        log_file = os.path.join(save_path, "train.log")

    gpplus.config.configure_logger(
        level=getattr(logging, args.log_level),
        log_to_file=log_file,
    )

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    n_jobs = None if args.n_jobs < 0 else args.n_jobs

    optimizer_kwargs = None
    if args.num_epochs > 1 and args.lr is not None:
        optimizer_kwargs = {**DEFAULT_ADAM_KWARGS, "lr": args.lr}

    posterior_example_indices = None
    if args.posterior_example_indices:
        posterior_example_indices = [
            int(x.strip()) for x in args.posterior_example_indices.split(",") if x.strip()
        ]

    drop_columns: list[int] | None
    if args.drop_columns.strip() == "":
        drop_columns = []
    else:
        drop_columns = [int(x.strip()) for x in args.drop_columns.split(",") if x.strip()]

    run_toa_pca_stgp(
        n_train=args.n_train,
        n_test=args.n_test,
        n_components=args.n_components,
        num_rff=args.num_rff,
        rff_sampling=args.rff_sampling,
        num_inits=args.num_inits,
        num_epochs=args.num_epochs,
        optimizer_kwargs=optimizer_kwargs,
        seed=args.seed,
        device=args.device,
        dtype=dtype,
        ard=args.ard,
        save_path=save_path,
        n_jobs=n_jobs,
        predict_chunk_size=args.predict_chunk_size,
        monitor_validation=args.monitor_validation,
        val_fraction=args.val_fraction,
        plot_validation=not args.no_plot,
        plot_posterior=args.plot_posterior and not args.no_plot,
        rel_tolerance=args.rel_tolerance,
        posterior_n_examples=args.posterior_n_examples,
        posterior_example_indices=posterior_example_indices,
        data_path=args.data_path,
        parallel_verbose=args.parallel_verbose,
        training_verbose=not args.no_training_log,
        log_every_n_epochs=args.log_every_n_epochs,
        save_checkpoint=not args.no_save_checkpoint,
        log_grain=not args.no_log_grain,
        drop_columns=drop_columns,
        pca_svd_solver=args.pca_svd_solver,
        response_noise_prior=args.response_noise_prior,
        noise_var_fraction=args.noise_var_fraction,
        noise_prior_log_scale=args.noise_prior_log_scale,
    )
