"""TOA RFFMTGPR with Adam noise-vs-other learning-rate groups (no scheduler)."""

from __future__ import annotations

import logging
import sys
from functools import partial
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
_MTGPR_DIR = Path(__file__).resolve().parent

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments_toa.paths import pin_toa_import_paths

pin_toa_import_paths(_MTGPR_DIR)

import gpplus
from gpplus.training.optimizer_param_groups import adam_noise_vs_other_param_groups
from mtgpr_experiment_utils import DEFAULT_ADAM_KWARGS
from toa_mtgpr_base import run_toa_mtgpr


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "TOA joint RFFMTGPR with Adam param groups: "
            "noise LR vs other hypers (no LR scheduler)"
        )
    )
    parser.add_argument("--n-train", type=int, default=16000)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument(
        "--rff-sampling",
        type=str,
        default="sorf",
        choices=("rff", "orf", "sorf"),
        help="Spectral feature sampling: RFF, ORF, or SORF",
    )
    parser.add_argument(
        "--num-rff",
        type=int,
        default=800,
        help="D (RFF/ORF/SORF frequencies); default min(512, n_train//3)",
    )
    parser.add_argument("--num-inits", type=int, default=1)
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=2000,
        help="Epochs per init (must be >1 for Adam grouped LRs)",
    )
    parser.add_argument(
        "--noise-lr",
        type=float,
        default=0.01,
        help="Adam LR for raw_noise / raw_task_noises",
    )
    parser.add_argument(
        "--other-lr",
        type=float,
        default=0.01,
        help="Adam LR for all other trainable parameters",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=("float32", "float64"),
    )
    parser.add_argument("--predict-chunk-size", type=int, default=512)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--ard", action="store_true", default=True)
    parser.add_argument("--no-ard", action="store_false", dest="ard")
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Results directory (default: .../toa_mt_grouped_adam_{rff_sampling})",
    )
    parser.add_argument("--monitor-validation", action="store_true", default=True)
    parser.add_argument(
        "--no-monitor-validation",
        action="store_false",
        dest="monitor_validation",
    )
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
    parser.add_argument(
        "--train-subset",
        type=str,
        default="maximin",
        choices=("random", "maximin"),
    )
    parser.add_argument("--no-log-grain", action="store_true")
    parser.add_argument(
        "--logit-cos",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--no-save-checkpoint", action="store_true")
    parser.add_argument(
        "--drop-columns",
        type=str,
        default="132,195,196,197,198,199,200,201,202,203,204,205,206,207,208",
    )
    parser.add_argument(
        "--response-noise-prior",
        action="store_true",
        default=True,
        help="Enable LogNormal per-task noise prior from training response columns",
    )
    parser.add_argument("--noise-var-fraction", type=float, default=1e-3)
    parser.add_argument("--noise-prior-log-scale", type=float, default=0.5)
    parser.add_argument(
        "--outputscale-prior",
        action="store_true",
        default=False,
        help="Enable Normal prior on log10 outputscale (MAP)",
    )
    parser.add_argument(
        "--outputscale-prior-loc",
        type=float,
        default=1.0,
        help="Normal prior mean for log10 outputscale (default: 0.0)",
    )
    parser.add_argument(
        "--outputscale-prior-scale",
        type=float,
        default=1.0,
        help="Normal prior std for log10 outputscale (default: 1.0)",
    )
    parser.add_argument(
        "--lengthscale-prior",
        action="store_true",
        default=True,
        help="Enable Normal prior on log10 lengthscale (MAP; shared loc/scale for ARD)",
    )
    parser.add_argument(
        "--lengthscale-prior-loc",
        type=float,
        default=0.6,
        help=(
            "Normal prior mean for log10 lengthscale (default: -2.0, matches init). "
            "For correct_sorf, try ~0.6 (~+2.6 vs legacy W scale)"
        ),
    )
    parser.add_argument(
        "--lengthscale-prior-scale",
        type=float,
        default=2.0,
        help="Normal prior std for log10 lengthscale (default: 2.0)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--parallel-verbose", type=int, default=10)
    parser.add_argument("--log-every-n-epochs", type=int, default=50)
    parser.add_argument(
        "--correct-sorf",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--no-training-log", action="store_true")
    args = parser.parse_args()

    if args.num_epochs <= 1:
        raise SystemExit("Grouped Adam requires --num-epochs > 1.")

    save_path = args.save_path
    if save_path is None:
        save_path = f"experiments_RFFMTGPR/results/july15/toa_mt_grouped_adam_{args.rff_sampling}_lr2_{args.other_lr}"

    log_file = args.log_file
    if log_file is None and args.device.startswith("cuda"):
        import os

        log_file = os.path.join(save_path, "train.log")

    gpplus.config.configure_logger(
        level=getattr(logging, args.log_level),
        log_to_file=log_file,
    )

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    n_jobs = None if args.n_jobs < 0 else args.n_jobs

    shared_adam = {k: v for k, v in DEFAULT_ADAM_KWARGS.items() if k != "lr"}
    param_groups_fn = partial(
        adam_noise_vs_other_param_groups,
        noise_lr=args.noise_lr,
        other_lr=args.other_lr,
        **shared_adam,
    )
    # Flat kwargs for metrics/fallback; per-group LRs come from param_groups_fn.
    optimizer_kwargs = {**DEFAULT_ADAM_KWARGS, "lr": args.other_lr}
    print(
        f"Grouped Adam LRs: noise_lr={args.noise_lr}, other_lr={args.other_lr} "
        f"(no scheduler)"
    )

    posterior_example_indices = None
    if args.posterior_example_indices:
        posterior_example_indices = [
            int(x.strip()) for x in args.posterior_example_indices.split(",") if x.strip()
        ]

    drop_columns = None
    if args.drop_columns:
        drop_columns = [int(x.strip()) for x in args.drop_columns.split(",") if x.strip()]

    run_toa_mtgpr(
        n_train=args.n_train,
        n_test=args.n_test,
        num_rff=args.num_rff,
        rff_sampling=args.rff_sampling,
        num_inits=args.num_inits,
        num_epochs=args.num_epochs,
        optimizer_kwargs=optimizer_kwargs,
        param_groups_fn=param_groups_fn,
        seed=args.seed,
        device=args.device,
        dtype=dtype,
        ard=args.ard,
        save_path=save_path,
        n_jobs=n_jobs,
        predict_chunk_size=args.predict_chunk_size,
        monitor_validation=args.monitor_validation,
        plot_validation=not args.no_plot,
        plot_posterior=args.plot_posterior,
        rel_tolerance=args.rel_tolerance,
        posterior_n_examples=args.posterior_n_examples,
        posterior_example_indices=posterior_example_indices,
        data_path=args.data_path,
        train_subset=args.train_subset,
        parallel_verbose=args.parallel_verbose,
        training_verbose=not args.no_training_log,
        log_every_n_epochs=args.log_every_n_epochs,
        save_checkpoint=not args.no_save_checkpoint,
        log_grain=not args.no_log_grain,
        logit_cos=args.logit_cos,
        drop_columns=drop_columns,
        response_noise_prior=args.response_noise_prior,
        noise_var_fraction=args.noise_var_fraction,
        noise_prior_log_scale=args.noise_prior_log_scale,
        outputscale_prior=args.outputscale_prior,
        outputscale_prior_loc=args.outputscale_prior_loc,
        outputscale_prior_scale=args.outputscale_prior_scale,
        lengthscale_prior=args.lengthscale_prior,
        lengthscale_prior_loc=args.lengthscale_prior_loc,
        lengthscale_prior_scale=args.lengthscale_prior_scale,
        correct_sorf=args.correct_sorf,
    )
