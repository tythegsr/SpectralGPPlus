"""S2 11-QoI TOA benchmark with GPPlus LRNN / Deep Basis Kernel."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
_LRNN_DIR = Path(__file__).resolve().parent
_MTGPR_DIR = _ROOT / "experiments_RFFMTGPR"
_RFF_DIR = _ROOT / "experiments_RFF"

# IDE RUN CONFIGURATION
# None trains all 11 QoIs. Example: QOI = ["algae", "fsnow"]
QOI: list[str] | None = None

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments_toa.paths import pin_toa_import_paths

pin_toa_import_paths(_MTGPR_DIR, _RFF_DIR, _LRNN_DIR)

import gpplus
from experiments_toa.s2_cli import add_s2_common_args, parse_example_indices, selected_task_names
from toa_lrnn_base import DEFAULT_LRNN_ADAM_KWARGS
from toa_s2_lrnn_base import run_s2_toa_lrnn


def run_s2_toa_lrnn_entry(**kwargs) -> dict:
    return run_s2_toa_lrnn(**kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="S2 11-QoI TOA with GPPlus LRNN / DBK")
    parser.add_argument("--n-train", type=int, default=16000)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 256, 512, 1024, 512, 256])
    parser.add_argument("--feature-rank", type=int, default=128)
    parser.add_argument("--activation", type=str, default="tanh", choices=("tanh", "relu", "gelu", "identity"))
    parser.add_argument("--variance-correction", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-inits", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32", choices=("float32", "float64"))
    parser.add_argument("--predict-chunk-size", type=int, default=512)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--monitor-validation", action="store_true", default=True)
    parser.add_argument("--no-monitor-validation", action="store_false", dest="monitor_validation")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--plot-posterior", action="store_true", default=True, dest="plot_posterior")
    parser.add_argument("--rel-tolerance", type=float, default=0.01)
    parser.add_argument("--posterior-n-examples", type=int, default=20)
    parser.add_argument("--posterior-example-indices", type=str, default=None)
    add_s2_common_args(parser, default_qoi=QOI)
    parser.add_argument("--no-save-checkpoint", action="store_true")
    parser.add_argument("--response-noise-prior", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--noise-var-fraction", type=float, default=0.01)
    parser.add_argument("--noise-prior-log-scale", type=float, default=0.01)
    parser.add_argument("--log-level", type=str, default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--parallel-verbose", type=int, default=10)
    parser.add_argument("--log-every-n-epochs", type=int, default=50)
    parser.add_argument("--no-training-log", action="store_true")
    args = parser.parse_args()

    hid_tag = "-".join(str(h) for h in args.hidden_dims)
    save_path = args.save_path or (
        f"experiments_LRNN/results/s2_toa_lrnn_{args.num_inits}inits_hiddims{hid_tag}_featr{args.feature_rank}"
    )
    log_file = args.log_file
    if log_file is None and args.device.startswith("cuda"):
        log_file = os.path.join(save_path, "train.log")
    gpplus.config.configure_logger(level=getattr(logging, args.log_level), log_to_file=log_file)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    optimizer_kwargs = None
    if args.num_epochs > 1:
        optimizer_kwargs = {
            **DEFAULT_LRNN_ADAM_KWARGS,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        }

    run_s2_toa_lrnn(
        n_train=args.n_train,
        n_test=args.n_test,
        hidden_dims=args.hidden_dims,
        feature_rank=args.feature_rank,
        activation=args.activation,
        variance_correction=args.variance_correction,
        num_inits=args.num_inits,
        num_epochs=args.num_epochs,
        optimizer_kwargs=optimizer_kwargs,
        seed=args.seed,
        device=args.device,
        dtype=dtype,
        save_path=save_path,
        n_jobs=None if args.n_jobs < 0 else args.n_jobs,
        predict_chunk_size=args.predict_chunk_size,
        monitor_validation=args.monitor_validation,
        plot_validation=not args.no_plot,
        plot_posterior=args.plot_posterior and not args.no_plot,
        rel_tolerance=args.rel_tolerance,
        posterior_n_examples=args.posterior_n_examples,
        posterior_example_indices=parse_example_indices(args.posterior_example_indices),
        data_path=args.data_path,
        parallel_verbose=args.parallel_verbose,
        training_verbose=not args.no_training_log,
        log_every_n_epochs=args.log_every_n_epochs,
        save_checkpoint=not args.no_save_checkpoint,
        response_noise_prior=args.response_noise_prior,
        noise_var_fraction=args.noise_var_fraction,
        noise_prior_log_scale=args.noise_prior_log_scale,
        input_variable=args.input_variable,
        task_names=selected_task_names(args),
        task_band_config=args.task_band_config,
    )
