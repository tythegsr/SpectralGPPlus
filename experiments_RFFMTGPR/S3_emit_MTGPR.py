"""S3 EMIT snow70 joint SORF+NIGP MTGPR (Woodbury inference).

Third example in the S1/S2/S3 series: trains on
``emit_test_data_processed_20262008.nc`` (reflectance + elevation) with
10k train / 5k val / remaining test. Uses ``RFFMTGPR`` with
``rff_sampling='sorf'`` and classic NIGP.

Independent single-task SORF lives in ``experiments_SORF/S3_emit_SORF.py``.
Minibatch inducing-point SVGP is not used here; see ``experiments_SVGP``.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
_MTGPR_DIR = Path(__file__).resolve().parent
_RFF_DIR = _ROOT / "experiments_RFF"

# ---------------------------------------------------------------------------
# IDE RUN CONFIGURATION — edit these, then press Run.
# ---------------------------------------------------------------------------
QOI: list[str] | None = ["cosA", "sinA", "grain_size", "fsnow", "algae", "fsoil", "fPV", "fNPV", "liquid_water", "aot", "cwv", "dust"]
N_TRAIN = 100000
N_VAL = 5000
NUM_RFF = 5000
NUM_INITS = 1
NUM_EPOCHS = 2000
LR = 1e-3
SEED = 42
DEVICE = "cuda"
DTYPE = "float32"  # "float32" | "float64"
PREDICT_CHUNK_SIZE = 2048
N_JOBS = 1
ARD = True
SAVE_PATH: str | None = None
MONITOR_VALIDATION = True
PLOT = True
PLOT_POSTERIOR = True
REL_TOLERANCE = 0.01
POSTERIOR_N_EXAMPLES = 20
POSTERIOR_EXAMPLE_INDICES: str | None = None
CORRECT_SORF = True
SAVE_CHECKPOINT = True
RESPONSE_NOISE_PRIOR = False
NOISE_VAR_FRACTION = 0.01
NOISE_PRIOR_LOG_SCALE = 0.5
LOG_SCALE_QOI: list[str] | None = []
LOGIT_SCALE_QOI: list[str] | None = []
RANK_KERNEL = 2
NIGP = True
FREEZE_EPOCH_NIGP = 200
NONDEFAULT_ONLY = True
FILTER_VALID_LABELS = False
INCLUDE_ELEVATION = True
LOG_LEVEL = "INFO"
LOG_FILE: str | None = None
PARALLEL_VERBOSE = 10
LOG_EVERY_N_EPOCHS = 50
TRAINING_LOG = True
DATA_PATH: str | None = str(
    _ROOT / "experiments_toa" / "data 11 QoI" / "emit_test_data_processed_20262008.nc"
)
# ---------------------------------------------------------------------------

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments_toa.paths import pin_toa_import_paths

pin_toa_import_paths(_MTGPR_DIR, _RFF_DIR)

import gpplus
from experiments_toa.s2_cli import parse_example_indices
from mtgpr_experiment_utils import DEFAULT_ADAM_KWARGS
from emit_s3_mtgpr_base import parse_s3_task_names, run_s3_emit_mtgpr


def run_s3_emit_mtgpr_entry(**kwargs) -> dict:
    return run_s3_emit_mtgpr(**kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="S3 EMIT joint SORF+NIGP MTGPR (Woodbury; no SVGP)"
    )
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--n-val", type=int, default=N_VAL)
    parser.add_argument("--num-rff", type=int, default=NUM_RFF)
    parser.add_argument("--num-inits", type=int, default=NUM_INITS)
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--dtype", type=str, default=DTYPE, choices=("float32", "float64"))
    parser.add_argument("--predict-chunk-size", type=int, default=PREDICT_CHUNK_SIZE)
    parser.add_argument("--n-jobs", type=int, default=N_JOBS)
    parser.add_argument("--save-path", type=str, default=SAVE_PATH)
    parser.add_argument("--data-path", type=str, default=DATA_PATH)
    parser.add_argument(
        "--qoi",
        nargs="+",
        default=None,
        metavar="NAME",
        help="QoIs (default: IDE QOI list; all mapped EMIT QoIs allowed)",
    )
    parser.add_argument("--nigp", action=argparse.BooleanOptionalAction, default=NIGP)
    parser.add_argument("--freeze-epoch-nigp", type=int, default=FREEZE_EPOCH_NIGP)
    parser.add_argument("--rank-kernel", type=int, default=RANK_KERNEL)
    parser.add_argument("--correct-sorf", action=argparse.BooleanOptionalAction, default=CORRECT_SORF)
    parser.add_argument(
        "--nondefault-only",
        action=argparse.BooleanOptionalAction,
        default=NONDEFAULT_ONLY,
    )
    parser.add_argument(
        "--filter-valid-labels",
        action=argparse.BooleanOptionalAction,
        default=FILTER_VALID_LABELS,
    )
    parser.add_argument(
        "--include-elevation",
        action=argparse.BooleanOptionalAction,
        default=INCLUDE_ELEVATION,
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    nigp = bool(args.nigp)
    nigp_str = "_nigp" if nigp else ""
    freeze_str = (
        f"_freezeepochnigp{args.freeze_epoch_nigp}" if nigp and args.freeze_epoch_nigp > 0 else ""
    )
    elev_str = "_elev" if args.include_elevation else ""
    save_path = args.save_path or (
        f"experiments_RFFMTGPR/results/Aug20/s3_emit_mtgpr_{args.num_inits}inits_"
        f"numrff{args.num_rff}_lr{args.lr}{nigp_str}{freeze_str}{elev_str}_dtype{args.dtype}"
    )
    log_file = LOG_FILE
    if log_file is None and args.device.startswith("cuda"):
        log_file = os.path.join(save_path, "train.log")
    gpplus.config.configure_logger(level=getattr(logging, LOG_LEVEL), log_to_file=log_file)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    optimizer_kwargs = None
    if args.num_epochs > 1 and args.lr is not None:
        optimizer_kwargs = {**DEFAULT_ADAM_KWARGS, "lr": args.lr}

    qoi = parse_s3_task_names(args.qoi if args.qoi is not None else QOI)
    print(
        f"S3 EMIT MTGPR  nigp={nigp}  freeze_epoch_nigp={args.freeze_epoch_nigp}  "
        f"include_elevation={args.include_elevation}  qoi={qoi}  "
        f"n_train={args.n_train}  n_val={args.n_val}"
    )

    run_s3_emit_mtgpr(
        n_train=args.n_train,
        n_val=args.n_val,
        num_rff=args.num_rff,
        rff_sampling="sorf",
        num_inits=args.num_inits,
        num_epochs=args.num_epochs,
        optimizer_kwargs=optimizer_kwargs,
        seed=args.seed,
        device=args.device,
        dtype=dtype,
        ard=ARD,
        save_path=save_path,
        n_jobs=None if args.n_jobs < 0 else args.n_jobs,
        predict_chunk_size=args.predict_chunk_size,
        monitor_validation=MONITOR_VALIDATION,
        plot_validation=PLOT and not args.no_plot,
        plot_posterior=PLOT_POSTERIOR and PLOT and not args.no_plot,
        rel_tolerance=REL_TOLERANCE,
        posterior_n_examples=POSTERIOR_N_EXAMPLES,
        posterior_example_indices=parse_example_indices(POSTERIOR_EXAMPLE_INDICES),
        data_path=args.data_path,
        parallel_verbose=PARALLEL_VERBOSE,
        training_verbose=TRAINING_LOG,
        log_every_n_epochs=LOG_EVERY_N_EPOCHS,
        save_checkpoint=SAVE_CHECKPOINT,
        response_noise_prior=RESPONSE_NOISE_PRIOR,
        noise_var_fraction=NOISE_VAR_FRACTION,
        noise_prior_log_scale=NOISE_PRIOR_LOG_SCALE,
        correct_sorf=args.correct_sorf,
        log_scale_qoi=LOG_SCALE_QOI,
        logit_scale_qoi=LOGIT_SCALE_QOI,
        rank_kernel=args.rank_kernel,
        task_names=qoi,
        nigp=nigp,
        freeze_epoch_nigp=args.freeze_epoch_nigp,
        nondefault_only=args.nondefault_only,
        filter_valid_labels=args.filter_valid_labels,
        include_elevation=args.include_elevation,
    )
