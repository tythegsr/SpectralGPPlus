"""S4 independent SORF+NIGP (Woodbury RFFGPR) on EMIT or snow-TOA NetCDF.

Fourth example in the S1/S2/S3/S4 series: trains with radiance + geometry aux
(coszen, ele_km, RAA_TRUE) predicting S4 QoIs. ``DATA_PATH`` may be either a
processed EMIT file (``radiance``/``obs``/``state``) or a snow-TOA simulation
file (``toa_radiance`` + named QoIs). One ``RFFGPR`` per QoI with
``rff_sampling='sorf'`` and classic NIGP.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
_SORF_DIR = Path(__file__).resolve().parent
_RFF_DIR = _ROOT / "experiments_RFF"
_MTGPR_DIR = _ROOT / "experiments_RFFMTGPR"

# ---------------------------------------------------------------------------
# IDE RUN CONFIGURATION — edit these, then press Run.
# ---------------------------------------------------------------------------
QOI: list[str] | None = ["grain_size", "cos_i", "dust", "algae", "fsnow", "cwv", "lwc"] # all 10 S4 tasks
N_TRAIN = 100000
N_VAL = 5000
NUM_RFF = 800
NUM_INITS = 1
NUM_EPOCHS = 1000
LR = 1e-1
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
NOISE_PRIOR_LOG_SCALE = 0.25
# LOG_SCALE_QOI: list[str] | None = ["grain_size", "dust", "algae"]
# LOGIT_SCALE_QOI: list[str] | None = ["fsnow"]
# LOG_OFFSETS: dict[str, float] | None = {"dust": 1.0, "algae": 1.0}  # default dust C=1.0 from S2_LOG_OFFSETS
LOG_SCALE_QOI: list[str] | None = []
LOGIT_SCALE_QOI: list[str] | None = []
LOG_OFFSETS: dict[str, float] | None = None
NIGP = True
FREEZE_EPOCH_NIGP = 100
FILTER_VALID_LABELS = False
OFF_FLOOR_TRAIN_TASKS: list[str] | None = []
TASK_BAND_CONFIG: str | None = (
    None
)
LOG_LEVEL = "INFO"
LOG_FILE: str | None = None
PARALLEL_VERBOSE = 10
LOG_EVERY_N_EPOCHS = 50
TRAINING_LOG = True
# EMIT processed or snow-TOA simulation NetCDF (schema auto-detected).
# Examples:
#   emit_test_data_90to100_aotbelow02_20262608.nc
#   snow_toa_fsnow_90to100_20262608.nc
DATA_PATH: str | None = str(
    _ROOT
    / "experiments_toa"
    / "data 11 QoI"
    # / "emit_test_data_90to100_aotbelow02_20262608.nc"
    / "snow_toa_fsnow_90to100_20262608.nc"
)
# ---------------------------------------------------------------------------

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments_toa.paths import pin_toa_import_paths

pin_toa_import_paths(_MTGPR_DIR, _RFF_DIR, _SORF_DIR)

import gpplus
from experiments_toa.s2_cli import parse_example_indices
from experiments_toa.s2_constants import S4_LOGIT_BOUNDS
from mtgpr_experiment_utils import DEFAULT_ADAM_KWARGS
from emit_s4_sorf_base import parse_s4_task_names, run_s4_emit_sorf


def run_s4_emit_sorf_entry(**kwargs) -> dict:
    return run_s4_emit_sorf(**kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="S4 EMIT independent SORF+NIGP (Woodbury RFFGPR; radiance + geometry aux)"
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
        help="QoIs (default: all S4 tasks)",
    )
    parser.add_argument("--nigp", action=argparse.BooleanOptionalAction, default=NIGP)
    parser.add_argument("--freeze-epoch-nigp", type=int, default=FREEZE_EPOCH_NIGP)
    parser.add_argument("--correct-sorf", action=argparse.BooleanOptionalAction, default=CORRECT_SORF)
    parser.add_argument(
        "--filter-valid-labels",
        action=argparse.BooleanOptionalAction,
        default=FILTER_VALID_LABELS,
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    nigp = bool(args.nigp)
    nigp_str = "_nigp" if nigp else ""
    freeze_str = (
        f"_freezeepochnigp{args.freeze_epoch_nigp}" if nigp and args.freeze_epoch_nigp > 0 else ""
    )
    band_str = "_taskbandconfig" if TASK_BAND_CONFIG else ""
    of_str = "_offfloor" if OFF_FLOOR_TRAIN_TASKS else ""
    save_path = args.save_path or (
        f"experiments_SORF/results/Aug26/s4_emit_aotbelow02_sorf_{args.num_inits}inits_"
        f"numrff{args.num_rff}_lr{args.lr}{nigp_str}{freeze_str}"
        f"{band_str}{of_str}_dtype{args.dtype}"
    )
    log_file = LOG_FILE
    if log_file is None and args.device.startswith("cuda"):
        log_file = os.path.join(save_path, "train.log")
    gpplus.config.configure_logger(level=getattr(logging, LOG_LEVEL), log_to_file=log_file)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    optimizer_kwargs = None
    if args.num_epochs > 1 and args.lr is not None:
        optimizer_kwargs = {**DEFAULT_ADAM_KWARGS, "lr": args.lr}

    qoi = parse_s4_task_names(args.qoi if args.qoi is not None else QOI)
    print(
        f"S4 EMIT independent SORF  nigp={nigp}  freeze_epoch_nigp={args.freeze_epoch_nigp}  "
        f"qoi={qoi}  n_train={args.n_train}  n_val={args.n_val}  "
        f"log_qoi={LOG_SCALE_QOI}  logit_qoi={LOGIT_SCALE_QOI}  "
        f"off_floor={OFF_FLOOR_TRAIN_TASKS}  task_bands={TASK_BAND_CONFIG}"
    )

    run_s4_emit_sorf(
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
        logit_bounds=dict(S4_LOGIT_BOUNDS),
        log_offsets=LOG_OFFSETS,
        task_names=qoi,
        nigp=nigp,
        freeze_epoch_nigp=args.freeze_epoch_nigp,
        filter_valid_labels=args.filter_valid_labels,
        task_band_config=TASK_BAND_CONFIG,
        off_floor_train_tasks=OFF_FLOOR_TRAIN_TASKS,
    )
