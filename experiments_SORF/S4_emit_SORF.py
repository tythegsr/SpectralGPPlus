"""S4 independent SORF+NIGP (Woodbury RFFGPR) on EMIT or snow-TOA NetCDF.

Fourth example in the S1/S2/S3/S4 series: trains with radiance + geometry aux
(coszen, ele_km) predicting S4 QoIs. ``DATA_PATH`` may be either a
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
# QOI: list[str] | None = ["cos_i"]  # subset; None = all 11 S4 tasks
QOI: list[str] | None = ["cos_i", "grain_size", "dust", "algae", "cwv", "lwc", "aot"]  # subset; None = all 11 S4 tasks
N_TRAIN = 10000
N_VAL = 5000
NUM_RFF = 2400
NUM_INITS = 1
NUM_EPOCHS = 1000  # default for all QoIs
# Optional per-QoI overrides; unspecified QoIs use NUM_EPOCHS.
# Example: {"aot": 1500, "algae": 500}
NUM_EPOCHS_BY_QOI: dict[str, int] | None = {"cos_i": 1000, "grain_size": 1000, "dust": 1000, "algae": 201, "cwv": 1000, "lwc": 1000, "aot": 201}
# NUM_EPOCHS_BY_QOI: dict[str, int] | None = {"cos_i": 52}
LR = 1e-2
SEED = 42
DEVICE = "cuda"
DTYPE = "float64"  # "float32" | "float64"
PREDICT_CHUNK_SIZE = 0  # 0 = single batch over all test points; use 4096–8192 if OOM
# None = auto: CPU → min(n_tasks, cpu_count); single GPU → 1; multi-GPU → n_gpus
PREDICT_N_JOBS: int | None = None
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
SPECTRAL_KERNEL = "rbf"  # "rbf" | "matern32" (Matérn 3/2 via Student-t scale mixture)
SAVE_CHECKPOINT = True
# After training: evaluate checkpoints on ASD validation + write plots.
EVAL_ASD = True
ASD_PATH: str | None = str(
    _ROOT / "experiments_toa" / "data 11 QoI" / "asd_validation_set.nc"
)
RESPONSE_NOISE_PRIOR = False
NOISE_VAR_FRACTION = 0.1
NOISE_PRIOR_LOG_SCALE = 0.05
LOG_SCALE_QOI: list[str] | None = []
# LOGIT_SCALE_QOI: list[str] | None = ["fsnow"]
LOG_OFFSETS: dict[str, float] | None = {} # default dust C=1.0 from S2_LOG_OFFSETS
# LOG_SCALE_QOI: list[str] | None = []
LOGIT_SCALE_QOI: list[str] | None = []
# LOG_OFFSETS: dict[str, float] | None = None
NIGP = True
FREEZE_EPOCH_NIGP = 200
# Paper-style outer-loop slope refreshes after NIGP unlock (None = every epoch).
# E.g. 20 with FREEZE=200 and NUM_EPOCHS=1000 → ~20 ∇μ recomputes over the NIGP phase
# (uses each QoI's resolved epoch count when NUM_EPOCHS_BY_QOI is set).
NIGP_SLOPE_REFRESHES: int | None = 10
FILTER_VALID_LABELS = False
OFF_FLOOR_TRAIN_TASKS: list[str] | None = []
TASK_BAND_CONFIG: str | None = (
    "experiments_toa/configs/s4_task_bands_ASDfit.json"
)
LOG_LEVEL = "INFO"
LOG_FILE: str | None = None
PARALLEL_VERBOSE = 10
LOG_EVERY_N_EPOCHS = 50
VAL_LOG_EVERY_N_EPOCHS = 50
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
    / "snow_toa_fsnow_pure_flat_Sep09.nc"
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
from emit_s4_sorf_base import parse_s4_task_names, resolve_num_epochs_by_task, run_s4_emit_sorf
from experiments_toa.s4_asd_posttrain import run_s4_asd_eval_and_plots


def _parse_epochs_by_qoi(items: list[str] | None) -> dict[str, int] | None:
    """Parse CLI entries like ``aot=1500`` / ``algae:500`` into a dict."""
    if not items:
        return None
    out: dict[str, int] = {}
    for raw in items:
        token = str(raw).strip()
        if "=" in token:
            name, _, epochs_s = token.partition("=")
        elif ":" in token:
            name, _, epochs_s = token.partition(":")
        else:
            raise ValueError(
                f"Invalid --num-epochs-by-qoi entry {raw!r}; expected NAME=EPOCHS"
            )
        name = name.strip()
        if not name:
            raise ValueError(f"Empty QoI name in --num-epochs-by-qoi entry {raw!r}")
        out[name] = int(epochs_s.strip())
    return out


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
    parser.add_argument(
        "--num-epochs-by-qoi",
        nargs="+",
        default=None,
        metavar="NAME=EPOCHS",
        help="Per-QoI epoch overrides (e.g. aot=1500 algae=500); else NUM_EPOCHS_BY_QOI",
    )
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--dtype", type=str, default=DTYPE, choices=("float32", "float64"))
    parser.add_argument(
        "--predict-chunk-size",
        type=int,
        default=PREDICT_CHUNK_SIZE,
        help="Test points per Woodbury predict chunk (0 = single batch; try 4096–8192 if OOM)",
    )
    parser.add_argument(
        "--predict-n-jobs",
        type=int,
        default=-1 if PREDICT_N_JOBS is None else PREDICT_N_JOBS,
        help=(
            "Parallel QoI predicts after training (-1 = auto: CPU all QoIs, "
            "single GPU 1, multi-GPU n_gpus)"
        ),
    )
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
    parser.add_argument(
        "--nigp-slope-refreshes",
        type=int,
        default=NIGP_SLOPE_REFRESHES,
        metavar="N",
        help=(
            "Recompute NIGP ∇μ slopes N times after NIGP unlock "
            "(default: every epoch when omitted/None)"
        ),
    )
    parser.add_argument("--correct-sorf", action=argparse.BooleanOptionalAction, default=CORRECT_SORF)
    parser.add_argument(
        "--spectral-kernel",
        type=str,
        default=SPECTRAL_KERNEL,
        choices=("rbf", "matern32"),
        help="RFF spectral density: rbf or matern32 (Student-t scale mixture)",
    )
    parser.add_argument(
        "--filter-valid-labels",
        action=argparse.BooleanOptionalAction,
        default=FILTER_VALID_LABELS,
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--eval-asd",
        action=argparse.BooleanOptionalAction,
        default=EVAL_ASD,
        help="After training, evaluate on ASD validation and write plots",
    )
    parser.add_argument(
        "--asd-path",
        type=str,
        default=ASD_PATH,
        help="ASD validation NetCDF (default: asd_validation_set.nc)",
    )
    args = parser.parse_args()

    nigp = bool(args.nigp)
    nigp_str = "_nigp" if nigp else ""
    freeze_str = (
        f"_freezeepochnigp{args.freeze_epoch_nigp}" if nigp and args.freeze_epoch_nigp > 0 else ""
    )
    slope_refreshes_str = (
        f"_sloperefreshes{args.nigp_slope_refreshes}"
        if nigp and args.nigp_slope_refreshes is not None
        else ""
    )
    band_str = "_taskbandconfig" if TASK_BAND_CONFIG else ""
    of_str = "_offfloor" if OFF_FLOOR_TRAIN_TASKS else ""
    sk_str = f"_{args.spectral_kernel}" if args.spectral_kernel != "rbf" else ""
    save_path = args.save_path or (
        f"experiments_SORF/results/Sept23_1350nm_ASD/s4_emit_sorf_{args.num_inits}inits_"
        f"numrff{args.num_rff}_lr{args.lr}{nigp_str}{freeze_str}{slope_refreshes_str}"
        f"{band_str}{of_str}{sk_str}_dtype{args.dtype}"
    )
    log_file = LOG_FILE
    if log_file is None and args.device.startswith("cuda"):
        log_file = os.path.join(save_path, "train.log")
    gpplus.config.configure_logger(level=getattr(logging, LOG_LEVEL), log_to_file=log_file)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    qoi = parse_s4_task_names(args.qoi if args.qoi is not None else QOI)
    epochs_by_qoi = (
        _parse_epochs_by_qoi(args.num_epochs_by_qoi)
        if args.num_epochs_by_qoi is not None
        else NUM_EPOCHS_BY_QOI
    )
    epochs_resolved = resolve_num_epochs_by_task(qoi, args.num_epochs, epochs_by_qoi)
    optimizer_kwargs = None
    if max(epochs_resolved.values()) > 1 and args.lr is not None:
        optimizer_kwargs = {**DEFAULT_ADAM_KWARGS, "lr": args.lr}

    print(
        f"S4 EMIT independent SORF  nigp={nigp}  freeze_epoch_nigp={args.freeze_epoch_nigp}  "
        f"nigp_slope_refreshes={args.nigp_slope_refreshes}  "
        f"spectral_kernel={args.spectral_kernel}  "
        f"num_epochs={args.num_epochs}  epochs_by_qoi={epochs_by_qoi}  "
        f"qoi={qoi}  n_train={args.n_train}  n_val={args.n_val}  "
        f"log_qoi={LOG_SCALE_QOI}  logit_qoi={LOGIT_SCALE_QOI}  "
        f"off_floor={OFF_FLOOR_TRAIN_TASKS}  task_bands={TASK_BAND_CONFIG}"
    )

    metrics = run_s4_emit_sorf(
        n_train=args.n_train,
        n_val=args.n_val,
        num_rff=args.num_rff,
        rff_sampling="sorf",
        num_inits=args.num_inits,
        num_epochs=args.num_epochs,
        num_epochs_by_task=epochs_by_qoi,
        optimizer_kwargs=optimizer_kwargs,
        seed=args.seed,
        device=args.device,
        dtype=dtype,
        ard=ARD,
        save_path=save_path,
        n_jobs=None if args.n_jobs < 0 else args.n_jobs,
        predict_chunk_size=args.predict_chunk_size,
        predict_n_jobs=None if args.predict_n_jobs < 0 else args.predict_n_jobs,
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
        val_log_every_n_epochs=VAL_LOG_EVERY_N_EPOCHS,
        save_checkpoint=SAVE_CHECKPOINT,
        response_noise_prior=RESPONSE_NOISE_PRIOR,
        noise_var_fraction=NOISE_VAR_FRACTION,
        noise_prior_log_scale=NOISE_PRIOR_LOG_SCALE,
        correct_sorf=args.correct_sorf,
        spectral_kernel=args.spectral_kernel,
        log_scale_qoi=LOG_SCALE_QOI,
        logit_scale_qoi=LOGIT_SCALE_QOI,
        logit_bounds=dict(S4_LOGIT_BOUNDS),
        log_offsets=LOG_OFFSETS,
        task_names=qoi,
        nigp=nigp,
        freeze_epoch_nigp=args.freeze_epoch_nigp,
        nigp_slope_refreshes=args.nigp_slope_refreshes,
        filter_valid_labels=args.filter_valid_labels,
        task_band_config=TASK_BAND_CONFIG,
        off_floor_train_tasks=OFF_FLOOR_TRAIN_TASKS,
    )

    if args.eval_asd:
        if not SAVE_CHECKPOINT:
            print("ASD eval skipped: SAVE_CHECKPOINT is False")
        else:
            run_s4_asd_eval_and_plots(
                save_path,
                backend="sorf",
                device=args.device,
                asd_path=args.asd_path,
                checkpoint_title=metrics.get("title"),
                tasks=qoi,
                plot=PLOT and not args.no_plot,
                predict_chunk_size=args.predict_chunk_size,
                sim_path=args.data_path,
            )
