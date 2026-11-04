"""S5 Additive Hetero-NIGP with scheduled Noise SORF (homo → NIGP → het).

Fifth example in the S1–S5 series: same EMIT / snow-TOA setup as S4, but
observation noise follows a fixed three-phase Adam schedule:

  Hom  [0, EPOCHS_HOM):                       σ_eff² = σ_y,hom² (scalar)
  NIGP [EPOCHS_HOM, EPOCHS_HOM+EPOCHS_HET):   σ_eff² = σ_y,hom² + σ_nigp²
  Het  [EPOCHS_HOM+EPOCHS_HET, NUM_EPOCHS):   σ_eff² = σ_y,het² + σ_nigp²
                                              (Noise SORF unlocked for refinement)

EPOCHS_HET is the NIGP-only window length before hetero unlock.
σ_nigp slopes refresh at NIGP unlock and at the last epoch only.
ASD eval/plots reuse S4 post-train helpers.
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
QOI: list[str] | None = ["cos_i", "grain_size", "dust", "algae", "cwv", "lwc", "aot"]
N_TRAIN = 10000
N_VAL = 5000
NUM_RFF = 2048
NUM_RFF_HET = 256
NUM_INITS = 1
NUM_EPOCHS = 1000
NUM_EPOCHS_BY_QOI: dict[str, int] | None = None
# Three-phase schedule (0-based): Hom -> NIGP -> Het refinement.
# EPOCHS_HET = NIGP-only window before unlocking Noise SORF.
# Require EPOCHS_HOM + EPOCHS_HET < NUM_EPOCHS.
EPOCHS_HOM = 200
EPOCHS_HET = 1
HETERO_MODE = "additive"
LR = 1e-2
SEED = 42
DEVICE = "cuda"
DTYPE = "float64"
PREDICT_CHUNK_SIZE = 0
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
SPECTRAL_KERNEL = "rbf"
SAVE_CHECKPOINT = True
EVAL_ASD = True
ASD_PATH: str | None = str(
    _ROOT / "experiments_toa" / "data 11 QoI" / "asd_validation_set.nc"
)
RESPONSE_NOISE_PRIOR = False
NOISE_VAR_FRACTION = 0.1
NOISE_PRIOR_LOG_SCALE = 0.05
LOG_SCALE_QOI: list[str] | None = []
LOG_OFFSETS: dict[str, float] | None = {}
LOGIT_SCALE_QOI: list[str] | None = []
NIGP = True  # required for S5 schedule (hetero_sorf forces nigp=True in base)
FILTER_VALID_LABELS = False
OFF_FLOOR_TRAIN_TASKS: list[str] | None = []
# TASK_BAND_CONFIG: str | None = (
#     "experiments_toa/configs/s4_task_bands_ASDfit.json"
# )
TASK_BAND_CONFIG: str | None = (
    None
)
LOG_LEVEL = "INFO"
LOG_FILE: str | None = None
PARALLEL_VERBOSE = 10
LOG_EVERY_N_EPOCHS = 50
VAL_LOG_EVERY_N_EPOCHS = 50
TRAINING_LOG = True
DATA_PATH: str | None = str(
    _ROOT
    / "experiments_toa"
    / "data 11 QoI"
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
from gpplus.utils.hetero_noise import hetero_folder_tag
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


def run_s5_emit_hetero_sorf_entry(**kwargs) -> dict:
    kwargs.setdefault("hetero_sorf", True)
    kwargs.setdefault("title_prefix", "S5_EMIT")
    return run_s4_emit_sorf(**kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "S5 EMIT Additive Hetero-NIGP (Noise SORF schedule: homo → NIGP → het)"
        )
    )
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--n-val", type=int, default=N_VAL)
    parser.add_argument("--num-rff", type=int, default=NUM_RFF)
    parser.add_argument("--num-rff-het", type=int, default=NUM_RFF_HET)
    parser.add_argument("--num-inits", type=int, default=NUM_INITS)
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument(
        "--num-epochs-by-qoi",
        nargs="+",
        default=None,
        metavar="NAME=EPOCHS",
        help="Per-QoI epoch overrides (e.g. aot=1500 algae=500); else NUM_EPOCHS_BY_QOI",
    )
    parser.add_argument("--epochs-hom", type=int, default=EPOCHS_HOM)
    parser.add_argument(
        "--epochs-het",
        type=int,
        default=EPOCHS_HET,
        help="NIGP-only window length after Hom before unlocking Noise SORF",
    )
    parser.add_argument(
        "--hetero-mode",
        type=str,
        default=HETERO_MODE,
        choices=("additive",),
        help="σ_eff combination mode (v1: additive only)",
    )
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--dtype", type=str, default=DTYPE, choices=("float32", "float64"))
    parser.add_argument(
        "--predict-chunk-size",
        type=int,
        default=PREDICT_CHUNK_SIZE,
        help="Test points per Woodbury predict chunk (0 = single batch)",
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
    parser.add_argument("--correct-sorf", action=argparse.BooleanOptionalAction, default=CORRECT_SORF)
    parser.add_argument(
        "--spectral-kernel",
        type=str,
        default=SPECTRAL_KERNEL,
        choices=("rbf", "matern32"),
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

    hetero_tag = hetero_folder_tag(
        hetero_mode=args.hetero_mode,
        epochs_hom=args.epochs_hom,
        epochs_het=args.epochs_het,
        num_rff_het=args.num_rff_het,
    )
    sk_str = f"_{args.spectral_kernel}" if args.spectral_kernel != "rbf" else ""
    band_str = "_taskbandconfig" if TASK_BAND_CONFIG else ""
    of_str = "_offfloor" if OFF_FLOOR_TRAIN_TASKS else ""
    save_path = args.save_path or (
        f"experiments_SORF/results/Sept24/s5_emit_hetero_sorf_{args.num_inits}inits_"
        f"numrff{args.num_rff}_lr{args.lr}{hetero_tag}_nigp"
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
        f"S5 Hetero-NIGP  mode={args.hetero_mode}  "
        f"epohom={args.epochs_hom} epohet={args.epochs_het} rffhet={args.num_rff_het}  "
        f"spectral_kernel={args.spectral_kernel}  "
        f"num_epochs={args.num_epochs}  epochs_by_qoi={epochs_by_qoi}  "
        f"qoi={qoi}  n_train={args.n_train}  n_val={args.n_val}  "
        f"log_qoi={LOG_SCALE_QOI}  logit_qoi={LOGIT_SCALE_QOI}  "
        f"off_floor={OFF_FLOOR_TRAIN_TASKS}  task_bands={TASK_BAND_CONFIG}"
    )

    metrics = run_s5_emit_hetero_sorf_entry(
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
        nigp=True,
        freeze_epoch_nigp=0,
        nigp_slope_refreshes=None,
        filter_valid_labels=args.filter_valid_labels,
        task_band_config=TASK_BAND_CONFIG,
        off_floor_train_tasks=OFF_FLOOR_TRAIN_TASKS,
        hetero_sorf=True,
        num_rff_het=args.num_rff_het,
        hetero_mode=args.hetero_mode,
        epochs_hom=args.epochs_hom,
        epochs_het=args.epochs_het,
        title_prefix="S5_EMIT",
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
