"""S4 independent SVGP+NIGP on EMIT or snow-TOA NetCDF.

Fourth-example SVGP counterpart to ``experiments_SORF/S4_emit_SORF.py``: trains
with radiance + geometry aux (coszen, ele_km) predicting S4 QoIs. ``DATA_PATH``
may be either a processed EMIT file or a snow-TOA simulation file. One
``SVGPR`` per QoI with minibatch ELBO SGD (optional classic NIGP).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
_SVGP_DIR = Path(__file__).resolve().parent
_GP_DIR = _ROOT / "experiments_GP"
_MTGPR_DIR = _ROOT / "experiments_RFFMTGPR"
_RFF_DIR = _ROOT / "experiments_RFF"

# ---------------------------------------------------------------------------
# IDE RUN CONFIGURATION — edit these, then press Run.
# ---------------------------------------------------------------------------
QOI: list[str] | None = ["cos_i", "grain_size"]  # subset; None = all 11 S4 tasks
N_TRAIN = 20000
N_VAL = 5000
NUM_INDUCING = 512
LEARN_INDUCING_LOCATIONS = True
BATCH_SIZE = 1024
VARIATIONAL_LR: float | None = None
KL_BETA = 1.0
NUM_INITS = 1
NUM_EPOCHS = 300
LR = 0.05
ADAM_STOP_PATIENCE = 50
SEED = 42
DEVICE = "cuda"
DTYPE = "float32"  # "float32" | "float64"
PREDICT_CHUNK_SIZE = 4096
N_JOBS = 1
ARD = True
SAVE_PATH: str | None = None
MONITOR_VALIDATION = True
PLOT = True
PLOT_POSTERIOR = True
REL_TOLERANCE = 0.01
POSTERIOR_N_EXAMPLES = 20
POSTERIOR_EXAMPLE_INDICES: str | None = None
SAVE_CHECKPOINT = True
# After training: evaluate checkpoints on ASD validation + write plots.
EVAL_ASD = True
ASD_PATH: str | None = str(
    _ROOT / "experiments_toa" / "data 11 QoI" / "asd_validation_set.nc"
)
RESPONSE_NOISE_PRIOR = False
NOISE_VAR_FRACTION = 0.01
NOISE_PRIOR_LOG_SCALE = 0.5
LOG_SCALE_QOI: list[str] | None = []
LOGIT_SCALE_QOI: list[str] | None = []
LOG_OFFSETS: dict[str, float] | None = None
NIGP = True
FREEZE_EPOCH_NIGP = 50
N_PCA_COMPONENTS: int | None = None  # mutually exclusive with NIGP
PCA_SVD_SOLVER = "randomized"
FILTER_VALID_LABELS = False
OFF_FLOOR_TRAIN_TASKS: list[str] | None = []
TASK_BAND_CONFIG: str | None = None
LOG_LEVEL = "INFO"
LOG_FILE: str | None = None
PARALLEL_VERBOSE = 10
LOG_EVERY_N_EPOCHS = 10
VAL_LOG_EVERY_N_EPOCHS = 10
TRAINING_LOG = True
# EMIT processed or snow-TOA simulation NetCDF (schema auto-detected).
DATA_PATH: str | None = str(
    _ROOT
    / "experiments_toa"
    / "data 11 QoI"
    / "snow_toa_fsnow_90to100_20260309.nc"
)
# ---------------------------------------------------------------------------

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments_toa.paths import pin_toa_import_paths

pin_toa_import_paths(_MTGPR_DIR, _RFF_DIR, _GP_DIR, _SVGP_DIR)

import gpplus
from experiments_toa.s2_cli import parse_example_indices
from experiments_toa.s2_constants import S4_LOGIT_BOUNDS
from experiments_toa.s4_asd_posttrain import run_s4_asd_eval_and_plots
from emit_s4_svgp_base import parse_s4_task_names, run_s4_emit_svgp
from gp_experiment_utils import DEFAULT_ADAM_KWARGS


def run_s4_emit_svgp_entry(**kwargs) -> dict:
    return run_s4_emit_svgp(**kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "S4 independent SVGP+NIGP (minibatch inducing-point; "
            "radiance + geometry aux)"
        )
    )
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--n-val", type=int, default=N_VAL)
    parser.add_argument("--num-inducing", type=int, default=NUM_INDUCING)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--learn-inducing-locations",
        action=argparse.BooleanOptionalAction,
        default=LEARN_INDUCING_LOCATIONS,
    )
    parser.add_argument("--variational-lr", type=float, default=VARIATIONAL_LR)
    parser.add_argument("--kl-beta", type=float, default=KL_BETA)
    parser.add_argument("--num-inits", type=int, default=NUM_INITS)
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--adam-stop-patience", type=int, default=ADAM_STOP_PATIENCE)
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
        help="QoIs (default: IDE QOI / all S4 tasks)",
    )
    parser.add_argument("--nigp", action=argparse.BooleanOptionalAction, default=NIGP)
    parser.add_argument("--freeze-epoch-nigp", type=int, default=FREEZE_EPOCH_NIGP)
    parser.add_argument(
        "--n-pca-components",
        type=int,
        default=N_PCA_COMPONENTS,
        help="PCA dim (mutually exclusive with --nigp)",
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
    if nigp and args.n_pca_components is not None:
        parser.error("Cannot combine --nigp with --n-pca-components")

    nigp_str = "_nigp" if nigp else ""
    freeze_str = (
        f"_freezeepochnigp{args.freeze_epoch_nigp}"
        if nigp and args.freeze_epoch_nigp > 0
        else ""
    )
    pca_str = (
        f"_pca{args.n_pca_components}" if args.n_pca_components is not None else ""
    )
    band_str = "_taskbandconfig" if TASK_BAND_CONFIG else ""
    of_str = "_offfloor" if OFF_FLOOR_TRAIN_TASKS else ""
    save_path = args.save_path or (
        f"experiments_SVGP/results/s4_emit_svgp_{args.num_inits}inits_"
        f"M{args.num_inducing}_batch{args.batch_size}_lr{args.lr}"
        f"{nigp_str}{freeze_str}{pca_str}{band_str}{of_str}_dtype{args.dtype}"
    )
    log_file = LOG_FILE
    if log_file is None and args.device.startswith("cuda"):
        log_file = os.path.join(save_path, "train.log")
    gpplus.config.configure_logger(
        level=getattr(logging, LOG_LEVEL), log_to_file=log_file
    )
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; falling back to CPU (expect this to be slow).")
        device = "cpu"

    optimizer_kwargs = None
    if args.num_epochs > 1 and args.lr is not None:
        optimizer_kwargs = {**DEFAULT_ADAM_KWARGS, "lr": args.lr}

    qoi = parse_s4_task_names(args.qoi if args.qoi is not None else QOI)
    print(
        f"S4 EMIT independent SVGP  M={args.num_inducing}  batch={args.batch_size}  "
        f"nigp={nigp}  freeze_epoch_nigp={args.freeze_epoch_nigp}  "
        f"pca={args.n_pca_components}  qoi={qoi}  n_train={args.n_train}  "
        f"n_val={args.n_val}  log_qoi={LOG_SCALE_QOI}  logit_qoi={LOGIT_SCALE_QOI}  "
        f"device={device}  dtype={args.dtype}"
    )

    metrics = run_s4_emit_svgp(
        n_train=args.n_train,
        n_val=args.n_val,
        num_inducing=args.num_inducing,
        learn_inducing_locations=args.learn_inducing_locations,
        batch_size=args.batch_size,
        variational_lr=args.variational_lr,
        kl_beta=args.kl_beta,
        num_inits=args.num_inits,
        num_epochs=args.num_epochs,
        optimizer_kwargs=optimizer_kwargs,
        adam_stop_patience=args.adam_stop_patience,
        seed=args.seed,
        device=device,
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
        val_log_every_n_epochs=VAL_LOG_EVERY_N_EPOCHS,
        save_checkpoint=SAVE_CHECKPOINT,
        response_noise_prior=RESPONSE_NOISE_PRIOR,
        noise_var_fraction=NOISE_VAR_FRACTION,
        noise_prior_log_scale=NOISE_PRIOR_LOG_SCALE,
        log_scale_qoi=LOG_SCALE_QOI,
        logit_scale_qoi=LOGIT_SCALE_QOI,
        logit_bounds=dict(S4_LOGIT_BOUNDS),
        log_offsets=LOG_OFFSETS,
        task_names=qoi,
        nigp=nigp,
        freeze_epoch_nigp=args.freeze_epoch_nigp,
        n_pca_components=args.n_pca_components,
        pca_svd_solver=PCA_SVD_SOLVER,
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
                backend="svgp",
                device=device,
                asd_path=args.asd_path,
                checkpoint_title=metrics.get("title"),
                tasks=qoi,
                plot=PLOT and not args.no_plot,
                predict_chunk_size=args.predict_chunk_size,
                sim_path=args.data_path,
            )
