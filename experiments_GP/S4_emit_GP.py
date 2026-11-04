"""S4 PCA + partitioned exact GPR (ensemble) on EMIT / snow-TOA NetCDF.

Trains per-QoI exact GPRs on PCA scores with train-partition ensembling
(primary mode ``full``), then evaluates ASD field validation in-process and
writes ASD plots + spectral-neighbor analysis.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
_GP_DIR = Path(__file__).resolve().parent
_PCA_DIR = _ROOT / "experiments_PCA"
_MTGPR_DIR = _ROOT / "experiments_RFFMTGPR"
_RFF_DIR = _ROOT / "experiments_RFF"

# ---------------------------------------------------------------------------
# IDE RUN CONFIGURATION — edit these, then press Run.
# ---------------------------------------------------------------------------
QOI: list[str] | None = ["cos_i", "grain_size", "dust", "algae", "cwv", "lwc", "aot"]  # subset; None = all 11 S4 tasks
N_TRAIN = 10000
N_VAL = 5000
PARTITION_SIZE = 2000
PARTITION_SHUFFLE = True
TOP_M_PARTITIONS = 5
SINGLE_PARTITION_INDEX = 0
NUM_INITS = 4
NUM_EPOCHS = 1  # 1 → LBFGS; >1 → Adam
LR = 0.01
SEED = 42
DEVICE = "cpu"
DTYPE = "float64"  # "float32" | "float64"
N_JOBS = 1
ARD = True
SAVE_PATH: str | None = None
PLOT = True
PLOT_POSTERIOR = True
REL_TOLERANCE = 0.01
POSTERIOR_N_EXAMPLES = 20
POSTERIOR_EXAMPLE_INDICES: str | None = None
EVAL_ASD = True
ASD_PATH: str | None = str(
    _ROOT / "experiments_toa" / "data 11 QoI" / "asd_validation_set.nc"
)
RESPONSE_NOISE_PRIOR = False
NOISE_VAR_FRACTION = 0.25
NOISE_PRIOR_LOG_SCALE = 0.5
LOG_SCALE_QOI: list[str] | None = []
LOGIT_SCALE_QOI: list[str] | None = []
LOG_OFFSETS: dict[str, float] | None = {}
NIGP = False
FREEZE_EPOCH_NIGP = 0
# Per-QoI PCA dims from s2_pca_svd_analysis.py on pure_flat_Sep04 (supervised / full).
N_PCA_COMPONENTS: int | str = str(
    _ROOT
    / "experiments_toa"
    / "configs"
    / "s4_task_pca_components_ridge_full_pure_flat_Sep04.json"
)
PCA_SVD_SOLVER = "randomized"
FILTER_VALID_LABELS = False
TASK_BAND_CONFIG: str | None = None
LOG_LEVEL = "INFO"
LOG_FILE: str | None = None
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

pin_toa_import_paths(_MTGPR_DIR, _RFF_DIR, _GP_DIR, _PCA_DIR)

import gpplus
from experiments_toa.s2_cli import (
    parse_example_indices,
    parse_n_pca_components,
    pca_components_path_tag,
)
from experiments_toa.s2_constants import S4_LOGIT_BOUNDS
from emit_s4_gp_base import parse_s4_task_names, run_s4_emit_gp
from gp_experiment_utils import DEFAULT_ADAM_KWARGS, DEFAULT_LBFGS_KWARGS


def run_s4_emit_gp_entry(**kwargs) -> dict:
    return run_s4_emit_gp(**kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "S4 PCA + partitioned exact GPR (ensemble; radiance + geometry aux; "
            "in-process ASD plots)"
        )
    )
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--n-val", type=int, default=N_VAL)
    parser.add_argument("--partition-size", type=int, default=PARTITION_SIZE)
    parser.add_argument(
        "--partition-shuffle",
        action=argparse.BooleanOptionalAction,
        default=PARTITION_SHUFFLE,
    )
    parser.add_argument("--top-m-partitions", type=int, default=TOP_M_PARTITIONS)
    parser.add_argument(
        "--single-partition-index", type=int, default=SINGLE_PARTITION_INDEX
    )
    parser.add_argument("--num-inits", type=int, default=NUM_INITS)
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--dtype", type=str, default=DTYPE, choices=("float32", "float64"))
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
    parser.add_argument(
        "--n-pca-components",
        type=str,
        default=str(N_PCA_COMPONENTS),
        help="Shared PCA dim (int) or per-QoI JSON path (default: Sep04 ridge_full)",
    )
    parser.add_argument("--nigp", action=argparse.BooleanOptionalAction, default=NIGP)
    parser.add_argument("--freeze-epoch-nigp", type=int, default=FREEZE_EPOCH_NIGP)
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

    try:
        n_pca_components = parse_n_pca_components(args.n_pca_components)
    except ValueError as exc:
        parser.error(str(exc))
    if n_pca_components is None:
        parser.error("S4_emit_GP requires --n-pca-components (int >= 1 or JSON path)")

    pca_str = pca_components_path_tag(n_pca_components)
    nigp_str = "_nigp" if args.nigp else ""
    band_str = "_taskbandconfig" if TASK_BAND_CONFIG else ""
    save_path = args.save_path or (
        f"experiments_GP/results/Sept10/s4_emit_gp_{args.num_inits}inits_"
        f"part{args.partition_size}_lr{args.lr}{pca_str}{nigp_str}{band_str}"
        f"_dtype{args.dtype}"
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
        print("CUDA unavailable; falling back to CPU.")
        device = "cpu"

    if args.num_epochs <= 1:
        optimizer_kwargs = dict(DEFAULT_LBFGS_KWARGS)
    else:
        optimizer_kwargs = {**DEFAULT_ADAM_KWARGS, "lr": args.lr}

    qoi = parse_s4_task_names(args.qoi if args.qoi is not None else QOI)
    print(
        f"S4 EMIT PCA+partitioned GP  partition_size={args.partition_size}  "
        f"pca={n_pca_components}  nigp={args.nigp}  qoi={qoi}  "
        f"n_train={args.n_train}  device={device}  dtype={args.dtype}"
    )

    run_s4_emit_gp(
        n_train=args.n_train,
        n_val=args.n_val,
        partition_size=args.partition_size,
        partition_shuffle=args.partition_shuffle,
        top_m_partitions=args.top_m_partitions,
        single_partition_index=args.single_partition_index,
        num_inits=args.num_inits,
        num_epochs=args.num_epochs,
        optimizer_kwargs=optimizer_kwargs,
        seed=args.seed,
        device=device,
        dtype=dtype,
        ard=ARD,
        save_path=save_path,
        n_jobs=None if args.n_jobs < 0 else args.n_jobs,
        plot_posterior=PLOT_POSTERIOR and PLOT and not args.no_plot,
        rel_tolerance=REL_TOLERANCE,
        posterior_n_examples=POSTERIOR_N_EXAMPLES,
        posterior_example_indices=parse_example_indices(POSTERIOR_EXAMPLE_INDICES),
        data_path=args.data_path,
        n_pca_components=n_pca_components,
        pca_svd_solver=PCA_SVD_SOLVER,
        response_noise_prior=RESPONSE_NOISE_PRIOR,
        noise_var_fraction=NOISE_VAR_FRACTION,
        noise_prior_log_scale=NOISE_PRIOR_LOG_SCALE,
        log_scale_qoi=LOG_SCALE_QOI,
        logit_scale_qoi=LOGIT_SCALE_QOI,
        logit_bounds=dict(S4_LOGIT_BOUNDS),
        log_offsets=LOG_OFFSETS,
        task_names=qoi,
        nigp=bool(args.nigp),
        freeze_epoch_nigp=args.freeze_epoch_nigp,
        filter_valid_labels=args.filter_valid_labels,
        task_band_config=TASK_BAND_CONFIG,
        eval_asd=bool(args.eval_asd),
        asd_path=args.asd_path,
        plot_asd=PLOT and not args.no_plot,
    )
