"""S2 11-QoI TOA benchmark with PCA + partitioned exact GPR."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
_PCA_DIR = Path(__file__).resolve().parent
_GP_DIR = _ROOT / "experiments_GP"
_MTGPR_DIR = _ROOT / "experiments_RFFMTGPR"
_RFF_DIR = _ROOT / "experiments_RFF"

# ---------------------------------------------------------------------------
# IDE RUN CONFIGURATION — edit these, then press Run.
# ---------------------------------------------------------------------------
# QOI: list[str] | None = ["algae", "aot", "cos_i", "cwv", "dust", "grain_size", "liquid_water"]  # None = all 11; e.g. ["algae", "fsnow"]
QOI: list[str] | None = ["algae"]  # None = all 11; e.g. ["algae", "fsnow"]
N_TRAIN = 16000
N_TEST = 5000
# int (shared p) or path to per-QoI JSON from s2_pca_svd_analysis.py
N_COMPONENTS: int | str = (
    "experiments_toa/configs/s2_task_pca_components_ridge_full.json"
)
PARTITION_SIZE = 2000
NUM_INITS = 4
NUM_EPOCHS = 1
LR = 0.01
SEED = 42
DEVICE = "cpu"
DTYPE = "float64"  # "float32" | "float64"
PREDICT_CHUNK_SIZE = 512
N_JOBS = 1
ARD = True
SAVE_PATH: str | None = None
MONITOR_VALIDATION = True
PLOT = True
PLOT_POSTERIOR = True
REL_TOLERANCE = 0.01
POSTERIOR_N_EXAMPLES = 20
POSTERIOR_EXAMPLE_INDICES: str | None = None  # e.g. "0,3,7" or None
PARTITION_SHUFFLE = True
TOP_M_PARTITIONS = 5
SINGLE_PARTITION_INDEX = 0
PCA_SVD_SOLVER = "randomized"  # "auto" | "full" | "randomized" | "arpack"
RESPONSE_NOISE_PRIOR = False
NOISE_VAR_FRACTION = 0.25
NOISE_PRIOR_LOG_SCALE = 0.5
LOG_LEVEL = "INFO"
LOG_FILE: str | None = None
DATA_PATH: str | None = "experiments_toa/data 11 QoI/snow_toa_fsnow_only_20260308.nc"  # None = snow_toa_simulations_20262107.nc
INPUT_VARIABLE = "toa_reflectance"
X_TRANSFORM = "none"  # "none" | "log1p"
# None = auto from NetCDF attrs for log; [] disables. Logit has no NetCDF auto.
LOG_SCALE_QOI: list[str] | None = []
LOGIT_SCALE_QOI: list[str] | None = []
# Classic NIGP: learnable independent per-PCA-dim input noise (σ_x=10^SoftClamp(raw)).
# Init via initializer (library default Uniform(-4, -1) on raw_input_noise).
NIGP = True
# Freeze raw_input_noise for this many Adam epochs (0 = learn from start).
FREEZE_EPOCH_NIGP = 100
# Pair full-band PCA configs with s2_task_bands_all.json
TASK_BAND_CONFIG: str | None = (
    "experiments_toa/configs/s2_task_bands_all.json"
    # "experiments_toa/configs/s2_task_bands_from_corr_fsnow_only.json"
)  # None = s2_task_bands_default.json
# ---------------------------------------------------------------------------

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments_toa.paths import pin_toa_import_paths

pin_toa_import_paths(_MTGPR_DIR, _RFF_DIR, _GP_DIR, _PCA_DIR)

import gpplus
from experiments_toa.s2_cli import parse_example_indices, parse_task_names
from gp_experiment_utils import DEFAULT_ADAM_KWARGS
from toa_s2_pca_gpr_base import run_s2_toa_pca_gpr


def run_s2_toa_pca_gpr_entry(**kwargs) -> dict:
    return run_s2_toa_pca_gpr(**kwargs)


if __name__ == "__main__":
    if RESPONSE_NOISE_PRIOR:
        noise_str = f"_noisevarfrac{NOISE_VAR_FRACTION}_noisepriorlogscale{NOISE_PRIOR_LOG_SCALE}"
    else:
        noise_str = ""

    if TASK_BAND_CONFIG is not None:
        task_band_str = "_taskbandconfig"
    else:
        task_band_str = ""

    x_tf_str = f"_x{X_TRANSFORM}" if X_TRANSFORM and X_TRANSFORM != "none" else ""
    nigp_str = "_nigp" if NIGP else ""
    p_tag = N_COMPONENTS if isinstance(N_COMPONENTS, int) else "perQoI"

    save_path = SAVE_PATH or (
        f"experiments_PCA/results/Aug05/NIGP_check/s2_toa_pca_gpr_{NUM_INITS}inits"
        f"_p{p_tag}_part{PARTITION_SIZE}_"
        f"lr{LR}{noise_str}{task_band_str}{x_tf_str}{nigp_str}_"
        f"dtype{DTYPE}"
    )
    gpplus.config.configure_logger(level=getattr(logging, LOG_LEVEL), log_to_file=LOG_FILE)
    dtype = torch.float32 if DTYPE == "float32" else torch.float64
    optimizer_kwargs = None
    if NUM_EPOCHS > 1 and LR is not None:
        optimizer_kwargs = {**DEFAULT_ADAM_KWARGS, "lr": LR}

    print(
        f"PCA-GPR IDE config  prior={RESPONSE_NOISE_PRIOR}  "
        f"log_qoi={LOG_SCALE_QOI} logit_qoi={LOGIT_SCALE_QOI}  "
        f"x_transform={X_TRANSFORM}  nigp={NIGP}  freeze_epoch_nigp={FREEZE_EPOCH_NIGP}  "
        f"qoi={QOI}"
    )

    run_s2_toa_pca_gpr(
        n_train=N_TRAIN,
        n_test=N_TEST,
        n_components=N_COMPONENTS,
        partition_size=PARTITION_SIZE,
        num_inits=NUM_INITS,
        num_epochs=NUM_EPOCHS,
        optimizer_kwargs=optimizer_kwargs,
        seed=SEED,
        device=DEVICE,
        dtype=dtype,
        ard=ARD,
        save_path=save_path,
        n_jobs=None if N_JOBS < 0 else N_JOBS,
        predict_chunk_size=PREDICT_CHUNK_SIZE,
        monitor_validation=MONITOR_VALIDATION,
        plot_validation=PLOT,
        plot_posterior=PLOT_POSTERIOR and PLOT,
        rel_tolerance=REL_TOLERANCE,
        posterior_n_examples=POSTERIOR_N_EXAMPLES,
        posterior_example_indices=parse_example_indices(POSTERIOR_EXAMPLE_INDICES),
        data_path=DATA_PATH,
        pca_svd_solver=PCA_SVD_SOLVER,
        response_noise_prior=RESPONSE_NOISE_PRIOR,
        noise_var_fraction=NOISE_VAR_FRACTION,
        noise_prior_log_scale=NOISE_PRIOR_LOG_SCALE,
        input_variable=INPUT_VARIABLE,
        task_names=parse_task_names(QOI),
        task_band_config=TASK_BAND_CONFIG,
        partition_shuffle=PARTITION_SHUFFLE,
        top_m_partitions=TOP_M_PARTITIONS,
        single_partition_index=SINGLE_PARTITION_INDEX,
        x_transform=X_TRANSFORM,
        log_scale_qoi=LOG_SCALE_QOI,
        logit_scale_qoi=LOGIT_SCALE_QOI,
        nigp=NIGP,
        freeze_epoch_nigp=FREEZE_EPOCH_NIGP,
    )
