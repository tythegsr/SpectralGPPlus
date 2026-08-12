"""S2 11-QoI TOA benchmark with PCA + RFF/ORF/SORF independent GPs."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
_PCA_DIR = Path(__file__).resolve().parent
_RFF_DIR = _ROOT / "experiments_RFF"
_MTGPR_DIR = _ROOT / "experiments_RFFMTGPR"

# ---------------------------------------------------------------------------
# IDE RUN CONFIGURATION — edit these, then press Run.
# ---------------------------------------------------------------------------
QOI: list[str] | None = None  # None = all 11; e.g. ["algae", "fsnow"]
N_TRAIN = 16000
N_TEST = 5000
# int (shared p) or path to per-QoI JSON from s2_pca_svd_analysis.py
N_COMPONENTS: int | str = (
    "experiments_toa/configs/s2_task_pca_components_var99_subset.json"
)
RFF_SAMPLING = "sorf"  # "rff" | "orf" | "sorf"
NUM_RFF = 1600
NUM_INITS = 1
NUM_EPOCHS = 2000
LR = 0.01
SEED = 42
DEVICE = "cuda"
DTYPE = "float32"  # "float32" | "float64"
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
CORRECT_SORF = True
SAVE_CHECKPOINT = True
PCA_SVD_SOLVER = "randomized"  # "auto" | "full" | "randomized" | "arpack"
RESPONSE_NOISE_PRIOR = True
NOISE_VAR_FRACTION = 0.01
NOISE_PRIOR_LOG_SCALE = 0.5
LOG_LEVEL = "INFO"
LOG_FILE: str | None = None
PARALLEL_VERBOSE = 10
LOG_EVERY_N_EPOCHS = 50
TRAINING_LOG = True
DATA_PATH: str | None = None  # None = snow_toa_simulations_20262107.nc
INPUT_VARIABLE = "toa_radiance"
X_TRANSFORM = "none"  # "none" | "log1p"
# None = auto from NetCDF attrs for log; [] disables. Logit has no NetCDF auto.
LOG_SCALE_QOI: list[str] | None = ["algae", "dust", "grain_size", "liquid_water"]
LOGIT_SCALE_QOI: list[str] | None = ["cos_i", "aot"]
# Classic NIGP: learnable independent per-PCA-dim input noise (σ_x=10^SoftClamp(raw)).
# Init via initializer (library default Uniform(-4, -1) on raw_input_noise).
NIGP = False
# Freeze raw_input_noise for this many Adam epochs (0 = learn from start).
FREEZE_EPOCH_NIGP = 100
# Pair full-band PCA configs with s2_task_bands_all.json
TASK_BAND_CONFIG: str | None = (
    "experiments_toa/configs/s2_task_bands_from_corr.json"
)  # None = s2_task_bands_default.json
# ---------------------------------------------------------------------------

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments_toa.paths import pin_toa_import_paths

pin_toa_import_paths(_MTGPR_DIR, _RFF_DIR, _PCA_DIR)

import gpplus
from experiments_toa.s2_cli import parse_example_indices, parse_task_names
from mtgpr_experiment_utils import DEFAULT_ADAM_KWARGS
from toa_s2_pca_stgp_base import run_s2_toa_pca_stgp


def run_s2_toa_pca_rff_entry(**kwargs) -> dict:
    return run_s2_toa_pca_stgp(**kwargs)


if __name__ == "__main__":
    p_tag = N_COMPONENTS if isinstance(N_COMPONENTS, int) else "perQoI"
    nigp_str = "_nigp" if NIGP else ""
    save_path = SAVE_PATH or (
        f"experiments_PCA/results/s2_toa_pca_{RFF_SAMPLING}_{NUM_INITS}inits_p{p_tag}{nigp_str}"
    )
    log_file = LOG_FILE
    if log_file is None and DEVICE.startswith("cuda"):
        log_file = os.path.join(save_path, "train.log")
    gpplus.config.configure_logger(level=getattr(logging, LOG_LEVEL), log_to_file=log_file)
    dtype = torch.float32 if DTYPE == "float32" else torch.float64
    optimizer_kwargs = None
    if NUM_EPOCHS > 1 and LR is not None:
        optimizer_kwargs = {**DEFAULT_ADAM_KWARGS, "lr": LR}

    print(
        f"PCA-RFF IDE config  sampling={RFF_SAMPLING}  prior={RESPONSE_NOISE_PRIOR}  "
        f"log_qoi={LOG_SCALE_QOI} logit_qoi={LOGIT_SCALE_QOI}  "
        f"x_transform={X_TRANSFORM}  nigp={NIGP}  freeze_epoch_nigp={FREEZE_EPOCH_NIGP}  "
        f"qoi={QOI}"
    )

    run_s2_toa_pca_stgp(
        n_train=N_TRAIN,
        n_test=N_TEST,
        n_components=N_COMPONENTS,
        rff_sampling=RFF_SAMPLING,
        num_rff=NUM_RFF,
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
        parallel_verbose=PARALLEL_VERBOSE,
        training_verbose=TRAINING_LOG,
        log_every_n_epochs=LOG_EVERY_N_EPOCHS,
        save_checkpoint=SAVE_CHECKPOINT,
        response_noise_prior=RESPONSE_NOISE_PRIOR,
        noise_var_fraction=NOISE_VAR_FRACTION,
        noise_prior_log_scale=NOISE_PRIOR_LOG_SCALE,
        correct_sorf=CORRECT_SORF,
        pca_svd_solver=PCA_SVD_SOLVER,
        input_variable=INPUT_VARIABLE,
        task_names=parse_task_names(QOI),
        task_band_config=TASK_BAND_CONFIG,
        x_transform=X_TRANSFORM,
        log_scale_qoi=LOG_SCALE_QOI,
        logit_scale_qoi=LOGIT_SCALE_QOI,
        nigp=NIGP,
        freeze_epoch_nigp=FREEZE_EPOCH_NIGP,
    )
