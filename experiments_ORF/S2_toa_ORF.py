"""S2 11-QoI TOA benchmark with GPPlus ORF kernel (Woodbury inference)."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
_ORF_DIR = Path(__file__).resolve().parent
_RFF_DIR = _ROOT / "experiments_RFF"
_MTGPR_DIR = _ROOT / "experiments_RFFMTGPR"

# ---------------------------------------------------------------------------
# IDE RUN CONFIGURATION — edit these, then press Run.
# ---------------------------------------------------------------------------
QOI: list[str] | None = None  # None = all 11; e.g. ["algae", "fsnow"]
N_TRAIN = 16000
N_TEST = 5000
NUM_RFF = 1600
NUM_INITS = 1
NUM_EPOCHS = 1000
LR = 0.1
SEED = 42
DEVICE = "cuda"
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
SAVE_CHECKPOINT = True
RESPONSE_NOISE_PRIOR = True
NOISE_VAR_FRACTION = 0.001
NOISE_PRIOR_LOG_SCALE = 0.5
LOG_LEVEL = "INFO"
LOG_FILE: str | None = None
PARALLEL_VERBOSE = 10
LOG_EVERY_N_EPOCHS = 50
TRAINING_LOG = True
DATA_PATH: str | None = None  # None = snow_toa_simulations_20262107.nc
INPUT_VARIABLE = "toa_radiance"
X_TRANSFORM = "none"  # "none" | "log1p"
LOG_SCALE: bool | None = None
TASK_BAND_CONFIG: str | None = (
    "experiments_toa/configs/s2_task_bands_from_corr.json"
)  # None = s2_task_bands_default.json
# ---------------------------------------------------------------------------

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments_toa.paths import pin_toa_import_paths

pin_toa_import_paths(_MTGPR_DIR, _RFF_DIR, _ORF_DIR)

import gpplus
from experiments_toa.s2_cli import parse_example_indices, parse_task_names
from mtgpr_experiment_utils import DEFAULT_ADAM_KWARGS
from toa_s2_orf_base import run_s2_toa_orf


def run_s2_toa_orf_entry(**kwargs) -> dict:
    return run_s2_toa_orf(**kwargs)


if __name__ == "__main__":
    save_path = SAVE_PATH or (
        f"experiments_ORF/results/s2_toa_orf_{NUM_INITS}inits_numorf{NUM_RFF}_"
        f"lr{LR}_noisevarfrac{NOISE_VAR_FRACTION}_noisepriorlogscale{NOISE_PRIOR_LOG_SCALE}"
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
        f"ORF IDE config  prior={RESPONSE_NOISE_PRIOR}  "
        f"output_log_scale={LOG_SCALE}  x_transform={X_TRANSFORM}  qoi={QOI}"
    )

    run_s2_toa_orf(
        n_train=N_TRAIN,
        n_test=N_TEST,
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
        input_variable=INPUT_VARIABLE,
        task_names=parse_task_names(QOI),
        task_band_config=TASK_BAND_CONFIG,
        x_transform=X_TRANSFORM,
        log_scale=LOG_SCALE,
    )
