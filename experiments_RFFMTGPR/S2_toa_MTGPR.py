"""S2 11-QoI TOA benchmark with joint GPPlus RFFMTGPR (Woodbury)."""

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
QOI: list[str] | None = ["algae", "aot", "cos_i", "cwv", "dust", "grain_size", "liquid_water"]  # None = all 11; e.g. ["algae", "fsnow"]
N_TRAIN = 40000
N_TEST = 5000
RFF_SAMPLING = "sorf"  # "rff" | "orf" | "sorf"
NUM_RFF = 1600
NUM_INITS = 1
NUM_EPOCHS = 2000
LR = 0.1
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
RESPONSE_NOISE_PRIOR = False
NOISE_VAR_FRACTION = 0.01
NOISE_PRIOR_LOG_SCALE = 0.5
# None = auto from NetCDF attrs for log; [] disables. Logit has no NetCDF auto.
LOG_SCALE_QOI: list[str] | None = []
LOGIT_SCALE_QOI: list[str] | None = []
RANK_KERNEL = 1  # 0 = independent tasks (QoIs are nearly orthogonal)
# Classic NIGP on joint MT (diag task noise). Default off for baseline MT runs.
NIGP = True
FREEZE_EPOCH_NIGP = 100
LOG_LEVEL = "INFO"
LOG_FILE: str | None = None
PARALLEL_VERBOSE = 10
LOG_EVERY_N_EPOCHS = 50
TRAINING_LOG = True
DATA_PATH: str | None = "experiments_toa/data 11 QoI/snow_toa_fsnow_only_20260308.nc"  # None = snow_toa_simulations_20262107.nc
INPUT_VARIABLE = "toa_reflectance"
# Joint MT uses the band config "default" ranges (shared X), not per-QoI keeps.
TASK_BAND_CONFIG: str | None = (
    "experiments_toa/configs/s2_task_bands_all.json"
    # "experiments_toa/configs/s2_task_bands_from_corr_fsnow_only.json"
)  # None = s2_task_bands_default.json
# ---------------------------------------------------------------------------

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments_toa.paths import pin_toa_import_paths

pin_toa_import_paths(_MTGPR_DIR, _RFF_DIR)

import gpplus
from experiments_toa.s2_cli import parse_example_indices, parse_task_names
from mtgpr_experiment_utils import DEFAULT_ADAM_KWARGS
from toa_s2_mtgpr_base import run_s2_toa_mtgpr


def run_s2_toa_mtgpr_entry(**kwargs) -> dict:
    return run_s2_toa_mtgpr(**kwargs)


if __name__ == "__main__":
    save_path = SAVE_PATH or (
        f"experiments_RFFMTGPR/results/Aug08/s2_toa_mtgpr_{RFF_SAMPLING}_"
        f"{NUM_INITS}inits_numrff{NUM_RFF}_lr{LR}_dtype{DTYPE}"
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
        f"S2 MTGPR IDE config  sampling={RFF_SAMPLING}  prior={RESPONSE_NOISE_PRIOR}  "
        f"nigp={NIGP}  freeze_epoch_nigp={FREEZE_EPOCH_NIGP}  "
        f"qoi={QOI}  input={INPUT_VARIABLE}"
    )

    run_s2_toa_mtgpr(
        n_train=N_TRAIN,
        n_test=N_TEST,
        num_rff=NUM_RFF,
        rff_sampling=RFF_SAMPLING,  # type: ignore[arg-type]
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
        log_scale_qoi=LOG_SCALE_QOI,
        logit_scale_qoi=LOGIT_SCALE_QOI,
        rank_kernel=RANK_KERNEL,
        input_variable=INPUT_VARIABLE,
        task_names=parse_task_names(QOI),
        task_band_config=TASK_BAND_CONFIG,
        nigp=NIGP,
        freeze_epoch_nigp=FREEZE_EPOCH_NIGP,
    )
