"""S2 11-QoI TOA benchmark with GPPlus LRNN / Deep Basis Kernel."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
_LRNN_DIR = Path(__file__).resolve().parent
_MTGPR_DIR = _ROOT / "experiments_RFFMTGPR"
_RFF_DIR = _ROOT / "experiments_RFF"

# ---------------------------------------------------------------------------
# IDE RUN CONFIGURATION — edit these, then press Run.
# ---------------------------------------------------------------------------
QOI: list[str] | None = ["algae", "aot", "cos_i", "cwv", "dust", "grain_size", "liquid_water"]  # None = all 11; e.g. ["algae", "fsnow"]
N_TRAIN = 16000
N_TEST = 5000
HIDDEN_DIMS = [128, 256, 512, 1024, 2048, 1024, 512]
FEATURE_RANK = 256
ACTIVATION = "tanh"  # "tanh" | "relu" | "gelu" | "identity"
VARIANCE_CORRECTION = True
NUM_INITS = 1
NUM_EPOCHS = 300
LR = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 42
DEVICE = "cuda"
DTYPE = "float64"  # "float32" | "float64"
PREDICT_CHUNK_SIZE = 512
N_JOBS = 1
SAVE_PATH: str | None = None
MONITOR_VALIDATION = True
PLOT = True
PLOT_POSTERIOR = True
REL_TOLERANCE = 0.01
POSTERIOR_N_EXAMPLES = 20
POSTERIOR_EXAMPLE_INDICES: str | None = None  # e.g. "0,3,7" or None
SAVE_CHECKPOINT = True
RESPONSE_NOISE_PRIOR = False
NOISE_VAR_FRACTION = 0.01
NOISE_PRIOR_LOG_SCALE = 0.01
LOG_LEVEL = "INFO"
LOG_FILE: str | None = None
PARALLEL_VERBOSE = 10
LOG_EVERY_N_EPOCHS = 50
TRAINING_LOG = True
DATA_PATH: str | None = "experiments_toa/data 11 QoI/snow_toa_fsnow_only_20262707.nc"
INPUT_VARIABLE = "toa_radiance"
X_TRANSFORM = "none"  # "none" | "log1p" (before UniformScaler / StandardScaler)
# None = auto from NetCDF attrs for log; [] disables. Logit has no NetCDF auto.
LOG_SCALE_QOI: list[str] | None = ["algae", "dust", "grain_size", "liquid_water"]
LOGIT_SCALE_QOI: list[str] | None = ["cos_i", "aot"]
TASK_BAND_CONFIG: str | None = (
    "experiments_toa/configs/s2_task_bands_from_corr_fsnow_only.json"
)  # None = s2_task_bands_default.json
# ---------------------------------------------------------------------------

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments_toa.paths import pin_toa_import_paths

pin_toa_import_paths(_MTGPR_DIR, _RFF_DIR, _LRNN_DIR)

import gpplus
from experiments_toa.s2_cli import parse_example_indices, parse_task_names
from toa_lrnn_base import DEFAULT_LRNN_ADAM_KWARGS
from toa_s2_lrnn_base import run_s2_toa_lrnn


def run_s2_toa_lrnn_entry(**kwargs) -> dict:
    return run_s2_toa_lrnn(**kwargs)


if __name__ == "__main__":
    hid_tag = "-".join(str(h) for h in HIDDEN_DIMS)
    save_path = SAVE_PATH or (
        f"experiments_LRNN/results/July27/s2_toa_lrnn_{NUM_INITS}inits_hiddims{hid_tag}_featr{FEATURE_RANK}"
    )
    log_file = LOG_FILE
    if log_file is None and DEVICE.startswith("cuda"):
        log_file = os.path.join(save_path, "train.log")
    gpplus.config.configure_logger(level=getattr(logging, LOG_LEVEL), log_to_file=log_file)
    dtype = torch.float32 if DTYPE == "float32" else torch.float64
    optimizer_kwargs = None
    if NUM_EPOCHS > 1:
        optimizer_kwargs = {
            **DEFAULT_LRNN_ADAM_KWARGS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
        }

    x_tf_str = f"_x{X_TRANSFORM}" if X_TRANSFORM and X_TRANSFORM != "none" else ""
    if SAVE_PATH is None and x_tf_str:
        save_path = f"{save_path}{x_tf_str}"

    print(
        f"LRNN IDE config  prior={RESPONSE_NOISE_PRIOR}  "
        f"frac={NOISE_VAR_FRACTION}  log_scale={NOISE_PRIOR_LOG_SCALE}  "
        f"log_qoi={LOG_SCALE_QOI} logit_qoi={LOGIT_SCALE_QOI}  x_transform={X_TRANSFORM}  qoi={QOI}"
    )

    run_s2_toa_lrnn(
        n_train=N_TRAIN,
        n_test=N_TEST,
        hidden_dims=HIDDEN_DIMS,
        feature_rank=FEATURE_RANK,
        activation=ACTIVATION,
        variance_correction=VARIANCE_CORRECTION,
        num_inits=NUM_INITS,
        num_epochs=NUM_EPOCHS,
        optimizer_kwargs=optimizer_kwargs,
        seed=SEED,
        device=DEVICE,
        dtype=dtype,
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
        log_scale_qoi=LOG_SCALE_QOI,
        logit_scale_qoi=LOGIT_SCALE_QOI,
    )
