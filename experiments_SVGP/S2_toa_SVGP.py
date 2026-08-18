"""S2 11-QoI TOA benchmark with inducing-point SVGP trained by minibatch SGD.

The exact-GP entry (``experiments_GP/S2_toa_GP.py``) is O(n^3) and stops being
usable past a few thousand training points. This one places ``M`` inducing
points and optimizes an ELBO that decomposes over observations, so cost per
step is O(BATCH_SIZE * M^2) and independent of N.

NIGP and PCA are mutually exclusive here: NIGP learns a per-input-dimension
sigma_x, which has no meaning once the bands are rotated and truncated into a
PCA basis. Pick one.
"""

from __future__ import annotations

import logging
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
QOI: list[str] | None = ["algae"]  # None = all 11
N_TRAIN = 20000
N_TEST = 5000
NUM_INITS = 1
NUM_EPOCHS = 300
LR = 0.05
SEED = 42
DEVICE = "cuda"
DTYPE = "float32"  # "float32" | "float64"

# --- SVGP ------------------------------------------------------------------
NUM_INDUCING = 512
LEARN_INDUCING_LOCATIONS = True
BATCH_SIZE = 1024
# Separate Adam lr for q(u) and the inducing locations; None = same as LR.
VARIATIONAL_LR: float | None = None
# KL weight; 1.0 is the true ELBO, smaller anneals the regularizer.
KL_BETA = 1.0
ADAM_STOP_PATIENCE = 50

# --- NIGP xor PCA (exactly one of these may be active) ----------------------
NIGP = False
FREEZE_EPOCH_NIGP = 50  # epochs of standard SVGP before sigma_x is unfrozen
N_PCA_COMPONENTS: int | None = 12  # None disables PCA
PCA_SVD_SOLVER = "randomized"
# ---------------------------------------------------------------------------

PREDICT_CHUNK_SIZE = 4096
N_JOBS = 1
ARD = True
SAVE_PATH: str | None = None
MONITOR_VALIDATION = True
PLOT = True
PLOT_POSTERIOR = True
REL_TOLERANCE = 0.01
POSTERIOR_N_EXAMPLES = 20
POSTERIOR_EXAMPLE_INDICES: str | None = None  # e.g. "0,3,7" or None
RESPONSE_NOISE_PRIOR = False
NOISE_VAR_FRACTION = 0.001
NOISE_PRIOR_LOG_SCALE = 0.1
LOG_LEVEL = "INFO"
LOG_FILE: str | None = None
PARALLEL_VERBOSE = 10
LOG_EVERY_N_EPOCHS = 10
TRAINING_LOG = True
DATA_PATH: str | None = None
INPUT_VARIABLE = "toa_radiance"
X_TRANSFORM = "none"  # "none" | "log1p"
# Warps off by default while prototyping the variational path.
LOG_SCALE_QOI: list[str] | None = []
LOGIT_SCALE_QOI: list[str] | None = []
TASK_BAND_CONFIG: str | None = "experiments_toa/configs/s2_task_bands_from_corr.json"

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_GP_DIR) not in sys.path:
    sys.path.insert(0, str(_GP_DIR))

from experiments_toa.paths import pin_toa_import_paths

pin_toa_import_paths(_MTGPR_DIR, _RFF_DIR, _GP_DIR)

import gpplus
from experiments_toa.s2_cli import parse_example_indices, parse_task_names
from gp_experiment_utils import DEFAULT_ADAM_KWARGS
from toa_s2_gp_base import run_s2_toa_gp


def run_s2_toa_svgp_entry(**kwargs) -> dict:
    return run_s2_toa_gp(svgp=True, **kwargs)


if __name__ == "__main__":
    save_path = SAVE_PATH or "experiments_SVGP/results/s2_toa_svgp"
    gpplus.config.configure_logger(
        level=getattr(logging, LOG_LEVEL),
        log_to_file=LOG_FILE,
    )
    dtype = torch.float32 if DTYPE == "float32" else torch.float64

    device = DEVICE
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; falling back to CPU (expect this to be slow).")
        device = "cpu"

    print(
        f"SVGP IDE config  M={NUM_INDUCING}  batch={BATCH_SIZE}  epochs={NUM_EPOCHS}  "
        f"nigp={NIGP}  pca={N_PCA_COMPONENTS}  device={device}  dtype={DTYPE}  qoi={QOI}"
    )

    run_s2_toa_svgp_entry(
        n_train=N_TRAIN,
        n_test=N_TEST,
        num_inits=NUM_INITS,
        num_epochs=NUM_EPOCHS,
        optimizer_kwargs={**DEFAULT_ADAM_KWARGS, "lr": LR},
        seed=SEED,
        device=device,
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
        response_noise_prior=RESPONSE_NOISE_PRIOR,
        noise_var_fraction=NOISE_VAR_FRACTION,
        noise_prior_log_scale=NOISE_PRIOR_LOG_SCALE,
        input_variable=INPUT_VARIABLE,
        task_names=parse_task_names(QOI),
        task_band_config=TASK_BAND_CONFIG,
        x_transform=X_TRANSFORM,
        log_scale_qoi=LOG_SCALE_QOI,
        logit_scale_qoi=LOGIT_SCALE_QOI,
        n_pca_components=N_PCA_COMPONENTS,
        pca_svd_solver=PCA_SVD_SOLVER,
        num_inducing=NUM_INDUCING,
        learn_inducing_locations=LEARN_INDUCING_LOCATIONS,
        batch_size=BATCH_SIZE,
        variational_lr=VARIATIONAL_LR,
        kl_beta=KL_BETA,
        nigp=NIGP,
        freeze_epoch_nigp=FREEZE_EPOCH_NIGP,
        adam_stop_patience=ADAM_STOP_PATIENCE,
    )
