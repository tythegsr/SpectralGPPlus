"""S2 11-QoI TOA benchmark with GPPlus SORF kernel (Woodbury inference)."""

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
# QOI: list[str] | None = ["algae", "aot", "cos_i", "cwv", "dust", "grain_size", "liquid_water"] # None = all 11; e.g. ["algae", "fsnow"]
QOI: list[str] | None = None
N_TRAIN = 100000
N_TEST = 10000
NUM_RFF = 800
NUM_INITS = 1  # total random starts
INIT_BATCH_SIZE = 1  # concurrent GPU wave size (VRAM knob only; must divide NUM_INITS)
NUM_EPOCHS = 2000
LR = 0.1
# Adam early stop: epochs with no train-loss improvement (not validation).
STOP_PATIENCE = 40
SEED = 42
DEVICE = "cuda" # "cpu" | "cuda"
DTYPE = "float64"  # "float32" | "float64"
PREDICT_CHUNK_SIZE = 512
N_JOBS = 1
TRAIN_MODE = "independent"  # "independent" | "batched"
ARD = True
SAVE_PATH: str | None = None
MONITOR_VALIDATION = True
PLOT = True
PLOT_POSTERIOR = True
REL_TOLERANCE = 0.01
POSTERIOR_N_EXAMPLES = 20
POSTERIOR_EXAMPLE_INDICES: str | None = None  # e.g. "0,3,7" or None
CORRECT_SORF = True
SPECTRAL_KERNEL = "rbf"  # "rbf" | "matern32" (Matérn 3/2 via Student-t scale mixture)
SAVE_CHECKPOINT = True
RESPONSE_NOISE_PRIOR = False
NOISE_VAR_FRACTION = 5e-5
NOISE_PRIOR_LOG_SCALE = 1.0
# Parameter init overrides for raw_noise / raw_lengthscale / raw_outputscale / raw_input_noise.
# None or {} = library defaults. Example:
#   {"raw_noise": {"method": "uniform", "lower": -5.0, "upper": -1.0},
#    "raw_lengthscale": {"method": "normal", "mean": -1.0, "std": 1.0},
#    "raw_outputscale": {"method": "normal", "mean": -2.0, "std": 1.5},
#    "raw_input_noise": {"method": "uniform", "lower": -4.0, "upper": -1.0}}  # NIGP default
# July29 LOOK (adjusted_init_ls) used mean=-4; library default mean=-2 stalls aot.
INITIALIZER_PARAMETER_CONFIGS: dict | None = None
    # {
    # "raw_lengthscale": {"method": "normal", "mean": -4.0, "std": 1.0}
    # "raw_lengthscale": {"method": "normal", "mean": -2.64, "std": 1.2}
    # }
USE_ABLATION_INIT_OVERRIDES = False  # if True and INITIALIZER_PARAMETER_CONFIGS empty, use ablation winners
LOG_LEVEL = "INFO"
LOG_FILE: str | None = None
PARALLEL_VERBOSE = 10
LOG_EVERY_N_EPOCHS = 50
TRAINING_LOG = True
DATA_PATH: str | None = "experiments_toa/data 11 QoI/snow_toa_fsnow_70to100_20261208.nc"  # None = snow_toa_simulations_20262107.nc
INPUT_VARIABLE = "toa_reflectance"
X_TRANSFORM = "none"  # "none" | "log1p" (before UniformScaler / StandardScaler)
# None = auto from NetCDF attrs for log; [] disables. Logit has no NetCDF auto.
# LOG_SCALE_QOI: list[str] | None = ["algae", "dust", "grain_size"]
LOG_SCALE_QOI: list[str] | None = []
LOGIT_SCALE_QOI: list[str] | None = []
# Soft probabilistic bounds: same length/order as QOI. None = no bound on that side.
# Do not set bounds on log/logit-warped QoIs.
BOUND_MIN: list[float | None] | None = None
BOUND_MAX: list[float | None] | None = None
BOUND_PENALTY_K = 2.0
BOUND_PENALTY_LAMBDA = 0.0
BOUND_PENALTY_LAMBDA_LEARNABLE = False
BOUND_PENALTY_LAM_MIN = 0.0
BOUND_PENALTY_ALPHA = 10.0
BOUND_PENALTY_MAX_POINTS: int | None = 4096
# Catoni PAC-Bayes KL on learnable parameters (wraps base / bound-penalized MLL).
PAC_BAYES = False
PAC_BAYES_TEMPERATURE = 2.55
PAC_BAYES_PRIOR_STD = 0.75
PAC_BAYES_POSTERIOR_STD = 0.5
# Classic NIGP: learnable independent per-dimension input noise (σ_x=10^SoftClamp(raw)).
# Init via INITIALIZER_PARAMETER_CONFIGS["raw_input_noise"] (library default Uniform(-4, -1)).
NIGP = True
# Freeze raw_input_noise for this many Adam epochs (0 = learn from start).
FREEZE_EPOCH_NIGP = 100
# Paper-style outer-loop slope refreshes after NIGP unlock (None = every epoch).
# E.g. 20 with FREEZE=100 and NUM_EPOCHS=2000 → ~20 ∇μ recomputes over the NIGP phase.
NIGP_SLOPE_REFRESHES: int | None = 20
# Freeze likelihood noise for this many Adam epochs (0 = off). Skips Woodbury tr(Λ⁻¹).
FREEZE_EPOCH_NOISE = 0
# Mean: "constant" | "neural". NeuralMean + TRAIN_MODE="batched" is unsupported.
# Final MLP output is always dims=1 (Identity); NEURAL_MEAN_HIDDEN are hidden widths only.
MEAN_TYPE = "constant"  # "constant" | "neural"
NEURAL_MEAN_HIDDEN = [64, 16]
NEURAL_MEAN_ACTIVATION = "tanh"  # relu|tanh|gelu|silu|identity
TASK_BAND_CONFIG: str | None = (
    "experiments_toa/configs/s2_task_bands_all.json"
    # "experiments_toa/configs/s2_task_bands_from_corr_fsnow_only.json"
)  # None = s2_task_bands_default.json

# ---------------------------------------------------------------------------

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments_toa.paths import pin_toa_import_paths

pin_toa_import_paths(_MTGPR_DIR, _RFF_DIR, _SORF_DIR)

import gpplus
from experiments_toa.s2_cli import parse_example_indices, parse_task_names
from mtgpr_experiment_utils import DEFAULT_ADAM_KWARGS
from s2_sorf_defaults import sorf_defaults_from_recommendations
from toa_s2_sorf_base import run_s2_toa_sorf


def run_s2_toa_sorf_entry(**kwargs) -> dict:
    return run_s2_toa_sorf(**kwargs)


if __name__ == "__main__":
    _sorf_defs = sorf_defaults_from_recommendations()

    init_pcs = dict(INITIALIZER_PARAMETER_CONFIGS) if INITIALIZER_PARAMETER_CONFIGS else None
    if not init_pcs and USE_ABLATION_INIT_OVERRIDES:
        init_pcs = _sorf_defs.get("initializer_parameter_configs") or None
        if init_pcs == {}:
            init_pcs = None
    
    if RESPONSE_NOISE_PRIOR:
        noise_str = f"_noisevarfrac{NOISE_VAR_FRACTION}_noisepriorlogscale{NOISE_PRIOR_LOG_SCALE}"
    else:
        noise_str = ""

    if TASK_BAND_CONFIG is not None:
        task_band_str = f"_taskbandconfig"
    else:
        task_band_str = ""

    x_tf_str = f"_x{X_TRANSFORM}" if X_TRANSFORM and X_TRANSFORM != "none" else ""
    sk_str = f"_{SPECTRAL_KERNEL}" if SPECTRAL_KERNEL else ""
    nigp_str = f"_nigp" if NIGP else ""
    pac_bayes_str = f"_pacbayes" if PAC_BAYES else ""
    freeze_epoch_nigp_str = f"_freezeepochnigp{FREEZE_EPOCH_NIGP}" if FREEZE_EPOCH_NIGP > 0 and NIGP else ""
    slope_refreshes_str = (
        f"_sloperefreshes{NIGP_SLOPE_REFRESHES}"
        if NIGP and NIGP_SLOPE_REFRESHES is not None
        else ""
    )
    nnmean_str = (
        f"_nnmean{'x'.join(str(int(d)) for d in NEURAL_MEAN_HIDDEN)}"
        if MEAN_TYPE == "neural"
        else ""
    )

    ibs_str = f"_ibs{INIT_BATCH_SIZE}" if INIT_BATCH_SIZE < NUM_INITS else ""
    save_path = SAVE_PATH or (
        f"experiments_SORF/results/Aug12/s2_toa_sorf_{NUM_INITS}inits"
        f"{ibs_str}_numrff{NUM_RFF}_"
        f"lr{LR}{noise_str}{task_band_str}{x_tf_str}{sk_str}{nigp_str}{pac_bayes_str}"
        f"{freeze_epoch_nigp_str}{slope_refreshes_str}{nnmean_str}_"
        f"dtype{DTYPE}"
    )
    log_file = LOG_FILE
    if log_file is None and DEVICE.startswith("cuda"):
        log_file = os.path.join(save_path, "train.log")

    gpplus.config.configure_logger(
        level=getattr(logging, LOG_LEVEL),
        log_to_file=log_file,
    )
    dtype = torch.float32 if DTYPE == "float32" else torch.float64
    optimizer_kwargs = None
    if NUM_EPOCHS > 1 and LR is not None:
        optimizer_kwargs = {**DEFAULT_ADAM_KWARGS, "lr": LR}

    print(
        f"SORF IDE config  prior={RESPONSE_NOISE_PRIOR}  "
        f"frac={NOISE_VAR_FRACTION}  log_scale={NOISE_PRIOR_LOG_SCALE}  "
        f"log_qoi={LOG_SCALE_QOI} logit_qoi={LOGIT_SCALE_QOI}  "
        f"bound_min={BOUND_MIN} bound_max={BOUND_MAX}  "
        f"bound_lambda_learnable={BOUND_PENALTY_LAMBDA_LEARNABLE}  x_transform={X_TRANSFORM}  "
        f"pac_bayes={PAC_BAYES}  nigp={NIGP}  freeze_epoch_nigp={FREEZE_EPOCH_NIGP}  "
        f"nigp_slope_refreshes={NIGP_SLOPE_REFRESHES}  "
        f"stop_patience={STOP_PATIENCE}  "
        f"mean_type={MEAN_TYPE}  neural_mean_hidden={NEURAL_MEAN_HIDDEN}  "
        f"neural_mean_activation={NEURAL_MEAN_ACTIVATION}  "
        f"init_overrides={init_pcs or {}}  qoi={QOI}  "
        f"spectral_kernel={SPECTRAL_KERNEL}"
    )

    run_s2_toa_sorf(
        n_train=N_TRAIN,
        n_test=N_TEST,
        num_rff=NUM_RFF,
        num_inits=NUM_INITS,
        init_batch_size=INIT_BATCH_SIZE,
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
        initializer_parameter_configs=init_pcs,
        correct_sorf=CORRECT_SORF,
        spectral_kernel=SPECTRAL_KERNEL,
        input_variable=INPUT_VARIABLE,
        task_names=parse_task_names(QOI),
        task_band_config=TASK_BAND_CONFIG,
        x_transform=X_TRANSFORM,
        log_scale_qoi=LOG_SCALE_QOI,
        logit_scale_qoi=LOGIT_SCALE_QOI,
        bound_min=BOUND_MIN,
        bound_max=BOUND_MAX,
        bound_penalty_k=BOUND_PENALTY_K,
        bound_penalty_lambda=BOUND_PENALTY_LAMBDA,
        bound_penalty_lambda_learnable=BOUND_PENALTY_LAMBDA_LEARNABLE,
        bound_penalty_lam_min=BOUND_PENALTY_LAM_MIN,
        bound_penalty_alpha=BOUND_PENALTY_ALPHA,
        bound_penalty_max_points=BOUND_PENALTY_MAX_POINTS,
        pac_bayes=PAC_BAYES,
        pac_bayes_temperature=PAC_BAYES_TEMPERATURE,
        pac_bayes_prior_std=PAC_BAYES_PRIOR_STD,
        pac_bayes_posterior_std=PAC_BAYES_POSTERIOR_STD,
        nigp=NIGP,
        freeze_epoch_nigp=FREEZE_EPOCH_NIGP,
        nigp_slope_refreshes=NIGP_SLOPE_REFRESHES,
        freeze_epoch_noise=FREEZE_EPOCH_NOISE,
        adam_stop_patience=STOP_PATIENCE,
        train_mode=TRAIN_MODE,
        mean_type=MEAN_TYPE,
        neural_mean_hidden=NEURAL_MEAN_HIDDEN,
        neural_mean_activation=NEURAL_MEAN_ACTIVATION,
    )
