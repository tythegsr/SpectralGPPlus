"""R2: Concrete compressive strength with GPPlus SORF (Woodbury)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
_SORF_DIR = Path(__file__).resolve().parent
for p in (_ROOT, _SORF_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import gpplus
from r_uci_sorf_base import run_uci_sorf
from sorf_experiment_utils import DEFAULT_ADAM_KWARGS

# ---------------------------------------------------------------------------
# IDE RUN CONFIGURATION — edit these, then press Run.
# ---------------------------------------------------------------------------
TRAIN_FRAC = 2.0 / 3.0
NUM_RFF: int | None = 200  # D frequencies; None = min(512, max(64, n_train // 3))
NUM_INITS = 64
NUM_EPOCHS = 1  # 1 -> LBFGSScipy; >1 -> Adam
LR = 0.01
SEED = 42
DEVICE = "cpu"  # "cpu" | "cuda"
DTYPE = "float64"  # "float32" | "float64"
PREDICT_CHUNK_SIZE = 512
N_JOBS = -1  # -1 = all cores
ARD = True
CORRECT_SORF = True
SPECTRAL_KERNEL = "matern32"  # "rbf" | "matern32" (Matérn 3/2 via Student-t scale mixture)
SAVE_PATH: str | None = "experiments_SORF/results/concrete_compressive_strength_sorf"
MONITOR_VALIDATION = True
VAL_FRACTION = 0.1
LOG_EVERY_N_EPOCHS = 100  # validation metrics every N epochs
PLOT_VALIDATION = True
PLOT_TRUE_VS_PRED = True
# Classic NIGP: learnable independent per-dimension input noise (σ_x=10^SoftClamp(raw)).
NIGP = False
# Freeze raw_input_noise for this many Adam epochs (0 = learn from start).
FREEZE_EPOCH_NIGP = 100
# Paper-style outer-loop slope refreshes after NIGP unlock (None = every epoch).
NIGP_SLOPE_REFRESHES: int | None = None
# Catoni PAC-Bayes KL on learnable parameters (wraps Woodbury / NIGP MLL).
PAC_BAYES = True
PAC_BAYES_TEMPERATURE = 1.5
PAC_BAYES_PRIOR_STD = 0.5
PAC_BAYES_POSTERIOR_STD = 0.25
# ---------------------------------------------------------------------------

_DATASET = "concrete"


def main(
    *,
    train_frac: float = TRAIN_FRAC,
    num_rff: int | None = NUM_RFF,
    num_inits: int = NUM_INITS,
    num_epochs: int = NUM_EPOCHS,
    lr: float = LR,
    seed: int = SEED,
    device: str = DEVICE,
    dtype_name: str = DTYPE,
    predict_chunk_size: int = PREDICT_CHUNK_SIZE,
    n_jobs: int = N_JOBS,
    ard: bool = ARD,
    correct_sorf: bool = CORRECT_SORF,
    spectral_kernel: str = SPECTRAL_KERNEL,
    save_path: str | None = SAVE_PATH,
    monitor_validation: bool = MONITOR_VALIDATION,
    val_fraction: float = VAL_FRACTION,
    log_every_n_epochs: int = LOG_EVERY_N_EPOCHS,
    plot_validation: bool = PLOT_VALIDATION,
    plot_true_vs_pred: bool = PLOT_TRUE_VS_PRED,
    nigp: bool = NIGP,
    freeze_epoch_nigp: int = FREEZE_EPOCH_NIGP,
    nigp_slope_refreshes: int | None = NIGP_SLOPE_REFRESHES,
    pac_bayes: bool = PAC_BAYES,
    pac_bayes_temperature: float = PAC_BAYES_TEMPERATURE,
    pac_bayes_prior_std: float = PAC_BAYES_PRIOR_STD,
    pac_bayes_posterior_std: float = PAC_BAYES_POSTERIOR_STD,
) -> dict:
    gpplus.config.configure_logger()
    dtype = torch.float32 if dtype_name == "float32" else torch.float64
    jobs = None if n_jobs < 0 else n_jobs
    optimizer_kwargs = None
    if num_epochs > 1:
        optimizer_kwargs = {**DEFAULT_ADAM_KWARGS, "lr": lr}

    return run_uci_sorf(
        _DATASET,
        train_frac=train_frac,
        num_sorf=num_rff,
        seed=seed,
        num_inits=num_inits,
        num_epochs=num_epochs,
        device=device,
        dtype=dtype,
        save_path=save_path,
        ard=ard,
        correct_sorf=correct_sorf,
        spectral_kernel=spectral_kernel,
        predict_chunk_size=predict_chunk_size,
        n_jobs=jobs,
        optimizer_kwargs=optimizer_kwargs,
        monitor_validation=monitor_validation,
        val_fraction=val_fraction,
        log_every_n_epochs=log_every_n_epochs,
        plot_validation=plot_validation,
        plot_true_vs_pred=plot_true_vs_pred,
        nigp=nigp,
        freeze_epoch_nigp=freeze_epoch_nigp,
        nigp_slope_refreshes=nigp_slope_refreshes,
        pac_bayes=pac_bayes,
        pac_bayes_temperature=pac_bayes_temperature,
        pac_bayes_prior_std=pac_bayes_prior_std,
        pac_bayes_posterior_std=pac_bayes_posterior_std,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="R2 concrete compressive strength SORF")
    parser.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    parser.add_argument("--num-rff", type=int, default=NUM_RFF)
    parser.add_argument("--num-inits", type=int, default=NUM_INITS)
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--dtype", type=str, default=DTYPE, choices=("float32", "float64"))
    parser.add_argument("--predict-chunk-size", type=int, default=PREDICT_CHUNK_SIZE)
    parser.add_argument("--n-jobs", type=int, default=N_JOBS)
    parser.add_argument("--ard", action=argparse.BooleanOptionalAction, default=ARD)
    parser.add_argument("--correct-sorf", action=argparse.BooleanOptionalAction, default=CORRECT_SORF)
    parser.add_argument(
        "--spectral-kernel",
        type=str,
        default=SPECTRAL_KERNEL,
        choices=("rbf", "matern32"),
    )
    parser.add_argument("--save-path", type=str, default=SAVE_PATH)
    parser.add_argument(
        "--monitor-validation",
        action=argparse.BooleanOptionalAction,
        default=MONITOR_VALIDATION,
    )
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--log-every-n-epochs", type=int, default=LOG_EVERY_N_EPOCHS)
    parser.add_argument(
        "--plot-validation",
        action=argparse.BooleanOptionalAction,
        default=PLOT_VALIDATION,
    )
    parser.add_argument(
        "--plot-true-vs-pred",
        action=argparse.BooleanOptionalAction,
        default=PLOT_TRUE_VS_PRED,
    )
    parser.add_argument("--nigp", action=argparse.BooleanOptionalAction, default=NIGP)
    parser.add_argument("--freeze-epoch-nigp", type=int, default=FREEZE_EPOCH_NIGP)
    parser.add_argument(
        "--nigp-slope-refreshes",
        type=int,
        default=NIGP_SLOPE_REFRESHES,
    )
    parser.add_argument("--pac-bayes", action=argparse.BooleanOptionalAction, default=PAC_BAYES)
    parser.add_argument("--pac-bayes-temperature", type=float, default=PAC_BAYES_TEMPERATURE)
    parser.add_argument("--pac-bayes-prior-std", type=float, default=PAC_BAYES_PRIOR_STD)
    parser.add_argument("--pac-bayes-posterior-std", type=float, default=PAC_BAYES_POSTERIOR_STD)
    args = parser.parse_args()
    main(
        train_frac=args.train_frac,
        num_rff=args.num_rff,
        num_inits=args.num_inits,
        num_epochs=args.num_epochs,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        dtype_name=args.dtype,
        predict_chunk_size=args.predict_chunk_size,
        n_jobs=args.n_jobs,
        ard=args.ard,
        correct_sorf=args.correct_sorf,
        spectral_kernel=args.spectral_kernel,
        save_path=args.save_path,
        monitor_validation=args.monitor_validation,
        val_fraction=args.val_fraction,
        log_every_n_epochs=args.log_every_n_epochs,
        plot_validation=args.plot_validation,
        plot_true_vs_pred=args.plot_true_vs_pred,
        nigp=args.nigp,
        freeze_epoch_nigp=args.freeze_epoch_nigp,
        nigp_slope_refreshes=args.nigp_slope_refreshes,
        pac_bayes=args.pac_bayes,
        pac_bayes_temperature=args.pac_bayes_temperature,
        pac_bayes_prior_std=args.pac_bayes_prior_std,
        pac_bayes_posterior_std=args.pac_bayes_posterior_std,
    )
