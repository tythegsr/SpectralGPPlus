import gpplus
import torch
from gpplus.training.callbacks import (
    FinalParameterStorageCallback,
    IterationParameterCallback,
    EpochParameterCallback,
    LBFGSInnerMetricsCallbackV3,
)
from gpplus.training.optimizers import LBFGSScipy

X_STANDARDIZE_METHOD = 2  # 0=Gaussian (StandardScaler), 1=Uniform [0,1], 2=Uniform [-1,1]
STANDARDIZE_X = True
STANDARDIZE_Y = True

SF_kernel = None
SF_mean = None
SF_likelihood = None

NUM_RUNS = 20 # Number of training sets to run for each problem
TRAINER_LR = None # Using LBFGS
TRAINER_MIN_EPOCHS = 0  # Do not consider early stopping until at least this many epochs
TRAINER_NUM_EPOCHS = 1
TRAINER_NUM_INITS = 16 # Number of initializations to run for each training set
TRAINER_CONVERGENCE_PATIENCE = 10
# TRAINER_CHOLESKY_JITTER = 0
TRAINER_CHOLESKY_JITTER = 1e-6
TRAINER_MIN_LOSS_CHANGE = 1e-7
# TRAINER_OPTIMIZER_CLASS = torch.optim.Adam
TRAINER_OPTIMIZER_CLASS = gpplus.training.optimizers.LBFGSScipy
# TRAINER_OPTIMIZER_KWARGS = {"max_iter": 20, "tolerance_grad": 1e-5, "tolerance_change": 1e-9, "history_size": 10} #1
# TRAINER_OPTIMIZER_KWARGS = {"max_iter": 15000, "max_eval": 15000, "tolerance_grad": 1e-5, "tolerance_change": 1e-9, "history_size": 10} #2
TRAINER_OPTIMIZER_KWARGS = {"max_iter": 2000, "max_eval": 2500, "tolerance_grad": 1e-5, "tolerance_change": 1e-9, "history_size": 10} #2
# TRAINER_OPTIMIZER_KWARGS = {"max_iter": 20}   # 0 
TRAINER_LOG_LBFGS_INNER = True

TRAINER_INITIALIZER_CLASS = gpplus.training.parameter_initializer.DefaultParameterInitializer
TRAINER_GP_DEVICE = 'cpu'
TRAINER_AMP_DEVICE = 'cuda'
DTYPE_GP = torch.float64
DTYPE_PFN = torch.float32
NOISE_TYPE = "gaussian" # "gaussian" or "uniform" or "student_t"


# TRAINER_ANALYSIS = True # make settings into problems
# PLOT_METRICS = False

# True vs pred scatter + GP 1D / marginal plots for synthetic benchmarks (A4–A9);
# writes under <save_path>/plots/prediction_diagnostics_<title>/run_XXX/ (same title string as experiment JSON, path-sanitized)
PLOT_PREDICTION_DIAGNOSTICS = False
PREDICTION_DIAGNOSTIC_RUN_INDICES = (0,)  # 0-based run indices
PREDICTION_DIAGNOSTIC_MAX_MARGINAL_DIMS = 3  # partial dependence slices when dimensions > 1

SEED = 42
SEED_TRAINER = None

# Joblib worker count for the multi-init GP trainer. None lets the trainer pick
# its legacy default (max(1, cpu_count - 2) on CPU); set to 1 to force series.
TRAINER_N_JOBS = None
# BLAS/OpenMP threads cap per worker (and inside the main process when n_jobs==1).
# Default 1 so series and parallel runs see identical FP rounding.
TRAINER_INNER_MAX_NUM_THREADS = 1


def get_default_gp_callbacks(
    optimizer_class,
    callback_save_path: str | None = None,
    log_lbfgs_inner: bool = True,
    lbfgs_inner_extra_metrics: list | None = None,
):
    """
    Default GP callbacks for final experiments (A1–A9).

    Default GP callbacks for experiments_revisions_april (passed into train_eval_gp).
    """
    callbacks: list = [FinalParameterStorageCallback(save_file=None, verbose=False)]

    if optimizer_class is not None:
        # Only set epoch callback save path when logging to disk is desired.
        if callback_save_path is not None:
            epoch_save_file = f"{callback_save_path}/epoch_parameters.json"
        else:
            epoch_save_file = None

        # LBFGSScipy: per-iteration parameter logging + optional inner metrics
        if optimizer_class is LBFGSScipy or (
            isinstance(optimizer_class, type) and issubclass(optimizer_class, LBFGSScipy)
        ):
            callbacks.append(
                IterationParameterCallback(
                    save_file=None,
                    verbose=False,
                    save_every_n_iterations=20,
                )
            )
            if log_lbfgs_inner:
                callbacks.append(
                    LBFGSInnerMetricsCallbackV3(
                        log_record_every_n_iters=10,
                        log_metrics_every_n_iters=10,
                        log_nll=True,
                        log_nis=True,
                        log_loo=True,
                        log_kf=True,
                        log_residual_mse=True,
                        extra_metrics=lbfgs_inner_extra_metrics or [],
                    )
                )
        # Adam: per-epoch parameter logging
        elif optimizer_class is torch.optim.Adam or (
            isinstance(optimizer_class, type) and issubclass(optimizer_class, torch.optim.Adam)
        ):
            callbacks.append(
                EpochParameterCallback(
                    save_file=epoch_save_file,
                    verbose=False,
                    save_every_n_epochs=20,
                )
            )

    return callbacks

def MF_kernel(
    cont_cols=None,
    cat_cols=None,
    source_cols=None,
    cont_kernel=None,
    cat_kernel=None,
    source_kernel=None,
    cat_encoder=None,
    source_encoder=None,
    z_dim=2,
    fix_lengthscale_cat=False,
    fix_lengthscale_source=False,
    **kwargs
):
    """Factory function that creates and wraps the MF kernel."""
    return gpplus.kernels.LogScaleKernel(
        gpplus.kernels.MVMFKernel(
            cont_cols=cont_cols,
            cat_cols=cat_cols,
            source_cols=source_cols,
            cont_kernel=cont_kernel,
            cat_kernel=cat_kernel,
            source_kernel=source_kernel,
            cat_encoder=cat_encoder,
            source_encoder=source_encoder,
            z_dim=z_dim,
            fix_lengthscale_cat=fix_lengthscale_cat,
            fix_lengthscale_source=fix_lengthscale_source,
            **kwargs
        )
    )

# For current experiments, we are not using the following MF methods
# MF_mean = gpplus.means.MultiMean
# MF_likelihood = gpplus.likelihoods.MultiLikelihood
# MF_STANDARDIZATION_METHOD = 2 # 0: standardize all data according to all data, 1: standardize all data according to HF data only, 2: standardize each data source independently
