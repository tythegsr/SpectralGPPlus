import cProfile
import json
import pstats
import sys
import time
from datetime import datetime
from pathlib import Path

import gpytorch
import numpy as np
import torch
import torch.nn.functional as F
from gpplus.utils.metrics_functions import analyze_metrics, format_metric_value, plot_metrics
from gpplus.utils.onehot_encode_data import encode_qual_data, learn_encodings
from tabpfn import TabPFNRegressor

import gpplus
from gpplus.utils import set_seed, train_eval_gp, train_eval_PFN

# Ensure this folder is on sys.path so local imports (e.g. load_experimental_data.py) work
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import logging

import defaults
from load_experimental_data import generate_ackley_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Configure gpplus logger to show INFO messages
gpplus.config.configure_logger(level=logging.INFO)
# SEEK kernel (the only substantive change vs Kian baseline)
from gpplus.kernels import GaussianKernel, LogScaleKernel, PowerExponentialKernel, SEEKKernel, SEEKKernelTrunkHead


def ackley_GPvsPFN_SEEK(
    num_folds: int = defaults.NUM_FOLDS,
    num_test: int = 5000,
    train_size: int = 10,  # total training size is train_size * input_dim
    dimensions: int = 5,  # Ackley function dimensions
    num_runs: int = defaults.TRAINER_NUM_RUNS,
    num_epochs: int = defaults.TRAINER_NUM_EPOCHS,
    lr: float = defaults.TRAINER_LR,
    convergence_patience: int = defaults.TRAINER_CONVERGENCE_PATIENCE,
    min_loss_change: float = defaults.TRAINER_MIN_LOSS_CHANGE,
    optimizer_class=defaults.TRAINER_OPTIMIZER_CLASS,
    initializer_class=defaults.TRAINER_INITIALIZER_CLASS,
    gp_device: str = defaults.TRAINER_GP_DEVICE,
    amp_device: str = defaults.TRAINER_AMP_DEVICE,
    save_path: str | None = "./results/Ackley",
    title: str | None = None,
    standardize_X: bool = True,
    standardize_y: bool = True,
    x_standardize_method: int = 2,  # 0=Gaussian (StandardScaler), 1=Uniform [0,1], 2=Uniform [-1,1]
    noise_train: float = 0.0,
    noise_test: float = 0.0,
    noise_type: str = "gaussian",
    seed: int = defaults.SEED,
    seed_trainer: int | None = defaults.SEED_TRAINER,
    gp_dtype: torch.dtype = defaults.DTYPE_GP,
    pfn_dtype: torch.dtype = defaults.DTYPE_PFN,
    trainer_info: bool = True,
    run_models: str | None = None,  # None=run both, 'gp'=GP only, 'pfn'=PFN only
    kernel_type: str | None = None,  # None=default, 'Gaussian', 'PowerExponential', 'Matern'
    V2: bool = False,  # Whether to use log(y+1) transformation
    x_bounds: list[float] = [-5, 10],  # Ackley function bounds
    *,
    seek_use_bias: bool = True,
    seek_weight_trunk_layer_config: dict | None = None,
    seek_weight_head_config: dict | None = None,
    seek_bias_trunk_layer_config: dict | None = None,
    seek_bias_head_config: dict | None = None,
    seek_share_bias_trunk: bool = False,
):
    """Ackley Function: TabPFN vs GPPlus with SEEKKernel.

    Notes:
    - Data is generated from the Ackley function benchmark.
    - The GP covariance is SEEKKernel with an ensemble of continuous base kernels.
    - This is a more complex problem that should show neural network parameter changes.
    """

    if run_models == "pfn":
        # If only running PFN, skip GP restarts entirely.
        num_runs = 0

    if title is None:
        title = f"Ackley_{dimensions}D_{train_size}D_{num_runs}runs_noiseTest{noise_test}_noiseTrain{noise_train}"
    else: 
        title = f"Ackley_{dimensions}D_{title}_{train_size}D_{num_runs}runs_noiseTest{noise_test}_noiseTrain{noise_train}"

    # Generate data
    set_seed(seed)
    
    print(f" GP Device: {gp_device}")
    print(f" TabPFN Device: {amp_device}")
    regressor = TabPFNRegressor(device=amp_device)
    regressor = None
    if save_path is not None:
        plot_save_path = f"{save_path}/plots"
    else:
        plot_save_path = None

    # Calculate total samples needed
    train_per_fold = train_size * dimensions
    total_train = num_folds * train_per_fold
    total_samples = num_test + total_train
    
    # Generate all unique Sobol samples at once
    print(f"Generating {total_samples} unique Sobol samples for {dimensions}D Ackley function\n\tTest samples: {num_test} / Train samples: {total_train}")
    X_train_all, y_train_all, X_test_all, y_test_all = generate_ackley_data(
        n_train=total_train,
        n_test=num_test,
        dimensions=dimensions,
        x_bounds=x_bounds,
        train_noise=noise_train, 
        test_noise=noise_test,
        noise_type=noise_type,
        seed=seed,
        V2=V2,
    )

    input_dim = int(X_train_all.shape[1])
    train_per_fold = train_size * input_dim
    total_train = num_folds * train_per_fold

    X = torch.cat([X_test_all, X_train_all], dim=0)

    print("="*10)
    print(f"{title}: TabPFN vs GP Comparison")
    print("="*10)

    # Prepare encoded data once from already loaded X, y (no extra CSV/label encoding)
    qual_dict = learn_encodings(X)
    X_enc_train_all, cont_cols, cat_cols, source_cols = encode_qual_data(
        X_train_all, qual_dict=qual_dict, source_col=None
    )

    # Normalize to plain Python lists for SEEK
    cont_cols = list(cont_cols) if cont_cols is not None else []
    cat_cols = cat_cols or []
    source_cols = source_cols or []

    # If everything looks continuous, encode_qual_data can return empty cont_cols in some edge cases.
    if not cont_cols:
        cont_cols = list(range(input_dim))

    TabPFN_metrics: list[dict] = []
    GPPlus_metrics: list[dict] = []
    GPTrainer_info: list[dict] = []
    gp_model_info = None
    tabpfn_model_info = None
    X_scaling_type = "None"  # Initialize in case GP is not run

    # Randomize across all training data, then split across folds
    all_indices = torch.randperm(total_train)
    train_indices_2d = all_indices.reshape(num_folds, train_per_fold)
        
    total_start_time = time.time()
    for i in range(num_folds):
        fold_seed = seed_trainer if seed_trainer is not None else (seed + i)
        print(f"\n{'='*20} {title} FOLD {i+1}/{num_folds}: {fold_seed} {'='*20}")

        # Get training indices for this fold
        fold_train_indices = train_indices_2d[i]

        X_train = X_train_all[fold_train_indices]
        y_train = y_train_all[fold_train_indices]

        # =============================================================================
        # GP Section (SEEK)
        # =============================================================================
        if run_models in [None, "gp"]:
            print(f"\n--- {title} GP(SEEK) Training ---")

            # Reuse PFN split, convert to torch
            X_train = X_train.detach().clone().to(dtype=gp_dtype)
            X_test = X_test_all.detach().clone().to(dtype=gp_dtype)
            y_train = y_train.detach().clone().to(dtype=gp_dtype)
            y_test = y_test_all.detach().clone().to(dtype=gp_dtype)

            # Determine X scaling type
            if standardize_X:
                if x_standardize_method == 0:
                    Xscaler = gpplus.utils.StandardScaler()
                    X_scaling_type = "StandardScaler (Gaussian)"
                elif x_standardize_method == 1:
                    Xscaler = gpplus.utils.UniformScaler(scale_to_neg_one=False)
                    X_scaling_type = "UniformScaler [0, 1]"
                elif x_standardize_method == 2:
                    Xscaler = gpplus.utils.UniformScaler(scale_to_neg_one=True)
                    X_scaling_type = "UniformScaler [-1, 1]"
                else:
                    raise ValueError(f"x_standardize_method must be 0, 1, or 2, got {x_standardize_method}")
                Xscaler.fit(X_train[:, cont_cols])
                X_train[:, cont_cols] = Xscaler.transform(X_train[:, cont_cols])
                X_test[:, cont_cols] = Xscaler.transform(X_test[:, cont_cols])
            else:
                X_scaling_type = "None"

            # Normalize y
            Yscaler = gpplus.utils.StandardScaler()
            Yscaler.fit(y_train)
            y_train_mean = Yscaler.mean 
            y_train_std = Yscaler.std
            y_train_normal = Yscaler.transform(y_train)

            # --- kernel configuration ---
            cont_cols_seek = cont_cols
            if cont_cols_seek is None or len(cont_cols_seek) == 0:
                cont_cols_seek = list(range(X_train.shape[1]))
            cont_dim = len(cont_cols_seek)

            # --- SEEK kernel definition (uses gpplus/kernels/seek_kernel.py trunk/head API) ---
            # Choose base continuous kernels to ensemble inside SEEK.
            if kernel_type == "PowerExponential":
                continuous_kernels = [gpplus.kernels.PowerExponentialKernel(ard_num_dims=cont_dim)]
            elif kernel_type == "Matern":
                continuous_kernels = [gpplus.kernels.MaternKernel(nu=2.5, ard_num_dims=cont_dim)]
            else:
                # Default (and also kernel_type == "Gaussian" or None): Gaussian base kernel
                continuous_kernels = [GaussianKernel(ard_num_dims=cont_dim)]
                # continuous_kernels = [GaussianKernel(ard_num_dims=cont_dim), GaussianKernel(ard_num_dims=cont_dim), GaussianKernel(ard_num_dims=cont_dim)]
            # Default SEEK trunk/head configs (can be overridden via function args)
            act = torch.nn.Softplus

            kernel_mod = SEEKKernelTrunkHead(
                cont_cols=cont_cols_seek,
                cat_cols=cat_cols,
                source_cols=source_cols,
                continuous_kernels=continuous_kernels,
                use_bias=seek_use_bias,
                use_log_scale_kernel=True,
                use_exponential_wrapper=True,
                normalize=True,  # Keep L2 normalization ON for numerical stability and faster convergence
                act=act,
                share_bias_trunk=seek_share_bias_trunk,
                trunk_layer_config={
                    0: {"dims": 8, "activation": act},
                    # 1: {"dims": 2, "activation": act},
                },
                bias_trunk_layer_config={
                    0: {"dims": 1, "activation": act},
                },
                weight_head_configs=[
                    {"dims": 1, "activation": torch.nn.Identity},
                    # {"dims": 1, "activation": torch.nn.Identity},
                    # {"dims": 1, "activation": torch.nn.Identity},
                ],
                bias_head_config={"dims": 1, "activation": torch.nn.Identity},
            )

            # kernel_mod = SEEKKernel(
            #     cont_cols=cont_cols_seek,
            #     cat_cols=cat_cols,
            #     source_cols=source_cols,
            #     continuous_kernels=continuous_kernels,
            #     use_bias=seek_use_bias,
            #     use_exponential_wrapper=True,  # Set to True to enable exponential wrapper (but may cause outputscale collapse)
            #     weight_layer_config={
            #         0: {"dims": 8, "activation": act},
            #         1: {"dims": 2, "activation": act},
            #         # 2: {"dims": 16, "activation": act},
            #         # 3: {"dims": 4, "activation": act},
            #         # 4: {"dims": 2, "activation": act},
            #     },
            #     bias_layer_config={
            #         0: {"dims": 8, "activation": act},
            #         1: {"dims": 2, "activation": act},
            #         # 2: {"dims": 2, "activation": act},
            #     },
            # )

            # Create GP model
            model = gpplus.models.GPR(
                X_train,
                y_train_normal if standardize_y else y_train,
                kernel_module=kernel_mod,
                mean_module=gpytorch.means.ZeroMean(),
                likelihood=defaults.SF_likelihood,
            )
            if (i == 0) or (i == num_folds - 1):
                print(f"X_train: {X_train.shape}")
                print(f"X_test: {X_test.shape}")
                print(f"y_test mean: {y_test.mean().item()} / y_test std: {y_test.std().item()}")
                print(model)

            # Create trainer (train_eval_gp always returns 4 values)
            # Pass fold_index and callback_save_path to set on callbacks
            callback_save_path = save_path if save_path is not None else None
            gp_metric, y_pred_gp, output_std_gp, gp_trainer_info = train_eval_gp(
                model,
                X_test,
                y_test,
                num_epochs=num_epochs,
                seed=fold_seed,
                num_runs=num_runs,
                lr=lr,
                convergence_patience=convergence_patience,
                min_loss_change=min_loss_change,
                optimizer_class=optimizer_class,
                initializer_class=initializer_class,
                device=gp_device,
                y_train_mean=y_train_mean if standardize_y else None,
                y_train_std=y_train_std if standardize_y else None,
                source_cols=source_cols,
                trainer_info=trainer_info,
                fold_index=i,  # Pass fold index to set on callbacks
                callback_save_path=callback_save_path,  # Pass save path for callbacks
            )

            # Record and print GP metrics (inside the GP branch so gp_metric is always defined)
            GPPlus_metrics.append(gp_metric)
            if gp_trainer_info:
                gp_trainer_info["fold"] = i + 1
                gp_trainer_info["metrics"] = gp_metric
                GPTrainer_info.append(gp_trainer_info)

        # Always print GP results for this fold, including jitter and jitter_max if present
        print(f"\nGP Results (Fold {i+1}/{num_folds})")
        for k, v in gp_metric.items():
            if isinstance(v, (int, float, np.floating)):
                if np.isnan(v):
                    print(f"  {k}: NaN")
                else:
                    # Use scientific notation for jitter/jitter_max/noise via shared formatter
                    print(f"  {k}: {format_metric_value(str(k), float(v), precision=4)}")
            else:
                print(f"  {k}: {v}")

        # =============================================================================
        # TabPFN Section
        # =============================================================================
        if run_models in [None, "pfn"]:
            print(f"\n--- {title} TabPFN Training ---")

            if regressor is None:
                raise RuntimeError(
                    "TabPFN requested (run_models=None/'pfn') but `regressor` is None. "
                    "Instantiate a TabPFN regressor (e.g. VanillaDirectTabPFNRegressor) and pass it in."
                )
            
            tabpfn_metric, y_pred_tabpfn, output_std_tabpfn = train_eval_PFN(
                X_train,
                X_test,
                y_train_normal if standardize_y else y_train,
                y_test,
                amp_device=amp_device,
                amp_dtype=pfn_dtype,
                regressor=regressor,
                y_train_mean=y_train_mean if standardize_y else None,
                y_train_std=y_train_std if standardize_y else None,
                source_cols=source_cols,
            )
            
            TabPFN_metrics.append(tabpfn_metric)

            # Print results for this fold
            print(f"\nTabPFN Results (Fold {i+1}/{num_folds})")
            for k, v in tabpfn_metric.items():
                if isinstance(v, (int, float, np.floating)):
                    if np.isnan(v):
                        print(f"  {k}: NaN")
                    else:
                        print(f"  {k}: {format_metric_value(str(k), float(v), precision=4)}")
                else:
                    print(f"  {k}: {v}")
        
        # Collect model info from first fold
        if i == 0:
            # Calculate y_test mean and std (once, since test data is fixed)
            y_test_stats = {
                "y_test_mean": float(y_test_all.mean().item()),
                "y_test_std": float(y_test_all.std().item())
            }

            model_params_dict = {}
            if run_models in [None, "gp"] and GPPlus_metrics:
                # Extract model parameters from gp_metric (lengthscales, outputscale, noise, jitter, etc.)
                model_params_dict = {}
                # Include jitter_max so we can see the maximum jitter used in the best run
                param_keys = ["lengthscale", "outputscale", "noise", "jitter", "jitter_max", "raw_noise"]
                for key, value in gp_metric.items():
                    # Include any key that contains parameter names
                    if any(param_key in key.lower() for param_key in param_keys):
                        model_params_dict[key] = value
                    # Also include best_epoch if present
                    elif key in ["best_epoch", "best_loss"]:
                        model_params_dict[key] = value

            gp_model_info = None
            tabpfn_model_info = None
            if run_models in [None, "gp"]:
                gp_model_info = {
                    "model_str": str(model),
                    "kernel_type": kernel_type,
                    "kernel": "SEEKKernelTrunkHead" if kernel_type is None else f"SEEKKernelTrunkHead_base={kernel_type}",
                    "cat_cols": cat_cols,
                    "cont_cols": cont_cols,
                    "source_cols": source_cols,
                    "qual_dict": qual_dict,
                    "input_dim": X_train.shape[1],
                    "train_samples": int(train_per_fold),
                    "test_samples": num_test,
                    "standardize_X": standardize_X,
                    "standardize_y": standardize_y,
                    "x_standardize_method": x_standardize_method,
                    "X_scaling_type": X_scaling_type,
                    "dtype": str(gp_dtype),
                    "device": str(gp_device),
                    "num_epochs": num_epochs,
                    "num_runs": num_runs,
                    "lr": lr,
                    "optimizer": optimizer_class.__name__,
                    "convergence_patience": convergence_patience,
                    "min_loss_change": min_loss_change,
                    "initializer": initializer_class.__name__ if initializer_class else None,
                    "dimensions": dimensions,
                    "x_bounds": x_bounds,
                    "V2": V2,
                    **y_test_stats,
                    "num_folds": num_folds,
                    "seed": seed,
                    "seed_trainer": seed_trainer,
                    **model_params_dict,
                }

            if run_models in [None, "pfn"] and regressor is not None:
                tabpfn_model_info = {
                    "model_path": regressor.model_path,
                    "fit_mode": regressor.fit_mode,
                    "device": str(regressor.device_),
                    "inference_precision": regressor.inference_precision,
                    "random_state": regressor.random_state,
                    "use_autocast": regressor.use_autocast_,
                    "forced_inference_dtype": str(regressor.forced_inference_dtype_) if regressor.forced_inference_dtype_ else None,
                }

        fold_time = time.time() - total_start_time
        print(f"\nFold {i+1}/{num_folds} completed in {fold_time:.2f} seconds")

    # =============================================================================
    # Final Results Summary
    # =============================================================================
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)

    # Summaries via analyze_metrics
    TabPFN_summary = (
        analyze_metrics(TabPFN_metrics, print_summary=True, label="TabPFN", title=title)
        if run_models in [None, "pfn"]
        else None
    )
    GPPlus_summary = (
        analyze_metrics(GPPlus_metrics, print_summary=True, label="GP(SEEK)", title=title)
        if run_models in [None, "gp"]
        else None
    )

    if save_path is not None:
        if run_models is None:
            plot_metrics(
                TabPFN_metrics,
                GPPlus_metrics,
                labels=["TabPFN", "GP(SEEK)"],
                title=title,
                save_path=plot_save_path,
            )

        out_dir = Path(save_path)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            file_prefix = run_models if run_models is not None else "gpVpfn"

            combined_data: dict = {}
            if run_models in [None, "gp"]:
                combined_data["gp_data"] = {
                    "summary": GPPlus_summary,
                    "metrics": GPPlus_metrics,
                    "gp_model_info": gp_model_info,
                }
                if trainer_info and GPTrainer_info:
                    combined_data["gp_trainer_info"] = GPTrainer_info

            if run_models in [None, "pfn"]:
                combined_data["tabpfn_data"] = {
                    "summary": TabPFN_summary,
                    "metrics": TabPFN_metrics,
                    "pfn_model_info": tabpfn_model_info,
                }

            # Append defaults.py source for reproducibility (same folder as this script)
            _defaults_path = Path(__file__).resolve().parent / "defaults.py"
            if _defaults_path.is_file():
                combined_data["defaults_py"] = _defaults_path.read_text(encoding="utf-8")

            (out_dir / f"{file_prefix}_{title}.json").write_text(json.dumps(combined_data, indent=2))
        except Exception:
            pass
    print(f"\nTotal experiment time for {num_folds} folds: {time.time() - total_start_time:.2f}s")
    print("=" * 60)
    print(
        "Trainer details: "
        f"\n\tnumber of epochs: {num_epochs}"
        f"\n\tnumber of runs: {num_runs}"
        f"\n\tlearning rate: {lr}"
        f"\n\toptimizer: {optimizer_class}"
        f"\n\tconvergence patience: {convergence_patience}"
        f"\n\tdevice: {gp_device}"
        f"\n\tinitializer: {initializer_class}"
        f"\n\tcont_cols: {cont_cols}"
        f"\n\tcat_cols: {cat_cols}"
        f"\n\tsource_cols: {source_cols}"
        f"\n\tX_standardize: {standardize_X}"
        f"\n\tX_scaling_type: {X_scaling_type}"
        f"\n\ty_standardize: {standardize_y}"
        f"\n\tdimensions: {dimensions}"
        f"\n\tx_bounds: {x_bounds}"
        f"\n\tV2: {V2}"
    )
    print(f"Experiment details: \n\t{len(X_test_all)} test samples, {len(X_train)} train samples\n\tfolds: {num_folds}")

    return GPPlus_metrics, TabPFN_metrics


if __name__ == "__main__":
    # Example usage
    GPPlus_metrics, TabPFN_metrics = ackley_GPvsPFN_SEEK(
        num_folds=5,
        num_test=5000,
        train_size=10,
        dimensions=20,  # 20D Ackley function
        num_runs=16,
        num_epochs=100,
        lr=0.1,
        gp_device="cuda",
        amp_device="cuda",
        save_path="./results/Ackley/SEEK_Test",
        title="SEEK_tests",
        standardize_X=True,
        standardize_y=True,
        noise_train=0.005,
        noise_test=0.005,
        seed=42,
        run_models="gp",  # Only run GP for now
        # kernel_type="Gaussian",
        seek_use_bias=False,
        seek_share_bias_trunk=False,
        optimizer_class=torch.optim.Adam,
    )
