import torch
import os
import json
from pathlib import Path
from gpplus.utils.onehot_encode_data import encode_qual_data, learn_encodings
import gpplus
import time
from gpplus.utils.metrics_functions import analyze_metrics, plot_metrics
from gpplus.utils import set_seed
from gpplus.utils.train_eval import train_eval_gp, train_eval_PFN
from tabpfn import TabPFNRegressor
from load_experimental_data import ackley_function, generate_ackley_data
from gp_prediction_diagnostics import run_gp_prediction_diagnostics
import defaults
import gpytorch

# import warnings
# warnings.filterwarnings("ignore")
def ackley_GPvsPFN(
        num_runs=defaults.NUM_RUNS,
        num_test=5000,
        train_size=10,  # total training size is train_size * number of X input dimensions
        dimensions=5,
        x_bounds=[-5, 10],
        num_inits=defaults.TRAINER_NUM_INITS,
        num_epochs=defaults.TRAINER_NUM_EPOCHS,
        lr=defaults.TRAINER_LR,
        convergence_patience=defaults.TRAINER_CONVERGENCE_PATIENCE,
        min_epochs=defaults.TRAINER_MIN_EPOCHS,
        min_loss_change=defaults.TRAINER_MIN_LOSS_CHANGE,
        optimizer_class=defaults.TRAINER_OPTIMIZER_CLASS,
        optimizer_kwargs=defaults.TRAINER_OPTIMIZER_KWARGS,
        initializer_class=defaults.TRAINER_INITIALIZER_CLASS,
        gp_device=defaults.TRAINER_GP_DEVICE,
        amp_device=defaults.TRAINER_AMP_DEVICE,
        save_path='./results/Ackley',
        title=None,
        standardize_X=defaults.STANDARDIZE_X,
        standardize_y=defaults.STANDARDIZE_Y,
        x_standardize_method=defaults.X_STANDARDIZE_METHOD,  # 0=Gaussian (StandardScaler), 1=Uniform [0,1], 2=Uniform [-1,1]
        noise_train=0.0,
        noise_test=0.0,
        noise_type=defaults.NOISE_TYPE,
        seed=defaults.SEED,
        seed_trainer=defaults.SEED_TRAINER,
        V2=False,
        gp_dtype=defaults.DTYPE_GP,
        pfn_dtype=defaults.DTYPE_PFN,
        trainer_info=True,
        run_models=None,  # None=run both, 'gp'=GP only, 'pfn'=PFN only
        log_lbfgs_inner=defaults.TRAINER_LOG_LBFGS_INNER,
        single_dataset=False,
        # If True: one Sobol train set (and test set) for every run.
        # If False: draw a larger train pool, shuffle, and use disjoint slices per run (legacy).
    ):

    if run_models == 'pfn':
        num_inits = 0

    v2 = "V2" if V2 else ""
    if title is None:
        title = f"Ackley{v2}_{dimensions}Dx_{train_size}Dn_[{x_bounds[0]},{x_bounds[1]}]_{num_inits}inits_noiseTest{noise_test}_noiseTrain{noise_train}_x{num_runs}"
    else:
        title = f"Ackley{v2}_{title}_{dimensions}Dx_{train_size}Dn_[{x_bounds[0]},{x_bounds[1]}]_{num_inits}inits_noiseTest{noise_test}_noiseTrain{noise_train}_x{num_runs}"
    
    print(f" GP Device: {gp_device}")
    print(f" TabPFN Device: {amp_device}")
    regressor = TabPFNRegressor(device=amp_device, random_state=seed)
    if save_path is not None:
        plot_save_path = f"{save_path}/plots"
        callback_save_path = f"{save_path}/trainer_analysis/plots"
    else:
        plot_save_path = None
        callback_save_path = None

    # Generate data
    set_seed(seed)
    
    # Calculate total samples needed
    train_per_run = train_size * dimensions  # train_size * dimensions for Ackley
    if single_dataset:
        total_train = train_per_run
        total_samples = num_test + total_train
        print(
            f"Generating {total_samples} unique Sobol samples for {dimensions}D Ackley function\n\t"
            f"Test samples: {num_test} / Train samples: {train_per_run} "
            f"(single_dataset=True: same train data for all {num_runs} runs)"
        )
    else:
        num_runs_gen = max(num_runs, 20)
        total_train = num_runs_gen * train_per_run
        total_samples = num_test + total_train
        print(
            f"Generating {total_samples} unique Sobol samples for {dimensions}D Ackley function\n\t"
            f"Test samples: {num_test} / Train pool: {total_train} "
            f"(single_dataset=False: disjoint train slices, {train_per_run} points per run)"
        )
    
    # Generate train and test data in one call
    X_train_all, y_train_all, X_test_all, y_test_all = generate_ackley_data(
        n_train=total_train,
        n_test=num_test,
        dimensions=dimensions,
        x_bounds=x_bounds,
        train_noise=noise_train,
        test_noise=noise_test,
        noise_type=noise_type,
        seed=seed,
        V2=V2
    )
    X = torch.cat([X_test_all, X_train_all], dim=0)

    print("="*10)
    print(f"{title}: TabPFN vs GP Comparison")
    print("="*10)

    # Prepare encoded data once from already loaded X, y (no extra CSV/label encoding)
    qual_dict = learn_encodings(X)
    print(qual_dict)
    _, cont_cols, cat_cols, source_cols = encode_qual_data(X_train_all, qual_dict=qual_dict, source_col=None)
    # _, _, _, _ = encode_qual_data(X_test_all, qual_dict=qual_dict, source_col=None)
    # print(cat_cols)
    TabPFN_metrics = []
    GPPlus_metrics = []
    GPTrainer_info = []  # Accumulate trainer logs across runs

    if not single_dataset:
        # Randomize across the single source, then split across runs
        all_indices = torch.randperm(total_train)
        train_indices_2d = all_indices.reshape(num_runs_gen, train_per_run)
        
    total_start_time = time.time()
    for i in range(num_runs):
        run_seed = seed_trainer if seed_trainer is not None else (seed + i)
        print(f"\n{'='*20} {title} RUN {i+1}/{num_runs}: {run_seed} {'='*20}")

        if single_dataset:
            X_train = X_train_all
            y_train = y_train_all
        else:
            # Get training indices for this run
            run_train_indices = train_indices_2d[i]
            X_train = X_train_all[run_train_indices]
            y_train = y_train_all[run_train_indices]

        # Prepare data (standardization) - ALWAYS DO THIS for both GP and PFN
        # Reuse PFN split, convert to torch (unified)
        X_train = X_train.detach().clone().to(dtype=gp_dtype)
        X_test = X_test_all.detach().clone().to(dtype=gp_dtype)
        y_train = y_train.detach().clone().to(dtype=gp_dtype)
        y_test = y_test_all.detach().clone().to(dtype=gp_dtype)
        X_train_raw_for_pfn = X_train.detach().clone()
        X_test_raw_for_pfn = X_test.detach().clone()
        # Determine X scaling type
        Xscaler = None
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

        # Normalize the data
        Yscaler = gpplus.utils.StandardScaler()
        Yscaler.fit(y_train)
        y_train_mean = Yscaler.mean 
        y_train_std = Yscaler.std
        y_train_normal = Yscaler.transform(y_train)

        # =============================================================================
        # GP Section 
        # =============================================================================
        if run_models in [None, 'gp']:
            print(f"\n--- {title} GP Training ---")
            
            # Create GP model (default kernel like SF wing)
            model = gpplus.models.GPR(
                X_train,
                y_train_normal if standardize_y else y_train,
                kernel_module=defaults.SF_kernel,
                mean_module=defaults.SF_mean,
                likelihood=defaults.SF_likelihood,
            )
            if (i == 0) or (i == num_runs - 1):
                print(f"X_train: {X_train.shape}")
                print(f"X_test: {X_test.shape}")
                print(f"y_test mean: {y_test.mean().item()} / y_test std: {y_test.std().item()}")
                print(model)

            # Create trainer
            gp_metric, y_pred_gp, output_std_gp, gp_trainer_info = train_eval_gp(
                model,
                X_test,
                y_test,
                num_epochs=num_epochs,
                seed=run_seed,
                num_inits=num_inits,
                lr=lr,
                convergence_patience=convergence_patience,
                min_epochs=min_epochs,
                min_loss_change=min_loss_change,
                optimizer_class=optimizer_class,
                optimizer_kwargs=optimizer_kwargs,
                initializer_class=initializer_class,
                device=gp_device,
                y_train_mean=y_train_mean if standardize_y else None,
                y_train_std=y_train_std if standardize_y else None,
                source_cols=source_cols,
                trainer_info=trainer_info,  # Set to True if you want trainer info
                callbacks=defaults.get_default_gp_callbacks(
                    optimizer_class,
                    callback_save_path=callback_save_path,
                    log_lbfgs_inner=log_lbfgs_inner,
                ),
                callback_save_path=callback_save_path,
                log_lbfgs_inner=log_lbfgs_inner,
                cholesky_jitter=defaults.TRAINER_CHOLESKY_JITTER,
                n_jobs=defaults.TRAINER_N_JOBS,
                inner_max_num_threads=defaults.TRAINER_INNER_MAX_NUM_THREADS,
            )
            
            GPPlus_metrics.append(gp_metric)
            
            # Accumulate trainer info if available
            if gp_trainer_info:
                # Add run information to trainer log
                gp_trainer_info["run"] = i + 1
                gp_trainer_info["metrics"] = gp_metric  # Include metrics for this run
                GPTrainer_info.append(gp_trainer_info)

            print(f"\nGP Results (Run {i+1}/{num_runs})")
            for k, v in gp_metric.items():
                print(f"  {k}: {v:.4f}" if v is not None and isinstance(v, (int, float)) else f"  {k}: {v}")

            truth_fn = (
                (lambda X: torch.log(ackley_function(X, dimensions) + 1.0))
                if V2
                else (lambda X: ackley_function(X, dimensions))
            )
            run_gp_prediction_diagnostics(
                save_path=save_path,
                run_index=i,
                experiment_title=title,
                dimensions=dimensions,
                cont_cols=cont_cols,
                x_bounds=x_bounds,
                X_train_orig=X_train_raw_for_pfn,
                X_train_raw_for_pfn=X_train_raw_for_pfn,
                X_test_raw_for_pfn=X_test_raw_for_pfn,
                y_train=y_train,
                y_test=y_test,
                y_pred_gp=y_pred_gp,
                output_std_gp=output_std_gp,
                model=model,
                Xscaler=Xscaler,
                standardize_X=standardize_X,
                y_train_mean=y_train_mean if standardize_y else None,
                y_train_std=y_train_std if standardize_y else None,
                standardize_y=standardize_y,
                standardize_y_log_scale=False,
                log_scale_C=None,
                log_y_point_inverse="median",
                truth_fn=truth_fn,
                plot_prediction_diagnostics=defaults.PLOT_PREDICTION_DIAGNOSTICS,
                diagnostic_run_indices=tuple(defaults.PREDICTION_DIAGNOSTIC_RUN_INDICES),
                max_marginal_dims=defaults.PREDICTION_DIAGNOSTIC_MAX_MARGINAL_DIMS,
            )

        # =============================================================================
        # TabPFN Section
        # =============================================================================
        if run_models in [None, 'pfn']:
            print(f"\n--- {title} TabPFN Training ---")
            
            tabpfn_metric, y_pred_tabpfn, output_std_tabpfn = train_eval_PFN(
                X_train_raw_for_pfn,
                X_test_raw_for_pfn,
                y_train,
                y_test,
                amp_device=amp_device,
                amp_dtype=pfn_dtype,
                regressor=regressor,
                source_cols=source_cols,
            )
            TabPFN_metrics.append(tabpfn_metric)

            # Print results for this run
            print(f"\nTabPFN Results (Run {i+1}/{num_runs})")
            for k, v in tabpfn_metric.items():
                print(f"  {k}: {v:.4f}" if v is not None and isinstance(v, (int, float)) else f"  {k}: {v}")
        
        # Collect model info from first run
        if i == 0:
            y_test_stats = {
                "y_test_mean": float(y_test_all.mean().item()),
                "y_test_std": float(y_test_all.std().item())
            }

            if run_models in [None, 'gp']:
                gp_model_info = {
                    "model_str": str(model),
                    "cat_cols": cat_cols,
                    "cont_cols": cont_cols,
                    "source_cols": source_cols,
                    "qual_dict": qual_dict,
                    "input_dim": X_train.shape[1],
                    "train_samples": X_train.shape[0],
                    "test_samples": num_test,
                    "y_train_mean": float(y_train_mean.item()),
                    "y_train_std": float(y_train_std.item()),
                    "standardize_X": standardize_X,
                    "standardize_y": standardize_y,
                    "X_scaling_type": X_scaling_type,
                    "x_standardize_method": x_standardize_method,
                    "dtype": str(gp_dtype),
                    "device": str(gp_device),
                    "num_epochs": num_epochs,
                    "num_inits": num_inits,
                    "lr": lr,
                    "optimizer": optimizer_class.__name__,
                    "convergence_patience": convergence_patience,
                    "initializer": initializer_class.__name__ if initializer_class else None,
                    **y_test_stats,
                    "num_runs": num_runs,
                    "seed": seed,
                    "seed_trainer": seed_trainer,
                }
            
            if run_models in [None, 'pfn']:
                tabpfn_model_info = {
                    "model_path": regressor.model_path,
                    "fit_mode": regressor.fit_mode,
                    "device": str(regressor.device),
                    "inference_precision": regressor.inference_precision,
                    "random_state": regressor.random_state,
                    "use_autocast": regressor.use_autocast_,
                    "forced_inference_dtype": str(regressor.forced_inference_dtype_) if regressor.forced_inference_dtype_ else None,
                }
        
    # =============================================================================
    # Final Results Summary
    # =============================================================================
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)

    # Summaries via analyze_metrics
    TabPFN_summary = analyze_metrics(TabPFN_metrics, print_summary=True, label="TabPFN", title=title) if run_models in [None, 'pfn'] else None
    GPPlus_summary = analyze_metrics(GPPlus_metrics, print_summary=True, label="GP", title=title) if run_models in [None, 'gp'] else None
    
    # Add model info to GP summary if available
    
    if save_path is not None:
        if run_models is None:
            plot_metrics(TabPFN_metrics, GPPlus_metrics, labels=["TabPFN", "GP"], title=title, save_path=plot_save_path)
        # Save raw metrics and summaries
        out_dir = Path(save_path)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            # Determine file prefix based on run_models
            if run_models is not None:
                file_prefix = run_models
            else:
                file_prefix = "gpVpfn"
            
            # Combined single file: TabPFN data + GP data + GP model_info at the end
            combined_data = {}
            if run_models in [None, 'gp']:
                combined_data["gp_data"] = {
                    "summary": GPPlus_summary,
                    "metrics": GPPlus_metrics,
                    "gp_model_info": gp_model_info,
                }
            if run_models in [None, 'pfn']:
                combined_data["tabpfn_data"] = {
                    "summary": TabPFN_summary,
                    "metrics": TabPFN_metrics,
                    "pfn_model_info": tabpfn_model_info
                }
            # Append defaults.py source at end of JSON for reproducibility
            _defaults_path = Path(__file__).resolve().parent / "defaults.py"
            if _defaults_path.is_file():
                combined_data["defaults_py"] = _defaults_path.read_text(encoding="utf-8")
            (out_dir / f"{file_prefix}_{title}.json").write_text(json.dumps(combined_data, indent=2))
        except Exception:
            pass
        
        # Save trainer info if trainer_info is enabled
        if trainer_info and GPTrainer_info and run_models in [None, 'gp']:
            try:
                # Create trainer_analysis directory (same level as plots)
                trainer_analysis_dir = Path(save_path) / "trainer_analysis"
                trainer_analysis_dir.mkdir(parents=True, exist_ok=True)
                
                # Save raw trainer info keyed by run (run_1, run_2, ...)
                trainer_info_by_run = {
                    f"run_{entry.get('run', i + 1)}": entry
                    for i, entry in enumerate(GPTrainer_info)
                }
                trainer_info_data = {
                    "title": title,
                    "num_runs": num_runs,
                    "num_inits_per_run": num_inits,
                    "trainer_info": trainer_info_by_run,
                }
                
                # Save trainer info JSON (always use "gp" prefix for GP trainer info)
                trainer_info_file = trainer_analysis_dir / f"gp_{title}_GP_Trainer_Analysis.json"
                trainer_info_file.write_text(json.dumps(trainer_info_data, indent=2))
                print(f"\nTrainer info saved to: {trainer_info_file}")

                # Also augment the saved gp_data metrics with chosen final loss per run.
                try:
                    from plot_trainer_analysis_hyperparams import extract_runs_and_chosen
                    all_inits, chosen_list = extract_runs_and_chosen(trainer_info_data)
                    if chosen_list and "gp_data" in combined_data:
                        gp_section = combined_data.get("gp_data") or {}
                        gp_metrics_list = gp_section.get("metrics")
                        if isinstance(gp_metrics_list, list):
                            # One chosen record per run, in the same run order used for metrics.
                            for i_run, (metric_rec, chosen_rec) in enumerate(zip(gp_metrics_list, chosen_list)):
                                loss_val = chosen_rec.get("loss")
                                if loss_val is not None:
                                    metric_rec["loss_final"] = float(loss_val)
                        combined_path = out_dir / f"{file_prefix}_{title}.json"
                        try:
                            combined_path.write_text(json.dumps(combined_data, indent=2))
                        except Exception:
                            # If rewriting combined JSON fails, continue without aborting trainer analysis.
                            pass
                except Exception as e:
                    print(f"Augmenting gp_data with chosen final loss failed: {e}")
                try:
                    from plot_trainer_analysis_hyperparams import plot_trainer_analysis_from_data
                    plot_trainer_analysis_from_data(trainer_info_data, trainer_analysis_dir / "plots")
                except Exception as plot_e:
                    print(f"Trainer analysis plotting skipped: {plot_e}")
                try:
                    from plot_epoch_metrics import plot_iter_metrics_from_data
                    plot_iter_metrics_from_data(trainer_info_data, trainer_analysis_dir / "plots")
                except ValueError:
                    pass  # no epoch_metrics in data
                except Exception as e:
                    print(f"Epoch metrics plotting skipped: {e}")
                
            except Exception as e:
                print(f"Error saving trainer info: {e}")
                import traceback
                traceback.print_exc()
    print(f"\nTotal experiment time for {num_runs} runs: {time.time() - total_start_time:.2f}s")
    print("="*60)
    print(f"Trainer details: \n\tnumber of epochs: {num_epochs}\n\tnumber of inits: {num_inits}\n\tlearning rate: {lr}\n\toptimizer: {optimizer_class}\n\tconvergence patience: {convergence_patience}\n\tdevice: {gp_device}\n\tinitializer: {initializer_class}\n\tcont_cols: {cont_cols}\n\tcat_cols: {cat_cols}\n\tsource_cols: {source_cols}\n\tqual_dict: {qual_dict}\n\tX_standardize: {standardize_X}\n\tX_scaling_type: {X_scaling_type}\n\ty_standardize: {standardize_y}")
    print(f"Experiment details: \n\t{len(X_test_all)} test samples, {len(X_train)} train samples\n\truns: {num_runs}")

    return GPPlus_metrics, TabPFN_metrics


if __name__ == "__main__":
    # Gaussian GP baseline (match experiments_RFF/A4_ackley_40D_RFF.py: 40D, 40 pts/dim, noise 0.005, seed 42)
    ackley_GPvsPFN(
        num_runs=10,
        train_size=10,
        dimensions=10,
        num_inits=16,
        num_epochs=1,
        save_path="experiments_revisions_april/results_test_val/ackley_10D_gaussian",
        run_models="gp",
        single_dataset=True,
        noise_test=0.005,
        noise_train=0.005,
        x_standardize_method=defaults.X_STANDARDIZE_METHOD,
        seed=42,
    )
    # ackley_GPvsPFN(num_runs=5, train_size=20, dimensions=80, num_inits=4, num_epochs=10000, save_path='./results/Ackley/test', run_models='pfn')


