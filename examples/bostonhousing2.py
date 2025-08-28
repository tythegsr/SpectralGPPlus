#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 12:21:56 2025

@author: kian
"""
# from gpplus.models import gp_plus, GPR
from gpplus.models.gpr import GPR
from gpplus.test_functions.multi_fidelity import multi_fidelity_wing
# from gpplus.preprocessing import train_test_split_normalizeX
# from gpplus.preprocessing.normalizeX import standard, MFstandard
from gpplus.utils.set_seed import set_seed
# import gpplus.preprocessing
from sklearn.model_selection import train_test_split
import torch
# from gpplus.optim import fit_model_scipy
from gpplus.training import evaluate_gp_model
import time
# from gpplus.priors import LogHalfHorseshoePrior, MollifiedUniformPrior
# from gpytorch.priors import NormalPrior
import gpytorch
from gpplus.utils import InputTransformNet, OneHotEncoder
from gpytorch.constraints import GreaterThan, Positive
from gpplus.training import GPTrainer
import gpplus
from gpplus.means import MultipleMean
# from gpplus.means import FidelityMean
# mean3 = MultipleMean()
import torch
import torch.nn as nn
import torch.distributions as dist
# from gpplus.preprocessing.MFOps import MFPreprocessing
# from gpplus.training.parameter_initializer import SimpleParameterInitializer, DefaultParameterInitializer
from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.preprocessing import StandardScaler


import cProfile
profiler = cProfile.Profile()
# Load the Boston Housing dataset
df_boston = fetch_openml(data_id=531, as_frame=True)
X, y = df_boston.data, df_boston.target

# Display the dataset description and data
# display(Markdown("### Boston Housing Dataset"))
# display(Markdown(df_boston["DESCR"]))
# display(X.head())
seed = 42
set_seed(seed)
# qual_dict = {10: 4}
# n=500
# num = {'0': n, '1': n, '2': n, '3': n}
# noise_std = {'0': 0, '1': 0, '2': 0, '3': 0}
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)
x_scaler = StandardScaler()
X_train = x_scaler.fit_transform(X_train)   # Only fit on training set
X_test = x_scaler.transform(X_test)         # Use same transform on test set

# === 3. Normalize y (only if you're training on scaled targets) ===
y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test = y_scaler.transform(y_test.values.reshape(-1, 1))

# === 4. Convert to torch tensors ===
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test  = torch.tensor(y_test, dtype=torch.float32)

# X_train = torch.tensor(X_train.astype(np.float32).values)
# y_train = torch.tensor(y_train.astype(np.float32).values)
# X_test = torch.tensor(X_test.astype(np.float32).values)
# y_test = torch.tensor(y_test.astype(np.float32).values)


# ytrain_norm, ytrain_mean, ytrain_std = standard(ytrain)

time_model = time.time()

# likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Interval(1e-8, 1e-2), 
                                                     # noise_prior=LogHalfHorseshoePrior(scale=0.01, lb=1e-8)
                                                     # )
# mean = gpytorch.means.ConstantMean()
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
model = GPR(X_train, y_train.squeeze(-1))
# model2 = GPR(Xtrain, ytrain, likelihood=likelihood, mean_module=m_mean2, kernel_module=covar_module)


print("===== DEBUG INFO =====")
print("Seed:", torch.initial_seed())
print("X shape:", X_train.shape, "Y shape:", y_train.shape)
# print("LR:", optimizer.param_groups[0]['lr'])
print("Model:", model)
# print("Kernel params:", model.covar_module.raw_lengthscale)
print("CUDA:", next(model.parameters()).device)
print("======================")

print("Model class:", model.__class__)
print("Module:", model.__module__)


print("\n=== NEW MODEL Parameter Shapes ===")
for name, param in model.named_parameters():
    print(f"{name}: shape = {param.shape}, numel = {param.numel()}")

print("\n=== Parameter Values ===")   
for name, param in model.named_parameters():
    print(f"{name}: value={param.data}, grad={param.grad}")


# with gpytorch.settings.cholesky_jitter(1e-4), \
#     gpytorch.settings.max_cg_iterations(1000), \
#     gpytorch.settings.eval_cg_tolerance(1e-3), \
#     gpytorch.settings.max_lanczos_quadrature_iterations(20), \
#     gpytorch.settings.fast_pred_var(), \
#     gpytorch.settings.detach_test_caches(False), \
#     gpytorch.settings.skip_posterior_variances(False):
#     # Train with the special GPTrainer from gpplus
#     trainer = gpplus.training.GPTrainer(
#         model=model,
#         num_epochs=10000,
#         seed=seed,
#         num_runs=n_restarts,
#         optimizer_kwargs={
#             'lr': 1e-1,
#             # 'weight_decay': 1e-4,
#             # 'eps': 1e-8,
#         },
#         map_prior=True,
#         convergence_patience=32,
#         optimizer_class=torch.optim.Adam,
#         # map_prior=True,
#         initializer_class=initializer_class,  # Use the chosen initializer
#         device='cuda',
#     )
  
#     # with torch.amp.autocast('cuda',dtype=torch.float32, cache_enabled=True):
#     _ = trainer.train()  # Returns a dict of results; you might store if needed.

# optimizer = torch.optim.Adam(model.parameters(), lr=.01)
# optimizer = torch.optim.LBFGS
num_epochs=10000
num_runs = 64
if torch.cuda.is_available():
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    # profiler.enable()
    # Print initial memory usage
    print(f"Initial GPU memory: {torch.cuda.memory_allocated(device)/1024**2:.1f} MB")
    
    # Choose parameter initializer type
    # Option 1: SimpleParameterInitializer (faster, more robust)
    # initializer_class = SimpleParameterInitializer
    
    # Option 2: DefaultParameterInitializer (systematic exploration with Sobol sequences)
    # initializer_class = DefaultParameterInitializer
    
    # print(f"Using initializer: {initializer_class.__name__}")
    
    # Training with stability settings
    with gpytorch.settings.cholesky_jitter(1e-4), \
        gpytorch.settings.max_cg_iterations(1000), \
        gpytorch.settings.eval_cg_tolerance(1e-3), \
        gpytorch.settings.max_lanczos_quadrature_iterations(200), \
        gpytorch.settings.fast_pred_var(), \
        gpytorch.settings.detach_test_caches(False), \
        gpytorch.settings.skip_posterior_variances(False):
        # Train with the special GPTrainer from gpplus
        trainer = gpplus.training.GPTrainer(
            model=model,
            num_epochs=num_epochs,
            seed=seed,
            num_runs=num_runs, #<-----
            optimizer_kwargs={
                'lr': 1,
                # 'weight_decay': 1e-4,
                # 'eps': 1e-8,
            },
            map_prior=True,
            convergence_patience=100,
            optimizer_class=torch.optim.Adam,
            # optimizer_kwargs = {"lr": 1, "line_search_fn": "strong_wolfe"},
            # initializer_class=initializer_class,  # Use the chosen initializer
            device='cpu',
        )
        # with torch.amp.autocast('cuda',dtype=torch.float32, cache_enabled=True):
        _ = trainer.train()  # Returns a dict of results; you might store if needed.
    profiler.disable()
    # profiler.dump_stats(os.path.join(dir_path, f"profile_results_{seed}.prof"))
    end_event.record()
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)     
    
    # Calculate timing values
    total_time_minutes = elapsed_ms / 60000
    total_time_seconds = elapsed_ms / 1000
    avg_time_per_run_seconds = elapsed_ms / (num_runs * 1000)
    
    # Print final memory usage and timing
    print(f"Final GPU memory: {torch.cuda.memory_allocated(device)/1024**2:.1f} MB")
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated(device)/1024**2:.1f} MB")
    print(f"Total time: {total_time_minutes:.2f} minutes ({total_time_seconds:.1f} seconds)")
    print(f"Average time per run: {avg_time_per_run_seconds:.1f} seconds")

# print(f"Trainer seed: {trainer.seed}")
print("training done")
# trainer.train()
# print("\n=== Parameter Shapes ===")
# for name, param in model.named_parameters():
#     print(f"{name}: shape = {param.shape}, numel = {param.numel()}")

# from gpplus.training.mll_scipy import MarginalLogLikelihoodWrapper
# lik = MarginalLogLikelihoodWrapper(model)
# theta, _ = lik.pack_parameters()
# lik.unpack_parameters(theta)  # will crash here
# from gpplus.training.mll_scipy import fit_model_scipy
# from gpplus.training.mll_torch import fit_model_torch
# from gpplus.training.mll_torch import fit_model_torch_lbfgs
# fit_model_scipy(model=model,
#                       num_restarts=n_restarts)
print(f"\n{time.time() - time_model:.2f} s for {num_runs} restarts to create and fit model (seed: {seed})")

# print("\n=== Parameter values AFTER optimization ===")
# print(f"{time.time() - time_model:.2f} s for {num_runs} restarts to create and fit model (seed: {seed})\n")
# with torch.no_grad():
#     output = model(Xtrain)
#     print("Final MLL:", likelihood(output, ytrain))
#     print(f"likelihood noise: {likelihood.noise_covar.noise.item()}")  # this is the actual scalar noise value

    # Raw noise value (pre-transform)
    # print(f"likelihood raw noise: {likelihood.noise_covar.raw_noise.item()}")
# for name, param in model.named_parameters():
    # print(f"{name}: {param.data}")

# %%


# %%



def compute_metrics(mean, lower, upper, stddev, y_true):
    mse = torch.mean((mean - y_true) ** 2).item()
    mae = torch.mean(torch.abs(mean - y_true)).item()
    rmse = torch.sqrt(torch.mean((mean - y_true) ** 2)).item()
    nrmse = rmse / torch.std(y_true).item()

    # NIS = Normalized Interval Score (width of CI + penalty if outside)
    alpha = 0.05  # 95% CI
    interval_width = upper - lower
    below = (y_true < lower).float()
    above = (y_true > upper).float()
    penalty = (2 / alpha) * ((lower - y_true) * below + (y_true - upper) * above)
    nis = torch.mean(interval_width + penalty).item()

    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'NRMSE': nrmse,
        'NIS': nis
    }
with torch.no_grad():
    pred_mean, pred_lower, pred_upper, pred_std = evaluate_gp_model(model, X_test)
    # # Inverse-transform y_pred and y_test if trained on normalized y
    # # Reshape to (N, 1) for scaler
    pred_mean_np = pred_mean.cpu().numpy().reshape(-1, 1)
    pred_lower_np = pred_lower.cpu().numpy().reshape(-1, 1)
    pred_upper_np = pred_upper.cpu().numpy().reshape(-1, 1)
    y_test_np     = y_test.cpu().numpy().reshape(-1, 1)
    
    # Inverse transform
    pred_mean_orig  = y_scaler.inverse_transform(pred_mean_np)
    pred_lower_orig = y_scaler.inverse_transform(pred_lower_np)
    pred_upper_orig = y_scaler.inverse_transform(pred_upper_np)
    y_test_orig     = y_scaler.inverse_transform(y_test_np)
    
    # Convert back to tensors (flatten to [N])
    pred_mean = torch.tensor(pred_mean_orig.squeeze(), dtype=torch.float32)
    pred_lower = torch.tensor(pred_lower_orig.squeeze(), dtype=torch.float32)
    pred_upper = torch.tensor(pred_upper_orig.squeeze(), dtype=torch.float32)
    y_test = torch.tensor(y_test_orig.squeeze(), dtype=torch.float32)


    metrics = compute_metrics(pred_mean, pred_lower, pred_upper, pred_std, y_test)

# metrics = compute_metrics(mean_i, pred_lower_i, pred_upper_i, std_i, y_test)
print(
    f"\nSUM → RMSE: {metrics['RMSE']:.4f}, MSE: {metrics['MSE']:.4f}, MAE: {metrics['MAE']:.4f}, "
    f"NRMSE: {metrics['NRMSE']:.4f}, NIS: {metrics['NIS']:.4f}"
    )
