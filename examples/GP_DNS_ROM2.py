import random
import matplotlib.pyplot as plt
import torch
import numpy as np
import sys

import argparse
import pandas as pd
# from _GITBO import *
import warnings
warnings.filterwarnings("ignore")

import gpytorch
import gpplus
from gpplus.models import GPR
from gpplus.TestProblems_Utils import *


import time
from datetime import datetime
from gpplus.training.eval import evaluate_gp_model
from gpplus.utils.one_hot_to_latent_nn import OneHotToLatent
from gpplus.utils.matrix_encoder import MatrixEncoder, SourceMatrixEncoder


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    

def compute_metrics(y_true, y_hat, output_std=None):
    # Basic metrics
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy().reshape(-1)
    if isinstance(y_hat, torch.Tensor):
        y_hat = y_hat.detach().cpu().numpy().reshape(-1)
    if output_std is not None and isinstance(output_std, torch.Tensor):
        output_std = output_std.detach().cpu().numpy().reshape(-1)
        
    metrics = {
        "Time": time.time() - t1,
        "RRMSE": np.sqrt(mean_squared_error(y_true, y_hat)) / y_true.std(),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_hat)),
        "MSE": mean_squared_error(y_true, y_hat),
        # "MAE": mean_absolute_error(y_true, y_hat),
        # "R2": r2_score(y_true, y_hat)
    }

    # Add NIS if output_std is provided
    if output_std is not None:
        z = 2  # or 1.96 for 95% CI
        L = y_hat - z * output_std
        U = y_hat + z * output_std
        width = U - L
        below = (L - y_true) * (y_true < L)
        above = (y_true - U) * (y_true > U)
        interval_score = width + (2 / 0.05) * below + (2 / 0.05) * above
        NIS = interval_score.mean() / y_true.std()
        metrics["NIS"] = NIS
        return metrics
    
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score


print("DNS Dataset, GP")

tkwargs = {"device": torch.device(device), "dtype": torch.float32}
amp_dtype = tkwargs["dtype"]
amp_device = device
# n_samples = 1000
# Load data
# import pickle

# with open("../../Datasets/DNS-ROM.pkl", "rb") as f:
#     data = pickle.load(f)

data0 = pd.read_csv("../../Datasets/DNS_ROM_Old/Data_high.csv", header=None)
data1 = pd.read_csv("../../Datasets/DNS_ROM_Old/Data_LF1.csv", header=None)
data2 = pd.read_csv("../../Datasets/DNS_ROM_Old/Data_LF2.csv", header=None)
data3 = pd.read_csv("../../Datasets/DNS_ROM_Old/Data_LF3.csv", header=None)
data0_t = torch.tensor(data0.values, dtype=torch.float32)
data1_t = torch.tensor(data1.values, dtype=torch.float32)
data2_t = torch.tensor(data2.values, dtype=torch.float32)
data3_t = torch.tensor(data3.values, dtype=torch.float32)

data_all = torch.cat((data0_t, data1_t, data2_t, data3_t), dim=0)  # dim=0 → stack rows
data_NP = np.array(data_all)
# data1 = data["repetition_1"]
# X_all = np.vstack([data1["x_train"], data1["x_val"], data1["x_test"]])
# Y_all = np.vstack([data1["y_train"], data1["y_val"], data1["y_test"]])
# T_all = np.vstack([data1["t_train"], data1["t_val"], data1["t_test"]])
# T_all_OH = np.vstack([data1["t_train_OH"], data1["t_val_OH"], data1["t_test_OH"]])
col_to_encode = data_all[:, -2]
one_hot = pd.get_dummies(col_to_encode).astype(int)

# Ensure it has exactly 3 columns (0, 1, 2)
one_hot = torch.tensor(one_hot.values, dtype=torch.float32)
data_rest = torch.cat([data_all[:, :-2], data_all[:, -1:]], dim=1)
# Concatenate one-hot and original data (drop the original column 3 before concatenating)
print("Source feature OH encoded")
data_OH = torch.cat([one_hot, data_rest], dim=1)
X = data_OH[:,:-1]
X_np = np.array(X)

Y = data_OH[:,-1]
# print("Without standardizing y-output target")

# Y_mean = Y.mean()
# Y_std = Y.std()

# Y = (Y - Y_mean) / Y_std
# print("WITH standardizing y-output target before splitting")

num_epochs=100
num_runs = 8
lr = .1

full_metrics=[]
i = 0
t0 = time.time()
for seed in range(42,44):
    t1 = time.time()
    i += 1
    np.random.seed(seed)
    torch.manual_seed(seed)

    # source_encoder = SourceMatrixEncoder(
    #            num_sources=4,
    #            z_dim=2,
    #            initialization='normal',
    #            init_std=0.1
    #        )
    # print("Using A matrix encoder")

    source_architecture = {
        'hidden_dims': [],
        'activation': 'hardtanh',
        'dropout': 0.0
    }
    # use_probabilistic_embedding = True
    # n_samples = 10 if use_probabilistic_embedding else 1
    # source_encoder = None
    # if (data['column_indices']['source']):
    source_encoder = OneHotToLatent(
        input_dim=4,
        architecture_config=source_architecture,
        z_dim=2,
        num_passes=1,
        # probabilistic=True
        # num_passes_pred=20,
        # device=device,
        # dtype=torch.float32
    )
    print("Using MLP encoder")
    
    kernel = gpplus.kernels.CombinedKernel_MVMF(
        cont_cols=np.arange(4,10),
        # cat_cols=[],
        source_cols=np.arange(0, 4),
        # cat_encoder=cat_encoder,
        source_encoder=source_encoder,
        cat_combination_method="product",
        source_combination_method="product",
        # cont_kernel=gpplus.kernels.gaussian_kernel.GaussianKernel(ard_num_dims=6),
        source_kernel=gpplus.kernels.gaussian_kernel.GaussianKernel(),
        )
    print("source kernel only")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)
    X_train = X_train.to(torch.float32)
    X_test  = X_test.to(torch.float32)
    y_train = y_train.to(torch.float32)
    y_test  = y_test.to(torch.float32)
    y_train_std = y_train.std().item()

    # cont_cols = torch.arange(4,10, device=device)
    # # Example to skip one-hot columns at the end:
    # # cont_cols = torch.arange(X_train.shape[1] - n_onehot, device=device)
    
    # # ---- fit scaler on TRAIN ONLY
    # mu_X   = X_train[:, cont_cols].mean(dim=0)
    # std_X  = X_train[:, cont_cols].std(dim=0).clamp_min(1e-8)
    
    y_mean = y_train.mean()
    y_std  = y_train.std().clamp_min(1e-8)
    
    # # ---- transform X and y (use TRAIN stats for both train & test)
    # X_train_std = X_train.clone()
    # X_test_std  = X_test.clone()
    # X_train_std[:, cont_cols] = (X_train[:, cont_cols] - mu_X) / std_X
    # X_test_std[:,  cont_cols] = (X_test[:,  cont_cols] - mu_X) / std_X
    
    y_train_std = (y_train - y_mean) / y_std
    y_test_std  = (y_test  - y_mean) / y_std
    y_train_std2 = y_train_std.std().item()
    # xnp2=np.array(X_train_std.cpu())
    # ---- (optional) set a LogNormal prior on noise variance in standardized space
    # After standardizing y, Var[y] ≈ 1. Choose a weak prior center; e.g., 0.01 variance.
    # noise_var_center = 1e-2
    # prior_mean = float(np.log(noise_var_center))
    # noise_prior = gpytorch.priors.LogNormalPrior(loc=prior_mean, scale=1.0)
    prior_mean = np.log(y_train_std2**2)
    noise_prior = gpytorch.priors.LogNormalPrior(loc=prior_mean, scale=1.0)
    from gpytorch.constraints import GreaterThan
    model = GPR(X_train, y_train_std, 
                # kernel_module=gpplus.kernels.gaussian_kernel.GaussianKernel(lengthscale_constraint=Interval(-6, 4)),
                # kernel_module = gpplus.kernels.gaussian_kernel.GaussianKernel(ard_num_dims=X_train.shape[1]),
                # kernel_module = gpytorch.kernels.RBFKernel(),
                kernel_module = kernel,
                mean_module = gpplus.means.MultipleMean(encoded_cols=np.arange(0,4)),
                # mean_module=gpytorch.means.ZeroMean(),
                likelihood=gpytorch.likelihoods.GaussianLikelihood(noise_constraint = GreaterThan(1e-6), noise_prior=noise_prior), 
                seed=seed)

    if i == 1:  
        print(model)
    print("Cont kernel lengthscale:", model.covar_module.cont_kernel.lengthscale)
    print("Cont kernel lengthscale requires_grad:", model.covar_module.cont_kernel.lengthscale.requires_grad)   
    trainer = gpplus.training.GPTrainer(
        model=model,
        num_epochs=num_epochs,
        seed=seed,
        num_runs=num_runs, #<-----
        optimizer_kwargs={
            'lr': lr,
            # 'weight_decay': 1e-4,
            # 'eps': 1e-8,
        },
        convergence_patience=50,
        optimizer_class=torch.optim.Adam,
        device='cpu',
        map_prior=True,
        # scheduler = .95
        # optimizer_kwargs = {"lr": 1, "line_search_fn": "strong_wolfe"},
        # initializer_class=initializer_class,  # Use the chosen initializer
    )
    # with torch.amp.autocast('cuda',dtype=torch.float32, cache_enabled=True):
    # trainer.train_single_process(1)
    results = trainer.train()  # Returns a dict of results; you might store if needed.
    # y_pred, pred_lower, pred_upper, output_std = evaluate_gp_model(model, X_test)
    print("Cont kernel lengthscale:", model.covar_module.cont_kernel.lengthscale)
    print("Cont kernel lengthscale requires_grad:", model.covar_module.cont_kernel.lengthscale.requires_grad) 
    y_pred_std, pred_lower_std, pred_upper_std, output_std_std = evaluate_gp_model(model, X_test)
    # y_pred     = y_pred_std * y_std + y_mean
    # pred_lower = pred_lower_std * y_std + y_mean
    # pred_upper = pred_upper_std * y_std + y_mean
    # output_std = output_std_std * y_std
    metric = compute_metrics(y_test_std, y_pred_std, output_std_std)
    print(f"\nMetrics (Run # {i}):")
    for k, v in metric.items():
        print(f"{k}: {v:.4f}")
    full_metrics.append(metric)
        
print(f"\nTime: {time.time()-t0:.2f} s\n({num_runs} runs. Lr = {lr}. {num_epochs} epochs)")
df_metrics = pd.DataFrame(full_metrics)

# Calculate the mean for each metric across all seeds
avg_metrics = df_metrics.mean().to_dict()
std_metrics = df_metrics.std().to_dict()


# Print averages
print("Average metrics (± std):")
# print(f"{i} seeds, time: ({t00f - t00:.2f} s)")
for metric in avg_metrics.keys():
    mean_val = avg_metrics[metric]
    std_val = std_metrics[metric]
    print(f"{metric}: {mean_val:.6f} ± {std_val:.6f}")
print('\n')

# full_metrics=[]
# i = 0
# t0 = time.time()
# for seed in range(42,52):
#     t1 = time.time()
#     i += 1
#     np.random.seed(seed)
#     torch.manual_seed(seed)

#     source_encoder = SourceMatrixEncoder(
#                num_sources=4,
#                z_dim=2,
#                initialization='normal',
#                init_std=0.1
#            )
#     print("Using A matrix encoder")

#     # source_architecture = {
#     #     'hidden_dims': [],
#     #     'activation': 'hardtanh',
#     #     'dropout': 0.0
#     # }
#     # # use_probabilistic_embedding = True
#     # # n_samples = 10 if use_probabilistic_embedding else 1
#     # # source_encoder = None
#     # # if (data['column_indices']['source']):
#     # source_encoder = OneHotToLatent(
#     #     input_dim=4,
#     #     architecture_config=source_architecture,
#     #     z_dim=2,
#     #     num_passes=1,
#     #     # probabilistic=True
#     #     # num_passes_pred=20,
#     #     # device=device,
#     #     # dtype=torch.float32
#     # )
#     # print("Using MLP encoder")
    
#     kernel = gpplus.kernels.CombinedKernel_MVMF(
#         cont_cols=np.arange(4,10),
#         cat_cols=[],
#         source_cols=np.arange(0, 4),
#         # cat_encoder=cat_encoder,
#         source_encoder=source_encoder,
#         cat_combination_method="product",
#         source_combination_method="product",
#         cont_kernel=gpplus.kernels.gaussian_kernel.GaussianKernel(ard_num_dims=6),
#         source_kernel=gpplus.kernels.gaussian_kernel.GaussianKernel(),
#         )
#     X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)
#     X_train = X_train.to(torch.float32)
#     X_test  = X_test.to(torch.float32)
#     y_train = y_train.to(torch.float32)
#     y_test  = y_test.to(torch.float32)
#     y_train_std = y_train.std().item()

#     # cont_cols = torch.arange(4,10, device=device)
#     # # Example to skip one-hot columns at the end:
#     # # cont_cols = torch.arange(X_train.shape[1] - n_onehot, device=device)
    
#     # # ---- fit scaler on TRAIN ONLY
#     # mu_X   = X_train[:, cont_cols].mean(dim=0)
#     # std_X  = X_train[:, cont_cols].std(dim=0).clamp_min(1e-8)
    
#     y_mean = y_train.mean()
#     y_std  = y_train.std().clamp_min(1e-8)
    
#     # # ---- transform X and y (use TRAIN stats for both train & test)
#     # X_train_std = X_train.clone()
#     # X_test_std  = X_test.clone()
#     # X_train_std[:, cont_cols] = (X_train[:, cont_cols] - mu_X) / std_X
#     # X_test_std[:,  cont_cols] = (X_test[:,  cont_cols] - mu_X) / std_X
    
#     y_train_std = (y_train - y_mean) / y_std
#     y_test_std  = (y_test  - y_mean) / y_std
#     y_train_std2 = y_train_std.std().item()
#     # xnp2=np.array(X_train_std.cpu())
#     # ---- (optional) set a LogNormal prior on noise variance in standardized space
#     # After standardizing y, Var[y] ≈ 1. Choose a weak prior center; e.g., 0.01 variance.
#     # noise_var_center = 1e-2
#     # prior_mean = float(np.log(noise_var_center))
#     # noise_prior = gpytorch.priors.LogNormalPrior(loc=prior_mean, scale=1.0)
#     prior_mean = np.log(y_train_std2**2)
#     noise_prior = gpytorch.priors.LogNormalPrior(loc=prior_mean, scale=1.0)
#     from gpytorch.constraints import GreaterThan
#     model = GPR(X_train, y_train_std, 
#                 # kernel_module=gpplus.kernels.gaussian_kernel.GaussianKernel(lengthscale_constraint=Interval(-6, 4)),
#                 # kernel_module = gpplus.kernels.gaussian_kernel.GaussianKernel(ard_num_dims=X_train.shape[1]),
#                 # kernel_module = gpytorch.kernels.RBFKernel(),
#                 kernel_module = kernel,
#                 mean_module = gpplus.means.MultipleMean(encoded_cols=np.arange(0,4)),
#                 # mean_module=gpytorch.means.ZeroMean(),
#                 likelihood=gpytorch.likelihoods.GaussianLikelihood(noise_constraint = GreaterThan(1e-6), noise_prior=noise_prior), 
#                 seed=seed)

#     print(model)
#     trainer = gpplus.training.GPTrainer(
#         model=model,
#         num_epochs=num_epochs,
#         seed=seed,
#         num_runs=num_runs, #<-----
#         optimizer_kwargs={
#             'lr': lr,
#             # 'weight_decay': 1e-4,
#             # 'eps': 1e-8,
#         },
#         convergence_patience=50,
#         optimizer_class=torch.optim.Adam,
#         device='cpu',
#         map_prior=True,
#         # scheduler = .95
#         # optimizer_kwargs = {"lr": 1, "line_search_fn": "strong_wolfe"},
#         # initializer_class=initializer_class,  # Use the chosen initializer
#     )
#     # with torch.amp.autocast('cuda',dtype=torch.float32, cache_enabled=True):
#     # trainer.train_single_process(1)
#     results = trainer.train()  # Returns a dict of results; you might store if needed.
#     # y_pred, pred_lower, pred_upper, output_std = evaluate_gp_model(model, X_test)
#     y_pred_std, pred_lower_std, pred_upper_std, output_std_std = evaluate_gp_model(model, X_test)
#     # y_pred     = y_pred_std * y_std + y_mean
#     # pred_lower = pred_lower_std * y_std + y_mean
#     # pred_upper = pred_upper_std * y_std + y_mean
#     # output_std = output_std_std * y_std
#     metric = compute_metrics(y_test_std, y_pred_std, output_std_std)
#     print(f"\nMetrics (Run # {i}):")
#     for k, v in metric.items():
#         print(f"{k}: {v:.4f}")
#     full_metrics.append(metric)
        
# print(f"Time: {time.time()-t0:.2f} s ({num_runs} runs. Lr = {lr})")
# df_metrics = pd.DataFrame(full_metrics)

# # Calculate the mean for each metric across all seeds
# avg_metrics = df_metrics.mean().to_dict()
# std_metrics = df_metrics.std().to_dict()


# # Print averages
# print("Average metrics (± std):")
# # print(f"{i} seeds, time: ({t00f - t00:.2f} s)")
# for metric in avg_metrics.keys():
#     mean_val = avg_metrics[metric]
#     std_val = std_metrics[metric]
#     print(f"{metric}: {mean_val:.6f} ± {std_val:.6f}")
# print('\n')