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
    
import torch.nn.functional as F

def one_hot_encode_integer_columns(data: torch.Tensor, int_tol: float = 1e-6):
    """
    data: (N, D) torch tensor (float or int)
    Returns:
      encoded_data: torch.Tensor with integer columns one-hot encoded
      column_info: list of dicts describing how each original column was treated
    """
    if data.dtype not in (torch.float32, torch.float64):
        data = data.to(torch.float32)

    num_rows, num_cols = data.shape
    encoded_columns = []
    column_info = []

    for col_idx in range(num_cols):
        col = data[:, col_idx]

        is_integer_like = torch.all(torch.abs(col - torch.round(col)) < int_tol)
        if is_integer_like:
            col_long = torch.round(col).to(torch.long)
            min_val = int(col_long.min().item())
            max_val = int(col_long.max().item())
            num_classes = max_val - min_val + 1

            # Shift so the smallest value maps to 0, then one-hot
            class_indices = col_long - min_val
            ohe = F.one_hot(class_indices, num_classes=num_classes).to(data.dtype)
            encoded_columns.append(ohe)

            column_info.append({
                "column": col_idx,
                "type": "one_hot",
                "min_value": min_val,
                "max_value": max_val,
                "num_classes": num_classes
            })
        else:
            encoded_columns.append(col.unsqueeze(1))
            column_info.append({
                "column": col_idx,
                "type": "continuous"
            })

    encoded_data = torch.cat(encoded_columns, dim=1)
    return encoded_data, column_info
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score
tkwargs = {"device": torch.device(device), "dtype": torch.float32}
amp_dtype = tkwargs["dtype"]
amp_device = device

print("HOIP Dataset, GP")


tkwargs = {"device": torch.device(device), "dtype": torch.float32}
amp_dtype = tkwargs["dtype"]
amp_device = device
# n_samples = 1000
# Load data
data0 = pd.read_csv("../../Datasets/HOIP_data/sample_HF.csv", header=None)
# data1 = pd.read_csv("../../Datasets/HOIP_data/sample_low1.csv", header=None)
data2 = pd.read_csv("../../Datasets/HOIP_data/sample_low2.csv", header=None)
data3 = pd.read_csv("../../Datasets/HOIP_data/sample_low3.csv", header=None)
# all_data = [data0, data1, data2, data3]
all_data = [data0, data2, data3]

# data_numeric = data.drop(columns=["M2AXphase"])
# column_names = data_numeric.columns.tolist()

print("HOIP (all data OH encoded)")


data0_t = torch.tensor(data0.values, dtype=torch.float32)
# data1_t = torch.tensor(data1.values, dtype=torch.float32)
data2_t = torch.tensor(data2.values, dtype=torch.float32)
data3_t = torch.tensor(data3.values, dtype=torch.float32)

# data_all = torch.cat((data0_t, data1_t, data2_t, data3_t), dim=0)  # dim=0 → stack rows
data_all = torch.cat((data0_t, data2_t, data3_t), dim=0)  # dim=0 → stack rows

t00 = time.time()
encoded_tensor, info = one_hot_encode_integer_columns(data_all)
tensor = np.array(encoded_tensor)
for c in info:
    print(c)
print("Final shape:", tuple(encoded_tensor.shape))

# Convert to tensor
arr = torch.tensor(encoded_tensor, dtype=torch.float32)

num_epochs= 2000
num_runs = 1
lr = .1

X = arr[:, :-1]
y = arr[:, -1]
i = 0
full_metrics=[]
t0 = time.time()
for seed in range(42,52):
    t1 = time.time()
    i += 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Initialize architectures



    # Initialize encoders with custom architectures
    # cat_encoder = None
    cat_encoder = MatrixEncoder(
        input_dim=29,
        z_dim=2,
        initialization='normal',
        init_std=0.1
        )

    # cat_architecture = {
    #     'hidden_dims': [],
    #     'activation': 'relu',
    #     'dropout': 0.0
    # }
    # cat_encoder = OneHotToLatent(
    #     input_dim=29,
    #     architecture_config=cat_architecture,
    #     z_dim=2,
    #     num_passes=1,
    #     # probabilistic=True
    #     # num_passes_pred=20,
    #     # device=device,
    #     # dtype=torch.float32
    # # )
    # source_encoder = SourceMatrixEncoder(
    #            num_sources=3,
    #            z_dim=2,
    #            initialization='normal',
    #            init_std=0.1
    #        )
    
    source_architecture = {
        'hidden_dims': [],
        'activation': 'relu',
        'dropout': 0.0
    }

    source_encoder = OneHotToLatent(
        input_dim=3,
        architecture_config=source_architecture,
        z_dim=2,
        num_passes=1,
        # probabilistic=True
        # num_passes_pred=20,
        # device=device,
        # dtype=torch.float32
    )
    
    
    kernel = gpplus.kernels.CombinedKernel_MVMF(
        # cont_cols=[],
        cat_cols=[np.arange(0, 10),np.arange(10,13),np.arange(13,29)],
        source_cols=np.arange(29, 32),
        # cat_encoder=cat_encoder,
        source_encoder=source_encoder,
        cat_combination_method="additive",
        source_combination_method="product",
        # cont_kernel= 
        # cat_kernel=gpplus.kernels.gaussian_kernel.GaussianKernel(),
        # source_kernel=gpplus.kernels.gaussian_kernel.GaussianKernel(),
        cat_encoder_type="matrix" 
    )
    # If using CUDA:
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # print(f'Seed: {seed}')
    

    DIM = X.shape[1]
    metrics=[]
    # for idx in range(4, arr.shape[1]):
    # Targets: column 6 = Porosity, column 7 = Hardness

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    # print(X_train[0])
    # print(X_test[0])

    # print(y_train[0])
    # print(y_test[0])

    y_train_std = y_train.std().item()
    prior_mean = np.log(y_train_std**2)
    noise_prior = gpytorch.priors.LogNormalPrior(loc=prior_mean, scale=1.0)
    from gpytorch.constraints import GreaterThan
    model = GPR(X_train, y_train, 
                # kernel_module=gpplus.kernels.gaussian_kernel.GaussianKernel(lengthscale_constraint=Interval(-6, 4)),
                # kernel_module=gpplus.kernels.gaussian_kernel.GaussianKernel(ard_num_dims=X_train.shape[1]),
                # kernel_module = gpytorch.kernels.RBFKernel(),
                kernel_module = kernel,
                mean_module = gpplus.means.MultipleMean(encoded_cols=np.arange(29,32)),
                # mean_module = gpplus.means.MultipleMean(qual_dict={4:3}),
                # mean_module=gpytorch.means.ZeroMean(),
                likelihood=gpytorch.likelihoods.GaussianLikelihood(noise_constraint = GreaterThan(1e-6), noise_prior=noise_prior), 
                seed=seed)
    
    # model.covar_module.inspect_kernel_references()
    if i == 1: print(model)
    # from gpplus.training.parameter_initializer import DefaultParameterInitializer
    from gpplus.training.parameter_initializer_original import DefaultParameterInitializer

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
        convergence_patience=20,
        optimizer_class=torch.optim.Adam,
        device='cpu',
        map_prior=True,
        # scheduler = .95,
        # optimizer_kwargs = {"lr": 1, "line_search_fn": "strong_wolfe"},
        initializer_class=DefaultParameterInitializer,  # Use the chosen initializer
    )
    # with torch.amp.autocast('cuda',dtype=torch.float32, cache_enabled=True):
    results = trainer.train()  # Returns a dict of results; you might store if needed.
    y_pred, pred_lower, pred_upper, output_std = evaluate_gp_model(model, X_test)
    metric = compute_metrics(y_test, y_pred, output_std)
    print(f"\nMetrics (Run # {i}):")
    for k, v in metric.items():
        print(f"{k}: {v:.4f}")
    full_metrics.append(metric)
# full_metrics.append(metrics)
t00f = time.time()

df_metrics = pd.DataFrame(full_metrics)

# Calculate the mean for each metric across all seeds
avg_metrics = df_metrics.mean().to_dict()
std_metrics = df_metrics.std().to_dict()

print(f"\nTime: {time.time()-t0:.2f} s\n({num_runs} runs. Lr = {lr}. {num_epochs} epochs)")
print(f"Average metrics {i} seeds (± std):")
# print(f"{i} seeds, time: ({t00f - t00:.2f} s)")
for metric in avg_metrics.keys():
    mean_val = avg_metrics[metric]
    std_val = std_metrics[metric]
    print(f"{metric}: {mean_val:.6f} ± {std_val:.6f}")
print('\n')


# %%


# print("V2 (Single forward pass, OH encoded source)")

# data1_t = torch.tensor(data1.values, dtype=torch.float32)
# data2_t = torch.tensor(data2.values, dtype=torch.float32)
# data3_t = torch.tensor(data3.values, dtype=torch.float32)

# data_all = torch.cat((data1_t, data2_t, data3_t), dim=0)  # dim=0 → stack rows
# t00 = time.time()
# arr = torch.tensor(data_all, dtype=torch.float32)
# col_to_encode = data_all[:, 3]

# # One-hot encode using pandas get_dummies
# one_hot = pd.get_dummies(col_to_encode).astype(int)

# # Ensure it has exactly 3 columns (0, 1, 2)
# one_hot = one_hot.reindex(columns=[0, 1, 2], fill_value=0)
# one_hot = torch.tensor(one_hot.values, dtype=torch.float32)

# data_rest = torch.cat([data_all[:, :3], data_all[:, 4:]], dim=1)

# # Concatenate one-hot and original data (drop the original column 3 before concatenating)
# data_OH = torch.cat([one_hot, data_rest], dim=1)

# # Convert to tensor
# arr = torch.tensor(data_OH, dtype=torch.float32)


# X = arr[:, :-1]   
# i = 0
# full_metrics=[]
# for seed in range(42,52):
#     t1 = time.time()
#     i += 1
#     np.random.seed(seed)
#     torch.manual_seed(seed)
    
#     # If using CUDA:
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#     # print(f'Seed: {seed}')
    

#     DIM = X.shape[1]
#     metrics=[]
#     # for idx in range(4, arr.shape[1]):
#     # Targets: column 6 = Porosity, column 7 = Hardness
#     y = arr[:, -1]

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
#     y_train_std = y_train.std().item()
#     prior_mean = np.log(y_train_std**2)
#     noise_prior = gpytorch.priors.LogNormalPrior(loc=prior_mean, scale=1.0)
#     from gpytorch.constraints import GreaterThan
#     model = GPR(X_train, y_train, 
#                 # kernel_module=gpplus.kernels.gaussian_kernel.GaussianKernel(lengthscale_constraint=Interval(-6, 4)),
#                 kernel_module=gpplus.kernels.gaussian_kernel.GaussianKernel(ard_num_dims=X_train.shape[1]),
#                 # kernel_module = gpytorch.kernels.RBFKernel(),
#                 # mean_module=gpytorch.means.ZeroMean(),
#                 likelihood=gpytorch.likelihoods.GaussianLikelihood(noise_constraint = GreaterThan(1e-6), noise_prior=noise_prior), 
#                 seed=seed)


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
#     results = trainer.train()  # Returns a dict of results; you might store if needed.
#     y_pred, pred_lower, pred_upper, output_std = evaluate_gp_model(model, X_test)
#     metric = compute_metrics(y_test, y_pred, output_std)
#     # print("\nMetrics:")
#     # for k, v in metric.items():
#     #     print(f"{k}: {v:.4f}")
#     full_metrics.append(metric)
# # full_metrics.append(metrics)
# t00f = time.time()

# df_metrics = pd.DataFrame(full_metrics)

# # Calculate the mean for each metric across all seeds
# avg_metrics = df_metrics.mean().to_dict()
# std_metrics = df_metrics.std().to_dict()

# print("Average metrics (± std):")
# # print(f"{i} seeds, time: ({t00f - t00:.2f} s)")
# for metric in avg_metrics.keys():
#     mean_val = avg_metrics[metric]
#     std_val = std_metrics[metric]
#     print(f"{metric}: {mean_val:.6f} ± {std_val:.6f}")
# print('\n')

# print("V3 (Forward pass for each source)")
# t0 = time.time()
# run = 0
# for data in all_data:
#     t00 = time.time()
#     arr = torch.tensor(data.to_numpy(), dtype=torch.float32)

#     # mask = ~torch.isnan(arr).any(dim=1)

#     # Apply mask to filter out all rows with any NaN
#     # arr = arr[mask]

#     arr = torch.tensor(arr, dtype=torch.float32)
#     run +=1
#     X = arr[:, :3]   
#     i = 0
#     full_metrics=[]
#     for seed in range(42,52):
#         t1 = time.time()
#         i += 1
#         np.random.seed(seed)
#         torch.manual_seed(seed)
        
#         # If using CUDA:
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed_all(seed)
#         # print(f'Seed: {seed}')
        
    
#         DIM = X.shape[1]
#         metrics=[]
#         # for idx in range(4, arr.shape[1]):
#         # Targets: column 6 = Porosity, column 7 = Hardness
#         y = arr[:, -1]
    
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
#         y_train_std = y_train.std().item()
#         prior_mean = np.log(y_train_std**2)
#         noise_prior = gpytorch.priors.LogNormalPrior(loc=prior_mean, scale=1.0)
#         from gpytorch.constraints import GreaterThan
#         model = GPR(X_train, y_train, 
#                     # kernel_module=gpplus.kernels.gaussian_kernel.GaussianKernel(lengthscale_constraint=Interval(-6, 4)),
#                     kernel_module=gpplus.kernels.gaussian_kernel.GaussianKernel(ard_num_dims=X_train.shape[1]),
#                     # kernel_module = gpytorch.kernels.RBFKernel(),
#                     # mean_module=gpytorch.means.ZeroMean(),
#                     likelihood=gpytorch.likelihoods.GaussianLikelihood(noise_constraint = GreaterThan(1e-6), noise_prior=noise_prior), 
#                     seed=seed)


#         trainer = gpplus.training.GPTrainer(
#             model=model,
#             num_epochs=num_epochs,
#             seed=seed,
#             num_runs=num_runs, #<-----
#             optimizer_kwargs={
#                 'lr': lr,
#                 # 'weight_decay': 1e-4,
#                 # 'eps': 1e-8,
#             },
#             convergence_patience=50,
#             optimizer_class=torch.optim.Adam,
#             device='cpu',
#             map_prior=True,
#             # scheduler = .95
#             # optimizer_kwargs = {"lr": 1, "line_search_fn": "strong_wolfe"},
#             # initializer_class=initializer_class,  # Use the chosen initializer
#         )
#         # with torch.amp.autocast('cuda',dtype=torch.float32, cache_enabled=True):
#         results = trainer.train()  # Returns a dict of results; you might store if needed.
#         y_pred, pred_lower, pred_upper, output_std = evaluate_gp_model(model, X_test)
#         metric = compute_metrics(y_test, y_pred, output_std)
#         # print("\nMetrics:")
#         # for k, v in metric.items():
#         #     print(f"{k}: {v:.4f}")
#         full_metrics.append(metric)
#     # full_metrics.append(metrics)
#     t00f = time.time()
    
#     df_metrics = pd.DataFrame(full_metrics)
    
#     # Calculate the mean for each metric across all seeds
#     avg_metrics = df_metrics.mean().to_dict()
#     std_metrics = df_metrics.std().to_dict()

#     print(f"Source {run-1} Average metrics (± std):")
#     # print(f"{i} seeds, time: ({t00f - t00:.2f} s)")
#     for metric in avg_metrics.keys():
#         mean_val = avg_metrics[metric]
#         std_val = std_metrics[metric]
#         print(f"{metric}: {mean_val:.6f} ± {std_val:.6f}")
#     print('\n')

    
# data1 = pd.read_csv("../../Datasets/HOIP_data/HOIP_noisy.csv", header=None)
# data2 = pd.read_csv("../../Datasets/HOIP_data/sample_low2.csv", header=None)
# data3 = pd.read_csv("../../Datasets/HOIP_data/sample_low3.csv", header=None)
# all_data = [data1, data2, data3]

# t0 = time.time()
# for data in all_data:
#     t00 = time.time()
#     arr = torch.tensor(data.to_numpy(), dtype=torch.float32)

#     # mask = ~torch.isnan(arr).any(dim=1)

#     # Apply mask to filter out all rows with any NaN
#     # arr = arr[mask]

#     arr = torch.tensor(arr, dtype=torch.float32)

#     X = arr[:, :3]   
#     DIM = X.shape[1]
#     i = 0
#     full_metrics=[]
#     for seed in range(42,52):
#         t1 = time.time()
#         i += 1
#         np.random.seed(seed)
#         torch.manual_seed(seed)
        
#         # If using CUDA:
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed_all(seed)
#         # print(f'Seed: {seed}')
        
    
#         metrics=[]
#         # for idx in range(4, arr.shape[1]):
#         # Targets: column 6 = Porosity, column 7 = Hardness
#         y = arr[:, -1]
    
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
#         y_train_std = y_train.std().item()
#         prior_mean = np.log(y_train_std**2)
#         noise_prior = gpytorch.priors.LogNormalPrior(loc=prior_mean, scale=1.0)
#         from gpytorch.constraints import GreaterThan
#         model = GPR(X_train, y_train, 
#                     # kernel_module=gpplus.kernels.gaussian_kernel.GaussianKernel(lengthscale_constraint=Interval(-6, 4)),
#                     kernel_module=gpplus.kernels.gaussian_kernel.GaussianKernel(ard_num_dims=X_train.shape[1]),
#                     # kernel_module = gpytorch.kernels.RBFKernel(),
#                     # mean_module=gpytorch.means.ZeroMean(),
#                     likelihood=gpytorch.likelihoods.GaussianLikelihood(noise_constraint = GreaterThan(1e-6), noise_prior=noise_prior), 
#                     seed=seed)


#         trainer = gpplus.training.GPTrainer(
#             model=model,
#             num_epochs=num_epochs,
#             seed=seed,
#             num_runs=num_runs, #<-----
#             optimizer_kwargs={
#                 'lr': lr,
#                 # 'weight_decay': 1e-4,
#                 # 'eps': 1e-8,
#             },
#             convergence_patience=50,
#             optimizer_class=torch.optim.Adam,
#             device='cpu',
#             map_prior=True,
#             # scheduler = .95
#             # optimizer_kwargs = {"lr": 1, "line_search_fn": "strong_wolfe"},
#             # initializer_class=initializer_class,  # Use the chosen initializer
#         )
#         # with torch.amp.autocast('cuda',dtype=torch.float32, cache_enabled=True):
#         results = trainer.train()  # Returns a dict of results; you might store if needed.
#         y_pred, pred_lower, pred_upper, output_std = evaluate_gp_model(model, X_test)
#         metric = compute_metrics(y_test, y_pred, output_std)
#         t1f = time.time()
#         # print(f"\nMetrics: ({t1f-t1:.2f} s)")
#         # for k, v in metric.items():
#         #     print(f"{k}: {v:.4f}")
#         full_metrics.append(metric)
#     t00f = time.time()
#     df_metrics = pd.DataFrame(full_metrics)
    
#     # Calculate the mean for each metric across all seeds
#     avg_metrics = df_metrics.mean().to_dict()
#     std_metrics = df_metrics.std().to_dict()

#     print("\nAverage metrics (± std):")
#     # print(f"{i} seeds, time: ({t00f - t00:.2f} s)")
#     for metric in avg_metrics.keys():
#         mean_val = avg_metrics[metric]
#         std_val = std_metrics[metric]
#         print(f"{metric}: {mean_val:.6f} ± {std_val:.6f}")
