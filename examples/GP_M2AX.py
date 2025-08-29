import random
import matplotlib.pyplot as plt
import torch
import numpy as np
import sys
import argparse
import pandas as pd
import warnings
import time
from datetime import datetime

# from _GITBO import *
warnings.filterwarnings("ignore")

import gpytorch
import gpplus
from gpplus.models import GPR
from gpplus.TestProblems_Utils import *
from gpplus.training.eval import evaluate_gp_model
from gpplus.utils.one_hot_to_latent_nn import OneHotToLatent
from gpplus.utils.matrix_encoder import MatrixEncoder, SourceMatrixEncoder
from gpplus.utils.metrics_functions import compute_metrics
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

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

tkwargs = {"device": torch.device(device), "dtype": torch.float32}
amp_dtype = tkwargs["dtype"]
amp_device = device

print("M2AX Dataset, GP")

# Load data
data1 = pd.read_csv("../../Datasets/data_M.csv")

cat_cols = ['Msiteelement', 'Asiteelement', 'Xsiteelement']
data = data1
# Factorize each categorical column separately
for col in cat_cols:
    data[col], _ = pd.factorize(data[col])

data = torch.tensor(data.to_numpy(), dtype=torch.float32)

# Now call your one-hot encoder
encoded_cols, info = one_hot_encode_integer_columns(data[:,:3])

encoded_data = torch.cat([encoded_cols, data[:,3:]], dim=1)

t00 = time.time()

# Print column info
tensor = np.array(encoded_data)
for c in info:
    print(c)
print("Final shape:", tuple(encoded_cols.shape))

# Convert to tensor
arr = torch.tensor(encoded_data, dtype=torch.float32)

X = arr[:, :len(encoded_cols[0])]   
xnp = np.array(X)
# y = arr[:, -1]
# print("E, Young's Modulus")
# y = arr[:, -2]
# print("G, Shear Modulus")
y = arr[:, -3]
print("B, Bulk Modulus")

num_epochs = 1000
num_runs = 16
lr = 1

i = 0
full_metrics = []
t0 = time.time()
for seed in range(42, 44):
    t1 = time.time()
    i += 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # If using CUDA:
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Initialize architectures
    cat_architecture = {
        'hidden_dims': [],
        'activation': 'relu',
        'dropout': 0.2
    }

    # Initialize encoders with custom architectures
    # cat_encoder = MatrixEncoder(
    #     input_dim=len(X[0]),
    #     z_dim=2,
    #     initialization='normal',
    #     init_std=0.1
    # )

    # source_architecture = {
    #     'hidden_dims': [5],
    #     'activation': 'hardtanh',
    #     'dropout': 0.0
    # }

    # source_encoder = OneHotToLatent(
    #     input_dim=4,
    #     architecture_config=source_architecture,
    #     z_dim=2,
    #     num_passes=1,
    # )
    
    kernel = gpplus.kernels.CombinedKernel_MVMF(
        # cont_cols=[],
        cat_cols=[np.arange(0, 10), np.arange(10, 22), np.arange(22, 24)],
        # source_cols=np.arange(29, 32),
        # cat_encoder=cat_encoder,
        cat_encoder="nn",
        # source_encoder=source_encoder,
        cat_combination_method="additive",
        # source_combination_method="additive",
        cat_kernel=gpplus.kernels.gaussian_kernel.GaussianKernel(),
        # source_kernel=gpplus.kernels.gaussian_kernel.GaussianKernel(),
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_train = X_train.to(torch.float32)
    X_test = X_test.to(torch.float32)
    y_train = y_train.to(torch.float32)
    y_test = y_test.to(torch.float32)
    
    y_train_std = y_train.std().item()
    prior_mean = np.log(y_train_std**2)
    noise_prior = gpytorch.priors.LogNormalPrior(loc=prior_mean, scale=1.0)
    from gpytorch.constraints import GreaterThan
    
    model = GPR(X_train, y_train, 
                kernel_module=kernel,
                likelihood=gpytorch.likelihoods.GaussianLikelihood(noise_constraint=GreaterThan(1e-6), noise_prior=noise_prior), 
                seed=seed)

    if i == 1: 
        print(model)
    
    from gpplus.training.parameter_initializer_original import DefaultParameterInitializer

    trainer = gpplus.training.GPTrainer(
        model=model,
        num_epochs=num_epochs,
        seed=seed,
        num_runs=num_runs,
        optimizer_kwargs={
            'lr': lr,
        },
        convergence_patience=500,
        optimizer_class=torch.optim.Adam,
        device='cpu',
        map_prior=True,
        initializer_class=DefaultParameterInitializer,
    )
    
    results = trainer.train()
    y_pred, pred_lower, pred_upper, output_std = evaluate_gp_model(model, X_test)
    metric = compute_metrics(y_test, y_pred, output_std, start_time=t1)
    
    print(f"\nMetrics (Run # {i}):")
    for k, v in metric.items():
        print(f"{k}: {v:.4f}")
    
    full_metrics.append(metric)

t00f = time.time()

df_metrics = pd.DataFrame(full_metrics)

# Calculate the mean for each metric across all seeds
avg_metrics = df_metrics.mean().to_dict()
std_metrics = df_metrics.std().to_dict()

print(f"\nTime: {time.time()-t0:.2f} s\n({num_runs} runs. Lr = {lr}. {num_epochs} epochs)")
print(f"Average metrics {i} seeds (± std):")
for metric in avg_metrics.keys():
    mean_val = avg_metrics[metric]
    std_val = std_metrics[metric]
    print(f"{metric}: {mean_val:.6f} ± {std_val:.6f}")
print('\n')
