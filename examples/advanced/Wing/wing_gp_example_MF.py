import time

import numpy as np
import torch
from scipy.stats.qmc import Sobol
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import gpplus
from examples.data.data_gen import wing_mixed_variables
from gpplus.models import GPR
from gpplus.training.callbacks import PrintInitialParametersCallback
from gpplus.training.eval import evaluate_gp_model
from gpplus.utils import set_seed


def compute_metrics(y_true, y_hat, output_std=None, start_time=None):
    """
    Compute basic metrics for predictions.

    Args:
        y_true: True values (1D array)
        y_hat: Predicted values (1D array)
        output_std: Standard deviation of predictions (optional)
        start_time: Start time for timing (optional)

    Returns:
        dict: Dictionary with computed metrics
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy().reshape(-1)
    if isinstance(y_hat, torch.Tensor):
        y_hat = y_hat.detach().cpu().numpy().reshape(-1)
    if output_std is not None and isinstance(output_std, torch.Tensor):
        output_std = output_std.detach().cpu().numpy().reshape(-1)

    # Only add time if start_time is provided
    if start_time is not None:
        metrics = {
            "Time": time.time() - start_time,
            "RRMSE": np.sqrt(mean_squared_error(y_true, y_hat)) / y_true.std(),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_hat)),
            "MSE": mean_squared_error(y_true, y_hat),
        }
    else:
        metrics = {
            "RRMSE": np.sqrt(mean_squared_error(y_true, y_hat)) / y_true.std(),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_hat)),
            "MSE": mean_squared_error(y_true, y_hat),
        }

    # Add NIS if output_std is provided
    if output_std is not None:
        z = 1.96
        L = y_hat - z * output_std
        U = y_hat + z * output_std
        width = U - L
        below = (L - y_true) * (y_true < L)
        above = (y_true - U) * (y_true > U)
        interval_score = width + (2 / 0.05) * below + (2 / 0.05) * above
        NIS = interval_score.mean() / y_true.std()
        metrics["NIS"] = NIS
        return metrics

    return metrics


# Generate Sobol sequence for X inputs
seed = 42
set_seed(seed)

# Generate training and test data for all fidelity levels s0-s3
sources = ["s0", "s1", "s2", "s3"]
num_train_per_source = [100, 100, 100, 100]
num_test_per_source = [2500, 2500, 2500, 2500]  # 10k total / 4 sources = 2.5k per source

l_bound = torch.tensor([150.0, 220.0, 6.0, -10.0, 16.0, 0.5, 0.08, 2.5, 1700.0, 0.025])
u_bound = torch.tensor([200.0, 300.0, 10.0, 10.0, 45.0, 1.0, 0.18, 6.0, 2500.0, 0.08])


# Generate test data
print("\nGenerating test data...")
X_test_data = []
y_test_data = []
for i, source in enumerate(sources):
    sobol = Sobol(d=10, seed=seed)
    X_sobol_raw = torch.tensor(sobol.random(num_test_per_source[i]))

    # Scale Sobol samples to the proper bounds
    X_sobol = X_sobol_raw * (u_bound - l_bound) + l_bound

    print(f"Test source: {source}")
    result = wing_mixed_variables(
        X=X_sobol,
        source=source,
    )

    # Create one-hot encoding for source
    source_one_hot = torch.zeros(num_test_per_source[i], 4)
    source_one_hot[:, i] = 1

    # Concatenate original features with source one-hot encoding
    X_with_source = torch.cat([X_sobol, source_one_hot], dim=1)

    X_test_data.append(X_with_source)
    y_test_data.append(result)
    print(f"  Test samples: {X_with_source.shape[0]}, Result range: [{result.min():.4f}, {result.max():.4f}]")

# Concatenate all test data (10,000x14)
X_test = torch.cat(X_test_data, dim=0)
y_test = torch.cat(y_test_data, dim=0)

# Generate training data
print("Generating training data...")
X_train_data = []
y_train_data = []
for i, source in enumerate(sources):
    sobol = Sobol(d=10, seed=seed)
    X_sobol_raw = torch.tensor(sobol.random(num_train_per_source[i]))

    # Scale Sobol samples to the proper bounds
    X_sobol = X_sobol_raw * (u_bound - l_bound) + l_bound

    print(f"Training source: {source}")
    result = wing_mixed_variables(
        X=X_sobol,
        source=source,
    )

    # Create one-hot encoding for source (s0=[1,0,0,0], s1=[0,1,0,0], etc.)
    source_one_hot = torch.zeros(num_train_per_source[i], 4)
    source_one_hot[:, i] = 1

    # Concatenate original features with source one-hot encoding
    X_with_source = torch.cat([X_sobol, source_one_hot], dim=1)

    X_train_data.append(X_with_source)
    y_train_data.append(result)
    print(f"  Training samples: {X_with_source.shape[0]}, Result range: [{result.min():.4f}, {result.max():.4f}]")
# Concatenate all training data (400x14)
X_train = torch.cat(X_train_data, dim=0)
y_train = torch.cat(y_train_data, dim=0)

cont_cols = np.arange(0, 10)
source_cols = np.arange(10, 14)

# Shuffle both training and test datasets
print("Shuffling datasets...")
train_indices = torch.randperm(X_train.shape[0])
test_indices = torch.randperm(X_test.shape[0])

X_train = X_train[train_indices]
y_train = y_train[train_indices]
X_test = X_test[test_indices]
y_test = y_test[test_indices]

print(f"\nFinal training dataset shape: X={X_train.shape}, y={y_train.shape}")
print(f"Final test dataset shape: X={X_test.shape}, y={y_test.shape}")

# Print ranges for each of the 10 original features (excluding source one-hot)
print("\nX feature ranges (10 original features):")
for i in range(10):
    print(f"  Feature {i}: [{X_train[:, i].min():.4f}, {X_train[:, i].max():.4f}]")

print(f"\ny range: [{y_train.min():.4f}, {y_train.max():.4f}]")

# Standardize training and test inputs (continuous features)
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train[:, :10])
X_test_scaled = scaler_X.transform(X_test[:, :10])

# Combine scaled continuous features with source one-hot for training
X_train_scaled = torch.cat(
    [
        torch.tensor(X_train_scaled, dtype=torch.float64),
        X_train[:, 10:],  # Source one-hot columns
    ],
    dim=1,
)

# Combine scaled continuous features with source one-hot for test
X_test_scaled = torch.cat(
    [
        torch.tensor(X_test_scaled, dtype=torch.float64),
        X_test[:, 10:],  # Source one-hot columns
    ],
    dim=1,
)


# Standardize y
scaler_y = StandardScaler()
y_train_scaled = torch.tensor(scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten())

print("Standardized data shapes:")
print(f"  X_train: {X_train_scaled.shape}, y_train: {y_train_scaled.shape}")
print(f"  X_test: {X_test_scaled.shape}, y_test: {y_test.shape}")

t1 = time.time()

source_encoder = gpplus.utils.encoders.MatrixEncoder(input_dim=4, z_dim=2)
source_encoder2 = gpplus.utils.encoders.NeuralEncoder(
    input_dim=4, architecture_config={"hidden_dims": [], "activation": "hardtanh", "dropout": 0.0}, z_dim=2
)

# Create model
kernel = gpplus.kernels.CombinedKernel(
    cont_cols=cont_cols,
    cat_cols=None,
    source_cols=source_cols,
    source_encoder=source_encoder,
    # source_encoder=source_encoder2,
)

model = GPR(
    X_train_scaled,
    y_train_scaled,
    # kernel_module=gpplus.kernels.GaussianKernel(),
    kernel_module=kernel,
    mean_module=gpplus.means.MultipleMean(encoded_cols=source_cols),
    # mean_module=gpytorch.means.ConstantMean(),
    likelihood=gpplus.likelihoods.MultiLikelihood(encoded_cols=source_cols, training_data=X_train_scaled),
    # likelihood=gpytorch.likelihoods.GaussianLikelihood(),
)

num_epochs = 1000
num_runs = 4
lr = 0.1

print(model)
# from gpplus.training.parameter_initializer_kian import DefaultParameterInitializer
# Create trainer
trainer = gpplus.training.GPTrainer(
    model=model,
    num_epochs=num_epochs,
    seed=seed,
    num_runs=num_runs,
    optimizer_kwargs={"lr": lr},
    convergence_patience=50,
    optimizer_class=torch.optim.Adam,
    device="cuda",
    callbacks=[PrintInitialParametersCallback()],
    # initializer_class=DefaultParameterInitializer
)

print("Training model...")
results = trainer.train()

# Evaluate on standardized test data
y_pred_scaled, pred_lower_scaled, pred_upper_scaled, output_std_scaled = evaluate_gp_model(model, X_test_scaled)

# Transform predictions back to original scale for proper metrics
y_test_orig = y_test  # Already in original scale
y_pred_orig = scaler_y.inverse_transform(y_pred_scaled.numpy().reshape(-1, 1)).flatten()
output_std_orig = output_std_scaled * scaler_y.scale_[0]  # Scale the uncertainty


# Compute metrics on original scale

metric = compute_metrics(y_test_orig, y_pred_orig, output_std_orig, start_time=t1)

print("Metrics:")
for k, v in metric.items():
    print(f"  {k}: {v:.4f}")
