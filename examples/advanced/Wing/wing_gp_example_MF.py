import time

import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import gpplus
from examples.data.data_gen import load_data_wing_MV_MF
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


# Generate data using load_data_wing_MV_MF
seed = 42
set_seed(seed)

# Define training and test samples per source
n_train = {"s0": 100, "s1": 100, "s2": 100, "s3": 100}
n_test = {"s0": 2500, "s1": 2500, "s2": 2500, "s3": 2500}

print("\nGenerating data using load_data_wing_MV_MF...")
data = load_data_wing_MV_MF(
    seed=seed,
    n_train=n_train,
    n_test=n_test,
    noise_levels=[0.0, 0.0, 0.0, 0.0],
    shuffle=True,
    qual_dict={},  # No categorical variables for wing problem
    return_one_hot=False,  # if true, only one-hot encodings are returned
)

# Extract data
X_train = data["x_train_full"]
y_train = data["y_train_full"]
X_test = data["x_test_full"]
y_test = data["y_test_full"]

# Get column information from metadata
cont_cols = list(range(4, 14))     # or [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
source_cols = list(range(0, 4))    # or [0, 1, 2, 3]

print(f"Xtrainshape: {X_train.shape}")
print(f"Xtestshape: {X_test.shape}")
print(f"\nFinal training dataset shape: X={X_train.shape}, y={y_train.shape}")
print(f"Final test dataset shape: X={X_test.shape}, y={y_test.shape}")


# Print ranges for each of the 10 original features (excluding source one-hot)
print("\nX feature ranges (10 original features):")
for i in range(10):
    print(f"  Feature {i}: [{X_train[:, i].min():.4f}, {X_train[:, i].max():.4f}]")

print(f"\ny range: [{y_train.min():.4f}, {y_train.max():.4f}]")

# Standardize training and test inputs (continuous features)
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train[:, cont_cols])
X_test_scaled = scaler_X.transform(X_test[:, cont_cols])

# Combine scaled continuous features with source one-hot for training
X_train_scaled = torch.cat(
    [
        torch.tensor(X_train_scaled, dtype=torch.float64),
        X_train[:, source_cols],  # Source one-hot columns
    ],
    dim=1,
)

# Combine scaled continuous features with source one-hot for test
X_test_scaled = torch.cat(
    [
        torch.tensor(X_test_scaled, dtype=torch.float64),
        X_test[:, source_cols],  # Source one-hot columns
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
kernel = gpplus.kernels.MVMFKernel(
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
    mean_module=gpplus.means.MultiMean(source_cols=source_cols),
    # mean_module=gpytorch.means.ConstantMean(),
    likelihood=gpplus.likelihoods.MultiLikelihood(encoded_cols=source_cols, training_data=X_train_scaled),
    # likelihood=gpytorch.likelihoods.GaussianLikelihood(),
)

num_inits = 4

print(model)
# Create trainer
trainer = gpplus.training.GPTrainer(
    model=model,
    seed=seed,
    num_epochs=10000,  # Not required if not using Adam
    num_inits=num_inits,
    stop_conditions=[
        gpplus.training.ConvergencePatienceStopCondition(patience=50),
        gpplus.training.MinLossChangeStopCondition(min_loss_change=1e-7),
    ],
    # optimizer_class=torch.optim.Adam,
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
y_pred_orig = scaler_y.inverse_transform(y_pred_scaled.detach().cpu().numpy().reshape(-1, 1)).flatten()
output_std_orig = output_std_scaled.detach().cpu().numpy() * scaler_y.scale_[0]  # Scale the uncertainty


# Compute metrics on original scale
metric = compute_metrics(y_test_orig, y_pred_orig, output_std_orig, start_time=t1)

print("Metrics:")
for k, v in metric.items():
    print(f"  {k}: {v:.4f}")
