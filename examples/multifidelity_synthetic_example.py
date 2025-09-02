#!/usr/bin/env python3
"""
Multifidelity Synthetic Data Example
====================================

This script demonstrates a complete multifidelity Gaussian Process setup using:
1. Multimean: Different mean functions for each fidelity level
2. Multinoise: Different noise levels for each fidelity level
3. Source encoding: Neural network encoding of fidelity levels

The example generates synthetic data from mathematical equations with different fidelity levels.
"""

import os
import random
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# GPyTorch and GPPlus imports
import gpytorch
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.mlls import ExactMarginalLogLikelihood

from gpplus.kernels import CombinedKernel_MVMF
from gpplus.likelihood import MultiLikelihood
from gpplus.means import MultipleMean
from gpplus.models import GPR
from gpplus.training import GPTrainer
from gpplus.training.callbacks import PrintLossCallback
from gpplus.training.eval import evaluate_gp_model
from gpplus.utils.matrix_encoder import MatrixEncoder, SourceMatrixEncoder
from gpplus.utils.metrics_functions import compute_metrics, compute_per_source_metrics
from gpplus.utils.one_hot_to_latent_nn import OneHotToLatent

# Set device - use CPU to avoid device mismatch issues
device = torch.device("cpu")
print(f"Using device: {device}")


def generate_synthetic_data(n_samples_per_fidelity, noise_levels, seed=42):
    """
    Generate synthetic multifidelity data from mathematical equations.

    Args:
        n_samples_per_fidelity: Dict with number of samples per fidelity level
        noise_levels: Dict with noise standard deviation per fidelity level
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing training data, test data, and metadata
    """
    np.random.seed(seed)

    # Define the base function: f(x) = sin(2πx) + 0.5*sin(4πx) + 0.25*sin(8πx)
    def high_fidelity_function(x):
        """High fidelity function with multiple frequency components."""
        return np.sin(2 * np.pi * x) + 0.5 * np.sin(4 * np.pi * x) + 0.25 * np.sin(8 * np.pi * x)

    def medium_fidelity_function(x):
        """Medium fidelity function with fewer frequency components."""
        return np.sin(2 * np.pi * x) + 0.5 * np.sin(4 * np.pi * x)

    def low_fidelity_function(x):
        """Low fidelity function with only basic frequency."""
        return np.sin(2 * np.pi * x)

    # Generate input data in [0, 1]
    x_train_list = []
    y_train_list = []
    source_train_list = []

    x_test_list = []
    y_test_list = []
    source_test_list = []

    fidelity_functions = {
        "s0": high_fidelity_function,  # High fidelity
        "s1": medium_fidelity_function,  # Medium fidelity
        "s2": low_fidelity_function,  # Low fidelity
    }

    # Generate training data
    for source, n_samples in n_samples_per_fidelity.items():
        # Generate input points
        x_source = np.random.uniform(0, 1, n_samples).reshape(-1, 1)

        # Generate outputs using appropriate fidelity function
        y_source = fidelity_functions[source](x_source.flatten())

        # Add noise - handle both dict and list formats
        if isinstance(noise_levels, dict):
            # Dictionary format: noise_levels = {'s0': 0.01, 's1': 0.05, 's2': 0.15}
            if source in noise_levels:
                noise_std = noise_levels[source]
            else:
                noise_std = 0.1
                print(f"Warning: No noise level specified for {source}, using default {noise_std}")
        elif isinstance(noise_levels, (list, tuple, np.ndarray)):
            # List format: noise_levels = [0.01, 0.05, 0.15] - use index
            source_index = list(fidelity_functions.keys()).index(source)
            if source_index < len(noise_levels):
                noise_std = noise_levels[source_index]
            else:
                noise_std = 0.1
                print(f"Warning: Noise level list too short for {source}, using default {noise_std}")
        else:
            # Single value format: noise_levels = 0.1 (apply to all)
            noise_std = noise_levels
            print(f"Using single noise level {noise_std} for all fidelities")

        noise = np.random.normal(0, noise_std, n_samples)
        y_source += noise

        # Create source encoding (one-hot)
        source_encoding = np.zeros((n_samples, len(fidelity_functions)))
        source_encoding[:, list(fidelity_functions.keys()).index(source)] = 1

        # Split into train/test (80/20)
        n_train = int(0.8 * n_samples)
        n_test = n_samples - n_train

        # Training data
        x_train_list.append(x_source[:n_train])
        y_train_list.append(y_source[:n_train])
        source_train_list.append(source_encoding[:n_train])

        # Test data
        x_test_list.append(x_source[n_train:])
        y_test_list.append(y_source[n_train:])
        source_test_list.append(source_encoding[n_train:])

    # Combine all data
    x_train = np.vstack(x_train_list)
    y_train = np.concatenate(y_train_list)
    source_train = np.vstack(source_train_list)

    x_test = np.vstack(x_test_list)
    y_test = np.concatenate(y_test_list)
    source_test = np.vstack(source_test_list)

    # Convert to tensors
    x_train = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
    source_train = torch.tensor(source_train, dtype=torch.float32, device=device)

    x_test = torch.tensor(x_test, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test, dtype=torch.float32, device=device)
    source_test = torch.tensor(source_test, dtype=torch.float32, device=device)

    # Create data dictionary
    data = {
        "x_train": x_train,
        "y_train": y_train,
        "source_train": source_train,
        "x_test": x_test,
        "y_test": y_test,
        "source_test": source_test,
        "metadata": {
            "source_names": list(fidelity_functions.keys()),
            "n_train": n_samples_per_fidelity,
            "noise_levels": noise_levels,
            "fidelity_functions": fidelity_functions,
        },
    }

    return data


def create_synthetic_multifidelity_function(X=None, n_samples=100, fidelity_level=0, noise_std=0.0, random_state=None):
    """
    Create synthetic multifidelity data similar to the reference implementation.

    Args:
        X: Input features (if None, generates random features)
        n_samples: Number of samples to generate
        fidelity_level: Fidelity level (0=high, 1=medium, 2=low)
        noise_std: Standard deviation of noise to add
        random_state: Random seed for reproducibility

    Returns:
        X, y: Input features and target values
    """
    if random_state is not None:
        np.random.seed(random_state)

    if X is None:
        # Generate random input features in [0, 1]
        X = np.random.uniform(0, 1, (n_samples, 1))

    # Define fidelity functions
    if fidelity_level == 0:  # High fidelity
        y = np.sin(2 * np.pi * X[:, 0]) + 0.5 * np.sin(4 * np.pi * X[:, 0]) + 0.25 * np.sin(8 * np.pi * X[:, 0])
    elif fidelity_level == 1:  # Medium fidelity
        y = np.sin(2 * np.pi * X[:, 0]) + 0.5 * np.sin(4 * np.pi * X[:, 0])
    elif fidelity_level == 2:  # Low fidelity
        y = np.sin(2 * np.pi * X[:, 0])
    else:
        raise ValueError("Only 3 fidelity levels (0, 1, 2) have been implemented")

    # Add noise if specified
    if noise_std > 0.0:
        y += np.random.normal(0, noise_std, y.shape)

    return X, y


def generate_multifidelity_dataset(n_samples_per_fidelity, noise_levels, random_state=42):
    """
    Generate a complete multifidelity dataset using the synthetic function.

    Args:
        n_samples_per_fidelity: Dict with number of samples per fidelity level
        noise_levels: Dict with noise standard deviation per fidelity level
        random_state: Random seed for reproducibility

    Returns:
        X_all, y_all: Combined input features and target values
    """
    X_list = []
    y_list = []

    for level, num_samples in n_samples_per_fidelity.items():
        if num_samples > 0:
            # Generate data for this fidelity level
            X_level, y_level = create_synthetic_multifidelity_function(
                n_samples=num_samples,
                fidelity_level=int(level),
                noise_std=noise_levels[level],
                random_state=random_state,
            )

            # Add fidelity indicator column
            fidelity_col = np.ones(num_samples).reshape(-1, 1) * float(level)
            X_level = np.hstack([X_level, fidelity_col])

            X_list.append(X_level)
            y_list.append(y_level)

    # Combine all data
    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)

    return X_all, y_all


def main():
    """Main function to run the complete multifidelity example."""

    print("=" * 60)
    print("MULTIFIDELITY SYNTHETIC DATA EXAMPLE")
    print("=" * 60)

    # Configuration
    n_samples_per_fidelity = {
        "s0": 100,  # High fidelity: 100 samples
        "s1": 150,  # Medium fidelity: 150 samples
        "s2": 200,  # Low fidelity: 200 samples
    }

    # Different noise levels for each fidelity - you can use any of these formats:

    # Option 1: Dictionary format (recommended)
    noise_levels = {
        "s0": 0.001,  # High fidelity: very low noise (high quality)
        "s1": 0.01,  # Medium fidelity: low noise
        "s2": 0.05,  # Low fidelity: medium noise (low quality)
    }

    # Option 2: List format (noise levels in order: s0, s1, s2)
    # noise_levels = [0.01, 0.05, 0.15]

    # Option 3: Single value (same noise for all fidelities)
    # noise_levels = 0.1

    print(f"Fidelity noise levels:")
    if isinstance(noise_levels, dict):
        for source, noise in noise_levels.items():
            print(f"  {source}: σ = {noise:.3f}")
    elif isinstance(noise_levels, (list, tuple, np.ndarray)):
        for i, noise in enumerate(noise_levels):
            print(f"  s{i}: σ = {noise:.3f}")
    else:
        print(f"  All fidelities: σ = {noise_levels:.3f}")

    print(f"Generating synthetic data with {sum(n_samples_per_fidelity.values())} total samples...")
    print(f"Fidelity levels: {list(n_samples_per_fidelity.keys())}")
    print(f"Noise levels: {noise_levels}")

    # Generate synthetic data
    data = generate_synthetic_data(n_samples_per_fidelity, noise_levels, seed=seed)

    print(f"Data generated successfully!")
    print(f"Training samples: {len(data['x_train'])}")
    print(f"Test samples: {len(data['x_test'])}")

    # Create multifidelity GP model
    print("\nCreating multifidelity GP model...")

    x_train = data["x_train"]
    y_train = data["y_train"]
    source_train = data["source_train"]

    # Combine input features (continuous + source)
    x_train_combined = torch.cat([x_train, source_train], dim=1)

    # Define column indices - use the working reference approach
    continuous_cols = [0]  # x coordinate
    source_cols = [1, 2, 3]  # one-hot encoded sources (0-based indexing)
    # source_cols = np.arange(1, 4)  # one-hot encoded sources

    # Create source encoder - following working examples
    source_architecture = {
        "hidden_dims": [],
        "activation": "hardtanh",  # Use hardtanh like in working examples
        "dropout": 0.0,
    }

    source_encoder = OneHotToLatent(
        input_dim=len(data["metadata"]["source_names"]), architecture_config=source_architecture, z_dim=2, num_passes=1
    )
    # source_encoder = MatrixEncoder(
    #     input_dim=len(data['metadata']['source_names']),
    #     z_dim=2,
    #     initialization='normal',
    #     init_std=0.5
    # )

    # Create combined kernel
    kernel = CombinedKernel_MVMF(
        cont_cols=continuous_cols,
        cat_cols=[],  # No categorical columns
        source_cols=source_cols,
        source_encoder=source_encoder,
        source_combination_method="additive",
    )

    # Create multimean - use exact working reference approach
    multimean = MultipleMean(encoded_cols=source_cols)

    # Data standardization - following working examples
    y_mean = y_train.mean()
    y_std = y_train.std().clamp_min(1e-8)
    y_train_std = (y_train - y_mean) / y_std

    # Create noise prior - following working examples
    y_train_std_val = y_train_std.std().item()
    prior_mean = np.log(y_train_std_val**2)
    noise_prior = gpytorch.priors.LogNormalPrior(loc=prior_mean, scale=1.0)

    # Create multinoise likelihood - following working examples
    multinoise_likelihood = MultiLikelihood(encoded_cols=x_train_combined[:, source_cols], noise_prior=noise_prior)

    # Create the GP model
    model = GPR(
        train_x=x_train_combined,
        train_y=y_train_std,  # Use standardized y
        mean_module=multimean,
        kernel_module=kernel,
        likelihood=multinoise_likelihood,
        seed=42,
    ).to(device)

    # Ensure the model is properly initialized
    model.eval()
    model.train()

    print(f"Model created with:")
    print(f"  - Multimean: {type(model.mean_module).__name__}")
    print(f"  - Multinoise: {type(model.likelihood).__name__}")
    print(f"  - Source encoder: {type(model.covar_module.source_encoder).__name__}")

    print(f"Model: {model}")

    # Show the ACTUAL model's covariance structure and parameters
    print(f"\n" + "=" * 60)
    print("ACTUAL MODEL COVARIANCE INFORMATION")
    print("=" * 60)

    # Print the actual kernel parameters
    print(f"\nKernel Parameters:")
    if hasattr(model.covar_module, "cont_kernel") and model.covar_module.cont_kernel is not None:
        print(f"  Continuous kernel:")
        if hasattr(model.covar_module.cont_kernel, "lengthscale"):
            print(f"    Lengthscale: {model.covar_module.cont_kernel.lengthscale}")
        if hasattr(model.covar_module.cont_kernel, "outputscale"):
            print(f"    Outputscale: {model.covar_module.cont_kernel.outputscale}")

    if hasattr(model.covar_module, "source_kernel") and model.covar_module.source_kernel is not None:
        print(f"  Source kernel:")
        if hasattr(model.covar_module.source_kernel, "lengthscale"):
            print(f"    Lengthscale: {model.covar_module.source_kernel.lengthscale}")
        if hasattr(model.covar_module.source_kernel, "outputscale"):
            print(f"    Outputscale: {model.covar_module.source_kernel.outputscale}")

    # Print the actual source encoder parameters
    if hasattr(model.covar_module, "source_encoder") and model.covar_module.source_encoder is not None:
        print(f"\nSource Encoder Parameters:")
        if hasattr(model.covar_module.source_encoder, "projection_matrix"):
            proj_matrix = model.covar_module.source_encoder.projection_matrix
            print(f"  Projection matrix shape: {proj_matrix.shape}")
            print(f"  Projection matrix:\n{proj_matrix}")

    # Print the actual likelihood noise parameters
    print(f"\nLikelihood Parameters:")
    print(f"  Noise parameters: {model.likelihood.noise}")
    print(f"  Raw noise: {model.likelihood.raw_noise}")

    # Print the actual mean parameters
    print(f"\nMean Parameters:")
    if hasattr(model.mean_module, "means"):
        for i, mean_func in enumerate(model.mean_module.means):
            print(f"  Mean {i} ({type(mean_func).__name__}):")
            for name, param in mean_func.named_parameters():
                print(f"    {name}: {param}")
    else:
        print(f"  Mean module: {type(model.mean_module).__name__}")
        for name, param in model.mean_module.named_parameters():
            print(f"    {name}: {param}")

    print("\n" + "=" * 60)

    # Train the model - following working examples
    print("\nTraining multifidelity GP model...")
    import time

    training_start_time = time.time()

    # Debug: Check initial noise parameters
    print(f"Initial noise parameters: {model.likelihood.noise}")

    trainer = GPTrainer(
        model=model,
        num_epochs=1500,  # Slightly increased for SOURCE_0 to get RRMSE < 0.2
        seed=42,
        num_runs=4,  # Keep it small - it's just 1D!
        optimizer_kwargs={"lr": 0.1},  # Use 0.1 like working examples
        convergence_patience=20,  # Keep it simple
        optimizer_class=torch.optim.Adam,
        device="cpu",
        map_prior=True,
    )

    trainer.train()
    training_time = time.time() - training_start_time
    print(f"Training completed in {training_time:.2f} seconds!")

    # Debug: Check model parameters after training
    print(f"\nModel Parameters After Training:")
    print(f"  Likelihood noise parameters: {model.likelihood.noise}")

    # Debug: Print mean hyperparameters after training
    print(f"\nFinal Mean Hyperparameters:")
    if hasattr(model.mean_module, "means"):
        for i, mean_func in enumerate(model.mean_module.means):
            print(f"  Mean {i} ({type(mean_func).__name__}):")
            # Show all named parameters for each mean function
            for name, param in mean_func.named_parameters():
                print(f"    {name}: {param}")
    else:
        print(f"  Mean module structure: {model.mean_module}")
        if hasattr(model.mean_module, "parameters"):
            for name, param in model.mean_module.named_parameters():
                print(f"    {name}: {param}")

    # Print individual noise levels for each fidelity
    if hasattr(model.likelihood, "noise") and len(model.likelihood.noise) > 1:
        print(f"  Individual fidelity noise levels:")
        for i, noise_level in enumerate(model.likelihood.noise):
            print(f"    Fidelity {i}: σ = {noise_level:.6f}")

    if hasattr(model.covar_module, "base_kernel") and hasattr(model.covar_module.base_kernel, "lengthscale"):
        print(f"  Kernel lengthscale: {model.covar_module.base_kernel.lengthscale}")
    if hasattr(model.covar_module, "lengthscale"):
        print(f"  Kernel lengthscale: {model.covar_module.lengthscale}")
    print(f"  Model state: {'Trained' if model.training else 'Evaluation'}")

    # Evaluate the model using metrics functions
    print("\nEvaluating model performance...")
    model.eval()

    # Prepare test data
    x_test = data["x_test"]
    y_test = data["y_test"]
    source_test = data["source_test"]
    x_test_combined = torch.cat([x_test, source_test], dim=1)

    # Make predictions with timing
    pred_start_time = time.time()

    # Use the existing working evaluate_gp_model function
    y_pred, pred_lower, pred_upper, output_std = evaluate_gp_model(model, x_test_combined)

    prediction_time = time.time() - pred_start_time
    print(f"Prediction time: {prediction_time:.4f} seconds")

    # Debug: Check prediction statistics
    print(f"\nPrediction Statistics:")
    print(f"  Mean range: [{y_pred.min():.3f}, {y_pred.max():.3f}]")
    print(f"  Std range: [{output_std.min():.3f}, {output_std.max():.3f}]")
    print(f"  Std mean: {output_std.mean():.3f}")
    print(f"  Std std: {output_std.std():.3f}")

    # Debug: Check what noise is actually being applied
    print(f"\nNoise Application Debug:")
    print(f"  Model likelihood noise parameters: {model.likelihood.noise}")

    # Check the actual noise being applied to each source
    source_names = data["metadata"]["source_names"]
    for i, source_name in enumerate(source_names):
        source_mask = source_test[:, i] == 1
        if source_mask.any():
            source_std = output_std[source_mask]
            print(
                f"    {source_name}: std range [{source_std.min():.3f}, {source_std.max():.3f}], mean {source_std.mean():.3f}"
            )

    # Check if uncertainties are reasonable
    if output_std.mean() > 1.0:
        print(f"  WARNING: Standard deviations are very large! Mean std = {output_std.mean():.3f}")
        print(f"  This suggests the model may not be properly trained or has convergence issues.")

    # Check if uncertainties are too small (indicating noise not being applied)
    if output_std.mean() < 0.01:
        print(f"  WARNING: Standard deviations are very small! Mean std = {output_std.mean():.3f}")
        print(f"  This suggests the likelihood noise may not be properly applied.")
        print(f"  Expected noise levels: {noise_levels}")

    # Use the imported metrics functions - evaluate on standardized data like working examples
    y_test_std = (y_test - y_mean) / y_std
    overall_metrics = compute_metrics(y_test_std, y_pred, output_std, start_time=training_start_time)
    per_source_metrics = compute_per_source_metrics(y_test_std, y_pred, output_std, x_test_combined, source_cols)

    # Print results
    print("\n" + "=" * 50)
    print("MODEL PERFORMANCE RESULTS")
    print("=" * 50)

    print(f"\nOVERALL METRICS:")
    for metric_name, value in overall_metrics.items():
        if isinstance(value, float):
            print(f"  {metric_name}: {value:.4f}")
        else:
            print(f"  {metric_name}: {value}")

    print(f"\nPER-SOURCE METRICS:")
    for source_name, source_metrics in per_source_metrics["per_source"].items():
        print(f"\n{source_name.upper()}:")
        for metric_name, value in source_metrics.items():
            if isinstance(value, float):
                print(f"  {metric_name}: {value:.4f}")
            else:
                print(f"  {metric_name}: {value}")

    # Simple plotting
    print("\nGenerating plots...")
    x_test_np = x_test.cpu().numpy()
    y_test_np = y_test.cpu().numpy()
    source_test_np = source_test.cpu().numpy()
    pred_mean_np = y_pred.cpu().numpy()
    pred_std_np = output_std.cpu().numpy()

    source_names = data["metadata"]["source_names"]
    colors = ["red", "blue", "green"]

    fig, axes = plt.subplots(1, len(source_names), figsize=(15, 5))
    if len(source_names) == 1:
        axes = [axes]

    for i, (source_name, color) in enumerate(zip(source_names, colors)):
        # Find indices for this source
        source_mask = source_test_np[:, i] == 1

        if source_mask.any():
            ax = axes[i]

            # Plot true values
            ax.scatter(
                x_test_np[source_mask, 0],
                y_test_np[source_mask],
                c=color,
                alpha=0.6,
                label=f"True ({source_name})",
                s=20,
            )

            # Plot predictions with uncertainty
            ax.scatter(
                x_test_np[source_mask, 0],
                pred_mean_np[source_mask],
                c="black",
                alpha=0.8,
                label=f"Predicted ({source_name})",
                s=30,
                marker="x",
            )

            # Plot uncertainty bands - use 1.96 for 95% CI (not 2)
            ax.fill_between(
                x_test_np[source_mask, 0].flatten(),
                pred_mean_np[source_mask].flatten() - 1.96 * pred_std_np[source_mask].flatten(),
                pred_mean_np[source_mask].flatten() + 1.96 * pred_std_np[source_mask].flatten(),
                alpha=0.2,
                color=color,
                label=f"95% CI ({source_name})",
            )

            # Debug: Print std stats for this source
            source_std = pred_std_np[source_mask]
            print(
                f"    {source_name} std stats: min={source_std.min():.3f}, max={source_std.max():.3f}, mean={source_std.mean():.3f}"
            )

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(f"{source_name.upper()} Fidelity")
            ax.legend()
            ax.grid(True, alpha=0.3)

    # plt.tight_layout()
    # plt.savefig("multifidelity_results.png", dpi=300, bbox_inches='tight')
    # print(f"Plot saved to: multifidelity_results.png")
    # plt.show()

    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
