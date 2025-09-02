import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class LinearKernel(gpytorch.kernels.Kernel):
    """
    Linear kernel that preserves linear relationships in the latent space.
    This is particularly useful for matrix encoders to maintain interpretability.

    k(x, x') = x^T x' + bias
    """

    def __init__(self, bias=1.0, **kwargs):
        super().__init__(**kwargs)
        self.bias = bias

    def forward(self, x1, x2=None, diag=False, **kwargs):
        if x2 is None:
            x2 = x1

        if diag:
            # For diagonal elements, compute x^T x + bias
            return torch.sum(x1 * x1, dim=-1) + self.bias
        else:
            # For off-diagonal elements, compute x1^T x2 + bias
            return torch.matmul(x1, x2.transpose(-2, -1)) + self.bias

    def evaluate(self, x1, x2=None, diag=False, **kwargs):
        """Evaluate the kernel and return actual tensor values."""
        return self.forward(x1, x2, diag, **kwargs)


class SimpleLinearKernel:
    """
    Simple linear kernel that directly returns tensor values without lazy evaluation.
    This is easier to work with for testing and analysis.

    k(x, x') = x^T x' + bias
    """

    def __init__(self, bias=1.0):
        self.bias = bias

    def __call__(self, x1, x2=None, diag=False):
        if x2 is None:
            x2 = x1

        if diag:
            # For diagonal elements, compute x^T x + bias
            return torch.sum(x1 * x1, dim=-1) + self.bias
        else:
            # For off-diagonal elements, compute x1^T x2 + bias
            return torch.matmul(x1, x2.transpose(-2, -1)) + self.bias


class MatrixEncoder(nn.Module):
    """
    Matrix-based encoder for categorical variables as described in Section 4 of the GP+ paper.

    This encoder uses learnable matrices to map one-hot encoded categorical variables
    to latent representations, providing a more interpretable and computationally
    efficient alternative to neural network encoders.

    Args:
        input_dim: Dimension of input (one-hot vectors)
        z_dim: Dimension of latent space (default=2)
        initialization: Initialization method for the matrix ('normal', 'uniform', 'orthogonal')
        init_std: Standard deviation for normal initialization (default=0.1)
    """

    def __init__(self, input_dim, z_dim=2, initialization="normal", init_std=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim

        # Create learnable projection matrix
        self.projection_matrix = nn.Parameter(torch.empty(input_dim, z_dim))

        # Initialize the matrix
        if initialization == "normal":
            nn.init.normal_(self.projection_matrix, mean=0.0, std=init_std)
        elif initialization == "uniform":
            nn.init.uniform_(self.projection_matrix, -init_std, init_std)
        elif initialization == "orthogonal":
            nn.init.orthogonal_(self.projection_matrix, gain=init_std)
        else:
            raise ValueError(f"Unknown initialization method: {initialization}")

    def forward(self, x_onehot):
        """
        Forward pass: project one-hot encoded inputs to latent space.

        Args:
            x_onehot: One-hot encoded input tensor of shape [batch_size, input_dim]

        Returns:
            Latent representations of shape [batch_size, z_dim]
        """
        # Add dimension checks
        if x_onehot.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {x_onehot.shape[-1]}")

        # Matrix multiplication: x_onehot @ projection_matrix
        # This is equivalent to: torch.mm(x_onehot, self.projection_matrix)
        return torch.matmul(x_onehot.float(), self.projection_matrix)

    def get_projection_matrix(self):
        """Return the current projection matrix for analysis."""
        return self.projection_matrix.data.clone()

    def set_projection_matrix(self, matrix):
        """Set the projection matrix to specific values."""
        if matrix.shape != (self.input_dim, self.z_dim):
            raise ValueError(f"Expected matrix shape {(self.input_dim, self.z_dim)}, got {matrix.shape}")
        self.projection_matrix.data = matrix.clone()


class SourceMatrixEncoder(nn.Module):
    """
    Specialized matrix encoder for source/fidelity variables.

    This encoder is designed specifically for multi-fidelity modeling where
    different sources have different levels of fidelity. The matrix learns
    the relationships between different fidelity levels.

    Args:
        num_sources: Number of different sources/fidelity levels
        z_dim: Dimension of latent space (default=2)
        initialization: Initialization method for the matrix
        init_std: Standard deviation for normal initialization (default=0.1)
    """

    def __init__(self, num_sources, z_dim=2, initialization="normal", init_std=0.1):
        super().__init__()
        self.num_sources = num_sources
        self.z_dim = z_dim

        # Create learnable projection matrix for sources
        self.projection_matrix = nn.Parameter(torch.empty(num_sources, z_dim))

        # Initialize the matrix
        if initialization == "normal":
            nn.init.normal_(self.projection_matrix, mean=0.0, std=init_std)
        elif initialization == "uniform":
            nn.init.uniform_(self.projection_matrix, -init_std, init_std)
        elif initialization == "orthogonal":
            nn.init.orthogonal_(self.projection_matrix, gain=init_std)
        else:
            raise ValueError(f"Unknown initialization method: {initialization}")

    def forward(self, x_source):
        """
        Forward pass: project source one-hot encoded inputs to latent space.

        Args:
            x_source: One-hot encoded source tensor of shape [batch_size, num_sources]

        Returns:
            Latent representations of shape [batch_size, z_dim]
        """
        # Add dimension checks
        if x_source.shape[-1] != self.num_sources:
            raise ValueError(f"Expected input dimension {self.num_sources}, got {x_source.shape[-1]}")

        # Matrix multiplication: x_source @ projection_matrix
        return torch.matmul(x_source.float(), self.projection_matrix)

    def get_projection_matrix(self):
        """Return the current projection matrix for analysis."""
        return self.projection_matrix.data.clone()

    def set_projection_matrix(self, matrix):
        """Set the projection matrix to specific values."""
        if matrix.shape != (self.num_sources, self.z_dim):
            raise ValueError(f"Expected matrix shape {(self.num_sources, self.z_dim)}, got {matrix.shape}")
        self.projection_matrix.data = matrix.clone()


def analyze_projection_matrix(encoder, save_path=None, title="Projection Matrix Analysis"):
    """
    Analyze and visualize the learned projection matrix.

    Args:
        encoder: MatrixEncoder or SourceMatrixEncoder instance
        save_path: Path to save the visualization (optional)
        title: Title for the plot
    """
    matrix = encoder.get_projection_matrix().cpu().numpy()

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=16)

    # 1. Heatmap of the projection matrix
    im1 = axes[0, 0].imshow(matrix, cmap="RdBu_r", aspect="auto")
    axes[0, 0].set_title("Projection Matrix Heatmap")
    axes[0, 0].set_xlabel("Latent Dimensions")
    axes[0, 0].set_ylabel("Input Dimensions")
    plt.colorbar(im1, ax=axes[0, 0])

    # 2. Bar plot of matrix norms
    matrix_norms = np.linalg.norm(matrix, axis=1)
    axes[0, 1].bar(range(len(matrix_norms)), matrix_norms)
    axes[0, 1].set_title("Row-wise Matrix Norms")
    axes[0, 1].set_xlabel("Input Dimensions")
    axes[0, 1].set_ylabel("L2 Norm")

    # 3. Scatter plot of latent space
    if matrix.shape[1] >= 2:
        axes[1, 0].scatter(matrix[:, 0], matrix[:, 1], alpha=0.7)
        axes[1, 0].set_title("Latent Space Projection")
        axes[1, 0].set_xlabel("Latent Dimension 1")
        axes[1, 0].set_ylabel("Latent Dimension 2")

        # Add labels for each point
        for i, (x, y) in enumerate(zip(matrix[:, 0], matrix[:, 1])):
            axes[1, 0].annotate(f"{i}", (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)

    # 4. Statistical summary
    axes[1, 1].text(0.1, 0.9, f"Matrix Shape: {matrix.shape}", transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.8, f"Mean: {matrix.mean():.4f}", transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.7, f"Std: {matrix.std():.4f}", transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.6, f"Min: {matrix.min():.4f}", transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.5, f"Max: {matrix.max():.4f}", transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.4, f"Condition Number: {np.linalg.cond(matrix):.2f}", transform=axes[1, 1].transAxes)
    axes[1, 1].set_title("Matrix Statistics")
    axes[1, 1].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Matrix analysis saved to: {save_path}")

    plt.close()  # Close the figure instead of showing it

    return matrix


def compare_encoders(neural_encoder, matrix_encoder, test_input, save_path=None):
    """
    Compare the outputs of neural network and matrix encoders.

    Args:
        neural_encoder: OneHotToLatent encoder instance
        matrix_encoder: MatrixEncoder instance
        test_input: Test input tensor
        save_path: Path to save the comparison plot (optional)
    """
    with torch.no_grad():
        neural_output = neural_encoder(test_input)
        matrix_output = matrix_encoder(test_input)

    # Convert to numpy for plotting
    neural_np = neural_output.cpu().numpy()
    matrix_np = matrix_output.cpu().numpy()

    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot neural network output
    if neural_np.shape[1] >= 2:
        axes[0].scatter(neural_np[:, 0], neural_np[:, 1], alpha=0.7, c="blue")
        axes[0].set_title("Neural Network Encoder Output")
        axes[0].set_xlabel("Latent Dimension 1")
        axes[0].set_ylabel("Latent Dimension 2")

    # Plot matrix encoder output
    if matrix_np.shape[1] >= 2:
        axes[1].scatter(matrix_np[:, 0], matrix_np[:, 1], alpha=0.7, c="red")
        axes[1].set_title("Matrix Encoder Output")
        axes[1].set_xlabel("Latent Dimension 1")
        axes[1].set_ylabel("Latent Dimension 2")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Encoder comparison saved to: {save_path}")

    plt.close()  # Close the figure instead of showing it

    # Print statistics
    print(f"Neural encoder output - Mean: {neural_np.mean():.4f}, Std: {neural_np.std():.4f}")
    print(f"Matrix encoder output - Mean: {matrix_np.mean():.4f}, Std: {matrix_np.std():.4f}")

    return neural_np, matrix_np


def analyze_linear_relationships(encoder, test_inputs, save_path=None):
    """
    Analyze linear relationships in the matrix encoder outputs.

    Args:
        encoder: MatrixEncoder or SourceMatrixEncoder instance
        test_inputs: Test input tensor (one-hot encoded)
        save_path: Path to save the analysis plot (optional)
    """
    # Ensure test_inputs are on the same device as the encoder
    device = next(encoder.parameters()).device
    test_inputs = test_inputs.to(device)

    with torch.no_grad():
        outputs = encoder(test_inputs)

    outputs_np = outputs.cpu().numpy()
    matrix = encoder.get_projection_matrix().cpu().numpy()
    test_inputs_np = test_inputs.cpu().numpy()

    # Create analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Linear Relationship Analysis", fontsize=16)

    # 1. Input vs Output correlation
    for i in range(min(3, outputs_np.shape[1])):
        for j in range(min(3, test_inputs_np.shape[1])):
            correlation = np.corrcoef(test_inputs_np[:, j], outputs_np[:, i])[0, 1]
            axes[0, 0].scatter(
                test_inputs_np[:, j], outputs_np[:, i], alpha=0.6, label=f"Input{j}->Output{i} (r={correlation:.3f})"
            )

    axes[0, 0].set_xlabel("Input Values")
    axes[0, 0].set_ylabel("Output Values")
    axes[0, 0].set_title("Input-Output Correlations")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Linear regression analysis
    from sklearn.linear_model import LinearRegression

    lr = LinearRegression()
    lr.fit(test_inputs_np, outputs_np)
    predicted = lr.predict(test_inputs_np)

    axes[0, 1].scatter(outputs_np.flatten(), predicted.flatten(), alpha=0.6)
    axes[0, 1].plot([outputs_np.min(), outputs_np.max()], [outputs_np.min(), outputs_np.max()], "r--", lw=2)
    axes[0, 1].set_xlabel("Actual Outputs")
    axes[0, 1].set_ylabel("Linear Regression Predictions")
    axes[0, 1].set_title("Linear Regression Analysis")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Matrix rank analysis
    u, s, v = np.linalg.svd(matrix)
    axes[1, 0].plot(range(1, len(s) + 1), s, "bo-")
    axes[1, 0].set_xlabel("Singular Value Index")
    axes[1, 0].set_ylabel("Singular Value")
    axes[1, 0].set_title("Singular Value Decomposition")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale("log")

    # 4. Linear transformation visualization
    if matrix.shape[1] >= 2:
        # Create unit vectors in input space
        unit_vectors = torch.eye(matrix.shape[0], device=device)
        with torch.no_grad():
            transformed = encoder(unit_vectors)

        transformed_np = transformed.cpu().numpy()
        axes[1, 1].scatter(transformed_np[:, 0], transformed_np[:, 1], s=100, alpha=0.7)

        # Add arrows from origin to show transformation
        for i, (x, y) in enumerate(transformed_np):
            axes[1, 1].arrow(0, 0, x, y, head_width=0.05, head_length=0.05, fc="red", ec="red", alpha=0.7)
            axes[1, 1].annotate(f"Input{i}", (x, y), xytext=(5, 5), textcoords="offset points")

        axes[1, 1].set_xlabel("Latent Dimension 1")
        axes[1, 1].set_ylabel("Latent Dimension 2")
        axes[1, 1].set_title("Linear Transformation of Unit Vectors")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_aspect("equal")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Linear relationship analysis saved to: {save_path}")

    plt.close()  # Close the figure instead of showing it

    # Print analysis results
    print(f"Matrix rank: {np.linalg.matrix_rank(matrix)}")
    print(f"Condition number: {np.linalg.cond(matrix):.2f}")
    print(f"Linear regression R² score: {lr.score(test_inputs_np, outputs_np):.4f}")

    return matrix, outputs_np
