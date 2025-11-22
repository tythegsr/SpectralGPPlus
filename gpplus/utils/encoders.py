import math

import gpytorch
import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def __init__(self, input_dim, z_dim=2, initialization="normal", init_std=0.1, seed=None):
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim

        if seed is None:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()

        self.seed = seed
        # Create learnable projection matrix
        self.projection_matrix = nn.Parameter(torch.empty(input_dim, z_dim))

        # Store initialization type and parameters in the module for the parameter initializer
        self._param_init_types = {"projection_matrix": initialization}
        self._param_init_params = {"projection_matrix": {"init_std": init_std}}

        # Initialize the matrix
        generator = torch.Generator().manual_seed(self.seed)
        if initialization == "normal":
            nn.init.normal_(self.projection_matrix, mean=0.0, std=init_std, generator=generator)
        elif initialization == "uniform":
            nn.init.uniform_(self.projection_matrix, -init_std, init_std, generator=generator)
        elif initialization == "orthogonal":
            nn.init.orthogonal_(self.projection_matrix, gain=init_std, generator=generator)
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
        return torch.matmul(x_onehot, self.projection_matrix)

    def get_projection_matrix(self):
        """Return the current projection matrix for analysis."""
        return self.projection_matrix.data.clone()

    def set_projection_matrix(self, matrix):
        """Set the projection matrix to specific values."""
        if matrix.shape != (self.input_dim, self.z_dim):
            raise ValueError(f"Expected matrix shape {(self.input_dim, self.z_dim)}, got {matrix.shape}")
        self.projection_matrix.data = matrix.clone()


class NeuralEncoder(gpytorch.Module):
    def __init__(
        self, input_dim, architecture_config=None, z_dim=2, num_passes=1, num_passes_pred=None, probabilistic=False
    ):
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.num_passes = num_passes
        self.is_probabilistic = probabilistic

        """
        OneHot: A module for encoding categorical (one-hot) features into a latent space, 
        with support for both deterministic and probabilistic encoders.
        
        This class transforms one-hot encoded inputs into a continuous latent representation. 
        It supports two modes:
            - Deterministic: Outputs a single latent vector per input.
            - Probabilistic: Outputs parameters for a Gaussian distribution and samples 
              multiple latent representations (Monte Carlo approach).
        
        Args:
            input_dim (int): Dimensionality of input (number of one-hot categories).
            architecture_config (dict, optional): Dictionary specifying encoder architecture:
                {
                    'hidden_dims': list[int],  # Hidden layer sizes (empty list for no hidden layers)
                    'activation': str,          # Activation function ('relu', 'tanh', etc.)
                    'dropout': float            # Dropout probability (default: None)
                }
                Defaults to {'hidden_dims': [5], 'activation': 'relu', 'dropout': None}.
            z_dim (int): Dimensionality of the latent space. Default = 2.
            num_passes (int): Number of forward passes during training (probabilistic case).
            num_passes_pred (int, optional): Number of forward passes during prediction.
                If None, defaults to num_passes.
            probabilistic (bool): Whether to use probabilistic encoder. Default = True.
        
        Attributes:
            fci (nn.Module): First linear mapping or initial VAE layer.
            fce (nn.Module): Final VAE layer (only for probabilistic mode with hidden layers).
            hidden_layers (list): Names of intermediate hidden VAE layers (probabilistic mode).
            is_probabilistic (bool): Indicates whether the encoder is probabilistic.
        
        Forward Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim].
            epsilon (torch.Tensor, optional): Noise tensor for sampling in probabilistic mode.
                If None, sampled internally.
            visualize (bool): If True, returns full latent distribution parameters.
                Otherwise, returns compressed latent projection.
        
        Returns:
            torch.Tensor: 
                - If probabilistic: Sampled latent representation or full distribution parameters.
                - If deterministic: Latent embedding of input.
        """
        if self.z_dim != 2:
            raise NotImplementedError(
                "Current class only supports 2D embeddings. Higher D still needs to be implemented!"
            )
        # self.is_probabilistic = probabilistic
        if (num_passes_pred is None) or (num_passes == 1):
            self.num_passes_pred = self.num_passes
        else:
            self.num_passes_pred = self.num_passes

        if architecture_config is None:
            architecture_config = {"hidden_dims": [], "activation": "tanh", "dropout": None}

        activation_map = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU(),
            "hardtanh": nn.Hardtanh(),
        }
        activation = activation_map.get(architecture_config["activation"].lower(), nn.ReLU())

        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in architecture_config["hidden_dims"]:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(activation)
            if architecture_config.get("dropout"):
                encoder_layers.append(nn.Dropout(architecture_config["dropout"]))
            prev_dim = hidden_dim

        if self.is_probabilistic:  # Multiple forward passes, Probabilistic version
            # VAE output dimension (5 parameters: mu1, mu2, L21, L11, L22)
            vae_output_dim = 5

            # First layer: input to VAE latent space
            self.fci = Linear_VAE(in_features=input_dim, out_features=vae_output_dim, bias=True, name="fci")

            # Store hidden layer configuration
            self.hidden_layers = architecture_config["hidden_dims"]

            # Create hidden layers if any exist
            if self.hidden_layers:
                prev_dim = vae_output_dim  # Start from fci output

                for i, hidden_dim in enumerate(self.hidden_layers):
                    name = f"h{i + 1}"  # h1, h2, h3, etc.
                    layer = Linear_VAE(prev_dim, hidden_dim, name=name)
                    setattr(self, name, layer)
                    prev_dim = hidden_dim

                # Final layer: hidden layers to VAE output
                self.fce = Linear_VAE(in_features=prev_dim, out_features=vae_output_dim, bias=True, name="fce")
            else:
                # No hidden layers: fci directly outputs VAE parameters
                self.fce = None

        else:  # Determinisitic version (n_passes = 1)
            inner_layers = []
            prev_dim = input_dim

            for hidden_dim in architecture_config["hidden_dims"]:
                inner_layers.append(nn.Linear(prev_dim, hidden_dim))
                inner_layers.append(activation)
                if architecture_config.get("dropout"):
                    inner_layers.append(nn.Dropout(architecture_config["dropout"]))
                prev_dim = hidden_dim

            parameter_head = nn.Linear(prev_dim, self.z_dim)
            self.fci = nn.Sequential(*inner_layers, parameter_head)

    # def forward(self, x_onehot, num_passes=1, epsilon=None):
    def forward(self, x, epsilon=None, visualize=False):
        if self.is_probabilistic:
            if x.dim() > 1:
                xtemp, inverse_indices = torch.unique(x, dim=0, return_inverse=True)  # shape [S, onehot_dim]
            else:
                xtemp = x
            if self.hidden_layers:
                xtemp = self.fci(xtemp)
                for i in range(len(self.hidden_layers)):
                    xtemp = F.relu(getattr(self, f"h{i + 1}")(xtemp))
                out = self.fce(xtemp)
            else:
                # No hidden layers: fci directly outputs VAE parameters
                out = self.fci(xtemp)

            if epsilon is None:
                epsilon = torch.normal(mean=0, std=1, size=[self.input_dim, 2], device=x.device, dtype=x.dtype)

            if x.dim() > 1:
                Mu_1 = out[:, 0:1]
                Mu_2 = out[:, 1:2]
                L21 = out[:, 2:3]
                L11 = torch.abs(out[:, 3:4])
                L22 = torch.abs(out[:, 4:5])
                epsilon_1 = epsilon[:, 0:1]
                epsilon_2 = epsilon[:, 1:2]

                z1 = Mu_1 + L11 * epsilon_1
                z2 = Mu_2 + L21 * epsilon_1 + L22 * epsilon_2
                z = torch.cat([z1, z2], dim=1)
                if visualize is True:
                    return z
                else:
                    z_projected = x[:, :] @ z  # [400, 4] @ [4, 2] → [400, 2]
                    return z_projected

        else:
            return self.fci(x)


class Linear_VAE(gpytorch.Module):
    def __init__(self, in_features, out_features, bias=True, name=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.name = str(name) if name is not None else ""

        """
        Linear_VAE: A linear layer with learnable weight and bias parameters 
        that supports prior distributions for Bayesian inference.
        
        This module behaves like a fully connected (linear) layer but integrates 
        Bayesian priors (Normal) on weights and optionally on biases, making it 
        suitable for probabilistic encoders like VAEs.
        
        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            bias (bool, optional): If True, adds a learnable bias to the output.
                Default = True.
            name (str, optional): Name prefix for parameter priors. Used to register
                priors in GPyTorch. Default = None.
        
        Attributes:
            weight (torch.nn.Parameter): Learnable weight matrix of shape 
                [out_features, in_features], initialized with Kaiming uniform.
            bias (torch.nn.Parameter or None): Learnable bias vector of shape 
                [out_features], initialized with uniform distribution.
            priors: 
                - Weight prior: Normal(0, 0.2)
                - Bias prior (if enabled): Normal(0, 0.05)
        
        Methods:
            reset_parameters():
                Initializes weights with Kaiming uniform initialization and bias 
                uniformly within calculated bounds.
            forward(input: torch.Tensor) -> torch.Tensor:
                Computes the linear transformation of the input: input @ weight^T + bias.
            extra_repr() -> str:
                Returns a string representation with in_features and out_features.
        
        Example:
            >>> layer = Linear_VAE(in_features=4, out_features=5, bias=True, name="fci")
            >>> x = torch.randn(10, 4)
            >>> output = layer(x)
            >>> print(output.shape)  # torch.Size([10, 5])
        """

        # Weight
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.register_prior(f"{self.name}_prior_weight", gpytorch.priors.NormalPrior(0.0, 0.2), lambda m: m.weight)

        # Bias
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            self.register_prior(f"{self.name}_prior_bias", gpytorch.priors.NormalPrior(0.0, 0.05), lambda m: m.bias)
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        input = input.to(self.weight.dtype)
        return nn.functional.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}"
