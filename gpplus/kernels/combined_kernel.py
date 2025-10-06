import gpytorch
import torch

from ..utils.encoders import MatrixEncoder, NeuralEncoder
from .gaussian_kernel import GaussianKernel


class CombinedKernel(gpytorch.kernels.Kernel):
    def __init__(
        self,
        cont_cols: list = None,
        cat_cols: list = None,
        source_cols: list = None,
        cont_kernel: gpytorch.kernels.Kernel = None,
        cat_kernel: gpytorch.kernels.Kernel = None,
        source_kernel: gpytorch.kernels.Kernel = None,
        cat_encoder=None,  # Accepts: "matrix", "nn", or a list of encoders [enc1, enc2, enc3]
        source_encoder=None,
        cat_combination_method: str = "additive",
        source_combination_method: str = "additive",
        source_z_dim: int = 2,
        cat_z_dim: int = 2,
        **kwargs,
    ):
        """
        Multi-Variable Multi-Fidelity Combined Kernel V2

        Args:
            cont_cols: Indices of continuous features
            cat_cols: List of categorical feature groups. Each group can be:
                - A list of indices (e.g., [0, 1, 2])
                - A range (e.g., np.arange(0, 10))
                - A single index (e.g., [5])
            source_cols: Indices of source/fidelity features
            cont_kernel: Kernel for continuous features (default: scaled RBF)
            cat_kernel: Kernel for categorical features (default: RBF for NN, Linear for Matrix)
            source_kernel: Kernel for source features (default: RBF for NN, Linear for Matrix)
            cat_encoder: Encoder function for categorical features (default: NeuralEncoder)
                Can be an encoder object, "matrix", "nn", or a list of encoders [enc1, enc2, enc3]
                that will be matched with categorical groups
            source_encoder: Encoder function for source features (default: NeuralEncoder)
            cat_combination_method: Combination method for hybrid kernels (default: "additive")
            source_combination_method: Combination method for hybrid kernels (default: "additive")
        """
        super().__init__(**kwargs)

        if cont_cols is None:
            cont_cols = []
        if cat_cols is None:
            cat_cols = []
        if source_cols is None:
            source_cols = []

        self.cont_cols = cont_cols
        self.cat_cols = cat_cols
        self.source_cols = source_cols
        self.cat_combination_method = cat_combination_method
        self.source_combination_method = source_combination_method
        self.has_cont = len(cont_cols) > 0
        self.has_cat = len(cat_cols) > 0
        self.has_source = len(source_cols) > 0

        # Continuous kernel
        if self.has_cont:
            self._process_cont(cont_kernel)
        else:
            self.cont_kernel = None

        # Categorical kernel
        if self.has_cat:
            self._process_cat(cat_encoder, cat_kernel)
        else:
            self.cat_kernel = None
            self.cat_encoder = None

        # Source kernel
        if self.has_source:
            self._process_source(source_encoder, source_kernel)
        else:
            self.source_kernel = None
            self.source_encoder = None

        # Register outputscale parameter
        self.register_parameter("outputscale", torch.nn.Parameter(torch.tensor(1.0)))
        self.register_constraint(param_name="outputscale", constraint=gpytorch.constraints.GreaterThan(1e-6))

    def forward(self, x1, x2=None, diag=False, **kwargs):
        device = x1.device
        n_sources = len(self.source_cols)

        # Handle x2 for cross-covariance
        if x2 is None:
            x2 = x1

        # Compute continuous kernel
        if self.has_cont:
            k_cont = self.cont_kernel(
                x1.index_select(-1, torch.tensor(self.cont_cols)),
                x2.index_select(-1, torch.tensor(self.cont_cols)),
                diag=diag,
                **kwargs,
            )
            result = k_cont
        else:
            result = torch.ones(x1.shape[0], x2.shape[0], device=device)

        # Compute categorical kernel
        if self.has_cat:
            # Handle both single encoder and list of encoders
            if not hasattr(self.cat_encoder, "__iter__"):
                encoders = [self.cat_encoder]
            else:
                encoders = self.cat_encoder

            for i, (encoder, col_group) in enumerate(zip(encoders, self.cat_cols)):
                x1_group = x1.index_select(-1, torch.tensor(col_group))
                x2_group = x2.index_select(-1, torch.tensor(col_group))

                z1_c = encoder(x1_group)
                z2_c = encoder(x2_group)
                k_cat_group = self.cat_kernel(z1_c, z2_c, diag=diag, **kwargs)

                if i == 0:
                    result_cat = k_cat_group
                else:
                    result_cat = result_cat.mul(k_cat_group)

            if self.has_cont:
                result = result.mul(result_cat)
            else:
                result = result_cat

        # Compute source kernel
        if self.has_source:
            use_eps = isinstance(self.source_encoder, NeuralEncoder) and getattr(
                self.source_encoder, "is_probabilistic", True
            )
            if use_eps:
                epsilon = torch.normal(mean=0, std=1, size=[n_sources, 2], device=x1.device, dtype=x1.dtype)
                z1_s = self.source_encoder(x1.index_select(-1, torch.tensor(self.source_cols)), epsilon=epsilon)
                z2_s = self.source_encoder(x2.index_select(-1, torch.tensor(self.source_cols)), epsilon=epsilon)
                k_source = self.source_kernel(z1_s, z2_s, diag=diag, **kwargs)
                result = result.mul(k_source)
            else:
                z1_s = self.source_encoder(x1.index_select(-1, torch.tensor(self.source_cols)))
                z2_s = self.source_encoder(x2.index_select(-1, torch.tensor(self.source_cols)))
                k_source = self.source_kernel(z1_s, z2_s, diag=diag, **kwargs)
                result = result.mul(k_source)

        # Multiply final result by outputscale
        return result.mul(self.outputscale)

    def _process_cont(self, cont_kernel):
        """Process continuous kernel."""
        if cont_kernel is None:  # Default cont_kernel
            gauss_k = GaussianKernel(ard_num_dims=len(self.cont_cols))
            self.cont_kernel = gauss_k
        else:  # User-provided cont_kernel
            self.cont_kernel = cont_kernel

    def _process_cat(self, cat_encoder, cat_kernel):
        """Process categorical encoders and kernel."""
        temp_cat_encoder = []
        # Wraps single list into a list so loop works
        if len(self.cat_cols) > 0 and not hasattr(self.cat_cols[0], "__iter__"):
            self.cat_cols = [self.cat_cols]
        for i, cat_group in enumerate(self.cat_cols):
            if isinstance(cat_encoder, str):
                if cat_encoder == "nn":
                    encoder = NeuralEncoder(len(cat_group))
                elif cat_encoder == "matrix":  # Default cat_encoder
                    encoder = MatrixEncoder(len(cat_group), z_dim=2)
                else:
                    raise ValueError("Only string inputs for cat_encoder are 'nn' and 'matrix'")
            elif cat_encoder is not None:
                if cat_encoder[i].input_dim == len(self.cat_cols[i]):
                    # Use the corresponding encoder from the list
                    encoder = cat_encoder[i]
                    # Validate that the encoder has the right input size
                    if hasattr(encoder, "input_size") and encoder.input_size != len(cat_group):
                        raise ValueError(
                            f"Encoder {i} has input_size {encoder.input_size} but \
                            group {i} has {len(cat_group)} columns"
                        )
                else:
                    raise ValueError("Numbers of encoders provided does not match number of categorical groups")
            else:
                encoder = MatrixEncoder(len(cat_group), z_dim=2)
            temp_cat_encoder.append(encoder)
        # Set cat_kernel
        if cat_kernel is None:
            shared_gauss_k = GaussianKernel(ard_num_dims=temp_cat_encoder[0].z_dim)
            shared_gauss_k.lengthscale.requires_grad_(False)
            shared_gauss_k.lengthscale.data = torch.ones(temp_cat_encoder[0].z_dim) * 0.0
            self.cat_kernel = shared_gauss_k
        else:
            self.cat_kernel = cat_kernel
        # Set cat_encoder
        if len(temp_cat_encoder) == 1:
            self.cat_encoder = temp_cat_encoder[0]
        else:
            self.cat_encoder = temp_cat_encoder
            for i, encoder in enumerate(self.cat_encoder):
                self.register_module(f"cat_encoder_{i}", encoder)

    def _process_source(self, source_encoder, source_kernel):
        """Process source encoder and kernel."""
        # Initialize encoder
        if isinstance(source_encoder, str):
            if source_encoder == "nn":
                encoder = NeuralEncoder(len(self.source_cols))
            elif source_encoder == "matrix":  # Default source_encoder
                encoder = MatrixEncoder(len(self.source_cols), z_dim=2)
            else:
                raise ValueError("Only string inputs for source_encoder are 'nn' and 'matrix'")
        elif source_encoder is not None:
            if source_encoder.input_dim == len(self.source_cols):
                # Use the corresponding encoder from the list
                encoder = source_encoder
                # Validate that the encoder has the right input size
                if hasattr(encoder, "input_size") and encoder.input_size != len(self.source_cols):
                    raise ValueError(
                        f"Encoder has input_size {encoder.input_size} but \
                        source has {len(self.source_cols)} columns"
                    )
            else:
                raise ValueError("Numbers of encoders provided does not match number of source groups")
        else:
            encoder = MatrixEncoder(len(self.source_cols), z_dim=2)

        # Set source_kernel
        if source_kernel is None:
            gauss_k_source = GaussianKernel(ard_num_dims=encoder.z_dim)
            gauss_k_source.lengthscale.requires_grad_(False)
            gauss_k_source.lengthscale.data = torch.ones(encoder.z_dim) * 0.0
            self.source_kernel = gauss_k_source
        else:
            self.source_kernel = source_kernel

        # Set source_encoder
        self.source_encoder = encoder
