import gpytorch
import torch

from ..utils.matrix_encoder import MatrixEncoder
from ..utils.one_hot_to_latent_nn import OneHotToLatent
from .gaussian_kernel import GaussianKernel


class CombinedKernel_MVMF(gpytorch.kernels.Kernel):
    def __init__(
        self,
        cont_cols: list = None,
        cat_cols: list = None,
        source_cols: list = None,
        cont_kernel: gpytorch.kernels.Kernel = None,
        cat_kernel: gpytorch.kernels.Kernel = None,
        source_kernel: gpytorch.kernels.Kernel = None,
        cat_encoder=None,
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
            cat_encoder: Encoder function for categorical features (default: OneHotToLatent)
                Can be an encoder object, "matrix", "nn", or a list of encoders [enc1, enc2, enc3]
                that will be matched with categorical groups
            source_encoder: Encoder function for source features (default: OneHotToLatent)
            cat_combination_method: Combination method for hybrid kernels (default: "additive")
            source_combination_method: Combination method for hybrid kernels (default: "additive")
        """
        super().__init__(**kwargs)

        # Initialize with None values first
        if cont_cols is None:
            cont_cols = []
        if cat_cols is None:
            cat_cols = []
        if source_cols is None:
            source_cols = []

        # Create all objects first without assigning to self
        # Continuous kernel
        has_cont = len(cont_cols) > 0
        if has_cont:
            if cont_kernel is None:
                gauss_k = GaussianKernel(ard_num_dims=len(cont_cols))
                final_cont_kernel = gauss_k
            else:
                final_cont_kernel = cont_kernel
        else:
            final_cont_kernel = None

        # Handle categorical variables - support both original single group and new multiple groups
        if cat_cols is not None and len(cat_cols) > 0:
            has_cat = True

            # Check if this is the new multiple group format (list of lists/arrays)
            if len(cat_cols) > 0 and hasattr(cat_cols[0], "__iter__") and not isinstance(cat_cols[0], (str, bytes)):
                # New multiple group format
                multiple_groups = True
                cat_encoders = []

                # Process each categorical group
                for i, cat_group in enumerate(cat_cols):
                    # Convert to list if it's a numpy array or range
                    if hasattr(cat_group, "__iter__") and not isinstance(cat_group, (list, tuple)):
                        cat_group = list(cat_group)
                    elif not isinstance(cat_group, (list, tuple)):
                        cat_group = [cat_group]

                    # Determine encoder type for this group
                    if isinstance(cat_encoder, str):
                        encoder_type = cat_encoder
                        # Create encoder for this group based on string type
                        if encoder_type == "matrix":
                            encoder = MatrixEncoder(len(cat_group), z_dim=2)
                        else:  # nn
                            encoder = OneHotToLatent(len(cat_group))
                    elif isinstance(cat_encoder, list) and len(cat_encoder) == len(cat_cols):
                        # Use the corresponding encoder from the list
                        encoder = cat_encoder[i]
                        # Validate that the encoder has the right input size
                        if hasattr(encoder, "input_size") and encoder.input_size != len(cat_group):
                            raise ValueError(
                                f"Encoder {i} has input_size {encoder.input_size} \
                                but group {i} has {len(cat_group)} columns"
                            )
                        elif hasattr(encoder, "num_classes") and encoder.num_classes != len(cat_group):
                            raise ValueError(
                                f"Encoder {i} has num_classes {encoder.num_classes} \
                                but group {i} has {len(cat_group)} columns"
                            )
                    else:
                        # Use matrix encoder for small groups (≤10), NN for larger ones
                        if len(cat_group) <= 10:
                            encoder_type = "matrix"
                        else:
                            encoder_type = "nn"

                        # Create encoder for this group
                        if encoder_type == "matrix":
                            encoder = MatrixEncoder(len(cat_group), z_dim=2)
                        else:  # nn
                            encoder = OneHotToLatent(len(cat_group))

                    cat_encoders.append(encoder)

                # Create a single shared kernel for all categorical groups
                if cat_kernel is None:
                    # Use the first encoder's z_dim (assuming they're all the same)
                    shared_gauss_k = GaussianKernel(ard_num_dims=cat_encoders[0].z_dim)
                    shared_gauss_k.lengthscale.requires_grad_(False)  # Fix lengthscale
                    shared_gauss_k.lengthscale.data = torch.ones(cat_encoders[0].z_dim) * 0.0  # Fixed lengthscale
                    final_cat_kernel = shared_gauss_k
                else:
                    final_cat_kernel = cat_kernel

                # For multiple groups, final_cat_encoder is the list of individual encoders
                final_cat_encoder = cat_encoders
            else:
                # Original single group format - keep existing behavior
                multiple_groups = False

                # Initialize encoder
                if cat_encoder is None:
                    final_cat_encoder = OneHotToLatent(len(cat_cols))
                elif isinstance(cat_encoder, str):
                    if cat_encoder == "matrix":
                        final_cat_encoder = MatrixEncoder(len(cat_cols), z_dim=2)
                    elif cat_encoder == "nn":
                        final_cat_encoder = OneHotToLatent(len(cat_cols))
                    else:
                        raise ValueError(f"cat_encoder must be 'matrix', 'nn', or an encoder object, got {cat_encoder}")
                else:
                    final_cat_encoder = cat_encoder

                # Initialize kernel
                if cat_kernel is None:
                    gauss_k_cat = GaussianKernel(ard_num_dims=final_cat_encoder.z_dim)
                    gauss_k_cat.lengthscale.requires_grad_(False)  # Fix lengthscale
                    gauss_k_cat.lengthscale.data = torch.ones(final_cat_encoder.z_dim) * 0.0  # Fixed lengthscale
                    final_cat_kernel = gauss_k_cat
                else:
                    final_cat_kernel = cat_kernel
        else:
            has_cat = False
            multiple_groups = False
            final_cat_encoder = cat_encoder
            final_cat_kernel = cat_kernel

        # Only initialize source kernel and encoder if there are multiple sources
        has_source = len(source_cols) > 1
        if has_source:
            # Initialize encoder
            if source_encoder is None:
                final_source_encoder = OneHotToLatent(len(source_cols))
            else:
                final_source_encoder = source_encoder

            # Initialize kernel
            if source_kernel is None:
                gauss_k_source = GaussianKernel(ard_num_dims=final_source_encoder.z_dim)
                gauss_k_source.lengthscale.requires_grad_(False)
                gauss_k_source.lengthscale.data = torch.ones(final_source_encoder.z_dim) * 0.0
                final_source_kernel = gauss_k_source
            else:
                final_source_kernel = source_kernel
        else:
            final_source_kernel = None
            final_source_encoder = None

        # NOW assign ALL self variables at the end in consistent order
        self.cont_cols = cont_cols
        self.cat_cols = cat_cols
        self.source_cols = source_cols
        self.cont_kernel = final_cont_kernel
        self.cat_kernel = final_cat_kernel
        self.source_kernel = final_source_kernel
        self.cat_encoder = final_cat_encoder
        self.source_encoder = final_source_encoder
        self.cat_combination_method = cat_combination_method
        self.source_combination_method = source_combination_method
        self.has_cont = has_cont
        self.has_cat = has_cat
        self.has_source = has_source
        self.multiple_groups = multiple_groups

        # NOW register all modules in consistent order
        self._register_all_modules()

        # Register outputscale parameter
        self.register_parameter("outputscale", torch.nn.Parameter(torch.tensor(1.0)))
        self.register_constraint(param_name="outputscale", constraint=gpytorch.constraints.GreaterThan(1e-6))

    def _register_all_modules(self):
        """
        Register all modules in a consistent order to ensure predictable model structure.
        This method is called at the end of __init__ to guarantee consistent ordering.
        """
        # Always register in the same order: continuous → source → categorical

        # 1. Continuous components
        if self.has_cont and self.cont_kernel is not None:
            self.register_module("cont_kernel", self.cont_kernel)

        # 2. Source components
        if self.has_source and self.source_kernel is not None:
            self.register_module("source_kernel", self.source_kernel)
        if self.has_source and self.source_encoder is not None:
            self.register_module("source_encoder", self.source_encoder)

        # 3. Categorical components
        if self.has_cat and self.cat_kernel is not None:
            self.register_module("cat_kernel", self.cat_kernel)
        if self.has_cat:
            if self.multiple_groups:
                # Register multiple group encoders
                for i, encoder in enumerate(self.cat_encoder):
                    self.register_module(f"cat_encoder_{i}", encoder)
            else:
                # Register single group encoder
                if self.cat_encoder is not None:
                    self.register_module("cat_encoder", self.cat_encoder)

    def forward(self, x1, x2=None, diag=False, **kwargs):
        device = x1.device
        cont_cols_tensor = (
            torch.tensor(self.cont_cols, dtype=torch.long, device=device)
            if (self.cont_cols is not None and len(self.cont_cols) > 0)
            else torch.tensor([], dtype=torch.long, device=device)
        )
        source_cols_tensor = (
            torch.tensor(self.source_cols, dtype=torch.long, device=device)
            if (self.source_cols is not None and len(self.source_cols) > 0)
            else torch.tensor([], dtype=torch.long, device=device)
        )
        n_sources = len(self.source_cols) if (self.source_cols is not None and len(self.source_cols) > 0) else 0

        # Calculate expected dimensions properly
        total_cont = len(self.cont_cols) if (self.cont_cols is not None and len(self.cont_cols) > 0) else 0
        total_cat = (
            sum(len(group) for group in self.cat_cols)
            if self.has_cat and self.multiple_groups
            else (len(self.cat_cols) if self.has_cat else 0)
        )
        total_source = len(self.source_cols) if (self.source_cols is not None and len(self.source_cols) > 0) else 0
        expected_dim = total_cont + total_cat + total_source
        actual_dim = x1.shape[-1]

        if actual_dim != expected_dim:
            raise ValueError(f"Expected input dimension {expected_dim}, got {actual_dim}")

        # Handle x2 for cross-covariance
        if x2 is None:
            x2 = x1
        else:
            if x2.shape[-1] != expected_dim:
                raise ValueError(f"Expected x2 dimension {expected_dim}, got {x2.shape[-1]}")

        # Compute continuous kernel
        if self.has_cont:
            k_cont = self.cont_kernel(
                x1.index_select(-1, cont_cols_tensor), x2.index_select(-1, cont_cols_tensor), diag=diag, **kwargs
            )
            result = k_cont
        else:
            result = torch.ones(x1.shape[0], x2.shape[0], device=device)

        if self.has_cat:
            if self.multiple_groups:
                # Multiple groups - compute kernel for each group using shared kernel
                for i, (encoder, col_group) in enumerate(zip(self.cat_encoder, self.cat_cols)):
                    # Extract columns for this group
                    col_tensor = torch.tensor(col_group, dtype=torch.long, device=device)
                    x1_group = x1.index_select(-1, col_tensor)
                    x2_group = x2.index_select(-1, col_tensor)

                    # Encode and compute kernel for this group using shared kernel
                    z1_c = encoder(x1_group)
                    z2_c = encoder(x2_group)
                    k_cat_group = self.cat_kernel(z1_c, z2_c, diag=diag, **kwargs)

                    # Combine with existing result
                    if i == 0:
                        result_cat = k_cat_group
                    else:
                        # Use multiplication for combining kernels
                        result_cat = result_cat.mul(k_cat_group)

                # Combine with existing result
                if self.has_cont:
                    result = result.mul(result_cat)
                else:
                    result = result_cat
            else:
                # Original single group behavior
                cat_cols_tensor = torch.tensor(self.cat_cols, dtype=torch.long, device=device)
                z1_c = self.cat_encoder(x1.index_select(-1, cat_cols_tensor))
                z2_c = self.cat_encoder(x2.index_select(-1, cat_cols_tensor))
                k_cat = self.cat_kernel(z1_c, z2_c, diag=diag, **kwargs)
                if self.has_cont:
                    result = result.mul(k_cat)
                else:
                    result = k_cat

        if self.has_source:
            use_eps = isinstance(self.source_encoder, OneHotToLatent) and getattr(
                self.source_encoder, "is_probabilistic", True
            )
            if use_eps:
                epsilon = torch.normal(mean=0, std=1, size=[n_sources, 2], device=x1.device, dtype=x1.dtype)
                z1_s = self.source_encoder(x1.index_select(-1, source_cols_tensor), epsilon=epsilon)
                z2_s = self.source_encoder(x2.index_select(-1, source_cols_tensor), epsilon=epsilon)
                k_source = self.source_kernel(z1_s, z2_s, diag=diag, **kwargs)
                result = result.mul(k_source)
            else:
                z1_s = self.source_encoder(x1.index_select(-1, source_cols_tensor))
                z2_s = self.source_encoder(x2.index_select(-1, source_cols_tensor))
                k_source = self.source_kernel(z1_s, z2_s, diag=diag, **kwargs)
                result = result.mul(k_source)

        return result.mul(self.outputscale)
