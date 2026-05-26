import gpytorch
import torch

from ..utils.encoders import Encoder
from .gaussian_kernel import GaussianKernel


class MVMFKernel(gpytorch.kernels.Kernel):
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
        z_dim=2,
        fix_lengthscale_cat=False,
        fix_lengthscale_source=False,
        **kwargs,
    ):
        """
        Multi-Variable Multi-Fidelity Combined Kernel

        Args:
            cont_cols: Indices of continuous features
            cat_cols: List of categorical feature groups. Each group can be:
                - A list of indices (e.g., [0, 1, 2])
                - A range (e.g., np.arange(0, 10))
                - A single index (e.g., [5])
            source_cols: Indices of source/fidelity features
            cont_kernel: Kernel for continuous features (default: scaled RBF)
            cat_kernel: Kernel for categorical features. Single kernel used for all groups.
            source_kernel: Kernel for source features.
            cat_encoder: Encoder or list[Encoder] for categorical features.
                If there are grouped categorical columns, pass one encoder per group.
            source_encoder: Encoder for source features.
            z_dim: Dimension of the latent space for the encoders (default: 2)
            fix_lengthscale_cat: Whether to fix the lengthscale of the categorical kernel (default: False)
            fix_lengthscale_source: Whether to fix the lengthscale of the source kernel (default: False)
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
        self.z_dim = z_dim
        self.fix_lengthscale_cat = fix_lengthscale_cat
        self.fix_lengthscale_source = fix_lengthscale_source

        self._process_cont(cont_kernel)
        self._process_cat(cat_encoder, cat_kernel)
        self._process_source(source_encoder, source_kernel)

    def forward(self, x1, x2=None, diag=False, last_dim_is_batch=False, **kwargs):
        device = x1.device

        # Handle x2 for cross-covariance
        if x2 is None:
            x2 = x1

        # Compute continuous kernel
        if self.cont_kernel is not None:
            cont_idx = torch.as_tensor(self.cont_cols, device=device)
            k_cont = self.cont_kernel(
                x1.index_select(-1, cont_idx),
                x2.index_select(-1, cont_idx),
                diag=diag,
                last_dim_is_batch=last_dim_is_batch,
                **kwargs,
            )
            result = k_cont
        else:
            result = torch.ones(x1.shape[0], x2.shape[0], device=device)

        # Compute categorical kernel
        if self.cat_kernel is not None and self.cat_encoder is not None:
            encoders = self.cat_encoder if isinstance(self.cat_encoder, torch.nn.ModuleList) else [self.cat_encoder]
            # Encode each categorical group and collect encoded outputs
            z1_cat_list = []
            z2_cat_list = []
            for encoder, col_group in zip(encoders, self.cat_cols):
                cat_idx = torch.as_tensor(col_group, device=device)
                x1_group = x1.index_select(-1, cat_idx)
                x2_group = x2.index_select(-1, cat_idx)

                z1_c, z2_c = encoder.encode_pair(x1_group, x2_group)
                z1_cat_list.append(z1_c)
                z2_cat_list.append(z2_c)

            # Concatenate all encoded outputs
            z1_cat_concat = torch.cat(z1_cat_list, dim=-1)
            z2_cat_concat = torch.cat(z2_cat_list, dim=-1)

            # Apply single cat_kernel to concatenated outputs
            result_cat = self.cat_kernel(
                z1_cat_concat,
                z2_cat_concat,
                diag=diag,
                last_dim_is_batch=last_dim_is_batch,
                **kwargs,
            )

            if self.cont_kernel is not None:
                result = result.mul(result_cat)
            else:
                result = result_cat

        # Compute source kernel
        if self.source_kernel is not None and self.source_encoder is not None:
            source_idx = torch.as_tensor(self.source_cols, device=device)
            x1_source = x1.index_select(-1, source_idx)
            x2_source = x2.index_select(-1, source_idx)
            z1_s, z2_s = self.source_encoder.encode_pair(x1_source, x2_source)
            k_source = self.source_kernel(
                z1_s,
                z2_s,
                diag=diag,
                last_dim_is_batch=last_dim_is_batch,
                **kwargs,
            )
            result = result.mul(k_source)

        # Multiply final result by outputscale
        return result

    def _process_cont(self, cont_kernel):
        """Process continuous kernel."""
        if cont_kernel is not None or len(self.cont_cols) > 0:
            if cont_kernel is None:  # Default cont_kernel
                gauss_k = GaussianKernel(ard_num_dims=len(self.cont_cols))
                self.cont_kernel = gauss_k
            else:  # User-provided cont_kernel
                self.cont_kernel = cont_kernel
        else:
            self.cont_kernel = None

    def _coerce_encoder_type(self, encoder_type, name):
        if encoder_type not in {"matrix", "nn"}:
            raise ValueError(f"String {name} must be 'matrix' or 'nn'")
        return encoder_type

    def _make_default_kernel(self, kernel, z_dim, fix_lengthscale):
        if kernel is not None:
            return kernel
        gauss_k = GaussianKernel(ard_num_dims=z_dim)
        if fix_lengthscale:
            gauss_k.raw_lengthscale.requires_grad_(False)
            gauss_k.raw_lengthscale.data = torch.ones(z_dim) * 0.0
        else:
            gauss_k.raw_lengthscale.requires_grad_(True)
        return gauss_k

    def _resolve_encoder_list(self, encoder_spec, col_groups, name):
        # Default behavior: matrix encoder per group.
        if encoder_spec is None:
            encoders = [Encoder(input_dim=len(group), encoder_type="matrix", z_dim=self.z_dim) for group in col_groups]
        elif isinstance(encoder_spec, Encoder):
            if len(col_groups) != 1:
                raise ValueError(
                    f"A single {name} can only be used with one group. "
                    f"For grouped columns, pass a list of Encoder objects."
                )
            encoders = [encoder_spec]
        elif isinstance(encoder_spec, (str, bytes)):
            encoder_type = self._coerce_encoder_type(encoder_spec.lower(), name)
            encoders = [
                Encoder(
                    input_dim=len(group),
                    encoder_type=encoder_type,
                    z_dim=self.z_dim,
                )
                for group in col_groups
            ]
        elif hasattr(encoder_spec, "__iter__"):
            encoders = list(encoder_spec)
        else:
            raise TypeError(f"{name} must be an Encoder or a list of Encoder objects")

        if len(encoders) != len(col_groups):
            raise ValueError(f"Number of {name} entries must match number of groups")

        for i, (encoder, group) in enumerate(zip(encoders, col_groups)):
            if not isinstance(encoder, Encoder):
                raise TypeError(f"{name}[{i}] must be an Encoder instance")
            if encoder.input_dim != len(group):
                raise ValueError(f"{name}[{i}].input_dim={encoder.input_dim} does not match group size={len(group)}")
        return encoders

    def _resolve_single_encoder(self, encoder_spec, input_dim, name):
        if encoder_spec is None:
            encoder = Encoder(input_dim=input_dim, encoder_type="matrix", z_dim=self.z_dim)
        elif isinstance(encoder_spec, (str, bytes)):
            encoder_type = self._coerce_encoder_type(encoder_spec.lower(), name)
            encoder = Encoder(input_dim=input_dim, encoder_type=encoder_type, z_dim=self.z_dim)
        else:
            encoder = encoder_spec

        if not isinstance(encoder, Encoder):
            raise TypeError(f"{name} must be an Encoder instance")
        if encoder.input_dim != input_dim:
            raise ValueError(f"{name}.input_dim={encoder.input_dim} does not match size={input_dim}")
        return encoder

    def _process_cat(self, cat_encoder, cat_kernel):
        """Process categorical encoders and kernel."""
        if len(self.cat_cols) > 0 and not hasattr(self.cat_cols[0], "__iter__"):
            self.cat_cols = [self.cat_cols]

        if len(self.cat_cols) > 0:
            encoders = self._resolve_encoder_list(cat_encoder, self.cat_cols, "cat_encoder")
            if len(encoders) == 1:
                self.cat_encoder = encoders[0]
            else:
                self.cat_encoder = torch.nn.ModuleList(encoders)
            total_z_dim = sum(encoder.z_dim for encoder in encoders)
            self.cat_kernel = self._make_default_kernel(
                kernel=cat_kernel,
                z_dim=total_z_dim,
                fix_lengthscale=self.fix_lengthscale_cat,
            )
        elif cat_kernel is not None or cat_encoder is not None:
            raise ValueError("cat_cols must be non-empty when cat_encoder or cat_kernel is provided")
        else:
            self.cat_kernel = None
            self.cat_encoder = None

    def _process_source(self, source_encoder, source_kernel):
        """Process source encoder and kernel."""
        if len(self.source_cols) > 0:
            encoder = self._resolve_single_encoder(
                encoder_spec=source_encoder,
                input_dim=len(self.source_cols),
                name="source_encoder",
            )
            self.source_kernel = self._make_default_kernel(
                kernel=source_kernel,
                z_dim=encoder.z_dim,
                fix_lengthscale=self.fix_lengthscale_source,
            )
            self.source_encoder = encoder
        elif source_kernel is not None or source_encoder is not None:
            raise ValueError("source_cols must be non-empty when source_encoder or source_kernel is provided")
        else:
            self.source_kernel = None
            self.source_encoder = None
