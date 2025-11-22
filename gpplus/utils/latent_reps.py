import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def get_latent_representations(model, qual_dict=None):
    """
    Generate latent representations with robust column index handling

    Args:
        model: Trained GP model
        qual_dict: Dictionary to create proper grouped OH encodings for checking latent representations.
                   If provided, will be used to generate proper combinations for grouped categorical encodings.
                   If there is only one categorical variable in your encoding, this is not necessary.
                   Not necessary if using separate categorical encoders.


    Returns:
        dict: encoder_data_dict mapping encoder names to their data. Keys include
              'cat_encoder', 'cat_encoder_i', and optionally 'source_encoder'.
              If no encoders are found, returns None.
    """
    model.eval()
    # Handle both direct CombinedKernel and LogScaleKernel wrapping CombinedKernel
    if hasattr(model.covar_module, "base_kernel"):
        combined_kernel = model.covar_module.base_kernel
    else:
        combined_kernel = model.covar_module

    # Check if categorical variables exist
    # Look for categorical encoders - prefer numbered ones, fallback to generic
    cat_encoders = [attr for attr in dir(combined_kernel) if attr.startswith("cat_encoder_")]
    if not cat_encoders and hasattr(combined_kernel, "cat_encoder"):
        cat_encoders = ["cat_encoder"]

    # Get encoder dimensions directly from the encoders
    cat_dims = []
    for encoder_name in cat_encoders:
        latent_net = getattr(combined_kernel, encoder_name)
        if hasattr(latent_net, "input_dim"):
            cat_dims.append(latent_net.input_dim)
        else:
            print(f"Warning: Encoder {encoder_name} has no input_dim attribute")
            continue

    # It's possible there are no categorical encoders but a source encoder exists.
    # So do not early-return here; we'll build whatever encoders exist.

    # Dictionary to store encoder data with names
    encoder_data_dict = {}

    # Loop through all categorical encoders
    for i, encoder_name in enumerate(cat_encoders):
        latent_net = getattr(combined_kernel, encoder_name)

        # Determine encoder dimensions
        if qual_dict is not None and encoder_name == "cat_encoder":
            # For grouped categorical encoding, use qual_dict to get proper dimensions
            cat_vars = sorted(qual_dict.items())
            encoder_dims = [dim for _, dim in cat_vars]
        else:
            # For individual encoders, get dimension from the encoder itself
            if hasattr(latent_net, "input_dim"):
                encoder_dims = [latent_net.input_dim]
            else:
                print(f"Warning: Encoder {encoder_name} has no input_dim attribute, skipping")
                continue

        # Skip if no dimensions found
        if not encoder_dims:
            print(f"Warning: No dimensions found for encoder {encoder_name}, skipping")
            continue

        # Generate all possible combinations for this encoder's variables
        indices = torch.cartesian_prod(*[torch.arange(dim) for dim in encoder_dims])

        # Ensure indices is 2D (add dimension if it's 1D)
        if indices.dim() == 1:
            indices = indices.unsqueeze(1)

        # Create one-hot encoded combinations
        one_hots = [F.one_hot(indices[:, j], num_classes=dim) for j, dim in enumerate(encoder_dims)]
        encoder_combinations = torch.cat(one_hots, dim=1)

        # Get latent representations
        with torch.no_grad():
            latent_reps = latent_net(encoder_combinations.to(dtype=model.train_inputs[0].dtype))

        # Store in dictionary with encoder name
        encoder_data_dict[encoder_name] = {
            "combinations": encoder_combinations,
            "indices": indices,
            "latent_reps": latent_reps,
            "input_dim": getattr(latent_net, "input_dim", encoder_combinations.shape[1]),
        }

    # Optionally include source encoder, if present and there are multiple sources
    if hasattr(combined_kernel, "source_encoder") and combined_kernel.source_encoder is not None:
        try:
            n_sources = len(getattr(combined_kernel, "source_cols", []) or [])
        except Exception:
            n_sources = 0
        if n_sources and n_sources > 1:
            source_net = combined_kernel.source_encoder
            source_indices = torch.arange(n_sources).unsqueeze(1)
            source_combinations = torch.eye(n_sources)
            with torch.no_grad():
                source_latent = source_net(source_combinations.to(dtype=model.train_inputs[0].dtype))
            encoder_data_dict["source_encoder"] = {
                "combinations": source_combinations,
                "indices": source_indices,
                "latent_reps": source_latent,
                "input_dim": getattr(source_net, "input_dim", n_sources),
            }

    if len(encoder_data_dict) == 0:
        return None

    return encoder_data_dict


def plot_encoders(model, qual_dict=None, save_path=None):
    """
    Plot latent representations for all categorical encoders in the model.

    Args:
        model: Trained GP model
        qual_dict: Dictionary mapping categorical variable names to their dimensions
        save_path: Optional path to save the plot
    """
    # Get all encoder data from your fixed function
    encoder_data_dict = get_latent_representations(model, qual_dict)

    if encoder_data_dict is None:
        print("No encoder data found")
        return

    # Create subplots for each encoder
    encoder_names = list(encoder_data_dict.keys())
    num_encoders = len(encoder_names)
    fig, axes = plt.subplots(1, num_encoders, figsize=(5 * num_encoders, 5))
    if num_encoders == 1:
        axes = [axes]

    for i, encoder_name in enumerate(encoder_names):
        encoder_data = encoder_data_dict[encoder_name]
        indices = encoder_data["indices"]
        latent_reps = encoder_data["latent_reps"]

        # Plot this encoder
        ax = axes[i]

        # Get unique categories for coloring
        unique_cats = indices.unique(dim=0)
        colors = plt.cm.tab10(torch.arange(len(unique_cats)).float() / len(unique_cats))

        for j, cat_combo in enumerate(unique_cats):
            # Find indices that match this category combination
            mask = (indices == cat_combo).all(dim=1)
            points = latent_reps[mask]

            ax.scatter(
                points[:, 0], points[:, 1], label=f"Cat {j}: {cat_combo.tolist()}", alpha=0.7, s=50, c=[colors[j]]
            )
            # Annotate each point with its index
            point_indices = torch.where(mask)[0]
            for idx in point_indices:
                x = latent_reps[idx, 0].item()
                y = latent_reps[idx, 1].item()
                ax.text(x, y, str(int(idx.item())), fontsize=8, ha="center", va="bottom")

        ax.set_xlabel("Latent Dimension 1")
        ax.set_ylabel("Latent Dimension 2")
        ax.set_title(f"{encoder_name}")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
