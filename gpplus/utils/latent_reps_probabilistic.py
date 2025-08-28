import torch
import matplotlib.pyplot as plt
import os

def get_latent_representations_probabilistic(encoder, device='cpu', num_samples=100, ind=None, save_dir=None, z_dim=2):
    """
    Generate and visualize probabilistic latent representations from a source encoder.
    
    This function samples latent vectors from a probabilistic source encoder for each 
    input category (one-hot encoded), by generating multiple latent samples per source 
    using Gaussian noise. It then plots the latent embeddings in 2D.
    
    Args:
        encoder (torch.nn.Module): 
            The source encoder model used to map one-hot inputs to latent space.
        device (str, optional): 
            Device on which to perform computations ('cpu' or 'cuda'). Default is 'cpu'.
        num_samples (int, optional): 
            Number of latent samples to generate per source. Default is 100.
        ind (int or None, optional): 
            Index used for plot title and saved filename for differentiation. Default is None.
        save_dir (str or None, optional): 
            Directory to save the plot. If None, the plot is shown instead of saved. Default is None.
        z_dim (int, optional): 
            Dimensionality of the latent space. Default is 2.
    
    Returns:
        None
            Displays or saves a 2D scatter plot of latent representations. Each category 
            is represented with distinct markers for visualization.
    
    Notes:
        - This function assumes the encoder supports a forward pass with an `epsilon` argument for probabilistic sampling.
        - The plot uses markers to distinguish different source categories.
        - Works with 'OneHotToLatent' source encoder from gpplus.utils.one_hot_to_latent_nn.py
    """

    num_sources = encoder.input_dim
    onehots = torch.eye(num_sources, device=device)

    all_z = []
    all_labels = []
    # z_plot

    with torch.no_grad():
        for i in range(num_sources):
            source_vec = onehots[i].unsqueeze(0).repeat(num_samples, 1)  # shape [num_samples, input_dim]
            epsilon = torch.randn(num_samples, z_dim, device=device)
            z_samples = encoder(source_vec, epsilon=epsilon, visualize=True).cpu()
            all_z.append(z_samples)
            all_labels.append(i)

    plt.figure(figsize=(8, 6))
    markers = ['o', 's', '^', 'D']
    
    for i in range(len(all_z)):
        z_np = all_z[i].numpy()
        plt.scatter(z_np[:, 0], z_np[:, 1], label=f"Source {i}", alpha=0.5, s=10, marker=markers[i % len(markers)])
    
    plt.legend()
    plt.title(f"Probabilistic Latent Source Encoder Samples {ind}")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    
    if save_dir is not None:
        save_path = os.path.join(save_dir, f'probailistic_latent_space_plot{ind}.png')
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
