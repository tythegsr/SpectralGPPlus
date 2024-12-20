import torch

def generate_data(
    nx: int,
    ny: int,
    nsamples: int,         
):
    """
    Given a function y(s, x), for each sample s generate the set of output values y over the grid x.
    Input: s. Size nsamples.
    Output: y(s, x). Size: nsamples*nx*ny
    """
    
    # Generate simulation samples
    train_samples = torch.linspace(1, 5, nsamples) # Shape (nsamples, )

    # Generate output grid
    Np = nx * ny
    train_x1 = torch.linspace(0, 1, nx)
    train_x2 = torch.linspace(0, 1, ny)

    train_x1, train_x2 = torch.meshgrid(train_x1, train_x2)
    train_x1, train_x2 = train_x1.flatten(), train_x2.flatten()
    grid_points = torch.stack([train_x1, train_x2], dim=-1)  # Shape: (Np, 2)

    all_train_x = []
    all_train_y = []

    for sim in range(nsamples):
        # Add a column for the simulation feature x3 and stack with the inputs
        x3 = train_samples[sim].item()
        sim_column = torch.full((Np, 1), x3)  # Shape: (Np, 1)
        x = torch.cat([grid_points, sim_column], dim=-1)  # Shape: (Np, 3)

        # Generate outputs for each point in the grid: (Np, 1)
        outputs = torch.sin(x[:, 0] * x3 * torch.pi) * torch.cos(x[:, 1] * x3 * torch.pi)
        outputs += 0.05 * torch.randn_like(outputs)  # Add some noise
        outputs = outputs.unsqueeze(-1)

        # Append to the lists
        all_train_x.append(x)
        all_train_y.append(outputs)

    # Concatenate the results for all simulations
    train_x = torch.cat(all_train_x, dim=0)  # Shape: (Np * nsamples, 3)
    train_y = torch.cat(all_train_y, dim=0)  # Shape: (Np * nsamples, 1)

    return train_x, train_y.squeeze(-1) # (Np * nsamples, 3), (Np * nsamples)