import torch
import matplotlib.pyplot as plt

from gpplus.eval.evaluator import Evaluator

def plot_2d(
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    evaluator: Evaluator,
    nx: int,
    ny: int,
    sample_index: int = 0,
):
    prediction, std_pred = evaluator.evaluate(test_x.unsqueeze(0))
    prediction = prediction.squeeze()

    # Initialize the figure with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Get data for output
    Np = nx*ny
    init = Np*sample_index
    end = init + Np
    x = test_x[init:end, :2]
    y = test_y[init:end]
    pred = prediction[init:end]

    
    # Original output
    mean_gt = y.numpy().reshape(nx, ny)
    im0 = axs[0].imshow(mean_gt, extent=(x[:,0].min(), x[:,0].max(), x[:,1].min(), x[:,1].max()), origin='lower', cmap='viridis', alpha=0.8)
    axs[0].set_title('Ground truth')
    axs[0].set_xlabel('X1')
    axs[0].set_ylabel('X2')
    plt.colorbar(im0, ax=axs[0], label='Ground Truth')

    # Predicted output
    mean_pred = pred.detach().numpy().reshape(nx,ny)
    im1 = axs[1].imshow(mean_pred, extent=(0, 1, 0, 1), origin='lower', cmap='viridis', alpha=0.8)
    axs[1].set_title('Predicted Mean')
    axs[1].set_xlabel('X1')
    axs[1].set_ylabel('X2')
    plt.colorbar(im1, ax=axs[1], label='Mean Prediction')

    # Error (Original - Predicted)
    error = mean_gt - mean_pred
    im2 = axs[2].imshow(error, extent=(0, 1, 0, 1), origin='lower', cmap='coolwarm', alpha=0.8)
    axs[2].set_title('Error (Ground Truth - Predicted)')
    axs[2].set_xlabel('X1')
    axs[2].set_ylabel('X2')
    plt.colorbar(im2, ax=axs[2], label='Error')

    plt.tight_layout()
    plt.savefig('./gp-mogp-kernel.png')