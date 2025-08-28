import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import math

import numpy as np
# import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch

def plot_violin(df_results, save_path):
    """
    Creates two subplots:
      - Left subplot: NRMSE (log scale)
      - Right subplot: NIS   (log scale)
    Each subplot:
      - Violinplot per kernel
      - Individual scatter points
      - Lines indicating Q1, Median, Q3
      - A point (dot) indicating mean
    """
    # Extract kernel labels
    kernels = df_results["Kernel"].unique()

    # For each kernel, gather arrays of NRMSE and NIS over seeds
    nrmse_data = [df_results[df_results["Kernel"] == k]["NRMSE"].values for k in kernels]
    nis_data   = [df_results[df_results["Kernel"] == k]["NIS"].values   for k in kernels]

    # Create figure with 2 subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Violin Plots of NRMSE and NIS")

    # -------------------------------------------------------
    # 1) Left Subplot: NRMSE
    # -------------------------------------------------------
    positions = np.arange(len(kernels))
    parts1 = ax1.violinplot(nrmse_data, positions=positions, widths=0.5, showextrema=False)

    # Color the violins (optional)
    for pc in parts1['bodies']:
        pc.set_facecolor("red")
        pc.set_edgecolor("black")
        pc.set_alpha(0.5)

    # Overlay scatter plots for individual points
    for i, vals in enumerate(nrmse_data):
        ax1.scatter(
            np.full_like(vals, i, dtype=float),  # x-values (all = i)
            vals,                                # y-values
            color="red", alpha=0.6, s=10
        )

    # Overlay lines for Q1, Median, Q3, plus a dot for Mean
    for i, vals in enumerate(nrmse_data):
        q1     = np.percentile(vals, 25)
        median = np.median(vals)
        q3     = np.percentile(vals, 75)
        mean_  = np.mean(vals)

        # Q1 line
        ax1.plot([i - 0.15, i + 0.15], [q1, q1], color="black", lw=2)
        # Median line
        ax1.plot([i - 0.2, i + 0.2], [median, median], color="darkred", lw=2)
        # Q3 line
        ax1.plot([i - 0.15, i + 0.15], [q3, q3], color="black", lw=2)
        # Dot for mean
        ax1.scatter(i, mean_, color="blue", marker="o", s=40, zorder=3)

    # Set log scale if desired
    ax1.set_yscale("log")

    # Labeling, ticks, etc.
    ax1.set_xticks(range(len(kernels)))
    ax1.set_xticklabels(kernels, rotation=45, ha='right')
    ax1.set_ylabel("NRMSE (log scale)", color="red")
    ax1.set_title("NRMSE per Kernel")

    # -------------------------------------------------------
    # 2) Right Subplot: NIS
    # -------------------------------------------------------
    parts2 = ax2.violinplot(nis_data, positions=positions, widths=0.5, showextrema=False)

    # Color the violins
    for pc in parts2['bodies']:
        pc.set_facecolor("blue")
        pc.set_edgecolor("black")
        pc.set_alpha(0.5)

    # Scatter points
    for i, vals in enumerate(nis_data):
        ax2.scatter(
            np.full_like(vals, i, dtype=float),
            vals,
            color="blue", alpha=0.6, s=10
        )

    # Q1, Median, Q3, Mean
    for i, vals in enumerate(nis_data):
        q1     = np.percentile(vals, 25)
        median = np.median(vals)
        q3     = np.percentile(vals, 75)
        mean_  = np.mean(vals)

        ax2.plot([i - 0.15, i + 0.15], [q1, q1], color="black", lw=2)
        ax2.plot([i - 0.2, i + 0.2], [median, median], color="darkblue", lw=2)
        ax2.plot([i - 0.15, i + 0.15], [q3, q3], color="black", lw=2)
        ax2.scatter(i, mean_, color="red", marker="o", s=40, zorder=3)

    ax2.set_yscale("log")
    ax2.set_xticks(range(len(kernels)))
    ax2.set_xticklabels(kernels, rotation=45, ha='right')
    ax2.set_ylabel("NIS (log scale)", color="blue")
    ax2.set_title("NIS per Kernel")

    # -------------------------------------------------------
    # Final layout, save, and close
    # -------------------------------------------------------
    plt.tight_layout()
    plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {save_path}")

def plot_predictions_vs_true(
    test_y, pred_mean, pred_var,
    source_indices, source_names,
    title="Predictions vs True Values", 
    save_dir="D:/gpplus_examples/examples/wing_MV_MF/main_wing_MV_MF",
    dpi=300,
    close_fig=True,
    ):
    """
    Plots true values (test_y) vs predictions (pred_mean) with confidence intervals.
    
    Args:
        test_y (array-like): True target values.
        pred_mean (array-like): Predicted mean values.
        pred_var (array-like): Predicted variances.
        source_indices (array-like): Array of source indices for each data point.
        source_names (list): List of source names corresponding to indices.
        title_base (str): Base string for plot titles.
        save_dir (str): Directory to save plots.
    """
    # Convert to numpy arrays if not already
    test_y = np.asarray(test_y)
    pred_mean = np.asarray(pred_mean)
    pred_var = np.asarray(pred_var)
    source_indices = np.asarray(source_indices)

    # Calculate standard deviation and confidence intervals
    pred_std = np.sqrt(pred_var)
    lower_bound = pred_mean - 2 * pred_std  # ~95% CI
    upper_bound = pred_mean + 2 * pred_std
    
    # print(source_names)
    # Create plots for each source
    for source_idx, source_name in enumerate(source_names):
        # Filter data for this source
        mask = source_indices == source_idx
        if not np.any(mask):
            continue  # Skip if no data for this source
            
        test_y_source = test_y[mask]
        pred_mean_source = pred_mean[mask]
        lower_bound_source = lower_bound[mask]
        upper_bound_source = upper_bound[mask]
        
        # Sort by PREDICTED values for smooth confidence bands
        sorted_indices = np.argsort(test_y_source)
        pred_mean_sorted = pred_mean_source[sorted_indices]
        test_y_sorted = test_y_source[sorted_indices]
        lower_bound_sorted = lower_bound_source[sorted_indices]
        upper_bound_sorted = upper_bound_source[sorted_indices]
        
        # Calculate RRMSE compared to source 's0'
        if source_name != 's0':
            mask_s0 = source_indices == 0
            test_y_s0 = test_y[mask_s0]
            rrmse = np.sqrt(np.mean((test_y_sorted - test_y_s0) ** 2)) / np.mean(test_y_s0)
        else:
            rrmse = 0

        # Create the plot
        plt.figure(figsize=(8, 8))

        # Plot the perfect prediction line (y=x)
        min_val = min(np.min(pred_mean_sorted), np.min(test_y_sorted))
        max_val = max(np.max(pred_mean_sorted), np.max(test_y_sorted))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')

        # Confidence interval (shaded)
        plt.fill_between(
            test_y_sorted,
            lower_bound_sorted,
            upper_bound_sorted,
            color='lightgreen',
            alpha=0.5,
            label='95% Confidence Interval'
        )
        
        # True vs Pred
        plt.scatter(
            test_y_sorted,
            pred_mean_sorted,
            color='blue',
            marker='.',
            alpha=0.7,
            label='Test Data'
        )

        # Compute MSE for this source
        mse = np.mean((pred_mean_sorted - test_y_sorted) ** 2)
        nrmse = np.sqrt(mse) / np.std(test_y_sorted)
        # Add MSE as text on the plot (top left corner)
        plt.text(
            0.05, 0.95,
            f"MSE: {mse:.4f}\nNRMSE: {nrmse:.4f}\nRRMSE (s0 vs {source_name}): {rrmse:.4f}",
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )

        # Customize plot
        plt.xlabel('Target Value')
        plt.ylabel('Prediction')
        plt.title(f"{title} - {source_name} Source")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save plot
        save_path = os.path.join(save_dir, f"gp_predictions_{source_name.lower()}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
        if close_fig:
            plt.close()

def plot_latent_space(latent_reps, labels, title, save_path, cat_dims=None, qual_dict=None, cat_combos=None):
    """
    Plot 2D projection of latent space
    
    Args:
        latent_reps: Latent space representations
        labels: Either source names (strings) or categorical indices (integers)
        title: Plot title
        save_path: Where to save the plot
        cat_dims: List of dimensions for each categorical variable (only needed for categorical plots)
        qual_dict: Dictionary mapping column indices to number of levels (only needed for categorical plots)
        cat_combos: One-hot encoded categorical combinations (only needed for categorical plots)
    """
    # Reduce to 2D if needed
    if latent_reps.shape[1] > 2:
        tsne = TSNE(n_components=2)
        proj = tsne.fit_transform(latent_reps.cpu().numpy())
    else:
        proj = latent_reps.cpu().numpy()
    
    plt.figure(figsize=(15, 15))
    
    if isinstance(labels[0], str):
        # Source latent space plot
        plt.scatter(proj[:, 0], proj[:, 1], c=range(len(labels)), cmap='viridis', s=100)
        
        # Add text labels for each point
        for i, txt in enumerate(labels):
            plt.annotate(txt, (proj[i, 0], proj[i, 1]), fontsize=10, 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.title(f"{title}\nSource Fidelity Levels")
        
    else:
        # Categorical latent space plot
        if cat_dims is None or qual_dict is None or cat_combos is None:
            raise ValueError("cat_dims, qual_dict, and cat_combos must be provided for categorical plots")
        
        # Get all labels first
        all_labels = []
        for i in range(len(proj)):
            var_values = []
            current_pos = 0
            
            # For each categorical variable
            for var_idx, (col_idx, num_levels) in enumerate(sorted(qual_dict.items())):
                # Get the one-hot encoded values for this variable
                var_one_hot = cat_combos[i, current_pos:current_pos + num_levels]
                # Convert one-hot to actual level (1-based)
                level = torch.argmax(var_one_hot).item() + 1
                var_values.append(f'Var{col_idx}={level}')
                current_pos += num_levels
            
            # Print debug information
            # print(f"{i:4d} | {', '.join(var_values)}")
            
            all_labels.append(', '.join(var_values))
        
        # Plot points with colors based on their index
        scatter = plt.scatter(proj[:, 0], proj[:, 1], c=range(len(proj)), cmap='viridis', s=100, alpha=0.7)
        
        # Add labels with spiral layout
        for i, (x, y) in enumerate(proj):
            # Calculate spiral position
            angle = 2 * np.pi * i / len(proj)
            radius = 15 + (i % 3) * 5  # Vary radius slightly to avoid overlap
            dx = radius * np.cos(angle)
            dy = radius * np.sin(angle)
            
            # Add label with the calculated offset
            plt.annotate(all_labels[i], (x, y), fontsize=8,
                        xytext=(dx, dy), textcoords='offset points',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', pad=2),
                        ha='center', va='center')
        
        plt.title(f"{title}\nCategorical Variables and Their Values")
    
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_data_distribution(data, save_dir):
    """
    Plot the distribution of data to verify data generation.
    
    Args:
        data: Dictionary containing the generated data
        save_dir: Directory to save the plots
    """
    # Convert tensors to numpy for plotting
    x_train = data['x_train_full'].cpu().numpy()
    y_train = data['y_train_full'].cpu().numpy()
    source_train = data['source_train_full'].cpu().numpy()
    
    # Get column indices
    cols = data['column_indices']
    source_cols = cols['source']
    continuous_cols = cols['continuous']
    categorical_cols = cols['categorical']
    
    # Print diagnostic information
    print("\nData Generation Diagnostics:")
    print(f"Total number of samples: {len(x_train)}")
    print(f"Number of categorical columns: {len(categorical_cols)}")
    print(f"Number of continuous columns: {len(continuous_cols)}")
    print(f"Number of source columns: {len(source_cols)}")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Distribution of continuous variable (L)
    plt.subplot(2, 2, 1)
    plt.hist(x_train[:, continuous_cols[0]], bins=30)
    plt.title('Distribution of L (Continuous Variable)')
    plt.xlabel('L value')
    plt.ylabel('Count')
    
    # 2. Distribution of categorical variables
    plt.subplot(2, 2, 2)
    
    # Get the categorical data
    cat_data = x_train[:, categorical_cols]
    
    # Only process categorical data if it exists
    if len(categorical_cols) > 0:
        # Get original categorical values
        E_values = data['metadata'].get('E_values', [])
        K_values = data['metadata'].get('K_values', [])
        I_values = data['metadata'].get('I_values', [])
        
        print("\nExpected categorical values:")
        print(f"E values: {E_values}")
        print(f"K values: {K_values}")
        print(f"I values: {I_values}")
        
        # Count occurrences of each categorical value
        E_counts = np.zeros(len(E_values))
        K_counts = np.zeros(len(K_values))
        I_counts = np.zeros(len(I_values))
        
        # Count occurrences
        for i, val in enumerate(E_values):
            E_counts[i] = np.sum(cat_data[:, i] == 1)
        for i, val in enumerate(K_values):
            K_counts[i] = np.sum(cat_data[:, i + len(E_values)] == 1)
        for i, val in enumerate(I_values):
            I_counts[i] = np.sum(cat_data[:, i + len(E_values) + len(K_values)] == 1)
        
        # Create bar plot for all categorical variables
        x = np.arange(len(E_values) + len(K_values) + len(I_values))
        width = 0.35
        
        # Create labels for all bars
        labels = [f'E={v}' for v in E_values] + [f'K={v}' for v in K_values] + [f'I={v}' for v in I_values]
        
        # Plot bars
        plt.bar(x, np.concatenate([E_counts, K_counts, I_counts]))
        plt.title('Distribution of Categorical Variables')
        plt.xticks(x, labels, rotation=45, ha='right')
        plt.ylabel('Count')
        
        # Add grid for better readability
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    else:
        plt.text(0.5, 0.5, 'No categorical variables present', 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes)
        plt.title('Distribution of Categorical Variables')
    
    # 3. Scatter plot of L vs output for each source
    plt.subplot(2, 2, 3)
    for source_idx, source_name in enumerate(data['metadata']['source_names']):
        mask = source_train == source_idx
        plt.scatter(x_train[mask, continuous_cols[0]], y_train[mask], 
                   label=f'Source {source_name}', alpha=0.6)
    plt.title('L vs Output by Source')
    plt.xlabel('L value')
    plt.ylabel('Output')
    plt.legend()
    
    # 4. Box plot of output by categorical values
    plt.subplot(2, 2, 4)
    if len(categorical_cols) > 0:
        box_data = []
        labels = []
        
        # Group data by categorical values
        for i, val in enumerate(E_values):
            mask = cat_data[:, i] == 1
            box_data.append(y_train[mask])
            labels.append(f'E={val}')
        for i, val in enumerate(K_values):
            mask = cat_data[:, i + len(E_values)] == 1
            box_data.append(y_train[mask])
            labels.append(f'K={val}')
        for i, val in enumerate(I_values):
            mask = cat_data[:, i + len(E_values) + len(K_values)] == 1
            box_data.append(y_train[mask])
            labels.append(f'I={val}')
        
        plt.boxplot(box_data, labels=labels)
        plt.title('Output Distribution by Categorical Values')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Output')
    else:
        plt.text(0.5, 0.5, 'No categorical variables present', 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes)
        plt.title('Output Distribution by Categorical Values')
    
    # Adjust layout and save
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'data_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create separate plot for categorical distributions only if categorical variables exist
    if len(categorical_cols) > 0:
        fig_cat = plt.figure(figsize=(15, 5))
        
        # E values
        plt.subplot(1, 3, 1)
        plt.bar(range(len(E_values)), E_counts)
        plt.title('Distribution of E values')
        plt.xticks(range(len(E_values)), [f'E={v}' for v in E_values], rotation=45)
        plt.ylabel('Count')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # K values
        plt.subplot(1, 3, 2)
        plt.bar(range(len(K_values)), K_counts)
        plt.title('Distribution of K values')
        plt.xticks(range(len(K_values)), [f'K={v}' for v in K_values], rotation=45)
        plt.ylabel('Count')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # I values
        plt.subplot(1, 3, 3)
        plt.bar(range(len(I_values)), I_counts)
        plt.title('Distribution of I values')
        plt.xticks(range(len(I_values)), [f'I={v}' for v in I_values], rotation=45)
        plt.ylabel('Count')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        save_path_cat = os.path.join(save_dir, 'categorical_distribution.png')
        plt.savefig(save_path_cat, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print actual counts
        print("\nActual categorical value counts:")
        print("E values counts:", E_counts)
        print("K values counts:", K_counts)
        print("I values counts:", I_counts)
    
    print(f"\nPlots saved to:")
    print(f"- {save_path}")
    if len(categorical_cols) > 0:
        print(f"- {save_path_cat}")
