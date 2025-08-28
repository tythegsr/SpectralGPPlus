import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import math

import numpy as np
# import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import gpytorch
import gpplus
import gpplus.utils as utils
from torch.quasirandom import SobolEngine
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ZeroMean
from gpytorch.kernels import RBFKernel, ScaleKernel, ProductKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.manifold import TSNE
# from gpytorch.utils import Callback
from gpplus.training import GPTrainer
from gpplus.training.callbacks import PrintLossCallback
from scipy.interpolate import make_interp_spline

def wing_mixed_variables(X, source='s0'):
    """
    Compute wing weight given input variables.
    
    Args:
        X (np.ndarray): Input array of shape [n_samples, 10] with columns:
            0: Sw (wing area, sq ft)
            1: Wfw (weight of fuel in the wing, lb)
            2: A (aspect ratio)
            3: Gama (quarter-chord sweep angle, degrees)
            4: q (dynamic pressure at cruise, lb/sq ft)
            5: lamb (taper ratio)
            6: tc (airfoil thickness to chord ratio)
            7: Nz (ultimate load factor)
            8: Wdg (flight design gross weight, lb)
            9: Wp (paint weight, lb/sq ft)
        source (str): Source of the data
    Returns:
        np.ndarray: Wing weight values for each input sample
    """
    Sw = X[..., 0]
    Wfw = X[..., 1]
    A = X[..., 2]
    Gama = X[..., 3] * (torch.pi/180.0)  # Convert to radians
    q = X[..., 4]
    lamb = X[..., 5]
    tc = X[..., 6]
    Nz = X[..., 7]
    Wdg = X[..., 8]
    Wp = X[..., 9]
    cos_Gama = torch.cos(Gama)
    # Wing weight calculation
    if source == 's0':
        result = 0.036 * Sw**0.758 * Wfw**0.0035 * (A/(cos_Gama) ** 2) ** 0.6 * \
        q**0.006 * lamb**0.04 * ((100 * tc)/(cos_Gama))**(-0.3) * \
        (Nz * Wdg) ** 0.49 + Sw * Wp
    elif source == 's1':
        result = 0.036 * Sw**0.758 * Wfw**0.0035 * (A/(cos_Gama) ** 2) ** 0.6 * \
        q**0.006 * lamb**0.04 * ((100 * tc)/(cos_Gama))**(-0.3) * \
        (Nz * Wdg) ** 0.49 + 1 * Wp
    elif source == 's2':
        result = 0.036 * Sw**0.8 * Wfw**0.0035 * (A/(cos_Gama) ** 2) ** 0.6 * \
        q**0.006 * lamb**0.04 * ((100 * tc)/(cos_Gama))**(-0.3) * \
        (Nz * Wdg) ** 0.49 + 1 * Wp
    elif source == 's3':
        result = 0.036 * Sw**0.9 * Wfw**0.0035 * (A/(cos_Gama) ** 2) ** 0.6 * \
        q**0.006 * lamb**0.04 * ((100 * tc)/(cos_Gama))**(-0.3) * \
        (Nz * Wdg) ** 0.49 + 0 * Wp
    

    return result

def analyze_buckling_categorical_ordering():
    """
    Analyze the expected output ordering for different categorical combinations
    in the buckling problem to help understand latent space organization.
    """
    # Define the categorical values
    # E_values = [73.1, 200.0]
    # K_values = [0.5, 2.0] 
    # I_values = [29.5, 9.49]

    E_values = [73.1, 200.0]
    K_values = [0.5, 0.7, 1.0, 2.0]
    I_values = [9.49, 12.1, 29.5]
    
    # Use a fixed L value for comparison
    L = 1.0  # middle of the range
    
    print("Buckling Problem Categorical Analysis:")
    print("=" * 50)
    
    results = []
    
    for i, E in enumerate(E_values):
        for j, K in enumerate(K_values):
            for k, I in enumerate(I_values):
                # Calculate buckling load for s0 (simpler formula)
                P_s0 = np.pi * E * I / (L * K) ** 2
                
                # Calculate buckling load for s1 (more complex formula)
                P_s1 = ((np.pi * E * I / (L * K) ** 2) + L) ** 1.1
                
                results.append({
                    'E_idx': i, 'K_idx': j, 'I_idx': k,
                    'E_val': E, 'K_val': K, 'I_val': I,
                    'P_s0': P_s0, 'P_s1': P_s1,
                    'combo': f"E{i}K{j}I{k}"
                })
    
    # Sort by P_s0 (the simpler formula)
    results.sort(key=lambda x: x['P_s0'])
    
    print("Categorical combinations ordered by expected output magnitude (P_s0):")
    print("Format: E_idx K_idx I_idx | E_val K_val I_val | P_s0 | P_s1 | combo")
    print("-" * 80)
    
    for i, result in enumerate(results):
        print(f"{result['E_idx']:2d} {result['K_idx']:2d} {result['I_idx']:2d} | "
              f"{result['E_val']:5.1f} {result['K_val']:4.1f} {result['I_val']:5.2f} | "
              f"{result['P_s0']:8.1f} | {result['P_s1']:8.1f} | {result['combo']}")
    
    print("\nExpected latent space ordering (from lowest to highest output):")
    print(" -> ".join([r['combo'] for r in results]))
    
    return results

def buckling_mixed_variables(X, source='s0'):
    """
    Compute wing weight given input variables.
    
    Args:
        X (np.ndarray): Input array of shape [n_samples, 10] with columns:
            0: L (length of the beam, m)
            1: E (Young's modulus, Pa)
            2: K (shear modulus, Pa)
            3: I (moment of inertia, m^4)
        source (str): Source of the data
    Returns:
        np.ndarray: Wing weight values for each input sample
    """
    L = X[..., 0]
    E = X[..., 1]
    K = X[..., 2]
    I = X[..., 3]
     
    
    # Wing weight calculation
    if source == 's0':
        P = np.pi * E * I / (L * K) ** 2
    elif source == 's1':
        P = ((np.pi * E * I / (L *K) ** 2) + L) ** 1.1

    return P

def borehole_mixed_variables(X, source='s0'):
    """
    Compute borehole water flow rate given input variables.
    
    Args:
        X (np.ndarray): Input array of shape [n_samples, 8] with columns:
            0: rw (radius of borehole, m)
            1: r (radius of influence, m)
            2: Tu (transmissivity of upper aquifer, m^2/yr)
            3: Hu (potentiometric head of upper aquifer, m)
            4: Tl (transmissivity of lower aquifer, m^2/yr)
            5: Hl (potentiometric head of lower aquifer, m)
            6: L (length of borehole, m)
            7: Kw (hydraulic conductivity of borehole, m/yr)
        source (str): Source of the data
    Returns:
        np.ndarray: Water flow rate values for each input sample
    """
    rw = X[..., 0]
    r = X[..., 1]
    Tu = X[..., 2]
    Hu = X[..., 3]
    Tl = X[..., 4]
    Hl = X[..., 5]
    L = X[..., 6]
    Kw = X[..., 7]
    
    # Borehole water flow rate calculation
    if source == 's0':
        numerator = 2 * torch.pi * Tu * (Hu - Hl)
        denominator = torch.log(r / rw) * (1 + 2 * L * Tu / (torch.log(r / rw) * rw**2 * Kw) + Tu / Tl)
        result = numerator / denominator
    elif source == 's1':
        numerator = 2 * torch.pi * Tu * (Hu - 0.8 * Hl)
        denominator = torch.log(r / rw) * (1 + 2 * L * Tu / (torch.log(r / rw) * rw**2 * Kw) + Tu / Tl)
        result = numerator / denominator
    elif source == 's2':
        numerator = 2 * torch.pi * Tu * (Hu - 3 * Hl)
        denominator = torch.log(r / rw) * (1 + 8 * L * Tu / (torch.log(r / rw) * rw**2 * Kw) + 0.75 * Tu / Tl)
        result = numerator / denominator
    elif source == 's3':
        numerator = 2 * torch.pi * Tu * (1.1 * Hu - Hl)
        denominator = torch.log(4 * r / rw) * (1 + 3 * L * Tu / (torch.log(r / rw) * rw**2 * Kw) + Tu / Tl)
        result = numerator / denominator
    elif source == 's4':
        numerator = 2 * torch.pi * Tu * (1.05 * Hu - Hl)
        denominator = torch.log(2 * r / rw) * (1 + 2 * L * Tu / (torch.log(r / rw) * rw**2 * Kw) + Tu / Tl)
        result = numerator / denominator

    return result

def analyze_borehole_source_distributions(save_dir=None):
    """
    Analyze the expected output distributions and characteristics for different sources
    in the borehole problem to help understand multi-fidelity behavior.
    
    This function generates sample data for each source and analyzes:
    1. Output value ranges and distributions
    2. Statistical properties (mean, std, min, max)
    3. Relationships between sources
    4. Visualizations of the distributions
    
    Args:
        save_dir (str, optional): Directory to save the analysis plot. If None, saves in current directory.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    
    # Define variable names for better labeling
    var_names = [
        'rw (radius of borehole, m)',
        'r (radius of influence, m)', 
        'Tu (transmissivity of upper aquifer, m²/yr)',
        'Hu (potentiometric head of upper aquifer, m)',
        'Tl (transmissivity of lower aquifer, m²/yr)',
        'Hl (potentiometric head of lower aquifer, m)',
        'L (length of borehole, m)',
        'Kw (hydraulic conductivity of borehole, m/yr)'
    ]
    
    # Define bounds for borehole problem (8 variables)
    l_bound = torch.tensor([0.05, 100., 63070., 990., 63.1, 700., 1120., 9855.])
    u_bound = torch.tensor([0.15, 50000., 115600., 1110., 116., 820., 1680., 12045.])
    
    # Generate sample data for analysis
    n_samples = 10000
    torch.manual_seed(42)  # Fixed seed for reproducible analysis
    
    # Generate random samples using Sobol sequence for better coverage
    sobol_engine = SobolEngine(dimension=8, scramble=True, seed=42)
    x_raw = utils.scale(sobol_engine.draw(n_samples).float(), l_bound, u_bound)
    
    # Calculate outputs for each source
    sources = ['s0', 's1', 's2', 's3', 's4']
    outputs = {}
    
    print("Borehole Problem Source Analysis:")
    print("=" * 60)
    
    for source in sources:
        y = borehole_mixed_variables(x_raw, source)
        outputs[source] = y
        
        # Calculate statistics
        y_np = y.cpu().numpy()
        print(f"\n{source.upper()} Statistics:")
        print(f"  Mean: {y_np.mean():.2f}")
        print(f"  Std:  {y_np.std():.2f}")
        print(f"  Min:  {y_np.min():.2f}")
        print(f"  Max:  {y_np.max():.2f}")
        print(f"  Range: {y_np.max() - y_np.min():.2f}")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Borehole Problem: Multi-Fidelity Source Analysis', fontsize=16, fontweight='bold')
    
    # 1. Output distributions (histograms)
    ax1 = axes[0, 0]
    for i, source in enumerate(sources):
        y_np = outputs[source].cpu().numpy()
        ax1.hist(y_np, bins=50, alpha=0.6, label=source, density=True)
    ax1.set_xlabel('Water Flow Rate (m³/yr)')
    ax1.set_ylabel('Density')
    ax1.set_title('Output Distributions by Source')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot comparison
    ax2 = axes[0, 1]
    y_data = [outputs[source].cpu().numpy() for source in sources]
    bp = ax2.boxplot(y_data, labels=sources, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax2.set_ylabel('Water Flow Rate (m³/yr)')
    ax2.set_title('Output Distributions (Box Plot)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Source comparison scatter (s0 vs others)
    ax3 = axes[0, 2]
    y_s0 = outputs['s0'].cpu().numpy()
    for source in sources[1:]:
        y_other = outputs[source].cpu().numpy()
        ax3.scatter(y_s0, y_other, alpha=0.3, label=f'{source} vs s0', s=10)
    ax3.plot([y_s0.min(), y_s0.max()], [y_s0.min(), y_s0.max()], 'k--', alpha=0.5, label='y=x')
    ax3.set_xlabel('s0 Output')
    ax3.set_ylabel('Other Source Output')
    ax3.set_title('Source Comparison (vs s0)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Statistical summary table
    ax4 = axes[1, 0]
    ax4.axis('tight')
    ax4.axis('off')
    
    # Create summary table
    summary_data = []
    for source in sources:
        y_np = outputs[source].cpu().numpy()
        summary_data.append([
            source,
            f"{y_np.mean():.1f}",
            f"{y_np.std():.1f}",
            f"{y_np.min():.1f}",
            f"{y_np.max():.1f}",
            f"{y_np.max() - y_np.min():.1f}"
        ])
    
    table = ax4.table(cellText=summary_data,
                     colLabels=['Source', 'Mean', 'Std', 'Min', 'Max', 'Range'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax4.set_title('Statistical Summary', fontweight='bold')
    
    # 5. Variable importance analysis (correlation with output)
    ax5 = axes[1, 1]
    correlations = {}
    for source in sources:
        y_np = outputs[source].cpu().numpy()
        x_np = x_raw.cpu().numpy()
        corrs = [np.corrcoef(x_np[:, i], y_np)[0, 1] for i in range(8)]
        correlations[source] = corrs
    
    # Plot correlations
    x_pos = np.arange(8)
    width = 0.15
    for i, source in enumerate(sources):
        ax5.bar(x_pos + i*width, correlations[source], width, label=source, alpha=0.7)
    
    ax5.set_xlabel('Variable Index')
    ax5.set_ylabel('Correlation with Output')
    ax5.set_title('Variable-Output Correlations by Source')
    ax5.set_xticks(x_pos + width * 2)
    ax5.set_xticklabels([f'Var {i+1}' for i in range(8)])
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Fidelity progression analysis
    ax6 = axes[1, 2]
    # Calculate how outputs change across fidelity levels
    y_s0 = outputs['s0'].cpu().numpy()
    fidelity_changes = {}
    for source in sources[1:]:
        y_other = outputs[source].cpu().numpy()
        fidelity_changes[source] = (y_other - y_s0) / y_s0 * 100  # Percentage change
    
    # Plot fidelity changes
    for source in sources[1:]:
        ax6.hist(fidelity_changes[source], bins=50, alpha=0.6, label=f'{source} vs s0', density=True)
    
    ax6.set_xlabel('Percentage Change from s0 (%)')
    ax6.set_ylabel('Density')
    ax6.set_title('Fidelity Level Changes')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the analysis
    if save_dir is not None:
        # Ensure the directory exists
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "borehole_source_analysis.png")
    else:
        save_path = "borehole_source_analysis.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nAnalysis saved to: {save_path}")
    
    # Additional insights
    print("\nKey Insights:")
    print("-" * 40)
    
    # Find the source with highest/lowest outputs
    means = {source: outputs[source].mean().item() for source in sources}
    max_source = max(means, key=means.get)
    min_source = min(means, key=means.get)
    
    print(f"• Highest average output: {max_source} ({means[max_source]:.1f})")
    print(f"• Lowest average output: {min_source} ({means[min_source]:.1f})")
    
    # Calculate fidelity differences
    y_s0 = outputs['s0'].cpu().numpy()
    for source in sources[1:]:
        y_other = outputs[source].cpu().numpy()
        mean_diff = ((y_other.mean() - y_s0.mean()) / y_s0.mean()) * 100
        print(f"• {source} vs s0: {mean_diff:+.1f}% mean difference")
    
    # Variable importance
    print("\nVariable Importance (correlation with s0 output):")
    y_s0 = outputs['s0'].cpu().numpy()
    x_np = x_raw.cpu().numpy()
    for i in range(8):
        corr = np.corrcoef(x_np[:, i], y_s0)[0, 1]
        print(f"  {var_names[i]}: {corr:.3f}")
    
    plt.show()
    
    return outputs, correlations

#################################################################
# 2. Data Loading
#################################################################

def load_data_wing_MV_MF(
    seed: int,
    n_train: dict = {'s0': 200, 's1': 500, 's2': 1000},
    n_test: dict = {'s0': 50, 's1': 100, 's2': 200},
    noise_levels: list = [0.0, 0.01, 0.05],
    shuffle: bool = True,
    qual_dict: dict = {0: 10, 5: 10},
    return_one_hot: bool = True,
):
    """
    Generate multi-fidelity data with variable samples per source for both train and test
    
    Args:
        n_train (dict): Training samples per source (e.g., {'s0': 200, 's1': 500})
        n_test (dict): Test samples per source (same keys as n_train)
        Other params same as before
        
    Returns:
        dict: {
            'x_train_full': tensor,  # Concatenated features + source
            'y_train_full': tensor,
            'source_train_full': tensor,  # Source indices
            'noise_train_full': tensor,  # Noise levels
            'train_counts': dict,  # Actual counts per source
            'x_test_full': tensor,  # Concatenated test features + source
            'y_test_full': tensor,
            'source_test_full': tensor,
            'test_counts': dict,
            'metadata': {
                'x_dim': int,
                'source_dim': int,
                'source_names': list,
                'continuous_cols': list,
                'new_continuous_cols': list,
                'discrete_cols': list,
                'y_std': float,
                'noise_levels': list
            }
        }
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Setup sources
    fidelity_levels = list(n_train.keys())
    source_to_idx = {source: i for i, source in enumerate(fidelity_levels)}
    source_dim = len(fidelity_levels)
    
    # Get column types
    continuous_cols, discrete_cols = utils.get_column_types(qual_dict)
    num_continuous = len(continuous_cols)
    
    # Define bounds
    l_bound = torch.tensor([150., 220., 6., -10., 16., 0.5, 0.08, 2.5, 1700., 0.025])
    u_bound = torch.tensor([200., 300., 10., 10., 45., 1., 0.18, 6., 2500., 0.08])

    # Initialize Sobol engine
    sobol_engine = SobolEngine(dimension=10, scramble=True, seed=seed)

    def process_data(n_samples, is_train=True):
        x_full, y_full, source_full = [], [], []
        counts = {}
        
        for source, n in n_samples.items():
            # Generate raw features (n x 10)
            x_raw = utils.scale(sobol_engine.draw(n).float(), l_bound, u_bound)
            
            # Handle qualitative variables
            for col_idx, n_levels in qual_dict.items():
                levels = torch.linspace(l_bound[col_idx], u_bound[col_idx], steps=n_levels)
                x_raw[:, col_idx] = levels[torch.randint(0, n_levels, (n,))]
            # print('x_raw: ', x_raw.shape)
            # print(x_raw[0])
            # Process features
            if return_one_hot:
                # One-hot encoded categoricals (n x num_categorical)
                x_categorical = utils.one_hot_encoding(x_raw, qual_dict)
                # print('shape x_cat: ', x_categorical.shape)
                # Continuous features (n x 8)
                x_continuous = x_raw[:, continuous_cols]
                # print('shape x_cont: ', x_continuous.shape)
                # Combined processed features (n x (num_categorical + 8))

                # x_processed = torch.cat([x_categorical, x_continuous], dim=1).to(device)
                x_processed = torch.cat([x_categorical, x_continuous], dim=1)

                # print(x_processed.shape)
            else:
                x_processed = x_raw  # (n x 10)
                # x_processed = x_raw.to(device)  # (n x 10)

            # Source vector (n x source_dim)
            source_vec = torch.zeros(n, source_dim)
            # source_vec = torch.zeros(n, source_dim).to(device)
            source_vec[:, source_to_idx[source]] = 1

            # Get outputs
            if source == 's0':
                y = wing_mixed_variables(x_raw, source)
            elif source == 's1':
                y = wing_mixed_variables(x_raw, source)
                # x_processed[:, [1, 3, 4, 5, 6, 7]] = x_processed[:, [1, 3, 4, 5, 6, 7]].median(axis=0).values

            elif source == 's2':
                y = wing_mixed_variables(x_raw, source)
                # x_processed[:, [1, 3, 4, 5, 6, 7, 9]] = x_processed[:, [1, 3, 4, 5, 6, 7, 9]].median(axis=0).values

            elif source == 's3':
                y = wing_mixed_variables(x_raw, source)
                # x_processed[:, [1, 3, 4, 5, 6, 7, 9]] = x_processed[:, [1, 3, 4, 5, 6, 7, 9]].median(axis=0).values

            # For training: add noise variations
            noise = noise_levels[source_to_idx[source]] if source in noise_levels else 0.0
            if is_train:
                if noise > 0:
                    noisy_y = y + noise * torch.randn_like(y) * y.std()
                else:
                    noisy_y = y.clone()
                x_full.append(torch.cat([source_vec, x_processed], dim=1))
                y_full.append(noisy_y)
                source_full.append(torch.full((n,), source_to_idx[source]))
                counts[source] = n
            else:
                counts[source] = n
                x_full.append(torch.cat([source_vec, x_processed], dim=1))
                y_full.append(y)
                source_full.append(torch.full((n,), source_to_idx[source]))

        # Concatenate all
        # print(x_full.shape)
        x_full = torch.cat(x_full, dim=0)
        # x_full = torch.cat(x_full, dim=0).to(device)
        # print(x_full.shape)
        y_full = torch.cat(y_full, dim=0)
        # y_full = torch.cat(y_full, dim=0).to(device)
        source_full = torch.cat(source_full, dim=0)
        # source_full = torch.cat(source_full, dim=0).to(device)
        
        if is_train and shuffle:
            idx = torch.randperm(x_full.shape[0])
            x_full = x_full[idx]
            y_full = y_full[idx]
            source_full = source_full[idx]
            # idx = torch.randperm(x_full.shape[0]).to(device)
            # x_full = x_full[idx].to(device)
            # y_full = y_full[idx].to(device)
            # source_full = source_full[idx].to(device)
            
        return x_full, y_full, source_full, counts

    # Process data
    x_train, y_train, source_train, train_counts = process_data(n_train, True)
    x_test, y_test, source_test, test_counts = process_data(n_test, False)

    # Calculate y_std per source
    y_std_per_source = {}
    for source in fidelity_levels:
        source_idx = source_to_idx[source]
        mask = source_test == source_idx
        y_std_per_source[source] = y_test[mask].std().item()

    # Calculate column indices
    source_cols = list(range(source_dim))
    if return_one_hot:
        num_categorical = sum(qual_dict.values())
        categorical_cols = list(range(source_dim, source_dim + num_categorical))
        continuous_cols = list(range(source_dim + num_categorical, 
                                  source_dim + num_categorical + num_continuous))
        # print('\n', num_categorical, '\n', categorical_cols, '\n', continuous_cols)
    else:
        categorical_cols = []
        continuous_cols = list(range(source_dim, source_dim + num_continuous))

    # Verify tensor sizes
    expected_dim = source_dim + (num_categorical if return_one_hot else 0) + num_continuous
    assert x_train.shape[1] == expected_dim, (
        f"Feature dimension mismatch. Expected {expected_dim} features (source:{source_dim} + "
        f"cat:{num_categorical if return_one_hot else 0} + cont:{num_continuous}), "
        f"got {x_train.shape[1]}. Check one-hot encoding implementation."
    )

    return {
        'x_train_full': x_train,
        'y_train_full': y_train,
        'source_train_full': source_train,
        'noise_train_full': torch.cat([
            torch.full((n,), torch.tensor(noise_levels[i] if i < len(noise_levels) else 0.0))
            for i, (source, n) in enumerate(n_train.items())
        ]),
        'train_counts': train_counts,
        'x_test_full': x_test, 
        'y_test_full': y_test,
        'source_test_full': source_test,
        'test_counts': test_counts,
        'column_indices': {
            'original_columns': list(range(10)),
            'source': source_cols,
            'categorical': categorical_cols,
            'continuous': continuous_cols
        },
        'metadata': {
            'source_names': fidelity_levels,
            'y_std': y_std_per_source,  # Dictionary with y_std per source
            'expected_dim': expected_dim,
            'num_continuous': num_continuous,
            'num_categorical': len(categorical_cols),
            'num_source': len(source_cols),
            'input_dim': 10,  # Original input dimension before processing
            'noise_levels': noise_levels,

        }
    }

def load_data_buckling_MF(
    seed: int,
    n_train: dict = {'s0': 200, 's1': 500},
    n_test: dict = {'s0': 50, 's1': 100},
    noise_levels: list = [0.0, 0.01],
    shuffle: bool = True,
    qual_dict: dict = {1: 2, 2: 4, 3: 3},
    return_one_hot: bool = True,
):
    """
    Generate multi-fidelity data with variable samples per source for both train and test
    
    Args:
        n_train (dict): Training samples per source (e.g., {'s0': 200, 's1': 500})
        n_test (dict): Test samples per source (same keys as n_train)
        Other params same as before
        
    Returns:
        dict: {
            'x_train_full': tensor,  # Concatenated features + source
            'y_train_full': tensor,
            'source_train_full': tensor,  # Source indices
            'noise_train_full': tensor,  # Noise levels
            'train_counts': dict,  # Actual counts per source
            'x_test_full': tensor,  # Concatenated test features + source
            'y_test_full': tensor,
            'source_test_full': tensor,
            'test_counts': dict,
            'metadata': {
                'x_dim': int,
                'source_dim': int,
                'source_names': list,
                'continuous_cols': list,
                'new_continuous_cols': list,
                'discrete_cols': list,
                'y_std': float,
                'noise_levels': list
            }
        }
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Setup sources
    fidelity_levels = list(n_train.keys())
    source_to_idx = {source: i for i, source in enumerate(fidelity_levels)}
    source_dim = len(fidelity_levels)
    
    # Get column types from qual_dict
    continuous_cols, discrete_cols = utils.get_column_types(qual_dict, num_features=4)
    num_continuous = len(continuous_cols)
    
    # Define bounds
    l_bound = torch.tensor([0.5, 73.1, 0.5, 9.49])
    u_bound = torch.tensor([1.5, 200., 2., 29.5])
    
    # Define specific categorical values
    E_values = torch.tensor([73.1, 200.0])  # Column 1: E can only be 73.1 or 200
    K_values = torch.tensor([0.5, 0.7, 1.0, 2.0])  # Column 2: K can only be 0.5, 0.7, 1, or 2
    I_values = torch.tensor([9.49, 12.1, 29.5])  # Column 3: I can only be 12.1, 29.5, or 9.49
    
    # E_values = torch.tensor([73.1, 200.0])  # Column 1: E can only be 73.1 or 200
    # K_values = torch.tensor([0.5, 2.0])  # Column 2: K can only be 0.5 or 2
    # I_values = torch.tensor([9.49, 29.5])  # Column 3: I can only be 29.5 or 9.49

    # Initialize Sobol engine
    sobol_engine = SobolEngine(dimension=4, scramble=True, seed=seed)

    def process_data(n_samples, is_train=True):
        x_full, y_full, source_full = [], [], []
        counts = {}
        
        for source, n in n_samples.items():
            # Generate raw features (n x 4)
            x_raw = utils.scale(sobol_engine.draw(n).float(), l_bound, u_bound)
            
            # Handle qualitative variables using the specific categorical values
            # Ensure even distribution of categorical values
            
            # Column 1: E values (2 options)
            n_per_E = n // len(E_values)
            E_indices = torch.cat([
                torch.full((n_per_E,), i) for i in range(len(E_values))
            ])
            # Add any remaining samples
            if len(E_indices) < n:
                E_indices = torch.cat([E_indices, torch.randint(0, len(E_values), (n - len(E_indices),))])
            E_indices = E_indices[torch.randperm(n)]  # Shuffle
            x_raw[:, 1] = E_values[E_indices]
            
            # Column 2: K values (4 options)
            n_per_K = n // len(K_values)
            K_indices = torch.cat([
                torch.full((n_per_K,), i) for i in range(len(K_values))
            ])
            # Add any remaining samples
            if len(K_indices) < n:
                K_indices = torch.cat([K_indices, torch.randint(0, len(K_values), (n - len(K_indices),))])
            K_indices = K_indices[torch.randperm(n)]  # Shuffle
            x_raw[:, 2] = K_values[K_indices]
            
            # Column 3: I values (3 options)
            n_per_I = n // len(I_values)
            I_indices = torch.cat([
                torch.full((n_per_I,), i) for i in range(len(I_values))
            ])
            # Add any remaining samples
            if len(I_indices) < n:
                I_indices = torch.cat([I_indices, torch.randint(0, len(I_values), (n - len(I_indices),))])
            I_indices = I_indices[torch.randperm(n)]  # Shuffle
            x_raw[:, 3] = I_values[I_indices]
            
            # Print sample of raw data for verification
            # if source == 's0' and n == n_samples['s0']:
            #     print("\nSample of raw data before one-hot encoding:")
            #     print("First 5 rows:")
            #     print(x_raw[:5])
            #     print("\nUnique values in each categorical column:")
            #     print("E values:", torch.unique(x_raw[:, 1]))
            #     print("K values:", torch.unique(x_raw[:, 2]))
            #     print("I values:", torch.unique(x_raw[:, 3]))
            
            # Process features
            if return_one_hot:
                # One-hot encoded categoricals
                x_categorical = utils.one_hot_encoding(x_raw, qual_dict)
                # Continuous features
                x_continuous = x_raw[:, continuous_cols]
                # Combined processed features
                x_processed = torch.cat([x_categorical, x_continuous], dim=1)
                
                # Print sample of processed data for verification
                # if source == 's0' and n == n_samples['s0']:
                #     print("\nSample of processed data after one-hot encoding:")
                #     print("First 5 rows:")
                #     print(x_processed[:5])
                #     print("\nOne-hot encoding dimensions:")
                #     print("E encoding:", x_categorical[:, :2].sum(dim=0))
                #     print("K encoding:", x_categorical[:, 2:6].sum(dim=0))
                #     print("I encoding:", x_categorical[:, 6:].sum(dim=0))
            else:
                x_processed = x_raw

            # Source vector (n x source_dim)
            source_vec = torch.zeros(n, source_dim)
            source_vec[:, source_to_idx[source]] = 1

            # Get outputs
            if source == 's0':
                y = buckling_mixed_variables(x_raw, source)
            elif source == 's1':
                y = buckling_mixed_variables(x_raw, source)

            # For training: add noise variations
            noise = noise_levels[source_to_idx[source]] if source in noise_levels else 0.0
            if is_train:
                if noise > 0:
                    noisy_y = y + noise * torch.randn_like(y) * y.std()
                else:
                    noisy_y = y.clone()
                x_full.append(torch.cat([source_vec, x_processed], dim=1))
                y_full.append(noisy_y)
                source_full.append(torch.full((n,), source_to_idx[source]))
                counts[source] = n
            else:
                counts[source] = n
                x_full.append(torch.cat([source_vec, x_processed], dim=1))
                y_full.append(y)
                source_full.append(torch.full((n,), source_to_idx[source]))

        # Concatenate all
        x_full = torch.cat(x_full, dim=0)
        y_full = torch.cat(y_full, dim=0)
        source_full = torch.cat(source_full, dim=0)
        
        if is_train and shuffle:
            idx = torch.randperm(x_full.shape[0])
            x_full = x_full[idx]
            y_full = y_full[idx]
            source_full = source_full[idx]
            
        return x_full, y_full, source_full, counts

    # Process data
    x_train, y_train, source_train, train_counts = process_data(n_train, True)
    x_test, y_test, source_test, test_counts = process_data(n_test, False)

    # Calculate y_std per source
    y_std_per_source = {}
    for source in fidelity_levels:
        source_idx = source_to_idx[source]
        mask = source_test == source_idx
        y_std_per_source[source] = y_test[mask].std().item()

    # Calculate column indices for the processed data (after one-hot encoding)
    source_cols = list(range(source_dim))
    if return_one_hot:
        num_categorical = sum(qual_dict.values())
        categorical_cols = list(range(source_dim, source_dim + num_categorical))
        new_continuous_cols = list(range(source_dim + num_categorical, 
                                  source_dim + num_categorical + num_continuous))
    else:
        categorical_cols = []
        new_continuous_cols = list(range(source_dim, source_dim + num_continuous))

    # Verify tensor sizes
    expected_dim = source_dim + (num_categorical if return_one_hot else 0) + num_continuous
    assert x_train.shape[1] == expected_dim, (
        f"Feature dimension mismatch. Expected {expected_dim} features (source:{source_dim} + "
        f"cat:{num_categorical if return_one_hot else 0} + cont:{num_continuous}), "
        f"got {x_train.shape[1]}. Check one-hot encoding implementation."
    )

    return {
        'x_train_full': x_train,
        'y_train_full': y_train,
        'source_train_full': source_train,
        'noise_train_full': torch.cat([
            torch.full((n,), torch.tensor(noise_levels[i] if i < len(noise_levels) else 0.0))
            for i, (source, n) in enumerate(n_train.items())
        ]),
        'train_counts': train_counts,
        'x_test_full': x_test, 
        'y_test_full': y_test,
        'source_test_full': source_test,
        'test_counts': test_counts,
        'column_indices': {
            'original_columns': list(range(4)),
            'source': source_cols,
            'categorical': categorical_cols,
            'continuous': new_continuous_cols  # These are indices in the processed data
        },
        'metadata': {
            'source_names': fidelity_levels,
            'y_std': y_std_per_source,  # Dictionary with y_std per source
            'expected_dim': expected_dim,
            'num_continuous': num_continuous,
            'num_categorical': len(categorical_cols),
            'num_source': len(source_cols),
            'input_dim': 4,  # Original input dimension before processing
            'noise_levels': noise_levels,
            'E_values': E_values.tolist(),
            'K_values': K_values.tolist(),
            'I_values': I_values.tolist()
        }
    }

def load_data_borehole_MV_MF(
    seed: int,
    n_train: dict = {'s0': 200, 's1': 500, 's2': 1000, 's3': 2000, 's4': 4000},
    n_test: dict = {'s0': 50, 's1': 100, 's2': 200, 's3': 400, 's4': 800},
    noise_levels: list = [0.0, 0.01, 0.05, 0.1],
    shuffle: bool = True,
    qual_dict: dict = {},
    return_one_hot: bool = True,
):
    """
    Generate multi-fidelity borehole data with variable samples per source for both train and test
    
    Args:
        n_train (dict): Training samples per source (e.g., {'s0': 200, 's1': 500, 's2': 1000, 's3': 2000})
        n_test (dict): Test samples per source (same keys as n_train)
        Other params same as before
        
    Returns:
        dict: {
            'x_train_full': tensor,  # Concatenated features + source
            'y_train_full': tensor,
            'source_train_full': tensor,  # Source indices
            'noise_train_full': tensor,  # Noise levels
            'train_counts': dict,  # Actual counts per source
            'x_test_full': tensor,  # Concatenated test features + source
            'y_test_full': tensor,
            'source_test_full': tensor,
            'test_counts': dict,
            'metadata': {
                'x_dim': int,
                'source_dim': int,
                'source_names': list,
                'continuous_cols': list,
                'new_continuous_cols': list,
                'discrete_cols': list,
                'y_std': float,
                'noise_levels': list
            }
        }
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Setup sources
    fidelity_levels = list(n_train.keys())
    source_to_idx = {source: i for i, source in enumerate(fidelity_levels)}
    source_dim = len(fidelity_levels)
    
    # Get column types
    continuous_cols, discrete_cols = utils.get_column_types(qual_dict, num_features=8)
    num_continuous = len(continuous_cols)
    
    # Define bounds for borehole problem (8 variables)
    l_bound = torch.tensor([0.05, 100., 63070., 990., 63.1, 700., 1120., 9855.])
    u_bound = torch.tensor([0.15, 50000., 115600., 1110., 116., 820., 1680., 12045.])

    # Initialize Sobol engine
    sobol_engine = SobolEngine(dimension=8, scramble=True, seed=seed)

    def process_data(n_samples, is_train=True):
        x_full, y_full, source_full = [], [], []
        counts = {}
        
        for source, n in n_samples.items():
            # Generate raw features (n x 8)
            x_raw = utils.scale(sobol_engine.draw(n).float(), l_bound, u_bound)
            
            # Handle qualitative variables if any
            for col_idx, n_levels in qual_dict.items():
                levels = torch.linspace(l_bound[col_idx], u_bound[col_idx], steps=n_levels)
                x_raw[:, col_idx] = levels[torch.randint(0, n_levels, (n,))]
            
            # Process features
            if return_one_hot:
                # One-hot encoded categoricals (n x num_categorical)
                x_categorical = utils.one_hot_encoding(x_raw, qual_dict)
                # Continuous features (n x num_continuous)
                x_continuous = x_raw[:, continuous_cols]
                # Combined processed features
                x_processed = torch.cat([x_categorical, x_continuous], dim=1)
            else:
                x_processed = x_raw  # (n x 8)

            # Source vector (n x source_dim)
            source_vec = torch.zeros(n, source_dim)
            source_vec[:, source_to_idx[source]] = 1

            # Get outputs
            if source == 's0':
                y = borehole_mixed_variables(x_raw, source)
            elif source == 's1':
                y = borehole_mixed_variables(x_raw, source)
            elif source == 's2':
                y = borehole_mixed_variables(x_raw, source)
            elif source == 's3':
                y = borehole_mixed_variables(x_raw, source)
            elif source == 's4':
                y = borehole_mixed_variables(x_raw, source)

            # For training: add noise variations
            noise = noise_levels[source_to_idx[source]] if source in noise_levels else 0.0
            if is_train:
                if noise > 0:
                    noisy_y = y + noise * torch.randn_like(y) * y.std()
                else:
                    noisy_y = y.clone()
                x_full.append(torch.cat([source_vec, x_processed], dim=1))
                y_full.append(noisy_y)
                source_full.append(torch.full((n,), source_to_idx[source]))
                counts[source] = n
            else:
                counts[source] = n
                x_full.append(torch.cat([source_vec, x_processed], dim=1))
                y_full.append(y)
                source_full.append(torch.full((n,), source_to_idx[source]))

        # Concatenate all
        x_full = torch.cat(x_full, dim=0)
        y_full = torch.cat(y_full, dim=0)
        source_full = torch.cat(source_full, dim=0)
        
        if is_train and shuffle:
            idx = torch.randperm(x_full.shape[0])
            x_full = x_full[idx]
            y_full = y_full[idx]
            source_full = source_full[idx]
            
        return x_full, y_full, source_full, counts

    # Process data
    x_train, y_train, source_train, train_counts = process_data(n_train, True)
    x_test, y_test, source_test, test_counts = process_data(n_test, False)

    # Calculate y_std per source
    y_std_per_source = {}
    for source in fidelity_levels:
        source_idx = source_to_idx[source]
        mask = source_test == source_idx
        y_std_per_source[source] = y_test[mask].std().item()

    # Calculate column indices
    source_cols = list(range(source_dim))
    if return_one_hot:
        num_categorical = sum(qual_dict.values())
        categorical_cols = list(range(source_dim, source_dim + num_categorical))
        continuous_cols = list(range(source_dim + num_categorical, 
                                  source_dim + num_categorical + num_continuous))
    else:
        categorical_cols = []
        continuous_cols = list(range(source_dim, source_dim + num_continuous))

    # Verify tensor sizes
    expected_dim = source_dim + (num_categorical if return_one_hot else 0) + num_continuous
    assert x_train.shape[1] == expected_dim, (
        f"Feature dimension mismatch. Expected {expected_dim} features (source:{source_dim} + "
        f"cat:{num_categorical if return_one_hot else 0} + cont:{num_continuous}), "
        f"got {x_train.shape[1]}. Check one-hot encoding implementation."
    )

    return {
        'x_train_full': x_train,
        'y_train_full': y_train,
        'source_train_full': source_train,
        'noise_train_full': torch.cat([
            torch.full((n,), torch.tensor(noise_levels[i] if i < len(noise_levels) else 0.0))
            for i, (source, n) in enumerate(n_train.items())
        ]),
        'train_counts': train_counts,
        'x_test_full': x_test, 
        'y_test_full': y_test,
        'source_test_full': source_test,
        'test_counts': test_counts,
        'column_indices': {
            'original_columns': list(range(8)),
            'source': source_cols,
            'categorical': categorical_cols,
            'continuous': continuous_cols
        },
        'metadata': {
            'source_names': fidelity_levels,
            'y_std': y_std_per_source,  # Dictionary with y_std per source
            'expected_dim': expected_dim,
            'num_continuous': num_continuous,
            'num_categorical': len(categorical_cols),
            'num_source': len(source_cols),
            'input_dim': 8,  # Original input dimension before processing
            'noise_levels': noise_levels,
        }
    }

def get_data(problem: str, seed: int, save_dir=None, **kwargs):
    """
    Dispatches to the appropriate data-loading function based on `problem`.
    Extra keyword args (`**kwargs`) can be passed on to the loader if needed.
    
    Args:
        problem (str): Problem name ("wing_MV_MF", "buckling_MF", "borehole_MV_MF")
        seed (int): Random seed
        save_dir (str, optional): Directory to save analysis plots
        **kwargs: Additional arguments passed to the data loader
    """
    if problem == "wing_MV_MF":
        return load_data_wing_MV_MF(seed=seed, **kwargs)
    elif problem == "buckling_MF":
        # Analyze categorical ordering for buckling problem
        analyze_buckling_categorical_ordering()
        return load_data_buckling_MF(seed=seed, **kwargs)
    elif problem == "borehole_MV_MF":
        # Analyze source distributions for borehole problem
        analyze_borehole_source_distributions(save_dir=save_dir)
        return load_data_borehole_MV_MF(seed=seed, **kwargs)
    else:
        raise ValueError(f"Unknown problem: {problem}")