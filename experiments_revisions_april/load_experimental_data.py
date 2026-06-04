import torch
import os
import pandas as pd
import numpy as np
from math import sqrt
import torch.nn.functional as F
from sklearn.datasets import fetch_openml


def _sample_noise_like(
    reference: torch.Tensor,
    noise_scale: torch.Tensor | float,
    noise_type: str,
    student_t_df: float = 4.0,
) -> torch.Tensor:
    """Sample zero-mean noise with target std set by ``noise_scale``."""
    if noise_type == "gaussian":
        return torch.randn_like(reference) * noise_scale
    if noise_type == "uniform":
        return (torch.rand_like(reference) - 0.5) * 2 * noise_scale * sqrt(3)
    if noise_type in ("student_t", "student-t", "t"):
        if student_t_df <= 2.0:
            raise ValueError("student_t_df must be > 2.0 to have finite variance.")
        raw = torch.distributions.StudentT(df=student_t_df).sample(reference.shape)
        raw = raw.to(dtype=reference.dtype, device=reference.device)
        return (raw / sqrt(student_t_df / (student_t_df - 2.0))) * noise_scale
    raise ValueError(
        f"Unknown noise_type: {noise_type}. Use 'gaussian', 'uniform', or 'student_t'"
    )


def load_m2ax_data(print_info=False):
    """
    Load the M2AX dataset and optionally print detailed information about the data.
    Converts categorical element names to integers using label encoding.
    
    Args:
        print_info (bool): If True, prints detailed information about the loaded data
        
    Returns:
        tuple: (X, y) where X is features and y are targets
    """
    # Path to the data file (from one folder back, "data/data_M.csv")
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'data_M.csv')
    data_path = os.path.abspath(data_path)
    
    # Load the data
    df = pd.read_csv(data_path)

    # Label-encode the first three categorical columns before casting to float
    categorical_columns = df.columns[:3]
    df_encoded = df.copy()
    for col in categorical_columns:
        df_encoded[col] = pd.factorize(df[col])[0]

    # Convert to numpy array for processing
    arr = df_encoded.values.astype(np.float64)
    
    # Remove rows with any NaN values
    mask = ~np.isnan(arr).any(axis=1)
    arr = arr[mask]
    
    # Features: first 3 columns (3 encoded element names)
    X = torch.tensor(arr[:, :3], dtype=torch.float64)  # shape: (n_samples, 3)
    
    # Target: Bulk Modulus (column -3)
    y = torch.tensor(arr[:, -3], dtype=torch.float64)

    return X, y


def generate_hartmann_data(n_samples=10000, noise_level=0.0, noise_type='gaussian'):
    """
    Generate data for the 6D Hartmann function using Sobol sequences.
    
    The 6D Hartmann function is defined as:
    f(x) = -sum_{i=1}^{4} α_i * exp(-sum_{j=1}^{6} A_{ij} * (x_j - P_{ij})^2)
    
    where x ∈ [0,1]^6
    
    Args:
        n_samples (int): Number of samples to generate
        noise_level (float): Noise level as a fraction of the standard deviation of y
        noise_type (str): Type of noise ('gaussian' or 'uniform')
        
    Returns:
        X (torch.Tensor): Input samples of shape (n_samples, 6)
        y (torch.Tensor): Function values of shape (n_samples,)
    """
    # Hartmann 6D function constants
    alpha = torch.tensor([1.0, 1.2, 3.0, 3.2], dtype=torch.float64)
    
    A = torch.tensor([
        [10,   3,   17,  3.5, 1.7, 8],
        [0.05, 10,  17,  0.1, 8,   14],
        [3,    3.5, 1.7, 10,  17,  8],
        [17,   8,   0.05, 10,  0.1, 14]
    ], dtype=torch.float64)
    
    P = 1e-4 * torch.tensor([
        [1312, 1696, 5569, 124,  8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381]
    ], dtype=torch.float64)
    
    # Generate Sobol samples in [0,1]^6
    sobol = torch.quasirandom.SobolEngine(dimension=6, scramble=True)
    X = sobol.draw(n_samples).to(dtype=torch.float64)
    
    # Compute Hartmann function values using vectorized operations
    y = torch.zeros(n_samples, dtype=torch.float64)
    
    for k in range(4):  # 4 terms in the sum
        # Compute (x_j - P_{kj})^2 for all samples and dimensions
        diff = X - P[k:k+1, :]  # Broadcasting: (n_samples, 6) - (1, 6)
        squared_diff = diff ** 2  # (n_samples, 6)
        
        # Compute sum over dimensions: sum_j A_{kj} * (x_j - P_{kj})^2
        inner_sum = torch.sum(A[k:k+1, :] * squared_diff, dim=1)  # (n_samples,)
        
        # Add alpha_k * exp(-inner_sum) to the total
        y += alpha[k] * torch.exp(-inner_sum)
    
    # Apply negative sign
    y = -y
    
    # Add noise if specified
    if noise_level > 0:
        y_std = y.std()
        noise_scale = noise_level * y_std
        
        noise = _sample_noise_like(y, noise_scale, noise_type)
        
        y = y + noise
    
    return X, y


def load_am_dataset_data(print_info=False):
    """
    Load the AM dataset from the .pt file.
    
    Args:
        print_info (bool): If True, prints detailed information about the loaded data
        
    Returns:
        tuple: (X, y_porosity, y_hardness) where X is features and y are targets
    """
    # Path to the data file (from one folder back, "data/am_data.pt")
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'am_data.pt')
    data_path = os.path.abspath(data_path)
    
    # Load the data
    data = torch.load(data_path)
    
    arr = data
    mask = ~torch.isnan(arr).any(dim=1)
    
    # Apply mask to filter out all rows with any NaN
    arr = arr[mask]
    
    X = arr[:, :6]   # shape: (540, 6) or (539, 6) after removing NaN sample
    
    # Targets: column 6 = Porosity, column 7 = Hardness
    y_porosity = arr[:, 6]
    y_hardness = arr[:, 7]
    
    return X, y_porosity, y_hardness


def wing_mixed_variables(X: torch.Tensor, source: str = "s0") -> torch.Tensor:
    """
    Local copy of the wing function with source-specific variants.
    X shape: (n, 10)
    """
    Sw = X[..., 0]
    Wfw = X[..., 1]
    A = X[..., 2]
    Gama = X[..., 3] * (torch.pi / 180.0)
    q = X[..., 4]
    lamb = X[..., 5]
    tc = X[..., 6]
    Nz = X[..., 7]
    Wdg = X[..., 8]
    Wp = X[..., 9]
    cos_Gama = torch.cos(Gama)

    if source == "s0":
        return (
            0.036
            * Sw**0.758
            * Wfw**0.0035
            * (A / (cos_Gama) ** 2) ** 0.6
            * q**0.006
            * lamb**0.04
            * ((100 * tc) / (cos_Gama)) ** (-0.3)
            * (Nz * Wdg) ** 0.49
            + Sw * Wp
        )
    elif source == "s1":
        return (
            0.036
            * Sw**0.758
            * Wfw**0.0035
            * (A / (cos_Gama) ** 2) ** 0.6
            * q**0.006
            * lamb**0.04
            * ((100 * tc) / (cos_Gama)) ** (-0.3)
            * (Nz * Wdg) ** 0.49
            + 1 * Wp
        )
    elif source == "s2":
        return (
            0.036
            * Sw**0.8
            * Wfw**0.0035
            * (A / (cos_Gama) ** 2) ** 0.6
            * q**0.006
            * lamb**0.04
            * ((100 * tc) / (cos_Gama)) ** (-0.3)
            * (Nz * Wdg) ** 0.49
            + 1 * Wp
        )
    elif source == "s3":
        return (
            0.036
            * Sw**0.9
            * Wfw**0.0035
            * (A / (cos_Gama) ** 2) ** 0.6
            * q**0.006
            * lamb**0.04
            * ((100 * tc) / (cos_Gama)) ** (-0.3)
            * (Nz * Wdg) ** 0.49
            + 0 * Wp
        )
    else:
        raise ValueError(f"Unknown source: {source}")


def generate_mf_wing_data(train_samples_per_source: list[int], test_samples_per_source: list[int], 
                         seed: int = None, train_noise: list[float] = None, test_noise: list[float] = None, 
                         noise_type: str = 'gaussian') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate multi-fidelity Wing data by drawing a single Sobol batch (no repeats),
    then splitting into test/train per source. Compute test std after the split
    and scale both train and test noise by that std.

    Returns:
      - X_train, y_train: Training data with 11 features (10 continuous + 1 source class column in {0,1,2,3})
      - X_test, y_test: Test data with 11 features (10 continuous + 1 source class column in {0,1,2,3})
    """
    if seed is not None:
        torch.manual_seed(seed)
    else:
        seed = 42
        torch.manual_seed(seed)

    # Defaults and validation for per-source noise (4 sources)
    if train_noise is None:
        train_noise = [0.0, 0.0, 0.0, 0.0]
    if test_noise is None:
        test_noise = [0.0, 0.0, 0.0, 0.0]
    if isinstance(train_noise, (int, float)):
        train_noise = [float(train_noise)] * 4
    if isinstance(test_noise, (int, float)):
        test_noise = [float(test_noise)] * 4
    if len(train_noise) != 4 or len(test_noise) != 4:
        raise ValueError("train_noise and test_noise must be length-4 (one scalar per source)")

    sources = ["s0", "s1", "s2", "s3"]

    # Bounds for the 10 continuous features
    l_bound = torch.tensor([150.0, 220.0, 6.0, -10.0, 16.0, 0.5, 0.08, 2.5, 1700.0, 0.025], dtype=torch.float64)
    u_bound = torch.tensor([200.0, 300.0, 10.0, 10.0, 45.0, 1.0, 0.18, 6.0, 2500.0, 0.08], dtype=torch.float64)

    # Total samples per source and overall
    total_per_source = [tr + te for tr, te in zip(train_samples_per_source, test_samples_per_source)]
    total_n = sum(total_per_source)

    # Draw all Sobol samples at once (scrambled => randomized QMC) and scale to bounds
    sobol = torch.quasirandom.SobolEngine(dimension=10, scramble=True, seed=seed)
    X_raw_all = sobol.draw(total_n).to(dtype=torch.float64)
    X_raw_all = X_raw_all * (u_bound - l_bound) + l_bound

    # Assign contiguous blocks per source (no repeats globally)
    src_indices = []
    start = 0
    for idx, n in enumerate(total_per_source):
        src_indices.extend([idx] * n)
        start += n
    src_indices_tensor = torch.tensor(src_indices, dtype=torch.long)

    # Compute clean targets once per source
    y_clean_all = torch.empty(total_n, dtype=torch.float64)
    offset = 0
    for idx, (src, n) in enumerate(zip(sources, total_per_source)):
        if n == 0:
            continue
        x_block = X_raw_all[offset:offset + n]
        y_clean_all[offset:offset + n] = wing_mixed_variables(x_block, source=src)
        offset += n

    # Split per source into test then train; get test std after split
    X_train_list: list[torch.Tensor] = []
    y_train_list: list[torch.Tensor] = []
    X_test_list: list[torch.Tensor] = []
    y_test_list: list[torch.Tensor] = []

    cursor = 0
    for idx, (src, n_total, n_test, n_train) in enumerate(
        zip(sources, total_per_source, test_samples_per_source, train_samples_per_source)
    ):
        if n_total == 0:
            continue
        x_block = X_raw_all[cursor:cursor + n_total]
        y_block = y_clean_all[cursor:cursor + n_total]

        # Split: first n_test -> test, remaining -> train
        x_test_block = x_block[:n_test] if n_test > 0 else torch.empty((0, 10), dtype=torch.float64)
        y_test_block = y_block[:n_test] if n_test > 0 else torch.empty((0,), dtype=torch.float64)
        x_train_block = x_block[n_test:] if n_train > 0 else torch.empty((0, 10), dtype=torch.float64)
        y_train_block = y_block[n_test:] if n_train > 0 else torch.empty((0,), dtype=torch.float64)

        # Test std after split (per source) as a Python float
        test_std_value: float
        if y_test_block.numel() > 1:
            test_std_value = float(y_test_block.std().item())
        else:
            test_std_value = 0.0

        # Apply noise scaled by test std
        if n_train > 0 and train_noise[idx] > 0 and test_std_value > 0.0:
            noise = _sample_noise_like(
                y_train_block, train_noise[idx] * test_std_value, noise_type
            )
            y_train_block = y_train_block + noise

        if n_test > 0 and test_noise[idx] > 0 and test_std_value > 0.0:
            noise = _sample_noise_like(
                y_test_block, test_noise[idx] * test_std_value, noise_type
            )
            y_test_block = y_test_block + noise

        # Append source id as 11th feature
        if n_train > 0:
            src_col_train = torch.full((n_train, 1), float(idx), dtype=torch.float64)
            X_train_list.append(torch.cat([x_train_block, src_col_train], dim=1))
            y_train_list.append(y_train_block)

        if n_test > 0:
            src_col_test = torch.full((n_test, 1), float(idx), dtype=torch.float64)
            X_test_list.append(torch.cat([x_test_block, src_col_test], dim=1))
            y_test_list.append(y_test_block)

        cursor += n_total

    X_train = torch.cat(X_train_list, dim=0) if X_train_list else torch.empty((0, 11), dtype=torch.float64)
    y_train = torch.cat(y_train_list, dim=0) if y_train_list else torch.empty((0,), dtype=torch.float64)
    X_test = torch.cat(X_test_list, dim=0) if X_test_list else torch.empty((0, 11), dtype=torch.float64)
    y_test = torch.cat(y_test_list, dim=0) if y_test_list else torch.empty((0,), dtype=torch.float64)

    return X_train, y_train, X_test, y_test


def load_2dplanes_data(print_info=False):
    """
    Load the planes dataset and optionally print detailed information about the data.
    Converts categorical element names to integers using label encoding.
    
    Args:
        print_info (bool): If True, prints detailed information about the loaded data
        
    Returns:
        tuple: (X, y) where X is features and y are targets
    """
    # Path to the preferred CSV and a fallback TSV if CSV is missing
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    csv_path = os.path.join(data_dir, 'data_2dplanes.csv')
    tsv_path = os.path.join(data_dir, '215_2dplanes.tsv')

    # Load the data with graceful fallback
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
    elif os.path.isfile(tsv_path):
        df = pd.read_csv(tsv_path, sep='\t', header=None)
    else:
        raise FileNotFoundError(f"Could not find 2dplanes data. Expected one of: {csv_path} or {tsv_path}")
    

    # Convert to numpy array for processing; if header-like first row exists, drop it
    try:
        arr = df.values.astype(np.float64)
    except ValueError:
        # Drop the first row and retry conversion
        df = df.iloc[1:].reset_index(drop=True)
        arr = df.values.astype(np.float64)
    
    
    X = torch.tensor(arr[:, :-1], dtype=torch.float64)  # shape: (n_samples, 10)
    

    y = torch.tensor(arr[:, -1], dtype=torch.float64)
    
    return X, y


def load_pumadyn32_data(print_info: bool = False):
    """
    Load DELVE/OpenML puma32H data.

    Args:
        print_info (bool): If True, print feature/target shapes.

    Returns:
        tuple: (X, y) as torch.float64 tensors.
    """
    dataset = fetch_openml(name="puma32H", version=1, as_frame=True)
    X_df = dataset.data.copy()
    y_series = dataset.target

    X = torch.tensor(X_df.to_numpy(dtype=np.float64), dtype=torch.float64)
    y = torch.tensor(y_series.to_numpy(dtype=np.float64), dtype=torch.float64)

    if print_info:
        print(f"[pumadyn32] X shape: {tuple(X.shape)}, y shape: {tuple(y.shape)}")
        print("[pumadyn32] Using all 32 input columns")

    return X, y


def load_elevators_data(print_info: bool = False):
    """
    Load the elevators regression dataset from OpenML.

    Args:
        print_info (bool): If True, print feature/target shapes.

    Returns:
        tuple: (X, y) as torch.float64 tensors.
    """
    dataset = fetch_openml(name="elevators", version=1, as_frame=True)
    X_df = dataset.data.copy()
    y_series = dataset.target

    X = torch.tensor(X_df.to_numpy(dtype=np.float64), dtype=torch.float64)
    y = torch.tensor(y_series.to_numpy(dtype=np.float64), dtype=torch.float64)

    if print_info:
        print(f"[elevators] X shape: {tuple(X.shape)}, y shape: {tuple(y.shape)}")
        print("[elevators] Using all input columns")

    return X, y


def generate_pumadyn32_train_test_data(
    train_samples: int,
    seed: int | None = None,
    split_seed: int = 0,
    train_pool_size: int = 5192,
    test_pool_size: int = 3000,
    print_info: bool = False,
):
    """
    Load puma32H with a fixed pool split, then sample train points per run.

    Args:
        train_samples (int): Number of training samples to draw from the fixed train pool.
        seed (int | None): Random seed for per-run training subset selection.
        split_seed (int): Seed used once to create the fixed train/test pools.
        train_pool_size (int): Size of the fixed training pool.
        test_pool_size (int): Size of the fixed test pool.
        print_info (bool): If True, print split information.

    Returns:
        tuple: (X_train, y_train, X_test, y_test) as torch.float64 tensors.
    """
    if train_samples < 0:
        raise ValueError("train_samples must be non-negative.")
    if train_pool_size < 0 or test_pool_size < 0:
        raise ValueError("train_pool_size and test_pool_size must be non-negative.")

    X, y = load_pumadyn32_data(print_info=False)
    total_available = X.shape[0]
    total_pool = train_pool_size + test_pool_size

    if total_pool > total_available:
        raise ValueError(
            f"Requested pool size {total_pool} (train_pool+test_pool), but only "
            f"{total_available} are available in puma32H."
        )
    if train_samples > train_pool_size:
        raise ValueError(
            f"train_samples ({train_samples}) cannot exceed train_pool_size ({train_pool_size})."
        )

    # Fixed split created once by split_seed: first test_pool_size as test, next as train pool.
    split_generator = torch.Generator()
    split_generator.manual_seed(split_seed)
    split_perm = torch.randperm(total_available, generator=split_generator)
    pool_idx = split_perm[:total_pool]
    test_pool_idx = pool_idx[:test_pool_size]
    train_pool_idx = pool_idx[test_pool_size:]

    # Per-run training subset from the fixed train pool.
    if seed is not None:
        train_generator = torch.Generator()
        train_generator.manual_seed(seed)
        train_subperm = torch.randperm(train_pool_size, generator=train_generator)
    else:
        train_subperm = torch.randperm(train_pool_size)
    train_idx = train_pool_idx[train_subperm[:train_samples]]

    X_test = X[test_pool_idx]
    y_test = y[test_pool_idx]
    X_train = X[train_idx]
    y_train = y[train_idx]

    if print_info:
        print(
            f"[pumadyn32] Fixed pools -> train_pool: {train_pool_size}, "
            f"test_pool: {test_pool_size}, features: {X.shape[1]}"
        )
        print(
            f"[pumadyn32] Current run -> train subset: {X_train.shape[0]}, "
            f"test: {X_test.shape[0]}, split_seed: {split_seed}, run_seed: {seed}"
        )

    return X_train, y_train, X_test, y_test


def generate_elevators_train_test_data(
    train_samples: int,
    seed: int | None = None,
    split_seed: int = 0,
    train_pool_size: int = 13599,
    test_pool_size: int = 3000,
    print_info: bool = False,
):
    """
    Load elevators with a fixed pool split, then sample train points per run.

    Args:
        train_samples (int): Number of training samples to draw from the fixed train pool.
        seed (int | None): Random seed for per-run training subset selection.
        split_seed (int): Seed used once to create the fixed train/test pools.
        train_pool_size (int): Size of the fixed training pool.
        test_pool_size (int): Size of the fixed test pool.
        print_info (bool): If True, print split information.

    Returns:
        tuple: (X_train, y_train, X_test, y_test) as torch.float64 tensors.
    """
    if train_samples < 0:
        raise ValueError("train_samples must be non-negative.")
    if train_pool_size < 0 or test_pool_size < 0:
        raise ValueError("train_pool_size and test_pool_size must be non-negative.")

    X, y = load_elevators_data(print_info=False)
    total_available = X.shape[0]
    total_pool = train_pool_size + test_pool_size

    if total_pool > total_available:
        raise ValueError(
            f"Requested pool size {total_pool} (train_pool+test_pool), but only "
            f"{total_available} are available in elevators."
        )
    if train_samples > train_pool_size:
        raise ValueError(
            f"train_samples ({train_samples}) cannot exceed train_pool_size ({train_pool_size})."
        )

    # Fixed split created once by split_seed: first test_pool_size as test, next as train pool.
    split_generator = torch.Generator()
    split_generator.manual_seed(split_seed)
    split_perm = torch.randperm(total_available, generator=split_generator)
    pool_idx = split_perm[:total_pool]
    test_pool_idx = pool_idx[:test_pool_size]
    train_pool_idx = pool_idx[test_pool_size:]

    # Per-run training subset from the fixed train pool.
    if seed is not None:
        train_generator = torch.Generator()
        train_generator.manual_seed(seed)
        train_subperm = torch.randperm(train_pool_size, generator=train_generator)
    else:
        train_subperm = torch.randperm(train_pool_size)
    train_idx = train_pool_idx[train_subperm[:train_samples]]

    X_test = X[test_pool_idx]
    y_test = y[test_pool_idx]
    X_train = X[train_idx]
    y_train = y[train_idx]

    if print_info:
        print(
            f"[elevators] Fixed pools -> train_pool: {train_pool_size}, "
            f"test_pool: {test_pool_size}, features: {X.shape[1]}"
        )
        print(
            f"[elevators] Current run -> train subset: {X_train.shape[0]}, "
            f"test: {X_test.shape[0]}, split_seed: {split_seed}, run_seed: {seed}"
        )

    return X_train, y_train, X_test, y_test


def buckling_mixed_variables(X: torch.Tensor, source: str = "s0") -> torch.Tensor:
    """
    Compute buckling load given input variables.
    
    Args:
        X (torch.Tensor): Input array of shape [n_samples, 4] with columns:
            0: L (length of the beam, m)
            1: E (Young's modulus, Pa) 
            2: K (shear modulus, Pa)
            3: I (moment of inertia, m^4)
        source (str): Source of the data ('s0' or 's1')
    Returns:
        torch.Tensor: Buckling load values for each input sample
    """
    L = X[..., 0]
    E = X[..., 1]
    K = X[..., 2]
    I = X[..., 3]

    # Buckling load calculation
    if source == "s0":
        P = torch.pi * E * I / (L * K) ** 2
    elif source == "s1":
        P = ((torch.pi * E * I / (L * K) ** 2) + L) ** 1.1
    else:
        raise ValueError(f"Unknown source: {source}. Only 's0' and 's1' are supported for buckling.")

    return P


def generate_mf_buckling_data_with_folds(train_samples_per_source: list[int], test_samples_per_source: list[int], 
                                         num_runs: int = 4, seed: int = None, train_noise: list[float] = None, 
                                         test_noise: list[float] = None, noise_type: str = 'gaussian', 
                                         return_categorical: bool = True) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Generate multi-fidelity Buckling data with pre-stratified folds:
      - Use Sobol sequences to produce EVEN amounts of E, I, and K categorical inputs
      - Generate train data directly as num_runs with even categorical distributions
      - Generate test data with even categorical distributions
      - Each fold has perfectly balanced categorical distributions
    """
    if seed is not None:
        torch.manual_seed(seed)
    else:
        seed = torch.randint(0, 1000000)
        torch.manual_seed(seed)
    
    # Default noise values
    if train_noise is None:
        train_noise = [0.0] * len(train_samples_per_source)
    if test_noise is None:
        test_noise = [0.0] * len(test_samples_per_source)
    
    # Validate inputs
    if len(train_samples_per_source) != len(test_samples_per_source):
        raise ValueError("train_samples_per_source and test_samples_per_source must have same length")
    if len(train_noise) != len(train_samples_per_source):
        raise ValueError("train_noise must be length-2 (one scalar per source)")
    if len(test_noise) != len(test_samples_per_source):
        raise ValueError("test_noise must be length-2 (one scalar per source)")
    
    sources = ['s0', 's1']  # Two sources
    
    # Categorical index values (0-based) and actual physical values
    # Use strictly positive physical values to avoid 0/0 or division-by-zero in buckling formula
    E_values = torch.tensor([0, 1], dtype=torch.long)  # category indices
    K_values = torch.tensor([0, 1, 2, 3], dtype=torch.long)  # category indices
    I_values = torch.tensor([0, 1, 2], dtype=torch.long)  # category indices
    # Actual physical values for buckling problem
    E_phys = torch.tensor([73.1, 200.0], dtype=torch.float64)  # Young's modulus values
    K_phys = torch.tensor([0.5, 0.7, 1.0, 2.0], dtype=torch.float64)  # Shear modulus values
    I_phys = torch.tensor([9.49, 12.1, 29.5], dtype=torch.float64)  # Moment of inertia values
    
    # Generate all continuous L values per source at once (test + train) using single seed per source
    # This matches the pattern used in other problems: draw all samples for a source at once, then split
    # Use seed offset per source to ensure each source gets a unique sequence
    L_vals_per_source = {}
    for src_idx, (n_test, n_train) in enumerate(zip(test_samples_per_source, train_samples_per_source)):
        total_n = n_test + n_train
        if total_n == 0:
            continue
        # Use seed offset per source to get unique sequences (but consistent within source)
        # This ensures test and train are contiguous within each source's sequence
        sobol_seed = (seed + src_idx * 1000) if seed is not None else None
        sobol = torch.quasirandom.SobolEngine(1, scramble=True, seed=sobol_seed)
        # Draw all samples for this source at once (test + train together)
        L_vals_all = sobol.draw(total_n).squeeze() + 0.5  # L in [0.5, 1.5]
        L_vals_per_source[src_idx] = L_vals_all
    
    # Generate all data (test + train) per source, compute targets once, then split
    X_test_list = []
    y_test_list = []
    X_train_folds = []
    y_train_folds = []
    test_std_per_source = {}  # Store test std for each source
    
    total_train_samples = sum(train_samples_per_source)
    if total_train_samples > 0:
        # Pre-allocate lists for all folds
        for _ in range(num_runs):
            X_train_folds.append([])
            y_train_folds.append([])
    
    for src_idx, (src, n_test, n_train) in enumerate(zip(sources, test_samples_per_source, train_samples_per_source)):
        total_n = n_test + n_train
        if total_n == 0:
            continue
        
        L_vals_all = L_vals_per_source[src_idx]
        all_cat_assignments = []
        
        # Generate test categorical assignments (even distribution)
        if n_test > 0:
            # E values (2 options)
            n_per_E_test = n_test // len(E_values)
            remaining_E_test = n_test % len(E_values)
            E_indices_test = []
            for i in range(len(E_values)):
                count = n_per_E_test + (1 if i < remaining_E_test else 0)
                E_indices_test.append(torch.full((count,), i))
            E_indices_test = torch.cat(E_indices_test)
            E_indices_test = E_indices_test[torch.randperm(n_test)]
            
            # K values (4 options)
            n_per_K_test = n_test // len(K_values)
            remaining_K_test = n_test % len(K_values)
            K_indices_test = []
            for i in range(len(K_values)):
                count = n_per_K_test + (1 if i < remaining_K_test else 0)
                K_indices_test.append(torch.full((count,), i))
            K_indices_test = torch.cat(K_indices_test)
            K_indices_test = K_indices_test[torch.randperm(n_test)]
            
            # I values (3 options)
            n_per_I_test = n_test // len(I_values)
            remaining_I_test = n_test % len(I_values)
            I_indices_test = []
            for i in range(len(I_values)):
                count = n_per_I_test + (1 if i < remaining_I_test else 0)
                I_indices_test.append(torch.full((count,), i))
            I_indices_test = torch.cat(I_indices_test)
            I_indices_test = I_indices_test[torch.randperm(n_test)]
            
            # Store test assignments
            for i in range(n_test):
                all_cat_assignments.append({
                    'e': int(E_indices_test[i].item()),
                    'k': int(K_indices_test[i].item()),
                    'i': int(I_indices_test[i].item())
                })
        
        # Generate train categorical assignments (per fold with exact distributions)
        if n_train > 0:
            target_per_fold = n_train // num_runs
            remainder = n_train % num_runs
            num_E = len(E_values)
            num_K = len(K_values)
            num_I = len(I_values)
            
            for fold in range(num_runs):
                fold_target = target_per_fold + (1 if fold < remainder else 0)
                
                # Calculate EXACT counts for each categorical value in this fold
                E_base = fold_target // num_E
                E_rem = fold_target % num_E
                E_counts = [E_base + (1 if i < E_rem else 0) for i in range(num_E)]
                
                K_base = fold_target // num_K
                K_rem = fold_target % num_K
                K_counts = [K_base + (1 if i < K_rem else 0) for i in range(num_K)]
                
                I_base = fold_target // num_I
                I_rem = fold_target % num_I
                rotation_offset = (fold + src_idx * num_runs) % num_I
                I_counts = [I_base + (1 if (i + rotation_offset) % num_I < I_rem else 0) for i in range(num_I)]
                
                # Build assignments for this fold
                cat_assignments = []
                for e_idx in range(num_E):
                    for _ in range(E_counts[e_idx]):
                        cat_assignments.append({'e': e_idx})
                
                k_list = []
                for k_idx in range(num_K):
                    for _ in range(K_counts[k_idx]):
                        k_list.append(k_idx)
                if seed is not None:
                    torch.manual_seed(seed + src_idx * 2000 + fold * 100 + 1)
                k_perm = torch.randperm(len(k_list))
                k_list = [k_list[i] for i in k_perm.tolist()]
                
                i_list = []
                for i_idx in range(num_I):
                    for _ in range(I_counts[i_idx]):
                        i_list.append(i_idx)
                if seed is not None:
                    torch.manual_seed(seed + src_idx * 2000 + fold * 100 + 2)
                i_perm = torch.randperm(len(i_list))
                i_list = [i_list[i] for i in i_perm.tolist()]
                
                for i in range(fold_target):
                    cat_assignments[i]['k'] = k_list[i]
                    cat_assignments[i]['i'] = i_list[i]
                
                if seed is not None:
                    torch.manual_seed(seed + src_idx * 2000 + fold * 100 + 3)
                perm = torch.randperm(len(cat_assignments))
                cat_assignments = [cat_assignments[i] for i in perm.tolist()]
                
                all_cat_assignments.extend(cat_assignments)
        
        # Build all data (test + train) for this source at once
        x_all = torch.zeros((total_n, 4), dtype=torch.float64)
        for i, assignment in enumerate(all_cat_assignments):
            x_all[i, 0] = L_vals_all[i]
            x_all[i, 1] = E_phys[assignment['e']]
            x_all[i, 2] = K_phys[assignment['k']]
            x_all[i, 3] = I_phys[assignment['i']]
        
        # Compute targets ONCE for all data (test + train)
        y_all = buckling_mixed_variables(x_all, source=src)
        
        # Convert to categorical indices if requested (AFTER computing y)
        if return_categorical:
            for i, assignment in enumerate(all_cat_assignments):
                x_all[i, 1] = float(assignment['e'])
                x_all[i, 2] = float(assignment['k'])
                x_all[i, 3] = float(assignment['i'])
        
        # Split into test and train
        if n_test > 0:
            x_test_block = x_all[:n_test]
            y_test_clean = y_all[:n_test]
            
            # Compute test std for noise scaling
            if y_test_clean.numel() > 1:
                test_std_value = float(y_test_clean.std().item())
            else:
                test_std_value = 0.0
            test_std_per_source[src_idx] = test_std_value
            
            # Add noise to test data
            y_test_block = y_test_clean.clone()
            if test_noise[src_idx] > 0 and test_std_value > 0.0:
                noise = _sample_noise_like(
                    y_test_block, test_noise[src_idx] * test_std_value, noise_type
                )
                y_test_block = y_test_block + noise
            
            source_column = torch.full((x_test_block.shape[0], 1), src_idx, dtype=torch.float64)
            X_test_list.append(torch.cat([x_test_block, source_column], dim=1))
            y_test_list.append(y_test_block)
        
        if n_train > 0:
            x_train_all = x_all[n_test:]
            y_train_all = y_all[n_test:]
            
            # Add noise to train data
            test_std_value = test_std_per_source.get(src_idx, 0.0)
            if train_noise[src_idx] > 0 and test_std_value > 0.0:
                noise = _sample_noise_like(
                    y_train_all, train_noise[src_idx] * test_std_value, noise_type
                )
                y_train_all = y_train_all + noise
            
            # Split train into folds
            target_per_fold = n_train // num_runs
            remainder = n_train % num_runs
            fold_start = 0
            for fold in range(num_runs):
                fold_target = target_per_fold + (1 if fold < remainder else 0)
                fold_end = fold_start + fold_target
                
                x_fold = x_train_all[fold_start:fold_end]
                y_fold = y_train_all[fold_start:fold_end]
                
                source_column = torch.full((x_fold.shape[0], 1), src_idx, dtype=torch.float64)
                x_fold_with_source = torch.cat([x_fold, source_column], dim=1)
                
                X_train_folds[fold].append(x_fold_with_source)
                y_train_folds[fold].append(y_fold)
                
                fold_start = fold_end
    
    # Combine test data
    X_test_all = torch.cat(X_test_list, dim=0)
    y_test_all = torch.cat(y_test_list, dim=0)
    
    # Concatenate folds from all sources
    for fold in range(num_runs):
        X_train_folds[fold] = torch.cat(X_train_folds[fold], dim=0)
        y_train_folds[fold] = torch.cat(y_train_folds[fold], dim=0)
    
    return X_train_folds, y_train_folds, X_test_all, y_test_all


def generate_mf_buckling_data(train_samples_per_source: list[int], test_samples_per_source: list[int], 
                              seed: int = None, train_noise: list[float] = None, test_noise: list[float] = None, 
                              noise_type: str = 'gaussian', return_categorical: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate multi-fidelity Buckling data following the data_gen.py method:
      - Use Sobol sequences to produce EVEN amounts of E, I, and K categorical inputs
      - Draw a single Sobol batch per source (no repeats globally)
      - Split into test then train per source
      - Compute per-source test std after the split
      - Scale both train and test additive noise by that test std

    Returns:
      - X_train, y_train: Training data with 5 features (4 continuous + 1 source class column in {0,1})
        Where columns are [L (cont), E, K, I (categorical or values per return_categorical), source]
      - X_test, y_test: Test data with 5 features (same schema as X_train)
    """
    if seed is not None:
        torch.manual_seed(seed)
    # else:
    #     seed = 42
    #     torch.manual_seed(seed)

    # Defaults and validation for per-source noise (2 sources)
    if train_noise is None:
        train_noise = [0.0, 0.0]
    if test_noise is None:
        test_noise = [0.0, 0.0]
    if isinstance(train_noise, (int, float)):
        train_noise = [float(train_noise)] * 2
    if isinstance(test_noise, (int, float)):
        test_noise = [float(test_noise)] * 2
    if len(train_noise) != 2 or len(test_noise) != 2:
        raise ValueError("train_noise and test_noise must be length-2 (one scalar per source)")

    sources = ["s0", "s1"]

    # Bounds for the 4 continuous features (L, E, K, I)
    l_bound = torch.tensor([0.5, 73.1, 0.5, 9.49], dtype=torch.float64)
    u_bound = torch.tensor([1.5, 200.0, 2.0, 29.5], dtype=torch.float64)

    # Define specific categorical values (same as data_gen.py)
    E_values = torch.tensor([73.1, 200.0], dtype=torch.float64)  # Column 1: E can only be 73.1 or 200
    K_values = torch.tensor([0.5, 0.7, 1.0, 2.0], dtype=torch.float64)  # Column 2: K can only be 0.5, 0.7, 1, or 2
    I_values = torch.tensor([9.49, 12.1, 29.5], dtype=torch.float64)  # Column 3: I can only be 9.49, 12.1, or 29.5

    # Total samples per source and overall
    total_per_source = [tr + te for tr, te in zip(train_samples_per_source, test_samples_per_source)]
    total_n = sum(total_per_source)

    # Draw all Sobol samples at once and scale to bounds
    sobol = torch.quasirandom.SobolEngine(dimension=4, scramble=True, seed=seed)
    X_raw_all = sobol.draw(total_n).to(dtype=torch.float64)
    X_raw_all = X_raw_all * (u_bound - l_bound) + l_bound

    # Compute clean targets once per source in contiguous blocks
    y_clean_all = torch.empty(total_n, dtype=torch.float64)
    X_src_col_all = torch.empty((total_n, 1), dtype=torch.float64)
    cursor = 0
    
    for idx, (src, n_total, n_test, n_train) in enumerate(
        zip(sources, total_per_source, test_samples_per_source, train_samples_per_source)
    ):
        if n_total == 0:
            continue
        
        # Generate TEST data with even distribution
        if n_test > 0:
            x_test_block = X_raw_all[cursor:cursor + n_test].clone()
            
            # Column 1: E values (2 options) - ensure even distribution for TEST
            n_per_E_test = n_test // len(E_values)
            remaining_E_test = n_test % len(E_values)
            E_indices_test = []
            for i in range(len(E_values)):
                count = n_per_E_test + (1 if i < remaining_E_test else 0)
                E_indices_test.append(torch.full((count,), i))
            E_indices_test = torch.cat(E_indices_test)
            E_indices_test = E_indices_test[torch.randperm(n_test)]
            x_test_block[:, 1] = E_values[E_indices_test]

            # Column 2: K values (4 options) - ensure even distribution for TEST
            n_per_K_test = n_test // len(K_values)
            remaining_K_test = n_test % len(K_values)
            K_indices_test = []
            for i in range(len(K_values)):
                count = n_per_K_test + (1 if i < remaining_K_test else 0)
                K_indices_test.append(torch.full((count,), i))
            K_indices_test = torch.cat(K_indices_test)
            K_indices_test = K_indices_test[torch.randperm(n_test)]
            x_test_block[:, 2] = K_values[K_indices_test]

            # Column 3: I values (3 options) - ensure even distribution for TEST
            n_per_I_test = n_test // len(I_values)
            remaining_I_test = n_test % len(I_values)
            I_indices_test = []
            for i in range(len(I_values)):
                count = n_per_I_test + (1 if i < remaining_I_test else 0)
                I_indices_test.append(torch.full((count,), i))
            I_indices_test = torch.cat(I_indices_test)
            I_indices_test = I_indices_test[torch.randperm(n_test)]
            x_test_block[:, 3] = I_values[I_indices_test]

            # Compute targets for test data
            y_test_clean = buckling_mixed_variables(x_test_block, source=src)
            
            # Store categorical indices if requested
            if return_categorical:
                x_test_block[:, 1] = E_indices_test.to(torch.float64)
                x_test_block[:, 2] = K_indices_test.to(torch.float64)
                x_test_block[:, 3] = I_indices_test.to(torch.float64)
            
            # Store test data
            y_clean_all[cursor:cursor + n_test] = y_test_clean
            X_raw_all[cursor:cursor + n_test] = x_test_block
            X_src_col_all[cursor:cursor + n_test, 0] = float(idx)

        # Generate TRAIN data with even distribution
        if n_train > 0:
            x_train_block = X_raw_all[cursor + n_test:cursor + n_total].clone()
            
            # Column 1: E values (2 options) - ensure even distribution for TRAIN
            n_per_E_train = n_train // len(E_values)
            remaining_E_train = n_train % len(E_values)
            E_indices_train = []
            for i in range(len(E_values)):
                count = n_per_E_train + (1 if i < remaining_E_train else 0)
                E_indices_train.append(torch.full((count,), i))
            E_indices_train = torch.cat(E_indices_train)
            E_indices_train = E_indices_train[torch.randperm(n_train)]
            x_train_block[:, 1] = E_values[E_indices_train]

            # Column 2: K values (4 options) - ensure even distribution for TRAIN
            n_per_K_train = n_train // len(K_values)
            remaining_K_train = n_train % len(K_values)
            K_indices_train = []
            for i in range(len(K_values)):
                count = n_per_K_train + (1 if i < remaining_K_train else 0)
                K_indices_train.append(torch.full((count,), i))
            K_indices_train = torch.cat(K_indices_train)
            K_indices_train = K_indices_train[torch.randperm(n_train)]
            x_train_block[:, 2] = K_values[K_indices_train]

            # Column 3: I values (3 options) - ensure even distribution for TRAIN
            n_per_I_train = n_train // len(I_values)
            remaining_I_train = n_train % len(I_values)
            I_indices_train = []
            for i in range(len(I_values)):
                count = n_per_I_train + (1 if i < remaining_I_train else 0)
                I_indices_train.append(torch.full((count,), i))
            I_indices_train = torch.cat(I_indices_train)
            I_indices_train = I_indices_train[torch.randperm(n_train)]
            x_train_block[:, 3] = I_values[I_indices_train]

            # Compute targets for train data
            y_train_clean = buckling_mixed_variables(x_train_block, source=src)
            
            # Store categorical indices if requested
            if return_categorical:
                x_train_block[:, 1] = E_indices_train.to(torch.float64)
                x_train_block[:, 2] = K_indices_train.to(torch.float64)
                x_train_block[:, 3] = I_indices_train.to(torch.float64)
            
            # Store train data
            y_clean_all[cursor + n_test:cursor + n_total] = y_train_clean
            X_raw_all[cursor + n_test:cursor + n_total] = x_train_block
            X_src_col_all[cursor + n_test:cursor + n_total, 0] = float(idx)

        cursor += n_total

    # Split per source into test then train; get test std after split and add noise scaled by it
    X_train_list: list[torch.Tensor] = []
    y_train_list: list[torch.Tensor] = []
    X_test_list: list[torch.Tensor] = []
    y_test_list: list[torch.Tensor] = []

    cursor = 0
    for idx, (src, n_total, n_test, n_train) in enumerate(
        zip(sources, total_per_source, test_samples_per_source, train_samples_per_source)
    ):
        if n_total == 0:
            continue
        x_block = X_raw_all[cursor:cursor + n_total]
        y_block = y_clean_all[cursor:cursor + n_total]
        src_block = X_src_col_all[cursor:cursor + n_total]

        # Split: first n_test -> test, remaining -> train
        x_test_block = x_block[:n_test] if n_test > 0 else torch.empty((0, 4), dtype=torch.float64)
        y_test_block = y_block[:n_test] if n_test > 0 else torch.empty((0,), dtype=torch.float64)
        src_test_block = src_block[:n_test] if n_test > 0 else torch.empty((0, 1), dtype=torch.float64)
        x_train_block = x_block[n_test:] if n_train > 0 else torch.empty((0, 4), dtype=torch.float64)
        y_train_block = y_block[n_test:] if n_train > 0 else torch.empty((0,), dtype=torch.float64)
        src_train_block = src_block[n_test:] if n_train > 0 else torch.empty((0, 1), dtype=torch.float64)

        # Test std after split (per source)
        if y_test_block.numel() > 1:
            test_std_value = float(y_test_block.std().item())
        else:
            test_std_value = 0.0

        # Apply noise scaled by test std
        if n_train > 0 and train_noise[idx] > 0 and test_std_value > 0.0:
            noise = _sample_noise_like(
                y_train_block, train_noise[idx] * test_std_value, noise_type
            )
            y_train_block = y_train_block + noise

        if n_test > 0 and test_noise[idx] > 0 and test_std_value > 0.0:
            noise = _sample_noise_like(
                y_test_block, test_noise[idx] * test_std_value, noise_type
            )
            y_test_block = y_test_block + noise

        # Append source column and collect
        X_test_list.append(torch.cat([x_test_block, src_test_block], dim=1))
        y_test_list.append(y_test_block)
        X_train_list.append(torch.cat([x_train_block, src_train_block], dim=1))
        y_train_list.append(y_train_block)

        cursor += n_total

    X_train = torch.cat(X_train_list, dim=0) if X_train_list else torch.empty((0, 5), dtype=torch.float64)
    y_train = torch.cat(y_train_list, dim=0) if y_train_list else torch.empty((0,), dtype=torch.float64)
    X_test = torch.cat(X_test_list, dim=0) if X_test_list else torch.empty((0, 5), dtype=torch.float64)
    y_test = torch.cat(y_test_list, dim=0) if y_test_list else torch.empty((0,), dtype=torch.float64)

    return X_train, y_train, X_test, y_test


def borehole_mixed_variables(X: torch.Tensor, source: str = "s0") -> torch.Tensor:
    """
    Compute borehole water flow rate given input variables (torch implementation).

    Args:
        X (torch.Tensor): Input array of shape [n_samples, 8] with columns:
            0: rw (radius of borehole, m)
            1: r (radius of influence, m)
            2: Tu (transmissivity of upper aquifer, m^2/yr)
            3: Hu (potentiometric head of upper aquifer, m)
            4: Tl (transmissivity of lower aquifer, m^2/yr)
            5: Hl (potentiometric head of lower aquifer, m)
            6: L (length of borehole, m)
            7: Kw (hydraulic conductivity of borehole, m/yr)
        source (str): Source/fidelity level ("s0".."s4")
    Returns:
        torch.Tensor: Flow rate values for each input sample
    """
    rw = X[..., 0]
    r = X[..., 1]
    Tu = X[..., 2]
    Hu = X[..., 3]
    Tl = X[..., 4]
    Hl = X[..., 5]
    L = X[..., 6]
    Kw = X[..., 7]

    if source == "s0":
        numerator = 2 * torch.pi * Tu * (Hu - Hl)
        denominator = torch.log(r / rw) * (1 + 2 * L * Tu / (torch.log(r / rw) * rw**2 * Kw) + Tu / Tl)
        result = numerator / denominator
    elif source == "s1":
        numerator = 2 * torch.pi * Tu * (Hu - 0.8 * Hl)
        denominator = torch.log(r / rw) * (1 + 2 * L * Tu / (torch.log(r / rw) * rw**2 * Kw) + Tu / Tl)
        result = numerator / denominator
    elif source == "s2":
        numerator = 2 * torch.pi * Tu * (Hu - 3 * Hl)
        denominator = torch.log(r / rw) * (1 + 8 * L * Tu / (torch.log(r / rw) * rw**2 * Kw) + 0.75 * Tu / Tl)
        result = numerator / denominator
    elif source == "s3":
        numerator = 2 * torch.pi * Tu * (1.1 * Hu - Hl)
        denominator = torch.log(4 * r / rw) * (1 + 3 * L * Tu / (torch.log(r / rw) * rw**2 * Kw) + Tu / Tl)
        result = numerator / denominator
    elif source == "s4":
        numerator = 2 * torch.pi * Tu * (1.05 * Hu - Hl)
        denominator = torch.log(2 * r / rw) * (1 + 2 * L * Tu / (torch.log(r / rw) * rw**2 * Kw) + Tu / Tl)
        result = numerator / denominator
    else:
        raise ValueError(f"Unknown source: {source}. Only s0..s4 are supported for borehole.")

    return result


def generate_mf_borehole_data(
    train_samples_per_source: list[int],
    test_samples_per_source: list[int],
    *,
    seed: int | None = None,
    train_noise: list[float] | float | None = None,
    test_noise: list[float] | float | None = None,
    noise_type: str = 'gaussian',
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate multi-fidelity Borehole data consistent with wing/buckling helpers.

    - Draw a single Sobol batch across all sources (no repeats globally)
    - Split per source: first test, remaining train
    - Scale additive noise by per-source test std (post-split), like wing/buckling

    Returns:
      - X_train, y_train with 9 features (8 continuous + 1 numeric source column)
      - X_test, y_test with same schema
    """
    if seed is not None:
        torch.manual_seed(seed)
    # else:
    #     seed = 42
    #     torch.manual_seed(seed)

    # Defaults and validation for per-source noise (5 sources)
    num_sources = 5
    if train_noise is None:
        train_noise = [0.0] * num_sources
    if test_noise is None:
        test_noise = [0.0] * num_sources
    if isinstance(train_noise, (int, float)):
        train_noise = [float(train_noise)] * num_sources
    if isinstance(test_noise, (int, float)):
        test_noise = [float(test_noise)] * num_sources
    if len(train_noise) != num_sources or len(test_noise) != num_sources:
        raise ValueError(f"train_noise and test_noise must be length-{num_sources} (one scalar per source)")

    sources = ["s0", "s1", "s2", "s3", "s4"]

    # Bounds for the 8 continuous features
    l_bound = torch.tensor([0.05, 100.0, 63070.0, 990.0, 63.1, 700.0, 1120.0, 9855.0], dtype=torch.float64)
    u_bound = torch.tensor([0.15, 50000.0, 115600.0, 1110.0, 116.0, 820.0, 1680.0, 12045.0], dtype=torch.float64)

    # Totals and Sobol draws
    total_per_source = [tr + te for tr, te in zip(train_samples_per_source, test_samples_per_source)]
    total_n = sum(total_per_source)

    sobol = torch.quasirandom.SobolEngine(dimension=8, scramble=True, seed=seed)
    X_raw_all = sobol.draw(total_n).to(dtype=torch.float64)
    X_raw_all = X_raw_all * (u_bound - l_bound) + l_bound

    # Compute clean targets per contiguous source block
    y_clean_all = torch.empty(total_n, dtype=torch.float64)
    src_ids_all = torch.empty((total_n, 1), dtype=torch.float64)
    cursor = 0
    for idx, (src, n) in enumerate(zip(sources, total_per_source)):
        if n == 0:
            continue
        x_block = X_raw_all[cursor:cursor + n]
        y_clean_all[cursor:cursor + n] = borehole_mixed_variables(x_block, source=src)
        src_ids_all[cursor:cursor + n, 0] = float(idx)
        cursor += n

    # Split into test then train per source; add scaled noise
    X_train_list: list[torch.Tensor] = []
    y_train_list: list[torch.Tensor] = []
    X_test_list: list[torch.Tensor] = []
    y_test_list: list[torch.Tensor] = []

    cursor = 0
    for idx, (src, n_total, n_test, n_train) in enumerate(
        zip(sources, total_per_source, test_samples_per_source, train_samples_per_source)
    ):
        if n_total == 0:
            continue
        x_block = X_raw_all[cursor:cursor + n_total]
        y_block = y_clean_all[cursor:cursor + n_total]
        s_block = src_ids_all[cursor:cursor + n_total]

        x_test_block = x_block[:n_test] if n_test > 0 else torch.empty((0, 8), dtype=torch.float64)
        y_test_block = y_block[:n_test] if n_test > 0 else torch.empty((0,), dtype=torch.float64)
        s_test_block = s_block[:n_test] if n_test > 0 else torch.empty((0, 1), dtype=torch.float64)
        x_train_block = x_block[n_test:] if n_train > 0 else torch.empty((0, 8), dtype=torch.float64)
        y_train_block = y_block[n_test:] if n_train > 0 else torch.empty((0,), dtype=torch.float64)
        s_train_block = s_block[n_test:] if n_train > 0 else torch.empty((0, 1), dtype=torch.float64)

        # Per-source test std
        if y_test_block.numel() > 1:
            test_std_value = float(y_test_block.std().item())
        else:
            test_std_value = 0.0

        # Apply noise scaled by test std
        if n_train > 0 and train_noise[idx] > 0 and test_std_value > 0.0:
            noise = _sample_noise_like(
                y_train_block, train_noise[idx] * test_std_value, noise_type
            )
            y_train_block = y_train_block + noise

        if n_test > 0 and test_noise[idx] > 0 and test_std_value > 0.0:
            noise = _sample_noise_like(
                y_test_block, test_noise[idx] * test_std_value, noise_type
            )
            y_test_block = y_test_block + noise

        # Append numeric source id as 9th feature
        if n_train > 0:
            X_train_list.append(torch.cat([x_train_block, s_train_block], dim=1))
            y_train_list.append(y_train_block)
        if n_test > 0:
            X_test_list.append(torch.cat([x_test_block, s_test_block], dim=1))
            y_test_list.append(y_test_block)

        cursor += n_total

    X_train = torch.cat(X_train_list, dim=0) if X_train_list else torch.empty((0, 9), dtype=torch.float64)
    y_train = torch.cat(y_train_list, dim=0) if y_train_list else torch.empty((0,), dtype=torch.float64)
    X_test = torch.cat(X_test_list, dim=0) if X_test_list else torch.empty((0, 9), dtype=torch.float64)
    y_test = torch.cat(y_test_list, dim=0) if y_test_list else torch.empty((0,), dtype=torch.float64)

    return X_train, y_train, X_test, y_test


def ackley_function(X: torch.Tensor, dimensions: int = None) -> torch.Tensor:
    """
    Compute the Ackley function for given input variables.
    
    The Ackley function is defined as:
    f(x) = -20 * exp(-0.2 * sqrt(1/d * sum(x_i^2))) - exp(1/d * sum(cos(2*pi*x_i))) + 20 + e
    
    where x ∈ [-32.768, 32.768]^d and d is the number of dimensions
    
    Args:
        X (torch.Tensor): Input array of shape [n_samples, d] where d is the number of dimensions
        dimensions (int): Number of dimensions (optional, inferred from X if not provided)
        
    Returns:
        torch.Tensor: Ackley function values for each input sample
    """
    if dimensions is None:
        dimensions = X.shape[1]
    
    # Constants
    a = 20.0
    b = 0.2
    c = 2 * torch.pi
    
    # Compute the two main terms
    sum_squares = torch.sum(X**2, dim=1)
    sum_cos = torch.sum(torch.cos(c * X), dim=1)
    
    # Ackley function
    term1 = -a * torch.exp(-b * torch.sqrt(sum_squares / dimensions))
    term2 = -torch.exp(sum_cos / dimensions)
    result = term1 + term2 + a + torch.e
    
    return result


def generate_ackley_data(n_train: int, n_test: int, dimensions: int = 2, x_bounds: list[float] = [-5, 10], train_noise: float = 0.0, 
                        test_noise: float = 0.0, noise_type: str = 'gaussian', seed: int = None, V2: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate train and test data for the Ackley function using Sobol sequences.
    
    Args:
        n_train (int): Number of training samples to generate
        n_test (int): Number of test samples to generate
        dimensions (int): Number of dimensions for the Ackley function
        train_noise (float): Noise level for training data as a fraction of std
        test_noise (float): Noise level for test data as a fraction of std
        noise_type (str): Type of noise ('gaussian' or 'uniform')
        seed (int): Random seed for reproducibility
        
    Returns:
        X_train, y_train, X_test, y_test: Train and test data
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    l_bound = x_bounds[0]
    u_bound = x_bounds[1]
    
    # Generate ALL samples at once to avoid repeats
    total_samples = n_train + n_test
    sobol = torch.quasirandom.SobolEngine(dimension=dimensions, scramble=True)
    X_all = sobol.draw(total_samples).to(dtype=torch.float64)
    
    # Scale to Ackley bounds
    X_all = X_all * (u_bound - l_bound) + l_bound
    
    # Compute Ackley function values
    y_all = ackley_function(X_all, dimensions)

    if V2:
        y_all = torch.log(y_all+1)
    
    # Split into train and test
    X_train = X_all[:n_train]
    y_train = y_all[:n_train]
    X_test = X_all[n_train:]
    y_test = y_all[n_train:]
    
    # Add noise separately to train and test
    # Both train and test noise are based on TEST std
    y_test_std = y_test.std()
    
    if train_noise > 0:
        noise_scale = train_noise * y_test_std
        noise = _sample_noise_like(y_train, noise_scale, noise_type)
        y_train = y_train + noise
    
    if test_noise > 0:
        noise_scale = test_noise * y_test_std
        noise = _sample_noise_like(y_test, noise_scale, noise_type)
        y_test = y_test + noise
    
    return X_train, y_train, X_test, y_test


def tabpfn_1d_sin_plus_x_function(X: torch.Tensor) -> torch.Tensor:
    """f(x) = sin(x) + x for X of shape (n, 1). TabPFN-style 1D toy (Hollmann et al.)."""
    x = X[:, 0]
    return torch.sin(x) + x


def tabpfn_1d_sin_2pi_x_plus_x_function(X: torch.Tensor) -> torch.Tensor:
    """f(x) = sin(2*pi*x) + x for X of shape (n, 1)."""
    x = X[:, 0]
    return torch.sin(2 * torch.pi * x) + x


def tabpfn_1d_sin_2pi_x_function(X: torch.Tensor) -> torch.Tensor:
    """f(x) = sin(2*pi*x) for X of shape (n, 1)."""
    x = X[:, 0]
    return torch.sin(2 * torch.pi * x)


def tabpfn_1d_sin_2pi_x_windowed_function(X: torch.Tensor) -> torch.Tensor:
    """
    f(x) = sin(2*pi*x) on [-1, 1], and 0 outside.
    Useful for extended-domain tests on [-2, 2].
    """
    x = X[:, 0]
    mask = (x >= -1.0) & (x <= 1.0)
    y_mid = torch.sin(2 * torch.pi * x)
    return torch.where(mask, y_mid, torch.zeros_like(x))


def tabpfn_1d_sin_2pi_x_plus_x_windowed_function(X: torch.Tensor) -> torch.Tensor:
    """
    f(x) = sin(2*pi*x) + x on [-1, 1], and 0 outside.
    Useful for extended-domain tests on [-2, 2].
    """
    x = X[:, 0]
    mask = (x >= -1.0) & (x <= 1.0)
    y_mid = torch.sin(2 * torch.pi * x) + x
    return torch.where(mask, y_mid, torch.zeros_like(x))


def tabpfn_1d_x_squared_function(X: torch.Tensor) -> torch.Tensor:
    """f(x) = x^2 for X of shape (n, 1)."""
    return X[:, 0] ** 2


def tabpfn_1d_abs_x_function(X: torch.Tensor) -> torch.Tensor:
    """f(x) = |x| for X of shape (n, 1)."""
    return torch.abs(X[:, 0])


def tabpfn_1d_linear_function(
    X: torch.Tensor,
    slope: float = 3.0,
    intercept: float = 0.0,
) -> torch.Tensor:
    """f(x) = slope * x + intercept for X of shape (n, 1)."""
    x = X[:, 0]
    return slope * x + intercept


def tabpfn_1d_step_function(
    X: torch.Tensor,
    x_bounds: tuple[float, float] = (-0.5, 0.5),
    step_values: tuple[float, ...] = (-0.4, -0.2, 0.1, 0.4),
) -> torch.Tensor:
    """
    Piecewise-constant step function on equal-width bins along x (TabPFN Figure 3 style).

    Default: four steps on [-0.5, 0.5] with plateaus -0.4, -0.2, 0.1, 0.4.
    """
    l_bound, u_bound = x_bounds
    x = X[:, 0]
    n_steps = len(step_values)
    width = (u_bound - l_bound) / n_steps
    bin_idx = torch.floor((x - l_bound) / width).long().clamp(0, n_steps - 1)
    levels = torch.tensor(step_values, dtype=x.dtype, device=x.device)
    return levels[bin_idx]


def _generate_tabpfn_1d_toy_data(
    n_train: int,
    n_test: int,
    y_fn,
    x_bounds: list[float],
    test_x_bounds: list[float] | None,
    train_noise: float,
    test_noise: float,
    noise_type: str,
    seed: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sobol samples in 1D, evaluate y_fn(X), add noise; train/test can use different domains."""
    if seed is not None:
        torch.manual_seed(seed)

    train_l_bound = x_bounds[0]
    train_u_bound = x_bounds[1]
    if test_x_bounds is None:
        test_l_bound, test_u_bound = train_l_bound, train_u_bound
    else:
        test_l_bound, test_u_bound = test_x_bounds[0], test_x_bounds[1]

    sobol = torch.quasirandom.SobolEngine(dimension=1, scramble=True)
    X_train = sobol.draw(n_train).to(dtype=torch.float64)
    X_test = sobol.draw(n_test).to(dtype=torch.float64)
    X_train = X_train * (train_u_bound - train_l_bound) + train_l_bound
    X_test = X_test * (test_u_bound - test_l_bound) + test_l_bound

    y_train = y_fn(X_train)
    y_test = y_fn(X_test)

    y_test_std = y_test.std()

    if train_noise > 0:
        noise_scale = train_noise * y_test_std
        noise = _sample_noise_like(y_train, noise_scale, noise_type)
        y_train = y_train + noise

    if test_noise > 0:
        noise_scale = test_noise * y_test_std
        noise = _sample_noise_like(y_test, noise_scale, noise_type)
        y_test = y_test + noise

    return X_train, y_train, X_test, y_test


def generate_tabpfn_1d_sin_plus_x_data(
    n_train: int,
    n_test: int,
    dimensions: int = 1,
    x_bounds: list[float] | None = None,
    test_x_bounds: list[float] | None = None,
    train_noise: float = 0.0,
    test_noise: float = 0.0,
    noise_type: str = "gaussian",
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Train/test data for f(x) = sin(x) + x on an interval (default [-0.5, 0.5], as in TabPFN Fig. 3).
    """
    if dimensions != 1:
        raise ValueError(f"TabPFN 1D toys require dimensions=1, got {dimensions}")
    bounds = [-0.5, 0.5] if x_bounds is None else x_bounds
    return _generate_tabpfn_1d_toy_data(
        n_train, n_test, tabpfn_1d_sin_plus_x_function, bounds, test_x_bounds, train_noise, test_noise, noise_type, seed
    )


def generate_tabpfn_1d_x_squared_data(
    n_train: int,
    n_test: int,
    dimensions: int = 1,
    x_bounds: list[float] | None = None,
    test_x_bounds: list[float] | None = None,
    train_noise: float = 0.0,
    test_noise: float = 0.0,
    noise_type: str = "gaussian",
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Train/test data for f(x) = x^2 (default x in [-0.5, 0.5])."""
    if dimensions != 1:
        raise ValueError(f"TabPFN 1D toys require dimensions=1, got {dimensions}")
    bounds = [-0.5, 0.5] if x_bounds is None else x_bounds
    return _generate_tabpfn_1d_toy_data(
        n_train, n_test, tabpfn_1d_x_squared_function, bounds, test_x_bounds, train_noise, test_noise, noise_type, seed
    )


def generate_tabpfn_1d_abs_x_data(
    n_train: int,
    n_test: int,
    dimensions: int = 1,
    x_bounds: list[float] | None = None,
    test_x_bounds: list[float] | None = None,
    train_noise: float = 0.0,
    test_noise: float = 0.0,
    noise_type: str = "gaussian",
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Train/test data for f(x) = |x| (default x in [-0.5, 0.5])."""
    if dimensions != 1:
        raise ValueError(f"TabPFN 1D toys require dimensions=1, got {dimensions}")
    bounds = [-0.5, 0.5] if x_bounds is None else x_bounds
    return _generate_tabpfn_1d_toy_data(
        n_train, n_test, tabpfn_1d_abs_x_function, bounds, test_x_bounds, train_noise, test_noise, noise_type, seed
    )


def generate_tabpfn_1d_step_data(
    n_train: int,
    n_test: int,
    dimensions: int = 1,
    x_bounds: list[float] | None = None,
    test_x_bounds: list[float] | None = None,
    step_values: tuple[float, ...] = (-0.4, -0.2, 0.1, 0.4),
    train_noise: float = 0.0,
    test_noise: float = 0.0,
    noise_type: str = "gaussian",
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Train/test data for a 4-step piecewise constant on equal bins (TabPFN Fig. 3 defaults).
    Bins cover x_bounds; step_values must have one value per bin (default four plateaus).
    """
    if dimensions != 1:
        raise ValueError(f"TabPFN 1D toys require dimensions=1, got {dimensions}")
    bounds = [-0.5, 0.5] if x_bounds is None else x_bounds
    tb = (bounds[0], bounds[1])

    def y_fn(X):
        return tabpfn_1d_step_function(X, x_bounds=tb, step_values=step_values)

    return _generate_tabpfn_1d_toy_data(
        n_train, n_test, y_fn, bounds, test_x_bounds, train_noise, test_noise, noise_type, seed
    )


def generate_tabpfn_1d_sin_2pi_x_plus_x_data(
    n_train: int,
    n_test: int,
    dimensions: int = 1,
    x_bounds: list[float] | None = None,
    test_x_bounds: list[float] | None = None,
    train_noise: float = 0.0,
    test_noise: float = 0.0,
    noise_type: str = "gaussian",
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Train/test data for f(x) = sin(2*pi*x) + x (default x in [-1, 1])."""
    if dimensions != 1:
        raise ValueError(f"TabPFN 1D toys require dimensions=1, got {dimensions}")
    bounds = [-1.0, 1.0] if x_bounds is None else x_bounds
    return _generate_tabpfn_1d_toy_data(
        n_train,
        n_test,
        tabpfn_1d_sin_2pi_x_plus_x_function,
        bounds,
        test_x_bounds,
        train_noise,
        test_noise,
        noise_type,
        seed,
    )


def generate_tabpfn_1d_sin_2pi_x_data(
    n_train: int,
    n_test: int,
    dimensions: int = 1,
    x_bounds: list[float] | None = None,
    test_x_bounds: list[float] | None = None,
    train_noise: float = 0.0,
    test_noise: float = 0.0,
    noise_type: str = "gaussian",
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Train/test data for f(x) = sin(2*pi*x) (default x in [-1, 1])."""
    if dimensions != 1:
        raise ValueError(f"TabPFN 1D toys require dimensions=1, got {dimensions}")
    bounds = [-1.0, 1.0] if x_bounds is None else x_bounds
    return _generate_tabpfn_1d_toy_data(
        n_train,
        n_test,
        tabpfn_1d_sin_2pi_x_function,
        bounds,
        test_x_bounds,
        train_noise,
        test_noise,
        noise_type,
        seed,
    )


def generate_tabpfn_1d_sin_2pi_x_windowed_data(
    n_train: int,
    n_test: int,
    dimensions: int = 1,
    x_bounds: list[float] | None = None,
    test_x_bounds: list[float] | None = None,
    train_noise: float = 0.0,
    test_noise: float = 0.0,
    noise_type: str = "gaussian",
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Train/test data for windowed f(x) = sin(2*pi*x), zero outside [-1, 1] (default x in [-2, 2])."""
    if dimensions != 1:
        raise ValueError(f"TabPFN 1D toys require dimensions=1, got {dimensions}")
    bounds = [-2.0, 2.0] if x_bounds is None else x_bounds
    return _generate_tabpfn_1d_toy_data(
        n_train,
        n_test,
        tabpfn_1d_sin_2pi_x_windowed_function,
        bounds,
        test_x_bounds,
        train_noise,
        test_noise,
        noise_type,
        seed,
    )


def generate_tabpfn_1d_sin_2pi_x_plus_x_windowed_data(
    n_train: int,
    n_test: int,
    dimensions: int = 1,
    x_bounds: list[float] | None = None,
    test_x_bounds: list[float] | None = None,
    train_noise: float = 0.0,
    test_noise: float = 0.0,
    noise_type: str = "gaussian",
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Train/test data for windowed f(x) = sin(2*pi*x)+x, zero outside [-1, 1] (default x in [-2, 2])."""
    if dimensions != 1:
        raise ValueError(f"TabPFN 1D toys require dimensions=1, got {dimensions}")
    bounds = [-2.0, 2.0] if x_bounds is None else x_bounds
    return _generate_tabpfn_1d_toy_data(
        n_train,
        n_test,
        tabpfn_1d_sin_2pi_x_plus_x_windowed_function,
        bounds,
        test_x_bounds,
        train_noise,
        test_noise,
        noise_type,
        seed,
    )


def generate_tabpfn_1d_linear_homoscedastic_data(
    n_train: int,
    n_test: int,
    dimensions: int = 1,
    x_bounds: list[float] | None = None,
    test_x_bounds: list[float] | None = None,
    slope: float = 1.0,
    intercept: float = 0.0,
    train_noise: float = 0.0,
    test_noise: float = 0.0,
    noise_type: str = "gaussian",
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Linear 1D data with homoscedastic noise."""
    if dimensions != 1:
        raise ValueError(f"TabPFN 1D toys require dimensions=1, got {dimensions}")
    bounds = [-1.0, 1.0] if x_bounds is None else x_bounds

    def y_fn(X):
        return tabpfn_1d_linear_function(X, slope=slope, intercept=intercept)

    return _generate_tabpfn_1d_toy_data(
        n_train,
        n_test,
        y_fn,
        bounds,
        test_x_bounds,
        train_noise,
        test_noise,
        noise_type,
        seed,
    )


def generate_tabpfn_1d_linear_heteroscedastic_data(
    n_train: int,
    n_test: int,
    dimensions: int = 1,
    x_bounds: list[float] | None = None,
    test_x_bounds: list[float] | None = None,
    slope: float = 1.0,
    intercept: float = 0.0,
    train_noise: float = 0.0,
    test_noise: float = 0.0,
    noise_type: str = "gaussian",
    seed: int | None = None,
    hetero_min_scale: float = 0.1,
    hetero_max_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Linear 1D data with heteroscedastic noise increasing left->right.
    Noise std is scaled linearly from hetero_min_scale to hetero_max_scale across each split domain.
    """
    if dimensions != 1:
        raise ValueError(f"TabPFN 1D toys require dimensions=1, got {dimensions}")
    if seed is not None:
        torch.manual_seed(seed)
    bounds = [-1.0, 1.0] if x_bounds is None else x_bounds
    train_l, train_u = bounds[0], bounds[1]
    if test_x_bounds is None:
        test_l, test_u = train_l, train_u
    else:
        test_l, test_u = test_x_bounds[0], test_x_bounds[1]

    sobol = torch.quasirandom.SobolEngine(dimension=1, scramble=True)
    X_train = sobol.draw(n_train).to(dtype=torch.float64)
    X_test = sobol.draw(n_test).to(dtype=torch.float64)
    X_train = X_train * (train_u - train_l) + train_l
    X_test = X_test * (test_u - test_l) + test_l

    y_train = tabpfn_1d_linear_function(X_train, slope=slope, intercept=intercept)
    y_test = tabpfn_1d_linear_function(X_test, slope=slope, intercept=intercept)

    y_test_std = y_test.std()

    def _linear_scale(x, l_bound, u_bound):
        denom = max(u_bound - l_bound, 1e-12)
        w = ((x - l_bound) / denom).clamp(0.0, 1.0)
        return hetero_min_scale + (hetero_max_scale - hetero_min_scale) * w

    if train_noise > 0:
        base = train_noise * y_test_std
        scale = _linear_scale(X_train[:, 0], train_l, train_u)
        noise = _sample_noise_like(y_train, base * scale, noise_type)
        y_train = y_train + noise

    if test_noise > 0:
        base = test_noise * y_test_std
        scale = _linear_scale(X_test[:, 0], test_l, test_u)
        noise = _sample_noise_like(y_test, base * scale, noise_type)
        y_test = y_test + noise

    return X_train, y_train, X_test, y_test


def rastrigin_function(X: torch.Tensor, dimensions: int = None, shift: torch.Tensor | float | None = None) -> torch.Tensor:
    """
    Compute the Rastrigin function for given input variables.
    
    The Rastrigin function is defined as:
    f(x) = 10*d + sum_{i=1}^d [x_i^2 - 10*cos(2*pi*x_i)]
    
    For shifted Rastrigin: f(x - shift)
    
    where x ∈ [-5.12, 5.12]^d and d is the number of dimensions
    
    Args:
        X (torch.Tensor): Input array of shape [n_samples, d] where d is the number of dimensions
        dimensions (int): Number of dimensions (optional, inferred from X if not provided)
        shift (torch.Tensor, float, or None): Shift vector for shifted Rastrigin. 
            If None, computes standard Rastrigin.
            If float, applies same shift to all dimensions.
            If torch.Tensor of shape (d,), applies per-dimension shift.
            If torch.Tensor of shape (1, d), broadcasts to all samples.
        
    Returns:
        torch.Tensor: Rastrigin function values for each input sample
    """
    if dimensions is None:
        dimensions = X.shape[1]
    
    # Apply shift if provided
    if shift is not None:
        if isinstance(shift, (int, float)):
            # Scalar shift: apply to all dimensions
            X_shifted = X - shift
        elif isinstance(shift, torch.Tensor):
            if shift.dim() == 0:
                # Scalar tensor
                X_shifted = X - shift
            elif shift.dim() == 1:
                # Vector of shape (d,)
                if shift.shape[0] != dimensions:
                    raise ValueError(f"Shift vector length {shift.shape[0]} must match dimensions {dimensions}")
                X_shifted = X - shift.unsqueeze(0)  # Broadcast to all samples
            elif shift.dim() == 2 and shift.shape[0] == 1:
                # Shape (1, d) - broadcast to all samples
                if shift.shape[1] != dimensions:
                    raise ValueError(f"Shift vector length {shift.shape[1]} must match dimensions {dimensions}")
                X_shifted = X - shift
            else:
                raise ValueError(f"Shift tensor must be scalar, 1D vector, or 2D with shape (1, d), got shape {shift.shape}")
        else:
            raise ValueError(f"Shift must be float, int, torch.Tensor, or None, got {type(shift)}")
    else:
        X_shifted = X
    
    # Rastrigin function: 10*d + sum(x_i^2 - 10*cos(2*pi*x_i))
    term1 = torch.sum(X_shifted**2, dim=1)  # sum(x_i^2)
    term2 = -10 * torch.sum(torch.cos(2 * torch.pi * X_shifted), dim=1)  # -10*sum(cos(2*pi*x_i))
    result = 10 * dimensions + term1 + term2
    
    return result


def generate_rastrigin_data(n_train: int, n_test: int, dimensions: int = 2, x_bounds: list[float] = [-5.12, 5.12], train_noise: float = 0.0, 
                           test_noise: float = 0.0, noise_type: str = 'gaussian', seed: int = None, 
                           shift: torch.Tensor | float | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate train and test data for the Rastrigin function using Sobol sequences.
    
    Args:
        n_train (int): Number of training samples to generate
        n_test (int): Number of test samples to generate
        dimensions (int): Number of dimensions for the Rastrigin function
        train_noise (float): Noise level for training data as a fraction of std
        test_noise (float): Noise level for test data as a fraction of std
        noise_type (str): Type of noise ('gaussian' or 'uniform')
        seed (int): Random seed for reproducibility
        shift (torch.Tensor, float, or None): Shift vector for shifted Rastrigin.
            If None, generates standard Rastrigin data.
            If float, applies same shift to all dimensions.
            If torch.Tensor of shape (d,), applies per-dimension shift.
        
    Returns:
        X_train, y_train, X_test, y_test: Train and test data
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    l_bound = x_bounds[0]
    u_bound = x_bounds[1]
    
    # Generate ALL samples at once to avoid repeats
    total_samples = n_train + n_test
    sobol = torch.quasirandom.SobolEngine(dimension=dimensions, scramble=True)
    X_all = sobol.draw(total_samples).to(dtype=torch.float64)
    
    # Scale to Rastrigin bounds
    X_all = X_all * (u_bound - l_bound) + l_bound
    
    # Compute Rastrigin function values (with optional shift)
    y_all = rastrigin_function(X_all, dimensions, shift=shift)
    
    # Split into train and test
    X_train = X_all[:n_train]
    y_train = y_all[:n_train]
    X_test = X_all[n_train:]
    y_test = y_all[n_train:]
    
    # Add noise separately to train and test
    # Both train and test noise are based on TEST std
    y_test_std = y_test.std()
    
    if train_noise > 0:
        noise_scale = train_noise * y_test_std
        noise = _sample_noise_like(y_train, noise_scale, noise_type)
        y_train = y_train + noise
    
    if test_noise > 0:
        noise_scale = test_noise * y_test_std
        noise = _sample_noise_like(y_test, noise_scale, noise_type)
        y_test = y_test + noise
    
    return X_train, y_train, X_test, y_test

def rosenbrock_function(X: torch.Tensor, dimensions: int = None) -> torch.Tensor:
    """
    Compute the Rosenbrock function for given input variables.
    
    The Rosenbrock function is defined as:
    f(x) = sum_{i=1}^{d-1} [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
    
    where x ∈ [-5, 10]^d and d is the number of dimensions
    
    Args:
        X (torch.Tensor): Input array of shape [n_samples, d] where d is the number of dimensions
        dimensions (int): Number of dimensions (optional, inferred from X if not provided)
        
    Returns:
        torch.Tensor: Rosenbrock function values for each input sample
    """
    if dimensions is None:
        dimensions = X.shape[1]
    
    # Rosenbrock function: sum over i=1 to d-1 of [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
    # Vectorized computation: X[:, :-1] are x_i, X[:, 1:] are x_{i+1}
    x_i = X[:, :-1]  # All x_i for i=0 to d-2
    x_i_plus_1 = X[:, 1:]  # All x_{i+1} for i=0 to d-2
    
    term1 = 100 * (x_i_plus_1 - x_i**2)**2
    term2 = (1 - x_i)**2
    result = torch.sum(term1 + term2, dim=1)
    
    return result


def generate_rosenbrock_data(n_train: int, n_test: int, dimensions: int = 2, x_bounds: list[float] = [-5, 10], train_noise: float = 0.0, 
                            test_noise: float = 0.0, noise_type: str = 'gaussian', seed: int = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate train and test data for the Rosenbrock function using Sobol sequences.
    
    Args:
        n_train (int): Number of training samples to generate
        n_test (int): Number of test samples to generate
        dimensions (int): Number of dimensions for the Rosenbrock function
        train_noise (float): Noise level for training data as a fraction of std
        test_noise (float): Noise level for test data as a fraction of std
        noise_type (str): Type of noise ('gaussian' or 'uniform')
        seed (int): Random seed for reproducibility
        
    Returns:
        X_train, y_train, X_test, y_test: Train and test data
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    l_bound = x_bounds[0]
    u_bound = x_bounds[1]
    
    # Generate ALL samples at once to avoid repeats
    total_samples = n_train + n_test
    sobol = torch.quasirandom.SobolEngine(dimension=dimensions, scramble=True)
    X_all = sobol.draw(total_samples).to(dtype=torch.float64)
    
    # Scale to Rosenbrock bounds
    X_all = X_all * (u_bound - l_bound) + l_bound
    
    # Compute Rosenbrock function values
    y_all = rosenbrock_function(X_all, dimensions)
    
    # Split into train and test
    X_train = X_all[:n_train]
    y_train = y_all[:n_train]
    X_test = X_all[n_train:]
    y_test = y_all[n_train:]
    
    # Add noise separately to train and test
    # Both train and test noise are based on TEST std
    y_test_std = y_test.std()
    
    if train_noise > 0:
        noise_scale = train_noise * y_test_std
        noise = _sample_noise_like(y_train, noise_scale, noise_type)
        y_train = y_train + noise
    
    if test_noise > 0:
        noise_scale = test_noise * y_test_std
        noise = _sample_noise_like(y_test, noise_scale, noise_type)
        y_test = y_test + noise
    
    return X_train, y_train, X_test, y_test


def zakharov_function(X: torch.Tensor, dimensions: int = None) -> torch.Tensor:
    """
    Compute the Zakharov function for given input variables.
    
    The Zakharov function is defined as:
    f(x) = sum_{i=1}^{d} x_i^2 + (sum_{i=1}^{d} 0.5*i*x_i)^2 + (sum_{i=1}^{d} 0.5*i*x_i)^4
    
    where x ∈ [-5, 10]^d and d is the number of dimensions
    
    Args:
        X (torch.Tensor): Input array of shape [n_samples, d] where d is the number of dimensions
        dimensions (int): Number of dimensions (optional, inferred from X if not provided)
        
    Returns:
        torch.Tensor: Zakharov function values for each input sample
    """
    if dimensions is None:
        dimensions = X.shape[1]
    
    # First term: sum of squares
    term1 = torch.sum(X**2, dim=1)
    
    # Second and third terms: sum of 0.5*i*x_i
    # Create indices [0.5, 1.0, 1.5, ..., 0.5*d] for each sample
    i_values = torch.arange(1, dimensions + 1, dtype=X.dtype, device=X.device) * 0.5
    weighted_sum = torch.sum(X * i_values.unsqueeze(0), dim=1)
    
    # Second term: (weighted_sum)^2
    term2 = weighted_sum**2
    
    # Third term: (weighted_sum)^4
    term3 = weighted_sum**4
    
    result = term1 + term2 + term3
    
    return result


def generate_zakharov_data(n_train: int, n_test: int, dimensions: int = 2, x_bounds: list[float] = [-5, 10], train_noise: float = 0.0, 
                            test_noise: float = 0.0, noise_type: str = 'gaussian', seed: int = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate train and test data for the Zakharov function using Sobol sequences.
    
    Args:
        n_train (int): Number of training samples to generate
        n_test (int): Number of test samples to generate
        dimensions (int): Number of dimensions for the Zakharov function
        x_bounds (list[float]): Bounds for each dimension [lower, upper] (default: [-5, 10])
        train_noise (float): Noise level for training data as a fraction of std
        test_noise (float): Noise level for test data as a fraction of std
        noise_type (str): Type of noise ('gaussian' or 'uniform')
        seed (int): Random seed for reproducibility
        
    Returns:
        X_train, y_train, X_test, y_test: Train and test data
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    l_bound = x_bounds[0]
    u_bound = x_bounds[1]
    
    # Generate ALL samples at once to avoid repeats
    total_samples = n_train + n_test
    sobol = torch.quasirandom.SobolEngine(dimension=dimensions, scramble=True)
    X_all = sobol.draw(total_samples).to(dtype=torch.float64)
    
    # Scale to Zakharov bounds
    X_all = X_all * (u_bound - l_bound) + l_bound
    
    # Compute Zakharov function values
    y_all = zakharov_function(X_all, dimensions)
    
    # Split into train and test
    X_train = X_all[:n_train]
    y_train = y_all[:n_train]
    X_test = X_all[n_train:]
    y_test = y_all[n_train:]
    
    # Add noise separately to train and test
    # Both train and test noise are based on TEST std
    y_test_std = y_test.std()
    
    if train_noise > 0:
        noise_scale = train_noise * y_test_std
        noise = _sample_noise_like(y_train, noise_scale, noise_type)
        y_train = y_train + noise
    
    if test_noise > 0:
        noise_scale = test_noise * y_test_std
        noise = _sample_noise_like(y_test, noise_scale, noise_type)
        y_test = y_test + noise
    
    return X_train, y_train, X_test, y_test


def michalewicz_function(X: torch.Tensor, dimensions: int = None, m: float = 10.0) -> torch.Tensor:
    """
    Compute the Michalewicz function for given input variables.
    
    The Michalewicz function is defined as:
    f(x) = -sum_{i=1}^{d} sin(x_i) * sin^{2m}(i*x_i^2/π)
    
    where x ∈ [0, π]^d and d is the number of dimensions
    m is typically 10
    
    Args:
        X (torch.Tensor): Input array of shape [n_samples, d] where d is the number of dimensions
        dimensions (int): Number of dimensions (optional, inferred from X if not provided)
        m (float): Steepness parameter (default: 10.0)
        
    Returns:
        torch.Tensor: Michalewicz function values for each input sample
    """
    if dimensions is None:
        dimensions = X.shape[1]
    
    # Create indices [1, 2, 3, ..., d] for each sample
    i_values = torch.arange(1, dimensions + 1, dtype=X.dtype, device=X.device)
    
    # Compute sin(x_i) * sin^{2m}(i*x_i^2/π) for each dimension
    sin_x = torch.sin(X)
    sin_power_arg = i_values.unsqueeze(0) * X**2 / torch.pi
    sin_power = torch.sin(sin_power_arg) ** (2 * m)
    
    # Sum over all dimensions and negate
    result = -torch.sum(sin_x * sin_power, dim=1)
    
    return result


def generate_michalewicz_data(n_train: int, n_test: int, dimensions: int = 2, x_bounds: list[float] = [0, 3.141592653589793], train_noise: float = 0.0, 
                               test_noise: float = 0.0, noise_type: str = 'gaussian', seed: int = None, 
                               m: float = 10.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate train and test data for the Michalewicz function using Sobol sequences.
    
    Args:
        n_train (int): Number of training samples to generate
        n_test (int): Number of test samples to generate
        dimensions (int): Number of dimensions for the Michalewicz function
        x_bounds (list[float]): Bounds for each dimension [lower, upper] (default: [0, π])
        train_noise (float): Noise level for training data as a fraction of std
        test_noise (float): Noise level for test data as a fraction of std
        noise_type (str): Type of noise ('gaussian' or 'uniform')
        seed (int): Random seed for reproducibility
        m (float): Steepness parameter for Michalewicz function (default: 10.0)
        
    Returns:
        X_train, y_train, X_test, y_test: Train and test data
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    l_bound = x_bounds[0]
    u_bound = x_bounds[1]
    
    # Generate ALL samples at once to avoid repeats
    total_samples = n_train + n_test
    sobol = torch.quasirandom.SobolEngine(dimension=dimensions, scramble=True)
    X_all = sobol.draw(total_samples).to(dtype=torch.float64)
    
    # Scale to Michalewicz bounds
    X_all = X_all * (u_bound - l_bound) + l_bound
    
    # Compute Michalewicz function values
    y_all = michalewicz_function(X_all, dimensions, m=m)
    
    # Split into train and test
    X_train = X_all[:n_train]
    y_train = y_all[:n_train]
    X_test = X_all[n_train:]
    y_test = y_all[n_train:]
    
    # Add noise separately to train and test
    # Both train and test noise are based on TEST std
    y_test_std = y_test.std()
    
    if train_noise > 0:
        noise_scale = train_noise * y_test_std
        noise = _sample_noise_like(y_train, noise_scale, noise_type)
        y_train = y_train + noise
    
    if test_noise > 0:
        noise_scale = test_noise * y_test_std
        noise = _sample_noise_like(y_test, noise_scale, noise_type)
        y_test = y_test + noise
    
    return X_train, y_train, X_test, y_test


def keane_bump_function(X: torch.Tensor, dimensions: int = None) -> torch.Tensor:
    """
    Compute the Keane Bump function for given input variables.
    
    The Keane Bump function is defined as:
    f(x) = - | ( Σ_{i=1}^{d} cos⁴(x_i) - 2 Π_{i=1}^{d} cos²(x_i) ) / √( Σ_{i=1}^{d} i * x_i² ) |
    
    where x ∈ [0, 10]^d and d is the number of dimensions
    
    Note: Constraints are ignored for prediction/regression tasks.
    The constraints are:
    - c₁(x) = 0.75 - Π_{i=1}^{d} x_i ≤ 0
    - c₂(x) = Σ_{i=1}^{d} x_i - 7.5d ≤ 0
    
    Args:
        X (torch.Tensor): Input array of shape [n_samples, d] where d is the number of dimensions
        dimensions (int): Number of dimensions (optional, inferred from X if not provided)
        
    Returns:
        torch.Tensor: Keane Bump function values for each input sample
    """
    if dimensions is None:
        dimensions = X.shape[1]
    
    # Compute numerator: Σ_{i=1}^{d} cos⁴(x_i) - 2 Π_{i=1}^{d} cos²(x_i)
    cos_x = torch.cos(X)
    cos_squared = cos_x ** 2
    cos_fourth = cos_squared ** 2
    
    # Sum of cos⁴ terms
    sum_cos_fourth = torch.sum(cos_fourth, dim=1)
    
    # Product of cos² terms
    prod_cos_squared = torch.prod(cos_squared, dim=1)
    
    # Numerator
    numerator = sum_cos_fourth - 2 * prod_cos_squared
    
    # Compute denominator: √( Σ_{i=1}^{d} i * x_i² )
    # Create indices for each dimension: [1, 2, 3, ..., d]
    indices = torch.arange(1, dimensions + 1, dtype=X.dtype, device=X.device).unsqueeze(0)
    x_squared = X ** 2
    weighted_sum = torch.sum(indices * x_squared, dim=1)
    denominator = torch.sqrt(weighted_sum)
    
    # Avoid division by zero (add small epsilon)
    epsilon = 1e-10
    denominator = torch.clamp(denominator, min=epsilon)
    
    # Compute f(x) = - | numerator / denominator |
    result = -torch.abs(numerator / denominator)
    
    return result


def generate_keane_bump_data(n_train: int, n_test: int, dimensions: int = 30, x_bounds: list[float] = [0, 10], train_noise: float = 0.0, 
                              test_noise: float = 0.0, noise_type: str = 'gaussian', seed: int = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate train and test data for the Keane Bump function using Sobol sequences.
    
    Args:
        n_train (int): Number of training samples to generate
        n_test (int): Number of test samples to generate
        dimensions (int): Number of dimensions for the Keane Bump function (default: 30)
        train_noise (float): Noise level for training data as a fraction of std
        test_noise (float): Noise level for test data as a fraction of std
        noise_type (str): Type of noise ('gaussian' or 'uniform')
        seed (int): Random seed for reproducibility
        
    Returns:
        X_train, y_train, X_test, y_test: Train and test data
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    l_bound = x_bounds[0]
    u_bound = x_bounds[1]
    
    # Generate ALL samples at once to avoid repeats
    total_samples = n_train + n_test
    sobol = torch.quasirandom.SobolEngine(dimension=dimensions, scramble=True)
    X_all = sobol.draw(total_samples).to(dtype=torch.float64)
    
    # Scale to Keane Bump bounds [0, 10]
    X_all = X_all * (u_bound - l_bound) + l_bound
    
    # Compute Keane Bump function values
    y_all = keane_bump_function(X_all, dimensions)
    
    # Split into train and test
    X_train = X_all[:n_train]
    y_train = y_all[:n_train]
    X_test = X_all[n_train:]
    y_test = y_all[n_train:]
    
    # Add noise separately to train and test
    # Both train and test noise are based on TEST std
    y_test_std = y_test.std()
    
    if train_noise > 0:
        noise_scale = train_noise * y_test_std
        noise = _sample_noise_like(y_train, noise_scale, noise_type)
        y_train = y_train + noise
    
    if test_noise > 0:
        noise_scale = test_noise * y_test_std
        noise = _sample_noise_like(y_test, noise_scale, noise_type)
        y_test = y_test + noise
    
    return X_train, y_train, X_test, y_test


def rover_trajectory_function(X: torch.Tensor, num_design_points: int = 30, 
                               start_location: torch.Tensor = None, 
                               end_location: torch.Tensor = None,
                               obstacles: list = None) -> torch.Tensor:
    """
    Compute the rover trajectory reward function for given design points.
    
    The trajectory is defined by fitting a B-spline through 30 design points in a 2D plane.
    Reward function: f(x) = c(x) + 5, where c(x) is a penalty function that penalizes
    collisions with obstacles by -20.
    
    Args:
        X (torch.Tensor): Input array of shape [n_samples, 60] where 60 = 30 design points × 2D
        num_design_points (int): Number of design points (default: 30)
        start_location (torch.Tensor): Start location [x, y] (default: [0.0, 0.0])
        end_location (torch.Tensor): End location [x, y] (default: [10.0, 10.0])
        obstacles (list): List of obstacles, each as [center_x, center_y, radius]
        
    Returns:
        torch.Tensor: Reward function values for each input sample
    """
    if X.shape[1] != num_design_points * 2:
        raise ValueError(f"X must have {num_design_points * 2} dimensions (got {X.shape[1]})")
    
    n_samples = X.shape[0]
    
    # Default start and end locations
    if start_location is None:
        start_location = torch.tensor([0.0, 0.0], dtype=X.dtype, device=X.device)
    if end_location is None:
        end_location = torch.tensor([10.0, 10.0], dtype=X.dtype, device=X.device)
    
    # Default obstacles: 15 circular obstacles
    if obstacles is None:
        # Create 15 obstacles with random but fixed positions
        torch.manual_seed(42)  # Fixed seed for reproducible obstacles
        obstacles = []
        for i in range(15):
            center_x = 2.0 + (i % 5) * 2.0
            center_y = 2.0 + (i // 5) * 2.0
            radius = 0.5 + (i % 3) * 0.3
            obstacles.append([center_x, center_y, radius])
        obstacles = torch.tensor(obstacles, dtype=X.dtype, device=X.device)
    else:
        obstacles = torch.tensor(obstacles, dtype=X.dtype, device=X.device)
    
    # Reshape X to [n_samples, num_design_points, 2]
    design_points = X.reshape(n_samples, num_design_points, 2)
    
    # Create full trajectory including start and end points
    # Add start and end to design points
    start_expanded = start_location.unsqueeze(0).unsqueeze(0).expand(n_samples, 1, 2)
    end_expanded = end_location.unsqueeze(0).unsqueeze(0).expand(n_samples, 1, 2)
    full_points = torch.cat([start_expanded, design_points, end_expanded], dim=1)  # [n_samples, num_design_points+2, 2]
    
    # Simple B-spline approximation: use piecewise linear interpolation with smoothing
    # For simplicity, we'll sample points along the trajectory
    num_trajectory_samples = 100
    t = torch.linspace(0, 1, num_trajectory_samples, dtype=X.dtype, device=X.device)
    
    # Interpolate along the path using linear interpolation between control points
    trajectory_points = torch.zeros(n_samples, num_trajectory_samples, 2, dtype=X.dtype, device=X.device)
    
    num_control = full_points.shape[1]
    for i in range(n_samples):
        # Use linear interpolation between control points
        for j, t_val in enumerate(t):
            # Map t_val [0, 1] to segment index
            segment_idx = t_val * (num_control - 1)
            idx_low = torch.floor(segment_idx).long().clamp(0, num_control - 2)
            idx_high = (idx_low + 1).clamp(0, num_control - 1)
            alpha = (segment_idx - idx_low.float()).clamp(0, 1)
            
            point_low = full_points[i, idx_low]
            point_high = full_points[i, idx_high]
            trajectory_points[i, j] = (1 - alpha) * point_low + alpha * point_high
    
    # Compute collision penalty (vectorized)
    penalty = torch.zeros(n_samples, dtype=X.dtype, device=X.device)
    total_collisions = torch.zeros(n_samples, dtype=X.dtype, device=X.device)
    
    for obs_idx in range(obstacles.shape[0]):
        obs_center = obstacles[obs_idx, :2]  # [2]
        obs_radius = obstacles[obs_idx, 2]  # scalar
        
        # Compute distance from each trajectory point to obstacle center
        # trajectory_points: [n_samples, num_trajectory_samples, 2]
        # obs_center: [2]
        obs_center_expanded = obs_center.unsqueeze(0).unsqueeze(0)  # [1, 1, 2]
        distances = torch.norm(trajectory_points - obs_center_expanded, dim=2)  # [n_samples, num_trajectory_samples]
        
        # Check for collisions (distance < radius)
        collisions = distances < obs_radius  # [n_samples, num_trajectory_samples]
        has_collision = collisions.any(dim=1)  # [n_samples]
        
        # For samples with collision: compute maximum penetration depth
        penetration_depths = torch.clamp(obs_radius - distances, min=0.0)  # [n_samples, num_trajectory_samples]
        max_penetration = torch.max(penetration_depths, dim=1)[0]  # [n_samples]
        penalty += max_penetration * 10.0  # Scale penetration penalty
        
        # For samples without collision: negative of minimum distance (larger distance = better)
        min_dist = torch.min(distances, dim=1)[0]  # [n_samples]
        penalty -= min_dist * (~has_collision).float()  # Only apply to non-colliding samples
        
        # Count collisions
        total_collisions += has_collision.float()
    
    # Reward function: f(x) = c(x) + 5, where c(x) includes collision penalty
    collision_penalty = -20.0 * total_collisions
    reward = penalty + collision_penalty + 5.0
    
    return reward


def generate_rover_trajectory_data(n_train: int, n_test: int, dimensions: int = 60, 
                                    train_noise: float = 0.0, test_noise: float = 0.0, 
                                    noise_type: str = 'gaussian', seed: int = None,
                                    start_location: torch.Tensor = None,
                                    end_location: torch.Tensor = None,
                                    obstacles: list = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate train and test data for the 60D Rover Trajectory Planning problem using Sobol sequences.
    
    The problem involves optimizing a rover's trajectory defined by 30 design points in a 2D plane
    (60 dimensions total: 30 points × 2 coordinates).
    
    Args:
        n_train (int): Number of training samples to generate
        n_test (int): Number of test samples to generate
        dimensions (int): Number of dimensions (default: 60 for 30 design points × 2D)
        train_noise (float): Noise level for training data as a fraction of std
        test_noise (float): Noise level for test data as a fraction of std
        noise_type (str): Type of noise ('gaussian' or 'uniform')
        seed (int): Random seed for reproducibility
        start_location (torch.Tensor): Start location [x, y] (default: [0.0, 0.0])
        end_location (torch.Tensor): End location [x, y] (default: [10.0, 10.0])
        obstacles (list): List of obstacles, each as [center_x, center_y, radius]
        
    Returns:
        X_train, y_train, X_test, y_test: Train and test data
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Bounds for design points in 2D plane (reasonable range for trajectory planning)
    l_bound = 0.0
    u_bound = 10.0
    
    # Generate ALL samples at once to avoid repeats
    total_samples = n_train + n_test
    sobol = torch.quasirandom.SobolEngine(dimension=dimensions, scramble=True)
    X_all = sobol.draw(total_samples).to(dtype=torch.float64)
    
    # Scale to bounds [0, 10]
    X_all = X_all * (u_bound - l_bound) + l_bound
    
    # Compute rover trajectory reward function values
    y_all = rover_trajectory_function(
        X_all, 
        num_design_points=dimensions // 2,
        start_location=start_location,
        end_location=end_location,
        obstacles=obstacles
    )
    
    # Split into train and test
    X_train = X_all[:n_train]
    y_train = y_all[:n_train]
    X_test = X_all[n_train:]
    y_test = y_all[n_train:]
    
    # Add noise separately to train and test
    # Both train and test noise are based on TEST std
    y_test_std = y_test.std()
    
    if train_noise > 0:
        noise_scale = train_noise * y_test_std
        noise = _sample_noise_like(y_train, noise_scale, noise_type)
        y_train = y_train + noise
    
    if test_noise > 0:
        noise_scale = test_noise * y_test_std
        noise = _sample_noise_like(y_test, noise_scale, noise_type)
        y_test = y_test + noise
    
    return X_train, y_train, X_test, y_test


def powell_function(X: torch.Tensor, dimensions: int = None) -> torch.Tensor:
    """
    Compute the Powell function for given input variables.
    
    The Powell function is defined as:
    f(x) = Σ_{i=1}^{d/4} [ (x_{4i-3} + 10x_{4i-2})^2 + 5(x_{4i-1} - x_{4i})^2 + (x_{4i-2} - 2x_{4i-1})^4 + 10(x_{4i-3} - x_{4i})^4 ]
    
    where x ∈ [-4, 5]^d and d is the number of dimensions (must be a multiple of 4)
    
    Global minimum: f(x*) = 0, at x* = (0,..., 0)
    
    Args:
        X (torch.Tensor): Input array of shape [n_samples, d] where d is the number of dimensions
        dimensions (int): Number of dimensions (optional, inferred from X if not provided)
        
    Returns:
        torch.Tensor: Powell function values for each input sample
    """
    if dimensions is None:
        dimensions = X.shape[1]
    
    # Check that dimensions is a multiple of 4
    if dimensions % 4 != 0:
        raise ValueError(f"Dimensions must be a multiple of 4, got {dimensions}")
    
    n_terms = dimensions // 4
    result = torch.zeros(X.shape[0], dtype=X.dtype, device=X.device)
    
    # Compute each term in the sum
    for i in range(n_terms):
        # Indices: 4i-3, 4i-2, 4i-1, 4i (converted to 0-based: 4i-4, 4i-3, 4i-2, 4i-1)
        idx_base = i * 4
        x_4i_minus_3 = X[:, idx_base]      # x_{4i-3} (0-based: 4i-4)
        x_4i_minus_2 = X[:, idx_base + 1]  # x_{4i-2} (0-based: 4i-3)
        x_4i_minus_1 = X[:, idx_base + 2]  # x_{4i-1} (0-based: 4i-2)
        x_4i = X[:, idx_base + 3]          # x_{4i} (0-based: 4i-1)
        
        # Term 1: (x_{4i-3} + 10x_{4i-2})^2
        term1 = (x_4i_minus_3 + 10 * x_4i_minus_2) ** 2
        
        # Term 2: 5(x_{4i-1} - x_{4i})^2
        term2 = 5 * (x_4i_minus_1 - x_4i) ** 2
        
        # Term 3: (x_{4i-2} - 2x_{4i-1})^4
        term3 = (x_4i_minus_2 - 2 * x_4i_minus_1) ** 4
        
        # Term 4: 10(x_{4i-3} - x_{4i})^4
        term4 = 10 * (x_4i_minus_3 - x_4i) ** 4
        
        # Add all terms for this i
        result += term1 + term2 + term3 + term4
    
    return result


def generate_powell_data(n_train: int, n_test: int, dimensions: int = 4, x_bounds: list[float] = [-4, 5], train_noise: float = 0.0, 
                         test_noise: float = 0.0, noise_type: str = 'gaussian', seed: int = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate train and test data for the Powell function using Sobol sequences.
    
    Args:
        n_train (int): Number of training samples to generate
        n_test (int): Number of test samples to generate
        dimensions (int): Number of dimensions for the Powell function (must be a multiple of 4, default: 4)
        x_bounds (list[float]): Bounds for each dimension [lower, upper] (default: [-4, 5])
        train_noise (float): Noise level for training data as a fraction of std
        test_noise (float): Noise level for test data as a fraction of std
        noise_type (str): Type of noise ('gaussian' or 'uniform')
        seed (int): Random seed for reproducibility
        
    Returns:
        X_train, y_train, X_test, y_test: Train and test data
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Check that dimensions is a multiple of 4
    if dimensions % 4 != 0:
        raise ValueError(f"Dimensions must be a multiple of 4, got {dimensions}")
    
    l_bound = x_bounds[0]
    u_bound = x_bounds[1]
    
    # Generate ALL samples at once to avoid repeats
    total_samples = n_train + n_test
    sobol = torch.quasirandom.SobolEngine(dimension=dimensions, scramble=True)
    X_all = sobol.draw(total_samples).to(dtype=torch.float64)
    
    # Scale to Powell bounds
    X_all = X_all * (u_bound - l_bound) + l_bound
    
    # Compute Powell function values
    y_all = powell_function(X_all, dimensions)
    
    # Split into train and test
    X_train = X_all[:n_train]
    y_train = y_all[:n_train]
    X_test = X_all[n_train:]
    y_test = y_all[n_train:]
    
    # Add noise separately to train and test
    # Both train and test noise are based on TEST std
    y_test_std = y_test.std()
    
    if train_noise > 0:
        noise_scale = train_noise * y_test_std
        noise = _sample_noise_like(y_train, noise_scale, noise_type)
        y_train = y_train + noise
    
    if test_noise > 0:
        noise_scale = test_noise * y_test_std
        noise = _sample_noise_like(y_test, noise_scale, noise_type)
        y_test = y_test + noise
    
    return X_train, y_train, X_test, y_test


def griewank_function(X: torch.Tensor, dimensions: int = None) -> torch.Tensor:
    """
    Compute the Griewank function for given input variables.
    
    The Griewank function is defined as:
    f(x) = sum_{i=1}^{d} (x_i^2 / 4000) - product_{i=1}^{d} cos(x_i / sqrt(i)) + 1
    
    where x ∈ [-600, 600]^d and d is the number of dimensions
    
    Global minimum: f(x*) = 0, at x* = (0,...,0)
    
    Args:
        X (torch.Tensor): Input array of shape [n_samples, d] where d is the number of dimensions
        dimensions (int): Number of dimensions (optional, inferred from X if not provided)
        
    Returns:
        torch.Tensor: Griewank function values for each input sample
    """
    if dimensions is None:
        dimensions = X.shape[1]
    
    # First term: sum_{i=1}^{d} (x_i^2 / 4000)
    sum_squares = torch.sum(X**2, dim=1) / 4000.0
    
    # Second term: product_{i=1}^{d} cos(x_i / sqrt(i))
    # Create indices [1, 2, 3, ..., d] for each sample
    i_values = torch.arange(1, dimensions + 1, dtype=X.dtype, device=X.device)
    sqrt_i = torch.sqrt(i_values)  # [sqrt(1), sqrt(2), ..., sqrt(d)]
    
    # Compute cos(x_i / sqrt(i)) for each dimension
    cos_terms = torch.cos(X / sqrt_i.unsqueeze(0))  # [n_samples, d]
    
    # Product over all dimensions
    product_cos = torch.prod(cos_terms, dim=1)  # [n_samples]
    
    # Griewank function: sum_squares - product_cos + 1
    result = sum_squares - product_cos + 1.0
    
    return result


def generate_griewank_data(n_train: int, n_test: int, dimensions: int = 2, x_bounds: list[float] = [-600, 600], train_noise: float = 0.0, 
                            test_noise: float = 0.0, noise_type: str = 'gaussian', seed: int = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate train and test data for the Griewank function using Sobol sequences.
    
    Args:
        n_train (int): Number of training samples to generate
        n_test (int): Number of test samples to generate
        dimensions (int): Number of dimensions for the Griewank function
        x_bounds (list[float]): Bounds for each dimension [lower, upper] (default: [-600, 600])
        train_noise (float): Noise level for training data as a fraction of std
        test_noise (float): Noise level for test data as a fraction of std
        noise_type (str): Type of noise ('gaussian' or 'uniform')
        seed (int): Random seed for reproducibility
        
    Returns:
        X_train, y_train, X_test, y_test: Train and test data
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    l_bound = x_bounds[0]
    u_bound = x_bounds[1]
    
    # Generate ALL samples at once to avoid repeats
    total_samples = n_train + n_test
    sobol = torch.quasirandom.SobolEngine(dimension=dimensions, scramble=True)
    X_all = sobol.draw(total_samples).to(dtype=torch.float64)
    
    # Scale to Griewank bounds
    X_all = X_all * (u_bound - l_bound) + l_bound
    
    # Compute Griewank function values
    y_all = griewank_function(X_all, dimensions)
    
    # Split into train and test
    X_train = X_all[:n_train]
    y_train = y_all[:n_train]
    X_test = X_all[n_train:]
    y_test = y_all[n_train:]
    
    # Add noise separately to train and test
    # Both train and test noise are based on TEST std
    y_test_std = y_test.std()
    
    if train_noise > 0:
        noise_scale = train_noise * y_test_std
        noise = _sample_noise_like(y_train, noise_scale, noise_type)
        y_train = y_train + noise
    
    if test_noise > 0:
        noise_scale = test_noise * y_test_std
        noise = _sample_noise_like(y_test, noise_scale, noise_type)
        y_test = y_test + noise
    
    return X_train, y_train, X_test, y_test


def dixon_price_function(X: torch.Tensor, dimensions: int = None) -> torch.Tensor:
    """
    Compute the Dixon-Price function for given input variables.
    
    The Dixon-Price function is defined as:
    f(x) = (x_1 - 1)^2 + sum_{i=2}^{d} i * (2x_i^2 - x_{i-1})^2
    
    where x ∈ [-10, 10]^d and d is the number of dimensions
    
    Global minimum: f(x*) = 0, at x_i = 2^((2^(i-1) - 1) / 2^(i-1)) for i = 1, ..., d
    
    Args:
        X (torch.Tensor): Input array of shape [n_samples, d] where d is the number of dimensions
        dimensions (int): Number of dimensions (optional, inferred from X if not provided)
        
    Returns:
        torch.Tensor: Dixon-Price function values for each input sample
    """
    if dimensions is None:
        dimensions = X.shape[1]
    
    # First term: (x_1 - 1)^2
    first_term = (X[:, 0] - 1.0) ** 2
    
    # Sum term: sum_{i=2}^{d} i * (2x_i^2 - x_{i-1})^2
    # For i=2 to d: i * (2x_i^2 - x_{i-1})^2
    # x_i corresponds to X[:, i-1] (0-indexed)
    # x_{i-1} corresponds to X[:, i-2] (0-indexed)
    
    sum_term = torch.zeros(X.shape[0], dtype=X.dtype, device=X.device)
    
    for i in range(2, dimensions + 1):  # i from 2 to d
        x_i = X[:, i - 1]  # x_i (0-indexed: column i-1)
        x_i_minus_1 = X[:, i - 2]  # x_{i-1} (0-indexed: column i-2)
        
        # Compute (2x_i^2 - x_{i-1})^2
        term = (2.0 * x_i**2 - x_i_minus_1) ** 2
        
        # Multiply by i and add to sum
        sum_term += float(i) * term
    
    # Dixon-Price function: first_term + sum_term
    result = first_term + sum_term
    
    return result


def generate_dixon_price_data(n_train: int, n_test: int, dimensions: int = 2, x_bounds: list[float] = [-10, 10], train_noise: float = 0.0, 
                               test_noise: float = 0.0, noise_type: str = 'gaussian', seed: int = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate train and test data for the Dixon-Price function using Sobol sequences.
    
    Args:
        n_train (int): Number of training samples to generate
        n_test (int): Number of test samples to generate
        dimensions (int): Number of dimensions for the Dixon-Price function
        x_bounds (list[float]): Bounds for each dimension [lower, upper] (default: [-10, 10])
        train_noise (float): Noise level for training data as a fraction of std
        test_noise (float): Noise level for test data as a fraction of std
        noise_type (str): Type of noise ('gaussian' or 'uniform')
        seed (int): Random seed for reproducibility
        
    Returns:
        X_train, y_train, X_test, y_test: Train and test data
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    l_bound = x_bounds[0]
    u_bound = x_bounds[1]
    
    # Generate ALL samples at once to avoid repeats
    total_samples = n_train + n_test
    sobol = torch.quasirandom.SobolEngine(dimension=dimensions, scramble=True)
    X_all = sobol.draw(total_samples).to(dtype=torch.float64)
    
    # Scale to Dixon-Price bounds
    X_all = X_all * (u_bound - l_bound) + l_bound
    
    # Compute Dixon-Price function values
    y_all = dixon_price_function(X_all, dimensions)
    
    # Split into train and test
    X_train = X_all[:n_train]
    y_train = y_all[:n_train]
    X_test = X_all[n_train:]
    y_test = y_all[n_train:]
    
    # Add noise separately to train and test
    # Both train and test noise are based on TEST std
    y_test_std = y_test.std()
    
    if train_noise > 0:
        noise_scale = train_noise * y_test_std
        noise = _sample_noise_like(y_train, noise_scale, noise_type)
        y_train = y_train + noise
    
    if test_noise > 0:
        noise_scale = test_noise * y_test_std
        noise = _sample_noise_like(y_test, noise_scale, noise_type)
        y_test = y_test + noise
    
    return X_train, y_train, X_test, y_test


def styblinski_tang_function(X: torch.Tensor, dimensions: int = None) -> torch.Tensor:
    """
    Compute the Styblinski-Tang function for given input variables.
    
    The Styblinski-Tang function is defined as:
    f(x) = (1/2) * sum_{i=1}^{d} (x_i^4 - 16x_i^2 + 5x_i)
    
    where x ∈ [-5, 5]^d and d is the number of dimensions
    
    Global minimum: f(x*) = -39.16599d, at x* = (-2.903534, ..., -2.903534)
    
    Args:
        X (torch.Tensor): Input array of shape [n_samples, d] where d is the number of dimensions
        dimensions (int): Number of dimensions (optional, inferred from X if not provided)
        
    Returns:
        torch.Tensor: Styblinski-Tang function values for each input sample
    """
    if dimensions is None:
        dimensions = X.shape[1]
    
    # Compute (x_i^4 - 16x_i^2 + 5x_i) for each dimension
    x_squared = X ** 2
    x_fourth = x_squared ** 2
    term = x_fourth - 16.0 * x_squared + 5.0 * X
    
    # Sum over all dimensions
    sum_term = torch.sum(term, dim=1)
    
    # Multiply by 1/2
    result = 0.5 * sum_term
    
    return result


def generate_styblinski_tang_data(n_train: int, n_test: int, dimensions: int = 2, x_bounds: list[float] = [-5, 5], train_noise: float = 0.0, 
                                   test_noise: float = 0.0, noise_type: str = 'gaussian', seed: int = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate train and test data for the Styblinski-Tang function using Sobol sequences.
    
    Args:
        n_train (int): Number of training samples to generate
        n_test (int): Number of test samples to generate
        dimensions (int): Number of dimensions for the Styblinski-Tang function
        x_bounds (list[float]): Bounds for each dimension [lower, upper] (default: [-5, 5])
        train_noise (float): Noise level for training data as a fraction of std
        test_noise (float): Noise level for test data as a fraction of std
        noise_type (str): Type of noise ('gaussian' or 'uniform')
        seed (int): Random seed for reproducibility
        
    Returns:
        X_train, y_train, X_test, y_test: Train and test data
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    l_bound = x_bounds[0]
    u_bound = x_bounds[1]
    
    # Generate ALL samples at once to avoid repeats
    total_samples = n_train + n_test
    sobol = torch.quasirandom.SobolEngine(dimension=dimensions, scramble=True)
    X_all = sobol.draw(total_samples).to(dtype=torch.float64)
    
    # Scale to Styblinski-Tang bounds
    X_all = X_all * (u_bound - l_bound) + l_bound
    
    # Compute Styblinski-Tang function values
    y_all = styblinski_tang_function(X_all, dimensions)
    
    # Split into train and test
    X_train = X_all[:n_train]
    y_train = y_all[:n_train]
    X_test = X_all[n_train:]
    y_test = y_all[n_train:]
    
    # Add noise separately to train and test
    # Both train and test noise are based on TEST std
    y_test_std = y_test.std()
    
    if train_noise > 0:
        noise_scale = train_noise * y_test_std
        noise = _sample_noise_like(y_train, noise_scale, noise_type)
        y_train = y_train + noise
    
    if test_noise > 0:
        noise_scale = test_noise * y_test_std
        noise = _sample_noise_like(y_test, noise_scale, noise_type)
        y_test = y_test + noise
    
    return X_train, y_train, X_test, y_test


def load_dns_rom_data_all(print_info=False):
    """
    Load all DNS ROM multi-fidelity data from CSV files without splitting.
    
    CSV files are located in experiments_kian/data/DNS_ROM/:
    - Data_high.csv (source 0)
    - Data_LF1.csv (source 1)
    - Data_LF2.csv (source 2)
    - Data_LF3.csv (source 3)
    
    Each CSV has format: [6 input features, source_id, y]
    
    Args:
        print_info (bool): If True, prints information about the loaded data
        
    Returns:
        X, y: All data with 7 features (6 continuous + 1 source column in {0,1,2,3})
    """
    # Path to DNS ROM data directory
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'DNS_ROM')
    data_dir = os.path.abspath(data_dir)
    
    # Load CSV files
    csv_files = [
        os.path.join(data_dir, 'Data_high.csv'),  # source 0
        os.path.join(data_dir, 'Data_LF1.csv'),  # source 1
        os.path.join(data_dir, 'Data_LF2.csv'),  # source 2
        os.path.join(data_dir, 'Data_LF3.csv'),  # source 3
    ]
    
    # Load and combine all data
    all_data_list = []
    for source_idx, csv_path in enumerate(csv_files):
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Could not find DNS ROM data file: {csv_path}")
        
        df = pd.read_csv(csv_path, header=None)
        arr = df.values.astype(np.float64)
        
        # Verify format: should have 8 columns (6 features + source + y)
        if arr.shape[1] != 8:
            raise ValueError(f"Expected 8 columns in {csv_path}, got {arr.shape[1]}")
        
        # Verify source column matches expected source
        source_col = arr[:, 6].astype(int)
        if not np.all(source_col == source_idx):
            # If source column doesn't match, set it to the expected source
            arr[:, 6] = source_idx
        
        all_data_list.append(arr)
    
    # Combine all sources
    all_data = np.vstack(all_data_list)
    
    # Extract features (first 6 columns), source (column 6), and y (column 7)
    X_features = torch.tensor(all_data[:, :6], dtype=torch.float64)
    source_ids = torch.tensor(all_data[:, 6], dtype=torch.float64)
    y_all = torch.tensor(all_data[:, 7], dtype=torch.float64)
    
    # Append source id as 7th feature
    source_col = source_ids.unsqueeze(1)
    X = torch.cat([X_features, source_col], dim=1)
    
    if print_info:
        print(f"Loaded DNS ROM data:")
        print(f"  Total samples: {len(X)}")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y_all.shape}")
        for source_idx in range(4):
            source_mask = source_ids == source_idx
            n_samples = source_mask.sum().item()
            print(f"  Source {source_idx}: {n_samples} samples")
    
    return X, y_all


def load_dns_rom_hf_data(print_info=False):
    """
    Load only high-fidelity DNS ROM data from Data_high.csv (single-fidelity version).
    
    CSV file is located in experiments_kian/data/DNS_ROM/:
    - Data_high.csv
    
    Each CSV has format: [6 input features, source_id, y]
    
    Args:
        print_info (bool): If True, prints information about the loaded data
        
    Returns:
        X, y: Data with 6 features (no source column)
    """
    # Path to DNS ROM data directory
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'DNS_ROM')
    data_dir = os.path.abspath(data_dir)
    
    # Load only high-fidelity CSV file
    csv_path = os.path.join(data_dir, 'Data_high.csv')
    
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Could not find DNS ROM data file: {csv_path}")
    
    df = pd.read_csv(csv_path, header=None)
    arr = df.values.astype(np.float64)
    
    # Verify format: should have 8 columns (6 features + source + y)
    if arr.shape[1] != 8:
        raise ValueError(f"Expected 8 columns in {csv_path}, got {arr.shape[1]}")
    
    # Extract features (first 6 columns) and y (column 7)
    # Don't include source column for single-fidelity
    X = torch.tensor(arr[:, :6], dtype=torch.float64)
    y = torch.tensor(arr[:, 7], dtype=torch.float64)
    
    if print_info:
        print(f"Loaded DNS ROM high-fidelity data:")
        print(f"  Total samples: {len(X)}")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
    
    return X, y
