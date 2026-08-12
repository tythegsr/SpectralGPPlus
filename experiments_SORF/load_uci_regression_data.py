"""Download-on-first-run loaders for UCI / Kaggle regression benchmarks."""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = _REPO_ROOT / "experiments" / "data"

FISH_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00504/qsar_fish_toxicity.csv"
)
CONCRETE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/"
    "Concrete_Data.xls"
)
AIRFOIL_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat"
)
FIAT_KAGGLE_SLUG = "paolocons/another-fiat-500-dataset-1538-rows"
FIAT_CSV_NAME = "automobile_dot_it_used_fiat_500_in_Italy_dataset_filtered.csv"

DATASET_DIRS = {
    "fish": "QSAR_fish_toxicity",
    "concrete": "concrete_compressive_strength",
    "fiat": "Another-Dataset-on-used-Fiat-500",
    "airfoil": "airfoil_self_noise",
}

DATASET_TARGETS = {
    "fish": "LC50",
    "concrete": "Concrete compressive strength",
    "fiat": "price",
    "airfoil": "sound_pressure_level",
}


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download_url(url: str, dest: Path) -> Path:
    dest = Path(dest)
    _ensure_dir(dest.parent)
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    print(f"[download] {url} -> {dest}")
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise
    return dest


def _to_xy_tensors(arr: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected 2D array with >=2 columns, got shape {arr.shape}")
    X = torch.tensor(arr[:, :-1], dtype=torch.float64)
    y = torch.tensor(arr[:, -1], dtype=torch.float64)
    return X, y


def load_qsar_fish_toxicity(print_info: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Load QSAR fish toxicity (UCI 504). Target: LC50."""
    cache_dir = _ensure_dir(DATA_ROOT / DATASET_DIRS["fish"])
    csv_path = cache_dir / "qsar_fish_toxicity.csv"
    _download_url(FISH_URL, csv_path)
    df = pd.read_csv(csv_path, sep=";", header=None)
    arr = df.to_numpy(dtype=np.float64)
    X, y = _to_xy_tensors(arr)
    if print_info:
        print(f"[fish] X={tuple(X.shape)}, y={tuple(y.shape)} from {csv_path}")
    return X, y


def _load_concrete_via_openml(csv_path: Path) -> pd.DataFrame:
    from sklearn.datasets import fetch_openml

    # OpenML data_id 44959 == UCI concrete compressive strength (n=1030, 8 features).
    print("[concrete] downloading via OpenML data_id=44959")
    bundle = fetch_openml(data_id=44959, as_frame=True, parser="auto")
    X_df = bundle.data.copy()
    y = bundle.target
    frame = X_df.copy()
    target_name = y.name if getattr(y, "name", None) else "strength"
    frame[target_name] = y.to_numpy()
    frame.to_csv(csv_path, index=False)
    print(f"[concrete] cached CSV at {csv_path}")
    return frame


def load_concrete_compressive_strength(
    print_info: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load concrete compressive strength (UCI 165 / OpenML 44959). Target: strength (MPa)."""
    cache_dir = _ensure_dir(DATA_ROOT / DATASET_DIRS["concrete"])
    csv_path = cache_dir / "Concrete_Data.csv"
    if csv_path.is_file() and csv_path.stat().st_size > 0:
        df = pd.read_csv(csv_path)
    else:
        # Prefer OpenML: UCI ships .xls and modern pandas/xlrd cannot read it reliably.
        try:
            df = _load_concrete_via_openml(csv_path)
        except Exception as openml_exc:
            xls_path = cache_dir / "Concrete_Data.xls"
            try:
                _download_url(CONCRETE_URL, xls_path)
                df = None
                for engine in ("xlrd", "openpyxl", None):
                    try:
                        df = pd.read_excel(xls_path, engine=engine)
                        break
                    except Exception:
                        continue
                if df is None:
                    raise RuntimeError("pandas could not read Concrete_Data.xls") from openml_exc
                df.to_csv(csv_path, index=False)
                print(f"[concrete] cached CSV at {csv_path}")
            except Exception as xls_exc:
                raise RuntimeError(
                    "Failed to load concrete dataset via OpenML and UCI .xls. "
                    f"OpenML error: {openml_exc}; XLS error: {xls_exc}"
                ) from xls_exc
    arr = df.to_numpy(dtype=np.float64)
    X, y = _to_xy_tensors(arr)
    if print_info:
        print(f"[concrete] X={tuple(X.shape)}, y={tuple(y.shape)} from {csv_path}")
    return X, y


def load_airfoil_self_noise(print_info: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Load airfoil self-noise (UCI 291). Target: scaled sound pressure level (dB)."""
    cache_dir = _ensure_dir(DATA_ROOT / DATASET_DIRS["airfoil"])
    dat_path = cache_dir / "airfoil_self_noise.dat"
    _download_url(AIRFOIL_URL, dat_path)
    arr = np.loadtxt(dat_path, dtype=np.float64)
    X, y = _to_xy_tensors(arr)
    if print_info:
        print(f"[airfoil] X={tuple(X.shape)}, y={tuple(y.shape)} from {dat_path}")
    return X, y


def _find_fiat_csv(search_root: Path) -> Path | None:
    preferred = search_root / FIAT_CSV_NAME
    if preferred.is_file():
        return preferred
    matches = list(search_root.rglob("*.csv"))
    if not matches:
        return None
    for path in matches:
        if path.name == FIAT_CSV_NAME:
            return path
    return matches[0]


def _download_fiat_via_kagglehub(cache_dir: Path) -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is required to download the Fiat 500 dataset. "
            "Install it (`pip install kagglehub`) or place the CSV manually at "
            f"{cache_dir / FIAT_CSV_NAME}"
        ) from exc

    try:
        downloaded = Path(kagglehub.dataset_download(FIAT_KAGGLE_SLUG))
    except Exception as exc:
        raise RuntimeError(
            "Failed to download Fiat 500 via kagglehub. Ensure Kaggle credentials "
            f"(KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json) are set, or place "
            f"the CSV at {cache_dir / FIAT_CSV_NAME}. Original error: {exc}"
        ) from exc

    found = _find_fiat_csv(downloaded)
    if found is None:
        raise RuntimeError(
            f"kagglehub downloaded {downloaded} but no CSV was found. "
            f"Place {FIAT_CSV_NAME} at {cache_dir / FIAT_CSV_NAME}."
        )
    dest = cache_dir / FIAT_CSV_NAME
    if found.resolve() != dest.resolve():
        shutil.copy2(found, dest)
    return dest


def load_fiat_500(print_info: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Load used Fiat 500 (Kaggle). Target: price. One-hot encodes `model`."""
    cache_dir = _ensure_dir(DATA_ROOT / DATASET_DIRS["fiat"])
    csv_path = cache_dir / FIAT_CSV_NAME
    if not (csv_path.is_file() and csv_path.stat().st_size > 0):
        csv_path = _download_fiat_via_kagglehub(cache_dir)

    df = pd.read_csv(csv_path)
    if "price" not in df.columns:
        raise ValueError(f"Fiat CSV missing 'price' column. Columns: {list(df.columns)}")

    y = df["price"].to_numpy(dtype=np.float64)
    feature_df = df.drop(columns=["price"])
    if "model" in feature_df.columns:
        feature_df = pd.get_dummies(feature_df, columns=["model"], drop_first=False)
    # Coerce any remaining object columns.
    for col in feature_df.columns:
        if feature_df[col].dtype == object:
            feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce")
    if feature_df.isna().any().any():
        n_bad = int(feature_df.isna().any(axis=1).sum())
        raise ValueError(f"Fiat features contain {n_bad} rows with NaNs after encoding.")

    X = torch.tensor(feature_df.to_numpy(dtype=np.float64), dtype=torch.float64)
    y_t = torch.tensor(y, dtype=torch.float64)
    if print_info:
        print(
            f"[fiat] X={tuple(X.shape)}, y={tuple(y_t.shape)} from {csv_path}; "
            f"columns={list(feature_df.columns)}"
        )
    return X, y_t


_LOADERS = {
    "fish": load_qsar_fish_toxicity,
    "concrete": load_concrete_compressive_strength,
    "fiat": load_fiat_500,
    "airfoil": load_airfoil_self_noise,
}


def load_dataset(
    dataset: str,
    print_info: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = dataset.strip().lower()
    if key not in _LOADERS:
        raise ValueError(f"Unknown dataset {dataset!r}. Choose from {sorted(_LOADERS)}")
    return _LOADERS[key](print_info=print_info)


def make_train_test_split(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    train_frac: float = 2.0 / 3.0,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shuffle once and split into train/test with the given train fraction."""
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")
    n = int(X.shape[0])
    if n < 2:
        raise ValueError(f"Need at least 2 samples to split, got {n}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = max(1, min(n - 1, int(round(train_frac * n))))
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def load_train_test(
    dataset: str,
    *,
    train_frac: float = 2.0 / 3.0,
    seed: int = 42,
    print_info: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    X, y = load_dataset(dataset, print_info=print_info)
    X_train, y_train, X_test, y_test = make_train_test_split(
        X, y, train_frac=train_frac, seed=seed
    )
    if print_info:
        print(
            f"[{dataset}] split seed={seed} train_frac={train_frac:.4f}: "
            f"n_train={X_train.shape[0]}, n_test={X_test.shape[0]}"
        )
    return X_train, y_train, X_test, y_test
