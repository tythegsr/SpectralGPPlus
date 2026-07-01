"""Save/load full TOA RFFGPR (single-task) checkpoints for later inference."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

_ROOT = Path(__file__).resolve().parents[1]
_MTGPR_DIR = _ROOT / "experiments_RFFMTGPR"
if str(_MTGPR_DIR) not in sys.path:
    sys.path.insert(0, str(_MTGPR_DIR))

from gpplus.models import RFFGPR
from gpplus.utils import StandardScaler, UniformScaler
from toa_mtgpr_checkpoint import CHECKPOINT_VERSION, scaler_from_dict, scaler_to_dict


@dataclass
class ToaStgpBundle:
    model: RFFGPR
    task_name: str
    x_scaler: StandardScaler | UniformScaler | None
    y_scaler: StandardScaler | None
    standardize_x: bool
    standardize_y: bool
    x_standardize_method: int
    train_idx: torch.Tensor
    val_idx: torch.Tensor
    test_idx: torch.Tensor
    title: str
    seed: int
    best_train_loss: float
    n_train: int
    n_test: int
    n_val: int
    data_path: str | None
    rel_tolerance: float
    dtype: torch.dtype
    log_grain: bool = False
    input_column_indices: torch.Tensor | None = None
    rff_sampling: str = "rff"


def save_toa_stgp_checkpoint(
    path: str | Path,
    *,
    model: RFFGPR,
    task_name: str,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    x_scaler: StandardScaler | UniformScaler | None,
    y_scaler: StandardScaler | None,
    standardize_x: bool,
    standardize_y: bool,
    x_standardize_method: int,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    title: str,
    seed: int,
    best_train_loss: float,
    n_train: int,
    n_test: int,
    n_val: int,
    data_path: str | None,
    rel_tolerance: float,
    dtype: torch.dtype,
    model_config: dict[str, Any],
    log_grain: bool = False,
    input_column_indices: torch.Tensor | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if input_column_indices is None:
        input_column_indices = torch.arange(train_x.shape[-1], dtype=torch.int64)
    payload = {
        "version": CHECKPOINT_VERSION,
        "model_class": "RFFGPR",
        "task_name": task_name,
        "title": title,
        "seed": seed,
        "best_train_loss": best_train_loss,
        "n_train": n_train,
        "n_test": n_test,
        "n_val": n_val,
        "data_path": os.path.abspath(data_path) if data_path else None,
        "rel_tolerance": rel_tolerance,
        "dtype": str(dtype).replace("torch.", ""),
        "model_config": model_config,
        "standardize_x": standardize_x,
        "x_standardize_method": x_standardize_method,
        "standardize_y": standardize_y,
        "log_grain": log_grain,
        "input_column_indices": input_column_indices.detach().cpu().to(torch.int64),
        "x_scaler": scaler_to_dict(x_scaler),
        "y_scaler": scaler_to_dict(y_scaler),
        "train_x": train_x.detach().cpu(),
        "train_y": train_y.detach().cpu(),
        "train_idx": train_idx.detach().cpu().to(torch.int64),
        "val_idx": val_idx.detach().cpu().to(torch.int64),
        "test_idx": test_idx.detach().cpu().to(torch.int64),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    torch.save(payload, path)
    return path


def _dtype_from_str(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"Unsupported checkpoint dtype: {name!r}")


def load_toa_stgp_checkpoint(path: str | Path, device: str = "cpu") -> ToaStgpBundle:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No checkpoint found at {path}")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    version = payload.get("version")
    if version != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported checkpoint version {version!r} (expected {CHECKPOINT_VERSION}).")

    dtype = _dtype_from_str(payload["dtype"])
    model_config = dict(payload["model_config"])
    train_x = payload["train_x"].to(dtype=dtype, device=device)
    train_y = payload["train_y"].to(dtype=dtype, device=device)

    model = RFFGPR(train_x, train_y, **model_config)
    model.load_state_dict(payload["state_dict"])
    model = model.to(device=device, dtype=dtype)
    model.eval()
    model.invalidate_feature_cache()

    x_scaler = scaler_from_dict(payload.get("x_scaler"))
    y_scaler = scaler_from_dict(payload.get("y_scaler"))
    if "input_column_indices" in payload:
        input_column_indices = payload["input_column_indices"].to(torch.int64)
    else:
        input_column_indices = torch.arange(train_x.shape[-1], dtype=torch.int64)

    rff_sampling = str(model_config.get("rff_sampling", "rff"))

    return ToaStgpBundle(
        model=model,
        task_name=str(payload["task_name"]),
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        standardize_x=bool(payload["standardize_x"]),
        standardize_y=bool(payload["standardize_y"]),
        x_standardize_method=int(payload["x_standardize_method"]),
        train_idx=payload["train_idx"],
        val_idx=payload["val_idx"],
        test_idx=payload["test_idx"],
        title=str(payload["title"]),
        seed=int(payload["seed"]),
        best_train_loss=float(payload["best_train_loss"]),
        n_train=int(payload["n_train"]),
        n_test=int(payload["n_test"]),
        n_val=int(payload["n_val"]),
        data_path=payload.get("data_path"),
        rel_tolerance=float(payload.get("rel_tolerance", 0.01)),
        dtype=dtype,
        log_grain=bool(payload.get("log_grain", False)),
        input_column_indices=input_column_indices,
        rff_sampling=rff_sampling,
    )


def checkpoint_path_for_run(save_path: str | Path, title: str, task_name: str) -> Path:
    return Path(save_path) / f"checkpoint_{title}_{task_name}.pt"
