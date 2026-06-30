"""Save/load full TOA RFFMTGPR checkpoints for later inference."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from gpplus.models import RFFMTGPR
from gpplus.utils import StandardScaler, UniformScaler

CHECKPOINT_VERSION = 1


def scaler_to_dict(scaler: StandardScaler | UniformScaler | None) -> dict[str, Any] | None:
    if scaler is None:
        return None
    if isinstance(scaler, StandardScaler):
        if scaler.mean is None or scaler.std is None:
            raise ValueError("StandardScaler has not been fitted.")
        return {
            "type": "StandardScaler",
            "mean": scaler.mean.detach().cpu(),
            "std": scaler.std.detach().cpu(),
        }
    if isinstance(scaler, UniformScaler):
        if scaler.min is None or scaler.max is None:
            raise ValueError("UniformScaler has not been fitted.")
        return {
            "type": "UniformScaler",
            "min": scaler.min.detach().cpu(),
            "max": scaler.max.detach().cpu(),
            "scale_min": scaler.scale_min,
            "scale_max": scaler.scale_max,
        }
    raise TypeError(f"Unsupported scaler type: {type(scaler)!r}")


def scaler_from_dict(data: dict[str, Any] | None) -> StandardScaler | UniformScaler | None:
    if data is None:
        return None
    kind = data["type"]
    if kind == "StandardScaler":
        scaler = StandardScaler()
        scaler.mean = data["mean"]
        scaler.std = data["std"]
        return scaler
    if kind == "UniformScaler":
        scale_min = data["scale_min"]
        scale_max = data["scale_max"]
        scale_to_neg_one = scale_min == -1 and scale_max == 1
        scaler = UniformScaler(scale_to_neg_one=scale_to_neg_one)
        if not scale_to_neg_one:
            scaler.feature_range = (scale_min, scale_max)
        scaler.min = data["min"]
        scaler.max = data["max"]
        scaler.scale_min = scale_min
        scaler.scale_max = scale_max
        return scaler
    raise ValueError(f"Unknown scaler type: {kind!r}")


@dataclass
class ToaMtgprBundle:
    model: RFFMTGPR
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


def save_toa_mtgpr_checkpoint(
    path: str | Path,
    *,
    model: RFFMTGPR,
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


def load_toa_mtgpr_checkpoint(path: str | Path, device: str = "cpu") -> ToaMtgprBundle:
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

    model = RFFMTGPR(train_x, train_y, **model_config)
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

    return ToaMtgprBundle(
        model=model,
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
    )


def checkpoint_path_for_run(save_path: str | Path, title: str) -> Path:
    return Path(save_path) / f"checkpoint_{title}.pt"
