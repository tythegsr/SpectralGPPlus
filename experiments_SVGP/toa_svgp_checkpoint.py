"""Save/load full TOA single-task SVGP checkpoints for inference."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

_ROOT = Path(__file__).resolve().parents[1]
_SVGP_DIR = Path(__file__).resolve().parent
_GP_DIR = _ROOT / "experiments_GP"
_MTGPR_DIR = _ROOT / "experiments_RFFMTGPR"
_PCA_DIR = _ROOT / "experiments_PCA"

for _p in (_ROOT, _GP_DIR, _MTGPR_DIR, _PCA_DIR, _SVGP_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gpplus.models import SVGPR
from gpplus.utils import StandardScaler, UniformScaler
from gpplus.utils.fs_path import ensure_parent, fs_path
from toa_mtgpr_checkpoint import CHECKPOINT_VERSION, scaler_from_dict, scaler_to_dict

_SVGPR_INIT_KEYS = (
    "num_inducing",
    "learn_inducing_locations",
    "nigp",
    "seed",
)


def _fs_path(path: Path) -> str:
    return fs_path(path)


@dataclass
class ToaSvgpBundle:
    model: SVGPR
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
    input_column_indices: torch.Tensor | None = None
    pca_meta: dict[str, Any] | None = None
    x_transform: str = "none"
    log_scale_qoi: list[str] | None = None
    logit_scale_qoi: list[str] | None = None
    input_variable: str = "toa_reflectance"


def save_toa_svgp_checkpoint(
    path: str | Path,
    *,
    model: SVGPR,
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
    input_column_indices: torch.Tensor | None = None,
    pca_meta: dict[str, Any] | None = None,
    x_transform: str = "none",
    log_scale_qoi: list[str] | None = None,
    logit_scale_qoi: list[str] | None = None,
    input_variable: str = "toa_reflectance",
) -> Path:
    path = Path(path)
    ensure_parent(path)
    if input_column_indices is None:
        input_column_indices = torch.arange(train_x.shape[-1], dtype=torch.int64)
    payload = {
        "version": CHECKPOINT_VERSION,
        "model_class": "SVGPR",
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
        "input_column_indices": input_column_indices.detach().cpu().to(torch.int64),
        "pca_meta": pca_meta,
        "x_transform": x_transform or "none",
        "log_scale_qoi": list(log_scale_qoi or []),
        "logit_scale_qoi": list(logit_scale_qoi or []),
        "input_variable": input_variable,
        "x_scaler": scaler_to_dict(x_scaler),
        "y_scaler": scaler_to_dict(y_scaler),
        "train_x": train_x.detach().cpu(),
        "train_y": train_y.detach().cpu(),
        "train_idx": train_idx.detach().cpu().to(torch.int64),
        "val_idx": val_idx.detach().cpu().to(torch.int64),
        "test_idx": test_idx.detach().cpu().to(torch.int64),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    torch.save(payload, _fs_path(path))
    return path.resolve()


def _dtype_from_str(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"Unsupported checkpoint dtype: {name!r}")


def load_toa_svgp_checkpoint(path: str | Path, device: str = "cpu") -> ToaSvgpBundle:
    path = Path(path)
    fs = _fs_path(path)
    if not os.path.isfile(fs):
        raise FileNotFoundError(f"No checkpoint found at {path}")

    payload = torch.load(fs, map_location="cpu", weights_only=False)
    version = payload.get("version")
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version {version!r} (expected {CHECKPOINT_VERSION})."
        )

    dtype = _dtype_from_str(payload["dtype"])
    model_config = dict(payload["model_config"])
    train_x = payload["train_x"].to(dtype=dtype, device=device)
    train_y = payload["train_y"].to(dtype=dtype, device=device)

    init_kwargs = {k: v for k, v in model_config.items() if k in _SVGPR_INIT_KEYS}
    model = SVGPR(train_x, train_y, **init_kwargs)
    model.load_state_dict(payload["state_dict"])
    model = model.to(device=device, dtype=dtype)
    model.eval()

    x_scaler = scaler_from_dict(payload.get("x_scaler"))
    y_scaler = scaler_from_dict(payload.get("y_scaler"))
    if "input_column_indices" in payload:
        input_column_indices = payload["input_column_indices"].to(torch.int64)
    else:
        input_column_indices = torch.arange(train_x.shape[-1], dtype=torch.int64)

    return ToaSvgpBundle(
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
        input_column_indices=input_column_indices,
        pca_meta=payload.get("pca_meta"),
        x_transform=str(payload.get("x_transform", "none")),
        log_scale_qoi=list(payload.get("log_scale_qoi") or []),
        logit_scale_qoi=list(payload.get("logit_scale_qoi") or []),
        input_variable=str(payload.get("input_variable", "toa_reflectance")),
    )


def checkpoint_path_for_run(save_path: str | Path, title: str, task_name: str) -> Path:
    return Path(save_path) / f"checkpoint_{title}_{task_name}.pt"
