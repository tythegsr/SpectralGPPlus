"""Marginal slice plots: y vs one input with others fixed (GP RFF + TabPFN + truth)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from gpplus.training import evaluate_rff_gp_model

WING_FEATURE_NAMES = (
    "Sw",
    "Wfw",
    "A",
    "Gama",
    "q",
    "lamb",
    "tc",
    "Nz",
    "Wdg",
    "Wp",
)

WING_FEATURE_LABELS = (
    "Sw (sq ft)",
    "Wfw (lb)",
    "A",
    "Gama (deg)",
    "q (lb/sq ft)",
    "lamb",
    "tc",
    "Nz",
    "Wdg (lb)",
    "Wp (lb/sq ft)",
)

WING_L_BOUND = torch.tensor(
    [150.0, 220.0, 6.0, -10.0, 16.0, 0.5, 0.08, 2.5, 1700.0, 0.025],
    dtype=torch.float64,
)
WING_U_BOUND = torch.tensor(
    [200.0, 300.0, 10.0, 10.0, 45.0, 1.0, 0.18, 6.0, 2500.0, 0.08],
    dtype=torch.float64,
)


def sanitize_plot_subdir(title: str) -> str:
    t = (title or "experiment").strip()
    for c in '\\/:*?"<>|':
        t = t.replace(c, "_")
    return t.rstrip(" .")


def resolve_fixed_row(
    X_train_orig: torch.Tensor,
    *,
    fixed_point: str = "median",
    custom_row: torch.Tensor | None = None,
) -> torch.Tensor:
    if custom_row is not None:
        return custom_row.detach().clone().reshape(-1)
    if fixed_point == "median":
        return X_train_orig.median(dim=0).values.detach().clone()
    if fixed_point == "mean":
        return X_train_orig.mean(dim=0).detach().clone()
    raise ValueError(f"fixed_point must be 'median' or 'mean', got {fixed_point!r}")


def build_slice_grid(
    fixed_row: torch.Tensor,
    dim: int,
    l_bound: torch.Tensor,
    u_bound: torch.Tensor,
    n_grid: int,
) -> torch.Tensor:
    grid = torch.linspace(
        float(l_bound[dim].item()),
        float(u_bound[dim].item()),
        n_grid,
        dtype=fixed_row.dtype,
    )
    rows = []
    for val in grid:
        row = fixed_row.clone()
        row[dim] = val
        rows.append(row)
    return torch.stack(rows, dim=0), grid


def predict_rff_denormalized(
    model,
    X_scaled: torch.Tensor,
    *,
    y_mean: torch.Tensor | float | None,
    y_std: torch.Tensor | float | None,
    chunk_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    model.invalidate_feature_cache()
    mean, _, _, std = evaluate_rff_gp_model(model, X_scaled, chunk_size=chunk_size)
    if y_mean is not None and y_std is not None:
        if not isinstance(y_mean, torch.Tensor):
            y_mean = torch.tensor(float(y_mean), dtype=mean.dtype, device=mean.device)
        if not isinstance(y_std, torch.Tensor):
            y_std = torch.tensor(float(y_std), dtype=std.dtype, device=std.device)
        y_mean = y_mean.to(device=mean.device, dtype=mean.dtype).squeeze()
        y_std = y_std.to(device=std.device, dtype=std.dtype).squeeze()
        mean = mean * y_std + y_mean
        std = std * y_std
    return mean, std


def predict_tabpfn_on_grid(regressor, X_orig_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X_np = np.asarray(X_orig_np, dtype=np.float64)
    full = regressor.predict(X_np, output_type="full", quantiles=[0.025, 0.975])
    mean = full.get("mean")
    if mean is None:
        raise RuntimeError("TabPFN predict(output_type='full') did not return 'mean'")
    if isinstance(mean, torch.Tensor):
        mean = mean.detach().cpu().numpy()
    mean = np.asarray(mean, dtype=np.float64).ravel()

    logits = full.get("logits")
    criterion = full.get("criterion")
    if logits is None or criterion is None or not hasattr(criterion, "variance"):
        raise RuntimeError("TabPFN full output missing logits/criterion for variance")
    if isinstance(logits, np.ndarray):
        logits = torch.tensor(logits)
    variance = criterion.variance(logits)
    if isinstance(variance, torch.Tensor):
        variance = variance.detach().cpu().numpy()
    std = np.sqrt(np.asarray(variance, dtype=np.float64).ravel())
    return mean, std


def save_gp_tabpfn_marginal_slices(
    *,
    rff_model,
    tabpfn_regressor,
    X_train_orig: torch.Tensor,
    x_scaler,
    standardize_x: bool,
    y_mean: torch.Tensor | float | None,
    y_std: torch.Tensor | float | None,
    truth_fn: Callable[[torch.Tensor], torch.Tensor],
    out_dir: str | Path,
    title: str,
    slice_dims: list[int] | None = None,
    l_bound: torch.Tensor | None = None,
    u_bound: torch.Tensor | None = None,
    feature_names: tuple[str, ...] | None = None,
    feature_labels: tuple[str, ...] | None = None,
    fixed_point: str = "median",
    custom_fixed_row: torch.Tensor | None = None,
    n_grid: int = 200,
    interval_z: float = 1.96,
    predict_chunk_size: int = 512,
) -> list[Path]:
    """
    One PNG per swept dimension: true wing function vs RFF-GP vs TabPFN with 95% PI bands.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    l_bound = l_bound if l_bound is not None else WING_L_BOUND
    u_bound = u_bound if u_bound is not None else WING_U_BOUND
    feature_names = feature_names or WING_FEATURE_NAMES
    feature_labels = feature_labels or WING_FEATURE_LABELS
    n_dims = X_train_orig.shape[1]
    if slice_dims is None:
        slice_dims = list(range(n_dims))

    fixed_row = resolve_fixed_row(
        X_train_orig,
        fixed_point=fixed_point,
        custom_row=custom_fixed_row,
    )
    device_model = next(rff_model.parameters()).device
    saved: list[Path] = []

    for dim in slice_dims:
        if dim < 0 or dim >= n_dims:
            raise ValueError(f"slice dim {dim} out of range for input_dim={n_dims}")

        X_orig, grid = build_slice_grid(fixed_row, dim, l_bound, u_bound, n_grid)
        x_np = grid.detach().cpu().numpy().ravel()

        X_scaled = X_orig.clone()
        if standardize_x and x_scaler is not None:
            X_scaled = x_scaler.transform(X_scaled.to(X_train_orig.device))

        mean_rff, std_rff = predict_rff_denormalized(
            rff_model,
            X_scaled.to(device_model),
            y_mean=y_mean,
            y_std=y_std,
            chunk_size=predict_chunk_size,
        )
        mean_rff_np = mean_rff.detach().cpu().numpy().ravel()
        std_rff_np = std_rff.detach().cpu().numpy().ravel()

        X_orig_np = X_orig.detach().cpu().numpy()
        mean_pfn_np, std_pfn_np = predict_tabpfn_on_grid(tabpfn_regressor, X_orig_np)

        with torch.no_grad():
            y_true_np = (
                truth_fn(X_orig.to(device=X_train_orig.device))
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
                .ravel()
            )

        name = feature_names[dim] if dim < len(feature_names) else f"x{dim}"
        x_label = feature_labels[dim] if dim < len(feature_labels) else f"x[{dim}]"

        fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=120)
        ax.plot(x_np, y_true_np, color="#D95F02", lw=2.0, label="True f(x)", zorder=4)

        ax.fill_between(
            x_np,
            mean_rff_np - interval_z * std_rff_np,
            mean_rff_np + interval_z * std_rff_np,
            color="#1B9E77",
            alpha=0.2,
            linewidth=0.0,
            label=f"RFF-GP ±{interval_z}σ",
            zorder=1,
        )
        ax.plot(x_np, mean_rff_np, color="#1B9E77", lw=1.8, label="RFF-GP mean", zorder=3)

        ax.fill_between(
            x_np,
            mean_pfn_np - interval_z * std_pfn_np,
            mean_pfn_np + interval_z * std_pfn_np,
            color="#7570B3",
            alpha=0.18,
            linewidth=0.0,
            label=f"TabPFN ±{interval_z}σ",
            zorder=1,
        )
        ax.plot(x_np, mean_pfn_np, color="#7570B3", lw=1.8, label="TabPFN mean", zorder=2)

        tr_x = X_train_orig[:, dim].detach().cpu().numpy()
        y_min = float(
            np.nanmin(
                np.concatenate(
                    [
                        y_true_np,
                        mean_rff_np - interval_z * std_rff_np,
                        mean_pfn_np - interval_z * std_pfn_np,
                    ]
                )
            )
        )
        y_max = float(
            np.nanmax(
                np.concatenate(
                    [
                        y_true_np,
                        mean_rff_np + interval_z * std_rff_np,
                        mean_pfn_np + interval_z * std_pfn_np,
                    ]
                )
            )
        )
        rug_y = y_min - 0.05 * (y_max - y_min + 1e-9)
        ax.scatter(
            tr_x,
            np.full(tr_x.shape, rug_y),
            s=10,
            c="0.3",
            alpha=0.4,
            marker="|",
            linewidths=0.8,
            label="Train x",
            zorder=0,
        )

        ax.set_xlabel(x_label)
        ax.set_ylabel("y (original scale)")
        ax.set_title(f"{title}\nmarginal slice: {name} (fixed_point={fixed_point})")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.28)
        fig.tight_layout()
        fp = out_dir / f"dim{dim}_{name}.png"
        fig.savefig(fp, bbox_inches="tight")
        plt.close(fig)
        saved.append(fp)

    return saved
