"""Phase 1: kernel approximation quality and W statistics (no GP training)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common import (
    SAMPLING_MODES,
    DROP_COLUMNS_DEFAULT,
    exact_rbf_kernel,
    results_dir,
    save_json,
)

from experiments_toa.data import load_toa_data
from toa_mtgpr_base import drop_input_columns
from gpplus.utils import UniformScaler, set_seed
from gpplus.utils.rff_utils import featurize_rbf, init_rbf_weights


def _w_stats(w: torch.Tensor, block_size: int) -> dict:
    """Column-norm / correlation / conditioning stats for W (d, D)."""
    d, D = w.shape
    norms = w.norm(dim=0)
    grams = (w.transpose(0, 1) @ w) / d  # scale-free-ish inner products
    # Off-diagonal absolute correlations within first full block (if any)
    b = min(block_size, D)
    block = w[:, :b]
    g_block = block.transpose(0, 1) @ block
    # Cosine similarities
    nn = block.norm(dim=0).clamp_min(1e-12)
    cos = g_block / (nn.unsqueeze(0) * nn.unsqueeze(1))
    off = cos[~torch.eye(b, dtype=torch.bool, device=cos.device)]
    # Conditioning of W^T W
    wtw = w.transpose(0, 1) @ w
    evals = torch.linalg.eigvalsh(wtw.double()).clamp_min(1e-30)
    cond = float(evals.max() / evals.min())
    return {
        "col_norm_mean": float(norms.mean()),
        "col_norm_std": float(norms.std(unbiased=False)),
        "col_norm_min": float(norms.min()),
        "col_norm_max": float(norms.max()),
        "within_block_abs_cos_mean": float(off.abs().mean()) if off.numel() else None,
        "within_block_abs_cos_max": float(off.abs().max()) if off.numel() else None,
        "cond_WtW": cond,
        "eig_min_WtW": float(evals.min()),
        "eig_max_WtW": float(evals.max()),
        "d": d,
        "D": D,
        "n_blocks": int((D + block_size - 1) // block_size),
    }


@torch.no_grad()
def kernel_quality_for_mode(
    x: torch.Tensor,
    *,
    mode: str,
    D: int,
    raw_lengthscale: torch.Tensor,
    n_pairs: int,
    seed: int,
) -> dict:
    set_seed(seed)
    device, dtype = x.device, x.dtype
    w = init_rbf_weights(
        x.shape[-1],
        D,
        device=device,
        dtype=dtype,
        rff_sampling=mode,  # type: ignore[arg-type]
    )
    z = featurize_rbf(x, w, raw_lengthscale, D)
    # Subsample pairs for kernel comparison
    n = x.shape[0]
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 17)
    idx_a = torch.randint(0, n, (n_pairs,), generator=g)
    idx_b = torch.randint(0, n, (n_pairs,), generator=g)
    xa, xb = x[idx_a], x[idx_b]
    k_exact = exact_rbf_kernel(xa, xb, raw_lengthscale).diag()
    za, zb = z[idx_a], z[idx_b]
    k_hat = (za * zb).sum(-1)
    err = k_hat - k_exact
    return {
        "rff_sampling": mode,
        "num_rff": D,
        "n_pairs": n_pairs,
        "kernel_bias": float(err.mean()),
        "kernel_mse": float((err * err).mean()),
        "kernel_mae": float(err.abs().mean()),
        "kernel_exact_mean": float(k_exact.mean()),
        "kernel_hat_mean": float(k_hat.mean()),
        "feature_frob": float(z.norm()),
        "feature_col_norm_mean": float(z.norm(dim=0).mean()),
        "w_stats": _w_stats(w, block_size=x.shape[-1]),
    }


def run_phase1(
    *,
    n_points: int = 512,
    n_pairs: int = 2000,
    D_list: list[int] | None = None,
    seed: int = 42,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> dict:
    if D_list is None:
        D_list = [64, 128, 270, 540, 800, 1600]

    set_seed(seed)
    x_train, _, _, _, _, _, _, _, _ = load_toa_data(
        n_train=n_points,
        n_test=min(100, n_points),
        n_val=0,
        seed=seed,
        train_subset="maximin",
    )
    x_raw = x_train.to(device=device, dtype=dtype)
    x_raw, _ = drop_input_columns(x_raw, DROP_COLUMNS_DEFAULT)
    scaler = UniformScaler()
    scaler.fit(x_raw)
    x = scaler.transform(x_raw)

    d = x.shape[-1]
    raw_ls = torch.zeros(d, device=device, dtype=dtype)  # scale = 1

    rows = []
    for D in D_list:
        for mode in SAMPLING_MODES:
            row = kernel_quality_for_mode(
                x, mode=mode, D=D, raw_lengthscale=raw_ls, n_pairs=n_pairs, seed=seed
            )
            rows.append(row)
            print(
                f"D={D:4d} {mode:4s}: mse={row['kernel_mse']:.4e} "
                f"bias={row['kernel_bias']:+.4e} "
                f"abs_cos={row['w_stats']['within_block_abs_cos_mean']:.4f} "
                f"cond={row['w_stats']['cond_WtW']:.2e}"
            )

    out = {
        "phase": 1,
        "n_points": n_points,
        "n_pairs": n_pairs,
        "input_dim": d,
        "raw_lengthscale": "zeros (scale=1)",
        "D_list": D_list,
        "seed": seed,
        "rows": rows,
    }
    path = results_dir() / "phase1_kernel_quality.json"
    save_json(path, out)
    print(f"Saved {path}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 1: kernel MSE and W stats")
    p.add_argument("--n-points", type=int, default=512)
    p.add_argument("--n-pairs", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--D-list",
        type=str,
        default="64,128,270,540,800,1600",
        help="Comma-separated frequency counts",
    )
    args = p.parse_args()
    D_list = [int(x) for x in args.D_list.split(",") if x.strip()]
    run_phase1(
        n_points=args.n_points,
        n_pairs=args.n_pairs,
        D_list=D_list,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
