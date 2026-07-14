"""
Offline diagnostics for RFF vs ORF vs SORF on Ackley-scale inputs.

1) Kernel MSE / bias vs exact RBF (Yu-aligned)
2) Feature geometry (norms, orthogonality, cond(Phi^T Phi))
3) W-swap: train on one sampling mode, freeze hypers, redraw W from another
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
_RFF = _ROOT / "experiments_RFF"
for p in (_ROOT, _RFF):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gpplus.utils.rff_utils import featurize_rbf, init_rbf_weights
from gpplus.utils import set_seed, UniformScaler, StandardScaler, compute_metrics
from load_experimental_data import generate_ackley_data
from gpplus.models import RFFGPR
from gpplus.training import (
    GPTrainer,
    RFFParameterInitializer,
    RFFWoodburyMarginalLogLikelihood,
    evaluate_rff_gp_model,
)
from rff_experiment_utils import DEFAULT_ADAM_KWARGS, compute_n_val, unpack_train_val_test


MODES = ("rff", "orf", "sorf")


def _rff_base_kernel(model: RFFGPR):
    cov = model.covar_module
    return cov.base_kernel if hasattr(cov, "base_kernel") else cov


def _exact_rbf(x: torch.Tensor, y: torch.Tensor, lengthscale: torch.Tensor) -> torch.Tensor:
    """SE kernel matching featurize_rbf: scale coords by 10^(ls/2)."""
    scale = torch.pow(10.0, lengthscale / 2.0)
    xs = x * scale
    ys = y * scale
    diff = xs.unsqueeze(1) - ys.unsqueeze(0)
    return torch.exp(-0.5 * (diff * diff).sum(dim=-1))


def _approx_kernel(z: torch.Tensor) -> torch.Tensor:
    return z @ z.transpose(-1, -2)


def kernel_mse_study(
    x: torch.Tensor,
    *,
    D_list: list[int],
    lengthscale: torch.Tensor,
    n_pairs: int,
    seeds: list[int],
) -> list[dict]:
    n = x.shape[0]
    rows = []
    for D in D_list:
        for mode in MODES:
            mses, biases = [], []
            for seed in seeds:
                set_seed(seed)
                W = init_rbf_weights(
                    x.shape[1], D, device=x.device, dtype=x.dtype, rff_sampling=mode
                )
                z = featurize_rbf(x, W, lengthscale, num_samples=D)
                K_hat = _approx_kernel(z)
                K = _exact_rbf(x, x, lengthscale)
                g = torch.Generator(device="cpu")
                g.manual_seed(seed)
                i = torch.randint(0, n, (n_pairs,), generator=g)
                j = torch.randint(0, n, (n_pairs,), generator=g)
                err = K_hat[i, j] - K[i, j]
                mses.append(float((err * err).mean().item()))
                biases.append(float(err.mean().item()))
            rows.append(
                {
                    "mode": mode,
                    "D": D,
                    "mse_mean": float(np.mean(mses)),
                    "mse_std": float(np.std(mses)),
                    "bias_mean": float(np.mean(biases)),
                    "bias_std": float(np.std(biases)),
                }
            )
    return rows


def feature_geometry(
    x: torch.Tensor,
    *,
    D: int,
    lengthscale: torch.Tensor,
    seed: int,
) -> list[dict]:
    rows = []
    d = x.shape[1]
    for mode in MODES:
        set_seed(seed)
        W = init_rbf_weights(d, D, device=x.device, dtype=x.dtype, rff_sampling=mode)
        z = featurize_rbf(x, W, lengthscale, num_samples=D)
        w_norms = W.norm(dim=0)
        b = min(d, D)
        Wb = W[:, :b]
        Wb_n = Wb / Wb.norm(dim=0, keepdim=True).clamp_min(1e-12)
        gram = (Wb_n.T @ Wb_n).abs()
        off = gram[~torch.eye(b, dtype=torch.bool, device=gram.device)]
        phi_t_phi = z.T @ z
        evals = torch.linalg.eigvalsh(phi_t_phi).clamp_min(1e-18)
        rows.append(
            {
                "mode": mode,
                "D": D,
                "w_norm_mean": float(w_norms.mean().item()),
                "w_norm_std": float(w_norms.std().item()),
                "w_norm_chi_d_target": math.sqrt(d),
                "block_abs_cos_mean": float(off.mean().item()) if off.numel() else 0.0,
                "block_abs_cos_max": float(off.max().item()) if off.numel() else 0.0,
                "cond_phiTphi": float((evals.max() / evals.min()).item()),
                "eff_rank_phiTphi": float((evals.sum() / evals.max()).item()),
                "phi_fro": float(z.norm().item()),
            }
        )
    return rows


def _train_rffgp(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    mode: str,
    D: int,
    num_inits: int,
    seed: int,
    device: str,
    dtype: torch.dtype,
    num_epochs: int = 200,
    lr: float = 1.0,
) -> RFFGPR:
    model = RFFGPR(x_train, y_train, num_rff=D, ard=True, rff_sampling=mode)
    trainer = GPTrainer(
        model,
        mll_class=RFFWoodburyMarginalLogLikelihood,
        num_epochs=num_epochs,
        num_inits=num_inits,
        seed=seed,
        device=device,
        dtype=dtype,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs={**DEFAULT_ADAM_KWARGS, "lr": lr},
        initializer_class=RFFParameterInitializer,
        n_jobs=1,
        inner_max_num_threads=1,
        cholesky_jitter=1e-6,
    )
    runs = trainer.train()
    successful = [r for r in runs if r.get("loss") is not None and r.get("state_dict") is not None]
    if not successful:
        raise RuntimeError(f"All training runs failed for mode={mode}")
    best = min(successful, key=lambda r: r["loss"])
    model.load_state_dict(best["state_dict"])
    return model


def w_swap_study(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    D: int,
    num_inits: int,
    device: str,
    dtype: torch.dtype,
    seed: int,
    num_epochs: int = 200,
    lr: float = 1.0,
) -> list[dict]:
    rows = []
    for train_mode in MODES:
        set_seed(seed)
        model = _train_rffgp(
            x_train,
            y_train,
            mode=train_mode,
            D=D,
            num_inits=num_inits,
            seed=seed,
            device=device,
            dtype=dtype,
            num_epochs=num_epochs,
            lr=lr,
        )
        base = _rff_base_kernel(model)
        raw_ls = base.raw_lengthscale.detach().clone()
        noise = model.likelihood.noise.detach().clone()
        raw_os = None
        if hasattr(model.covar_module, "raw_outputscale"):
            raw_os = model.covar_module.raw_outputscale.detach().clone()

        for eval_mode in MODES:
            set_seed(seed + 91)
            m2 = RFFGPR(
                x_train, y_train, num_rff=D, ard=True, rff_sampling=eval_mode
            ).to(device=device, dtype=dtype)
            base2 = _rff_base_kernel(m2)
            with torch.no_grad():
                base2.raw_lengthscale.copy_(raw_ls)
                m2.likelihood.noise = noise
                if raw_os is not None and hasattr(m2.covar_module, "raw_outputscale"):
                    m2.covar_module.raw_outputscale.copy_(raw_os)
                base2.resample_weights(spectral=False, rff_sampling=eval_mode)
            if hasattr(m2, "invalidate_feature_cache"):
                m2.invalidate_feature_cache()

            with torch.no_grad():
                pred_mean, _, _, _ = evaluate_rff_gp_model(m2, x_test, chunk_size=512)
            y_np = y_test.detach().cpu().numpy().reshape(-1)
            mean_np = pred_mean.detach().cpu().numpy().reshape(-1)
            metrics = compute_metrics(y_np, mean_np)
            rows.append(
                {
                    "train_mode": train_mode,
                    "eval_W_mode": eval_mode,
                    "same_mode": train_mode == eval_mode,
                    "RRMSE": float(metrics["RRMSE"]),
                    "RMSE": float(metrics["RMSE"]),
                }
            )
    return rows


def _prepare_ackley(
    dimensions: int,
    train_size: int,
    noise: float,
    seed: int,
    device: str,
    dtype: torch.dtype,
):
    set_seed(seed)
    n_train = train_size * dimensions
    n_val = compute_n_val(n_train, 0.2)
    data = generate_ackley_data(
        n_train=n_train,
        n_test=2000,
        n_val=n_val,
        dimensions=dimensions,
        x_bounds=[-5.0, 10.0],
        train_noise=noise,
        test_noise=noise,
        noise_type="gaussian",
        seed=seed,
    )
    x_train, y_train, _x_val, _y_val, x_test, y_test = unpack_train_val_test(data)
    x_scaler = UniformScaler(feature_range=(-1.0, 1.0), scale_to_neg_one=True)
    x_scaler.fit(x_train)
    x_train = x_scaler.transform(x_train)
    x_test = x_scaler.transform(x_test)
    y_scaler = StandardScaler()
    y_scaler.fit(y_train.unsqueeze(-1))
    y_train = y_scaler.transform(y_train.unsqueeze(-1)).squeeze(-1)
    y_test = y_scaler.transform(y_test.unsqueeze(-1)).squeeze(-1)
    return (
        x_train.to(device=device, dtype=dtype),
        y_train.to(device=device, dtype=dtype).reshape(-1),
        x_test.to(device=device, dtype=dtype),
        y_test.to(device=device, dtype=dtype).reshape(-1),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="RFF/ORF/SORF offline diagnostics")
    parser.add_argument("--dimensions", type=int, default=10)
    parser.add_argument("--train-size", type=int, default=40)
    parser.add_argument("--noise", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32", choices=("float32", "float64"))
    parser.add_argument("--D-list", type=str, default="50,100,200,400")
    parser.add_argument("--geom-D", type=int, default=200)
    parser.add_argument("--swap-D", type=int, default=200)
    parser.add_argument("--swap-inits", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=200, help="Adam epochs for W-swap training")
    parser.add_argument("--lr", type=float, default=1.0, help="Adam learning rate for W-swap")
    parser.add_argument("--n-pairs", type=int, default=2000)
    parser.add_argument("--mse-seeds", type=str, default="0,1,2,3,4")
    parser.add_argument(
        "--save-path",
        type=str,
        default="experiments_RFF/results/ackley_compare_adam_gpu/diagnostics.json",
    )
    parser.add_argument("--skip-swap", action="store_true")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable. Use gpplus_tydev / gpplus2026 (CUDA PyTorch)."
        )

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    D_list = [int(x) for x in args.D_list.split(",") if x.strip()]
    mse_seeds = [int(x) for x in args.mse_seeds.split(",") if x.strip()]

    x_train, y_train, x_test, y_test = _prepare_ackley(
        args.dimensions,
        args.train_size,
        args.noise,
        args.seed,
        args.device,
        dtype,
    )
    lengthscale = torch.zeros(args.dimensions, dtype=dtype, device=args.device)

    print("=== Kernel MSE study ===")
    mse_rows = kernel_mse_study(
        x_train,
        D_list=D_list,
        lengthscale=lengthscale,
        n_pairs=args.n_pairs,
        seeds=mse_seeds,
    )
    for r in mse_rows:
        print(
            f"  {r['mode']:<4} D={r['D']:>4}  MSE={r['mse_mean']:.6e}±{r['mse_std']:.2e}  "
            f"bias={r['bias_mean']:.4e}"
        )

    print("=== Feature geometry ===")
    geom_rows = feature_geometry(
        x_train, D=args.geom_D, lengthscale=lengthscale, seed=args.seed
    )
    for r in geom_rows:
        print(
            f"  {r['mode']:<4} w_norm={r['w_norm_mean']:.3f}±{r['w_norm_std']:.3f} "
            f"(chi~{r['w_norm_chi_d_target']:.3f})  "
            f"block|cos|={r['block_abs_cos_mean']:.4f}  "
            f"cond={r['cond_phiTphi']:.3e}  eff_rank={r['eff_rank_phiTphi']:.2f}"
        )

    swap_rows: list[dict] = []
    if not args.skip_swap:
        print(f"=== W-swap study (Adam ep={args.num_epochs}, lr={args.lr}, {args.device}) ===")
        swap_rows = w_swap_study(
            x_train,
            y_train,
            x_test,
            y_test,
            D=args.swap_D,
            num_inits=args.swap_inits,
            device=args.device,
            dtype=dtype,
            seed=args.seed,
            num_epochs=args.num_epochs,
            lr=args.lr,
        )
        for r in swap_rows:
            print(
                f"  train={r['train_mode']:<4} eval_W={r['eval_W_mode']:<4}  "
                f"RRMSE={r['RRMSE']:.6f}"
            )

    out = {
        "config": {
            "dimensions": args.dimensions,
            "train_size": args.train_size,
            "noise": args.noise,
            "seed": args.seed,
            "D_list": D_list,
            "geom_D": args.geom_D,
            "swap_D": args.swap_D,
        },
        "kernel_mse": mse_rows,
        "feature_geometry": geom_rows,
        "w_swap": swap_rows,
    }
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {save_path}")


if __name__ == "__main__":
    main()
