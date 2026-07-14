"""Phase 3: D-sweep (kernel + short TOA) and low-d Ackley matched comparison."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from common import DROP_COLUMNS_DEFAULT, SAMPLING_MODES, results_dir, save_json
from phase1_kernel_quality import run_phase1

from gpplus.models import RFFGPR
from gpplus.training import (
    GPTrainer,
    RFFParameterInitializer,
    RFFWoodburyMarginalLogLikelihood,
    evaluate_rff_gp_model,
)
from gpplus.training.optimizers import LBFGSScipy
from gpplus.utils import StandardScaler, UniformScaler, compute_metrics, set_seed
from toa_stgp_base import run_toa_stgp

import importlib.util
from pathlib import Path as _Path

_rff_load = _Path(__file__).resolve().parents[2] / "experiments_RFF" / "load_experimental_data.py"
_spec = importlib.util.spec_from_file_location("rff_load_experimental_data", _rff_load)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
generate_ackley_data = _mod.generate_ackley_data

_rff_utils = _Path(__file__).resolve().parents[2] / "experiments_RFF" / "rff_experiment_utils.py"
_spec2 = importlib.util.spec_from_file_location("rff_experiment_utils_diag", _rff_utils)
_mod2 = importlib.util.module_from_spec(_spec2)
assert _spec2.loader is not None
_spec2.loader.exec_module(_mod2)
DEFAULT_LBFGS_KWARGS = _mod2.DEFAULT_LBFGS_KWARGS
unpack_train_val_test = _mod2.unpack_train_val_test


def run_ackley_matched(
    *,
    dimensions: int = 10,
    train_size: int = 40,
    num_rff: int = 128,
    num_test: int = 2000,
    seed: int = 42,
    num_inits: int = 4,
    device: str = "cuda",
    dtype: torch.dtype = torch.float64,
) -> dict:
    """Matched Ackley bake-off: only rff_sampling differs."""
    n_train = train_size * dimensions
    data = generate_ackley_data(
        n_train=n_train,
        n_test=num_test,
        n_val=0,
        dimensions=dimensions,
        x_bounds=[-5.0, 10.0],
        train_noise=0.0,
        test_noise=0.0,
        seed=seed,
    )
    x_train, y_train, _, _, x_test, y_test = unpack_train_val_test(data)
    x_train = x_train.to(dtype=dtype)
    y_train = y_train.to(dtype=dtype).reshape(-1)
    x_test = x_test.to(dtype=dtype)
    y_test = y_test.to(dtype=dtype).reshape(-1)

    x_scaler = UniformScaler(scale_to_neg_one=True)
    x_scaler.fit(x_train)
    x_train_s = x_scaler.transform(x_train)
    x_test_s = x_scaler.transform(x_test)
    y_scaler = StandardScaler()
    y_scaler.fit(y_train.unsqueeze(-1))
    y_train_s = y_scaler.transform(y_train.unsqueeze(-1)).squeeze(-1)
    y_mean = y_scaler.mean.reshape(-1).cpu()
    y_std = y_scaler.std.reshape(-1).cpu()

    rows = []
    for mode in SAMPLING_MODES:
        set_seed(seed)
        model = RFFGPR(
            x_train_s,
            y_train_s,
            num_rff=num_rff,
            ard=True,
            rff_sampling=mode,  # type: ignore[arg-type]
        )
        trainer = GPTrainer(
            model,
            mll_class=RFFWoodburyMarginalLogLikelihood,
            num_epochs=1,
            num_inits=num_inits,
            seed=seed,
            device=device,
            dtype=dtype,
            optimizer_class=LBFGSScipy,
            optimizer_kwargs=dict(DEFAULT_LBFGS_KWARGS),
            initializer_class=RFFParameterInitializer,
            n_jobs=1,
            inner_max_num_threads=1,
            cholesky_jitter=1e-6,
        )
        t0 = time.time()
        runs = trainer.train()
        train_time = time.time() - t0
        successful = [
            r for r in runs if r.get("loss") is not None and r.get("state_dict") is not None
        ]
        if not successful:
            raise RuntimeError(f"Ackley training failed for {mode}")
        best = min(successful, key=lambda r: r["loss"])
        model.load_state_dict(best["state_dict"])
        model.eval()
        model.invalidate_feature_cache()
        pred_mean, _, _, pred_std = evaluate_rff_gp_model(
            model, x_test_s.to(device=device), chunk_size=512
        )
        pred_mean = pred_mean.detach().cpu() * y_std + y_mean
        pred_std = pred_std.detach().cpu() * y_std
        computed = compute_metrics(y_test.cpu(), pred_mean, output_std=pred_std)
        noise = float(model.likelihood.noise.detach().cpu().reshape(-1)[0])
        row = {
            "rff_sampling": mode,
            "dimensions": dimensions,
            "n_train": int(x_train.shape[0]),
            "num_rff": num_rff,
            "RRMSE": float(computed["RRMSE"]),
            "RMSE": float(computed["RMSE"]),
            "R2": float(computed.get("R2", float("nan"))),
            "NLPD": float(computed["NLPD"]) if "NLPD" in computed else None,
            "best_train_loss": float(best["loss"]),
            "noise": noise,
            "Training_Time": train_time,
        }
        rows.append(row)
        print(
            f"Ackley d={dimensions} {mode}: RRMSE={row['RRMSE']:.4g} "
            f"RMSE={row['RMSE']:.4g} loss={row['best_train_loss']:.4g}"
        )
    return {
        "dimensions": dimensions,
        "train_size_per_dim": train_size,
        "num_rff": num_rff,
        "num_test": num_test,
        "seed": seed,
        "num_inits": num_inits,
        "rows": rows,
    }


def run_toa_d_sweep(
    *,
    n_train: int = 2000,
    n_test: int = 2000,
    D_list: list[int] | None = None,
    seed: int = 42,
    num_epochs: int = 80,
    device: str = "cuda",
) -> dict:
    if D_list is None:
        D_list = [200, 400, 800]
    out_root = results_dir() / "phase3_toa_d_sweep"
    rows = []
    for D in D_list:
        for mode in SAMPLING_MODES:
            save_path = str(out_root / f"D{D}_{mode}")
            print(f"\n=== TOA D-sweep D={D} mode={mode} ===")
            metrics = run_toa_stgp(
                n_train=n_train,
                n_test=n_test,
                num_rff=D,
                rff_sampling=mode,  # type: ignore[arg-type]
                seed=seed,
                num_inits=1,
                num_epochs=num_epochs,
                device=device,
                dtype=torch.float32,
                save_path=save_path,
                train_subset="maximin",
                logit_cos=False,
                log_grain=True,
                drop_columns=DROP_COLUMNS_DEFAULT,
                optimizer_kwargs={
                    "lr": 1.0,
                    "betas": (0.9, 0.999),
                    "eps": 1e-08,
                    "weight_decay": 0.0,
                    "amsgrad": False,
                },
                plot_posterior=False,
                plot_validation=False,
                save_checkpoint=False,
                posterior_n_examples=0,
                monitor_validation=False,
            )
            rows.append(
                {
                    "num_rff": D,
                    "rff_sampling": mode,
                    "y_cos_RRMSE": metrics.get("y_cos_RRMSE"),
                    "y_grain_RRMSE": metrics.get("y_grain_RRMSE"),
                    "y_cos_RMSE": metrics.get("y_cos_RMSE"),
                    "y_grain_RMSE": metrics.get("y_grain_RMSE"),
                    "y_cos_noise": metrics.get("y_cos_noise"),
                    "y_grain_noise": metrics.get("y_grain_noise"),
                    "y_cos_best_train_loss": metrics.get("y_cos_best_train_loss"),
                    "y_grain_best_train_loss": metrics.get("y_grain_best_train_loss"),
                }
            )
    return {"n_train": n_train, "n_test": n_test, "D_list": D_list, "rows": rows}


def run_phase3(
    *,
    device: str = "cuda",
    skip_toa_sweep: bool = False,
    skip_ackley: bool = False,
) -> dict:
    print("=== Phase 3a: kernel D-sweep ===")
    kernel = run_phase1(
        n_points=512,
        n_pairs=2000,
        D_list=[64, 128, 270, 540, 800, 1600],
        seed=42,
        device="cpu",
        dtype=torch.float64,
    )
    kernel_path = results_dir() / "phase3_kernel_d_sweep.json"
    save_json(
        kernel_path,
        {"phase": "3a", **{k: kernel[k] for k in kernel if k != "phase"}},
    )

    ackley = None
    if not skip_ackley:
        print("\n=== Phase 3b: low-d Ackley matched ===")
        ackley_10 = run_ackley_matched(device=device)
        ackley_40 = run_ackley_matched(
            dimensions=40,
            train_size=20,
            num_rff=200,
            num_test=2000,
            num_inits=2,
            device=device,
        )
        ackley = {"ackley_10d": ackley_10, "ackley_40d": ackley_40}

    toa_sweep = None
    if not skip_toa_sweep:
        print("\n=== Phase 3c: TOA D-sweep (short train) ===")
        toa_sweep = run_toa_d_sweep(device=device, num_epochs=80)

    payload = {
        "phase": 3,
        "kernel_d_sweep_path": str(kernel_path),
        "ackley": ackley,
        "toa_d_sweep": toa_sweep,
    }
    path = results_dir() / "phase3_sweeps.json"
    save_json(path, payload)
    print(f"Saved {path}")
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 3 sweeps")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--skip-toa-sweep", action="store_true")
    p.add_argument("--skip-ackley", action="store_true")
    args = p.parse_args()
    run_phase3(
        device=args.device,
        skip_toa_sweep=args.skip_toa_sweep,
        skip_ackley=args.skip_ackley,
    )


if __name__ == "__main__":
    main()
