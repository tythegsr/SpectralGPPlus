"""Phase 0: matched TOA bake-off for rff / orf / sorf."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common import SAMPLING_MODES, DROP_COLUMNS_DEFAULT, results_dir, save_json

from mtgpr_experiment_utils import DEFAULT_ADAM_KWARGS
from toa_stgp_base import run_toa_stgp


def run_phase0(
    *,
    n_train: int = 2000,
    n_test: int = 5000,
    num_rff: int = 800,
    seed: int = 42,
    num_inits: int = 1,
    num_epochs: int = 200,
    lr: float = 1.0,
    device: str = "cuda",
    dtype: str = "float32",
    train_subset: str = "maximin",
    logit_cos: bool = False,
    log_grain: bool = True,
    modes: tuple[str, ...] = SAMPLING_MODES,
) -> dict:
    out_root = results_dir() / "phase0_matched"
    out_root.mkdir(parents=True, exist_ok=True)
    torch_dtype = torch.float32 if dtype == "float32" else torch.float64

    summary_rows = []
    for mode in modes:
        save_path = str(out_root / mode)
        print("\n" + "#" * 72)
        print(f"Phase 0 matched bake-off: {mode}")
        print("#" * 72)
        metrics = run_toa_stgp(
            n_train=n_train,
            n_test=n_test,
            num_rff=num_rff,
            rff_sampling=mode,  # type: ignore[arg-type]
            seed=seed,
            num_inits=num_inits,
            num_epochs=num_epochs,
            device=device,
            dtype=torch_dtype,
            save_path=save_path,
            train_subset=train_subset,
            logit_cos=logit_cos,
            log_grain=log_grain,
            drop_columns=DROP_COLUMNS_DEFAULT,
            optimizer_kwargs={**dict(DEFAULT_ADAM_KWARGS), "lr": lr},
            plot_posterior=False,
            plot_validation=True,
            save_checkpoint=True,
            posterior_n_examples=0,
        )
        row = {
            "rff_sampling": mode,
            "n_train": n_train,
            "n_test": n_test,
            "num_rff": num_rff,
            "seed": seed,
            "num_inits": num_inits,
            "num_epochs": num_epochs,
            "train_subset": train_subset,
            "logit_cos": logit_cos,
            "log_grain": log_grain,
            "y_cos_RRMSE": metrics.get("y_cos_RRMSE"),
            "y_cos_RMSE": metrics.get("y_cos_RMSE"),
            "y_cos_R2": metrics.get("y_cos_R2"),
            "y_cos_NLPD": metrics.get("y_cos_NLPD"),
            "y_cos_noise": metrics.get("y_cos_noise"),
            "y_cos_best_train_loss": metrics.get("y_cos_best_train_loss"),
            "y_grain_RRMSE": metrics.get("y_grain_RRMSE"),
            "y_grain_RMSE": metrics.get("y_grain_RMSE"),
            "y_grain_R2": metrics.get("y_grain_R2"),
            "y_grain_NLPD": metrics.get("y_grain_NLPD"),
            "y_grain_noise": metrics.get("y_grain_noise"),
            "y_grain_best_train_loss": metrics.get("y_grain_best_train_loss"),
            "Training_Time": metrics.get("Training_Time"),
            "save_path": save_path,
            "y_cos_checkpoint_path": metrics.get("y_cos_checkpoint_path"),
            "y_grain_checkpoint_path": metrics.get("y_grain_checkpoint_path"),
        }
        summary_rows.append(row)
        print(
            f"[{mode}] cos RRMSE={row['y_cos_RRMSE']:.6g} noise={row['y_cos_noise']:.3g} "
            f"| grain RRMSE={row['y_grain_RRMSE']:.6g} noise={row['y_grain_noise']:.3g}"
        )

    payload = {
        "phase": 0,
        "protocol": {
            "n_train": n_train,
            "n_test": n_test,
            "num_rff": num_rff,
            "seed": seed,
            "num_inits": num_inits,
            "num_epochs": num_epochs,
            "lr": lr,
            "train_subset": train_subset,
            "logit_cos": logit_cos,
            "log_grain": log_grain,
            "device": device,
            "dtype": dtype,
        },
        "rows": summary_rows,
    }
    path = results_dir() / "phase0_matched_summary.json"
    save_json(path, payload)
    print(f"\nSaved matched summary to {path}")
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 0 matched TOA bake-off")
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--n-test", type=int, default=5000)
    p.add_argument("--num-rff", type=int, default=800)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-inits", type=int, default=1)
    p.add_argument("--num-epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="float32", choices=("float32", "float64"))
    p.add_argument("--train-subset", type=str, default="maximin", choices=("random", "maximin"))
    p.add_argument("--logit-cos", action="store_true", default=False)
    p.add_argument("--no-logit-cos", action="store_false", dest="logit_cos")
    p.add_argument("--modes", type=str, default="rff,orf,sorf")
    args = p.parse_args()
    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    run_phase0(
        n_train=args.n_train,
        n_test=args.n_test,
        num_rff=args.num_rff,
        seed=args.seed,
        num_inits=args.num_inits,
        num_epochs=args.num_epochs,
        lr=args.lr,
        device=args.device,
        dtype=args.dtype,
        train_subset=args.train_subset,
        logit_cos=args.logit_cos,
        modes=modes,
    )


if __name__ == "__main__":
    main()
