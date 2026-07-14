"""Phase 2: optimization / conditioning diagnostics and W-swap ablation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from common import SAMPLING_MODES, results_dir, save_json, load_json

from gpplus.utils.rff_utils import (
    init_rbf_weights,
    woodbury_factor,
    woodbury_middle_matrix,
    woodbury_jitter_for_dtype,
)
from gpplus.training import evaluate_rff_gp_model
from gpplus.utils import compute_metrics, set_seed
from toa_stgp_checkpoint import load_toa_stgp_checkpoint
from toa_y_transform import inverse_y_single

# load_toa_stgp_checkpoint kept for API compatibility; prefer _load_checkpoint_flexible.


def _lengthscale_stats(model) -> dict:
    ls = model._rff_kernel.lengthscale.detach().float().reshape(-1)
    return {
        "lengthscale_mean": float(ls.mean()),
        "lengthscale_std": float(ls.std(unbiased=False)),
        "lengthscale_min": float(ls.min()),
        "lengthscale_max": float(ls.max()),
        "outputscale": float(model.covar_module.outputscale.detach()),
        "noise": float(model.likelihood.noise.detach().reshape(-1)[0]),
    }


@torch.no_grad()
def _woodbury_spectrum(model) -> dict:
    z = model.train_features()
    noise = model.likelihood.noise.reshape(-1)[0]
    jitter = woodbury_jitter_for_dtype(z.dtype)
    M = woodbury_middle_matrix(noise, z, jitter=jitter)
    evals = torch.linalg.eigvalsh(M.double()).clamp_min(1e-30)
    chol, _ = woodbury_factor(noise, z, jitter=jitter)
    return {
        "m": int(z.shape[-1]),
        "n": int(z.shape[-2]),
        "phi_frob": float(z.norm()),
        "phi_col_norm_mean": float(z.norm(dim=0).mean()),
        "M_eig_min": float(evals.min()),
        "M_eig_max": float(evals.max()),
        "M_cond": float(evals.max() / evals.min()),
        "chol_diag_min": float(chol.diag().abs().min()),
        "noise": float(noise),
    }


def analyze_checkpoint(ckpt_path: str | Path, device: str = "cuda") -> dict:
    bundle = _load_checkpoint_flexible(ckpt_path, device=device)
    model = bundle.model
    return {
        "checkpoint": str(ckpt_path),
        "task_name": bundle.task_name,
        "rff_sampling": bundle.rff_sampling,
        "title": bundle.title,
        "best_train_loss": bundle.best_train_loss,
        "hypers": _lengthscale_stats(model),
        "woodbury": _woodbury_spectrum(model),
        "W_shape": list(model._rff_kernel.randn_weights.shape),
    }


def _load_checkpoint_flexible(ckpt_path: str | Path, device: str = "cuda"):
    """
    Load TOA STGP checkpoints even when current SORF init uses d_pad rows
    while older checkpoints stored truncated (d, D) weights.
    """
    from toa_mtgpr_checkpoint import CHECKPOINT_VERSION, scaler_from_dict
    from toa_stgp_checkpoint import ToaStgpBundle, _dtype_from_str
    from gpplus.models import RFFGPR

    path = Path(ckpt_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported checkpoint version {payload.get('version')!r}")

    dtype = _dtype_from_str(payload["dtype"])
    model_config = dict(payload["model_config"])
    train_x = payload["train_x"].to(dtype=dtype, device=device)
    train_y = payload["train_y"].to(dtype=dtype, device=device)

    model = RFFGPR(train_x, train_y, **model_config)
    # Align randn_weights buffer shape with checkpoint before load_state_dict.
    ckpt_w = payload["state_dict"]["covar_module.base_kernel.randn_weights"]
    base = model._rff_kernel
    if tuple(base.randn_weights.shape) != tuple(ckpt_w.shape):
        base.register_buffer(
            "randn_weights",
            torch.empty(ckpt_w.shape, device=device, dtype=dtype),
        )
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
        logit_cos=bool(payload.get("logit_cos", True)),
        input_column_indices=input_column_indices,
        rff_sampling=str(model_config.get("rff_sampling", "rff")),
    )


def _load_raw_test_xy(bundle) -> tuple[torch.Tensor, torch.Tensor]:
    """Load original-scale test X/Y using checkpoint indices (or rebuild pools)."""
    from experiments_toa.data import load_toa_data

    data_path = bundle.data_path
    if data_path is None:
        data_path = str(Path(__file__).resolve().parents[2] / "toa_data_flattened.npz")

    test_idx = bundle.test_idx.cpu().long() if bundle.test_idx is not None else None
    if test_idx is None or test_idx.numel() == 0:
        # Older checkpoints omitted pool indices; rebuild fixed pools from seed.
        _, _, _, _, x_test, y_test, _, _, test_idx = load_toa_data(
            n_train=bundle.n_train,
            n_test=bundle.n_test,
            n_val=bundle.n_val,
            seed=bundle.seed,
            data_path=data_path,
            train_subset="maximin",
        )
        if bundle.input_column_indices is not None:
            x_test = x_test.index_select(-1, bundle.input_column_indices.cpu().long())
        return x_test, y_test

    data = np.load(data_path)
    X = torch.tensor(data["X"], dtype=torch.float64)
    y = torch.stack(
        [
            torch.tensor(data["y_cos"], dtype=torch.float64),
            torch.tensor(data["y_grain"], dtype=torch.float64),
        ],
        dim=1,
    )
    x_test = X[test_idx]
    y_test = y[test_idx]
    if bundle.input_column_indices is not None:
        x_test = x_test.index_select(-1, bundle.input_column_indices.cpu().long())
    return x_test, y_test


@torch.no_grad()
def _eval_bundle_on_test(bundle, device: str = "cuda") -> dict:
    model = bundle.model
    x_test_raw, y_test = _load_raw_test_xy(bundle)
    dtype = bundle.dtype
    x_test = x_test_raw.to(device="cpu", dtype=dtype)
    if bundle.x_scaler is not None and bundle.standardize_x:
        x_scaled = bundle.x_scaler.transform(x_test)
    else:
        x_scaled = x_test
    x_scaled = x_scaled.to(device=device, dtype=dtype)

    pred_mean, lower, upper, pred_std = evaluate_rff_gp_model(model, x_scaled, chunk_size=512)
    inv = inverse_y_single(
        pred_mean.detach().cpu(),
        pred_std.detach().cpu(),
        lower.detach().cpu(),
        upper.detach().cpu(),
        task_name=bundle.task_name,
        y_scaler=bundle.y_scaler,
        standardize_y=bundle.standardize_y,
        log_grain=bundle.log_grain,
        logit_cos=bundle.logit_cos,
        extended=True,
    )
    task = bundle.task_name
    y_true = y_test[:, 0 if task == "y_cos" else 1].cpu()
    computed = compute_metrics(
        y_true,
        inv.point,
        output_std=inv.std,
        lower_95=inv.lower,
        upper_95=inv.upper,
    )
    return {
        "rmse": float(computed["RMSE"]),
        "rrmse": float(computed["RRMSE"]),
        "r2": float(computed.get("R2", float("nan"))),
        "nlpd": float(computed["NLPD"]) if "NLPD" in computed else None,
        "n_test": int(y_true.numel()),
    }


@torch.no_grad()
def w_swap_ablation(
    sorf_ckpt: str | Path,
    *,
    modes: tuple[str, ...] = SAMPLING_MODES,
    seed: int = 42,
    device: str = "cuda",
) -> dict:
    """Freeze hypers from a SORF checkpoint; only redraw randn_weights under each mode."""
    set_seed(seed)
    bundle = _load_checkpoint_flexible(sorf_ckpt, device=device)
    model = bundle.model
    base = model._rff_kernel
    D = int(base.num_samples)
    # Input dim from training data (not W rows: SORF may use d_pad).
    d_in = int(model.train_inputs[0].shape[-1])
    rows = []

    model.invalidate_feature_cache()
    base_metrics = _eval_bundle_on_test(bundle, device=device)
    rows.append(
        {
            "weights": "checkpoint_original",
            "rff_sampling": bundle.rff_sampling,
            "W_shape": list(base.randn_weights.shape),
            "hypers": _lengthscale_stats(model),
            "woodbury": _woodbury_spectrum(model),
            "test": base_metrics,
        }
    )
    print(
        f"W-swap original ({bundle.rff_sampling}): RRMSE={base_metrics['rrmse']:.6g} "
        f"M_cond={rows[-1]['woodbury']['M_cond']:.3e}"
    )

    for mode in modes:
        set_seed(seed + (abs(hash(mode)) % 10_000))
        w = init_rbf_weights(
            d_in,
            D,
            device=base.randn_weights.device,
            dtype=base.randn_weights.dtype,
            rff_sampling=mode,  # type: ignore[arg-type]
        )
        # Resize buffer if SORF pad width differs from checkpoint W.
        if tuple(base.randn_weights.shape) != tuple(w.shape):
            base.register_buffer("randn_weights", w.clone())
        else:
            base.randn_weights.copy_(w)
        base._feature_cache_version = int(getattr(base, "_feature_cache_version", 0)) + 1
        model.invalidate_feature_cache()
        test_metrics = _eval_bundle_on_test(bundle, device=device)
        info = {
            "weights": f"redraw_{mode}",
            "rff_sampling": mode,
            "W_shape": list(base.randn_weights.shape),
            "hypers": _lengthscale_stats(model),
            "woodbury": _woodbury_spectrum(model),
            "test": test_metrics,
        }
        rows.append(info)
        print(
            f"W-swap {mode}: RRMSE={test_metrics['rrmse']:.6g} "
            f"noise={info['hypers']['noise']:.3g} "
            f"M_cond={info['woodbury']['M_cond']:.3e}"
        )

    return {
        "sorf_checkpoint": str(sorf_ckpt),
        "task_name": bundle.task_name,
        "frozen_hypers": True,
        "rows": rows,
    }


def run_phase2(
    *,
    phase0_summary: Path | None = None,
    device: str = "cuda",
    seed: int = 42,
) -> dict:
    if phase0_summary is None:
        phase0_summary = results_dir() / "phase0_matched_summary.json"
    if not Path(phase0_summary).is_file():
        raise FileNotFoundError(
            f"Need Phase 0 summary at {phase0_summary}. Run phase0_matched_bakeoff.py first."
        )
    summary = load_json(phase0_summary)

    conditioning = []
    sorf_cos_ckpt = None
    for row in summary["rows"]:
        for task, key in (("y_cos", "y_cos_checkpoint_path"), ("y_grain", "y_grain_checkpoint_path")):
            ckpt = row.get(key)
            if not ckpt or not Path(ckpt).is_file():
                print(f"Missing checkpoint for {row['rff_sampling']} {task}: {ckpt}")
                continue
            info = analyze_checkpoint(ckpt, device=device)
            conditioning.append(info)
            print(
                f"{row['rff_sampling']} {task}: noise={info['hypers']['noise']:.3g} "
                f"M_cond={info['woodbury']['M_cond']:.3e} "
                f"loss={info['best_train_loss']:.4g}"
            )
            if row["rff_sampling"] == "sorf" and task == "y_cos":
                sorf_cos_ckpt = ckpt

    if sorf_cos_ckpt is None:
        raise RuntimeError("No SORF y_cos checkpoint found for W-swap ablation.")

    print("\n=== W-swap ablation (frozen SORF hypers, redraw W) ===")
    wswap = w_swap_ablation(sorf_cos_ckpt, seed=seed, device=device)

    sorf_grain = next(
        (r.get("y_grain_checkpoint_path") for r in summary["rows"] if r["rff_sampling"] == "sorf"),
        None,
    )
    wswap_grain = None
    if sorf_grain and Path(sorf_grain).is_file():
        print("\n=== W-swap ablation (grain) ===")
        wswap_grain = w_swap_ablation(sorf_grain, seed=seed, device=device)

    payload = {
        "phase": 2,
        "conditioning": conditioning,
        "w_swap_y_cos": wswap,
        "w_swap_y_grain": wswap_grain,
    }
    path = results_dir() / "phase2_opt_and_wswap.json"
    save_json(path, payload)
    print(f"Saved {path}")
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 2 conditioning + W-swap")
    p.add_argument("--phase0-summary", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run_phase2(
        phase0_summary=Path(args.phase0_summary) if args.phase0_summary else None,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
