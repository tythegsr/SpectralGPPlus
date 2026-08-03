"""SMAC3 multi-fidelity tuner for S2 SORF algae (val RRMSE).

Searches PAC-Bayes, response-noise prior, lengthscale init, Adam lr, and soft
bound penalty (on/off + log-scale ``lambda`` in ``[1e-2, 1e2]``). When bounds
are on, algae uses floor ``bound_min=0`` (no upper bound). Fidelity =
``num_epochs``. Objective = ``algae_best_val_RRMSE``.

Install::

    pip install "smac>=2.1"

Smoke (1 short trial)::

    python experiments_SORF/smac_s2_sorf_tune.py --n-trials 1 --min-budget 50 --max-budget 50

Full algae tune (default fidelities 500/1000, Hyperband interpolates)::

    python experiments_SORF/smac_s2_sorf_tune.py --n-trials 30 --device cuda

Incumbent is written to ``<save-root>/incumbent.json`` for copy-paste into
``S2_toa_SORF.py`` IDE knobs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

_ROOT = Path(__file__).resolve().parents[1]
_SORF_DIR = Path(__file__).resolve().parent
_RFF_DIR = _ROOT / "experiments_RFF"
_MTGPR_DIR = _ROOT / "experiments_RFFMTGPR"

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments_toa.paths import pin_toa_import_paths

pin_toa_import_paths(_MTGPR_DIR, _RFF_DIR, _SORF_DIR)

from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    EqualsCondition,
    Float,
)
from smac import MultiFidelityFacade, Scenario

from mtgpr_experiment_utils import DEFAULT_ADAM_KWARGS
from toa_s2_sorf_base import run_s2_toa_sorf

TASK = "algae"
FAIL_COST = 1e3

# Match successful LOOK / IDE baselines (not searched).
FIXED = {
    "n_train": 16000,
    "n_test": 5000,
    "num_rff": 1600,
    "num_inits": 1,
    "ard": True,
    "correct_sorf": True,
    "dtype": torch.float64,
    "data_path": "experiments_toa/data 11 QoI/snow_toa_fsnow_only_20262707.nc",
    "input_variable": "toa_radiance",
    "task_band_config": "experiments_toa/configs/s2_task_bands_from_corr_fsnow_only.json",
    "x_transform": "none",
    "log_scale_qoi": None,
    "logit_scale_qoi": None,
    # Bound knobs held fixed; on/off + lambda are searched.
    "bound_penalty_k": 2.0,
    "bound_penalty_alpha": 10.0,
    "bound_penalty_max_points": 4096,
    "bound_penalty_lambda_learnable": False,
}


def build_configspace(seed: int) -> ConfigurationSpace:
    cs = ConfigurationSpace(seed=seed)
    lr = Float("lr", (1e-3, 5e-2), default=0.01, log=True)

    # Lengthscale init: family + family-specific knobs (raw / unconstrained space).
    ls_init_method = Categorical(
        "ls_init_method", ["normal"], default="normal"
    )
    ls_init_mean = Float("ls_init_mean", (-5.0, -1.0), default=-4.0)
    ls_init_std = Float("ls_init_std", (0.5, 2.0), default=1.0)

    response_noise_prior = Categorical(
        "response_noise_prior", [False, True], default=False
    )
    noise_var_fraction = Float(
        "noise_var_fraction", (1e-5, 1e-1), default=0.001, log=True
    )
    noise_prior_log_scale = Float(
        "noise_prior_log_scale", (0.1, 1.5), default=0.5
    )
    pac_bayes = Categorical("pac_bayes", [False, True], default=True)
    pac_bayes_temperature = Float(
        "pac_bayes_temperature", (0.3, 10.0), default=1.0, log=True
    )
    pac_bayes_prior_std = Float(
        "pac_bayes_prior_std", (0.3, 3.0), default=1.0, log=True
    )
    pac_bayes_posterior_std = Float(
        "pac_bayes_posterior_std", (0.05, 0.5), default=0.1, log=True
    )
    # Soft physical floor for algae; lambda searched wide (weak ↔ strong).
    bound_penalty = Categorical("bound_penalty", [False, True], default=False)
    bound_penalty_lambda = Float(
        "bound_penalty_lambda", (1e-4, 1e2), default=1.0, log=True
    )
    cs.add(
        [
            lr,
            ls_init_method,
            ls_init_mean,
            ls_init_std,
            response_noise_prior,
            noise_var_fraction,
            noise_prior_log_scale,
            pac_bayes,
            pac_bayes_temperature,
            pac_bayes_prior_std,
            pac_bayes_posterior_std,
            bound_penalty,
            bound_penalty_lambda,
        ]
    )
    cs.add(
        [
            EqualsCondition(ls_init_mean, ls_init_method, "normal"),
            EqualsCondition(ls_init_std, ls_init_method, "normal"),
            EqualsCondition(noise_var_fraction, response_noise_prior, True),
            EqualsCondition(noise_prior_log_scale, response_noise_prior, True),
            EqualsCondition(pac_bayes_temperature, pac_bayes, True),
            EqualsCondition(pac_bayes_prior_std, pac_bayes, True),
            EqualsCondition(pac_bayes_posterior_std, pac_bayes, True),
            EqualsCondition(bound_penalty_lambda, bound_penalty, True),
        ]
    )
    return cs


def lengthscale_init_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Build ``initializer_parameter_configs['raw_lengthscale']`` from SMAC cfg."""
    return {
        "method": "normal",
        "mean": float(cfg.get("ls_init_mean", -4.0)),
        "std": float(cfg.get("ls_init_std", 1.0)),
    }


def bound_kwargs_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Map SMAC bound flags to ``run_s2_toa_sorf`` bound kwargs."""
    bounds_on = bool(cfg.get("bound_penalty", False))
    if bounds_on:
        bound_min: list[float | None] | None = [0.0]
        bound_max: list[float | None] | None = [None]
        lam = float(cfg.get("bound_penalty_lambda", 1.0))
    else:
        bound_min = None
        bound_max = None
        lam = 0.0
    return {
        "bound_min": bound_min,
        "bound_max": bound_max,
        "bound_penalty_k": float(FIXED["bound_penalty_k"]),
        "bound_penalty_lambda": lam,
        "bound_penalty_alpha": float(FIXED["bound_penalty_alpha"]),
        "bound_penalty_max_points": FIXED["bound_penalty_max_points"],
        "bound_penalty_lambda_learnable": bool(FIXED["bound_penalty_lambda_learnable"]),
    }


def ide_bound_snippet(cfg: dict[str, Any]) -> dict[str, Any]:
    """IDE-facing bound knobs for ``S2_toa_SORF.py``."""
    bk = bound_kwargs_from_config(cfg)
    return {
        "BOUND_MIN": bk["bound_min"],
        "BOUND_MAX": bk["bound_max"],
        "BOUND_PENALTY_K": bk["bound_penalty_k"],
        "BOUND_PENALTY_LAMBDA": bk["bound_penalty_lambda"],
        "BOUND_PENALTY_LAMBDA_LEARNABLE": bk["bound_penalty_lambda_learnable"],
        "BOUND_PENALTY_ALPHA": bk["bound_penalty_alpha"],
        "BOUND_PENALTY_MAX_POINTS": bk["bound_penalty_max_points"],
    }


def _json_safe(value: Any) -> Any:
    """Convert ConfigSpace / NumPy scalars to plain Python JSON types."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    # NumPy scalars (incl. np.str_ / np.bool) before isinstance(str/bool):
    # those types subclass builtins but are not JSON-serializable.
    if type(value).__module__ == "numpy" and hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (ValueError, TypeError):
            pass
    if isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(value.item())
        except (ValueError, TypeError):
            pass
    return str(value)


def config_to_row(config: Configuration) -> dict[str, Any]:
    """Flatten a Configuration to a JSON-serializable dict."""
    return {k: _json_safe(config[k]) for k in config}


def append_trial_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = _json_safe(row)
    fieldnames = list(row.keys())
    write_header = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    jsonl = path.with_suffix(".jsonl")
    with jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def make_target(
    *,
    save_root: Path,
    device: str,
    n_train: int,
    n_test: int,
    num_rff: int,
) -> Any:
    trial_counter = {"i": 0}

    def target(config: Configuration, seed: int = 0, budget: float = 300.0) -> float:
        trial_counter["i"] += 1
        trial_id = trial_counter["i"]
        num_epochs = max(1, int(round(float(budget))))
        cfg = config_to_row(config)
        run_dir = save_root / "runs" / f"trial{trial_id:04d}_ep{num_epochs}"
        run_dir.mkdir(parents=True, exist_ok=True)

        pac_on = bool(cfg.get("pac_bayes", False))
        prior_on = bool(cfg.get("response_noise_prior", False))
        init_pcs = {"raw_lengthscale": lengthscale_init_from_config(cfg)}
        optimizer_kwargs = {**DEFAULT_ADAM_KWARGS, "lr": float(cfg["lr"])}
        bound_kw = bound_kwargs_from_config(cfg)

        print("=" * 72)
        print(f"SMAC trial {trial_id}  budget={num_epochs}  seed={seed}")
        print(f"  config={cfg}")

        t0 = time.time()
        cost = FAIL_COST
        metrics: dict[str, Any] = {}
        err: str | None = None
        try:
            metrics = run_s2_toa_sorf(
                n_train=n_train,
                n_test=n_test,
                num_rff=num_rff,
                num_inits=FIXED["num_inits"],
                num_epochs=num_epochs,
                optimizer_kwargs=optimizer_kwargs,
                seed=int(seed),
                device=device,
                dtype=FIXED["dtype"],
                ard=FIXED["ard"],
                save_path=str(run_dir),
                monitor_validation=True,
                plot_validation=False,
                plot_posterior=False,
                save_checkpoint=False,
                training_verbose=False,
                log_every_n_epochs=max(50, num_epochs // 10),
                parallel_verbose=0,
                response_noise_prior=prior_on,
                noise_var_fraction=float(cfg.get("noise_var_fraction", 0.001)),
                noise_prior_log_scale=float(cfg.get("noise_prior_log_scale", 0.5)),
                initializer_parameter_configs=init_pcs,
                correct_sorf=FIXED["correct_sorf"],
                data_path=FIXED["data_path"],
                input_variable=FIXED["input_variable"],
                task_band_config=FIXED["task_band_config"],
                x_transform=FIXED["x_transform"],
                log_scale_qoi=FIXED["log_scale_qoi"],
                logit_scale_qoi=FIXED["logit_scale_qoi"],
                pac_bayes=pac_on,
                pac_bayes_temperature=float(cfg.get("pac_bayes_temperature", 1.0)),
                pac_bayes_prior_std=float(cfg.get("pac_bayes_prior_std", 1.0)),
                pac_bayes_posterior_std=float(cfg.get("pac_bayes_posterior_std", 0.1)),
                task_names=[TASK],
                **bound_kw,
            )
            key = f"{TASK}_best_val_RRMSE"
            if key not in metrics or metrics[key] is None:
                raise KeyError(f"Missing {key} in metrics (keys={sorted(metrics)[:20]}...)")
            cost = float(metrics[key])
        except Exception as exc:  # noqa: BLE001 — SMAC must keep running
            err = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            print(f"  FAILED trial {trial_id}: {err}  -> cost={FAIL_COST}")

        wall_s = time.time() - t0
        row = {
            "trial_id": trial_id,
            "budget_epochs": num_epochs,
            "seed": int(seed),
            "cost_best_val_RRMSE": cost,
            "wall_time_s": wall_s,
            "error": err,
            "algae_RRMSE": metrics.get("algae_RRMSE"),
            "algae_best_val_NLL": metrics.get("algae_best_val_NLL"),
            "algae_noise": metrics.get("algae_noise"),
            "run_dir": str(run_dir),
            **{f"cfg_{k}": v for k, v in cfg.items()},
        }
        append_trial_row(save_root / "trials.csv", row)
        print(
            f"  done cost={cost:.6f}  test_RRMSE={row.get('algae_RRMSE')}  "
            f"wall={wall_s:.1f}s"
        )
        return cost

    return target


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-trials", type=int, default=60)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--save-root",
        type=str,
        default="experiments_SORF/results/smac_s2_sorf_algae_bounds",
    )
    p.add_argument("--n-train", type=int, default=FIXED["n_train"])
    p.add_argument("--n-test", type=int, default=FIXED["n_test"])
    p.add_argument("--num-rff", type=int, default=FIXED["num_rff"])
    p.add_argument(
        "--min-budget",
        type=int,
        default=300,
        help="Min fidelity (num_epochs). Use 50 for smoke.",
    )
    p.add_argument(
        "--max-budget",
        type=int,
        default=500,
        help="Max fidelity (num_epochs). Use 50 for smoke.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.min_budget < 1 or args.max_budget < args.min_budget:
        raise SystemExit(
            f"Need 1 <= min_budget <= max_budget, got "
            f"{args.min_budget}, {args.max_budget}"
        )

    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)
    cs = build_configspace(args.seed)
    target = make_target(
        save_root=save_root,
        device=args.device,
        n_train=args.n_train,
        n_test=args.n_test,
        num_rff=args.num_rff,
    )

    scenario = Scenario(
        cs,
        n_trials=args.n_trials,
        seed=args.seed,
        deterministic=True,
        min_budget=float(args.min_budget),
        max_budget=float(args.max_budget),
        output_directory=str(save_root / "smac_output"),
    )
    print(
        f"SMAC MultiFidelityFacade  n_trials={args.n_trials}  "
        f"budget=[{args.min_budget},{args.max_budget}]  "
        f"device={args.device}  save_root={save_root}"
    )
    smac = MultiFidelityFacade(scenario, target)
    incumbent = smac.optimize()
    # Prefer runhistory cost (already evaluated); avoid an extra max-budget re-train.
    try:
        cost = float(smac.runhistory.get_cost(incumbent))
    except Exception:  # noqa: BLE001
        cost = float("nan")
        print("Warning: could not read incumbent cost from runhistory.")

    inc_dict = config_to_row(incumbent)
    incumbent_payload = {
        "incumbent": inc_dict,
        "incumbent_cost_best_val_RRMSE": cost,
        "task": TASK,
        "n_trials": args.n_trials,
        "min_budget": args.min_budget,
        "max_budget": args.max_budget,
        "seed": args.seed,
        "save_root": str(save_root),
        "ide_snippet": {
            "LR": inc_dict.get("lr"),
            "INITIALIZER_PARAMETER_CONFIGS": {
                "raw_lengthscale": lengthscale_init_from_config(inc_dict),
            },
            "RESPONSE_NOISE_PRIOR": inc_dict.get("response_noise_prior"),
            "NOISE_VAR_FRACTION": inc_dict.get("noise_var_fraction", 0.001),
            "NOISE_PRIOR_LOG_SCALE": inc_dict.get("noise_prior_log_scale", 0.5),
            "PAC_BAYES": inc_dict.get("pac_bayes"),
            "PAC_BAYES_TEMPERATURE": inc_dict.get("pac_bayes_temperature", 1.0),
            "PAC_BAYES_PRIOR_STD": inc_dict.get("pac_bayes_prior_std", 1.0),
            "PAC_BAYES_POSTERIOR_STD": inc_dict.get("pac_bayes_posterior_std", 0.1),
            **ide_bound_snippet(inc_dict),
        },
    }
    (save_root / "incumbent.json").write_text(
        json.dumps(_json_safe(incumbent_payload), indent=2), encoding="utf-8"
    )
    summary = {
        "n_trials": args.n_trials,
        "min_budget": args.min_budget,
        "max_budget": args.max_budget,
        "seed": args.seed,
        "device": args.device,
        "save_root": str(save_root),
        "incumbent_cost": cost,
        "incumbent": inc_dict,
    }
    (save_root / "smac_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2), encoding="utf-8"
    )

    print("=" * 72)
    print("Incumbent config:")
    print(json.dumps(_json_safe(inc_dict), indent=2))
    print(f"Incumbent cost (algae_best_val_RRMSE): {cost:.6f}")
    print(f"Wrote {save_root / 'incumbent.json'}")
    print(f"IDE snippet keys under incumbent.json -> ide_snippet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
