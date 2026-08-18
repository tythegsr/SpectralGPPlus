"""SMAC3 multi-fidelity tuner for S2 SVGP algae vs Aug12 SORF+NIGP.

Jointly searches SVGP+PCA and SVGP+NIGP configs (no RFF/ORF/SORF). Fidelity =
``num_epochs``. Objective = ``algae_best_val_RRMSE``. After optimization the
incumbent is retrained at max budget and compared to the Aug12 algae test RRMSE.

Install::

    pip install "smac>=2.1"

Smoke (1 short trial)::

    python experiments_SVGP/smac_s2_svgp_tune.py --n-trials 1 --min-budget 50 --max-budget 50

Full algae tune::

    python experiments_SVGP/smac_s2_svgp_tune.py --n-trials 40 --device cuda

Incumbent is written to ``<save-root>/incumbent.json`` for copy-paste into
``experiments_SVGP/S2_toa_SVGP.py`` IDE knobs. Aug12 gap lands in
``aug12_comparison.json``.
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
_SVGP_DIR = Path(__file__).resolve().parent
_GP_DIR = _ROOT / "experiments_GP"
_RFF_DIR = _ROOT / "experiments_RFF"
_MTGPR_DIR = _ROOT / "experiments_RFFMTGPR"

for _p in (_ROOT, _GP_DIR, _RFF_DIR, _MTGPR_DIR, _SVGP_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments_toa.paths import pin_toa_import_paths

pin_toa_import_paths(_MTGPR_DIR, _RFF_DIR, _GP_DIR)

from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    EqualsCondition,
    Float,
)
from smac import MultiFidelityFacade, Scenario

from gp_experiment_utils import DEFAULT_ADAM_KWARGS
from toa_s2_gp_base import run_s2_toa_gp

TASK = "algae"
FAIL_COST = 1e3

AUG12_METRICS = (
    _ROOT
    / "experiments_SORF/results/Aug12"
    / "s2_toa_sorf_1inits_numrff800_lr0.1_taskbandconfig_rbf_nigp"
    "_freezeepochnigp100_sloperefreshes20_dtypefloat64"
    / "gp_S2_TOA_nTrain100000_nTest10000_sorfD800.json"
)

# Aug12-aligned fixed knobs (not searched).
FIXED = {
    "n_train": 100_000,
    "n_test": 10_000,
    "num_inits": 1,
    "ard": True,
    "dtype": torch.float32,
    "data_path": "experiments_toa/data 11 QoI/snow_toa_fsnow_70to100_20261208.nc",
    "input_variable": "toa_reflectance",
    "task_band_config": "experiments_toa/configs/s2_task_bands_all.json",
    "x_transform": "none",
    "log_scale_qoi": [],
    "logit_scale_qoi": [],
    "response_noise_prior": False,
    "pac_bayes": False,
    "learn_inducing_locations": True,
}


def build_configspace(seed: int) -> ConfigurationSpace:
    cs = ConfigurationSpace(seed=seed)
    variant = Categorical("variant", ["pca", "nigp"], default="pca")
    lr = Float("lr", (1e-3, 1.0), default=0.05, log=True)
    variational_lr_mode = Categorical(
        "variational_lr_mode", ["same", "custom"], default="same"
    )
    variational_lr = Float("variational_lr", (1e-3, 1.0), default=0.05, log=True)
    num_inducing = Categorical("num_inducing", [256, 512, 1024], default=512)
    batch_size = Categorical("batch_size", [256, 512, 1024, 2048], default=1024)
    kl_beta = Float("kl_beta", (0.5, 1.0), default=1.0)
    adam_stop_patience = Categorical(
        "adam_stop_patience", [50, 100, 200], default=100
    )
    n_pca_components = Categorical(
        "n_pca_components", [4, 8, 12, 16, 24], default=12
    )
    freeze_epoch_nigp = Categorical(
        "freeze_epoch_nigp", [0, 50, 100, 200], default=50
    )
    cs.add(
        [
            variant,
            lr,
            variational_lr_mode,
            variational_lr,
            num_inducing,
            batch_size,
            kl_beta,
            adam_stop_patience,
            n_pca_components,
            freeze_epoch_nigp,
        ]
    )
    cs.add(
        [
            EqualsCondition(variational_lr, variational_lr_mode, "custom"),
            EqualsCondition(n_pca_components, variant, "pca"),
            EqualsCondition(freeze_epoch_nigp, variant, "nigp"),
        ]
    )
    return cs


def _json_safe(value: Any) -> Any:
    """Convert ConfigSpace / NumPy scalars to plain Python JSON types."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
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


def kwargs_from_cfg(
    cfg: dict[str, Any],
    *,
    num_epochs: int,
    seed: int,
    device: str,
    n_train: int,
    n_test: int,
    save_path: str,
) -> dict[str, Any]:
    """Map a SMAC config dict onto ``run_s2_toa_gp(svgp=True, ...)`` kwargs."""
    variant = str(cfg.get("variant", "pca"))
    use_nigp = variant == "nigp"
    vlr_mode = str(cfg.get("variational_lr_mode", "same"))
    variational_lr = (
        float(cfg["variational_lr"]) if vlr_mode == "custom" else None
    )
    return dict(
        n_train=n_train,
        n_test=n_test,
        num_inits=FIXED["num_inits"],
        num_epochs=num_epochs,
        optimizer_kwargs={**DEFAULT_ADAM_KWARGS, "lr": float(cfg["lr"])},
        seed=int(seed),
        device=device,
        dtype=FIXED["dtype"],
        ard=FIXED["ard"],
        save_path=save_path,
        monitor_validation=True,
        plot_validation=False,
        plot_posterior=False,
        training_verbose=False,
        log_every_n_epochs=max(25, num_epochs // 10),
        parallel_verbose=0,
        n_jobs=1,
        predict_chunk_size=4096,
        response_noise_prior=FIXED["response_noise_prior"],
        pac_bayes=FIXED["pac_bayes"],
        data_path=FIXED["data_path"],
        input_variable=FIXED["input_variable"],
        task_band_config=FIXED["task_band_config"],
        x_transform=FIXED["x_transform"],
        log_scale_qoi=FIXED["log_scale_qoi"],
        logit_scale_qoi=FIXED["logit_scale_qoi"],
        task_names=[TASK],
        svgp=True,
        num_inducing=int(cfg["num_inducing"]),
        learn_inducing_locations=bool(FIXED["learn_inducing_locations"]),
        batch_size=int(cfg["batch_size"]),
        variational_lr=variational_lr,
        kl_beta=float(cfg.get("kl_beta", 1.0)),
        adam_stop_patience=int(cfg.get("adam_stop_patience", 100)),
        nigp=use_nigp,
        freeze_epoch_nigp=int(cfg.get("freeze_epoch_nigp", 50)) if use_nigp else 0,
        n_pca_components=None if use_nigp else int(cfg.get("n_pca_components", 12)),
    )


def ide_snippet_from_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """IDE-facing knobs for ``experiments_SVGP/S2_toa_SVGP.py``."""
    variant = str(cfg.get("variant", "pca"))
    use_nigp = variant == "nigp"
    vlr_mode = str(cfg.get("variational_lr_mode", "same"))
    return {
        "NIGP": use_nigp,
        "N_PCA_COMPONENTS": None if use_nigp else int(cfg.get("n_pca_components", 12)),
        "FREEZE_EPOCH_NIGP": int(cfg.get("freeze_epoch_nigp", 50)) if use_nigp else 50,
        "LR": float(cfg["lr"]),
        "VARIATIONAL_LR": (
            float(cfg["variational_lr"]) if vlr_mode == "custom" else None
        ),
        "NUM_INDUCING": int(cfg["num_inducing"]),
        "BATCH_SIZE": int(cfg["batch_size"]),
        "KL_BETA": float(cfg.get("kl_beta", 1.0)),
        "ADAM_STOP_PATIENCE": int(cfg.get("adam_stop_patience", 100)),
        "N_TRAIN": FIXED["n_train"],
        "N_TEST": FIXED["n_test"],
        "DTYPE": "float32",
        "QOI": [TASK],
    }


def make_target(
    *,
    save_root: Path,
    device: str,
    n_train: int,
    n_test: int,
) -> Any:
    trial_counter = {"i": 0}

    def target(config: Configuration, seed: int = 0, budget: float = 200.0) -> float:
        trial_counter["i"] += 1
        trial_id = trial_counter["i"]
        num_epochs = max(1, int(round(float(budget))))
        cfg = config_to_row(config)
        run_dir = save_root / "runs" / f"trial{trial_id:04d}_ep{num_epochs}"
        run_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 72)
        print(f"SMAC trial {trial_id}  budget={num_epochs}  seed={seed}")
        print(f"  config={cfg}")

        t0 = time.time()
        cost = FAIL_COST
        metrics: dict[str, Any] = {}
        err: str | None = None
        try:
            metrics = run_s2_toa_gp(
                **kwargs_from_cfg(
                    cfg,
                    num_epochs=num_epochs,
                    seed=seed,
                    device=device,
                    n_train=n_train,
                    n_test=n_test,
                    save_path=str(run_dir),
                )
            )
            key = f"{TASK}_best_val_RRMSE"
            if key not in metrics or metrics[key] is None:
                raise KeyError(
                    f"Missing {key} in metrics (keys={sorted(metrics)[:20]}...)"
                )
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
            "algae_best_val_RRMSE": metrics.get("algae_best_val_RRMSE"),
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


def load_aug12(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Aug12 metrics not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        "source": str(path),
        "n_train": raw.get("n_train"),
        "n_test": raw.get("n_test"),
        "num_rff": raw.get("num_rff"),
        "nigp": raw.get("nigp"),
        "algae_RRMSE": raw.get("algae_RRMSE"),
        "algae_NLPD": raw.get("algae_NLPD"),
        "aggregate_RRMSE": raw.get("aggregate_RRMSE"),
    }


def best_variant_rows(trials_csv: Path) -> dict[str, dict[str, Any] | None]:
    """Best successful PCA / NIGP rows by val cost from trials.csv."""
    out: dict[str, dict[str, Any] | None] = {"pca": None, "nigp": None}
    if not trials_csv.is_file():
        return out
    with trials_csv.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("error") not in (None, "", "None"):
                continue
            variant = row.get("cfg_variant")
            if variant not in out:
                continue
            try:
                cost = float(row["cost_best_val_RRMSE"])
            except (KeyError, TypeError, ValueError):
                continue
            if cost >= FAIL_COST:
                continue
            prev = out[variant]
            if prev is None or cost < float(prev["cost_best_val_RRMSE"]):
                out[variant] = row
    return out


def retrain_incumbent(
    cfg: dict[str, Any],
    *,
    num_epochs: int,
    seed: int,
    device: str,
    n_train: int,
    n_test: int,
    save_root: Path,
) -> dict[str, Any]:
    run_dir = save_root / "runs" / f"incumbent_ep{num_epochs}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print(f"Retraining incumbent at max_budget={num_epochs}")
    print(f"  config={cfg}")
    t0 = time.time()
    metrics = run_s2_toa_gp(
        **kwargs_from_cfg(
            cfg,
            num_epochs=num_epochs,
            seed=seed,
            device=device,
            n_train=n_train,
            n_test=n_test,
            save_path=str(run_dir),
        )
    )
    wall_s = time.time() - t0
    return {
        "run_dir": str(run_dir),
        "wall_time_s": wall_s,
        "algae_RRMSE": metrics.get("algae_RRMSE"),
        "algae_best_val_RRMSE": metrics.get("algae_best_val_RRMSE"),
        "algae_best_val_NLL": metrics.get("algae_best_val_NLL"),
        "algae_NLPD": metrics.get("algae_NLPD"),
        "Training_Time": metrics.get("Training_Time"),
    }


def write_aug12_comparison(
    save_root: Path,
    *,
    inc_dict: dict[str, Any],
    incumbent_cost: float,
    reeval: dict[str, Any],
    aug12_path: Path,
) -> dict[str, Any]:
    aug12 = load_aug12(aug12_path)
    aug12_rrmse = aug12.get("algae_RRMSE")
    inc_test = reeval.get("algae_RRMSE")
    gap = None
    if aug12_rrmse is not None and inc_test is not None:
        gap = float(inc_test) - float(aug12_rrmse)

    best_by_variant = best_variant_rows(save_root / "trials.csv")
    payload = {
        "task": TASK,
        "incumbent_variant": inc_dict.get("variant"),
        "incumbent_cost_best_val_RRMSE": incumbent_cost,
        "incumbent_test_RRMSE": inc_test,
        "incumbent_reeval": reeval,
        "aug12": aug12,
        "gap_test_RRMSE_minus_aug12": gap,
        "on_par": (
            gap is not None and gap <= 0.005  # within 0.5pp of Aug12
        ),
        "best_pca_trial": best_by_variant.get("pca"),
        "best_nigp_trial": best_by_variant.get("nigp"),
    }
    out = save_root / "aug12_comparison.json"
    out.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    print("=" * 72)
    print("Aug12 comparison")
    print(f"  incumbent variant     : {inc_dict.get('variant')}")
    print(f"  incumbent val RRMSE   : {incumbent_cost:.6f}")
    print(f"  incumbent test RRMSE  : {inc_test}")
    print(f"  Aug12 algae RRMSE     : {aug12_rrmse}")
    print(f"  gap (inc - aug12)     : {gap}")
    print(f"Wrote {out}")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--n-trials", type=int, default=60)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--save-root",
        type=str,
        default="experiments_SVGP/results/Aug15/smac_s2_svgp_algae",
    )
    p.add_argument("--n-train", type=int, default=FIXED["n_train"])
    p.add_argument("--n-test", type=int, default=FIXED["n_test"])
    p.add_argument(
        "--min-budget",
        type=int,
        default=200,
        help="Min fidelity (num_epochs). Use 50 for smoke.",
    )
    p.add_argument(
        "--max-budget",
        type=int,
        default=600,
        help="Max fidelity (num_epochs). Use 50 for smoke.",
    )
    p.add_argument("--aug12", type=Path, default=AUG12_METRICS)
    p.add_argument(
        "--skip-reeval",
        action="store_true",
        help="Skip max-budget incumbent retrain (still writes Aug12 stub from trials).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.min_budget < 1 or args.max_budget < args.min_budget:
        raise SystemExit(
            f"Need 1 <= min_budget <= max_budget, got "
            f"{args.min_budget}, {args.max_budget}"
        )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            f"device={args.device!r} requested but this torch build has no CUDA "
            f"({torch.__version__}). Use the gpplus2026 env."
        )

    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)
    cs = build_configspace(args.seed)
    target = make_target(
        save_root=save_root,
        device=args.device,
        n_train=args.n_train,
        n_test=args.n_test,
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
        f"device={args.device}  n_train={args.n_train}  save_root={save_root}"
    )
    smac = MultiFidelityFacade(scenario, target)
    incumbent = smac.optimize()
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
        "n_train": args.n_train,
        "n_test": args.n_test,
        "save_root": str(save_root),
        "ide_snippet": ide_snippet_from_cfg(inc_dict),
    }
    (save_root / "incumbent.json").write_text(
        json.dumps(_json_safe(incumbent_payload), indent=2), encoding="utf-8"
    )

    if args.skip_reeval:
        reeval = {
            "skipped": True,
            "algae_RRMSE": None,
            "note": "Pass without --skip-reeval to retrain at max_budget.",
        }
    else:
        reeval = retrain_incumbent(
            inc_dict,
            num_epochs=args.max_budget,
            seed=args.seed,
            device=args.device,
            n_train=args.n_train,
            n_test=args.n_test,
            save_root=save_root,
        )

    write_aug12_comparison(
        save_root,
        inc_dict=inc_dict,
        incumbent_cost=cost,
        reeval=reeval,
        aug12_path=args.aug12,
    )

    summary = {
        "n_trials": args.n_trials,
        "min_budget": args.min_budget,
        "max_budget": args.max_budget,
        "seed": args.seed,
        "device": args.device,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "save_root": str(save_root),
        "incumbent_cost": cost,
        "incumbent": inc_dict,
        "incumbent_reeval": reeval,
    }
    (save_root / "smac_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2), encoding="utf-8"
    )

    print("=" * 72)
    print("Incumbent config:")
    print(json.dumps(_json_safe(inc_dict), indent=2))
    print(f"Incumbent cost (algae_best_val_RRMSE): {cost:.6f}")
    print(f"Wrote {save_root / 'incumbent.json'}")
    print("IDE snippet keys under incumbent.json -> ide_snippet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
