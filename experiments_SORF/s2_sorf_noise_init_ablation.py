"""SORF OFAT ablation: init distributions (and optional noise priors) for S2 QoIs.

Default mode is **without** a response-noise prior: only raw_noise /
raw_lengthscale / raw_outputscale initializations are swept.

Use ``--with-response-noise-prior`` to also sweep prior fraction / log-scale.

Tasks: cos_i, algae, grain_size, fsnow, aot.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from copy import deepcopy
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

import gpplus
from mtgpr_experiment_utils import DEFAULT_ADAM_KWARGS
from toa_s2_sorf_base import run_s2_toa_sorf

ABLATION_TASKS: tuple[str, ...] = ("cos_i", "algae", "grain_size", "fsnow", "aot")
FAMILIES: tuple[str, ...] = ("prior", "noise", "lengthscale", "outputscale")

DEFAULT_SAVE_ROOT_NO_PRIOR = _SORF_DIR / "results" / "July18" / "ablation_noise_init_prior_off"
DEFAULT_SAVE_ROOT_WITH_PRIOR = _SORF_DIR / "results" / "ablation_noise_init"

# Held knobs shared by both modes.
BASELINE: dict[str, Any] = {
    "n_train": 16000,
    "n_test": 5000,
    "num_rff": 1600,
    "num_inits": 16,
    "num_epochs": 300,
    "lr": 0.01,
    "dtype": "float32",
    "ard": True,
    "noise_var_fraction": 0.001,
    "noise_prior_log_scale": 0.5,
    "correct_sorf": True,
    "log_scale_qoi": ["algae", "dust", "grain_size", "liquid_water"],
    "logit_scale_qoi": ["cos_i", "aot"],
}


def _cfg_label(cfg: dict[str, Any] | None) -> str:
    if not cfg:
        return "default"
    method = cfg.get("method", "?")
    if method == "uniform":
        return f"uniform[{cfg['lower']:g},{cfg['upper']:g}]"
    if method == "normal":
        return f"normal(mean={cfg['mean']:g},std={cfg['std']:g})"
    if method == "constant":
        return "constant(prior_target)"
    if method == "prior_target":
        return "constant(prior_target)"
    return str(cfg)


def build_trial_grid(*, response_noise_prior: bool) -> list[dict[str, Any]]:
    """OFAT trials. When ``response_noise_prior`` is False, skip the prior family."""
    trials: list[dict[str, Any]] = []
    prior_on = bool(response_noise_prior)

    def add(
        trial_id: str,
        family: str,
        *,
        response_noise_prior: bool = prior_on,
        noise_var_fraction: float = float(BASELINE["noise_var_fraction"]),
        noise_prior_log_scale: float = float(BASELINE["noise_prior_log_scale"]),
        init_overrides: dict[str, Any] | None = None,
        note: str = "",
    ) -> None:
        trials.append(
            {
                "trial_id": trial_id,
                "family": family,
                "response_noise_prior": response_noise_prior,
                "noise_var_fraction": noise_var_fraction,
                "noise_prior_log_scale": noise_prior_log_scale,
                "initializer_parameter_configs": deepcopy(init_overrides or {}),
                "note": note,
            }
        )

    if prior_on:
        add(
            "B0",
            "baseline",
            note="shared baseline (prior on, frac=1e-3, scale=0.5, default/constant-from-prior inits)",
        )
        add("P0", "prior", response_noise_prior=False, note="prior off")
        add("P1", "prior", noise_var_fraction=1e-4)
        add("P3", "prior", noise_var_fraction=1e-2)
        add("P4", "prior", noise_var_fraction=5e-2)
        add("P5", "prior", noise_var_fraction=2.5e-1)
        add("P6", "prior", noise_prior_log_scale=0.1)
        add("P8", "prior", noise_prior_log_scale=1.0)
        add("P9", "prior", noise_var_fraction=1e-5, note="lower prior-center")
    else:
        add(
            "B0",
            "baseline",
            response_noise_prior=False,
            note="shared baseline (prior off, default inits; noise ~ uniform[-5,-1])",
        )

    # With prior on: N0 is a free uniform draw (B0 uses constant-from-prior).
    # With prior off: N0 uniform[-5,-1] == library default (= B0) — skip it.
    if prior_on:
        add(
            "N0",
            "noise",
            init_overrides={"raw_noise": {"method": "uniform", "lower": -5.0, "upper": -1.0}},
        )
    add(
        "N1",
        "noise",
        init_overrides={"raw_noise": {"method": "uniform", "lower": -5.0, "upper": -3.0}},
    )
    add(
        "N2",
        "noise",
        init_overrides={"raw_noise": {"method": "uniform", "lower": -3.0, "upper": 0.0}},
    )
    add(
        "N3",
        "noise",
        init_overrides={"raw_noise": {"method": "normal", "mean": -3.0, "std": 1.0}},
    )
    add(
        "N5",
        "noise",
        init_overrides={"raw_noise": {"method": "uniform", "lower": -5.0, "upper": -4.0}},
        note="lower-range noise init",
    )

    add(
        "L1",
        "lengthscale",
        init_overrides={"raw_lengthscale": {"method": "normal", "mean": -1.0, "std": 1.0}},
    )
    add(
        "L2",
        "lengthscale",
        init_overrides={"raw_lengthscale": {"method": "normal", "mean": -3.0, "std": 1.0}},
    )
    add(
        "L3",
        "lengthscale",
        init_overrides={"raw_lengthscale": {"method": "uniform", "lower": -4.0, "upper": 0.0}},
    )
    add(
        "L4",
        "lengthscale",
        init_overrides={"raw_lengthscale": {"method": "uniform", "lower": -6.0, "upper": -4.0}},
        note="lower-range lengthscale init",
    )

    add(
        "O1",
        "outputscale",
        init_overrides={"raw_outputscale": {"method": "normal", "mean": 0.0, "std": 0.5}},
    )
    add(
        "O2",
        "outputscale",
        init_overrides={"raw_outputscale": {"method": "normal", "mean": -2.0, "std": 1.5}},
    )
    add(
        "O3",
        "outputscale",
        init_overrides={"raw_outputscale": {"method": "uniform", "lower": -3.0, "upper": 0.0}},
    )
    add(
        "O4",
        "outputscale",
        init_overrides={"raw_outputscale": {"method": "uniform", "lower": -5.0, "upper": -3.0}},
        note="lower-range outputscale init",
    )

    if not prior_on:
        for t in trials:
            t["response_noise_prior"] = False

    return trials


def _row_path(save_root: Path, task: str, trial_id: str) -> Path:
    return save_root / "rows" / f"{trial_id}__{task}.json"


def _extract_row(trial: dict[str, Any], task: str, metrics: dict[str, Any], wall_s: float) -> dict[str, Any]:
    rmse = metrics.get(f"{task}_RMSE", metrics.get("RMSE"))
    rrmse = metrics.get(f"{task}_RRMSE", metrics.get("aggregate_RRMSE"))
    noise = metrics.get(f"{task}_noise")
    noise_std = metrics.get(f"{task}_noise_std")
    best_loss = metrics.get(f"{task}_best_train_loss")
    ratio = None
    if noise_std is not None and rmse is not None and float(rmse) > 0:
        ratio = float(noise_std) / float(rmse)

    pcs = trial.get("initializer_parameter_configs") or {}
    return {
        "trial_id": trial["trial_id"],
        "family": trial["family"],
        "task": task,
        "note": trial.get("note", ""),
        "response_noise_prior": trial["response_noise_prior"],
        "noise_var_fraction": trial["noise_var_fraction"],
        "noise_prior_log_scale": trial["noise_prior_log_scale"],
        "raw_noise_init": _cfg_label(pcs.get("raw_noise")),
        "raw_lengthscale_init": _cfg_label(pcs.get("raw_lengthscale")),
        "raw_outputscale_init": _cfg_label(pcs.get("raw_outputscale")),
        "initializer_parameter_configs": pcs,
        "RRMSE": float(rrmse) if rrmse is not None else None,
        "RMSE": float(rmse) if rmse is not None else None,
        "noise": float(noise) if noise is not None else None,
        "noise_std": float(noise_std) if noise_std is not None else None,
        "noise_std_over_RMSE": ratio,
        "best_train_loss": float(best_loss) if best_loss is not None else None,
        "wall_time_s": float(wall_s),
        "n_train": BASELINE["n_train"],
        "n_test": BASELINE["n_test"],
        "num_epochs": BASELINE["num_epochs"],
        "num_inits": BASELINE["num_inits"],
        "num_rff": BASELINE["num_rff"],
    }


def _append_csv(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial_id",
        "family",
        "task",
        "note",
        "response_noise_prior",
        "noise_var_fraction",
        "noise_prior_log_scale",
        "raw_noise_init",
        "raw_lengthscale_init",
        "raw_outputscale_init",
        "RRMSE",
        "RMSE",
        "noise",
        "noise_std",
        "noise_std_over_RMSE",
        "best_train_loss",
        "wall_time_s",
        "n_train",
        "n_test",
        "num_epochs",
        "num_inits",
        "num_rff",
    ]
    write_header = not csv_path.is_file()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_one_trial(
    trial: dict[str, Any],
    task: str,
    *,
    save_root: Path,
    device: str,
    seed: int,
    skip_existing: bool,
) -> dict[str, Any] | None:
    row_file = _row_path(save_root, task, trial["trial_id"])
    if skip_existing and row_file.is_file():
        print(f"[skip] {trial['trial_id']} / {task} ({row_file.name})")
        return json.loads(row_file.read_text(encoding="utf-8"))

    run_dir = save_root / "runs" / f"{trial['trial_id']}__{task}"
    run_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float32 if BASELINE["dtype"] == "float32" else torch.float64
    optimizer_kwargs = {**DEFAULT_ADAM_KWARGS, "lr": BASELINE["lr"]}

    print("=" * 72)
    print(f"Trial {trial['trial_id']}  family={trial['family']}  task={task}")
    print(
        f"  prior={trial['response_noise_prior']}  "
        f"frac={trial['noise_var_fraction']}  "
        f"log_scale={trial['noise_prior_log_scale']}"
    )
    print(f"  inits={trial['initializer_parameter_configs'] or '{}'}")
    if trial.get("note"):
        print(f"  note={trial['note']}")

    t0 = time.time()
    metrics = run_s2_toa_sorf(
        n_train=BASELINE["n_train"],
        n_test=BASELINE["n_test"],
        num_rff=BASELINE["num_rff"],
        num_inits=BASELINE["num_inits"],
        num_epochs=BASELINE["num_epochs"],
        optimizer_kwargs=optimizer_kwargs,
        seed=seed,
        device=device,
        dtype=dtype,
        ard=BASELINE["ard"],
        save_path=str(run_dir),
        monitor_validation=False,
        plot_validation=False,
        plot_posterior=False,
        save_checkpoint=False,
        training_verbose=False,
        log_every_n_epochs=50,
        response_noise_prior=trial["response_noise_prior"],
        noise_var_fraction=trial["noise_var_fraction"],
        noise_prior_log_scale=trial["noise_prior_log_scale"],
        initializer_parameter_configs=trial["initializer_parameter_configs"] or None,
        correct_sorf=BASELINE["correct_sorf"],
        log_scale_qoi=BASELINE["log_scale_qoi"],
        logit_scale_qoi=BASELINE["logit_scale_qoi"],
        task_names=[task],
        parallel_verbose=0,
    )
    wall_s = time.time() - t0
    row = _extract_row(trial, task, metrics, wall_s)
    row_file.parent.mkdir(parents=True, exist_ok=True)
    row_file.write_text(json.dumps(row, indent=2), encoding="utf-8")
    _append_csv(save_root / "ablation_results.csv", row)
    print(
        f"  done RRMSE={row['RRMSE']:.6f}  noise_std/RMSE="
        f"{row['noise_std_over_RMSE'] if row['noise_std_over_RMSE'] is not None else float('nan'):.3f}  "
        f"wall={wall_s:.1f}s"
    )
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S2 SORF noise/prior/init OFAT ablation")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(ABLATION_TASKS),
        choices=list(ABLATION_TASKS),
        help="QoIs to run (default: all five)",
    )
    parser.add_argument(
        "--only-family",
        choices=list(FAMILIES) + ["baseline", "all"],
        default="all",
        help="Restrict to one OFAT family (plus baseline always available via family=baseline)",
    )
    parser.add_argument(
        "--with-response-noise-prior",
        action="store_true",
        help="Also sweep noise-prior settings (default: prior off; init-only study)",
    )
    parser.add_argument("--save-root", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--log-level", type=str, default="WARNING")
    args = parser.parse_args(argv)

    prior_on = bool(args.with_response_noise_prior)
    if args.save_root is None:
        save_root = DEFAULT_SAVE_ROOT_WITH_PRIOR if prior_on else DEFAULT_SAVE_ROOT_NO_PRIOR
    else:
        save_root = Path(args.save_root)

    gpplus.config.configure_logger(level=getattr(logging, args.log_level.upper()))
    save_root.mkdir(parents=True, exist_ok=True)

    trials = build_trial_grid(response_noise_prior=prior_on)
    if args.only_family != "all":
        if args.only_family == "baseline":
            trials = [t for t in trials if t["family"] == "baseline"]
        elif args.only_family == "prior" and not prior_on:
            print("No prior family in --no-prior mode; nothing to run for --only-family prior.")
            return 0
        else:
            trials = [
                t
                for t in trials
                if t["family"] in ("baseline", args.only_family)
            ]

    pairs = [(t, task) for t in trials for task in args.tasks]
    mode = "WITH prior" if prior_on else "NO prior"
    print(
        f"Ablation mode={mode}: {len(trials)} configs × {len(args.tasks)} tasks "
        f"= {len(pairs)} runs  -> {save_root}"
    )
    if args.dry_run:
        for t in trials:
            pcs = t["initializer_parameter_configs"] or {}
            print(
                f"  {t['trial_id']:4s}  family={t['family']:12s}  "
                f"prior={t['response_noise_prior']}  "
                f"frac={t['noise_var_fraction']:g}  "
                f"s={t['noise_prior_log_scale']:g}  "
                f"inits={{{', '.join(f'{k}:{_cfg_label(v)}' for k, v in pcs.items()) or 'default'}}}"
                + (f"  ({t['note']})" if t.get("note") else "")
            )
        return 0

    (save_root / "ablation_grid.json").write_text(
        json.dumps(
            {
                "mode": mode,
                "response_noise_prior": prior_on,
                "baseline": BASELINE,
                "trials": trials,
                "tasks": list(args.tasks),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    for trial, task in pairs:
        run_one_trial(
            trial,
            task,
            save_root=save_root,
            device=args.device,
            seed=args.seed,
            skip_existing=not args.no_skip_existing,
        )

    all_rows: list[dict[str, Any]] = []
    rows_dir = save_root / "rows"
    if rows_dir.is_dir():
        for path in sorted(rows_dir.glob("*.json")):
            all_rows.append(json.loads(path.read_text(encoding="utf-8")))
    (save_root / "ablation_results.json").write_text(
        json.dumps(all_rows, indent=2), encoding="utf-8"
    )
    csv_path = save_root / "ablation_results.csv"
    if csv_path.is_file():
        csv_path.unlink()
    for row in all_rows:
        _append_csv(csv_path, row)
    print(f"Wrote {len(all_rows)} rows to {save_root / 'ablation_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
