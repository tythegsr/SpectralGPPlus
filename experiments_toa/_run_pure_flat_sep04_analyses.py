"""Run TOA/ASD analysis suite on snow_toa_fsnow_pure_flat_Sep04.nc."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
SIM = _ROOT / "experiments_toa" / "data 11 QoI" / "snow_toa_fsnow_pure_flat_Sep04.nc"
ASD = _ROOT / "experiments_toa" / "data 11 QoI" / "asd_validation_set.nc"
OUT_ROOT = _ROOT / "experiments_toa" / "analysis_pure_flat_Sep04"
EVAL_DIR = (
    _ROOT
    / "experiments_SORF"
    / "results"
    / "Sept04_log"
    / "s4_emit_aotbelow02_sorf_1inits_numrff1000_lr0.04_nigp_freezeepochnigp200_sloperefreshes10_dtypefloat32"
    / "asd_validation_eval"
)
# Prefer ASD-forced training eval if present
EVAL_ASD = (
    _ROOT
    / "experiments_SORF"
    / "results"
    / "Sept04_log_asd"
    / "s4_emit_aotbelow02_sorf_1inits_numrff1000_lr0.04_nigp_freezeepochnigp200_sloperefreshes10_dtypefloat32"
    / "asd_validation_eval"
)

PY = sys.executable


def run(name: str, cmd: list[str]) -> tuple[str, int, float, str]:
    print("\n" + "=" * 72)
    print(f"RUN: {name}")
    print(" ".join(cmd))
    print("=" * 72, flush=True)
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        dt = time.time() - t0
        out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        # print last part of output
        tail = "\n".join(out.splitlines()[-40:])
        print(tail)
        print(f"[{name}] exit={proc.returncode}  {dt:.1f}s", flush=True)
        return name, proc.returncode, dt, out
    except Exception as exc:  # noqa: BLE001
        dt = time.time() - t0
        print(f"[{name}] FAILED: {exc!r}")
        return name, 1, dt, repr(exc)


def main() -> None:
    if not SIM.is_file():
        raise SystemExit(f"Missing sim: {SIM}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    eval_dir = EVAL_ASD if (EVAL_ASD / "asd_validation_metrics.json").is_file() else EVAL_DIR
    print(f"SIM={SIM}")
    print(f"OUT={OUT_ROOT}")
    print(f"EVAL={eval_dir} exists={(eval_dir / 'asd_validation_metrics.json').is_file()}")

    jobs: list[tuple[str, list[str]]] = [
        (
            "asd_spectral_neighbors",
            [
                PY,
                "experiments_toa/asd_spectral_neighbors.py",
                "--asd",
                str(ASD),
                "--sim",
                str(SIM),
                "--thr",
                "0.04",
                "--out-dir",
                str(OUT_ROOT / "asd_neighbors"),
                "--preds",
                str(eval_dir / "asd_validation_predictions.npz"),
            ],
        ),
        (
            "analyze_flat_multiplicity",
            [
                PY,
                "experiments_toa/analyze_flat_multiplicity.py",
                "--asd",
                str(ASD),
                "--sim",
                str(SIM),
                "--out-dir",
                str(OUT_ROOT / "multiplicity"),
                "--thr",
                "0.05",
            ],
        ),
        (
            "compare_asd_sim_neighbors",
            [
                PY,
                "experiments_toa/compare_asd_sim_neighbors.py",
                "--asd",
                str(ASD),
                "--sim",
                str(SIM),
                "--out-dir",
                str(OUT_ROOT / "nn_compare"),
                "--asd-eval-dir",
                str(eval_dir),
                "--k",
                "8",
            ],
        ),
        (
            "diagnose_asd_vs_synthetic",
            [
                PY,
                "experiments_toa/diagnose_asd_vs_synthetic.py",
                "--asd-path",
                str(ASD),
                "--sim-path",
                str(SIM),
                "--out",
                str(OUT_ROOT / "diagnose_asd_vs_synthetic.json"),
                "--no-metrics",
            ],
        ),
        (
            "s2_histogram_analysis",
            [
                PY,
                "experiments_toa/s2_histogram_analysis.py",
                "--data-path",
                str(SIM),
                "--out-dir",
                str(OUT_ROOT / "histograms"),
            ],
        ),
        (
            "s2_correlation_analysis",
            [
                PY,
                "experiments_toa/s2_correlation_analysis.py",
                "--data-path",
                str(SIM),
                "--out-dir",
                str(OUT_ROOT / "correlation"),
            ],
        ),
        (
            "s2_nonlinear_diagnostics",
            [
                PY,
                "experiments_toa/s2_nonlinear_diagnostics.py",
                "--data-path",
                str(SIM),
                "--out-dir",
                str(OUT_ROOT / "nonlinear"),
                "--n-subsample",
                "12000",
            ],
        ),
    ]

    # diagnose_asd_flat_sim needs eval metrics
    if (eval_dir / "asd_validation_metrics.json").is_file():
        jobs.insert(
            2,
            (
                "diagnose_asd_flat_sim",
                [
                    PY,
                    "experiments_toa/diagnose_asd_flat_sim.py",
                    "--asd",
                    str(ASD),
                    "--sim",
                    str(SIM),
                    "--eval-dir",
                    str(eval_dir),
                    "--out-dir",
                    str(OUT_ROOT / "diagnose_flat"),
                ],
            ),
        )
    else:
        print("SKIP diagnose_asd_flat_sim: no asd_validation_metrics.json")

    # Optional band selection
    band_cfg = _ROOT / "experiments_toa" / "s2_task_band_config.json"
    band_script = _ROOT / "experiments_toa" / "s2_per_qoi_band_selection.py"
    if band_script.is_file():
        cmd = [
            PY,
            str(band_script),
            "--data-path",
            str(SIM),
            "--out-dir",
            str(OUT_ROOT / "band_selection"),
        ]
        # only add band-config if flag exists — try without first via help? Just include if file exists
        if band_cfg.is_file():
            cmd.extend(["--band-config", str(band_cfg)])
        jobs.append(("s2_per_qoi_band_selection", cmd))

    results = []
    for name, cmd in jobs:
        results.append(run(name, cmd))

    # Write summary log
    log_lines = [f"SIM={SIM}", f"OUT={OUT_ROOT}", ""]
    for name, code, dt, out in results:
        status = "OK" if code == 0 else f"FAIL({code})"
        log_lines.append(f"{status:10s}  {dt:7.1f}s  {name}")
        # save per-job log
        (OUT_ROOT / f"_log_{name}.txt").write_text(out, encoding="utf-8")
    summary = "\n".join(log_lines)
    (OUT_ROOT / "_run_summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    n_fail = sum(1 for _, c, _, _ in results if c != 0)
    raise SystemExit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
