"""Profile a short Aug08-matched SORF+NIGP training run (10 Adam epochs).

Runs two phases on one QoI with production ``n_train`` / ``num_rff`` / float64:

1. Homoskedastic warm-start path (``freeze_epoch_nigp >= num_epochs``)
2. NIGP-active path (``freeze_epoch_nigp=0``)

Uses cProfile + ``GPPLUS_WOODBURY_CUDA_TIMING`` section averages.

Usage (repo root)::

    python -m experiments_toa.profile_sorf_10epoch
    python -m experiments_toa.profile_sorf_10epoch --qoi grain_size --epochs 10
"""

from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
_SORF = _ROOT / "experiments_SORF"
_RFF = _ROOT / "experiments_RFF"
_MTGPR = _ROOT / "experiments_RFFMTGPR"
for p in (_ROOT, _SORF, _RFF, _MTGPR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiments_toa.paths import pin_toa_import_paths

pin_toa_import_paths(_MTGPR, _RFF, _SORF)

import gpplus
from gpplus.training.nigp_mll import NIGPWoodburyMarginalLogLikelihood
from gpplus.utils import nigp_utils
from gpplus.utils.woodbury_mll_autograd import (
    cuda_timing_averages,
    reset_cuda_timing,
    set_cuda_timing,
)
from mtgpr_experiment_utils import DEFAULT_ADAM_KWARGS
from toa_s2_sorf_base import run_s2_toa_sorf

_NIGP_SECTION_MS: dict[str, list[float]] = {}


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _record_nigp_ms(name: str, ms: float) -> None:
    _NIGP_SECTION_MS.setdefault(name, []).append(ms)


class _CudaTimer:
    def __init__(self, name: str):
        self.name = name
        self.enabled = torch.cuda.is_available()
        self.start = None
        self.end = None

    def __enter__(self):
        if not self.enabled:
            self.t0 = time.perf_counter()
            return self
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()
        return self

    def __exit__(self, *exc):
        if not self.enabled:
            _record_nigp_ms(self.name, (time.perf_counter() - self.t0) * 1000.0)
            return False
        self.end.record()
        self.end.synchronize()
        _record_nigp_ms(self.name, float(self.start.elapsed_time(self.end)))
        return False


def _install_nigp_timers() -> None:
    """Wrap NIGP MLL pieces with CUDA event timers (profile-only)."""
    if getattr(_install_nigp_timers, "_installed", False):
        _NIGP_SECTION_MS.clear()
        return

    import gpplus.training.nigp_mll as nigp_mll_mod

    _orig_grad = nigp_utils.posterior_mean_grad_wrt_x
    _orig_fwd = NIGPWoodburyMarginalLogLikelihood.forward

    def timed_grad(*args, **kwargs):
        with _CudaTimer("nigp_grad_mu"):
            return _orig_grad(*args, **kwargs)

    def timed_fwd(self, *args, **kwargs):
        with _CudaTimer("nigp_mll_forward"):
            return _orig_fwd(self, *args, **kwargs)

    nigp_utils.posterior_mean_grad_wrt_x = timed_grad
    nigp_mll_mod.posterior_mean_grad_wrt_x = timed_grad
    NIGPWoodburyMarginalLogLikelihood.forward = timed_fwd
    _install_nigp_timers._installed = True
    _NIGP_SECTION_MS.clear()


def _print_nigp_avgs(label: str) -> None:
    if not _NIGP_SECTION_MS:
        print(f"[{label}] no NIGP section timings (freeze path or unused)")
        return
    print(f"\n=== [{label}] NIGP section avg (ms/call) ===")
    for k, vals in sorted(_NIGP_SECTION_MS.items()):
        print(f"  {k:20s}  {sum(vals)/len(vals):8.2f} ms  (n={len(vals)})")


def _print_cuda_avgs(label: str) -> None:
    avgs = cuda_timing_averages()
    if not avgs:
        print(f"[{label}] no CUDA section timings recorded")
        return
    total = sum(avgs.values())
    print(f"\n=== [{label}] Woodbury CUDA section avg (ms/call) ===")
    for k in sorted(avgs, key=avgs.get, reverse=True):
        pct = 100.0 * avgs[k] / total if total else 0.0
        print(f"  {k:16s}  {avgs[k]:8.2f} ms  ({pct:5.1f}%)")
    print(f"  {'SUM_SECTIONS':16s}  {total:8.2f} ms")


def _run_phase(
    *,
    label: str,
    qoi: str,
    n_train: int,
    n_test: int,
    num_rff: int,
    num_epochs: int,
    freeze_epoch_nigp: int,
    lr: float,
    seed: int,
    data_path: str | None,
    task_band_config: str | None,
    out_prof: Path,
    save_path: Path,
    nigp_slope_refreshes: int | None = None,
) -> dict:
    os.environ["GPPLUS_WOODBURY_CUDA_TIMING"] = "1"
    set_cuda_timing(True)
    reset_cuda_timing()
    _install_nigp_timers()
    _NIGP_SECTION_MS.clear()

    gpplus.config.configure_logger(level=getattr(__import__("logging"), "WARNING"))

    optimizer_kwargs = {**DEFAULT_ADAM_KWARGS, "lr": lr}
    profiler = cProfile.Profile()
    _cuda_sync()
    t0 = time.perf_counter()
    profiler.enable()
    try:
        metrics = run_s2_toa_sorf(
            n_train=n_train,
            n_test=n_test,
            num_rff=num_rff,
            num_inits=1,
            init_batch_size=1,
            num_epochs=num_epochs,
            optimizer_kwargs=optimizer_kwargs,
            seed=seed,
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float64,
            ard=True,
            save_path=str(save_path),
            n_jobs=1,
            predict_chunk_size=512,
            monitor_validation=False,
            plot_validation=False,
            plot_posterior=False,
            save_checkpoint=False,
            training_verbose=True,
            log_every_n_epochs=max(1, num_epochs // 2),
            response_noise_prior=False,
            correct_sorf=True,
            spectral_kernel="rbf",
            data_path=data_path,
            task_names=[qoi],
            task_band_config=task_band_config,
            x_transform="none",
            log_scale_qoi=[],
            logit_scale_qoi=[],
            nigp=True,
            freeze_epoch_nigp=freeze_epoch_nigp,
            nigp_slope_refreshes=nigp_slope_refreshes,
            freeze_epoch_noise=0,
            adam_stop_patience=10_000,  # never early-stop a short profile
            train_mode="independent",
            mean_type="constant",
        )
    finally:
        _cuda_sync()
        profiler.disable()
    wall = time.perf_counter() - t0

    out_prof.parent.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(str(out_prof))
    stats = pstats.Stats(profiler)
    print(f"\n=== [{label}] wall={wall:.1f}s  ({wall / max(num_epochs, 1):.2f}s/epoch) ===")
    print(f"Wrote {out_prof}")
    stats.strip_dirs().sort_stats("cumulative")
    stats.print_stats(35)
    print(f"\n=== [{label}] tottime top 25 ===")
    stats.sort_stats("tottime")
    stats.print_stats(25)
    _print_cuda_avgs(label)
    _print_nigp_avgs(label)

    train_t = float(metrics.get(f"{qoi}_Training_Time", metrics.get("Training_Time", wall)))
    return {
        "label": label,
        "wall_s": wall,
        "s_per_epoch": wall / max(num_epochs, 1),
        "train_time_s": train_t,
        "cuda_avgs_ms": cuda_timing_averages(),
        "nigp_avgs_ms": {
            k: sum(v) / len(v) for k, v in _NIGP_SECTION_MS.items() if v
        },
        "nigp_grad_mu_calls": len(_NIGP_SECTION_MS.get("nigp_grad_mu", [])),
        "rrmse": metrics.get(f"{qoi}_RRMSE"),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qoi", default="algae", help="Single QoI (default: algae)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--n-train", type=int, default=40000)
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--num-rff", type=int, default=1600)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--data-path",
        default="experiments_toa/data 11 QoI/snow_toa_fsnow_only_20260308.nc",
    )
    p.add_argument(
        "--task-band-config",
        default="experiments_toa/configs/s2_task_bands_all.json",
    )
    p.add_argument(
        "--phases",
        default="both",
        choices=("both", "freeze", "nigp"),
        help="Which training paths to profile",
    )
    p.add_argument(
        "--slope-refreshes",
        type=int,
        default=None,
        help="NIGP outer-loop slope refreshes (default: every epoch)",
    )
    p.add_argument("--out-dir", type=Path, default=Path("profiles/sorf_10epoch"))
    args = p.parse_args(argv)

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available; profile will be on CPU and not representative.")

    results = []
    if args.phases in ("both", "freeze"):
        results.append(
            _run_phase(
                label="freeze_homoskedastic",
                qoi=args.qoi,
                n_train=args.n_train,
                n_test=args.n_test,
                num_rff=args.num_rff,
                num_epochs=args.epochs,
                freeze_epoch_nigp=args.epochs + 1,
                lr=args.lr,
                seed=args.seed,
                data_path=args.data_path,
                task_band_config=args.task_band_config,
                out_prof=args.out_dir / f"{args.qoi}_freeze.prof",
                save_path=args.out_dir / f"run_{args.qoi}_freeze",
                nigp_slope_refreshes=args.slope_refreshes,
            )
        )
    if args.phases in ("both", "nigp"):
        slope_tag = (
            f"_sr{args.slope_refreshes}" if args.slope_refreshes is not None else ""
        )
        results.append(
            _run_phase(
                label=f"nigp_active{slope_tag}",
                qoi=args.qoi,
                n_train=args.n_train,
                n_test=args.n_test,
                num_rff=args.num_rff,
                num_epochs=args.epochs,
                freeze_epoch_nigp=0,
                lr=args.lr,
                seed=args.seed,
                data_path=args.data_path,
                task_band_config=args.task_band_config,
                out_prof=args.out_dir / f"{args.qoi}_nigp{slope_tag}.prof",
                save_path=args.out_dir / f"run_{args.qoi}_nigp{slope_tag}",
                nigp_slope_refreshes=args.slope_refreshes,
            )
        )

    print("\n========== PROFILE SUMMARY ==========")
    for r in results:
        print(
            f"{r['label']:22s}  {r['s_per_epoch']:7.2f} s/epoch  "
            f"wall={r['wall_s']:.1f}s  grad_mu_calls={r.get('nigp_grad_mu_calls', 0)}  "
            f"test_RRMSE={r['rrmse']}"
        )
        if r["cuda_avgs_ms"]:
            top = sorted(r["cuda_avgs_ms"].items(), key=lambda kv: -kv[1])[:5]
            print("   cuda top:", ", ".join(f"{k}={v:.1f}ms" for k, v in top))

    if len(results) == 2:
        a, b = results[0], results[1]
        if a["s_per_epoch"] > 0:
            print(
                f"\nNIGP/freeze epoch-time ratio: "
                f"{b['s_per_epoch'] / a['s_per_epoch']:.2f}x"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
