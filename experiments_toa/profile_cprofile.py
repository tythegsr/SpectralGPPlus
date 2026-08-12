"""Run an experiment script under cProfile and write a snakeviz-ready .prof file.

Usage (from repo root)::

    python -m experiments_toa.profile_cprofile --script experiments_SORF/S2_toa_SORF.py
    snakeviz profiles/S2_toa_SORF.prof

Shorten the IDE config first (few epochs, one QoI, validation/plots off) — see
``experiments_toa/PROFILING.md``.
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import runpy
import sys
from pathlib import Path


def _cuda_synchronize() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _default_out_path(script: Path) -> Path:
    return Path("profiles") / f"{script.stem}.prof"


def profile_script(
    script: Path,
    out: Path,
    *,
    sort: str = "cumulative",
    top: int = 40,
    script_args: list[str] | None = None,
) -> None:
    script = script.resolve()
    if not script.is_file():
        raise FileNotFoundError(f"Script not found: {script}")

    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    # Match ``python script.py ...`` argv for scripts that use argparse.
    saved_argv = sys.argv[:]
    sys.argv = [str(script), *(script_args or [])]

    profiler = cProfile.Profile()
    _cuda_synchronize()
    profiler.enable()
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        _cuda_synchronize()
        profiler.disable()
        sys.argv = saved_argv

    profiler.dump_stats(str(out))
    stats = pstats.Stats(profiler)
    stats.strip_dirs().sort_stats(sort)
    print(f"\n=== cProfile top {top} ({sort}) ===")
    stats.print_stats(top)
    print(f"\nWrote {out}")
    print(f"View with: snakeviz {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Profile a training script with cProfile (snakeviz-compatible)."
    )
    parser.add_argument(
        "--script",
        type=Path,
        required=True,
        help="Path to the experiment script (e.g. experiments_SORF/S2_toa_SORF.py)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .prof path (default: profiles/<script_stem>.prof)",
    )
    parser.add_argument(
        "--sort",
        default="cumulative",
        choices=("cumulative", "tottime", "calls", "filename"),
        help="pstats sort key for the printed summary",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=40,
        help="Number of rows in the printed summary",
    )
    parser.add_argument(
        "script_args",
        nargs="*",
        help="Optional args forwarded to the script as sys.argv[1:]",
    )
    args = parser.parse_args(argv)

    out = args.out if args.out is not None else _default_out_path(args.script)
    profile_script(
        args.script,
        out,
        sort=args.sort,
        top=args.top,
        script_args=args.script_args,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
