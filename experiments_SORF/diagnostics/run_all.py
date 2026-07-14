"""Run the full SORF-vs-RFF-vs-ORF diagnostic cascade."""

from __future__ import annotations

import argparse

from phase0_matched_bakeoff import run_phase0
from phase1_kernel_quality import run_phase1
from phase2_opt_and_wswap import run_phase2
from phase3_sweeps import run_phase3
from summarize import summarize


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--num-rff", type=int, default=800)
    p.add_argument("--num-epochs", type=int, default=200)
    p.add_argument("--skip-phase0", action="store_true")
    p.add_argument("--skip-phase1", action="store_true")
    p.add_argument("--skip-phase2", action="store_true")
    p.add_argument("--skip-phase3", action="store_true")
    p.add_argument("--skip-toa-sweep", action="store_true",
                   help="Skip Phase 3 TOA D-sweep (slow)")
    args = p.parse_args()

    if not args.skip_phase1:
        print("\n######## PHASE 1 ########\n")
        run_phase1()

    if not args.skip_phase0:
        print("\n######## PHASE 0 ########\n")
        run_phase0(
            n_train=args.n_train,
            num_rff=args.num_rff,
            num_epochs=args.num_epochs,
            device=args.device,
        )

    if not args.skip_phase2:
        print("\n######## PHASE 2 ########\n")
        run_phase2(device=args.device)

    if not args.skip_phase3:
        print("\n######## PHASE 3 ########\n")
        run_phase3(device=args.device, skip_toa_sweep=args.skip_toa_sweep)

    print("\n######## WRITEUP ########\n")
    summarize()


if __name__ == "__main__":
    main()
