"""Quick comparison of ASD field validation vs synthetic snow-TOA training data."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASD = _ROOT / "experiments_toa" / "data 11 QoI" / "asd_validation_set.nc"
DEFAULT_SIM = _ROOT / "experiments_toa" / "data 11 QoI" / "snow_toa_fsnow_90to100_20263108.nc"

QOI_PAIRS = [
    ("grain_radius_mean", "grain_size", "grain size (µm)"),
    ("cos_i", "cos_i", "cos incidence"),
    ("dust_conc_mean", "dust", "dust conc."),
    ("algae_conc_mean", "algae", "algae conc."),
    ("cwv", "cwv", "column water vapor"),
    ("lwc_mean", "liquid_water", "liquid water"),
    ("aod", "aot", "aerosol optical depth"),
]


def _pct(x: np.ndarray, ps=(1, 5, 50, 95, 99)) -> np.ndarray:
    return np.percentile(np.asarray(x, float), ps)


def compare(asd_path: Path, sim_path: Path, *, sim_subsample: int = 8000) -> None:
    with h5py.File(asd_path, "r") as fa, h5py.File(sim_path, "r") as fs:
        print(f"ASD: {asd_path.name}  (n={fa['toa_radiance'].shape[0]})")
        print(f"Sim: {sim_path.name}  (n={fs['toa_radiance'].shape[0]})")
        if "sampling" in fs.attrs:
            print(f"Sim sampling note: {fs.attrs['sampling']}")

        print("\n=== QoI composition (glaring gaps highlighted) ===")
        header = f"{'QoI':18s} {'ASD [min,max]':24s} {'Sim [min,max]':24s} {'Sim median':12s} {'ASD in sim range':16s}"
        print(header)
        print("-" * len(header))
        flags: list[str] = []
        for akey, skey, label in QOI_PAIRS:
            a = np.asarray(fa[akey][:], float)
            s = np.asarray(fs[skey][:], float)
            in_range = float(np.mean((a >= s.min()) & (a <= s.max())) * 100.0)
            line = (
                f"{label:18s} [{a.min():8.3g},{a.max():8.3g}]  "
                f"[{s.min():8.3g},{s.max():8.3g}]  "
                f"{np.median(s):10.3g}  {in_range:14.1f}%"
            )
            print(line)
            # Heuristic flags
            if np.median(a) < np.percentile(s, 5) or np.median(a) > np.percentile(s, 95):
                flags.append(f"  ! {label}: ASD median ({np.median(a):.3g}) outside sim 5–95% band [{np.percentile(s,5):.3g}, {np.percentile(s,95):.3g}]")
            if akey == "algae_conc_mean" and np.max(a) < np.percentile(s, 1):
                flags.append(
                    f"  !! ALGAE: ASD values ~{a.mean():.1f} vs sim median {np.median(s):.0f} "
                    f"(orders of magnitude below training distribution)"
                )
            if akey == "dust_conc_mean" and np.any(a < np.percentile(s, 10)):
                flags.append(f"  ! DUST: some ASD samples below sim 10th pct ({np.percentile(s,10):.1g})")

        print("\n=== Synthetic-only mixture state (ASD has no labels) ===")
        for k in ("fsnow", "fPV", "fNPV", "fsoil"):
            x = np.asarray(fs[k][:], float)
            print(f"  {k:6s}: [{x.min():.4g}, {x.max():.4g}]  mean={x.mean():.4g}  (training: 90–100% snow subset)")

        print("\n=== Geometry / viewing (model inputs) ===")
        sza = np.asarray(fa["sza"][:], float)
        coszen_asd = np.cos(np.radians(sza))
        coszen_sim = float(np.asarray(fs["coszen"][0], float))
        cos_i_a = np.asarray(fa["cos_i"][:], float)
        cos_i_s = np.asarray(fs["cos_i"][:], float)
        ele_a = float(np.asarray(fa["elevation_km"][0], float))
        ele_s = np.asarray(fs["ele_km"][:], float)
        raa_a = np.asarray(fa["RAA"][:], float)
        raa_s = np.asarray(fs["RAA_TRUE"][:], float)

        print(f"  coszen (aux):  ASD {coszen_asd.min():.3f}–{coszen_asd.max():.3f} (varies)  |  Sim FIXED at {coszen_sim:.4f}")
        print(f"  cos_i (label): ASD {cos_i_a.min():.3f}–{cos_i_a.max():.3f}  |  Sim {cos_i_s.min():.3f}–{cos_i_s.max():.3f}")
        print(f"  ele_km:        ASD fixed {ele_a:.1f} km  |  Sim {ele_s.min():.2f}–{ele_s.max():.2f} km (mean {ele_s.mean():.2f})")
        print(f"  RAA:           ASD all {raa_a[0]:.0f}°  |  Sim {raa_s.min():.0f}–{raa_s.max():.0f}° (mean {raa_s.mean():.0f})")
        if np.all(raa_a == 0) and raa_s.min() >= 110:
            flags.append("  !! RAA: ASD is 0° for all pixels; training spans 110–180°")
        if np.allclose(coszen_sim, coszen_sim) and np.std(coszen_asd) > 0.05:
            flags.append(
                f"  !! coszen aux: training uses fixed LUT coszen={coszen_sim:.4f}; "
                f"ASD spans {coszen_asd.min():.3f}–{coszen_asd.max():.3f}"
            )

        print("\n=== Algae / dust detail ===")
        alg_s, dust_s = np.asarray(fs["algae"][:], float), np.asarray(fs["dust"][:], float)
        alg_a = np.asarray(fa["algae_conc_mean"][:], float)
        dust_a = np.asarray(fa["dust_conc_mean"][:], float)
        print(f"  algae sim pct [1,5,50,95,99]: {_pct(alg_s)}")
        print(f"  algae ASD: {np.round(alg_a, 2)}")
        print(f"  dust  sim pct [1,5,50,95,99]: {_pct(dust_s)}")
        print(f"  dust  ASD: {np.round(dust_a, 1)}")
        n_alg_floor = int(np.sum(alg_s <= 0.02))
        print(f"  sim algae forced to ~0.01 when grain<400: {n_alg_floor}/{len(alg_s)} ({100*n_alg_floor/len(alg_s):.1f}%)")

        print("\n=== LWC / grain size extremes ===")
        lwc_a = np.asarray(fa["lwc_mean"][:], float)
        lwc_s = np.asarray(fs["liquid_water"][:], float)
        grain_a = np.asarray(fa["grain_radius_mean"][:], float)
        grain_s = np.asarray(fs["grain_size"][:], float)
        print(f"  LWC ASD: {np.round(lwc_a, 2)}  |  sim pct [5,50,95]: {_pct(lwc_s, (5,50,95))}")
        print(f"  grain ASD: {np.round(grain_a, 0)}  |  sim pct [5,50,95]: {_pct(grain_s, (5,50,95))}")
        if np.min(lwc_a) < np.percentile(lwc_s, 1):
            flags.append(f"  ! LWC: ASD min {lwc_a.min():.3g} below most sim draws (sim 1st pct={np.percentile(lwc_s,1):.3g})")

        print("\n=== Spectral radiance ===")
        rad_a = np.asarray(fa["toa_radiance"][:], float)
        rng = np.random.default_rng(0)
        n = min(sim_subsample, fs["toa_radiance"].shape[0])
        idx = rng.choice(fs["toa_radiance"].shape[0], n, replace=False)
        rad_s = np.asarray(fs["toa_radiance"][idx], float)
        wl = np.asarray(fa["wl"][:], float)
        mean_a = rad_a.mean(axis=0)
        mean_s = rad_s.mean(axis=0)
        diff = mean_a - mean_s
        ib = int(np.argmax(np.abs(diff)))
        print(f"  band-mean radiance: ASD avg={mean_a.mean():.2f}  sim={mean_s.mean():.2f}")
        print(f"  largest mean spectrum shift: {diff[ib]:+.2f} at {wl[ib]:.0f} nm")
        print(f"  per-scene mean radiance ASD: {np.round(rad_a.mean(axis=1), 2)}")
        print(f"  sim per-pixel mean rad pct [5,50,95]: {_pct(rad_s.mean(axis=1), (5,50,95))}")

        print("\n=== ASD retrieval meta (field labels are aggregated retrievals) ===")
        print(f"  sample_count per scene: {np.asarray(fa['sample_count'][:])}")
        print(f"  rmse_mean (fit quality): {np.round(np.asarray(fa['rmse_mean'][:]), 4)}")
        print(f"  dates: {[d.decode() if isinstance(d, bytes) else str(d) for d in fa['date'][:]]}")

        if flags:
            print("\n=== FLAGGED DIFFERENCES ===")
            for f in flags:
                print(f)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asd-path", type=Path, default=DEFAULT_ASD)
    p.add_argument("--sim-path", type=Path, default=DEFAULT_SIM)
    args = p.parse_args()
    compare(args.asd_path, args.sim_path)


if __name__ == "__main__":
    main()
