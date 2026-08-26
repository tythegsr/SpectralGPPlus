"""Map an EMIT pixel NetCDF onto the S2 11-QoI analysis schema.

EMIT files store ``reflectance`` + 15-D ISOFIT ``state`` (no ``wl``, no
``toa_radiance``). Processed files with ``obs`` also get ``cos_i`` from
``calc_new_angles_cosi`` (sinA/cosA + SZA/SAA/slope). This writes a sidecar with:

- ``toa_reflectance`` copied from ``reflectance``
- ``wl`` copied from a synthetic TOA file (same 285-band grid)
- mapped QoIs from state (+ ``cos_i`` when obs geometry is available)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from experiments_toa.emit_geometry import calc_cos_i_from_state_obs
from experiments_toa.merge_emit_chunks import EMIT_STATE_FEATURE_NAMES

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMIT = _ROOT / "split_files" / "emit_data_snow_70to100.nc"
DEFAULT_WL_SRC = (
    _ROOT / "experiments_toa" / "data 11 QoI" / "snow_toa_fsnow_70to100_20261208.nc"
)
DEFAULT_OUT = (
    _ROOT
    / "experiments_toa"
    / "reports"
    / "emit_snow70_s2_analysis"
    / "emit_snow70_s2like.nc"
)

CHUNK = 8192
EMIT_STATE_INDEX = {name: i for i, name in enumerate(EMIT_STATE_FEATURE_NAMES)}
IDX_Z = tuple(EMIT_STATE_INDEX[n] for n in ("z_snow", "z_pv", "z_npv", "z_soil"))
MAPPED_QOI = (
    "algae",
    "aot",
    "cosA",
    "cwv",
    "dust",
    "fNPV",
    "fPV",
    "fsnow",
    "fsoil",
    "grain_size",
    "liquid_water",
    "sinA",
)


def softmax_rows(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.clip(e.sum(axis=1, keepdims=True), 1e-30, None)


def mapped_qois(state: np.ndarray) -> dict[str, np.ndarray]:
    state = np.asarray(state, dtype=np.float64)
    frac = softmax_rows(state[:, list(IDX_Z)])
    return {
        "grain_size": state[:, EMIT_STATE_INDEX["grain_radius"]],
        "liquid_water": state[:, EMIT_STATE_INDEX["liquid_water"]],
        "dust": state[:, EMIT_STATE_INDEX["dust"]],
        "algae": state[:, EMIT_STATE_INDEX["algae"]],
        "aot": state[:, EMIT_STATE_INDEX["AOT660"]],
        "cwv": state[:, EMIT_STATE_INDEX["H20STR"]],
        "fsnow": frac[:, 0],
        "fPV": frac[:, 1],
        "fNPV": frac[:, 2],
        "fsoil": frac[:, 3],
        "sinA": state[:, EMIT_STATE_INDEX["sinA"]],
        "cosA": state[:, EMIT_STATE_INDEX["cosA"]],
    }


def export_emit_as_s2(
    emit_path: Path,
    out_path: Path,
    *,
    wl_src: Path,
    overwrite: bool = False,
) -> Path:
    emit_path = Path(emit_path)
    out_path = Path(out_path)
    wl_src = Path(wl_src)
    if not emit_path.is_file():
        raise FileNotFoundError(f"EMIT NetCDF not found: {emit_path}")
    if not wl_src.is_file():
        raise FileNotFoundError(f"Wavelength source NetCDF not found: {wl_src}")
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(wl_src, "r") as f_wl:
        if "wl" not in f_wl:
            raise KeyError(f"Missing 'wl' in {wl_src}")
        wl = np.asarray(f_wl["wl"][:], dtype=np.float64)

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    with h5py.File(emit_path, "r") as src, h5py.File(tmp_path, "w") as out:
        n, n_bands = src["reflectance"].shape
        if n_bands != wl.shape[0]:
            raise ValueError(
                f"EMIT n_bands={n_bands} does not match wl length {wl.shape[0]}"
            )
        n_state = int(src["state"].shape[1])
        if n_state != len(EMIT_STATE_FEATURE_NAMES):
            raise ValueError(
                f"Expected {len(EMIT_STATE_FEATURE_NAMES)} state features, got {n_state}"
            )

        state = np.asarray(src["state"][:], dtype=np.float64)
        qois = mapped_qois(state)

        has_radiance = "radiance" in src
        has_elevation = "elevation" in src
        has_obs = "obs" in src
        has_cos_i = False
        if has_obs:
            obs = np.asarray(src["obs"][:], dtype=np.float64)
            qois["cos_i"] = calc_cos_i_from_state_obs(state, obs)
            has_cos_i = True
        del state

        missing_vars = []
        if not has_radiance:
            missing_vars.append("toa_radiance")
        if not has_cos_i:
            missing_vars.append("cos_i")
        out.attrs["description"] = np.bytes_(
            b"EMIT pixels mapped to S2 analysis schema "
            b"(reflectance [+ radiance] + elevation input + QoIs)"
        )
        out.attrs["source"] = np.bytes_(str(emit_path.resolve()).encode("utf-8"))
        out.attrs["wl_source"] = np.bytes_(str(wl_src.resolve()).encode("utf-8"))
        out.attrs["n_samples"] = np.int64(n)
        if has_cos_i:
            out.attrs["cos_i_source"] = np.bytes_(
                b"calc_new_angles_cosi(sinA,cosA,SZA,SAA,slope)"
            )
        else:
            out.attrs["missing_qoi"] = np.bytes_(b"cos_i")
        out.attrs["missing_variables"] = np.bytes_(
            ",".join(missing_vars).encode("utf-8")
        )
        out.attrs["cover_mapping"] = np.bytes_(
            b"fsnow,fPV,fNPV,fsoil = softmax(z_snow,z_pv,z_npv,z_soil)"
        )
        out.attrs["aux_inputs"] = np.bytes_(
            b"elevation" if has_elevation else b""
        )
        out.attrs["output_log_scale"] = np.bytes_(b"false")

        out.create_dataset("wl", data=wl)
        ds_ref = out.create_dataset(
            "toa_reflectance",
            shape=(n, n_bands),
            dtype=src["reflectance"].dtype,
            chunks=(min(CHUNK, n), n_bands),
        )
        ds_rad = None
        if has_radiance:
            if src["radiance"].shape != src["reflectance"].shape:
                raise ValueError(
                    f"radiance shape {src['radiance'].shape} != "
                    f"reflectance shape {src['reflectance'].shape}"
                )
            ds_rad = out.create_dataset(
                "toa_radiance",
                shape=(n, n_bands),
                dtype=src["radiance"].dtype,
                chunks=(min(CHUNK, n), n_bands),
            )
        if has_elevation:
            elev = np.asarray(src["elevation"][:], dtype=np.float64).reshape(n)
            out.create_dataset("elevation", data=elev.astype(np.float32, copy=False))
        for name in MAPPED_QOI:
            out.create_dataset(name, data=np.asarray(qois[name], dtype=np.float64))
        if has_cos_i:
            out.create_dataset("cos_i", data=np.asarray(qois["cos_i"], dtype=np.float64))

        src_ref = src["reflectance"]
        src_rad = src["radiance"] if has_radiance else None
        for start in range(0, n, CHUNK):
            stop = min(start + CHUNK, n)
            ds_ref[start:stop] = src_ref[start:stop]
            if ds_rad is not None and src_rad is not None:
                ds_rad[start:stop] = src_rad[start:stop]
            if start == 0 or stop == n or (start // CHUNK) % 20 == 0:
                parts = ["reflectance"]
                if has_radiance:
                    parts.append("radiance")
                if has_elevation:
                    parts.append("elevation")
                print(f"  copied {'+'.join(parts)} {stop}/{n}", flush=True)

    tmp_path.replace(out_path)
    print(f"Wrote S2-schema NetCDF -> {out_path}  n={n} bands={n_bands}")
    return out_path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--emit-path", type=Path, default=DEFAULT_EMIT)
    p.add_argument("--out-path", type=Path, default=DEFAULT_OUT)
    p.add_argument("--wl-src", type=Path, default=DEFAULT_WL_SRC)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    export_emit_as_s2(
        args.emit_path,
        args.out_path,
        wl_src=args.wl_src,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
