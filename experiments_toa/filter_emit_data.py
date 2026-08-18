"""Filter emit_data.nc into emit_data_filtered.nc.

Drops:
  1. grain_radius == 0
  2. grain_radius == 500 AND liquid_water == 0 AND all z_* == 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from experiments_toa.merge_emit_chunks import EMIT_STATE_FEATURE_NAMES

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = _ROOT / "split_files" / "emit_data.nc"
DEFAULT_OUT = _ROOT / "split_files" / "emit_data_filtered.nc"

ZERO_ATOL = 1e-12
GRAIN_DEFAULT = 500.0
CHUNK = 8192

IDX_GRAIN = EMIT_STATE_FEATURE_NAMES.index("grain_radius")
IDX_LWC = EMIT_STATE_FEATURE_NAMES.index("liquid_water")
IDX_Z = tuple(
    EMIT_STATE_FEATURE_NAMES.index(name)
    for name in ("z_snow", "z_pv", "z_npv", "z_soil")
)

FILTER_RULES = (
    "drop grain_radius==0; drop grain_radius==500 AND liquid_water==0 "
    "AND z_snow=z_pv=z_npv=z_soil==0"
)


def _is_zero(x: np.ndarray) -> np.ndarray:
    return np.abs(x) < ZERO_ATOL


def keep_mask(state: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    grain = state[:, IDX_GRAIN]
    lwc = state[:, IDX_LWC]
    z = state[:, list(IDX_Z)]
    drop_grain0 = _is_zero(grain)
    drop_stuck = (
        (np.abs(grain - GRAIN_DEFAULT) < ZERO_ATOL)
        & _is_zero(lwc)
        & np.all(_is_zero(z), axis=1)
    )
    drop = drop_grain0 | drop_stuck
    keep = ~drop
    counts = {
        "n_source": int(state.shape[0]),
        "n_drop_grain0": int(drop_grain0.sum()),
        "n_drop_stuck": int(drop_stuck.sum()),
        "n_drop_overlap": int((drop_grain0 & drop_stuck).sum()),
        "n_drop": int(drop.sum()),
        "n_kept": int(keep.sum()),
    }
    return keep, counts


def filter_emit_file(
    src_path: Path,
    output_path: Path,
    *,
    overwrite: bool = False,
    compression: str | None = None,
) -> Path:
    src_path = Path(src_path)
    output_path = Path(output_path)
    if not src_path.is_file():
        raise FileNotFoundError(f"EMIT NetCDF not found: {src_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(src_path, "r") as src:
        state = np.asarray(src["state"][:])
        keep, counts = keep_mask(state)
        n_kept = counts["n_kept"]
        n_bands = int(src["reflectance"].shape[1])
        n_state = int(src["state"].shape[1])
        if n_state != len(EMIT_STATE_FEATURE_NAMES):
            raise ValueError(
                f"Expected {len(EMIT_STATE_FEATURE_NAMES)} state features, got {n_state}"
            )
        print(f"Source {src_path}")
        print(
            f"  n_source={counts['n_source']:,}  "
            f"drop_grain0={counts['n_drop_grain0']:,}  "
            f"drop_stuck={counts['n_drop_stuck']:,}  "
            f"overlap={counts['n_drop_overlap']:,}  "
            f"kept={n_kept:,}"
        )
        if n_kept == 0:
            raise RuntimeError("Filter kept 0 rows")

        root_attrs = {k: src.attrs[k] for k in src.attrs if k != "_NCProperties"}
        reflectance_feature = np.asarray(src["reflectance_feature"][:])
        state_feature = np.asarray(src["state_feature"][:])
        ref_dtype = src["reflectance"].dtype
        state_dtype = src["state"].dtype
        sample_dtype = src["sample"].dtype

        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        if tmp_path.exists():
            tmp_path.unlink()

        create_kw: dict = {}
        if compression:
            create_kw["compression"] = compression

        with h5py.File(tmp_path, "w") as out:
            for k, v in root_attrs.items():
                out.attrs[k] = v
            out.attrs["description"] = np.bytes_(
                b"Filtered EMIT reflectance and state (dropped grain=0 and "
                b"stuck grain=500 / liquid_water=0 / z_*=0 rows)"
            )
            out.attrs["source"] = np.bytes_(str(src_path.resolve()).encode("utf-8"))
            out.attrs["n_source"] = np.int64(counts["n_source"])
            out.attrs["n_kept"] = np.int64(n_kept)
            out.attrs["n_drop_grain0"] = np.int64(counts["n_drop_grain0"])
            out.attrs["n_drop_stuck"] = np.int64(counts["n_drop_stuck"])
            out.attrs["filter_rules"] = np.bytes_(FILTER_RULES.encode("utf-8"))
            out.attrs["state_feature_names"] = np.bytes_(
                ",".join(EMIT_STATE_FEATURE_NAMES).encode("utf-8")
            )

            ds_ref = out.create_dataset(
                "reflectance",
                shape=(n_kept, n_bands),
                dtype=ref_dtype,
                chunks=(min(8192, n_kept), n_bands),
                **create_kw,
            )
            ds_state = out.create_dataset(
                "state",
                shape=(n_kept, n_state),
                dtype=state_dtype,
                chunks=(min(8192, n_kept), n_state),
                **create_kw,
            )
            ds_sample = out.create_dataset(
                "sample",
                shape=(n_kept,),
                dtype=sample_dtype,
                chunks=(min(65536, n_kept),),
                **create_kw,
            )
            out.create_dataset("reflectance_feature", data=reflectance_feature)
            out.create_dataset("state_feature", data=state_feature)
            dt = h5py.string_dtype(encoding="utf-8")
            out.create_dataset(
                "state_feature_name",
                data=np.array(EMIT_STATE_FEATURE_NAMES, dtype=object),
                dtype=dt,
            )

            ds_state[:] = state[keep]
            ds_sample[:] = np.asarray(src["sample"][:])[keep]
            del state

            n_src = counts["n_source"]
            offset = 0
            src_ref = src["reflectance"]
            for start in range(0, n_src, CHUNK):
                stop = min(start + CHUNK, n_src)
                m = keep[start:stop]
                n_chunk = int(m.sum())
                if n_chunk == 0:
                    continue
                ds_ref[offset : offset + n_chunk] = src_ref[start:stop][m]
                offset += n_chunk
                if start % (CHUNK * 32) == 0:
                    print(f"  copied {offset:,} / {n_kept:,} kept rows")

            if offset != n_kept:
                raise RuntimeError(
                    f"Row count mismatch: wrote {offset}, expected {n_kept}"
                )

    if output_path.exists():
        output_path.unlink()
    tmp_path.replace(output_path)
    print(f"Wrote {output_path} ({output_path.stat().st_size / 1e9:.2f} GB)")
    return output_path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--compression",
        type=str,
        default=None,
        help="Optional h5py compression (e.g. gzip); default uncompressed for speed",
    )
    args = parser.parse_args()
    filter_emit_file(
        args.src,
        args.output,
        overwrite=args.overwrite,
        compression=args.compression,
    )


if __name__ == "__main__":
    main()
