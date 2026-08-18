"""Merge split_files/emit_data_chunk_*.nc into a single NetCDF."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import numpy as np

# ISOFIT snow / multi-surface state vector (15-D), confirmed order:
#   sinA, cosA, grain_radius, liquid_water, dust, algae,
#   z_snow, z_pv, z_npv, z_soil, veg_rank, npv_rank, soil_rank,
#   AOT660, H20STR
# sinA/cosA are aspect free-params (independent; sin^2+cos^2 is often ~2 at
# defaults), NOT cosine of solar incidence. cos_i is not present in this vector.
EMIT_STATE_FEATURE_NAMES: tuple[str, ...] = (
    "sinA",
    "cosA",
    "grain_radius",
    "liquid_water",
    "dust",
    "algae",
    "z_snow",
    "z_pv",
    "z_npv",
    "z_soil",
    "veg_rank",
    "npv_rank",
    "soil_rank",
    "AOT660",
    "H20STR",
)

_CHUNK_RE = re.compile(r"emit_data_chunk_(\d+)\.nc$", re.IGNORECASE)


def _chunk_index(path: Path) -> int:
    m = _CHUNK_RE.search(path.name)
    if not m:
        raise ValueError(f"Unexpected chunk name: {path.name}")
    return int(m.group(1))


def find_chunk_paths(split_dir: Path) -> list[Path]:
    paths = sorted(split_dir.glob("emit_data_chunk_*.nc"), key=_chunk_index)
    if not paths:
        raise FileNotFoundError(f"No emit_data_chunk_*.nc under {split_dir}")
    return paths


def merge_emit_chunks(
    split_dir: Path,
    output_path: Path,
    *,
    compression: str | None = None,
    overwrite: bool = False,
) -> Path:
    chunks = find_chunk_paths(split_dir)
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_per_chunk: list[int] = []
    for path in chunks:
        with h5py.File(path, "r") as f:
            n_per_chunk.append(int(f["reflectance"].shape[0]))
            n_bands = int(f["reflectance"].shape[1])
            n_state = int(f["state"].shape[1])
            ref_dtype = f["reflectance"].dtype
            state_dtype = f["state"].dtype
            sample_dtype = f["sample"].dtype
            reflectance_feature = np.asarray(f["reflectance_feature"][:])
            state_feature = np.asarray(f["state_feature"][:])
            root_attrs = {
                k: f.attrs[k]
                for k in f.attrs
                if k != "_NCProperties"
            }

    n_total = int(sum(n_per_chunk))
    if n_state != len(EMIT_STATE_FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(EMIT_STATE_FEATURE_NAMES)} state features, got {n_state}"
        )

    print(f"Merging {len(chunks)} chunks -> {output_path}")
    print(f"  n_total={n_total:,}  bands={n_bands}  state_dim={n_state}")

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    create_kw = {}
    if compression:
        create_kw["compression"] = compression

    with h5py.File(tmp_path, "w") as out:
        for k, v in root_attrs.items():
            out.attrs[k] = v
        out.attrs["description"] = np.bytes_(
            b"Merged cleaned reflectance and state arrays for ML training"
        )
        out.attrs["source_chunks"] = np.bytes_(
            ",".join(p.name for p in chunks).encode("utf-8")
        )
        out.attrs["state_feature_names"] = np.bytes_(
            ",".join(EMIT_STATE_FEATURE_NAMES).encode("utf-8")
        )

        ds_ref = out.create_dataset(
            "reflectance",
            shape=(n_total, n_bands),
            dtype=ref_dtype,
            chunks=(min(8192, n_total), n_bands),
            **create_kw,
        )
        ds_state = out.create_dataset(
            "state",
            shape=(n_total, n_state),
            dtype=state_dtype,
            chunks=(min(8192, n_total), n_state),
            **create_kw,
        )
        ds_sample = out.create_dataset(
            "sample",
            shape=(n_total,),
            dtype=sample_dtype,
            chunks=(min(65536, n_total),),
            **create_kw,
        )
        out.create_dataset("reflectance_feature", data=reflectance_feature)
        out.create_dataset("state_feature", data=state_feature)
        # String labels for state columns (UTF-8 variable-length).
        dt = h5py.string_dtype(encoding="utf-8")
        out.create_dataset(
            "state_feature_name",
            data=np.array(EMIT_STATE_FEATURE_NAMES, dtype=object),
            dtype=dt,
        )

        offset = 0
        for path, n in zip(chunks, n_per_chunk):
            print(f"  copying {path.name} ({n:,} rows) @ offset {offset:,}")
            with h5py.File(path, "r") as src:
                ds_ref[offset : offset + n] = src["reflectance"][:]
                ds_state[offset : offset + n] = src["state"][:]
                ds_sample[offset : offset + n] = src["sample"][:]
            offset += n

        if offset != n_total:
            raise RuntimeError(f"Row count mismatch: wrote {offset}, expected {n_total}")

    if output_path.exists():
        output_path.unlink()
    tmp_path.replace(output_path)
    print(f"Wrote {output_path} ({output_path.stat().st_size / 1e9:.2f} GB)")
    return output_path.resolve()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=root / "split_files",
        help="Directory containing emit_data_chunk_*.nc",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "split_files" / "emit_data.nc",
        help="Merged NetCDF path",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output if it already exists",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default=None,
        help="Optional h5py compression (e.g. gzip); default uncompressed for speed",
    )
    args = parser.parse_args()
    merge_emit_chunks(
        args.split_dir,
        args.output,
        compression=args.compression,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
