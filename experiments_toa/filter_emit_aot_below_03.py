"""Filter processed EMIT NetCDF to pixels with AOT660 <= 0.03.

Default source is emit_test_data_processed_90to100_20262508.nc (already
fsnow >= 0.90). Keeps the full processed schema.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from experiments_toa.merge_emit_chunks import EMIT_STATE_FEATURE_NAMES

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DATA_DIR = _ROOT / "experiments_toa" / "data 11 QoI"
DEFAULT_SRC = _DEFAULT_DATA_DIR / "emit_test_data_processed_90to100_20262508.nc"
DEFAULT_OUT = _DEFAULT_DATA_DIR / "emit_test_data_90to100_aotbelow03_20262608.nc"

CHUNK = 8192
DEFAULT_AOT_HI = 0.03
AOT_COL = EMIT_STATE_FEATURE_NAMES.index("AOT660")
ROW_DATASETS = (
    "radiance",
    "reflectance",
    "elevation",
    "obs",
    "state",
)
STATIC_DATASETS = (
    "sample",
    "radiance_feature",
    "reflectance_feature",
    "elevation_feature",
    "obs_feature",
    "state_feature",
)


def keep_mask(
    state: np.ndarray, *, aot_hi: float = DEFAULT_AOT_HI
) -> tuple[np.ndarray, dict[str, float]]:
    aot = np.asarray(state[:, AOT_COL], dtype=np.float64)
    keep = aot <= aot_hi
    counts = {
        "n_source": int(state.shape[0]),
        "n_kept": int(keep.sum()),
        "n_drop": int((~keep).sum()),
        "aot_min": float(aot.min()),
        "aot_max": float(aot.max()),
        "aot_mean": float(aot.mean()),
        "kept_aot_min": float(aot[keep].min()) if keep.any() else float("nan"),
        "kept_aot_max": float(aot[keep].max()) if keep.any() else float("nan"),
        "kept_aot_mean": float(aot[keep].mean()) if keep.any() else float("nan"),
    }
    return keep, counts


def _copy_row_dataset(
    src,
    out,
    name: str,
    keep: np.ndarray,
    n_kept: int,
    *,
    create_kw: dict,
) -> None:
    src_ds = src[name]
    shape = (n_kept,) + tuple(int(s) for s in src_ds.shape[1:])
    chunks = (min(CHUNK, n_kept),) + src_ds.chunks[1:] if src_ds.chunks else None
    if chunks is None:
        chunks = (min(CHUNK, n_kept),) + shape[1:]
    ds_out = out.create_dataset(
        name,
        shape=shape,
        dtype=src_ds.dtype,
        chunks=chunks,
        **create_kw,
    )
    n_src = int(keep.size)
    offset = 0
    for start in range(0, n_src, CHUNK):
        stop = min(start + CHUNK, n_src)
        m = keep[start:stop]
        n_chunk = int(m.sum())
        if n_chunk == 0:
            continue
        ds_out[offset : offset + n_chunk] = src_ds[start:stop][m]
        offset += n_chunk
    if offset != n_kept:
        raise RuntimeError(
            f"{name}: row count mismatch wrote {offset}, expected {n_kept}"
        )


def filter_aot_file(
    src_path: Path,
    output_path: Path,
    *,
    aot_hi: float = DEFAULT_AOT_HI,
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
    filter_rules = f"keep AOT660 <= {aot_hi:g} (drop AOT660 > {aot_hi:g})"

    with h5py.File(src_path, "r") as src:
        missing = [n for n in ROW_DATASETS if n not in src]
        if missing:
            raise KeyError(f"Missing required datasets in {src_path}: {missing}")
        state = np.asarray(src["state"][:])
        keep, counts = keep_mask(state, aot_hi=aot_hi)
        n_kept = int(counts["n_kept"])
        n_state = int(state.shape[1])
        if n_state != len(EMIT_STATE_FEATURE_NAMES):
            raise ValueError(
                f"Expected {len(EMIT_STATE_FEATURE_NAMES)} state features, got {n_state}"
            )
        print(f"Source {src_path}")
        print(
            f"  n_source={counts['n_source']:,}  "
            f"drop={counts['n_drop']:,}  kept={n_kept:,}  "
            f"aot source [{counts['aot_min']:.4g}, {counts['aot_max']:.4g}] "
            f"mean={counts['aot_mean']:.4g}  "
            f"kept aot [{counts['kept_aot_min']:.4g}, {counts['kept_aot_max']:.4g}] "
            f"mean={counts['kept_aot_mean']:.4g}"
        )
        if n_kept == 0:
            raise RuntimeError("Filter kept 0 rows")

        root_attrs = {k: src.attrs[k] for k in src.attrs if k != "_NCProperties"}
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
                f"Processed EMIT snow90 pixels with AOT660 <= {aot_hi:g}".encode("utf-8")
            )
            out.attrs["source"] = np.bytes_(str(src_path.resolve()).encode("utf-8"))
            out.attrs["n_source"] = np.int64(counts["n_source"])
            out.attrs["n_kept"] = np.int64(n_kept)
            out.attrs["aot_hi"] = np.float64(aot_hi)
            out.attrs["kept_aot_min"] = np.float64(counts["kept_aot_min"])
            out.attrs["kept_aot_max"] = np.float64(counts["kept_aot_max"])
            out.attrs["kept_aot_mean"] = np.float64(counts["kept_aot_mean"])
            out.attrs["filter_rules"] = np.bytes_(filter_rules.encode("utf-8"))
            out.attrs["state_feature_names"] = np.bytes_(
                ",".join(EMIT_STATE_FEATURE_NAMES).encode("utf-8")
            )

            for name in STATIC_DATASETS:
                if name not in src:
                    continue
                src_ds = src[name]
                if src_ds.ndim == 0 or src_ds.shape[0] != counts["n_source"]:
                    out.create_dataset(name, data=src_ds[()])
                else:
                    out.create_dataset(name, data=src_ds[keep])

            for name in ROW_DATASETS:
                print(f"  copying {name}...")
                _copy_row_dataset(
                    src, out, name, keep, n_kept, create_kw=create_kw
                )

            if "state_feature_name" in src:
                out.create_dataset(
                    "state_feature_name",
                    data=src["state_feature_name"][()],
                )
            if "obs_feature_name" in src:
                out.create_dataset(
                    "obs_feature_name",
                    data=src["obs_feature_name"][()],
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
    parser.add_argument(
        "--aot-hi",
        type=float,
        default=DEFAULT_AOT_HI,
        help="Keep pixels with AOT660 <= this value (default 0.03)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--compression",
        type=str,
        default=None,
        help="Optional h5py compression (e.g. gzip); default uncompressed for speed",
    )
    args = parser.parse_args()
    filter_aot_file(
        args.src,
        args.output,
        aot_hi=args.aot_hi,
        overwrite=args.overwrite,
        compression=args.compression,
    )


if __name__ == "__main__":
    main()
