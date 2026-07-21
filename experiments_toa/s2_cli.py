"""Shared CLI helpers for S2 TOA entry scripts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from experiments_toa.s2_constants import (
    S2_DEFAULT_BAND_CONFIG_PATH,
    S2_DEFAULT_DATA_PATH,
    S2_INPUT_VARIABLES,
    S2_TASK_NAMES,
)


def add_s2_common_args(
    parser: argparse.ArgumentParser,
    *,
    default_qoi: Sequence[str] | None = None,
) -> None:
    """Add shared S2 arguments, including an IDE-editable default QoI list."""
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(S2_DEFAULT_DATA_PATH),
        help="Path to S2 NetCDF (default: experiments_toa/data 11 QoI/...nc)",
    )
    parser.add_argument(
        "--input-variable",
        type=str,
        default="toa_reflectance",
        choices=S2_INPUT_VARIABLES,
        help="Spectral input variable in the NetCDF",
    )
    parser.add_argument(
        "--qoi",
        nargs="+",
        default=list(default_qoi) if default_qoi is not None else None,
        metavar="NAME",
        help=(
            "QoI name(s) to train sequentially, e.g. --qoi cos_i fsnow. "
            f"Script default: {list(default_qoi) if default_qoi is not None else None} "
            f"(None trains all {len(S2_TASK_NAMES)} QoIs)."
        ),
    )
    parser.add_argument(
        "--task-names",
        type=str,
        default=None,
        help=(
            "Legacy comma-separated alias for --qoi. "
            f"Default: None (train all {len(S2_TASK_NAMES)} QoIs)."
        ),
    )
    parser.add_argument(
        "--task-band-config",
        type=str,
        default=str(S2_DEFAULT_BAND_CONFIG_PATH),
        help="JSON mapping each QoI to original band indices/ranges",
    )


def parse_task_names(raw: str | Sequence[str] | None) -> list[str] | None:
    """Normalize a comma- or sequence-based QoI selection; ``None`` means all."""
    if raw is None:
        return None
    values = [raw] if isinstance(raw, str) else list(raw)
    names = [
        name
        for value in values
        for name in (x.strip() for x in str(value).split(","))
        if name
    ]
    unknown = [n for n in names if n not in S2_TASK_NAMES]
    if unknown:
        raise ValueError(f"Unknown QoI names: {unknown}. Valid: {list(S2_TASK_NAMES)}")
    if not names:
        raise ValueError("QoI selection produced an empty list")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate QoI names are not allowed: {duplicates}")
    return names


def selected_task_names(args: argparse.Namespace) -> list[str] | None:
    """Resolve the preferred ``--qoi`` option or legacy ``--task-names`` alias."""
    qoi = getattr(args, "qoi", None)
    legacy = getattr(args, "task_names", None)
    if qoi is not None and legacy is not None:
        raise ValueError("Use either --qoi or --task-names, not both.")
    return parse_task_names(qoi if qoi is not None else legacy)


def parse_example_indices(raw: str | None) -> list[int] | None:
    if raw is None or not str(raw).strip():
        return None
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def ensure_parent_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
