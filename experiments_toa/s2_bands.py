"""Parse and validate per-QoI original-band subsets for S2 TOA runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments_toa.s2_constants import (
    S2_DEFAULT_BAND_CONFIG_PATH,
    S2_DEFAULT_KEEP_RANGES,
    S2_INPUT_DIM,
    S2_TASK_NAMES,
)


def expand_band_spec(
    spec: Sequence[int | Sequence[int]] | Mapping[str, Any] | None,
    *,
    input_dim: int = S2_INPUT_DIM,
) -> list[int]:
    """
    Expand a band specification into an ordered list of unique indices.

    Accepted forms:
    - ``[0, 1, 2]`` flat list of indices
    - ``[[0, 130], [143, 190]]`` inclusive ranges
    - mixed ``[0, 1, [10, 12]]``
    - ``{"ranges": [[0, 130]], "indices": [200]}``
    - ``None`` -> default keep ranges
    """
    if spec is None:
        specs: list[Any] = [list(r) for r in S2_DEFAULT_KEEP_RANGES]
    elif isinstance(spec, Mapping):
        specs = []
        if "ranges" in spec:
            specs.extend(spec["ranges"])
        if "indices" in spec:
            specs.extend(spec["indices"])
        if not specs and "bands" in spec:
            specs.extend(spec["bands"])
        if not specs:
            raise ValueError(f"Empty band mapping: {spec!r}")
    else:
        specs = list(spec)

    out: list[int] = []
    seen: set[int] = set()
    for item in specs:
        if isinstance(item, (list, tuple)):
            if len(item) != 2:
                raise ValueError(f"Range must be [lo, hi], got {item!r}")
            lo, hi = int(item[0]), int(item[1])
            if lo > hi:
                raise ValueError(f"Range lo>hi: {item!r}")
            chunk = list(range(lo, hi + 1))
        else:
            chunk = [int(item)]
        for idx in chunk:
            if idx < 0 or idx >= input_dim:
                raise ValueError(f"Band index {idx} out of range for input_dim={input_dim}")
            if idx in seen:
                continue
            seen.add(idx)
            out.append(idx)
    if not out:
        raise ValueError("Band specification produced an empty selection.")
    return out


def default_keep_indices(*, input_dim: int = S2_INPUT_DIM) -> list[int]:
    return expand_band_spec(None, input_dim=input_dim)


def load_task_band_config(
    path: str | Path | None = None,
    *,
    task_names: Sequence[str] | None = None,
    input_dim: int = S2_INPUT_DIM,
) -> dict[str, list[int]]:
    """
    Load JSON band config and return ``{task_name: [band_indices...]}``.

    JSON schema::

        {
          "input_dim": 285,
          "default": [[0, 130], [143, 190], [213, 277]],
          "tasks": {
            "cos_i": [[0, 130], [143, 190]],
            "aot": {"ranges": [[0, 100]], "indices": [200]}
          }
        }
    """
    if path is None:
        path = S2_DEFAULT_BAND_CONFIG_PATH
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Task band config not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    cfg_dim = int(payload.get("input_dim", input_dim))
    if cfg_dim != input_dim:
        raise ValueError(
            f"Config input_dim={cfg_dim} does not match expected input_dim={input_dim}"
        )

    names = list(task_names) if task_names is not None else list(S2_TASK_NAMES)
    default_spec = payload.get("default", None)
    default_bands = expand_band_spec(default_spec, input_dim=input_dim)
    task_specs = payload.get("tasks", {}) or {}
    if not isinstance(task_specs, dict):
        raise ValueError("'tasks' must be a JSON object mapping task -> band spec")

    unknown = sorted(set(task_specs) - set(S2_TASK_NAMES))
    if unknown:
        raise ValueError(f"Unknown tasks in band config: {unknown}")
    requested_unknown = [n for n in names if n not in S2_TASK_NAMES]
    if requested_unknown:
        raise ValueError(f"Unknown requested task names: {requested_unknown}")

    out: dict[str, list[int]] = {}
    for name in names:
        if name in task_specs:
            out[name] = expand_band_spec(task_specs[name], input_dim=input_dim)
        else:
            out[name] = list(default_bands)
    return out


def normalize_task_input_band_indices(
    mapping: Mapping[str, Sequence[int | Sequence[int]]] | None,
    *,
    task_names: Sequence[str],
    input_dim: int = S2_INPUT_DIM,
    fallback: Sequence[int] | None = None,
) -> dict[str, list[int]]:
    """Validate/normalize an in-memory per-task band mapping."""
    names = list(task_names)
    fallback_bands = (
        list(fallback) if fallback is not None else default_keep_indices(input_dim=input_dim)
    )
    if mapping is None:
        return {name: list(fallback_bands) for name in names}

    unknown = sorted(set(mapping) - set(names))
    if unknown:
        raise ValueError(f"Unknown tasks in band mapping: {unknown}")

    out: dict[str, list[int]] = {}
    for name in names:
        if name in mapping:
            out[name] = expand_band_spec(mapping[name], input_dim=input_dim)
        else:
            out[name] = list(fallback_bands)
    return out


def band_config_metadata(
    bands_by_task: Mapping[str, Sequence[int]],
    *,
    wavelengths_nm: Sequence[float] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build JSON-serializable band metadata for metrics/checkpoints."""
    meta: dict[str, dict[str, Any]] = {}
    for name, bands in bands_by_task.items():
        idxs = [int(i) for i in bands]
        entry: dict[str, Any] = {
            "indices": idxs,
            "n_bands": len(idxs),
        }
        if wavelengths_nm is not None:
            entry["wavelength_nm"] = [float(wavelengths_nm[i]) for i in idxs]
        meta[name] = entry
    return meta
