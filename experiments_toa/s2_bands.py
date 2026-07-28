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

NComponentsSpec = int | Mapping[str, int] | str | Path


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


def load_task_pca_components_config(
    path: str | Path,
    *,
    task_names: Sequence[str] | None = None,
) -> tuple[dict[str, int], dict[str, Any]]:
    """
    Load per-QoI PCA component counts from JSON.

    Returns ``(components_by_task, metadata)`` where metadata retains selection /
    band_mode / thresholds and other non-task fields from the file.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Task PCA components config not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    if not isinstance(payload, dict):
        raise ValueError(f"PCA components config must be a JSON object, got {type(payload)}")

    task_specs = payload.get("tasks", {}) or {}
    if not isinstance(task_specs, dict):
        raise ValueError("'tasks' must be a JSON object mapping task -> n_components")

    names = list(task_names) if task_names is not None else list(S2_TASK_NAMES)
    requested_unknown = [n for n in names if n not in S2_TASK_NAMES]
    if requested_unknown:
        raise ValueError(f"Unknown requested task names: {requested_unknown}")

    unknown = sorted(set(task_specs) - set(S2_TASK_NAMES))
    if unknown:
        raise ValueError(f"Unknown tasks in PCA components config: {unknown}")

    default_raw = payload.get("default", None)
    default_p: int | None = int(default_raw) if default_raw is not None else None
    if default_p is not None and default_p < 1:
        raise ValueError(f"default n_components must be >= 1, got {default_p}")

    out: dict[str, int] = {}
    for name in names:
        if name in task_specs:
            p = int(task_specs[name])
        elif default_p is not None:
            p = default_p
        else:
            raise ValueError(
                f"PCA components config missing task {name!r} and has no 'default'"
            )
        if p < 1:
            raise ValueError(f"n_components for {name!r} must be >= 1, got {p}")
        out[name] = p

    meta = {k: v for k, v in payload.items() if k != "tasks"}
    meta["config_path"] = str(path)
    return out, meta


def resolve_task_pca_components(
    n_components: NComponentsSpec,
    *,
    task_names: Sequence[str] | None = None,
) -> tuple[dict[str, int], dict[str, Any]]:
    """
    Resolve a shared int, per-task mapping, or JSON path to ``{task: p}``.

    Returns ``(components_by_task, metadata)``. Metadata is empty for int/mapping
    specs; for JSON paths it includes fields from the file.
    """
    names = list(task_names) if task_names is not None else list(S2_TASK_NAMES)
    if isinstance(n_components, int):
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        return {name: int(n_components) for name in names}, {
            "selection": "fixed",
            "shared_n_components": int(n_components),
        }

    if isinstance(n_components, Mapping):
        out: dict[str, int] = {}
        for name in names:
            if name not in n_components:
                raise ValueError(f"n_components mapping missing task {name!r}")
            p = int(n_components[name])
            if p < 1:
                raise ValueError(f"n_components for {name!r} must be >= 1, got {p}")
            out[name] = p
        return out, {"selection": "mapping"}

    return load_task_pca_components_config(n_components, task_names=names)


def resolve_task_n_components(
    n_components: NComponentsSpec,
    task_name: str,
    *,
    components_by_task: Mapping[str, int] | None = None,
) -> int:
    """Look up ``p`` for one task (optionally from a pre-resolved mapping)."""
    if components_by_task is not None:
        if task_name not in components_by_task:
            raise ValueError(f"components_by_task missing task {task_name!r}")
        return int(components_by_task[task_name])
    by_task, _ = resolve_task_pca_components(n_components, task_names=[task_name])
    return int(by_task[task_name])
