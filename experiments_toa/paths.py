"""Import path helpers for TOA experiment scripts."""

from __future__ import annotations

import sys
from pathlib import Path


def repo_root() -> Path:
    """Repository root (parent of ``experiments_toa``)."""
    return Path(__file__).resolve().parents[1]


def ensure_repo_on_path() -> Path:
    """Ensure repo root is on ``sys.path`` so ``experiments_toa`` can be imported."""
    root = repo_root()
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    return root


def pin_toa_import_paths(*extra_dirs: str | Path) -> None:
    """
    Pin ``sys.path`` order for TOA runners.

    ``extra_dirs`` are listed first (highest priority), then the repo root.
    Later entries in ``extra_dirs`` have higher priority than earlier ones.
    """
    ensure_repo_on_path()
    ordered = [str(d) for d in extra_dirs] + [str(repo_root())]
    seen: set[str] = set()
    front: list[str] = []
    for path in ordered:
        if path not in seen:
            seen.add(path)
            front.append(path)
    sys.path[:] = front + [p for p in sys.path if p not in seen]
