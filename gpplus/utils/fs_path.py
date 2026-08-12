"""Windows long-path helpers (bypass MAX_PATH / WinError 206)."""

from __future__ import annotations

import os
from pathlib import Path


def fs_path(path: str | Path) -> str:
    """Absolute path string safe for Win32 paths longer than MAX_PATH (260)."""
    text = str(Path(path).expanduser().resolve())
    if os.name == "nt" and not text.startswith("\\\\?\\"):
        return "\\\\?\\" + text
    return text


def ensure_dir(path: str | Path) -> Path:
    """Create directory ``path`` (and parents), tolerating long Windows paths."""
    p = Path(path).expanduser()
    os.makedirs(fs_path(p), exist_ok=True)
    return p.resolve()


def ensure_parent(path: str | Path) -> Path:
    """Create parent directory of ``path``, tolerating long Windows paths."""
    p = Path(path).expanduser()
    ensure_dir(p.parent)
    return p
