"""Backward-compat shim; import from ``experiments_toa.data`` instead."""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from experiments_toa.data import *  # noqa: F403
