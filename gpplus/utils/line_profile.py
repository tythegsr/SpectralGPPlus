"""No-op ``@profile`` unless ``kernprof -l`` injected line_profiler into builtins."""

from __future__ import annotations

import builtins
from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable)

_builtin_profile = getattr(builtins, "profile", None)

if _builtin_profile is not None:
    profile = _builtin_profile
else:

    def profile(func: F) -> F:
        return func


__all__ = ["profile"]
