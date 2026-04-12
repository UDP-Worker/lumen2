from __future__ import annotations

from typing import Any


def run_filter_optimization(*args: Any, **kwargs: Any) -> Any:
    from .optimize import run_filter_optimization as _run_filter_optimization

    return _run_filter_optimization(*args, **kwargs)


__all__ = ["run_filter_optimization"]
