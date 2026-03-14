from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

INTERFACE_ROOT = Path(__file__).resolve().parent


def import_matlab_modules() -> tuple[Any, Any]:
    try:
        import matlab
        import matlab.engine
    except ImportError as exc:
        raise ImportError(
            "MATLAB Engine API for Python is required. Install the `matlabengine` "
            "package into this environment and ensure MATLAB is installed."
        ) from exc

    return matlab, matlab.engine


def add_matlab_paths(engine: Any, paths: Iterable[Path | str]) -> None:
    normalized_paths = {str(INTERFACE_ROOT.resolve())}
    normalized_paths.update(str(Path(path).resolve()) for path in paths)

    for matlab_path in sorted(normalized_paths):
        engine.addpath(matlab_path, nargout=0)


def start_matlab_engine(*, extra_paths: Iterable[Path | str] = ()) -> Any:
    _, matlab_engine = import_matlab_modules()
    engine = matlab_engine.start_matlab()
    add_matlab_paths(engine, extra_paths)
    return engine


@contextmanager
def matlab_engine_session(
    engine: Any | None = None,
    *,
    extra_paths: Iterable[Path | str] = (),
) -> Iterator[Any]:
    created_here = engine is None
    if created_here:
        engine = start_matlab_engine(extra_paths=extra_paths)
    else:
        add_matlab_paths(engine, extra_paths)

    try:
        yield engine
    finally:
        if created_here and engine is not None:
            engine.quit()


def to_matlab_row_vector(values: Sequence[float | int]) -> Any:
    matlab, _ = import_matlab_modules()
    return matlab.double([float(value) for value in values])


def to_numpy_1d(values: Any) -> NDArray[np.float64]:
    return np.asarray(values, dtype=float).reshape(-1)


def matlab_struct_get(structure: Any, field: str, default: Any = None) -> Any:
    if isinstance(structure, dict):
        return structure.get(field, default)

    if hasattr(structure, field):
        return getattr(structure, field)

    return default
