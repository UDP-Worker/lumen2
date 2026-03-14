from __future__ import annotations

import os
import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

INTERFACE_ROOT = Path(__file__).resolve().parent
_MANAGED_ENGINE_PROCESSES: dict[int, subprocess.Popen[Any]] = {}


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
    try:
        engine = matlab_engine.start_matlab()
    except UnicodeDecodeError as exc:
        if os.name != "nt":
            raise
        engine = _start_shared_engine_fallback(matlab_engine, exc)
    add_matlab_paths(engine, extra_paths)
    return engine


def close_matlab_engine(engine: Any) -> None:
    managed_process = _MANAGED_ENGINE_PROCESSES.pop(id(engine), None)

    try:
        engine.quit()
    finally:
        if managed_process is not None:
            _wait_for_managed_process(managed_process)


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
            close_matlab_engine(engine)


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


def _start_shared_engine_fallback(matlab_engine: Any, original_error: UnicodeDecodeError) -> Any:
    matlab_executable = _resolve_matlab_executable()
    share_name = f"lumen2_{os.getpid()}_{uuid4().hex[:8]}"
    startup_tokens = _build_shared_startup_tokens(share_name)
    process = subprocess.Popen(
        [str(matlab_executable), *startup_tokens],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    deadline = time.monotonic() + 60.0
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    "MATLAB exited before a shared engine session became available."
                ) from original_error

            if share_name in matlab_engine.find_matlab():
                engine = matlab_engine.connect_matlab(share_name)
                _MANAGED_ENGINE_PROCESSES[id(engine)] = process
                return engine

            time.sleep(1.0)
    except Exception:
        if process.poll() is None:
            process.kill()
        _wait_for_managed_process(process)
        raise

    if process.poll() is None:
        process.kill()
    _wait_for_managed_process(process)
    raise RuntimeError(
        "Timed out while starting a shared MATLAB session. "
        "Set the MATLAB_EXE environment variable if MATLAB is installed in a nonstandard location."
    ) from original_error


def _build_shared_startup_tokens(share_name: str) -> list[str]:
    tokens = ["-nodesktop", "-nosplash"]
    tokens.append("-r")
    tokens.append(f"matlab.engine.shareEngine('{share_name}')")
    return tokens


def _resolve_matlab_executable() -> Path:
    configured_path = os.environ.get("MATLAB_EXE")
    if configured_path:
        executable = Path(configured_path).expanduser().resolve()
        if executable.exists():
            return executable
        raise FileNotFoundError(f"MATLAB_EXE does not exist: {executable}")

    discovered = shutil.which("matlab.exe") or shutil.which("matlab")
    if discovered:
        return Path(discovered).resolve()

    windows_candidates: list[Path] = []
    for root in (Path("C:/Program Files/MATLAB"), Path("D:/Program Files/MATLAB")):
        if root.exists():
            windows_candidates.extend(root.glob("R*/bin/matlab.exe"))

    if windows_candidates:
        return sorted(windows_candidates, reverse=True)[0].resolve()

    raise FileNotFoundError(
        "Could not locate matlab.exe. Add it to PATH or set the MATLAB_EXE environment variable."
    )


def _wait_for_managed_process(process: subprocess.Popen[Any]) -> None:
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10.0)
