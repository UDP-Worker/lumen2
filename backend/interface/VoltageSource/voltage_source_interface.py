from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..matlab_bridge import (
    add_matlab_paths,
    close_matlab_engine,
    matlab_engine_session,
    matlab_struct_get,
    start_matlab_engine,
    to_matlab_row_vector,
    to_numpy_1d,
)

VOLTAGE_SOURCE_MATLAB_DIR = Path(__file__).resolve().parent


def connect_voltage_source(com_port: int, engine: Any | None = None) -> Any:
    matlab_engine = engine
    if matlab_engine is None:
        matlab_engine = start_matlab_engine(extra_paths=(VOLTAGE_SOURCE_MATLAB_DIR,))
    else:
        add_matlab_paths(matlab_engine, (VOLTAGE_SOURCE_MATLAB_DIR,))

    matlab_engine.silicon_extreme_api("connect", float(com_port), nargout=1)
    return matlab_engine


def disconnect_voltage_source(engine: Any, *, stop_engine: bool = False) -> None:
    try:
        add_matlab_paths(engine, (VOLTAGE_SOURCE_MATLAB_DIR,))
        engine.silicon_extreme_api("disconnect", nargout=1)
    finally:
        if stop_engine:
            close_matlab_engine(engine)


def configure_channel_limits(
    channels: Sequence[int],
    *,
    vmax: float | Sequence[float] | None = None,
    imax: float | Sequence[float] | None = None,
    engine: Any | None = None,
    com_port: int | None = None,
) -> None:
    with _voltage_source_context(engine, com_port=com_port) as matlab_engine:
        matlab_engine.silicon_extreme_api(
            "configure_limits",
            to_matlab_row_vector(channels),
            _to_matlab_scalar_or_vector(vmax),
            _to_matlab_scalar_or_vector(imax),
            nargout=1,
        )


def set_channel_voltage(
    channel: int,
    voltage: float,
    *,
    engine: Any | None = None,
    com_port: int | None = None,
) -> float:
    with _voltage_source_context(engine, com_port=com_port) as matlab_engine:
        result = matlab_engine.silicon_extreme_api(
            "set_voltage",
            float(channel),
            float(voltage),
            nargout=1,
        )

    return float(matlab_struct_get(result, "voltage"))


def set_channel_voltages(
    channel_to_voltage: Mapping[int, float] | Sequence[tuple[int, float]],
    *,
    engine: Any | None = None,
    com_port: int | None = None,
) -> dict[int, dict[str, float]]:
    channels, voltages = _normalize_assignments(channel_to_voltage)
    with _voltage_source_context(engine, com_port=com_port) as matlab_engine:
        result = matlab_engine.silicon_extreme_api(
            "set_voltages",
            to_matlab_row_vector(channels),
            to_matlab_row_vector(voltages),
            nargout=1,
        )

    return _snapshot_result_to_dict(result)


def apply_channel_voltages(
    channel_to_voltage: Mapping[int, float] | Sequence[tuple[int, float]],
    *,
    com_port: int,
    vmax: float | Sequence[float] | None = None,
    imax: float | Sequence[float] | None = None,
    settle_time_s: float = 0.5,
    engine: Any | None = None,
) -> dict[int, dict[str, float]]:
    channels, voltages = _normalize_assignments(channel_to_voltage)

    with _voltage_source_context(engine, com_port=com_port) as matlab_engine:
        result = matlab_engine.SWEEP_SiliconExtreme(
            float(com_port),
            to_matlab_row_vector(channels),
            to_matlab_row_vector(voltages),
            "Vmax",
            _to_matlab_scalar_or_vector(vmax),
            "Imax",
            _to_matlab_scalar_or_vector(imax),
            "settle_time_s",
            float(settle_time_s),
            "disconnect_when_done",
            False,
            nargout=1,
        )

    return _snapshot_result_to_dict(result)


def read_channel_voltage(
    channel: int,
    *,
    engine: Any | None = None,
    com_port: int | None = None,
) -> float:
    return _read_scalar("read_voltage", channel, engine=engine, com_port=com_port)


def read_channel_current(
    channel: int,
    *,
    engine: Any | None = None,
    com_port: int | None = None,
) -> float:
    return _read_scalar("read_current", channel, engine=engine, com_port=com_port)


def read_channel_power(
    channel: int,
    *,
    engine: Any | None = None,
    com_port: int | None = None,
) -> float:
    return _read_scalar("read_power", channel, engine=engine, com_port=com_port)


def snapshot_channels(
    channels: Sequence[int],
    *,
    engine: Any | None = None,
    com_port: int | None = None,
) -> dict[int, dict[str, float]]:
    with _voltage_source_context(engine, com_port=com_port) as matlab_engine:
        result = matlab_engine.silicon_extreme_api(
            "snapshot",
            to_matlab_row_vector(channels),
            nargout=1,
        )

    return _snapshot_result_to_dict(result)


@contextmanager
def _voltage_source_context(
    engine: Any | None,
    *,
    com_port: int | None,
):
    created_here = engine is None
    if created_here and com_port is None:
        raise ValueError("com_port is required when no MATLAB engine is provided.")

    with matlab_engine_session(engine, extra_paths=(VOLTAGE_SOURCE_MATLAB_DIR,)) as matlab_engine:
        if com_port is not None:
            matlab_engine.silicon_extreme_api("connect", float(com_port), nargout=1)

        try:
            yield matlab_engine
        finally:
            if created_here:
                try:
                    matlab_engine.silicon_extreme_api("disconnect", nargout=1)
                except Exception:
                    pass


def _read_scalar(
    action: str,
    channel: int,
    *,
    engine: Any | None,
    com_port: int | None,
) -> float:
    with _voltage_source_context(engine, com_port=com_port) as matlab_engine:
        return float(matlab_engine.silicon_extreme_api(action, float(channel), nargout=1))


def _normalize_assignments(
    assignments: Mapping[int, float] | Sequence[tuple[int, float]],
) -> tuple[list[int], list[float]]:
    if isinstance(assignments, Mapping):
        items: Iterable[tuple[int, float]] = assignments.items()
    else:
        items = assignments

    channels: list[int] = []
    voltages: list[float] = []
    for channel, voltage in items:
        channels.append(int(channel))
        voltages.append(float(voltage))

    if not channels:
        raise ValueError("At least one channel assignment is required.")

    return channels, voltages


def _to_matlab_scalar_or_vector(value: float | Sequence[float] | None) -> Any:
    if value is None:
        return to_matlab_row_vector(())

    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("Numeric limit values cannot be strings or bytes.")

    try:
        return float(value)
    except (TypeError, ValueError):
        return to_matlab_row_vector([float(item) for item in value])


def _snapshot_result_to_dict(result: Any) -> dict[int, dict[str, float]]:
    channels = to_numpy_1d(matlab_struct_get(result, "channels")).astype(int, copy=False)
    voltages = to_numpy_1d(matlab_struct_get(result, "voltages"))
    currents = to_numpy_1d(matlab_struct_get(result, "currents"))
    powers = to_numpy_1d(matlab_struct_get(result, "powers"))

    snapshot: dict[int, dict[str, float]] = {}
    for channel, voltage, current, power in zip(channels, voltages, currents, powers, strict=True):
        snapshot[int(channel)] = {
            "voltage": float(voltage),
            "current": float(current),
            "power": float(power),
        }

    return snapshot
