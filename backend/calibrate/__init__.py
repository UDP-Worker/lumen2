from __future__ import annotations

from typing import Any


def collect_control_targets(*args: Any, **kwargs: Any) -> Any:
    from .calibrate import collect_control_targets as _collect_control_targets

    return _collect_control_targets(*args, **kwargs)


def collect_model_reference(*args: Any, **kwargs: Any) -> Any:
    from .calibrate import collect_model_reference as _collect_model_reference

    return _collect_model_reference(*args, **kwargs)


def run_control_calibration(*args: Any, **kwargs: Any) -> Any:
    from .calibrate_control import run_control_calibration as _run_control_calibration

    return _run_control_calibration(*args, **kwargs)


def run_model_calibration(*args: Any, **kwargs: Any) -> Any:
    from .calibrate_model import run_model_calibration as _run_model_calibration

    return _run_model_calibration(*args, **kwargs)


def run_voltage_to_tunable_calibration(*args: Any, **kwargs: Any) -> Any:
    from .calibrate import run_voltage_to_tunable_calibration as _run_voltage_to_tunable_calibration

    return _run_voltage_to_tunable_calibration(*args, **kwargs)


__all__ = [
    "collect_control_targets",
    "collect_model_reference",
    "run_control_calibration",
    "run_model_calibration",
    "run_voltage_to_tunable_calibration",
]
