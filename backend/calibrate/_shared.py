from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from numpy.typing import NDArray

from backend.utils.complex_response_viewer import CurveGroup, build_curve_group

REPO_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_DATA_ROOT = REPO_ROOT / "backend" / "calibrate" / "data"


@dataclass(slots=True)
class CurveSweepBundle:
    name: str
    sweep_values: NDArray[np.float64]
    wavelength_nm: NDArray[np.float64]
    complex_response_matrix: NDArray[np.complex128]
    power_db_matrix: NDArray[np.float64]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_curve_group(self) -> CurveGroup:
        return build_curve_group(
            self.name,
            self.wavelength_nm,
            self.complex_response_matrix,
            sweep_values=self.sweep_values.tolist(),
        )

    def power_db_at(self, wavelength_nm: float) -> NDArray[np.float64]:
        return np.asarray(
            [
                np.interp(
                    float(wavelength_nm),
                    self.wavelength_nm,
                    self.power_db_matrix[index],
                )
                for index in range(self.power_db_matrix.shape[0])
            ],
            dtype=float,
        )


def load_raw_yaml(config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    with config_file.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{config_file} must contain a top-level YAML mapping.")
    return raw


def write_raw_yaml(config_path: str | Path, payload: Mapping[str, Any]) -> Path:
    config_file = Path(config_path).resolve()
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with config_file.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(payload), handle, allow_unicode=True, sort_keys=False)
    return config_file


def resolve_model_name(raw_config: Mapping[str, Any], resolved_config: Mapping[str, Any]) -> str:
    model_section = raw_config.get("model")
    if isinstance(model_section, Mapping) and isinstance(model_section.get("name"), str):
        return str(model_section["name"])
    return str(resolved_config["model"]["name"])


def ensure_output_dir(model_name: str) -> Path:
    output_dir = CALIBRATION_DATA_ROOT / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def resolve_zero_tunable_config(
    raw_config: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
) -> dict[str, float]:
    calibration_section = raw_config.get("calibration")
    zero_section = calibration_section.get("zero_config") if isinstance(calibration_section, Mapping) else None
    tunable_zero = zero_section.get("tunable") if isinstance(zero_section, Mapping) else None
    if isinstance(tunable_zero, Mapping):
        resolved: dict[str, float] = {}
        for name in resolved_config["parameters"]["tunable"]:
            if name not in tunable_zero:
                raise ValueError(f"calibration.zero_config.tunable.{name} is missing.")
            resolved[name] = float(tunable_zero[name])
        return resolved

    return {
        name: float(spec["value"])
        for name, spec in resolved_config["parameters"]["tunable"].items()
    }


def save_zero_tunable_config(
    config_path: str | Path,
    tunable_values: Mapping[str, float],
) -> Path:
    raw_config = load_raw_yaml(config_path)

    calibration_section = raw_config.get("calibration")
    if calibration_section is None:
        calibration_section = {}
        raw_config["calibration"] = calibration_section
    elif not isinstance(calibration_section, Mapping):
        raise ValueError("calibration must be a mapping.")

    zero_section = calibration_section.get("zero_config")
    if zero_section is None:
        zero_section = {}
        calibration_section["zero_config"] = zero_section
    elif not isinstance(zero_section, Mapping):
        raise ValueError("calibration.zero_config must be a mapping.")

    zero_section["tunable"] = {
        str(name): float(value)
        for name, value in tunable_values.items()
    }
    return write_raw_yaml(config_path, raw_config)


def power_db_to_complex_amplitude(power_db: Sequence[float] | NDArray[np.float64]) -> NDArray[np.complex128]:
    power_array = np.asarray(power_db, dtype=float)
    amplitude = np.power(10.0, power_array / 20.0)
    return amplitude.astype(np.complex128)


def extinction_ratio_db(power_db: Sequence[float] | NDArray[np.float64], baseline_db: float) -> NDArray[np.float64]:
    return float(baseline_db) - np.asarray(power_db, dtype=float)


def estimate_zero_crossing(
    x_values: Sequence[float] | NDArray[np.float64],
    extinction_values_db: Sequence[float] | NDArray[np.float64],
) -> float:
    x_array = np.asarray(x_values, dtype=float)
    y_array = np.asarray(extinction_values_db, dtype=float)
    if x_array.ndim != 1 or y_array.ndim != 1 or x_array.size != y_array.size:
        raise ValueError("x_values and extinction_values_db must be 1D arrays with the same length.")
    if x_array.size == 0:
        raise ValueError("At least one sample is required to estimate a zero crossing.")

    order = np.argsort(x_array)
    x_sorted = x_array[order]
    y_sorted = y_array[order]

    for index in range(len(x_sorted) - 1):
        y0 = y_sorted[index]
        y1 = y_sorted[index + 1]
        if y0 == 0.0:
            return float(x_sorted[index])
        if y0 * y1 < 0.0:
            x0 = x_sorted[index]
            x1 = x_sorted[index + 1]
            return float(x0 + (0.0 - y0) * (x1 - x0) / (y1 - y0))

    return float(x_sorted[np.argmin(np.abs(y_sorted))])


def write_json(destination: str | Path, payload: Mapping[str, Any]) -> Path:
    destination_path = Path(destination).resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with destination_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return destination_path


def save_curve_archives(destination: str | Path, bundles: Sequence[CurveSweepBundle]) -> Path:
    destination_path = Path(destination).resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, NDArray[np.generic]] = {}
    for bundle in bundles:
        prefix = sanitize_key(bundle.name)
        arrays[f"{prefix}__sweep_values"] = np.asarray(bundle.sweep_values, dtype=float)
        arrays[f"{prefix}__wavelength_nm"] = np.asarray(bundle.wavelength_nm, dtype=float)
        arrays[f"{prefix}__power_db"] = np.asarray(bundle.power_db_matrix, dtype=float)
        arrays[f"{prefix}__complex_real"] = np.asarray(bundle.complex_response_matrix.real, dtype=float)
        arrays[f"{prefix}__complex_imag"] = np.asarray(bundle.complex_response_matrix.imag, dtype=float)

    np.savez(destination_path, **arrays)
    return destination_path


def sanitize_key(text: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z_]+", "_", text.strip())
    return normalized or "curve"


def get_mapping(
    mapping: Mapping[str, Any] | None,
    key: str,
    *,
    default: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    if mapping is None:
        return {} if default is None else dict(default)
    value = mapping.get(key)
    if value is None:
        return {} if default is None else dict(default)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping.")
    return value


def get_sequence(
    mapping: Mapping[str, Any] | None,
    key: str,
    *,
    default: Sequence[Any] | None = None,
) -> list[Any]:
    if mapping is None or mapping.get(key) is None:
        return list(default or [])
    value = mapping[key]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{key} must be a sequence.")
    return list(value)


def bundle_from_power_sweeps(
    name: str,
    sweep_values: Sequence[float],
    wavelength_nm: Sequence[float] | NDArray[np.float64],
    power_db_matrix: Sequence[Sequence[float]] | NDArray[np.float64],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> CurveSweepBundle:
    wavelength_array = np.asarray(wavelength_nm, dtype=float)
    power_matrix = np.asarray(power_db_matrix, dtype=float)
    if power_matrix.ndim != 2:
        raise ValueError("power_db_matrix must be a 2D array.")
    if power_matrix.shape[0] != len(sweep_values):
        raise ValueError("power_db_matrix row count must match sweep_values length.")
    if power_matrix.shape[1] != wavelength_array.size:
        raise ValueError("power_db_matrix column count must match wavelength_nm length.")
    return CurveSweepBundle(
        name=name,
        sweep_values=np.asarray(sweep_values, dtype=float),
        wavelength_nm=wavelength_array,
        complex_response_matrix=power_db_to_complex_amplitude(power_matrix),
        power_db_matrix=power_matrix,
        metadata=dict(metadata or {}),
    )


def summarize_curve_bundle(bundle: CurveSweepBundle) -> dict[str, Any]:
    return {
        "name": bundle.name,
        "sweep_values": [float(value) for value in bundle.sweep_values],
        "wavelength_nm_range": [
            float(np.min(bundle.wavelength_nm)),
            float(np.max(bundle.wavelength_nm)),
        ],
        "sample_count": int(bundle.wavelength_nm.size),
    }
