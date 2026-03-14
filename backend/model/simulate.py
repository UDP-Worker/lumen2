from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import yaml

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.interface.matlab_bridge import matlab_engine_session

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATLAB_DIR = REPO_ROOT / "backend" / "model" / "MATLAB"


def load_model_config(config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    with config_file.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)

    if not isinstance(raw_config, dict):
        raise ValueError(f"Config file {config_file} must contain a YAML mapping at the top level.")

    return _resolve_model_config(raw_config, config_file)


def simulate_from_config(
    config_path: str | Path,
    *,
    engine: Any | None = None,
) -> dict[str, Any]:
    config = load_model_config(config_path)
    return simulate_from_dict(config, engine=engine)


def simulate_from_dict(
    config: Mapping[str, Any],
    *,
    engine: Any | None = None,
) -> dict[str, Any]:
    resolved_config = dict(config)
    matlab_dir = Path(resolved_config["model"]["matlab_dir"]).resolve()
    entrypoint = str(resolved_config["model"]["entrypoint"])
    payload = json.dumps(resolved_config, ensure_ascii=False)

    with matlab_engine_session(engine, extra_paths=(matlab_dir,)) as matlab_engine:
        matlab_result = matlab_engine.feval(entrypoint, payload, nargout=1)

    result = _convert_matlab_value(matlab_result)
    if not isinstance(result, dict):
        raise TypeError("MATLAB model must return a struct-like result.")

    return result


def save_result_npz(result: Mapping[str, Any], output_path: str | Path) -> Path:
    destination = Path(output_path).resolve()
    np.savez(
        destination,
        wavelength_nm=np.asarray(result["wavelength_nm"]),
        frequency_hz=np.asarray(result["frequency_hz"]),
        complex_response=np.asarray(result["complex_response"]),
        power_linear=np.asarray(result["power_linear"]),
        power_db=np.asarray(result["power_db"]),
    )
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a MATLAB photonic model from a YAML config.")
    parser.add_argument("config", type=Path, help="Path to a .yml/.yaml model config file.")
    parser.add_argument(
        "--save-npz",
        type=Path,
        default=None,
        help="Optional path to save the main simulation arrays as a .npz file.",
    )
    args = parser.parse_args(argv)

    result = simulate_from_config(args.config)
    if args.save_npz is not None:
        output_file = save_result_npz(result, args.save_npz)
        print(f"saved_npz: {output_file}")

    _print_summary(result)
    return 0


def _resolve_model_config(raw_config: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    model_section = _require_mapping(raw_config.get("model"), "model")
    simulation_section = _require_mapping(raw_config.get("simulation"), "simulation")
    wavelength_section = _require_mapping(simulation_section.get("wavelength_nm"), "simulation.wavelength_nm")
    outputs_section = _optional_mapping(raw_config.get("outputs"))
    parameters_section = _require_mapping(raw_config.get("parameters"), "parameters")

    config_dir = config_path.parent
    matlab_dir = _resolve_existing_path(
        model_section.get("matlab_dir", "backend/model/MATLAB"),
        config_dir=config_dir,
    )

    resolved = {
        "version": int(raw_config.get("version", 1)),
        "config_path": str(config_path),
        "model": {
            "name": _require_text(model_section.get("name"), "model.name"),
            "entrypoint": _require_text(
                model_section.get("entrypoint", model_section.get("name")),
                "model.entrypoint",
            ),
            "matlab_dir": str(matlab_dir),
        },
        "simulation": {
            "wavelength_nm": {
                "start": _require_float(wavelength_section.get("start"), "simulation.wavelength_nm.start"),
                "stop": _require_float(wavelength_section.get("stop"), "simulation.wavelength_nm.stop"),
                "step": _require_positive_float(
                    wavelength_section.get("step"),
                    "simulation.wavelength_nm.step",
                ),
            }
        },
        "outputs": {
            "observe_port": _require_text(outputs_section.get("observe_port", "C2"), "outputs.observe_port").upper()
        },
        "parameters": {
            "tunable": _resolve_tunable_parameters(parameters_section.get("tunable", {})),
            "fixed": _resolve_fixed_parameters(parameters_section.get("fixed", {})),
        },
    }

    wavelength_config = resolved["simulation"]["wavelength_nm"]
    if wavelength_config["stop"] <= wavelength_config["start"]:
        raise ValueError("simulation.wavelength_nm.stop must be greater than start.")

    return resolved


def _resolve_tunable_parameters(raw_tunable: Any) -> dict[str, dict[str, Any]]:
    tunable_section = _optional_mapping(raw_tunable)
    resolved: dict[str, dict[str, Any]] = {}
    for name, spec in tunable_section.items():
        spec_mapping = _require_mapping(spec, f"parameters.tunable.{name}")
        value = _normalize_yaml_scalar(spec_mapping.get("value"), f"parameters.tunable.{name}.value")
        bounds = spec_mapping.get("bounds")
        if bounds is not None:
            if not isinstance(bounds, list) or len(bounds) != 2:
                raise ValueError(f"parameters.tunable.{name}.bounds must be a 2-element list.")
            resolved_bounds = [
                _require_float(bounds[0], f"parameters.tunable.{name}.bounds[0]"),
                _require_float(bounds[1], f"parameters.tunable.{name}.bounds[1]"),
            ]
        else:
            resolved_bounds = None

        resolved[name] = {
            "value": value,
            "bounds": resolved_bounds,
        }

    return resolved


def _resolve_fixed_parameters(raw_fixed: Any) -> dict[str, Any]:
    fixed_section = _optional_mapping(raw_fixed)
    return {
        name: _normalize_yaml_scalar(value, f"parameters.fixed.{name}")
        for name, value in fixed_section.items()
    }


def _normalize_yaml_scalar(value: Any, field_name: str) -> Any:
    if isinstance(value, Mapping):
        keys = set(value.keys())
        if keys <= {"real", "imag"} and "real" in keys:
            return {
                "real": _require_float(value.get("real"), f"{field_name}.real"),
                "imag": _require_float(value.get("imag", 0.0), f"{field_name}.imag"),
            }

        return {str(key): _normalize_yaml_scalar(child, f"{field_name}.{key}") for key, child in value.items()}

    if isinstance(value, list):
        return [_normalize_yaml_scalar(item, f"{field_name}[]") for item in value]

    if isinstance(value, (bool, int, float, str)):
        return value

    raise TypeError(f"{field_name} must be a scalar, a complex mapping, or a list.")


def _convert_matlab_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _convert_matlab_value(child) for key, child in value.items()}

    if isinstance(value, (str, bool, int, float, complex)) or value is None:
        return value

    if isinstance(value, (list, tuple)):
        return [_convert_matlab_value(item) for item in value]

    try:
        array = np.asarray(value)
    except Exception:
        return value

    if array.dtype == object:
        return [_convert_matlab_value(item) for item in array.reshape(-1)]

    if array.shape == ():
        return array.item()

    if array.ndim == 2 and 1 in array.shape:
        return array.reshape(-1)

    return array


def _resolve_existing_path(raw_path: Any, *, config_dir: Path) -> Path:
    path_text = _require_text(raw_path, "model.matlab_dir")
    candidate = Path(path_text)

    if candidate.is_absolute():
        resolved = candidate.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"MATLAB directory does not exist: {resolved}")
        return resolved

    for base_dir in (config_dir, REPO_ROOT):
        resolved = (base_dir / candidate).resolve()
        if resolved.exists():
            return resolved

    raise FileNotFoundError(f"MATLAB directory does not exist: {path_text}")


def _require_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping.")
    return {str(key): child for key, child in value.items()}


def _optional_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("Expected a mapping.")
    return {str(key): child for key, child in value.items()}


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _require_float(value: Any, field_name: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric.")
    return float(value)


def _require_positive_float(value: Any, field_name: str) -> float:
    numeric_value = _require_float(value, field_name)
    if numeric_value <= 0:
        raise ValueError(f"{field_name} must be greater than 0.")
    return numeric_value


def _print_summary(result: Mapping[str, Any]) -> None:
    wavelength_nm = np.asarray(result["wavelength_nm"], dtype=float)
    power_db = np.asarray(result["power_db"], dtype=float)

    print(f"model_name: {result['model_name']}")
    print(f"port_name: {result['port_name']}")
    print(f"samples: {wavelength_nm.size}")
    print(f"wavelength_range_nm: {wavelength_nm.min():.6f} -> {wavelength_nm.max():.6f}")
    print(f"power_db_range: {power_db.min():.6f} -> {power_db.max():.6f}")


if __name__ == "__main__":
    raise SystemExit(main())
