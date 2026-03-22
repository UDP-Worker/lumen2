from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

try:
    from scipy.optimize import linear_sum_assignment as _scipy_linear_sum_assignment
except ImportError:
    _scipy_linear_sum_assignment = None

from backend.calibrate._shared import ensure_output_dir, load_raw_yaml, resolve_model_name
from backend.model import load_model_config
from backend.utils.complex_response_viewer import (
    ComplexCurve,
    CurveGroup,
    SelectionCancelledError,
    SelectionRecord,
    ViewerSelectionResult,
    build_curve_group,
    select_extinction_reference,
    select_variable_targets,
)


def collect_model_reference(
    groups: Sequence[CurveGroup] | Mapping[str, CurveGroup],
    *,
    title: str = "Model Calibration Reference",
) -> SelectionRecord:
    return select_extinction_reference(groups, title=title)


def collect_control_targets(
    groups: Sequence[CurveGroup] | Mapping[str, CurveGroup],
    *,
    title: str = "Control Calibration Targets",
    shared_baseline: bool = True,
) -> ViewerSelectionResult:
    return select_variable_targets(groups, title=title, shared_baseline=shared_baseline)


def run_voltage_to_tunable_calibration(
    config_path: str | Path,
    *,
    model_calibration_path: str | Path | None = None,
    control_calibration_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    grid_points: int = 64,
    min_overlap_db: float = 0.25,
    acceptance_cost: float = 0.35,
) -> dict[str, Any]:
    raw_config = load_raw_yaml(config_path)
    resolved_config = load_model_config(config_path)
    model_name = resolve_model_name(raw_config, resolved_config)
    destination_dir = Path(output_dir).resolve() if output_dir is not None else ensure_output_dir(model_name)
    destination_dir.mkdir(parents=True, exist_ok=True)

    model_json = (
        Path(model_calibration_path).resolve()
        if model_calibration_path is not None
        else destination_dir / "model_calibration.json"
    )
    control_json = (
        Path(control_calibration_path).resolve()
        if control_calibration_path is not None
        else destination_dir / "control_calibration.json"
    )

    model_payload = _load_json(model_json)
    control_payload = _load_json(control_json)

    parameter_names = list(model_payload["results"].keys())
    channel_names = list(control_payload["results"].keys())
    if not parameter_names or not channel_names:
        raise ValueError("Both model and control calibration results must be non-empty.")

    pair_cache: dict[tuple[str, str], dict[str, Any]] = {}
    cost_matrix = np.full((len(channel_names), len(parameter_names)), 1e6, dtype=float)

    for channel_index, channel_name in enumerate(channel_names):
        channel_result = control_payload["results"][channel_name]
        channel_values = _as_float_array(channel_result["sweep_values"])
        channel_er = _as_float_array(channel_result["extinction_ratio_db"])
        for parameter_index, parameter_name in enumerate(parameter_names):
            parameter_result = model_payload["results"][parameter_name]
            parameter_values = _as_float_array(parameter_result["sweep_values"])
            parameter_er = _as_float_array(parameter_result["extinction_ratio_db"])

            pair_result = _compare_calibration_curves(
                channel_values,
                channel_er,
                parameter_values,
                parameter_er,
                grid_points=grid_points,
                min_overlap_db=min_overlap_db,
            )
            pair_cache[(channel_name, parameter_name)] = pair_result
            cost_matrix[channel_index, parameter_index] = pair_result["cost"]

    row_indices, col_indices = _linear_sum_assignment(cost_matrix)
    assignments: dict[str, Any] = {}
    matched_channels: set[str] = set()
    matched_parameters: set[str] = set()

    for row_index, col_index in zip(row_indices, col_indices, strict=True):
        channel_name = channel_names[row_index]
        parameter_name = parameter_names[col_index]
        pair_result = pair_cache[(channel_name, parameter_name)]
        if not pair_result["valid"]:
            continue

        voltage_samples = np.asarray(pair_result["channel_samples"], dtype=float)
        parameter_samples = np.asarray(pair_result["parameter_samples"], dtype=float)
        order = np.argsort(voltage_samples)
        voltage_grid = voltage_samples[order]
        parameter_grid = parameter_samples[order]
        unique_voltage, unique_indices = np.unique(voltage_grid, return_index=True)
        unique_parameter = parameter_grid[unique_indices]

        assignments[channel_name] = {
            "parameter_name": parameter_name,
            "accepted": bool(pair_result["cost"] <= acceptance_cost),
            "match_cost": float(pair_result["cost"]),
            "orientation": pair_result["orientation"],
            "overlap_range_db": [
                float(pair_result["overlap_min_db"]),
                float(pair_result["overlap_max_db"]),
            ],
            "extinction_ratio_grid_db": [float(value) for value in pair_result["er_grid"]],
            "voltage_samples": [float(value) for value in voltage_samples],
            "parameter_samples": [float(value) for value in parameter_samples],
            "voltage_grid": [float(value) for value in unique_voltage],
            "parameter_grid": [float(value) for value in unique_parameter],
            "channel_zero_voltage": float(control_payload["results"][channel_name]["zero_voltage"]),
            "parameter_zero_value": float(model_payload["results"][parameter_name]["zero_tunable_value"]),
        }
        matched_channels.add(channel_name)
        matched_parameters.add(parameter_name)

    candidate_rankings = {
        channel_name: [
            {
                "parameter_name": parameter_name,
                "cost": float(pair_cache[(channel_name, parameter_name)]["cost"]),
                "valid": bool(pair_cache[(channel_name, parameter_name)]["valid"]),
            }
            for parameter_name in sorted(
                parameter_names,
                key=lambda candidate: pair_cache[(channel_name, candidate)]["cost"],
            )
        ]
        for channel_name in channel_names
    }

    payload = {
        "model_name": model_name,
        "config_path": str(Path(config_path).resolve()),
        "model_calibration_path": str(model_json),
        "control_calibration_path": str(control_json),
        "assignments": assignments,
        "candidate_rankings": candidate_rankings,
        "unmatched_channels": sorted(set(channel_names) - matched_channels),
        "unmatched_parameters": sorted(set(parameter_names) - matched_parameters),
        "acceptance_cost": float(acceptance_cost),
        "grid_points": int(grid_points),
        "min_overlap_db": float(min_overlap_db),
    }
    json_path = destination_dir / "voltage_to_tunable_mapping.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    payload["json_path"] = str(json_path.resolve())
    return payload


def _compare_calibration_curves(
    channel_values: NDArray[np.float64],
    channel_er: NDArray[np.float64],
    parameter_values: NDArray[np.float64],
    parameter_er: NDArray[np.float64],
    *,
    grid_points: int,
    min_overlap_db: float,
) -> dict[str, Any]:
    channel_er_sorted, channel_values_sorted = _prepare_inverse_curve(channel_values, channel_er)
    parameter_er_sorted, parameter_values_sorted = _prepare_inverse_curve(parameter_values, parameter_er)

    overlap_min = max(float(channel_er_sorted.min()), float(parameter_er_sorted.min()))
    overlap_max = min(float(channel_er_sorted.max()), float(parameter_er_sorted.max()))
    overlap_span = overlap_max - overlap_min
    if overlap_span <= min_overlap_db:
        return {
            "valid": False,
            "cost": 1e6,
            "orientation": "unknown",
            "overlap_min_db": overlap_min,
            "overlap_max_db": overlap_max,
            "er_grid": [],
            "channel_samples": [],
            "parameter_samples": [],
        }

    er_grid = np.linspace(overlap_min, overlap_max, grid_points)
    channel_samples = np.interp(er_grid, channel_er_sorted, channel_values_sorted)
    parameter_samples = np.interp(er_grid, parameter_er_sorted, parameter_values_sorted)

    channel_norm = _normalize_samples(channel_samples)
    parameter_norm = _normalize_samples(parameter_samples)
    direct_cost = float(np.sqrt(np.mean((channel_norm - parameter_norm) ** 2)))
    inverse_cost = float(np.sqrt(np.mean((channel_norm - (1.0 - parameter_norm)) ** 2)))
    direct = direct_cost <= inverse_cost

    return {
        "valid": True,
        "cost": direct_cost if direct else inverse_cost,
        "orientation": "same" if direct else "inverse",
        "overlap_min_db": overlap_min,
        "overlap_max_db": overlap_max,
        "er_grid": [float(value) for value in er_grid],
        "channel_samples": [float(value) for value in channel_samples],
        "parameter_samples": [float(value) for value in parameter_samples],
    }


def _prepare_inverse_curve(
    values: NDArray[np.float64],
    extinction_ratio_db: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    order = np.argsort(extinction_ratio_db)
    er_sorted = np.asarray(extinction_ratio_db[order], dtype=float)
    values_sorted = np.asarray(values[order], dtype=float)

    unique_er, unique_indices = np.unique(er_sorted, return_index=True)
    return unique_er, values_sorted[unique_indices]


def _normalize_samples(values: Sequence[float] | NDArray[np.float64]) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    lower = float(np.min(array))
    upper = float(np.max(array))
    if np.isclose(lower, upper):
        return np.zeros_like(array)
    return (array - lower) / (upper - lower)


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).resolve().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _as_float_array(values: Sequence[float]) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 2:
        raise ValueError("Calibration samples must be 1D arrays with at least two elements.")
    return array


def _linear_sum_assignment(cost_matrix: NDArray[np.float64]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    if _scipy_linear_sum_assignment is not None:
        row_indices, col_indices = _scipy_linear_sum_assignment(cost_matrix)
        return np.asarray(row_indices, dtype=np.int64), np.asarray(col_indices, dtype=np.int64)

    rows, cols = cost_matrix.shape
    if rows <= cols:
        assignment = _solve_rectangular_assignment(cost_matrix)
        row_indices = np.arange(rows, dtype=np.int64)
        col_indices = np.asarray(assignment, dtype=np.int64)
        return row_indices, col_indices

    reversed_assignment = _solve_rectangular_assignment(cost_matrix.T)
    row_indices = np.asarray(reversed_assignment, dtype=np.int64)
    col_indices = np.arange(cols, dtype=np.int64)
    return row_indices, col_indices


def _solve_rectangular_assignment(cost_matrix: NDArray[np.float64]) -> list[int]:
    rows, cols = cost_matrix.shape
    if cols > 20:
        raise RuntimeError(
            "SciPy is unavailable and the pure-Python assignment fallback only supports up to 20 columns."
        )

    @lru_cache(maxsize=None)
    def solve(row_index: int, used_mask: int) -> tuple[float, tuple[int, ...]]:
        if row_index == rows:
            return 0.0, ()

        best_cost = float("inf")
        best_assignment: tuple[int, ...] = ()
        for col_index in range(cols):
            if used_mask & (1 << col_index):
                continue
            tail_cost, tail_assignment = solve(row_index + 1, used_mask | (1 << col_index))
            candidate_cost = float(cost_matrix[row_index, col_index]) + tail_cost
            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_assignment = (col_index, *tail_assignment)
        return best_cost, best_assignment

    _, assignment = solve(0, 0)
    return list(assignment)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Map control voltages to model tunable parameters.")
    parser.add_argument("config", type=Path, help="Path to the model YAML config.")
    parser.add_argument("--model-calibration", type=Path, default=None)
    parser.add_argument("--control-calibration", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--grid-points", type=int, default=64)
    parser.add_argument("--min-overlap-db", type=float, default=0.25)
    parser.add_argument("--acceptance-cost", type=float, default=0.35)
    args = parser.parse_args(list(argv) if argv is not None else None)

    payload = run_voltage_to_tunable_calibration(
        args.config,
        model_calibration_path=args.model_calibration,
        control_calibration_path=args.control_calibration,
        output_dir=args.output_dir,
        grid_points=args.grid_points,
        min_overlap_db=args.min_overlap_db,
        acceptance_cost=args.acceptance_cost,
    )
    print(f"saved_json: {payload['json_path']}")
    print(f"assignment_count: {len(payload['assignments'])}")
    return 0


__all__ = [
    "ComplexCurve",
    "CurveGroup",
    "SelectionCancelledError",
    "SelectionRecord",
    "ViewerSelectionResult",
    "build_curve_group",
    "collect_control_targets",
    "collect_model_reference",
    "run_voltage_to_tunable_calibration",
    "select_extinction_reference",
    "select_variable_targets",
]


if __name__ == "__main__":
    raise SystemExit(main())
