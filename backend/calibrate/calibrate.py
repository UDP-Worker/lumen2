from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
from scipy.signal import correlate, correlation_lags, savgol_filter

from backend.calibrate._shared import (
    CurveSweepBundle,
    ensure_output_dir,
    get_mapping,
    load_raw_yaml,
    power_db_to_complex_amplitude,
    resolve_model_name,
    sanitize_key,
    write_json,
)
from backend.model import load_model_config


def collect_model_reference(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError(
        "The legacy viewer-based model reference flow has been removed. "
        "Run `backend.calibrate.calibrate_model` first to capture model curves."
    )


def collect_control_targets(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError(
        "The legacy viewer-based control-target flow has been removed. "
        "Run `backend.calibrate.calibrate_control` first to capture control curves."
    )


def run_voltage_to_tunable_calibration(
    config_path: str | Path,
    *,
    model_calibration_path: str | Path | None = None,
    control_calibration_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    channel_parameter_map: Mapping[str | int, str] | None = None,
    grid_points: int = 256,
    acceptance_cost: float = 0.2,
    min_correlation: float = 0.85,
    max_delta_nm: float = 0.05,
    smooth_window: int = 7,
    fit_loss: str = "soft_l1",
    fit_f_scale: float | None = None,
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
    if not isinstance(model_payload.get("results"), Mapping) or not model_payload["results"]:
        raise ValueError("Model calibration JSON must contain non-empty results.")
    if not isinstance(control_payload.get("results"), Mapping) or not control_payload["results"]:
        raise ValueError("Control calibration JSON must contain non-empty results.")

    model_archive_path = _resolve_archive_path(
        model_payload,
        model_json,
        default_filename="model_calibration_curves.npz",
    )
    control_archive_path = _resolve_archive_path(
        control_payload,
        control_json,
        default_filename="control_calibration_curves.npz",
    )

    parameter_names = list(_resolve_parameter_order(model_payload))
    channel_names = list(_resolve_channel_order(control_payload))
    resolved_mapping = _resolve_channel_parameter_map(
        raw_config,
        channel_parameter_map=channel_parameter_map,
        channel_names=channel_names,
        parameter_names=parameter_names,
    )
    if not resolved_mapping:
        raise ValueError(
            "Curve-shape calibration requires a channel-to-parameter relation. "
            "Provide it via CLI or calibration.voltage_to_tunable.channel_to_parameter in YAML."
        )

    model_archive = np.load(model_archive_path)
    control_archive = np.load(control_archive_path)
    try:
        mappings: dict[str, Any] = {}
        for channel_name, parameter_name in resolved_mapping.items():
            control_result = dict(control_payload["results"][channel_name])
            parameter_result = dict(model_payload["results"][parameter_name])

            control_bundle = _load_curve_bundle_from_result(
                control_archive,
                control_result,
                fallback_prefix=sanitize_key(f"ch{channel_name}"),
                bundle_name=f"ch{channel_name}",
                metadata={"channel": int(channel_name)},
            )
            model_bundle = _load_curve_bundle_from_result(
                model_archive,
                parameter_result,
                fallback_prefix=sanitize_key(parameter_name),
                bundle_name=parameter_name,
                metadata={"parameter_name": parameter_name},
            )

            mappings[channel_name] = _match_channel_to_parameter(
                channel_name=channel_name,
                parameter_name=parameter_name,
                control_bundle=control_bundle,
                model_bundle=model_bundle,
                control_result=control_result,
                parameter_result=parameter_result,
                max_delta_nm=max_delta_nm,
                smooth_window=smooth_window,
                fit_loss=fit_loss,
                fit_f_scale=fit_f_scale,
                acceptance_cost=acceptance_cost,
                min_correlation=min_correlation,
                mapping_grid_points=grid_points,
            )
    finally:
        model_archive.close()
        control_archive.close()

    payload = {
        "model_name": model_name,
        "config_path": str(Path(config_path).resolve()),
        "model_calibration_path": str(model_json),
        "control_calibration_path": str(control_json),
        "model_curve_archive": str(model_archive_path),
        "control_curve_archive": str(control_archive_path),
        "channel_to_parameter": resolved_mapping,
        "shape_matching": {
            "mapping_grid_points": int(grid_points),
            "acceptance_cost": float(acceptance_cost),
            "min_correlation": float(min_correlation),
            "max_delta_nm": float(max_delta_nm),
            "smooth_window": int(smooth_window),
            "fit_loss": fit_loss,
            "fit_f_scale": None if fit_f_scale is None else float(fit_f_scale),
        },
        "mappings": mappings,
        "unmapped_channels": sorted(set(channel_names) - set(resolved_mapping)),
        "unused_parameters": sorted(set(parameter_names) - set(resolved_mapping.values())),
    }
    json_path = write_json(destination_dir / "voltage_to_tunable_mapping.json", payload)
    payload["json_path"] = str(json_path)
    return payload


def _match_channel_to_parameter(
    *,
    channel_name: str,
    parameter_name: str,
    control_bundle: CurveSweepBundle,
    model_bundle: CurveSweepBundle,
    control_result: Mapping[str, Any],
    parameter_result: Mapping[str, Any],
    max_delta_nm: float,
    smooth_window: int,
    fit_loss: str,
    fit_f_scale: float | None,
    acceptance_cost: float,
    min_correlation: float,
    mapping_grid_points: int,
) -> dict[str, Any]:
    control_order = np.argsort(control_bundle.sweep_values)
    model_order = np.argsort(model_bundle.sweep_values)
    voltage_sorted = np.asarray(control_bundle.sweep_values[control_order], dtype=float)
    parameter_sorted = np.asarray(model_bundle.sweep_values[model_order], dtype=float)

    pair_results: list[list[dict[str, Any]]] = []
    cost_matrix = np.empty((len(control_order), len(model_order)), dtype=float)

    for row_index, control_curve_index in enumerate(control_order):
        control_wavelength, control_signal = _prepare_curve_signal(
            control_bundle,
            int(control_curve_index),
            smooth_window=smooth_window,
        )
        row_results: list[dict[str, Any]] = []
        for col_index, model_curve_index in enumerate(model_order):
            model_wavelength, model_signal = _prepare_curve_signal(
                model_bundle,
                int(model_curve_index),
                smooth_window=smooth_window,
            )
            fit_result = _fit_curve_pair(
                control_wavelength,
                control_signal,
                model_wavelength,
                model_signal,
                max_delta_nm=max_delta_nm,
                fit_loss=fit_loss,
                fit_f_scale=fit_f_scale,
            )
            row_results.append(fit_result)
            cost_matrix[row_index, col_index] = float(fit_result["fit_cost"])
        pair_results.append(row_results)

    increasing_path = _solve_monotone_path(cost_matrix, increasing=True)
    decreasing_path = _solve_monotone_path(cost_matrix, increasing=False)
    chosen_path = increasing_path if increasing_path["mean_cost"] <= decreasing_path["mean_cost"] else decreasing_path
    orientation = str(chosen_path["orientation"])
    matched_model_positions = np.asarray(chosen_path["path"], dtype=int)
    matched_parameter_sorted = parameter_sorted[matched_model_positions]

    selected_results = [
        pair_results[row_index][model_position]
        for row_index, model_position in enumerate(matched_model_positions)
    ]
    fit_costs = np.asarray([float(item["fit_cost"]) for item in selected_results], dtype=float)
    correlations = np.asarray([float(item["correlation"]) for item in selected_results], dtype=float)
    weights = 1.0 / np.maximum(fit_costs, 1e-6)
    monotone_parameter_sorted = _weighted_isotonic_regression(
        matched_parameter_sorted,
        weights=weights,
        increasing=(orientation == "increasing"),
    )

    unique_voltage, unique_parameter = _merge_duplicate_x(
        voltage_sorted,
        monotone_parameter_sorted,
        weights,
    )
    if unique_voltage.size >= 2:
        interpolator = PchipInterpolator(unique_voltage, unique_parameter, extrapolate=False)
        mapping_voltage_grid = np.linspace(
            float(unique_voltage[0]),
            float(unique_voltage[-1]),
            int(max(mapping_grid_points, unique_voltage.size)),
        )
        mapping_parameter_grid = np.asarray(interpolator(mapping_voltage_grid), dtype=float)
    else:
        mapping_voltage_grid = unique_voltage.copy()
        mapping_parameter_grid = unique_parameter.copy()

    curve_matches: list[dict[str, Any]] = []
    candidate_rankings: list[dict[str, Any]] = []
    for row_index, voltage_value in enumerate(voltage_sorted):
        selected = selected_results[row_index]
        model_position = int(matched_model_positions[row_index])
        actual_model_curve_index = int(model_order[model_position])
        actual_control_curve_index = int(control_order[row_index])

        curve_matches.append(
            {
                "sorted_curve_index": int(row_index),
                "control_curve_index": actual_control_curve_index,
                "model_curve_index": actual_model_curve_index,
                "voltage": float(voltage_value),
                "matched_parameter": float(matched_parameter_sorted[row_index]),
                "monotone_parameter": float(monotone_parameter_sorted[row_index]),
                "fit_cost": float(selected["fit_cost"]),
                "rmse": float(selected["rmse"]),
                "correlation": float(selected["correlation"]),
                "scale": float(selected["scale"]),
                "offset": float(selected["offset"]),
                "wavelength_shift_nm": float(selected["wavelength_shift_nm"]),
                "overlap_nm": [
                    float(selected["overlap_start_nm"]),
                    float(selected["overlap_stop_nm"]),
                ],
            }
        )

        ranking = sorted(
            (
                {
                    "model_curve_index": int(model_order[col_index]),
                    "parameter_value": float(parameter_sorted[col_index]),
                    "fit_cost": float(pair_results[row_index][col_index]["fit_cost"]),
                    "correlation": float(pair_results[row_index][col_index]["correlation"]),
                    "wavelength_shift_nm": float(pair_results[row_index][col_index]["wavelength_shift_nm"]),
                }
                for col_index in range(len(model_order))
            ),
            key=lambda item: item["fit_cost"],
        )
        candidate_rankings.append(
            {
                "sorted_curve_index": int(row_index),
                "control_curve_index": actual_control_curve_index,
                "voltage": float(voltage_value),
                "top_candidates": ranking[: min(5, len(ranking))],
            }
        )

    mean_fit_cost = float(np.mean(fit_costs))
    mean_correlation = float(np.mean(correlations))
    max_abs_delta_nm = float(
        np.max(np.abs([float(item["wavelength_shift_nm"]) for item in selected_results]))
    )

    return {
        "parameter_name": parameter_name,
        "accepted": bool(mean_fit_cost <= acceptance_cost and mean_correlation >= min_correlation),
        "orientation": orientation,
        "channel_zero_voltage": float(control_result.get("zero_voltage", 0.0)),
        "parameter_zero_value": _resolve_parameter_zero_value(parameter_name, parameter_result),
        "voltage_samples": [float(value) for value in voltage_sorted],
        "matched_parameter_samples": [float(value) for value in matched_parameter_sorted],
        "monotone_parameter_samples": [float(value) for value in monotone_parameter_sorted],
        "mapping_voltage_grid": [float(value) for value in mapping_voltage_grid],
        "mapping_parameter_grid": [float(value) for value in mapping_parameter_grid],
        "mean_fit_cost": mean_fit_cost,
        "mean_correlation": mean_correlation,
        "max_abs_delta_nm": max_abs_delta_nm,
        "curve_matches": curve_matches,
        "candidate_rankings": candidate_rankings,
    }


def _fit_curve_pair(
    control_wavelength_nm: NDArray[np.float64],
    control_signal: NDArray[np.float64],
    model_wavelength_nm: NDArray[np.float64],
    model_signal: NDArray[np.float64],
    *,
    max_delta_nm: float,
    fit_loss: str,
    fit_f_scale: float | None,
) -> dict[str, Any]:
    if max_delta_nm < 0.0:
        raise ValueError("max_delta_nm must be non-negative.")

    lower_bound = max(float(control_wavelength_nm[0]), float(model_wavelength_nm[0]) + max_delta_nm)
    upper_bound = min(float(control_wavelength_nm[-1]), float(model_wavelength_nm[-1]) - max_delta_nm)
    comparison_mask = (control_wavelength_nm >= lower_bound) & (control_wavelength_nm <= upper_bound)
    comparison_wavelength = np.asarray(control_wavelength_nm[comparison_mask], dtype=float)
    control_segment = np.asarray(control_signal[comparison_mask], dtype=float)
    minimum_points = max(16, min(control_wavelength_nm.size, model_wavelength_nm.size) // 2)

    if comparison_wavelength.size < minimum_points:
        return _invalid_fit_result(lower_bound, upper_bound)

    delta_initial = _estimate_initial_shift_nm(
        comparison_wavelength,
        control_segment,
        model_wavelength_nm,
        model_signal,
        max_delta_nm=max_delta_nm,
    )
    delta_initial = float(np.clip(delta_initial, -max_delta_nm, max_delta_nm))
    initial_model_segment = np.interp(
        comparison_wavelength + delta_initial,
        model_wavelength_nm,
        model_signal,
    )
    scale_initial, offset_initial = _fit_affine(initial_model_segment, control_segment)
    residual_scale = (
        float(fit_f_scale)
        if fit_f_scale is not None
        else max(float(np.ptp(control_segment)) * 0.05, 1e-6)
    )

    def residual(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        scale, offset, delta = parameters
        shifted_model = np.interp(
            comparison_wavelength + float(delta),
            model_wavelength_nm,
            model_signal,
        )
        return control_segment - (float(scale) * shifted_model + float(offset))

    result = least_squares(
        residual,
        x0=np.asarray([scale_initial, offset_initial, delta_initial], dtype=float),
        bounds=(
            np.asarray([0.0, -np.inf, -max_delta_nm], dtype=float),
            np.asarray([np.inf, np.inf, max_delta_nm], dtype=float),
        ),
        method="trf",
        loss=fit_loss,
        f_scale=residual_scale,
    )

    scale = float(result.x[0])
    offset = float(result.x[1])
    delta = float(result.x[2])
    shifted_model = np.interp(
        comparison_wavelength + delta,
        model_wavelength_nm,
        model_signal,
    )
    fitted_signal = scale * shifted_model + offset
    residual_vector = control_segment - fitted_signal
    rmse = float(np.sqrt(np.mean(np.square(residual_vector))))
    normalization = max(float(np.ptp(control_segment)), 1e-6)
    correlation = _pearson_correlation(control_segment, fitted_signal)

    return {
        "valid": True,
        "fit_cost": rmse / normalization,
        "rmse": rmse,
        "correlation": correlation,
        "scale": scale,
        "offset": offset,
        "wavelength_shift_nm": delta,
        "overlap_start_nm": float(comparison_wavelength[0]),
        "overlap_stop_nm": float(comparison_wavelength[-1]),
    }


def _estimate_initial_shift_nm(
    comparison_wavelength_nm: NDArray[np.float64],
    control_segment: NDArray[np.float64],
    model_wavelength_nm: NDArray[np.float64],
    model_signal: NDArray[np.float64],
    *,
    max_delta_nm: float,
) -> float:
    if comparison_wavelength_nm.size < 2 or max_delta_nm <= 0.0:
        return 0.0

    model_segment = np.interp(
        comparison_wavelength_nm,
        model_wavelength_nm,
        model_signal,
    )
    control_normalized = _normalize_for_correlation(control_segment)
    model_normalized = _normalize_for_correlation(model_segment)
    if np.allclose(control_normalized, 0.0) or np.allclose(model_normalized, 0.0):
        return 0.0

    correlation = correlate(control_normalized, model_normalized, mode="full", method="auto")
    lags = correlation_lags(control_normalized.size, model_normalized.size, mode="full")
    step_nm = float(np.median(np.diff(comparison_wavelength_nm)))
    if step_nm <= 0.0:
        return 0.0

    max_lag = max(1, int(round(max_delta_nm / step_nm)))
    lag_mask = np.abs(lags) <= max_lag
    if not np.any(lag_mask):
        return 0.0

    candidate_lags = lags[lag_mask]
    candidate_corr = correlation[lag_mask]
    best_lag = int(candidate_lags[int(np.argmax(candidate_corr))])
    return float(best_lag * step_nm)


def _prepare_curve_signal(
    bundle: CurveSweepBundle,
    curve_index: int,
    *,
    smooth_window: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    wavelength_nm = np.asarray(bundle.wavelength_nm, dtype=float)
    signal = np.asarray(np.abs(bundle.complex_response_matrix[curve_index]), dtype=float)
    finite_mask = np.isfinite(wavelength_nm) & np.isfinite(signal)
    wavelength_nm = wavelength_nm[finite_mask]
    signal = signal[finite_mask]

    if wavelength_nm.size < 2:
        raise ValueError(f"Curve {bundle.name}[{curve_index}] does not contain enough valid points.")

    order = np.argsort(wavelength_nm)
    wavelength_nm = wavelength_nm[order]
    signal = signal[order]

    if smooth_window > 2:
        window = min(int(smooth_window), int(signal.size))
        if window % 2 == 0:
            window -= 1
        if window >= 3:
            polyorder = min(2, window - 1)
            signal = savgol_filter(signal, window_length=window, polyorder=polyorder, mode="interp")

    return wavelength_nm, signal


def _solve_monotone_path(
    cost_matrix: NDArray[np.float64],
    *,
    increasing: bool,
) -> dict[str, Any]:
    if cost_matrix.ndim != 2:
        raise ValueError("cost_matrix must be 2D.")
    rows, cols = cost_matrix.shape
    if rows == 0 or cols == 0:
        raise ValueError("cost_matrix must be non-empty.")

    if not increasing:
        reversed_solution = _solve_monotone_path(cost_matrix[:, ::-1], increasing=True)
        reversed_path = np.asarray(reversed_solution["path"], dtype=int)
        return {
            "orientation": "decreasing",
            "mean_cost": float(reversed_solution["mean_cost"]),
            "path": [int(cols - 1 - value) for value in reversed_path],
        }

    dp = np.full((rows, cols), np.inf, dtype=float)
    previous = np.full((rows, cols), -1, dtype=int)
    dp[0, :] = cost_matrix[0, :]

    for row in range(1, rows):
        prefix_cost = np.empty(cols, dtype=float)
        prefix_index = np.empty(cols, dtype=int)
        best_cost = np.inf
        best_index = 0
        for col in range(cols):
            candidate_cost = dp[row - 1, col]
            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_index = col
            prefix_cost[col] = best_cost
            prefix_index[col] = best_index

        dp[row, :] = cost_matrix[row, :] + prefix_cost
        previous[row, :] = prefix_index

    end_col = int(np.argmin(dp[-1, :]))
    path = np.empty(rows, dtype=int)
    path[-1] = end_col
    for row in range(rows - 1, 0, -1):
        path[row - 1] = previous[row, path[row]]

    return {
        "orientation": "increasing",
        "mean_cost": float(dp[-1, end_col] / rows),
        "path": [int(value) for value in path],
    }


def _weighted_isotonic_regression(
    y_values: Sequence[float] | NDArray[np.float64],
    *,
    weights: Sequence[float] | NDArray[np.float64],
    increasing: bool,
) -> NDArray[np.float64]:
    y = np.asarray(y_values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if y.ndim != 1 or w.ndim != 1 or y.size != w.size:
        raise ValueError("y_values and weights must be 1D arrays with the same length.")
    if y.size == 0:
        return y.copy()
    if np.any(w <= 0.0):
        raise ValueError("Isotonic-regression weights must be positive.")
    if not increasing:
        return -_weighted_isotonic_regression(-y, weights=w, increasing=True)

    blocks: list[dict[str, float | int]] = []
    for index, (value, weight) in enumerate(zip(y, w, strict=True)):
        blocks.append(
            {
                "start": int(index),
                "end": int(index),
                "weight": float(weight),
                "value": float(value),
            }
        )
        while len(blocks) >= 2 and float(blocks[-2]["value"]) > float(blocks[-1]["value"]):
            right = blocks.pop()
            left = blocks.pop()
            merged_weight = float(left["weight"]) + float(right["weight"])
            merged_value = (
                float(left["value"]) * float(left["weight"]) + float(right["value"]) * float(right["weight"])
            ) / merged_weight
            blocks.append(
                {
                    "start": int(left["start"]),
                    "end": int(right["end"]),
                    "weight": merged_weight,
                    "value": merged_value,
                }
            )

    output = np.empty_like(y)
    for block in blocks:
        output[int(block["start"]) : int(block["end"]) + 1] = float(block["value"])
    return output


def _merge_duplicate_x(
    x_values: NDArray[np.float64],
    y_values: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if x_values.size != y_values.size or x_values.size != weights.size:
        raise ValueError("x_values, y_values, and weights must have the same length.")

    unique_x: list[float] = []
    unique_y: list[float] = []
    start = 0
    while start < x_values.size:
        end = start + 1
        while end < x_values.size and np.isclose(x_values[end], x_values[start]):
            end += 1
        group_weights = weights[start:end]
        group_values = y_values[start:end]
        unique_x.append(float(x_values[start]))
        unique_y.append(float(np.average(group_values, weights=group_weights)))
        start = end

    return np.asarray(unique_x, dtype=float), np.asarray(unique_y, dtype=float)


def _fit_affine(
    x_values: NDArray[np.float64],
    y_values: NDArray[np.float64],
) -> tuple[float, float]:
    design = np.column_stack([x_values, np.ones_like(x_values)])
    coefficients, *_ = np.linalg.lstsq(design, y_values, rcond=None)
    scale = max(float(coefficients[0]), 0.0)
    offset = float(coefficients[1])
    return scale, offset


def _normalize_for_correlation(values: NDArray[np.float64]) -> NDArray[np.float64]:
    centered = np.asarray(values, dtype=float) - float(np.mean(values))
    scale = float(np.std(centered))
    if scale <= 1e-12:
        return np.zeros_like(centered)
    return centered / scale


def _pearson_correlation(
    x_values: NDArray[np.float64],
    y_values: NDArray[np.float64],
) -> float:
    if x_values.size < 2 or y_values.size < 2:
        return 0.0
    x_normalized = _normalize_for_correlation(x_values)
    y_normalized = _normalize_for_correlation(y_values)
    if np.allclose(x_normalized, 0.0) or np.allclose(y_normalized, 0.0):
        return 1.0 if np.allclose(x_values, y_values) else 0.0
    return float(np.mean(x_normalized * y_normalized))


def _invalid_fit_result(
    overlap_start_nm: float,
    overlap_stop_nm: float,
) -> dict[str, Any]:
    return {
        "valid": False,
        "fit_cost": 1e6,
        "rmse": 1e6,
        "correlation": 0.0,
        "scale": 0.0,
        "offset": 0.0,
        "wavelength_shift_nm": 0.0,
        "overlap_start_nm": float(overlap_start_nm),
        "overlap_stop_nm": float(overlap_stop_nm),
    }


def _resolve_parameter_order(model_payload: Mapping[str, Any]) -> Sequence[str]:
    explicit_order = model_payload.get("parameter_order")
    if isinstance(explicit_order, Sequence) and not isinstance(explicit_order, (str, bytes, bytearray)):
        return [str(value) for value in explicit_order]
    return list(model_payload["results"].keys())


def _resolve_channel_order(control_payload: Mapping[str, Any]) -> Sequence[str]:
    explicit_order = control_payload.get("channel_order")
    if isinstance(explicit_order, Sequence) and not isinstance(explicit_order, (str, bytes, bytearray)):
        return [_normalize_channel_name(value) for value in explicit_order]
    return [_normalize_channel_name(value) for value in control_payload["results"].keys()]


def _resolve_channel_parameter_map(
    raw_config: Mapping[str, Any],
    *,
    channel_parameter_map: Mapping[str | int, str] | None,
    channel_names: Sequence[str],
    parameter_names: Sequence[str],
) -> dict[str, str]:
    raw_mapping = channel_parameter_map
    if raw_mapping is None:
        calibration_section = get_mapping(raw_config, "calibration")
        voltage_to_tunable = get_mapping(calibration_section, "voltage_to_tunable")
        candidate = voltage_to_tunable.get("channel_to_parameter")
        if candidate is None:
            candidate = calibration_section.get("channel_to_parameter")
        if isinstance(candidate, Mapping):
            raw_mapping = candidate

    if raw_mapping is None:
        return {}

    resolved: dict[str, str] = {}
    used_parameters: set[str] = set()
    valid_channels = set(channel_names)
    valid_parameters = set(parameter_names)
    for raw_channel, raw_parameter in raw_mapping.items():
        channel_name = _normalize_channel_name(raw_channel)
        parameter_name = str(raw_parameter).strip()
        if channel_name not in valid_channels:
            raise ValueError(f"Unknown control channel in channel-to-parameter map: {raw_channel!r}.")
        if parameter_name not in valid_parameters:
            raise ValueError(f"Unknown tunable parameter in channel-to-parameter map: {raw_parameter!r}.")
        if parameter_name in used_parameters:
            raise ValueError(
                "Each tunable parameter can only be assigned once in channel-to-parameter map. "
                f"Duplicate parameter: {parameter_name}."
            )
        resolved[channel_name] = parameter_name
        used_parameters.add(parameter_name)
    return resolved


def _normalize_channel_name(value: Any) -> str:
    text = str(value).strip()
    if text.lower().startswith("ch"):
        text = text[2:]
    return str(int(text))


def _load_curve_bundle_from_result(
    archive: Mapping[str, Any],
    result_entry: Mapping[str, Any],
    *,
    fallback_prefix: str,
    bundle_name: str,
    metadata: Mapping[str, Any] | None = None,
) -> CurveSweepBundle:
    prefix = str(result_entry.get("curve_archive_prefix") or fallback_prefix)
    required_keys = (
        f"{prefix}__sweep_values",
        f"{prefix}__wavelength_nm",
        f"{prefix}__power_db",
    )
    missing = [key for key in required_keys if key not in archive]
    if missing:
        raise ValueError(f"Curve archive is missing required arrays for prefix {prefix!r}: {missing}.")

    sweep_values = np.asarray(archive[f"{prefix}__sweep_values"], dtype=float)
    wavelength_nm = np.asarray(archive[f"{prefix}__wavelength_nm"], dtype=float)
    power_db_matrix = np.asarray(archive[f"{prefix}__power_db"], dtype=float)

    real_key = f"{prefix}__complex_real"
    imag_key = f"{prefix}__complex_imag"
    if real_key in archive and imag_key in archive:
        complex_response_matrix = np.asarray(archive[real_key], dtype=float) + 1j * np.asarray(
            archive[imag_key],
            dtype=float,
        )
    else:
        complex_response_matrix = power_db_to_complex_amplitude(power_db_matrix)

    return CurveSweepBundle(
        name=bundle_name,
        sweep_values=sweep_values,
        wavelength_nm=wavelength_nm,
        complex_response_matrix=np.asarray(complex_response_matrix, dtype=np.complex128),
        power_db_matrix=power_db_matrix,
        metadata=dict(metadata or {}),
    )


def _resolve_parameter_zero_value(parameter_name: str, parameter_result: Mapping[str, Any]) -> float:
    if "zero_tunable_value" in parameter_result:
        return float(parameter_result["zero_tunable_value"])
    raise ValueError(f"Model calibration result for {parameter_name!r} is missing zero_tunable_value.")


def _resolve_archive_path(
    payload: Mapping[str, Any],
    json_path: Path,
    *,
    default_filename: str,
) -> Path:
    archive_path = payload.get("curve_archive")
    if archive_path is None:
        return json_path.with_name(default_filename).resolve()
    return Path(str(archive_path)).resolve()


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).resolve().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _parse_channel_parameter_map(values: Sequence[str] | None) -> dict[str, str] | None:
    if values is None:
        return None

    parsed: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(
                "Channel-to-parameter mappings must use CHANNEL=PARAMETER syntax, "
                f"but got {item!r}."
            )
        channel_text, parameter_text = item.split("=", 1)
        parsed[_normalize_channel_name(channel_text)] = parameter_text.strip()
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Match control curves to model curves by shape and solve voltage-to-parameter mappings."
    )
    parser.add_argument("config", type=Path, help="Path to the model YAML config.")
    parser.add_argument("--model-calibration", type=Path, default=None)
    parser.add_argument("--control-calibration", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--channel-parameter-map",
        nargs="+",
        default=None,
        metavar="CHANNEL=PARAMETER",
        help="Explicit channel-to-parameter relation, for example `--channel-parameter-map 1=fai1 2=theta1`.",
    )
    parser.add_argument("--mapping-grid-points", "--grid-points", dest="grid_points", type=int, default=256)
    parser.add_argument("--acceptance-cost", type=float, default=0.2)
    parser.add_argument("--min-correlation", type=float, default=0.85)
    parser.add_argument("--max-delta-nm", type=float, default=0.05)
    parser.add_argument("--smooth-window", type=int, default=7)
    parser.add_argument("--fit-loss", type=str, default="soft_l1")
    parser.add_argument("--fit-f-scale", type=float, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    payload = run_voltage_to_tunable_calibration(
        args.config,
        model_calibration_path=args.model_calibration,
        control_calibration_path=args.control_calibration,
        output_dir=args.output_dir,
        channel_parameter_map=_parse_channel_parameter_map(args.channel_parameter_map),
        grid_points=args.grid_points,
        acceptance_cost=args.acceptance_cost,
        min_correlation=args.min_correlation,
        max_delta_nm=args.max_delta_nm,
        smooth_window=args.smooth_window,
        fit_loss=args.fit_loss,
        fit_f_scale=args.fit_f_scale,
    )
    print(f"saved_json: {payload['json_path']}")
    print(f"mapping_count: {len(payload['mappings'])}")
    return 0


__all__ = [
    "collect_control_targets",
    "collect_model_reference",
    "run_voltage_to_tunable_calibration",
]


if __name__ == "__main__":
    raise SystemExit(main())
