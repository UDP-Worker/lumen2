from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.signal import fftconvolve, savgol_filter

from backend.calibrate import calibrate as voltage_calibration
from backend.calibrate._shared import ensure_output_dir, load_raw_yaml, resolve_model_name, sanitize_key
from backend.model import load_model_config


DEFAULT_FEATURES = (
    "relative_db",
    "clip_6db",
    "clip_10db",
    "rank_clip_10db",
    "slope_clip_10db",
    "abs_slope_clip_10db",
    "valley_35pct",
    "highpass_0.12nm",
)


@dataclass(frozen=True)
class MethodSpec:
    variant: str
    feature: str
    allow_inverted: bool

    @property
    def name(self) -> str:
        suffix = "__abs" if self.allow_inverted else ""
        return f"{self.variant}__{self.feature}{suffix}"


@dataclass
class ChannelDiagnostics:
    channel: str
    parameter: str
    orientation: str
    mean_score: float
    median_score: float
    min_score: float
    edge_shift_fraction: float
    unique_parameter_count: int
    parameter_min: float
    parameter_max: float
    control_voltage: NDArray[np.float64]
    matched_parameter: NDArray[np.float64]
    matched_model_index: NDArray[np.int64]
    matched_shift_nm: NDArray[np.float64]
    matched_score: NDArray[np.float64]


def run_alignment_diagnostics(
    config_path: str | Path,
    *,
    model_calibration_path: str | Path | None = None,
    control_calibration_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    channel_parameter_map: Mapping[str | int, str] | None = None,
    max_delta_nm: float = 0.05,
    shift_count: int = 31,
    include_abs_correlation: bool = True,
    example_count: int = 5,
    top_methods: int = 3,
) -> dict[str, Any]:
    raw_config = load_raw_yaml(config_path)
    resolved_config = load_model_config(config_path)
    model_name = resolve_model_name(raw_config, resolved_config)
    data_dir = ensure_output_dir(model_name)
    destination_dir = Path(output_dir).resolve() if output_dir is not None else data_dir / "alignment_diagnostics"
    destination_dir.mkdir(parents=True, exist_ok=True)

    model_json = Path(model_calibration_path).resolve() if model_calibration_path else data_dir / "model_calibration.json"
    control_json = (
        Path(control_calibration_path).resolve() if control_calibration_path else data_dir / "control_calibration.json"
    )
    model_payload = _load_json(model_json)
    control_payload = _load_json(control_json)
    model_archive_path = _resolve_archive_path(model_payload, model_json, "model_calibration_curves.npz")
    control_archive_path = _resolve_archive_path(control_payload, control_json, "control_calibration_curves.npz")

    parameter_names = _resolve_parameter_order(model_payload)
    channel_names = _resolve_channel_order(control_payload)
    channel_to_parameter = _resolve_channel_parameter_map(
        raw_config,
        channel_parameter_map=channel_parameter_map,
        channel_names=channel_names,
        parameter_names=parameter_names,
    )
    if not channel_to_parameter:
        raise ValueError("No channel-to-parameter mapping was found.")

    control_osa = control_payload.get("settings", {}).get("osa_settings", {})
    osa_resolution_nm = float(control_osa.get("resolution_nm", 0.02)) if isinstance(control_osa, Mapping) else 0.02
    variants = _build_variant_specs(osa_resolution_nm)
    methods = [
        MethodSpec(variant=variant, feature=feature, allow_inverted=allow_inverted)
        for variant in variants
        for feature in DEFAULT_FEATURES
        for allow_inverted in ((False, True) if include_abs_correlation else (False,))
    ]

    shifts_nm = np.linspace(-float(max_delta_nm), float(max_delta_nm), int(shift_count))
    model_archive = np.load(model_archive_path)
    control_archive = np.load(control_archive_path)
    try:
        model_cache = _build_model_variant_cache(
            model_archive,
            parameter_names=parameter_names,
            variants=variants,
            osa_resolution_nm=osa_resolution_nm,
        )
        method_results: dict[str, dict[str, ChannelDiagnostics]] = {}
        method_summaries: list[dict[str, Any]] = []

        for method in methods:
            channel_results: dict[str, ChannelDiagnostics] = {}
            for channel, parameter in sorted(channel_to_parameter.items(), key=lambda item: int(item[0])):
                diagnostics = _evaluate_channel(
                    control_archive=control_archive,
                    model_cache=model_cache,
                    channel=channel,
                    parameter=parameter,
                    method=method,
                    shifts_nm=shifts_nm,
                )
                channel_results[channel] = diagnostics

            method_results[method.name] = channel_results
            summary = _summarize_method(method, channel_results)
            method_summaries.append(summary)

        method_summaries.sort(key=_method_sort_key, reverse=True)
        best_method_name = str(method_summaries[0]["method"])

        _write_summary_csv(destination_dir / "method_summary.csv", method_summaries)
        _plot_method_summary(destination_dir / "method_summary.png", method_summaries)
        for summary in method_summaries[: max(1, int(top_methods))]:
            method_name = str(summary["method"])
            _plot_best_match_grid(
                destination_dir / f"best_matches__{sanitize_key(method_name)}.png",
                method_name=method_name,
                channel_results=method_results[method_name],
                control_archive=control_archive,
                model_cache=model_cache,
                example_count=example_count,
            )
            _plot_mapping_path(
                destination_dir / f"mapping_path__{sanitize_key(method_name)}.png",
                method_name=method_name,
                channel_results=method_results[method_name],
            )

        payload = {
            "config_path": str(Path(config_path).resolve()),
            "model_calibration_path": str(model_json),
            "control_calibration_path": str(control_json),
            "model_curve_archive": str(model_archive_path),
            "control_curve_archive": str(control_archive_path),
            "channel_to_parameter": channel_to_parameter,
            "max_delta_nm": float(max_delta_nm),
            "shift_count": int(shift_count),
            "osa_resolution_nm": float(osa_resolution_nm),
            "best_method": best_method_name,
            "method_summaries": method_summaries,
            "outputs": {
                "summary_csv": str((destination_dir / "method_summary.csv").resolve()),
                "summary_plot": str((destination_dir / "method_summary.png").resolve()),
                "best_match_plot": str(
                    (destination_dir / f"best_matches__{sanitize_key(best_method_name)}.png").resolve()
                ),
                "mapping_path_plot": str(
                    (destination_dir / f"mapping_path__{sanitize_key(best_method_name)}.png").resolve()
                ),
            },
        }
        json_path = destination_dir / "alignment_diagnostics.json"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        payload["json_path"] = str(json_path.resolve())
        return payload
    finally:
        model_archive.close()
        control_archive.close()


def _evaluate_channel(
    *,
    control_archive: np.lib.npyio.NpzFile,
    model_cache: Mapping[str, Mapping[str, Any]],
    channel: str,
    parameter: str,
    method: MethodSpec,
    shifts_nm: NDArray[np.float64],
) -> ChannelDiagnostics:
    control_prefix = sanitize_key(f"ch{channel}")
    control_wavelength = np.asarray(control_archive[f"{control_prefix}__wavelength_nm"], dtype=float)
    control_power_db = np.asarray(control_archive[f"{control_prefix}__power_db"], dtype=float)
    control_voltage = np.asarray(control_archive[f"{control_prefix}__sweep_values"], dtype=float)

    model_item = model_cache[parameter]
    model_wavelength = np.asarray(model_item["wavelength_nm"], dtype=float)
    model_power_db = np.asarray(model_item["variants"][method.variant], dtype=float)
    model_parameter = np.asarray(model_item["sweep_values"], dtype=float)

    control_feature = np.vstack(
        [_normalize_rows(_feature(control_wavelength, row, method.feature)[None, :])[0] for row in control_power_db]
    )
    model_feature = np.vstack(
        [_normalize_rows(_feature(model_wavelength, row, method.feature)[None, :])[0] for row in model_power_db]
    )

    score_matrix, shift_matrix = _best_shifted_scores(
        control_wavelength=control_wavelength,
        control_feature=control_feature,
        model_wavelength=model_wavelength,
        model_feature=model_feature,
        shifts_nm=shifts_nm,
        allow_inverted=method.allow_inverted,
    )
    cost_matrix = 1.0 - score_matrix
    increasing = voltage_calibration._solve_monotone_path(cost_matrix, increasing=True)
    decreasing = voltage_calibration._solve_monotone_path(cost_matrix, increasing=False)
    chosen = increasing if increasing["mean_cost"] <= decreasing["mean_cost"] else decreasing
    path = np.asarray(chosen["path"], dtype=int)
    rows = np.arange(path.size)
    matched_score = score_matrix[rows, path]
    matched_shift = shift_matrix[rows, path]
    matched_parameter = model_parameter[path]

    return ChannelDiagnostics(
        channel=str(channel),
        parameter=str(parameter),
        orientation=str(chosen["orientation"]),
        mean_score=float(np.mean(matched_score)),
        median_score=float(np.median(matched_score)),
        min_score=float(np.min(matched_score)),
        edge_shift_fraction=float(np.mean(np.isclose(np.abs(matched_shift), np.max(np.abs(shifts_nm))))),
        unique_parameter_count=int(np.unique(np.round(matched_parameter, 9)).size),
        parameter_min=float(np.min(matched_parameter)),
        parameter_max=float(np.max(matched_parameter)),
        control_voltage=control_voltage,
        matched_parameter=matched_parameter,
        matched_model_index=path,
        matched_shift_nm=matched_shift,
        matched_score=matched_score,
    )


def _best_shifted_scores(
    *,
    control_wavelength: NDArray[np.float64],
    control_feature: NDArray[np.float64],
    model_wavelength: NDArray[np.float64],
    model_feature: NDArray[np.float64],
    shifts_nm: NDArray[np.float64],
    allow_inverted: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    control_count = control_feature.shape[0]
    model_count = model_feature.shape[0]
    best_score = np.full((control_count, model_count), -np.inf, dtype=float)
    best_shift = np.zeros((control_count, model_count), dtype=float)

    for shift_nm in shifts_nm:
        sample_wavelength = control_wavelength + float(shift_nm)
        valid = (sample_wavelength >= model_wavelength[0]) & (sample_wavelength <= model_wavelength[-1])
        if int(np.sum(valid)) < max(16, control_wavelength.size // 2):
            continue

        shifted_model = np.vstack(
            [
                np.interp(sample_wavelength[valid], model_wavelength, model_feature[index])
                for index in range(model_count)
            ]
        )
        shifted_model = _normalize_rows(shifted_model)
        control_window = _normalize_rows(control_feature[:, valid])
        scores = control_window @ shifted_model.T / float(np.sum(valid))
        if allow_inverted:
            scores = np.abs(scores)

        improved = scores > best_score
        best_score[improved] = scores[improved]
        best_shift[improved] = float(shift_nm)

    best_score[~np.isfinite(best_score)] = 0.0
    return best_score, best_shift


def _build_model_variant_cache(
    archive: np.lib.npyio.NpzFile,
    *,
    parameter_names: Sequence[str],
    variants: Sequence[str],
    osa_resolution_nm: float,
) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    for parameter in parameter_names:
        prefix = sanitize_key(parameter)
        wavelength = np.asarray(archive[f"{prefix}__wavelength_nm"], dtype=float)
        power_db = np.asarray(archive[f"{prefix}__power_db"], dtype=float)
        relative_db = power_db - np.nanmax(power_db, axis=1, keepdims=True)
        variant_arrays: dict[str, NDArray[np.float64]] = {}
        for variant in variants:
            variant_arrays[variant] = _model_variant(
                wavelength,
                relative_db,
                variant=variant,
                osa_resolution_nm=osa_resolution_nm,
            )
        cache[parameter] = {
            "wavelength_nm": wavelength,
            "sweep_values": np.asarray(archive[f"{prefix}__sweep_values"], dtype=float),
            "variants": variant_arrays,
        }
    return cache


def _model_variant(
    wavelength_nm: NDArray[np.float64],
    relative_db: NDArray[np.float64],
    *,
    variant: str,
    osa_resolution_nm: float,
) -> NDArray[np.float64]:
    if variant == "raw":
        return np.asarray(relative_db, dtype=float)

    if not variant.startswith("osa"):
        raise ValueError(f"Unknown model variant: {variant}")

    convolved = _convolve_linear_power(wavelength_nm, relative_db, fwhm_nm=osa_resolution_nm)
    if variant == "osa":
        return convolved

    floor_prefix = "osa_floor"
    if variant.startswith(floor_prefix):
        floor_db = float(variant[len(floor_prefix) :].replace("db", ""))
        return _add_extinction_floor(convolved, floor_db=floor_db)

    raise ValueError(f"Unknown model variant: {variant}")


def _convolve_linear_power(
    wavelength_nm: NDArray[np.float64],
    relative_db: NDArray[np.float64],
    *,
    fwhm_nm: float,
) -> NDArray[np.float64]:
    step_nm = float(np.median(np.diff(wavelength_nm)))
    if step_nm <= 0.0 or fwhm_nm <= step_nm:
        return np.asarray(relative_db, dtype=float)

    sigma_nm = fwhm_nm / 2.354820045
    half_width = max(3, int(np.ceil(5.0 * sigma_nm / step_nm)))
    offsets = np.arange(-half_width, half_width + 1, dtype=float) * step_nm
    kernel = np.exp(-0.5 * np.square(offsets / sigma_nm))
    kernel /= float(np.sum(kernel))
    pad = kernel.size // 2

    linear_power = np.power(10.0, relative_db / 10.0)
    convolved_rows: list[NDArray[np.float64]] = []
    for row in linear_power:
        padded = np.pad(row, (pad, pad), mode="edge")
        filtered = fftconvolve(padded, kernel, mode="same")[pad:-pad]
        convolved_rows.append(10.0 * np.log10(np.maximum(filtered, 1e-300)))
    output = np.vstack(convolved_rows)
    return output - np.nanmax(output, axis=1, keepdims=True)


def _add_extinction_floor(relative_db: NDArray[np.float64], *, floor_db: float) -> NDArray[np.float64]:
    linear = np.power(10.0, relative_db / 10.0)
    floor = np.power(10.0, -float(floor_db) / 10.0)
    with_floor = floor + (1.0 - floor) * linear
    output = 10.0 * np.log10(np.maximum(with_floor, 1e-300))
    return output - np.nanmax(output, axis=1, keepdims=True)


def _feature(
    wavelength_nm: NDArray[np.float64],
    power_db: NDArray[np.float64],
    feature_name: str,
) -> NDArray[np.float64]:
    relative_db = np.asarray(power_db, dtype=float) - float(np.nanmax(power_db))
    if feature_name == "relative_db":
        return _smooth_nm(wavelength_nm, relative_db, width_nm=0.02)
    if feature_name == "clip_6db":
        return _smooth_nm(wavelength_nm, np.clip(relative_db, -6.0, 0.0), width_nm=0.02)
    if feature_name == "clip_10db":
        return _smooth_nm(wavelength_nm, np.clip(relative_db, -10.0, 0.0), width_nm=0.02)
    if feature_name == "rank_clip_10db":
        return _rank01(np.clip(relative_db, -10.0, 0.0))
    if feature_name == "slope_clip_10db":
        clipped = _smooth_nm(wavelength_nm, np.clip(relative_db, -10.0, 0.0), width_nm=0.02)
        return np.gradient(clipped, wavelength_nm)
    if feature_name == "abs_slope_clip_10db":
        clipped = _smooth_nm(wavelength_nm, np.clip(relative_db, -10.0, 0.0), width_nm=0.02)
        return np.abs(np.gradient(clipped, wavelength_nm))
    if feature_name == "valley_35pct":
        threshold = float(np.nanquantile(relative_db, 0.35))
        return _smooth_nm(wavelength_nm, np.maximum(threshold - relative_db, 0.0), width_nm=0.02)
    if feature_name == "highpass_0.12nm":
        return relative_db - _smooth_nm(wavelength_nm, relative_db, width_nm=0.12)
    raise ValueError(f"Unknown feature: {feature_name}")


def _smooth_nm(
    wavelength_nm: NDArray[np.float64],
    values: NDArray[np.float64],
    *,
    width_nm: float,
) -> NDArray[np.float64]:
    if values.size < 5:
        return np.asarray(values, dtype=float)
    step_nm = float(np.median(np.diff(wavelength_nm)))
    if step_nm <= 0.0:
        return np.asarray(values, dtype=float)
    window = max(5, int(round(float(width_nm) / step_nm)))
    window = min(window, values.size)
    if window % 2 == 0:
        window -= 1
    if window < 5:
        return np.asarray(values, dtype=float)
    return savgol_filter(values, window_length=window, polyorder=min(2, window - 1), mode="interp")


def _normalize_rows(values: NDArray[np.float64]) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    centered = array - np.mean(array, axis=1, keepdims=True)
    scale = np.std(centered, axis=1, keepdims=True)
    return np.divide(centered, scale, out=np.zeros_like(centered), where=scale > 1e-12)


def _rank01(values: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(values)
    ranks = np.empty_like(values, dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, values.size)
    return ranks


def _build_variant_specs(osa_resolution_nm: float) -> tuple[str, ...]:
    # The names are intentionally compact because they become plot filenames.
    return ("raw", "osa", "osa_floor6db", "osa_floor10db", "osa_floor20db")


def _summarize_method(
    method: MethodSpec,
    channel_results: Mapping[str, ChannelDiagnostics],
) -> dict[str, Any]:
    rows = list(channel_results.values())
    mean_score = float(np.mean([row.mean_score for row in rows]))
    min_channel_score = float(np.min([row.mean_score for row in rows]))
    edge_shift_fraction = float(np.mean([row.edge_shift_fraction for row in rows]))
    unique_parameter_count = float(np.mean([row.unique_parameter_count for row in rows]))
    composite_score = mean_score - 0.10 * edge_shift_fraction + 0.02 * min(unique_parameter_count / 5.0, 1.0)
    return {
        "method": method.name,
        "variant": method.variant,
        "feature": method.feature,
        "allow_inverted": method.allow_inverted,
        "mean_score": mean_score,
        "min_channel_mean_score": min_channel_score,
        "edge_shift_fraction": edge_shift_fraction,
        "mean_unique_parameter_count": unique_parameter_count,
        "composite_score": float(composite_score),
        "channels": {
            row.channel: {
                "parameter": row.parameter,
                "orientation": row.orientation,
                "mean_score": row.mean_score,
                "median_score": row.median_score,
                "min_score": row.min_score,
                "edge_shift_fraction": row.edge_shift_fraction,
                "unique_parameter_count": row.unique_parameter_count,
                "parameter_range": [row.parameter_min, row.parameter_max],
            }
            for row in rows
        },
    }


def _method_sort_key(summary: Mapping[str, Any]) -> tuple[float, float, float]:
    return (
        float(summary["composite_score"]),
        float(summary["mean_score"]),
        -float(summary["edge_shift_fraction"]),
    )


def _plot_method_summary(path: Path, summaries: Sequence[Mapping[str, Any]]) -> None:
    top = list(summaries[: min(20, len(summaries))])
    labels = [str(item["method"]) for item in top][::-1]
    scores = [float(item["mean_score"]) for item in top][::-1]
    composite = [float(item["composite_score"]) for item in top][::-1]
    fig, ax = plt.subplots(figsize=(12, max(5, 0.35 * len(labels))), constrained_layout=True)
    y = np.arange(len(labels))
    ax.barh(y - 0.18, scores, height=0.35, label="mean score")
    ax.barh(y + 0.18, composite, height=0.35, label="composite")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Score")
    ax.set_title("Alignment Method Summary")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_best_match_grid(
    path: Path,
    *,
    method_name: str,
    channel_results: Mapping[str, ChannelDiagnostics],
    control_archive: np.lib.npyio.NpzFile,
    model_cache: Mapping[str, Mapping[str, Any]],
    example_count: int,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    for ax, channel in zip(axes.flat, sorted(channel_results, key=int)):
        result = channel_results[channel]
        control_prefix = sanitize_key(f"ch{channel}")
        control_wavelength = np.asarray(control_archive[f"{control_prefix}__wavelength_nm"], dtype=float)
        control_power = np.asarray(control_archive[f"{control_prefix}__power_db"], dtype=float)
        method_variant = method_name.split("__", 1)[0]
        model_item = model_cache[result.parameter]
        model_wavelength = np.asarray(model_item["wavelength_nm"], dtype=float)
        model_power = np.asarray(model_item["variants"][method_variant], dtype=float)

        indices = _example_indices(result.control_voltage.size, example_count)
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(indices)))
        for color, curve_index in zip(colors, indices, strict=True):
            measured = control_power[curve_index] - float(np.nanmax(control_power[curve_index]))
            model_index = int(result.matched_model_index[curve_index])
            shift = float(result.matched_shift_nm[curve_index])
            matched = np.interp(control_wavelength + shift, model_wavelength, model_power[model_index])
            matched = matched - float(np.nanmax(matched))
            voltage = float(result.control_voltage[curve_index])
            parameter = float(result.matched_parameter[curve_index])
            score = float(result.matched_score[curve_index])
            ax.plot(control_wavelength, measured, color=color, lw=1.6, alpha=0.8)
            ax.plot(control_wavelength, matched, color=color, lw=1.2, ls="--", alpha=0.8)
            ax.text(
                0.01,
                0.98 - 0.08 * len(ax.texts),
                f"{voltage:.2f}V -> {parameter:.2f}rad, s={score:.2f}",
                transform=ax.transAxes,
                fontsize=7,
                va="top",
            )
        ax.set_title(
            f"ch{channel} -> {result.parameter}\n"
            f"{result.orientation}, mean={result.mean_score:.3f}, edge={result.edge_shift_fraction:.2f}"
        )
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Power rel. max (dB)")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Best Actual/Theory Matches: {method_name}", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_mapping_path(
    path: Path,
    *,
    method_name: str,
    channel_results: Mapping[str, ChannelDiagnostics],
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    for ax, channel in zip(axes.flat, sorted(channel_results, key=int)):
        result = channel_results[channel]
        scatter = ax.scatter(
            result.control_voltage,
            result.matched_parameter,
            c=result.matched_score,
            cmap="viridis",
            s=20,
            vmin=0.0,
            vmax=1.0,
        )
        ax.plot(result.control_voltage, result.matched_parameter, lw=1.0, alpha=0.6)
        ax.set_title(
            f"ch{channel} -> {result.parameter}\n"
            f"mean={result.mean_score:.3f}, unique={result.unique_parameter_count}"
        )
        ax.set_xlabel("Voltage (V)")
        ax.set_ylabel("Matched parameter (rad)")
        ax.grid(True, alpha=0.3)
    fig.colorbar(scatter, ax=axes.ravel().tolist(), label="match score")
    fig.suptitle(f"Matched Monotone Paths: {method_name}", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _example_indices(size: int, count: int) -> list[int]:
    if size <= 0:
        return []
    count = max(1, min(int(count), int(size)))
    return [int(value) for value in np.linspace(0, size - 1, count).round()]


def _write_summary_csv(path: Path, summaries: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "variant",
                "feature",
                "allow_inverted",
                "composite_score",
                "mean_score",
                "min_channel_mean_score",
                "edge_shift_fraction",
                "mean_unique_parameter_count",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary[field] for field in writer.fieldnames})


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).resolve().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _resolve_archive_path(payload: Mapping[str, Any], json_path: Path, default_filename: str) -> Path:
    candidate = payload.get("curve_archive")
    if candidate is None:
        candidate = payload.get("curve_archive_path")
    if candidate is None:
        return json_path.parent / default_filename
    return Path(str(candidate)).resolve()


def _resolve_parameter_order(model_payload: Mapping[str, Any]) -> list[str]:
    order = model_payload.get("parameter_order")
    if isinstance(order, Sequence) and not isinstance(order, (str, bytes, bytearray)):
        return [str(value) for value in order]
    results = model_payload.get("results")
    if not isinstance(results, Mapping):
        raise ValueError("Model calibration JSON must contain results.")
    return [str(value) for value in results]


def _resolve_channel_order(control_payload: Mapping[str, Any]) -> list[str]:
    order = control_payload.get("channel_order")
    if isinstance(order, Sequence) and not isinstance(order, (str, bytes, bytearray)):
        return [str(int(value)) for value in order]
    results = control_payload.get("results")
    if not isinstance(results, Mapping):
        raise ValueError("Control calibration JSON must contain results.")
    return [str(int(value)) for value in results]


def _resolve_channel_parameter_map(
    raw_config: Mapping[str, Any],
    *,
    channel_parameter_map: Mapping[str | int, str] | None,
    channel_names: Sequence[str],
    parameter_names: Sequence[str],
) -> dict[str, str]:
    raw_mapping = channel_parameter_map
    if raw_mapping is None:
        calibration_section = raw_config.get("calibration")
        if isinstance(calibration_section, Mapping):
            voltage_to_tunable = calibration_section.get("voltage_to_tunable")
            if isinstance(voltage_to_tunable, Mapping):
                candidate = voltage_to_tunable.get("channel_to_parameter")
                if isinstance(candidate, Mapping):
                    raw_mapping = candidate

    if raw_mapping is None:
        return {}

    valid_channels = {str(int(channel)) for channel in channel_names}
    valid_parameters = {str(parameter) for parameter in parameter_names}
    resolved: dict[str, str] = {}
    for channel, parameter in raw_mapping.items():
        channel_name = str(int(channel))
        parameter_name = str(parameter)
        if channel_name not in valid_channels:
            raise ValueError(f"Unknown channel in channel_to_parameter: {channel_name}")
        if parameter_name not in valid_parameters:
            raise ValueError(f"Unknown parameter in channel_to_parameter: {parameter_name}")
        resolved[channel_name] = parameter_name
    return resolved


def _parse_channel_parameter_map(values: Sequence[str] | None) -> dict[str, str] | None:
    if values is None:
        return None
    parsed: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Channel map entries must use CHANNEL=PARAMETER syntax: {item!r}")
        channel, parameter = item.split("=", 1)
        parsed[str(int(channel.strip()))] = parameter.strip()
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare measured control curves against model curves using multiple alignment features."
    )
    parser.add_argument("config", type=Path, help="Path to model YAML.")
    parser.add_argument("--model-calibration", type=Path, default=None)
    parser.add_argument("--control-calibration", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--channel-parameter-map", nargs="+", default=None)
    parser.add_argument("--max-delta-nm", type=float, default=0.05)
    parser.add_argument("--shift-count", type=int, default=31)
    parser.add_argument("--no-abs-correlation", action="store_true")
    parser.add_argument("--example-count", type=int, default=5)
    parser.add_argument("--top-methods", type=int, default=3)
    args = parser.parse_args(list(argv) if argv is not None else None)

    payload = run_alignment_diagnostics(
        args.config,
        model_calibration_path=args.model_calibration,
        control_calibration_path=args.control_calibration,
        output_dir=args.output_dir,
        channel_parameter_map=_parse_channel_parameter_map(args.channel_parameter_map),
        max_delta_nm=args.max_delta_nm,
        shift_count=args.shift_count,
        include_abs_correlation=not args.no_abs_correlation,
        example_count=args.example_count,
        top_methods=args.top_methods,
    )

    print(f"saved_json: {payload['json_path']}")
    print(f"best_method: {payload['best_method']}")
    best_summary = payload["method_summaries"][0]
    print(f"mean_score: {best_summary['mean_score']:.6f}")
    print(f"edge_shift_fraction: {best_summary['edge_shift_fraction']:.6f}")
    print(f"best_match_plot: {payload['outputs']['best_match_plot']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
