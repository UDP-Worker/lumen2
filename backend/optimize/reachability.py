from __future__ import annotations

import argparse
import copy
import csv
import math
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
from numpy.typing import NDArray
from scipy.stats import qmc

if matplotlib.get_backend().lower() != "agg":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.interface.matlab_bridge import matlab_engine_session
from backend.model import load_model_config, simulate_from_dict
from backend.optimize.optimize import (
    OPTIMIZATION_DATA_ROOT,
    OptimizationTarget,
    _estimate_center_and_bandwidth_nm,
    _load_raw_yaml,
    _resolve_optimization_settings,
    _resolve_run_dir,
    _write_json,
)


DEFAULT_SAMPLE_COUNT = 1024
NEAREST_SAMPLE_COUNT = 12


def run_reachability_map(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    samples: int = DEFAULT_SAMPLE_COUNT,
    method: str = "sobol",
    seed: int = 42,
    center_window_nm: float | None = None,
    bandwidth_window_nm: float | None = None,
    peak_min_linear: float = 0.8,
    max_failures: int | None = None,
) -> dict[str, Any]:
    raw_config = _load_raw_yaml(config_path)
    resolved_config = load_model_config(config_path)
    model_name = str(resolved_config["model"]["name"])
    optimization_settings = _resolve_optimization_settings(raw_config, resolved_config)
    parameterization = optimization_settings["parameterization"]
    target = optimization_settings["target"]

    if parameterization.dimension == 0:
        raise ValueError("Reachability sampling requires at least one decision variable.")
    if samples <= 0:
        raise ValueError("samples must be greater than 0.")
    if peak_min_linear < 0.0:
        raise ValueError("peak_min_linear must be non-negative.")

    method = method.strip().lower()
    sample_vectors = _sample_vectors(
        method=method,
        requested_samples=int(samples),
        seed=int(seed),
        lower=parameterization.lower_bounds(),
        upper=parameterization.upper_bounds(),
    )

    run_dir = _resolve_reachability_run_dir(output_dir=output_dir, model_name=model_name)
    csv_path = run_dir / "reachability_samples.csv"
    plot_path = run_dir / "reachability_map.png"
    summary_path = run_dir / "reachability_summary.json"

    center_window = (
        float(center_window_nm)
        if center_window_nm is not None
        else float(target.bandwidth_3db_nm)
    )
    bandwidth_window = (
        float(bandwidth_window_nm)
        if bandwidth_window_nm is not None
        else float(target.bandwidth_3db_nm)
    )
    if center_window < 0.0 or bandwidth_window < 0.0:
        raise ValueError("center_window_nm and bandwidth_window_nm must be non-negative.")

    start_time = time.perf_counter()
    rows: list[dict[str, Any]] = []
    failures = 0
    fieldnames = _csv_fieldnames(parameterization)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        matlab_dir = Path(resolved_config["model"]["matlab_dir"]).resolve()
        with matlab_engine_session(extra_paths=(matlab_dir,)) as engine:
            for sample_index, vector in enumerate(sample_vectors, start=1):
                row = _evaluate_sample(
                    sample_index=sample_index,
                    vector=vector,
                    resolved_config=resolved_config,
                    parameterization=parameterization,
                    target=target,
                    engine=engine,
                )
                if row["status"] != "ok":
                    failures += 1
                    if max_failures is not None and failures > max_failures:
                        writer.writerow(row)
                        raise RuntimeError(
                            f"Reachability sampling exceeded max_failures={max_failures}."
                        )
                rows.append(row)
                writer.writerow(row)
                handle.flush()

    elapsed_seconds = time.perf_counter() - start_time
    valid_rows = [row for row in rows if row["status"] == "ok"]
    summary = _build_summary(
        config_path=config_path,
        run_dir=run_dir,
        elapsed_seconds=elapsed_seconds,
        method=method,
        requested_samples=int(samples),
        actual_samples=int(sample_vectors.shape[0]),
        seed=int(seed),
        parameterization=parameterization,
        target=target,
        center_window_nm=center_window,
        bandwidth_window_nm=bandwidth_window,
        peak_min_linear=float(peak_min_linear),
        rows=rows,
        valid_rows=valid_rows,
        csv_path=csv_path,
        plot_path=plot_path,
        summary_path=summary_path,
    )
    _plot_reachability_map(
        valid_rows=valid_rows,
        target=target,
        center_window_nm=center_window,
        bandwidth_window_nm=bandwidth_window,
        peak_min_linear=float(peak_min_linear),
        plot_path=plot_path,
    )
    _write_json(summary_path, summary)
    return summary


def _sample_vectors(
    *,
    method: str,
    requested_samples: int,
    seed: int,
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> NDArray[np.float64]:
    dimension = int(lower.size)
    if method == "sobol":
        exponent = max(0, math.ceil(math.log2(requested_samples)))
        sampler = qmc.Sobol(d=dimension, scramble=True, seed=seed)
        unit_samples = sampler.random_base2(m=exponent)
    elif method in {"latin", "lhs", "latin-hypercube", "latin_hypercube"}:
        sampler = qmc.LatinHypercube(d=dimension, seed=seed)
        unit_samples = sampler.random(n=requested_samples)
    elif method in {"random", "uniform"}:
        rng = np.random.default_rng(seed)
        unit_samples = rng.random((requested_samples, dimension), dtype=float)
    else:
        raise ValueError("method must be one of: sobol, lhs, random.")

    return np.asarray(qmc.scale(unit_samples, lower, upper), dtype=float)


def _evaluate_sample(
    *,
    sample_index: int,
    vector: Sequence[float],
    resolved_config: Mapping[str, Any],
    parameterization: Any,
    target: OptimizationTarget,
    engine: Any,
) -> dict[str, Any]:
    sample_start = time.perf_counter()
    decision_values = parameterization.decision_values(vector)
    tunable_values = parameterization.expand(vector)
    row: dict[str, Any] = {
        "sample_index": int(sample_index),
        "status": "ok",
        "error": "",
    }
    row.update({f"decision__{name}": value for name, value in decision_values.items()})
    row.update({f"tunable__{name}": value for name, value in tunable_values.items()})

    try:
        sample_config = _build_resolved_config(resolved_config, tunable_values)
        simulation_result = simulate_from_dict(sample_config, engine=engine)
        metrics = _extract_metrics(simulation_result, target)
        row.update(metrics)
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = str(exc)
    finally:
        row["elapsed_seconds"] = float(time.perf_counter() - sample_start)

    return row


def _build_resolved_config(
    resolved_config: Mapping[str, Any],
    tunable_values: Mapping[str, float],
) -> dict[str, Any]:
    sample_config = copy.deepcopy(dict(resolved_config))
    for name, value in tunable_values.items():
        sample_config["parameters"]["tunable"][name]["value"] = float(value)
    return sample_config


def _extract_metrics(
    simulation_result: Mapping[str, Any],
    target: OptimizationTarget,
) -> dict[str, float]:
    wavelength_nm = np.asarray(simulation_result["wavelength_nm"], dtype=float)
    power_linear = np.asarray(simulation_result["power_linear"], dtype=float)
    power_db = np.asarray(simulation_result["power_db"], dtype=float)
    if wavelength_nm.ndim != 1 or power_linear.ndim != 1 or wavelength_nm.size != power_linear.size:
        raise ValueError("Simulation result must contain matching 1D wavelength_nm and power_linear arrays.")

    peak_index = int(np.argmax(power_linear))
    peak_wavelength_nm = float(wavelength_nm[peak_index])
    peak_power_linear = float(power_linear[peak_index])
    peak_power_db = float(power_db[peak_index])
    center_nm, bandwidth_3db_nm = _estimate_center_and_bandwidth_nm(wavelength_nm, power_linear)

    center_error_nm = center_nm - target.center_nm
    bandwidth_error_nm = bandwidth_3db_nm - target.bandwidth_3db_nm
    peak_error_linear = peak_power_linear - 1.0
    target_bandwidth = max(float(target.bandwidth_3db_nm), 1e-12)
    normalized_distance_2d = math.hypot(
        center_error_nm / target_bandwidth,
        bandwidth_error_nm / target_bandwidth,
    )
    normalized_distance_3d = math.sqrt(
        (center_error_nm / target_bandwidth) ** 2
        + (bandwidth_error_nm / target_bandwidth) ** 2
        + peak_error_linear**2
    )

    return {
        "peak_wavelength_nm": peak_wavelength_nm,
        "peak_power_linear": peak_power_linear,
        "peak_power_db": peak_power_db,
        "measured_center_nm": float(center_nm),
        "measured_bandwidth_3db_nm": float(bandwidth_3db_nm),
        "center_error_nm": float(center_error_nm),
        "bandwidth_error_nm": float(bandwidth_error_nm),
        "peak_error_linear": float(peak_error_linear),
        "normalized_distance_2d": float(normalized_distance_2d),
        "normalized_distance_3d": float(normalized_distance_3d),
        "target_peak_power_linear": 1.0,
    }


def _build_summary(
    *,
    config_path: str | Path,
    run_dir: Path,
    elapsed_seconds: float,
    method: str,
    requested_samples: int,
    actual_samples: int,
    seed: int,
    parameterization: Any,
    target: OptimizationTarget,
    center_window_nm: float,
    bandwidth_window_nm: float,
    peak_min_linear: float,
    rows: Sequence[Mapping[str, Any]],
    valid_rows: Sequence[Mapping[str, Any]],
    csv_path: Path,
    plot_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    target_hits = [
        row
        for row in valid_rows
        if abs(float(row["center_error_nm"])) <= center_window_nm
        and abs(float(row["bandwidth_error_nm"])) <= bandwidth_window_nm
        and float(row["peak_power_linear"]) >= peak_min_linear
    ]
    nearest_rows = sorted(
        valid_rows,
        key=lambda row: float(row["normalized_distance_3d"]),
    )[:NEAREST_SAMPLE_COUNT]

    return {
        "config_path": str(Path(config_path).resolve()),
        "run_dir": str(run_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": float(elapsed_seconds),
        "method": method,
        "requested_samples": int(requested_samples),
        "actual_samples": int(actual_samples),
        "valid_samples": int(len(valid_rows)),
        "failed_samples": int(len(rows) - len(valid_rows)),
        "seed": int(seed),
        "parameterization": parameterization.summary(),
        "target": {
            **asdict(target),
            "peak_power_linear": 1.0,
        },
        "target_window": {
            "center_window_nm": float(center_window_nm),
            "bandwidth_window_nm": float(bandwidth_window_nm),
            "peak_min_linear": float(peak_min_linear),
            "hit_count": int(len(target_hits)),
            "hit_fraction": float(len(target_hits) / max(len(valid_rows), 1)),
        },
        "metric_ranges": _metric_ranges(valid_rows),
        "nearest_samples": [_compact_row(row) for row in nearest_rows],
        "artifacts": {
            "samples_csv": str(csv_path),
            "summary_json": str(summary_path),
            "scatter_plot": str(plot_path),
        },
    }


def _metric_ranges(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    metrics = (
        "measured_center_nm",
        "measured_bandwidth_3db_nm",
        "peak_power_linear",
        "normalized_distance_2d",
        "normalized_distance_3d",
    )
    ranges: dict[str, dict[str, float]] = {}
    for metric in metrics:
        values = np.asarray([float(row[metric]) for row in rows], dtype=float)
        if values.size == 0:
            ranges[metric] = {}
            continue
        ranges[metric] = {
            "min": float(np.min(values)),
            "p05": float(np.percentile(values, 5)),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }
    return ranges


def _compact_row(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "sample_index",
        "measured_center_nm",
        "measured_bandwidth_3db_nm",
        "peak_power_linear",
        "peak_wavelength_nm",
        "normalized_distance_2d",
        "normalized_distance_3d",
    )
    return {
        key: (float(row[key]) if key != "sample_index" else int(row[key]))
        for key in keys
    }


def _plot_reachability_map(
    *,
    valid_rows: Sequence[Mapping[str, Any]],
    target: OptimizationTarget,
    center_window_nm: float,
    bandwidth_window_nm: float,
    peak_min_linear: float,
    plot_path: Path,
) -> None:
    if not valid_rows:
        return

    centers = np.asarray([float(row["measured_center_nm"]) for row in valid_rows], dtype=float)
    bandwidths = np.asarray([float(row["measured_bandwidth_3db_nm"]) for row in valid_rows], dtype=float)
    peaks = np.asarray([float(row["peak_power_linear"]) for row in valid_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(9.0, 6.2), constrained_layout=True)
    scatter = ax.scatter(
        centers,
        bandwidths,
        c=peaks,
        s=18,
        cmap="viridis",
        alpha=0.78,
        edgecolors="none",
    )
    ax.scatter(
        [target.center_nm],
        [target.bandwidth_3db_nm],
        marker="*",
        s=220,
        color="#dc2626",
        edgecolors="#7f1d1d",
        linewidths=0.8,
        label="target",
        zorder=5,
    )
    ax.axvspan(
        target.center_nm - center_window_nm,
        target.center_nm + center_window_nm,
        color="#dc2626",
        alpha=0.08,
        linewidth=0,
    )
    ax.axhspan(
        target.bandwidth_3db_nm - bandwidth_window_nm,
        target.bandwidth_3db_nm + bandwidth_window_nm,
        color="#dc2626",
        alpha=0.08,
        linewidth=0,
    )
    ax.axvline(target.center_nm, color="#dc2626", linestyle="--", linewidth=1.0, alpha=0.75)
    ax.axhline(target.bandwidth_3db_nm, color="#dc2626", linestyle="--", linewidth=1.0, alpha=0.75)
    ax.set_xlabel("measured center wavelength (nm)")
    ax.set_ylabel("measured 3 dB bandwidth / FWHM (nm)")
    ax.set_title(f"Reachability map (target peak threshold {peak_min_linear:.3g})")
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("peak power (linear)")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)


def _csv_fieldnames(parameterization: Any) -> list[str]:
    fields = [
        "sample_index",
        "status",
        "error",
        "elapsed_seconds",
        "peak_wavelength_nm",
        "peak_power_linear",
        "peak_power_db",
        "measured_center_nm",
        "measured_bandwidth_3db_nm",
        "center_error_nm",
        "bandwidth_error_nm",
        "peak_error_linear",
        "normalized_distance_2d",
        "normalized_distance_3d",
        "target_peak_power_linear",
    ]
    fields.extend(f"decision__{variable.name}" for variable in parameterization.decision_variables)
    fields.extend(f"tunable__{name}" for name in parameterization.tunable_order)
    return fields


def _resolve_reachability_run_dir(
    *,
    output_dir: str | Path | None,
    model_name: str,
) -> Path:
    if output_dir is not None:
        run_dir = Path(output_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _resolve_run_dir(
        output_dir=OPTIMIZATION_DATA_ROOT / model_name / f"{timestamp}_reachability",
        model_name=model_name,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sample the tunable parameter space and plot filter reachability metrics."
    )
    parser.add_argument("config", type=Path, help="Path to the model YAML config.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory.")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument(
        "--method",
        choices=("sobol", "lhs", "random"),
        default="sobol",
        help="Sampling method. Sobol uses the next power-of-two sample count.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--center-window-nm", type=float, default=None)
    parser.add_argument("--bandwidth-window-nm", type=float, default=None)
    parser.add_argument("--peak-min-linear", type=float, default=0.8)
    parser.add_argument("--max-failures", type=int, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    summary = run_reachability_map(
        args.config,
        output_dir=args.output_dir,
        samples=args.samples,
        method=args.method,
        seed=args.seed,
        center_window_nm=args.center_window_nm,
        bandwidth_window_nm=args.bandwidth_window_nm,
        peak_min_linear=args.peak_min_linear,
        max_failures=args.max_failures,
    )
    target_window = summary["target_window"]
    nearest = summary["nearest_samples"][0] if summary["nearest_samples"] else None
    print(f"saved_summary: {summary['artifacts']['summary_json']}")
    print(f"run_dir: {summary['run_dir']}")
    print(f"valid_samples: {summary['valid_samples']} / {summary['actual_samples']}")
    print(f"target_window_hits: {target_window['hit_count']} ({target_window['hit_fraction']:.4%})")
    if nearest is not None:
        print(f"nearest_sample_index: {nearest['sample_index']}")
        print(f"nearest_center_nm: {nearest['measured_center_nm']:.6f}")
        print(f"nearest_bandwidth_3db_nm: {nearest['measured_bandwidth_3db_nm']:.6f}")
        print(f"nearest_peak_power_linear: {nearest['peak_power_linear']:.6f}")
        print(f"nearest_normalized_distance_3d: {nearest['normalized_distance_3d']:.6f}")
    print(f"scatter_plot: {summary['artifacts']['scatter_plot']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
