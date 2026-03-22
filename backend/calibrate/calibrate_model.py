from __future__ import annotations

import argparse
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from backend.calibrate._shared import (
    CurveSweepBundle,
    ensure_output_dir,
    extinction_ratio_db,
    get_mapping,
    load_raw_yaml,
    ordered_selection_records,
    resolve_model_name,
    resolve_zero_tunable_config,
    save_zero_tunable_config,
    save_curve_archives,
    summarize_curve_bundle,
    write_json,
)
from backend.interface.matlab_bridge import matlab_engine_session
from backend.model import load_model_config, simulate_from_dict
from backend.utils.complex_response_viewer import (
    SelectionCancelledError,
    SelectionRecord,
    TunableEditorPlot,
    TunableEditorResult,
    TunableParameterSpec,
    build_curve_group,
    edit_tunable_parameters,
    select_variable_targets,
)


def run_model_calibration(
    config_path: str | Path,
    *,
    parameters: Sequence[str] | None = None,
    num_samples: int | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    raw_config = load_raw_yaml(config_path)
    resolved_config = load_model_config(config_path)
    model_name = resolve_model_name(raw_config, resolved_config)
    destination_dir = Path(output_dir).resolve() if output_dir is not None else ensure_output_dir(model_name)
    destination_dir.mkdir(parents=True, exist_ok=True)
    logger = _build_model_calibration_logger(model_name, destination_dir)

    _require_explicit_zero_tunable_config(raw_config, resolved_config, config_path=config_path)
    zero_tunable = resolve_zero_tunable_config(raw_config, resolved_config)
    tunable_specs = resolved_config["parameters"]["tunable"]
    parameter_names = list(parameters or tunable_specs.keys())
    unknown = [name for name in parameter_names if name not in tunable_specs]
    if unknown:
        raise ValueError(f"Unknown tunable parameters requested: {unknown}")

    calibration_section = get_mapping(raw_config, "calibration")
    model_sweep_section = get_mapping(calibration_section, "model_sweep")
    base_config = deepcopy(resolved_config)
    for name, value in zero_tunable.items():
        base_config["parameters"]["tunable"][name]["value"] = float(value)

    bundles: list[CurveSweepBundle] = []
    results: dict[str, Any] = {}

    with matlab_engine_session() as engine:
        curve_groups: list[Any] = []
        target_maps_by_parameter: dict[str, dict[str, int]] = {}
        bundles_by_parameter: dict[str, CurveSweepBundle] = {}
        for parameter_name in parameter_names:
            sweep_values = _resolve_parameter_sweep_values(
                parameter_name,
                tunable_specs[parameter_name],
                model_sweep_section,
                num_samples=num_samples,
            )
            logger.info(
                "Starting model calibration selection for %s with %d sweep samples.",
                parameter_name,
                len(sweep_values),
            )
            bundle = _simulate_parameter_sweep(
                parameter_name,
                sweep_values,
                base_config,
                engine=engine,
            )
            bundles.append(bundle)
            bundles_by_parameter[parameter_name] = bundle
            parameter_curve_groups, target_to_index = bundle.to_single_curve_groups()
            curve_groups.extend(parameter_curve_groups)
            target_maps_by_parameter[parameter_name] = target_to_index

        viewer_result = select_variable_targets(
            curve_groups,
            title="Model calibration",
            shared_baseline=False,
            logger=logger,
        )

        for parameter_name in parameter_names:
            bundle = bundles_by_parameter[parameter_name]
            curve_selections = ordered_selection_records(
                target_maps_by_parameter[parameter_name],
                {
                    name: record
                    for name, record in viewer_result.selections.items()
                    if name in target_maps_by_parameter[parameter_name]
                },
                expected_count=len(bundle.sweep_values),
            )
            logger.info(
                "Completed model calibration selection for %s with %d curve-specific selections.",
                parameter_name,
                len(curve_selections),
            )
            through_power_samples = bundle.power_db_at_curve_wavelengths(
                [record.through_wavelength_nm for record in curve_selections]
            )
            extinction_power_samples = bundle.power_db_at_curve_wavelengths(
                [record.extinction_wavelength_nm for record in curve_selections]
            )
            extinction_samples = extinction_ratio_db(through_power_samples, extinction_power_samples)

            results[parameter_name] = {
                "curve_selections": _serialize_curve_selections(bundle, curve_selections),
                "zero_tunable_value": float(zero_tunable[parameter_name]),
                "sweep_values": [float(value) for value in bundle.sweep_values],
                "power_db_at_through_wavelength": [float(value) for value in through_power_samples],
                "power_db_at_extinction_wavelength": [
                    float(value) for value in extinction_power_samples
                ],
                "extinction_ratio_db": [float(value) for value in extinction_samples],
                "curve_summary": summarize_curve_bundle(bundle),
            }

    archive_path = save_curve_archives(destination_dir / "model_calibration_curves.npz", bundles)
    payload = {
        "model_name": model_name,
        "config_path": str(Path(config_path).resolve()),
        "curve_archive": str(archive_path),
        "zero_tunable_config": {name: float(value) for name, value in zero_tunable.items()},
        "parameter_order": parameter_names,
        "results": results,
    }
    json_path = write_json(destination_dir / "model_calibration.json", payload)
    payload["json_path"] = str(json_path)
    return payload


def _serialize_curve_selections(
    bundle: CurveSweepBundle,
    curve_selections: Sequence[SelectionRecord],
) -> list[dict[str, Any]]:
    return [
        {
            "curve_index": int(index),
            "curve_label": f"{bundle.name}={float(bundle.sweep_values[index]):+.6g}",
            "sweep_value": float(bundle.sweep_values[index]),
            "through_wavelength_nm": float(record.through_wavelength_nm),
            "extinction_wavelength_nm": float(record.extinction_wavelength_nm),
        }
        for index, record in enumerate(curve_selections)
    ]


def edit_model_zero_config(
    config_path: str | Path,
) -> TunableEditorResult:
    raw_config = load_raw_yaml(config_path)
    resolved_config = load_model_config(config_path)
    model_name = resolve_model_name(raw_config, resolved_config)
    destination_dir = ensure_output_dir(model_name)
    logger = _build_zero_config_logger(model_name, destination_dir)
    tunable_specs = resolved_config["parameters"]["tunable"]
    initial_values = resolve_zero_tunable_config(raw_config, resolved_config)
    base_config = deepcopy(resolved_config)
    logger.info("Launching zero-config editor for %s using config %s", model_name, Path(config_path).resolve())
    logger.info("Initial tunable values: %s", initial_values)

    parameter_specs = [
        TunableParameterSpec(
            name=name,
            value=float(initial_values[name]),
            lower_bound=(
                float(spec["bounds"][0])
                if isinstance(spec.get("bounds"), Sequence) and len(spec["bounds"]) == 2
                else None
            ),
            upper_bound=(
                float(spec["bounds"][1])
                if isinstance(spec.get("bounds"), Sequence) and len(spec["bounds"]) == 2
                else None
            ),
        )
        for name, spec in tunable_specs.items()
    ]

    with matlab_engine_session() as engine:
        def render_curves(values: dict[str, float]) -> TunableEditorPlot:
            run_config = deepcopy(base_config)
            for name, value in values.items():
                run_config["parameters"]["tunable"][name]["value"] = float(value)
            result = simulate_from_dict(run_config, engine=engine)

            power_db = np.asarray(result["power_db"], dtype=float)
            display_curve = np.power(10.0, power_db / 20.0).astype(np.complex128)
            constraint_status = result.get("parameters", {}).get("constraint_status", {})
            summary_lines = [
                f"Model: {result.get('model_name', model_name)}",
                f"Port: {result.get('port_name', resolved_config['outputs']['observe_port'])}",
                f"Samples: {power_db.size}",
                f"Power range (dB): {power_db.min():.4f} -> {power_db.max():.4f}",
            ]
            logger.info(
                "Zero-config simulation complete: samples=%d, power_range=(%.6f, %.6f)",
                power_db.size,
                float(power_db.min()),
                float(power_db.max()),
            )
            if isinstance(constraint_status, Mapping):
                summary_lines.append(f"Constraints satisfied: {bool(constraint_status.get('satisfied', True))}")
                violations = constraint_status.get("violations")
                if isinstance(violations, Sequence):
                    for violation in violations[:3]:
                        if isinstance(violation, Mapping):
                            summary_lines.append(
                                f"Violation: {violation.get('name', 'constraint')} "
                                f"(residual={float(violation.get('residual', 0.0)):.4g})"
                            )

            return TunableEditorPlot(
                groups=(
                    build_curve_group(
                        "current_response",
                        result["wavelength_nm"],
                        display_curve,
                    ),
                ),
                summary_lines=tuple(summary_lines),
            )

        def save_values(values: dict[str, float]) -> str:
            saved_path = save_zero_tunable_config(config_path, values)
            logger.info("Saved zero-config values to %s", saved_path)
            return str(saved_path)

        return edit_tunable_parameters(
            parameter_specs,
            render_curves=render_curves,
            save_values=save_values,
            title=f"Model zero config editor: {model_name}",
            save_button_text="Save Zero Config",
            logger=logger,
        )


def _resolve_parameter_sweep_values(
    parameter_name: str,
    tunable_spec: Mapping[str, Any],
    model_sweep_section: Mapping[str, Any],
    *,
    num_samples: int | None,
) -> NDArray[np.float64]:
    explicit_values_section = model_sweep_section.get("values")
    if isinstance(explicit_values_section, Mapping) and parameter_name in explicit_values_section:
        explicit_values = explicit_values_section[parameter_name]
        if not isinstance(explicit_values, Sequence) or isinstance(
            explicit_values, (str, bytes, bytearray)
        ):
            raise ValueError(f"calibration.model_sweep.values.{parameter_name} must be a sequence.")
        values = np.asarray([float(item) for item in explicit_values], dtype=float)
        if values.size < 2:
            raise ValueError(
                f"calibration.model_sweep.values.{parameter_name} must contain at least two values."
            )
        return values

    sample_count = int(num_samples or model_sweep_section.get("num_samples", 9))
    if sample_count < 2:
        raise ValueError("num_samples must be at least 2.")

    bounds = tunable_spec.get("bounds")
    if isinstance(bounds, Sequence) and len(bounds) == 2:
        return np.linspace(float(bounds[0]), float(bounds[1]), sample_count)

    center = float(tunable_spec["value"])
    half_span = float(model_sweep_section.get("default_half_span", 1.0))
    return np.linspace(center - half_span, center + half_span, sample_count)


def _simulate_parameter_sweep(
    parameter_name: str,
    sweep_values: Sequence[float],
    base_config: Mapping[str, Any],
    *,
    engine: Any,
) -> CurveSweepBundle:
    wavelength_nm: NDArray[np.float64] | None = None
    complex_rows: list[NDArray[np.complex128]] = []
    power_rows: list[NDArray[np.float64]] = []

    for sweep_value in sweep_values:
        run_config = deepcopy(base_config)
        run_config["parameters"]["tunable"][parameter_name]["value"] = float(sweep_value)
        result = simulate_from_dict(run_config, engine=engine)

        current_wavelength = np.asarray(result["wavelength_nm"], dtype=float)
        current_complex = np.asarray(result["complex_response"], dtype=np.complex128)
        current_power = np.asarray(result["power_db"], dtype=float)

        if wavelength_nm is None:
            wavelength_nm = current_wavelength
        elif current_wavelength.shape != wavelength_nm.shape or not np.allclose(
            current_wavelength,
            wavelength_nm,
        ):
            raise ValueError("Model sweep returned inconsistent wavelength grids.")

        complex_rows.append(current_complex)
        power_rows.append(current_power)

    if wavelength_nm is None:
        raise RuntimeError(f"No model results were collected for parameter '{parameter_name}'.")

    return CurveSweepBundle(
        name=parameter_name,
        sweep_values=np.asarray(sweep_values, dtype=float),
        wavelength_nm=wavelength_nm,
        complex_response_matrix=np.vstack(complex_rows),
        power_db_matrix=np.vstack(power_rows),
        metadata={"parameter_name": parameter_name},
    )


def _require_explicit_zero_tunable_config(
    raw_config: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    *,
    config_path: str | Path,
) -> None:
    calibration_section = raw_config.get("calibration")
    zero_section = calibration_section.get("zero_config") if isinstance(calibration_section, Mapping) else None
    tunable_zero = zero_section.get("tunable") if isinstance(zero_section, Mapping) else None

    if not isinstance(tunable_zero, Mapping):
        raise ValueError(
            "Model calibration requires calibration.zero_config.tunable in the YAML config. "
            f"Run `python -m backend.calibrate.calibrate_model {Path(config_path)} --edit-zero-config` first."
        )

    missing = [
        name
        for name in resolved_config["parameters"]["tunable"]
        if name not in tunable_zero
    ]
    if missing:
        raise ValueError(
            "calibration.zero_config.tunable is missing values for: "
            f"{missing}. Run `--edit-zero-config` to rewrite the full zero config."
        )


def _build_zero_config_logger(model_name: str, destination_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"backend.calibrate.zero_config.{model_name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    log_path = destination_dir / "zero_config_editor.log"
    resolved_log_path = log_path.resolve()
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == resolved_log_path:
            return logger
        logger.removeHandler(handler)
        handler.close()

    handler = logging.FileHandler(resolved_log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def _build_model_calibration_logger(model_name: str, destination_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"backend.calibrate.model_selection.{model_name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    log_path = destination_dir / "model_calibration_viewer.log"
    resolved_log_path = log_path.resolve()
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == resolved_log_path:
            return logger
        logger.removeHandler(handler)
        handler.close()

    handler = logging.FileHandler(resolved_log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate model tunable parameters against extinction ratio.")
    parser.add_argument("config", type=Path, help="Path to the model YAML config.")
    parser.add_argument(
        "--edit-zero-config",
        action="store_true",
        help="Open the interactive zero-config editor and write calibration.zero_config.tunable back to YAML.",
    )
    parser.add_argument(
        "--parameters",
        nargs="+",
        default=None,
        help="Optional subset of tunable parameters to calibrate.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Override the number of sweep samples per parameter.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional custom output directory.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        if args.edit_zero_config:
            result = edit_model_zero_config(args.config)
            print(f"saved_config: {result.saved_path or '--'}")
            print(f"parameter_count: {len(result.values)}")
            return 0

        payload = run_model_calibration(
            args.config,
            parameters=args.parameters,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
        )
    except SelectionCancelledError:
        return 1

    print(f"saved_json: {payload['json_path']}")
    print(f"model_name: {payload['model_name']}")
    print(f"parameter_count: {len(payload['results'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
