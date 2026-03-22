from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from backend.calibrate._shared import (
    CurveSweepBundle,
    bundle_from_power_sweeps,
    ensure_output_dir,
    estimate_zero_crossing,
    extinction_ratio_db,
    get_mapping,
    get_sequence,
    load_raw_yaml,
    ordered_selection_records,
    resolve_model_name,
    save_curve_archives,
    summarize_curve_bundle,
    write_json,
)
from backend.interface.OSA import read_power_at_wavelengths, read_spectrum
from backend.interface.VoltageSource import (
    configure_channel_limits,
    connect_voltage_source,
    disconnect_voltage_source,
    set_channel_voltages,
)
from backend.model import load_model_config
from backend.utils.complex_response_viewer import (
    SelectionCancelledError,
    SelectionRecord,
    select_variable_targets,
)


@dataclass(slots=True)
class ControlCalibrationSettings:
    channels: list[int]
    com_port: int
    vmax: float | None
    imax: float | None
    settle_time_s: float
    initialization_offsets: NDArray[np.float64]
    calibration_offsets: NDArray[np.float64]
    initial_voltages: dict[int, float]
    shared_baseline: bool
    osa_settings: dict[str, Any]


def run_control_calibration(
    config_path: str | Path,
    *,
    channels: Sequence[int] | None = None,
    com_port: int | None = None,
    vmax: float | None = None,
    imax: float | None = None,
    settle_time_s: float | None = None,
    initialization_offsets: Sequence[float] | None = None,
    calibration_offsets: Sequence[float] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    raw_config = load_raw_yaml(config_path)
    resolved_config = load_model_config(config_path)
    model_name = resolve_model_name(raw_config, resolved_config)
    destination_dir = Path(output_dir).resolve() if output_dir is not None else ensure_output_dir(model_name)
    destination_dir.mkdir(parents=True, exist_ok=True)
    logger = _build_control_calibration_logger(model_name, destination_dir)

    settings = _resolve_settings(
        raw_config,
        resolved_config,
        channels=channels,
        com_port=com_port,
        vmax=vmax,
        imax=imax,
        settle_time_s=settle_time_s,
        initialization_offsets=initialization_offsets,
        calibration_offsets=calibration_offsets,
    )

    current_voltages = dict(settings.initial_voltages)
    initialization_bundles: list[CurveSweepBundle] = []
    calibration_bundles: list[CurveSweepBundle] = []
    zeroing_history: dict[str, Any] = {}
    calibration_results: dict[str, Any] = {}
    initialization_curve_targets: dict[int, list[SelectionRecord]] = {}

    engine = connect_voltage_source(settings.com_port)
    try:
        if settings.vmax is not None or settings.imax is not None:
            configure_channel_limits(
                settings.channels,
                vmax=settings.vmax,
                imax=settings.imax,
                engine=engine,
            )

        _apply_voltages(current_voltages, engine=engine, settle_time_s=settings.settle_time_s)

        for channel in settings.channels:
            sweep_values = current_voltages[channel] + settings.initialization_offsets
            bundle = _capture_spectrum_bundle(
                channel,
                sweep_values,
                current_voltages,
                osa_settings=settings.osa_settings,
                settle_time_s=settings.settle_time_s,
                engine=engine,
            )
            initialization_bundles.append(bundle)

        initialization_groups: list[Any] = []
        initialization_target_maps: dict[int, dict[str, int]] = {}
        for bundle in initialization_bundles:
            groups, target_to_index = bundle.to_single_curve_groups()
            initialization_groups.extend(groups)
            initialization_target_maps[int(bundle.metadata["channel"])] = target_to_index

        viewer_result = select_variable_targets(
            initialization_groups,
            title="Control calibration: choose through/extinction wavelengths",
            shared_baseline=False,
            logger=logger,
        )

        for bundle in initialization_bundles:
            channel = int(bundle.metadata["channel"])
            bundle_selections = {
                name: record
                for name, record in viewer_result.selections.items()
                if name in initialization_target_maps[channel]
            }
            initialization_curve_targets[channel] = ordered_selection_records(
                initialization_target_maps[channel],
                bundle_selections,
                expected_count=len(bundle.sweep_values),
            )
        logger.info(
            "Initialization wavelength targets selected: %s",
            {
                str(channel): len(records)
                for channel, records in initialization_curve_targets.items()
            },
        )

        for bundle in initialization_bundles:
            channel = int(bundle.metadata["channel"])
            curve_selections = initialization_curve_targets[channel]
            candidate_voltages = current_voltages[channel] + settings.initialization_offsets
            through_power_samples, extinction_power_samples = _measure_target_powers(
                channel,
                candidate_voltages,
                current_voltages,
                curve_selections=curve_selections,
                osa_settings=settings.osa_settings,
                settle_time_s=settings.settle_time_s,
                engine=engine,
            )
            er_samples = extinction_ratio_db(through_power_samples, extinction_power_samples)
            zero_voltage = estimate_zero_crossing(candidate_voltages, er_samples)
            current_voltages[channel] = float(zero_voltage)
            _apply_voltages(current_voltages, engine=engine, settle_time_s=settings.settle_time_s)
            logger.info(
                "Zero search for channel %d used %d curve-specific selections and chose %.6f V.",
                channel,
                len(curve_selections),
                zero_voltage,
            )
            zeroing_history[str(channel)] = {
                "curve_selections": _serialize_curve_selections(bundle, curve_selections),
                "candidate_voltages": [float(value) for value in candidate_voltages],
                "power_db_at_through_wavelength": [float(value) for value in through_power_samples],
                "power_db_at_extinction_wavelength": [float(value) for value in extinction_power_samples],
                "extinction_ratio_db": [float(value) for value in er_samples],
                "chosen_zero_voltage": float(zero_voltage),
            }

        for channel in settings.channels:
            sweep_values = current_voltages[channel] + settings.calibration_offsets
            bundle = _capture_spectrum_bundle(
                channel,
                sweep_values,
                current_voltages,
                osa_settings=settings.osa_settings,
                settle_time_s=settings.settle_time_s,
                engine=engine,
            )
            calibration_bundles.append(bundle)
            curve_groups, target_to_index = bundle.to_single_curve_groups()
            viewer_result = select_variable_targets(
                curve_groups,
                title=f"Control calibration: channel {channel}",
                shared_baseline=False,
                logger=logger,
            )
            curve_selections = ordered_selection_records(
                target_to_index,
                viewer_result.selections,
                expected_count=len(bundle.sweep_values),
            )

            logger.info(
                "Formal calibration for channel %d is using %d curve-specific selections around zero %.6f V.",
                channel,
                len(curve_selections),
                current_voltages[channel],
            )
            through_power_samples = bundle.power_db_at_curve_wavelengths(
                [record.through_wavelength_nm for record in curve_selections]
            )
            extinction_power_samples = bundle.power_db_at_curve_wavelengths(
                [record.extinction_wavelength_nm for record in curve_selections]
            )
            er_samples = extinction_ratio_db(through_power_samples, extinction_power_samples)
            calibration_results[str(channel)] = {
                "curve_selections": _serialize_curve_selections(bundle, curve_selections),
                "zero_voltage": float(current_voltages[channel]),
                "sweep_values": [float(value) for value in bundle.sweep_values],
                "power_db_at_through_wavelength": [float(value) for value in through_power_samples],
                "power_db_at_extinction_wavelength": [
                    float(value) for value in extinction_power_samples
                ],
                "extinction_ratio_db": [float(value) for value in er_samples],
                "curve_summary": summarize_curve_bundle(bundle),
            }
    finally:
        disconnect_voltage_source(engine, stop_engine=True)

    initialization_archive = save_curve_archives(
        destination_dir / "control_initialization_curves.npz",
        initialization_bundles,
    )
    calibration_archive = save_curve_archives(
        destination_dir / "control_calibration_curves.npz",
        calibration_bundles,
    )

    payload = {
        "model_name": model_name,
        "config_path": str(Path(config_path).resolve()),
        "settings": {
            "channels": settings.channels,
            "com_port": settings.com_port,
            "vmax": settings.vmax,
            "imax": settings.imax,
            "settle_time_s": settings.settle_time_s,
            "initialization_offsets": [float(value) for value in settings.initialization_offsets],
            "calibration_offsets": [float(value) for value in settings.calibration_offsets],
            "initial_voltages": {str(key): float(value) for key, value in settings.initial_voltages.items()},
            "shared_baseline": settings.shared_baseline,
            "osa_settings": settings.osa_settings,
        },
        "channel_targets": {
            str(int(bundle.metadata["channel"])): _serialize_curve_selections(
                bundle,
                initialization_curve_targets[int(bundle.metadata["channel"])],
            )
            for bundle in initialization_bundles
        },
        "zero_voltages": {str(channel): float(value) for channel, value in current_voltages.items()},
        "initialization": {
            "curve_archive": str(initialization_archive),
            "curve_summaries": {
                bundle.name: summarize_curve_bundle(bundle) for bundle in initialization_bundles
            },
            "zeroing_history": zeroing_history,
        },
        "results": calibration_results,
        "curve_archive": str(calibration_archive),
    }
    json_path = write_json(destination_dir / "control_calibration.json", payload)
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


def _resolve_settings(
    raw_config: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    *,
    channels: Sequence[int] | None,
    com_port: int | None,
    vmax: float | None,
    imax: float | None,
    settle_time_s: float | None,
    initialization_offsets: Sequence[float] | None,
    calibration_offsets: Sequence[float] | None,
) -> ControlCalibrationSettings:
    calibration_section = get_mapping(raw_config, "calibration")
    control_section = get_mapping(calibration_section, "control")
    voltage_section = get_mapping(control_section, "voltage_source")
    osa_section = dict(get_mapping(control_section, "osa"))

    resolved_channels = [int(value) for value in (channels or get_sequence(control_section, "channels"))]
    if not resolved_channels:
        raise ValueError("Control calibration requires at least one channel.")

    resolved_com_port = com_port if com_port is not None else voltage_section.get("com_port")
    if resolved_com_port is None:
        raise ValueError("Control calibration requires com_port or calibration.control.voltage_source.com_port.")

    initial_voltage_section = control_section.get("initial_voltages")
    initial_voltages = {channel: 0.0 for channel in resolved_channels}
    if isinstance(initial_voltage_section, Mapping):
        for channel in resolved_channels:
            raw_value = initial_voltage_section.get(channel, initial_voltage_section.get(str(channel), 0.0))
            initial_voltages[channel] = float(raw_value)

    wavelength_section = resolved_config["simulation"]["wavelength_nm"]
    osa_section.setdefault("lam_start_nm", float(wavelength_section["start"]))
    osa_section.setdefault("lam_stop_nm", float(wavelength_section["stop"]))
    osa_section.setdefault("resolution_nm", max(0.02, float(wavelength_section["step"]) * 10.0))
    osa_section.setdefault("points_per_resolution", 5)
    osa_section.setdefault("speed", "2x")
    osa_section.setdefault("timeout_s", 150.0)
    osa_section.setdefault("plot_result", False)
    osa_section.setdefault("record", False)

    return ControlCalibrationSettings(
        channels=resolved_channels,
        com_port=int(resolved_com_port),
        vmax=float(vmax if vmax is not None else voltage_section.get("vmax"))
        if (vmax is not None or voltage_section.get("vmax") is not None)
        else None,
        imax=float(imax if imax is not None else voltage_section.get("imax"))
        if (imax is not None or voltage_section.get("imax") is not None)
        else None,
        settle_time_s=float(settle_time_s if settle_time_s is not None else control_section.get("settle_time_s", 0.5)),
        initialization_offsets=np.asarray(
            initialization_offsets
            if initialization_offsets is not None
            else get_sequence(
                control_section,
                "initialization_offsets",
                default=(-0.2, -0.1, 0.0, 0.1, 0.2),
            ),
            dtype=float,
        ),
        calibration_offsets=np.asarray(
            calibration_offsets
            if calibration_offsets is not None
            else get_sequence(
                control_section,
                "calibration_offsets",
                default=(-0.4, -0.2, 0.0, 0.2, 0.4),
            ),
            dtype=float,
        ),
        initial_voltages=initial_voltages,
        shared_baseline=bool(control_section.get("shared_baseline", True)),
        osa_settings=osa_section,
    )


def _build_control_calibration_logger(model_name: str, destination_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"backend.calibrate.control_selection.{model_name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    log_path = destination_dir / "control_calibration_viewer.log"
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


def _capture_spectrum_bundle(
    channel: int,
    sweep_values: Sequence[float],
    base_voltages: Mapping[int, float],
    *,
    osa_settings: Mapping[str, Any],
    settle_time_s: float,
    engine: Any,
) -> CurveSweepBundle:
    power_rows: list[NDArray[np.float64]] = []
    wavelength_nm: NDArray[np.float64] | None = None

    for value in sweep_values:
        trial = dict(base_voltages)
        trial[channel] = float(value)
        _apply_voltages(trial, engine=engine, settle_time_s=settle_time_s)
        spectrum = read_spectrum(engine=engine, **osa_settings)
        current_wavelength = np.asarray(spectrum.wavelength_nm, dtype=float)
        current_power = np.asarray(spectrum.power_dbm, dtype=float)

        if wavelength_nm is None:
            wavelength_nm = current_wavelength
        elif wavelength_nm.shape != current_wavelength.shape or not np.allclose(
            wavelength_nm,
            current_wavelength,
        ):
            raise ValueError("OSA returned inconsistent wavelength grids during control calibration.")

        power_rows.append(current_power)

    _apply_voltages(base_voltages, engine=engine, settle_time_s=settle_time_s)

    if wavelength_nm is None:
        raise RuntimeError(f"No spectra captured while sweeping channel {channel}.")

    return bundle_from_power_sweeps(
        f"ch{channel}",
        sweep_values,
        wavelength_nm,
        power_rows,
        metadata={"channel": channel},
    )


def _measure_target_powers(
    channel: int,
    sweep_values: Sequence[float],
    base_voltages: Mapping[int, float],
    *,
    curve_selections: Sequence[SelectionRecord],
    osa_settings: Mapping[str, Any],
    settle_time_s: float,
    engine: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if len(curve_selections) != len(sweep_values):
        raise ValueError("curve_selections must contain one selection record per sweep value.")

    through_powers: list[float] = []
    extinction_powers: list[float] = []
    for value, selection in zip(sweep_values, curve_selections, strict=True):
        trial = dict(base_voltages)
        trial[channel] = float(value)
        _apply_voltages(trial, engine=engine, settle_time_s=settle_time_s)
        _, selected_powers = read_power_at_wavelengths(
            [
                float(selection.through_wavelength_nm),
                float(selection.extinction_wavelength_nm),
            ],
            engine=engine,
            **osa_settings,
        )
        through_powers.append(float(selected_powers[0]))
        extinction_powers.append(float(selected_powers[1]))

    _apply_voltages(base_voltages, engine=engine, settle_time_s=settle_time_s)
    return np.asarray(through_powers, dtype=float), np.asarray(extinction_powers, dtype=float)


def _apply_voltages(
    channel_to_voltage: Mapping[int, float],
    *,
    engine: Any,
    settle_time_s: float,
) -> None:
    set_channel_voltages(channel_to_voltage, engine=engine)
    if settle_time_s > 0:
        time.sleep(float(settle_time_s))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate control voltages against extinction ratio.")
    parser.add_argument("config", type=Path, help="Path to the model YAML config.")
    parser.add_argument("--channels", nargs="+", type=int, default=None)
    parser.add_argument("--com-port", type=int, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--imax", type=float, default=None)
    parser.add_argument("--settle-time-s", type=float, default=None)
    parser.add_argument("--initialization-offsets", nargs="+", type=float, default=None)
    parser.add_argument("--calibration-offsets", nargs="+", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        payload = run_control_calibration(
            args.config,
            channels=args.channels,
            com_port=args.com_port,
            vmax=args.vmax,
            imax=args.imax,
            settle_time_s=args.settle_time_s,
            initialization_offsets=args.initialization_offsets,
            calibration_offsets=args.calibration_offsets,
            output_dir=args.output_dir,
        )
    except SelectionCancelledError:
        return 1

    print(f"saved_json: {payload['json_path']}")
    print(f"model_name: {payload['model_name']}")
    print(f"channel_count: {len(payload['results'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
