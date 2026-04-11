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
    get_mapping,
    get_sequence,
    load_raw_yaml,
    resolve_model_name,
    save_curve_archives,
    sanitize_key,
    summarize_curve_bundle,
    write_json,
)
from backend.interface.OSA import read_spectrum
from backend.interface.VoltageSource import (
    configure_channel_limits,
    connect_voltage_source,
    disconnect_voltage_source,
    set_channel_voltages,
)
from backend.model import load_model_config


@dataclass(slots=True)
class ControlCalibrationSettings:
    channels: list[int]
    com_port: int
    vmax: float
    imax: float
    settle_time_s: float
    calibration_offsets: NDArray[np.float64]
    zero_voltages: dict[int, float]
    osa_settings: dict[str, Any]


def run_control_calibration(
    config_path: str | Path,
    *,
    channels: Sequence[int] | None = None,
    com_port: int | None = None,
    vmax: float | None = None,
    imax: float | None = None,
    settle_time_s: float | None = None,
    calibration_offsets: Sequence[float] | None = None,
    zero_voltages: Mapping[int, float] | None = None,
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
        calibration_offsets=calibration_offsets,
        zero_voltages=zero_voltages,
    )
    logger.info(
        "Starting control curve capture: channels=%s, zero_voltages=%s, vmax=%.6f, imax=%.6f",
        settings.channels,
        settings.zero_voltages,
        settings.vmax,
        settings.imax,
    )

    current_voltages = dict(settings.zero_voltages)
    calibration_bundles: list[CurveSweepBundle] = []
    calibration_results: dict[str, Any] = {}

    engine = connect_voltage_source(settings.com_port)
    try:
        configure_channel_limits(
            settings.channels,
            vmax=settings.vmax,
            imax=settings.imax,
            engine=engine,
        )

        _apply_voltages(current_voltages, engine=engine, settle_time_s=settings.settle_time_s)

        for channel in settings.channels:
            sweep_values = settings.zero_voltages[channel] + settings.calibration_offsets
            _validate_voltage_sweep(
                channel,
                sweep_values,
                zero_voltage=settings.zero_voltages[channel],
                vmax=settings.vmax,
            )
            logger.info(
                "Capturing channel %d around zero voltage %.6f with %d sweep samples.",
                channel,
                settings.zero_voltages[channel],
                len(sweep_values),
            )
            bundle = _capture_spectrum_bundle(
                channel,
                sweep_values,
                current_voltages,
                osa_settings=settings.osa_settings,
                settle_time_s=settings.settle_time_s,
                engine=engine,
            )
            calibration_bundles.append(bundle)
            logger.info(
                "Captured %d curves for channel %d over sweep range %.6f -> %.6f.",
                len(bundle.sweep_values),
                channel,
                float(np.min(bundle.sweep_values)),
                float(np.max(bundle.sweep_values)),
            )
            calibration_results[str(channel)] = {
                "zero_voltage": float(settings.zero_voltages[channel]),
                "curve_archive_prefix": sanitize_key(bundle.name),
                "sweep_values": [float(value) for value in bundle.sweep_values],
                "curve_count": int(bundle.sweep_values.size),
                "curve_summary": summarize_curve_bundle(bundle),
            }
    finally:
        disconnect_voltage_source(engine, stop_engine=True)

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
            "calibration_offsets": [float(value) for value in settings.calibration_offsets],
            "zero_voltages": {str(key): float(value) for key, value in settings.zero_voltages.items()},
            "osa_settings": settings.osa_settings,
        },
        "channel_order": settings.channels,
        "zero_voltages": {str(channel): float(value) for channel, value in settings.zero_voltages.items()},
        "results": calibration_results,
        "curve_archive": str(calibration_archive),
    }
    json_path = write_json(destination_dir / "control_calibration.json", payload)
    payload["json_path"] = str(json_path)
    return payload


def _resolve_settings(
    raw_config: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    *,
    channels: Sequence[int] | None,
    com_port: int | None,
    vmax: float | None,
    imax: float | None,
    settle_time_s: float | None,
    calibration_offsets: Sequence[float] | None,
    zero_voltages: Mapping[int, float] | None,
) -> ControlCalibrationSettings:
    calibration_section = get_mapping(raw_config, "calibration")
    control_section = get_mapping(calibration_section, "control")
    voltage_section = get_mapping(control_section, "voltage_source")
    osa_section = dict(get_mapping(control_section, "osa"))

    resolved_channels = [int(value) for value in (channels or get_sequence(control_section, "channels"))]
    if not resolved_channels:
        raise ValueError("Control calibration requires at least one channel.")
    if len(set(resolved_channels)) != len(resolved_channels):
        raise ValueError("Control calibration channels must not contain duplicates.")

    resolved_com_port = com_port if com_port is not None else voltage_section.get("com_port")
    if resolved_com_port is None:
        raise ValueError("Control calibration requires com_port or calibration.control.voltage_source.com_port.")

    resolved_vmax_raw = vmax if vmax is not None else voltage_section.get("vmax")
    resolved_imax_raw = imax if imax is not None else voltage_section.get("imax")
    if resolved_vmax_raw is None or resolved_imax_raw is None:
        raise ValueError(
            "Control calibration requires both vmax and imax. "
            "Provide them via CLI or calibration.control.voltage_source."
        )
    if float(resolved_vmax_raw) <= 0.0:
        raise ValueError("Control calibration requires vmax to be positive.")
    if float(resolved_imax_raw) <= 0.0:
        raise ValueError("Control calibration requires imax to be positive.")

    zero_voltage_section = zero_voltages or _resolve_zero_voltage_mapping(control_section)
    resolved_zero_voltages: dict[int, float] = {}
    for channel in resolved_channels:
        if channel not in zero_voltage_section:
            raise ValueError(
                "Control calibration requires zero voltages for every channel. "
                f"Missing channel {channel}."
            )
        resolved_zero_voltages[channel] = float(zero_voltage_section[channel])
        _validate_voltage_limit(
            channel,
            resolved_zero_voltages[channel],
            vmax=float(resolved_vmax_raw),
            label="zero voltage",
        )

    wavelength_section = resolved_config["simulation"]["wavelength_nm"]
    osa_section.setdefault("lam_start_nm", float(wavelength_section["start"]))
    osa_section.setdefault("lam_stop_nm", float(wavelength_section["stop"]))
    osa_section.setdefault("resolution_nm", max(0.02, float(wavelength_section["step"]) * 10.0))
    osa_section.setdefault("points_per_resolution", 5)
    osa_section.setdefault("speed", "2x")
    osa_section.setdefault("timeout_s", 150.0)
    osa_section.setdefault("plot_result", False)
    osa_section.setdefault("record", False)

    resolved_offsets = np.asarray(
        calibration_offsets
        if calibration_offsets is not None
        else get_sequence(
            control_section,
            "calibration_offsets",
            default=(-0.4, -0.2, 0.0, 0.2, 0.4),
        ),
        dtype=float,
    )
    if resolved_offsets.ndim != 1 or resolved_offsets.size < 2:
        raise ValueError("Control calibration requires at least two calibration offsets.")

    return ControlCalibrationSettings(
        channels=resolved_channels,
        com_port=int(resolved_com_port),
        vmax=float(resolved_vmax_raw),
        imax=float(resolved_imax_raw),
        settle_time_s=float(settle_time_s if settle_time_s is not None else control_section.get("settle_time_s", 0.5)),
        calibration_offsets=resolved_offsets,
        zero_voltages=resolved_zero_voltages,
        osa_settings=osa_section,
    )


def _resolve_zero_voltage_mapping(control_section: Mapping[str, Any]) -> dict[int, float]:
    raw_mapping = control_section.get("zero_voltages")
    if raw_mapping is None:
        raw_mapping = control_section.get("initial_voltages")
    if not isinstance(raw_mapping, Mapping):
        raise ValueError(
            "Control calibration requires calibration.control.zero_voltages "
            "(or legacy calibration.control.initial_voltages)."
        )

    resolved: dict[int, float] = {}
    for key, value in raw_mapping.items():
        resolved[int(key)] = float(value)
    return resolved


def _build_control_calibration_logger(model_name: str, destination_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"backend.calibrate.control_curve_capture.{model_name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    log_path = destination_dir / "control_calibration.log"
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


def _apply_voltages(
    channel_to_voltage: Mapping[int, float],
    *,
    engine: Any,
    settle_time_s: float,
) -> None:
    set_channel_voltages(channel_to_voltage, engine=engine)
    if settle_time_s > 0:
        time.sleep(float(settle_time_s))


def _validate_voltage_sweep(
    channel: int,
    sweep_values: Sequence[float],
    *,
    zero_voltage: float,
    vmax: float,
) -> None:
    _validate_voltage_limit(channel, zero_voltage, vmax=vmax, label="zero voltage")
    for index, value in enumerate(sweep_values):
        _validate_voltage_limit(channel, float(value), vmax=vmax, label=f"sweep value #{index}")


def _validate_voltage_limit(
    channel: int,
    voltage: float,
    *,
    vmax: float,
    label: str,
) -> None:
    if abs(float(voltage)) > float(vmax):
        raise ValueError(
            f"Channel {channel} {label} {float(voltage):.6f} V exceeds configured |vmax| {float(vmax):.6f} V."
        )


def _parse_zero_voltage_overrides(values: Sequence[str] | None) -> dict[int, float] | None:
    if values is None:
        return None

    parsed: dict[int, float] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(
                "Zero-voltage overrides must use CHANNEL=VOLTAGE syntax, "
                f"but got {item!r}."
            )
        channel_text, voltage_text = item.split("=", 1)
        parsed[int(channel_text.strip())] = float(voltage_text.strip())
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Capture control-side voltage sweep curves for later shape-based calibration."
    )
    parser.add_argument("config", type=Path, help="Path to the model YAML config.")
    parser.add_argument("--channels", nargs="+", type=int, default=None)
    parser.add_argument("--com-port", type=int, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--imax", type=float, default=None)
    parser.add_argument("--settle-time-s", type=float, default=None)
    parser.add_argument("--calibration-offsets", nargs="+", type=float, default=None)
    parser.add_argument(
        "--zero-voltages",
        nargs="+",
        default=None,
        metavar="CHANNEL=VOLTAGE",
        help="Per-channel zero voltages, for example `--zero-voltages 1=0.0 2=0.15`.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    payload = run_control_calibration(
        args.config,
        channels=args.channels,
        com_port=args.com_port,
        vmax=args.vmax,
        imax=args.imax,
        settle_time_s=args.settle_time_s,
        calibration_offsets=args.calibration_offsets,
        zero_voltages=_parse_zero_voltage_overrides(args.zero_voltages),
        output_dir=args.output_dir,
    )

    print(f"saved_json: {payload['json_path']}")
    print(f"model_name: {payload['model_name']}")
    print(f"channel_count: {len(payload['results'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
