from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from ..matlab_bridge import (
    matlab_engine_session,
    matlab_struct_get,
    to_matlab_row_vector,
    to_numpy_1d,
)

OSA_MATLAB_DIR = Path(__file__).resolve().parent


@dataclass(slots=True)
class SpectrumResult:
    wavelength_nm: NDArray[np.float64]
    power_dbm: NDArray[np.float64]
    raw_power_dbm: NDArray[np.float64]
    selected_wavelength_nm: NDArray[np.float64] | None
    selected_power_dbm: NDArray[np.float64] | None
    settings: Any

    @property
    def data(self) -> NDArray[np.float64]:
        return np.column_stack((self.wavelength_nm, self.power_dbm))


def read_spectrum(
    *,
    lam_start_nm: float = 1548.0,
    lam_stop_nm: float = 1552.0,
    resolution_nm: float = 0.02,
    sensitivity: str = "high2",
    points_per_resolution: int = 5,
    speed: str = "2x",
    reflevel_up_dbm: float = 0.0,
    reflevel_down_dbm: float = -100.0,
    channel: str = "a",
    board_index: int = 0,
    primary_address: int = 1,
    timeout_s: float = 150.0,
    use_requested_range: bool = True,
    restore_defaults: bool = True,
    record: bool = False,
    save_path: str | Path | None = None,
    plot_result: bool = False,
    normalization: bool = False,
    normalization_reference_path: str | Path | None = None,
    target_wavelengths_nm: Sequence[float] | None = None,
    interpolation_method: str = "linear",
    engine: Any | None = None,
) -> SpectrumResult:
    args: list[Any] = [
        "lam_start_nm",
        float(lam_start_nm),
        "lam_stop_nm",
        float(lam_stop_nm),
        "resolution_nm",
        float(resolution_nm),
        "sensitivity",
        str(sensitivity),
        "points_per_resolution",
        float(points_per_resolution),
        "speed",
        str(speed),
        "reflevel_up_dbm",
        float(reflevel_up_dbm),
        "reflevel_down_dbm",
        float(reflevel_down_dbm),
        "channel",
        str(channel),
        "board_index",
        float(board_index),
        "primary_address",
        float(primary_address),
        "timeout_s",
        float(timeout_s),
        "use_requested_range",
        bool(use_requested_range),
        "restore_defaults",
        bool(restore_defaults),
        "record",
        bool(record),
        "plot_result",
        bool(plot_result),
        "normalization",
        bool(normalization),
        "interpolation_method",
        str(interpolation_method),
    ]

    if save_path is not None:
        args.extend(["save_path", str(Path(save_path))])

    if normalization_reference_path is not None:
        args.extend(["normalization_reference_path", str(Path(normalization_reference_path))])

    if target_wavelengths_nm is not None:
        args.extend(["target_wavelengths_nm", to_matlab_row_vector(target_wavelengths_nm)])

    with matlab_engine_session(engine, extra_paths=(OSA_MATLAB_DIR,)) as matlab_engine:
        result = matlab_engine.wavelength_sweep(*args, nargout=1)

    return _to_spectrum_result(result)


def read_power_at_wavelengths(
    target_wavelengths_nm: Sequence[float],
    **kwargs: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    spectrum = read_spectrum(target_wavelengths_nm=target_wavelengths_nm, **kwargs)
    if spectrum.selected_wavelength_nm is None or spectrum.selected_power_dbm is None:
        raise RuntimeError("MATLAB did not return sampled wavelength data.")

    return spectrum.selected_wavelength_nm, spectrum.selected_power_dbm


def _to_spectrum_result(result: Any) -> SpectrumResult:
    selected_wavelengths = matlab_struct_get(result, "selected_wavelength_nm")
    selected_powers = matlab_struct_get(result, "selected_power_dbm")

    return SpectrumResult(
        wavelength_nm=to_numpy_1d(matlab_struct_get(result, "wavelength_nm")),
        power_dbm=to_numpy_1d(matlab_struct_get(result, "power_dbm")),
        raw_power_dbm=to_numpy_1d(matlab_struct_get(result, "raw_power_dbm")),
        selected_wavelength_nm=_optional_array(selected_wavelengths),
        selected_power_dbm=_optional_array(selected_powers),
        settings=matlab_struct_get(result, "settings", {}),
    )


def _optional_array(values: Any) -> NDArray[np.float64] | None:
    if values is None:
        return None

    array = to_numpy_1d(values)
    if array.size == 0:
        return None

    return array
