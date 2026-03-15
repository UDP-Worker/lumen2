from __future__ import annotations

from typing import Mapping, Sequence

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
    """Collect one wavelength/baseline pair for model-parameter calibration."""

    return select_extinction_reference(groups, title=title)


def collect_control_targets(
    groups: Sequence[CurveGroup] | Mapping[str, CurveGroup],
    *,
    title: str = "Control Calibration Targets",
    shared_baseline: bool = True,
) -> ViewerSelectionResult:
    """Collect per-variable wavelength targets and extinction-ratio baselines."""

    return select_variable_targets(
        groups,
        title=title,
        shared_baseline=shared_baseline,
    )


__all__ = [
    "ComplexCurve",
    "CurveGroup",
    "SelectionCancelledError",
    "SelectionRecord",
    "ViewerSelectionResult",
    "build_curve_group",
    "collect_control_targets",
    "collect_model_reference",
    "select_extinction_reference",
    "select_variable_targets",
]
