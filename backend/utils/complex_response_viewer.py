from __future__ import annotations

import argparse
import colorsys
import logging
import math
import tkinter as tk
from dataclasses import dataclass, field
from tkinter import messagebox, ttk
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

DB_EPS = 1e-12
GLOBAL_TARGET = "__global__"
PALETTE = (
    "#d1495b",
    "#00798c",
    "#edae49",
    "#30638e",
    "#003d5b",
    "#8f2d56",
    "#66a182",
    "#6f4e7c",
)
_MATPLOTLIB_TK_IMPORTS: tuple[Any, Any, Any] | None = None


@dataclass(frozen=True, slots=True)
class ComplexCurve:
    wavelength_nm: NDArray[np.float64]
    complex_amplitude: NDArray[np.complex128]
    label: str
    sweep_value: Any | None = None

    def magnitude_db(self) -> NDArray[np.float64]:
        return 20.0 * np.log10(np.maximum(np.abs(self.complex_amplitude), DB_EPS))


@dataclass(frozen=True, slots=True)
class CurveGroup:
    name: str
    curves: tuple[ComplexCurve, ...]
    color: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SelectionRecord:
    through_wavelength_nm: float
    extinction_wavelength_nm: float


@dataclass(frozen=True, slots=True)
class ViewerSelectionResult:
    selections: dict[str, SelectionRecord]
    visible_groups: tuple[str, ...]
    x_limits_nm: tuple[float, float]
    y_limits_db: tuple[float, float]


@dataclass(frozen=True, slots=True)
class TunableParameterSpec:
    name: str
    value: float
    lower_bound: float | None = None
    upper_bound: float | None = None


@dataclass(frozen=True, slots=True)
class TunableEditorPlot:
    groups: tuple[CurveGroup, ...]
    summary_lines: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TunableEditorResult:
    values: dict[str, float]
    saved_path: str | None = None


class SelectionCancelledError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class _PlotRange:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def padded(self) -> "_PlotRange":
        xpad = max((self.xmax - self.xmin) * 0.02, 1e-6)
        ypad = max((self.ymax - self.ymin) * 0.05, 1e-6)
        return _PlotRange(
            xmin=self.xmin - xpad,
            xmax=self.xmax + xpad,
            ymin=self.ymin - ypad,
            ymax=self.ymax + ypad,
        )


def build_curve_group(
    name: str,
    wavelength_nm: ArrayLike,
    complex_response_matrix: ArrayLike,
    *,
    sweep_values: Sequence[Any] | None = None,
    curve_labels: Sequence[str] | None = None,
    color: str | None = None,
    curve_axis: int = 0,
    metadata: Mapping[str, Any] | None = None,
) -> CurveGroup:
    wavelength = _as_float_1d(wavelength_nm, "wavelength_nm")
    matrix = np.asarray(complex_response_matrix, dtype=np.complex128)
    if matrix.ndim == 1:
        matrix = matrix[np.newaxis, :]
    elif matrix.ndim != 2:
        raise ValueError("complex_response_matrix must be 1D or 2D.")
    if curve_axis not in {0, 1}:
        raise ValueError("curve_axis must be 0 or 1.")
    if curve_axis == 1:
        matrix = np.swapaxes(matrix, 0, 1)
    if matrix.shape[1] != wavelength.size:
        raise ValueError("complex_response_matrix wavelength dimension does not match wavelength_nm.")

    count = matrix.shape[0]
    if sweep_values is not None and len(sweep_values) != count:
        raise ValueError("sweep_values length must match the number of curves.")
    if curve_labels is not None and len(curve_labels) != count:
        raise ValueError("curve_labels length must match the number of curves.")

    curves = []
    for index in range(count):
        label = (
            str(curve_labels[index])
            if curve_labels is not None
            else f"{name}={sweep_values[index]}" if sweep_values is not None else f"{name}[{index}]"
        )
        curves.append(
            ComplexCurve(
                wavelength_nm=wavelength,
                complex_amplitude=matrix[index].copy(),
                label=label,
                sweep_value=None if sweep_values is None else sweep_values[index],
            )
        )
    return CurveGroup(name=name, curves=tuple(curves), color=color, metadata=metadata or {})


def select_extinction_reference(
    groups: Sequence[CurveGroup] | Mapping[str, CurveGroup],
    *,
    title: str = "Select Extinction Reference",
    logger: logging.Logger | None = None,
) -> SelectionRecord:
    result = _run_viewer(groups, mode="single", shared_baseline=False, title=title, logger=logger)
    return result.selections[GLOBAL_TARGET]


def select_variable_targets(
    groups: Sequence[CurveGroup] | Mapping[str, CurveGroup],
    *,
    title: str = "Select Calibration Targets",
    shared_baseline: bool = True,
    logger: logging.Logger | None = None,
) -> ViewerSelectionResult:
    return _run_viewer(
        groups,
        mode="per-variable",
        shared_baseline=shared_baseline,
        title=title,
        logger=logger,
    )


def edit_tunable_parameters(
    parameter_specs: Sequence[TunableParameterSpec],
    *,
    render_curves: Callable[[dict[str, float]], TunableEditorPlot],
    save_values: Callable[[dict[str, float]], str | None],
    title: str = "Edit Tunable Parameters",
    save_button_text: str = "Save",
    logger: logging.Logger | None = None,
) -> TunableEditorResult:
    return _TunableEditor(
        parameter_specs,
        render_curves=render_curves,
        save_values=save_values,
        title=title,
        save_button_text=save_button_text,
        logger=logger,
    ).show()


def _load_matplotlib_tk() -> tuple[Any, Any, Any]:
    global _MATPLOTLIB_TK_IMPORTS
    if _MATPLOTLIB_TK_IMPORTS is not None:
        return _MATPLOTLIB_TK_IMPORTS

    try:
        import matplotlib

        matplotlib.use("TkAgg")
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        from matplotlib.figure import Figure
    except Exception as exc:  # pragma: no cover - import/environment dependent
        raise RuntimeError(
            "The zero-config editor requires matplotlib with the TkAgg backend. "
            "Install 'matplotlib' in the active Python environment."
        ) from exc

    _MATPLOTLIB_TK_IMPORTS = (Figure, FigureCanvasTkAgg, NavigationToolbar2Tk)
    return _MATPLOTLIB_TK_IMPORTS


class _TunableEditor:
    def __init__(
        self,
        parameter_specs: Sequence[TunableParameterSpec],
        *,
        render_curves: Callable[[dict[str, float]], TunableEditorPlot],
        save_values: Callable[[dict[str, float]], str | None],
        title: str,
        save_button_text: str,
        logger: logging.Logger | None,
    ) -> None:
        if not parameter_specs:
            raise ValueError("At least one tunable parameter is required.")

        self.parameter_specs = tuple(parameter_specs)
        self.specs_by_name = {spec.name: spec for spec in self.parameter_specs}
        if len(self.specs_by_name) != len(self.parameter_specs):
            raise ValueError("Duplicate tunable parameter names are not allowed.")

        self.render_curves = render_curves
        self.save_values = save_values
        self.save_button_text = save_button_text
        self.logger = logger

        self.initial_values = {spec.name: float(spec.value) for spec in self.parameter_specs}
        self.current_values = dict(self.initial_values)
        self.saved_path: str | None = None
        self.groups: list[CurveGroup] = []
        self.colors: dict[str, str] = {}
        self.default_range = _PlotRange(0.0, 1.0, -1.0, 1.0)
        self.range = self.default_range
        self.last_cursor: tuple[float, float] | None = None
        self.result: TunableEditorResult | None = None
        self.cancelled = False
        self._pending_render_log_id: str | None = None
        self._syncing_limits = False
        self.figure: Any | None = None
        self.axes: Any | None = None
        self.figure_canvas: Any | None = None
        self.toolbar: Any | None = None
        self.cursor_vline: Any | None = None
        self.cursor_hline: Any | None = None
        self.cursor_annotation: Any | None = None

        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            raise RuntimeError("Unable to open tkinter tunable-parameter editor.") from exc

        self.root.title(title)
        self.root.geometry("1520x900")
        self.root.minsize(1200, 760)
        self.root.protocol("WM_DELETE_WINDOW", self.cancel)

        self.status = tk.StringVar(
            value=(
                "Adjust tunable values, click Simulate to refresh the curve, then save the zero config."
            )
        )
        self.summary = tk.StringVar(value="")
        self.summary_lines: tuple[str, ...] = ()
        self.cursor_x = tk.StringVar(value="--")
        self.cursor_y = tk.StringVar(value="--")
        self.value_vars = {
            spec.name: tk.StringVar(value=self._format_value(spec.value))
            for spec in self.parameter_specs
        }
        self.scales: dict[str, ttk.Scale] = {}

        self.parameter_canvas: tk.Canvas
        self.parameter_frame: ttk.Frame
        self.parameter_window: int
        self._build_ui()
        self._log_info(
            "Tunable editor initialized with parameters: %s",
            ", ".join(spec.name for spec in self.parameter_specs),
        )
        self._render_plot(initial=True)

    def show(self) -> TunableEditorResult:
        self.root.mainloop()
        if self.result is not None:
            self._log_info("Tunable editor closed with saved_path=%s", self.result.saved_path)
            return self.result
        if self.cancelled:
            self._log_info("Tunable editor cancelled by user.")
            raise SelectionCancelledError("Tunable-parameter editing was cancelled.")
        raise RuntimeError("Tunable-parameter editor closed without a result.")

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        controls = ttk.Frame(outer, width=420)
        controls.pack(side="left", fill="y")
        controls.pack_propagate(False)

        plot_wrap = ttk.Frame(outer)
        plot_wrap.pack(side="left", fill="both", expand=True, padx=(12, 0))
        toolbar_wrap = ttk.Frame(plot_wrap)
        toolbar_wrap.pack(fill="x")
        plot_surface = ttk.Frame(plot_wrap)
        plot_surface.pack(fill="both", expand=True)

        ttk.Label(
            controls,
            text=(
                "This editor runs the current model with the tunable values shown below. "
                "Use it to find an all-pass / zero-response operating point, then write it "
                "back to calibration.zero_config.tunable in the YAML file. The right-hand "
                "plot uses an embedded Matplotlib canvas with the standard navigation toolbar."
            ),
            wraplength=390,
            justify="left",
        ).pack(fill="x", pady=(0, 12))

        button_row = ttk.Frame(controls)
        button_row.pack(fill="x", pady=(0, 12))
        ttk.Button(button_row, text="Simulate", command=self._render_plot).pack(
            side="left",
            fill="x",
            expand=True,
        )
        ttk.Button(button_row, text="Reset", command=self._reset_values).pack(
            side="left",
            fill="x",
            expand=True,
            padx=(8, 0),
        )

        cursor_frame = ttk.LabelFrame(controls, text="Cursor", padding=10)
        cursor_frame.pack(fill="x", pady=(0, 12))
        ttk.Label(cursor_frame, text="Cursor wavelength (nm)").pack(anchor="w")
        ttk.Label(cursor_frame, textvariable=self.cursor_x, font=("Consolas", 10, "bold")).pack(
            anchor="w",
            pady=(0, 8),
        )
        ttk.Label(cursor_frame, text="Cursor magnitude (dB)").pack(anchor="w")
        ttk.Label(cursor_frame, textvariable=self.cursor_y, font=("Consolas", 10, "bold")).pack(
            anchor="w",
            pady=(0, 8),
        )
        ttk.Button(cursor_frame, text="Reset Zoom", command=self._reset_zoom).pack(fill="x")

        summary_frame = ttk.LabelFrame(controls, text="Simulation Summary", padding=10)
        summary_frame.pack(fill="x", pady=(0, 12))
        ttk.Label(
            summary_frame,
            textvariable=self.summary,
            justify="left",
            wraplength=380,
        ).pack(fill="x")

        parameter_section = ttk.LabelFrame(controls, text="Tunable Values", padding=8)
        parameter_section.pack(fill="both", expand=True)

        self.parameter_canvas = tk.Canvas(parameter_section, highlightthickness=0, height=360)
        scrollbar = ttk.Scrollbar(
            parameter_section,
            orient="vertical",
            command=self.parameter_canvas.yview,
        )
        self.parameter_canvas.configure(yscrollcommand=scrollbar.set)
        self.parameter_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="left", fill="y", padx=(8, 0))

        self.parameter_frame = ttk.Frame(self.parameter_canvas)
        self.parameter_window = self.parameter_canvas.create_window(
            (0, 0),
            window=self.parameter_frame,
            anchor="nw",
        )
        self.parameter_frame.bind("<Configure>", self._sync_parameter_scroll)
        self.parameter_canvas.bind("<Configure>", self._resize_parameter_frame)

        for spec in self.parameter_specs:
            row = ttk.LabelFrame(self.parameter_frame, text=spec.name, padding=8)
            row.pack(fill="x", pady=(0, 8))

            entry = ttk.Entry(row, textvariable=self.value_vars[spec.name])
            entry.pack(fill="x")
            entry.bind(
                "<Return>",
                lambda _event, name=spec.name: self._apply_entry_value(name, rerender=True),
            )
            entry.bind(
                "<FocusOut>",
                lambda _event, name=spec.name: self._apply_entry_value(name, rerender=False),
            )

            lower = spec.lower_bound
            upper = spec.upper_bound
            if (
                lower is not None
                and upper is not None
                and math.isfinite(lower)
                and math.isfinite(upper)
                and lower < upper
            ):
                scale = ttk.Scale(
                    row,
                    from_=float(lower),
                    to=float(upper),
                    command=lambda raw, name=spec.name: self._scale_changed(name, raw),
                )
                scale.set(float(spec.value))
                scale.pack(fill="x", pady=(6, 0))
                scale.bind(
                    "<ButtonRelease-1>",
                    lambda _event, name=spec.name: self._render_plot_for(name),
                )
                self.scales[spec.name] = scale

            bounds_text = "Bounds: unbounded"
            if lower is not None and upper is not None:
                bounds_text = f"Bounds: [{self._format_value(lower)}, {self._format_value(upper)}]"
            ttk.Label(row, text=bounds_text, justify="left").pack(anchor="w", pady=(6, 0))

        footer = ttk.Frame(controls)
        footer.pack(fill="x", pady=(12, 0))
        ttk.Button(
            footer,
            text=self.save_button_text,
            command=self._save_current_values,
        ).pack(side="left", fill="x", expand=True)
        ttk.Button(footer, text="Close", command=self._confirm).pack(
            side="left",
            fill="x",
            expand=True,
            padx=(8, 0),
        )

        self._build_matplotlib_plot(plot_surface, toolbar_wrap)
        ttk.Label(plot_wrap, textvariable=self.status, anchor="w").pack(fill="x", pady=(8, 0))

    def _sync_parameter_scroll(self, _event: tk.Event[tk.Misc]) -> None:
        self.parameter_canvas.configure(scrollregion=self.parameter_canvas.bbox("all"))

    def _resize_parameter_frame(self, event: tk.Event[tk.Misc]) -> None:
        self.parameter_canvas.itemconfigure(self.parameter_window, width=event.width)

    def _build_matplotlib_plot(self, plot_surface: ttk.Frame, toolbar_wrap: ttk.Frame) -> None:
        Figure, FigureCanvasTkAgg, NavigationToolbar2Tk = _load_matplotlib_tk()

        self.figure = Figure(figsize=(10.0, 7.5), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=0.10, right=0.98, top=0.97, bottom=0.11)
        self.axes.callbacks.connect("xlim_changed", self._axes_limits_changed)
        self.axes.callbacks.connect("ylim_changed", self._axes_limits_changed)

        self.figure_canvas = FigureCanvasTkAgg(self.figure, master=plot_surface)
        self.figure_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.figure_canvas.mpl_connect("motion_notify_event", self._motion)
        self.figure_canvas.mpl_connect("axes_leave_event", self._leave)
        self.figure_canvas.mpl_connect("figure_leave_event", self._leave)

        self.toolbar = NavigationToolbar2Tk(self.figure_canvas, toolbar_wrap, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(fill="x")
        self._log_info("Matplotlib plot surface initialized for tunable editor.")

    def _render_plot_for(self, name: str) -> None:
        self._apply_entry_value(name, rerender=True, show_error=False)

    def _scale_changed(self, name: str, raw_value: str) -> None:
        value = float(raw_value)
        self.current_values[name] = value
        self.value_vars[name].set(self._format_value(value))

    def _apply_entry_value(
        self,
        name: str,
        *,
        rerender: bool,
        show_error: bool = True,
    ) -> None:
        try:
            value = self._parse_value(name, self.value_vars[name].get())
        except ValueError as exc:
            self.value_vars[name].set(self._format_value(self.current_values[name]))
            if show_error:
                messagebox.showerror("Invalid value", str(exc))
            return

        self.current_values[name] = value
        scale = self.scales.get(name)
        if scale is not None:
            scale.set(value)
        self.value_vars[name].set(self._format_value(value))
        if rerender:
            self._render_plot()

    def _collect_values(self) -> dict[str, float]:
        values: dict[str, float] = {}
        for name in self.specs_by_name:
            values[name] = self._parse_value(name, self.value_vars[name].get())
        return values

    def _parse_value(self, name: str, raw_text: str) -> float:
        try:
            value = float(raw_text)
        except ValueError as exc:
            raise ValueError(f"'{name}' must be a valid floating-point number.") from exc

        spec = self.specs_by_name[name]
        if spec.lower_bound is not None and value < spec.lower_bound:
            raise ValueError(
                f"'{name}' must be greater than or equal to {self._format_value(spec.lower_bound)}."
            )
        if spec.upper_bound is not None and value > spec.upper_bound:
            raise ValueError(
                f"'{name}' must be less than or equal to {self._format_value(spec.upper_bound)}."
            )
        return float(value)

    def _reset_values(self) -> None:
        self.current_values = dict(self.initial_values)
        for name, value in self.current_values.items():
            self.value_vars[name].set(self._format_value(value))
            scale = self.scales.get(name)
            if scale is not None:
                scale.set(value)
        self._render_plot()

    def _save_current_values(self) -> None:
        try:
            values = self._collect_values()
        except ValueError as exc:
            messagebox.showerror("Invalid value", str(exc))
            return

        try:
            saved_path = self.save_values(values)
        except Exception as exc:  # pragma: no cover - GUI error path
            messagebox.showerror("Save failed", str(exc))
            self.status.set(f"Save failed: {exc}")
            self._log_exception("Zero-config save failed")
            return

        self.current_values = dict(values)
        self.saved_path = saved_path
        self._refresh_summary(self.summary_lines)
        self._log_info("Zero config saved to %s with values=%s", saved_path, values)
        self.status.set(
            f"Zero config saved to {saved_path or 'the configured destination'}. "
            "You can continue tuning or close the editor."
        )
        messagebox.showinfo(
            "Zero config saved",
            f"Saved calibration.zero_config.tunable to:\n{saved_path}",
        )

    def _render_plot(self, initial: bool = False) -> None:
        try:
            values = self._collect_values()
        except ValueError as exc:
            if initial:
                raise
            messagebox.showerror("Invalid value", str(exc))
            return

        self.status.set("Running model simulation...")
        self.root.update_idletasks()
        self._log_info("Running tunable-editor simulation with values=%s", values)

        try:
            plot = self.render_curves(values)
        except Exception as exc:  # pragma: no cover - GUI error path
            self.status.set(f"Simulation failed: {exc}")
            if not initial:
                messagebox.showerror("Simulation failed", str(exc))
            self._log_exception("Tunable-editor simulation failed")
            return

        groups = _normalize_groups(list(plot.groups))
        self.current_values = dict(values)
        self.groups = groups
        self.colors = {
            group.name: group.color or PALETTE[index % len(PALETTE)]
            for index, group in enumerate(self.groups)
        }
        self.default_range = _default_range(self.groups)
        self.range = self.default_range
        self._refresh_summary(plot.summary_lines)
        self._log_info(
            "Simulation produced %d group(s), %d curve(s), range=(%.6f, %.6f, %.6f, %.6f)",
            len(self.groups),
            sum(len(group.curves) for group in self.groups),
            self.default_range.xmin,
            self.default_range.xmax,
            self.default_range.ymin,
            self.default_range.ymax,
        )
        self.status.set(
            "Simulation updated. Use the Matplotlib toolbar or Reset Zoom to inspect the curve, then save the zero config."
        )
        self._redraw()
        self.root.update_idletasks()
        self._schedule_canvas_state_log("after-render")

    def _refresh_summary(self, lines: Sequence[str]) -> None:
        self.summary_lines = tuple(line for line in lines if line.strip())
        display_lines = list(self.summary_lines)
        if self.saved_path:
            display_lines.append(f"Last saved: {self.saved_path}")
        self.summary.set("\n".join(display_lines) if display_lines else "No additional summary.")

    def _motion(self, event: Any) -> None:
        if self.axes is None or event.inaxes is not self.axes or event.xdata is None or event.ydata is None:
            self._leave()
            return
        self.last_cursor = (float(event.xdata), float(event.ydata))
        self.cursor_x.set(f"{self.last_cursor[0]:.6f}")
        self.cursor_y.set(f"{self.last_cursor[1]:.4f}")
        self._update_cursor_overlay()

    def _leave(self, _event: Any | None = None) -> None:
        self.last_cursor = None
        self.cursor_x.set("--")
        self.cursor_y.set("--")
        self._update_cursor_overlay()

    def _reset_zoom(self) -> None:
        self.range = self.default_range
        self.status.set(
            "View reset to the latest simulation bounds."
        )
        self._apply_range_to_axes()
        self._schedule_canvas_state_log("after-reset-zoom")

    def _redraw(self) -> None:
        if self.axes is None or self.figure_canvas is None:
            return

        self.axes.clear()
        self.axes.set_facecolor("white")
        self.axes.grid(True, color="#d7d7d7", linewidth=0.8, alpha=0.65)
        self.axes.set_xlabel("Wavelength (nm)")
        self.axes.set_ylabel("20*log10(|E|) (dB)")

        legend_handles: list[Any] = []
        legend_labels: list[str] = []
        if not self.groups:
            self.axes.text(
                0.5,
                0.5,
                "No curves to display.",
                transform=self.axes.transAxes,
                ha="center",
                va="center",
            )
            self._log_info("Redraw skipped because there are no groups to display.")
        else:
            for group in self.groups:
                group_line: Any | None = None
                for index, curve in enumerate(group.curves):
                    order = np.argsort(curve.wavelength_nm)
                    x = curve.wavelength_nm[order]
                    y = curve.magnitude_db()[order]
                    if x.size < 2:
                        continue
                    color = _shade(self.colors[group.name], index, len(group.curves))
                    (line,) = self.axes.plot(x, y, color=color, linewidth=2.4, alpha=0.96)
                    marker_indices = sorted({0, x.size - 1, int(np.argmin(y)), int(np.argmax(y))})
                    self.axes.scatter(
                        x[marker_indices],
                        y[marker_indices],
                        s=24,
                        color=color,
                        zorder=3,
                    )
                    if group_line is None:
                        group_line = line
                if group_line is not None:
                    legend_handles.append(group_line)
                    legend_labels.append(group.name)

            if legend_handles:
                self.axes.legend(
                    legend_handles,
                    legend_labels,
                    loc="upper right",
                    frameon=True,
                    framealpha=0.92,
                )

        self._apply_range_to_axes(draw=False)
        self._install_cursor_overlay()
        self.figure_canvas.draw_idle()

    def _confirm(self) -> None:
        self._cancel_pending_render_log()
        self.result = TunableEditorResult(values=dict(self.current_values), saved_path=self.saved_path)
        self.root.destroy()

    def cancel(self) -> None:
        self._cancel_pending_render_log()
        self.cancelled = True
        self.root.destroy()

    @staticmethod
    def _format_value(value: float) -> str:
        return f"{float(value):.12g}"

    def _log_info(self, message: str, *args: Any) -> None:
        if self.logger is not None:
            self.logger.info(message, *args)

    def _log_exception(self, message: str) -> None:
        if self.logger is not None:
            self.logger.exception(message)

    def _schedule_canvas_state_log(self, reason: str) -> None:
        if self._pending_render_log_id is not None:
            try:
                self.root.after_cancel(self._pending_render_log_id)
            except tk.TclError:
                pass
        self._pending_render_log_id = self.root.after(
            120,
            lambda: self._redraw_and_log_canvas_state(reason),
        )

    def _redraw_and_log_canvas_state(self, reason: str) -> None:
        self._pending_render_log_id = None
        if self.figure_canvas is None:
            return
        self.figure_canvas.draw()
        self._log_canvas_state(reason)

    def _cancel_pending_render_log(self) -> None:
        if self._pending_render_log_id is None:
            return
        try:
            self.root.after_cancel(self._pending_render_log_id)
        except tk.TclError:
            pass
        self._pending_render_log_id = None

    def _axes_limits_changed(self, _axes: Any) -> None:
        if self.axes is None or self._syncing_limits:
            return
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        self.range = _PlotRange(
            xmin=float(min(xlim)),
            xmax=float(max(xlim)),
            ymin=float(min(ylim)),
            ymax=float(max(ylim)),
        )

    def _apply_range_to_axes(self, *, draw: bool = True) -> None:
        if self.axes is None:
            return
        self._syncing_limits = True
        try:
            self.axes.set_xlim(self.range.xmin, self.range.xmax)
            self.axes.set_ylim(self.range.ymin, self.range.ymax)
        finally:
            self._syncing_limits = False
        self._update_cursor_overlay(draw=draw)

    def _install_cursor_overlay(self) -> None:
        if self.axes is None:
            return
        self.cursor_vline = self.axes.axvline(
            0.0,
            color="#666666",
            linestyle="--",
            linewidth=0.9,
            alpha=0.8,
            visible=False,
        )
        self.cursor_hline = self.axes.axhline(
            0.0,
            color="#666666",
            linestyle="--",
            linewidth=0.9,
            alpha=0.8,
            visible=False,
        )
        self.cursor_annotation = self.axes.annotate(
            "",
            xy=(0.0, 0.0),
            xytext=(12, 12),
            textcoords="offset points",
            fontfamily="Consolas",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.25",
                "fc": "white",
                "ec": "#909090",
                "alpha": 0.94,
            },
        )
        self.cursor_annotation.set_visible(False)
        self._update_cursor_overlay(draw=False)

    def _update_cursor_overlay(self, *, draw: bool = True) -> None:
        if (
            self.axes is None
            or self.figure_canvas is None
            or self.cursor_vline is None
            or self.cursor_hline is None
            or self.cursor_annotation is None
        ):
            return

        if self.last_cursor is None:
            self.cursor_vline.set_visible(False)
            self.cursor_hline.set_visible(False)
            self.cursor_annotation.set_visible(False)
            if draw:
                self.figure_canvas.draw_idle()
            return

        xdata, ydata = self.last_cursor
        xmin, xmax = self.axes.get_xlim()
        ymin, ymax = self.axes.get_ylim()
        if not (xmin <= xdata <= xmax and ymin <= ydata <= ymax):
            self.cursor_vline.set_visible(False)
            self.cursor_hline.set_visible(False)
            self.cursor_annotation.set_visible(False)
            if draw:
                self.figure_canvas.draw_idle()
            return

        xoffset = 12 if xdata <= xmin + 0.7 * (xmax - xmin) else -112
        yoffset = 12 if ydata <= ymin + 0.7 * (ymax - ymin) else -44
        self.cursor_vline.set_xdata([xdata, xdata])
        self.cursor_hline.set_ydata([ydata, ydata])
        self.cursor_vline.set_visible(True)
        self.cursor_hline.set_visible(True)
        self.cursor_annotation.xy = (xdata, ydata)
        self.cursor_annotation.set_position((xoffset, yoffset))
        self.cursor_annotation.set_text(f"x={xdata:.6f} nm\ny={ydata:.4f} dB")
        self.cursor_annotation.set_visible(True)
        if draw:
            self.figure_canvas.draw_idle()

    def _log_canvas_state(self, reason: str) -> None:
        if self.logger is None or self.axes is None or self.figure_canvas is None:
            return

        lines = self.axes.get_lines()
        detailed_lines: list[str] = []
        for line in lines:
            xdata = np.asarray(line.get_xdata(), dtype=float)
            ydata = np.asarray(line.get_ydata(), dtype=float)
            if xdata.size == 0 or ydata.size == 0:
                continue
            detailed_lines.append(
                f"label={line.get_label()},points={xdata.size},"
                f"start=({xdata[0]:.6f},{ydata[0]:.4f}),end=({xdata[-1]:.6f},{ydata[-1]:.4f})"
            )
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        widget = self.figure_canvas.get_tk_widget()
        self.logger.info(
            "Figure state [%s]: canvas=%dx%d axes_lines=%d collections=%d texts=%d groups=%d xlim=(%.6f, %.6f) ylim=(%.6f, %.6f) detail=%s",
            reason,
            widget.winfo_width(),
            widget.winfo_height(),
            len(lines),
            len(self.axes.collections),
            len(self.axes.texts),
            len(self.groups),
            float(min(xlim)),
            float(max(xlim)),
            float(min(ylim)),
            float(max(ylim)),
            detailed_lines[:5],
        )

class _MatplotlibViewer:
    def __init__(
        self,
        groups: Sequence[CurveGroup],
        *,
        mode: str,
        shared_baseline: bool,
        title: str,
        logger: logging.Logger | None = None,
    ) -> None:
        self.groups = _normalize_groups(groups)
        self.mode = mode
        self.shared_baseline_requested = shared_baseline
        self.logger = logger
        self.colors = {
            group.name: group.color or PALETTE[index % len(PALETTE)]
            for index, group in enumerate(self.groups)
        }
        if mode == "single":
            self.targets = [GLOBAL_TARGET]
            self.target_category_by_name: dict[str, str] = {GLOBAL_TARGET: GLOBAL_TARGET}
            self.targets_by_category: dict[str, list[str]] = {GLOBAL_TARGET: [GLOBAL_TARGET]}
            self.target_categories = (GLOBAL_TARGET,)
        else:
            self.targets = [group.name for group in self.groups]
            self.target_category_by_name = {}
            self.targets_by_category = {}
            for group in self.groups:
                category = str(group.metadata.get("source_bundle", group.name))
                self.target_category_by_name[group.name] = category
                self.targets_by_category.setdefault(category, []).append(group.name)
            self.target_categories = tuple(self.targets_by_category.keys())
        self.default_range = _default_range(self.groups)
        self.range = self.default_range
        self.result: ViewerSelectionResult | None = None
        self.cancelled = False
        self._syncing_limits = False
        self.figure: Any | None = None
        self.axes: Any | None = None
        self.figure_canvas: Any | None = None
        self.toolbar: Any | None = None
        self.cursor_vline: Any | None = None
        self.cursor_hline: Any | None = None
        self.cursor_annotation: Any | None = None
        self.controls_canvas: tk.Canvas | None = None
        self.controls_inner: ttk.Frame | None = None
        self.controls_window: int | None = None

        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            raise RuntimeError("Unable to open tkinter calibration viewer.") from exc

        self.root.title(title)
        self.root.geometry("1380x840")
        self.root.minsize(1080, 720)
        self.root.protocol("WM_DELETE_WINDOW", self.cancel)

        self.visible_vars = {group.name: tk.BooleanVar(value=True) for group in self.groups}
        self.active_target = tk.StringVar(value=self.targets[0])
        self.active_category = tk.StringVar(value=self.target_category_by_name[self.targets[0]])
        self.selection_slot = tk.StringVar(value="through")
        self.cursor_x = tk.StringVar(value="--")
        self.cursor_y = tk.StringVar(value="--")
        self.through_wavelength_var = tk.StringVar()
        self.extinction_wavelength_var = tk.StringVar()
        self.status = tk.StringVar(
            value=(
                "Select 'Through' or 'Extinction' on the left, then left-click the plot to "
                "record that wavelength. Use the Matplotlib toolbar to zoom or pan."
            )
        )

        self.assignments: dict[str, SelectionRecord] = {}
        self.last_cursor: tuple[float, float] | None = None
        self.active_target_combo: ttk.Combobox | None = None
        self.active_category_combo: ttk.Combobox | None = None

        self.tree: ttk.Treeview
        self._build_ui()
        self._refresh_tree()
        if self.mode == "per-variable":
            self._activate_target(self.targets[0], show_active_category=True)
        else:
            self._load_active_selection()
        self._redraw()
        if self.shared_baseline_requested:
            self._log_info(
                "shared_baseline=True was requested, but the viewer now always captures per-target through/extinction wavelengths."
            )
        self._log_info(
            "Calibration viewer initialized: mode=%s groups=%s",
            self.mode,
            [group.name for group in self.groups],
        )

    def show(self) -> ViewerSelectionResult:
        self.root.mainloop()
        if self.result is not None:
            self._log_info("Calibration viewer closed with %d selection(s).", len(self.result.selections))
            return self.result
        if self.cancelled:
            self._log_info("Calibration viewer cancelled by user.")
            raise SelectionCancelledError("Calibration selection was cancelled.")
        raise RuntimeError("Calibration viewer closed without a result.")

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        controls_wrap = ttk.Frame(outer, width=350)
        controls_wrap.pack(side="left", fill="y")
        controls_wrap.pack_propagate(False)

        self.controls_canvas = tk.Canvas(controls_wrap, highlightthickness=0, width=330)
        controls_scrollbar = ttk.Scrollbar(
            controls_wrap,
            orient="vertical",
            command=self.controls_canvas.yview,
        )
        self.controls_canvas.configure(yscrollcommand=controls_scrollbar.set)
        self.controls_canvas.pack(side="left", fill="both", expand=True)
        controls_scrollbar.pack(side="left", fill="y", padx=(8, 0))

        self.controls_inner = ttk.Frame(self.controls_canvas, width=330)
        self.controls_window = self.controls_canvas.create_window(
            (0, 0),
            window=self.controls_inner,
            anchor="nw",
            width=330,
        )
        self.controls_inner.bind("<Configure>", self._sync_controls_scroll)
        self.controls_canvas.bind("<Configure>", self._resize_controls_panel)
        self.controls_canvas.bind("<Enter>", self._bind_controls_mousewheel)
        self.controls_canvas.bind("<Leave>", self._unbind_controls_mousewheel)
        controls = self.controls_inner

        plot_wrap = ttk.Frame(outer)
        plot_wrap.pack(side="left", fill="both", expand=True, padx=(12, 0))
        toolbar_wrap = ttk.Frame(plot_wrap)
        toolbar_wrap.pack(fill="x")
        plot_surface = ttk.Frame(plot_wrap)
        plot_surface.pack(fill="both", expand=True)

        ttk.Label(
            controls,
            text=(
                "The viewer plots 20*log10(|E|) in dB. For each curve target, choose a through "
                "wavelength and an extinction wavelength. Extinction ratio will be computed "
                "as P(through) - P(extinction)."
            ),
            wraplength=300,
            justify="left",
        ).pack(fill="x", pady=(0, 12))

        group_frame = ttk.LabelFrame(controls, text="Visible Curves", padding=10)
        group_frame.pack(fill="x", pady=(0, 12))
        for group in self.groups:
            row = ttk.Frame(group_frame)
            row.pack(fill="x", pady=2)
            tk.Label(row, width=2, bg=self.colors[group.name]).pack(side="left", padx=(0, 8))
            ttk.Checkbutton(
                row,
                text=f"{group.name} ({len(group.curves)} curves)",
                variable=self.visible_vars[group.name],
                command=self._redraw,
            ).pack(side="left", fill="x", expand=True)

        buttons = ttk.Frame(group_frame)
        buttons.pack(fill="x", pady=(8, 0))
        ttk.Button(buttons, text="Show All", command=self._show_all).pack(
            side="left", fill="x", expand=True
        )
        ttk.Button(buttons, text="Hide All", command=self._hide_all).pack(
            side="left", fill="x", expand=True, padx=(8, 0)
        )
        ttk.Button(buttons, text="Only Active", command=self._show_only_active).pack(
            side="left", fill="x", expand=True, padx=(8, 0)
        )

        cursor_frame = ttk.LabelFrame(controls, text="Cursor", padding=10)
        cursor_frame.pack(fill="x", pady=(0, 12))
        ttk.Label(cursor_frame, text="Cursor wavelength (nm)").pack(anchor="w")
        ttk.Label(cursor_frame, textvariable=self.cursor_x, font=("Consolas", 10, "bold")).pack(
            anchor="w", pady=(0, 8)
        )
        ttk.Label(cursor_frame, text="Cursor magnitude (dB)").pack(anchor="w")
        ttk.Label(cursor_frame, textvariable=self.cursor_y, font=("Consolas", 10, "bold")).pack(
            anchor="w", pady=(0, 8)
        )
        cursor_buttons = ttk.Frame(cursor_frame)
        cursor_buttons.pack(fill="x")
        ttk.Button(
            cursor_buttons,
            text="Use Through",
            command=lambda: self._use_cursor_for_slot("through"),
        ).pack(
            side="left", fill="x", expand=True
        )
        ttk.Button(
            cursor_buttons,
            text="Use Extinction",
            command=lambda: self._use_cursor_for_slot("extinction"),
        ).pack(
            side="left", fill="x", expand=True, padx=(8, 0)
        )
        ttk.Button(cursor_buttons, text="Reset Zoom", command=self._reset_zoom).pack(
            side="left", fill="x", expand=True, padx=(8, 0)
        )

        select_frame = ttk.LabelFrame(controls, text="Selections", padding=10)
        select_frame.pack(fill="both", expand=True)
        if self.mode == "per-variable":
            if len(self.target_categories) > 1:
                ttk.Label(select_frame, text="Active parameter/group").pack(anchor="w")
                self.active_category_combo = ttk.Combobox(
                    select_frame,
                    textvariable=self.active_category,
                    state="readonly",
                    values=self.target_categories,
                )
                self.active_category_combo.pack(fill="x", pady=(0, 8))
                self.active_category_combo.bind("<<ComboboxSelected>>", self._category_selected)

            ttk.Label(select_frame, text="Active target").pack(anchor="w")
            self.active_target_combo = ttk.Combobox(
                select_frame,
                textvariable=self.active_target,
                state="readonly",
                values=self.targets_by_category[self.active_category.get()],
            )
            self.active_target_combo.pack(fill="x", pady=(0, 8))
            self.active_target_combo.bind("<<ComboboxSelected>>", self._target_selected)

        ttk.Label(select_frame, text="Left click sets").pack(anchor="w")
        slot_frame = ttk.Frame(select_frame)
        slot_frame.pack(fill="x", pady=(0, 8))
        ttk.Radiobutton(
            slot_frame,
            text="Through",
            value="through",
            variable=self.selection_slot,
        ).pack(side="left")
        ttk.Radiobutton(
            slot_frame,
            text="Extinction",
            value="extinction",
            variable=self.selection_slot,
        ).pack(side="left", padx=(12, 0))

        ttk.Label(select_frame, text="Through wavelength (nm)").pack(anchor="w")
        ttk.Entry(select_frame, textvariable=self.through_wavelength_var).pack(fill="x", pady=(0, 8))
        ttk.Label(select_frame, text="Extinction wavelength (nm)").pack(anchor="w")
        ttk.Entry(select_frame, textvariable=self.extinction_wavelength_var).pack(fill="x", pady=(0, 8))

        apply_buttons = ttk.Frame(select_frame)
        apply_buttons.pack(fill="x", pady=(0, 8))
        ttk.Button(apply_buttons, text="Apply", command=self._apply_selection).pack(
            side="left", fill="x", expand=True
        )
        ttk.Button(apply_buttons, text="Clear", command=self._clear_selection).pack(
            side="left", fill="x", expand=True, padx=(8, 0)
        )

        self.tree = ttk.Treeview(
            select_frame,
            columns=("target", "through", "extinction"),
            show="headings",
            height=max(5, min(10, len(self.targets))),
        )
        for column, label, width, anchor in (
            ("target", "Target", 120, "w"),
            ("through", "Through (nm)", 120, "e"),
            ("extinction", "Extinction (nm)", 120, "e"),
        ):
            self.tree.heading(column, text=label)
            self.tree.column(column, width=width, anchor=anchor)
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self._tree_selected)

        footer = ttk.Frame(controls)
        footer.pack(fill="x", pady=(12, 0))
        ttk.Button(footer, text="Confirm", command=self._confirm).pack(
            side="left", fill="x", expand=True
        )
        ttk.Button(footer, text="Cancel", command=self.cancel).pack(
            side="left", fill="x", expand=True, padx=(8, 0)
        )

        self._build_matplotlib_plot(plot_surface, toolbar_wrap)
        ttk.Label(plot_wrap, textvariable=self.status, anchor="w").pack(fill="x", pady=(8, 0))

    def _sync_controls_scroll(self, _event: tk.Event[tk.Misc]) -> None:
        if self.controls_canvas is not None:
            self.controls_canvas.configure(scrollregion=self.controls_canvas.bbox("all"))

    def _resize_controls_panel(self, event: tk.Event[tk.Misc]) -> None:
        if self.controls_canvas is not None and self.controls_window is not None:
            self.controls_canvas.itemconfigure(self.controls_window, width=event.width)

    def _bind_controls_mousewheel(self, _event: tk.Event[tk.Misc]) -> None:
        self.root.bind_all("<MouseWheel>", self._controls_mousewheel, add="+")

    def _unbind_controls_mousewheel(self, _event: tk.Event[tk.Misc]) -> None:
        self.root.unbind_all("<MouseWheel>")

    def _controls_mousewheel(self, event: tk.Event[tk.Misc]) -> None:
        if self.controls_canvas is None:
            return
        delta = getattr(event, "delta", 0)
        if delta == 0:
            return
        self.controls_canvas.yview_scroll(int(-delta / 120), "units")

    def _build_matplotlib_plot(self, plot_surface: ttk.Frame, toolbar_wrap: ttk.Frame) -> None:
        Figure, FigureCanvasTkAgg, NavigationToolbar2Tk = _load_matplotlib_tk()

        self.figure = Figure(figsize=(10.0, 7.0), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=0.10, right=0.98, top=0.97, bottom=0.11)
        self.axes.callbacks.connect("xlim_changed", self._axes_limits_changed)
        self.axes.callbacks.connect("ylim_changed", self._axes_limits_changed)

        self.figure_canvas = FigureCanvasTkAgg(self.figure, master=plot_surface)
        self.figure_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.figure_canvas.mpl_connect("motion_notify_event", self._motion)
        self.figure_canvas.mpl_connect("button_press_event", self._button_press)
        self.figure_canvas.mpl_connect("axes_leave_event", self._leave)
        self.figure_canvas.mpl_connect("figure_leave_event", self._leave)

        self.toolbar = NavigationToolbar2Tk(self.figure_canvas, toolbar_wrap, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(fill="x")
        self._log_info("Matplotlib calibration viewer plot surface initialized.")

    def _show_all(self) -> None:
        for value in self.visible_vars.values():
            value.set(True)
        self._redraw()

    def _hide_all(self) -> None:
        for value in self.visible_vars.values():
            value.set(False)
        self._redraw()

    def _show_only_active(self) -> None:
        if self.mode == "per-variable" and len(self.target_categories) > 1:
            self._show_only_active_category()
            return

        active = self.active_target.get()
        for name, value in self.visible_vars.items():
            value.set(active == GLOBAL_TARGET or name == active)
        self._redraw()

    def _show_only_active_category(self) -> None:
        active_category = self.active_category.get()
        for name, value in self.visible_vars.items():
            value.set(self.target_category_by_name.get(name) == active_category)
        self._redraw()

    def _update_active_target_values(self) -> None:
        if self.active_target_combo is None:
            return
        current_targets = self.targets_by_category.get(self.active_category.get(), self.targets)
        self.active_target_combo.configure(values=current_targets)

    def _first_pending_target(self, targets: Sequence[str]) -> str | None:
        for target in targets:
            if target not in self.assignments:
                return target
        return None

    def _activate_target(self, target: str, *, show_active_category: bool) -> None:
        category = self.target_category_by_name.get(target, self.active_category.get())
        if self.active_category.get() != category:
            self.active_category.set(category)
        self._update_active_target_values()
        self.active_target.set(target)
        if show_active_category:
            for name, value in self.visible_vars.items():
                value.set(self.target_category_by_name.get(name) == category)
        self._load_active_selection()
        self._redraw()

    def _activate_category(self, category: str) -> None:
        if category not in self.targets_by_category:
            return
        self.active_category.set(category)
        self._update_active_target_values()
        target = self._first_pending_target(self.targets_by_category[category])
        if target is None:
            target = self.targets_by_category[category][0]
        self._activate_target(target, show_active_category=True)

    def _advance_to_next_target(self) -> bool:
        current_target = self.active_target.get()
        current_index = self.targets.index(current_target)
        search_order = self.targets[current_index + 1 :] + self.targets[: current_index + 1]
        for target in search_order:
            if target not in self.assignments:
                self._activate_target(target, show_active_category=True)
                self.status.set(
                    f"Moved to the next unfinished target: {target}. Select through first."
                )
                return True
        self.status.set("All curve targets have been assigned. Review selections or confirm.")
        return False

    def _category_selected(self, _event: tk.Event[tk.Misc]) -> None:
        self._activate_category(self.active_category.get())

    def _target_selected(self, _event: tk.Event[tk.Misc]) -> None:
        self._activate_target(self.active_target.get(), show_active_category=True)

    def _use_cursor_for_slot(self, slot: str) -> None:
        if self.last_cursor is None:
            return
        self._set_selection_wavelength(slot, self.last_cursor[0], source="cursor button")
        self._refresh_tree()

    def _set_selection_wavelength(self, slot: str, wavelength_nm: float, *, source: str) -> None:
        formatted = f"{float(wavelength_nm):.6f}"
        if slot == "through":
            self.through_wavelength_var.set(formatted)
            self.selection_slot.set("extinction")
            self.status.set(f"Through wavelength set to {formatted} nm from {source}.")
        else:
            self.extinction_wavelength_var.set(formatted)
            self.status.set(f"Extinction wavelength set to {formatted} nm from {source}.")
        self._log_info(
            "%s set %s wavelength to %.6f for %s.",
            source.capitalize(),
            slot,
            wavelength_nm,
            self.active_target.get(),
        )
        self._apply_selection_from_fields(
            show_error=False,
            reason=source,
            advance_to_next=(slot == "extinction"),
        )

    def _parse_selection_fields(self) -> SelectionRecord:
        through_text = self.through_wavelength_var.get().strip()
        extinction_text = self.extinction_wavelength_var.get().strip()
        if not through_text or not extinction_text:
            raise ValueError("Both through wavelength and extinction wavelength are required.")

        try:
            through_wavelength = float(through_text)
        except ValueError as exc:
            raise ValueError("Through wavelength must be a valid number in nm.") from exc
        try:
            extinction_wavelength = float(extinction_text)
        except ValueError as exc:
            raise ValueError("Extinction wavelength must be a valid number in nm.") from exc

        return SelectionRecord(
            through_wavelength_nm=through_wavelength,
            extinction_wavelength_nm=extinction_wavelength,
        )

    def _apply_selection(self) -> None:
        self._apply_selection_from_fields(
            show_error=True,
            reason="apply button",
            advance_to_next=True,
            finalize_when_complete=True,
        )

    def _apply_selection_from_fields(
        self,
        *,
        show_error: bool,
        reason: str,
        advance_to_next: bool = False,
        finalize_when_complete: bool = False,
    ) -> bool:
        try:
            record = self._parse_selection_fields()
        except ValueError as exc:
            if show_error:
                messagebox.showerror("Invalid selection", str(exc))
            return False

        active_target = self.active_target.get()
        self.assignments[active_target] = record
        self._log_info(
            "Selection applied for %s from %s: through=%.6f extinction=%.6f",
            active_target,
            reason,
            record.through_wavelength_nm,
            record.extinction_wavelength_nm,
        )
        self._refresh_tree()
        if advance_to_next and self._advance_to_next_target():
            return True
        if finalize_when_complete and self._all_targets_assigned():
            return self._finalize_result(show_error=show_error)
        self._redraw()
        return True

    def _clear_selection(self) -> None:
        active_target = self.active_target.get()
        self.assignments.pop(active_target, None)
        self.through_wavelength_var.set("")
        self.extinction_wavelength_var.set("")
        self._log_info("Selection cleared for %s.", active_target)
        self._refresh_tree()
        self._redraw()

    def _load_active_selection(self) -> None:
        active_target = self.active_target.get()
        if self.tree.exists(active_target):
            self.tree.selection_set(active_target)
            self.tree.focus(active_target)
            self.tree.see(active_target)
        self.selection_slot.set("through")
        record = self.assignments.get(active_target)
        if record is None:
            self.through_wavelength_var.set("")
            self.extinction_wavelength_var.set("")
            self.status.set(
                "Active target changed. Select the through wavelength first, then the extinction wavelength."
            )
            return
        self.through_wavelength_var.set(f"{record.through_wavelength_nm:.6f}")
        self.extinction_wavelength_var.set(f"{record.extinction_wavelength_nm:.6f}")
        self.status.set(
            "Active target changed. Through/extinction values loaded; the next left-click will set through."
        )

    def _refresh_tree(self) -> None:
        self.tree.delete(*self.tree.get_children())
        for target in self.targets:
            record = self.assignments.get(target)
            through_text = "--" if record is None else f"{record.through_wavelength_nm:.6f}"
            extinction_text = "--" if record is None else f"{record.extinction_wavelength_nm:.6f}"
            self.tree.insert(
                "",
                "end",
                iid=target,
                values=("global" if target == GLOBAL_TARGET else target, through_text, extinction_text),
            )
        active_target = self.active_target.get()
        if self.tree.exists(active_target):
            self.tree.selection_set(active_target)
            self.tree.focus(active_target)

    def _tree_selected(self, _event: tk.Event[tk.Misc]) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        target = selection[0]
        if target in self.targets and target != self.active_target.get():
            self._activate_target(target, show_active_category=True)

    def _motion(self, event: Any) -> None:
        if self.axes is None or event.inaxes is not self.axes or event.xdata is None or event.ydata is None:
            self._leave()
            return
        self.last_cursor = (float(event.xdata), float(event.ydata))
        self.cursor_x.set(f"{self.last_cursor[0]:.6f}")
        self.cursor_y.set(f"{self.last_cursor[1]:.4f}")
        self._update_cursor_overlay()

    def _button_press(self, event: Any) -> None:
        if (
            self.axes is None
            or event.inaxes is not self.axes
            or event.xdata is None
        ):
            return

        button = getattr(event, "button", None)
        if not (button == 1 or str(button).endswith("LEFT")):
            return

        toolbar_mode = str(getattr(self.toolbar, "mode", "")).strip().lower()
        if toolbar_mode:
            self._log_info("Left-click ignored because toolbar mode is active: %s", toolbar_mode)
            return

        xdata = float(event.xdata)
        ydata = float(event.ydata) if event.ydata is not None else float("nan")
        self.last_cursor = (xdata, ydata)
        self.cursor_x.set(f"{xdata:.6f}")
        self.cursor_y.set("--" if not math.isfinite(ydata) else f"{ydata:.4f}")
        slot = self.selection_slot.get()
        self._set_selection_wavelength(slot, xdata, source="left click")
        self._refresh_tree()
        self._update_cursor_overlay()

    def _leave(self, _event: Any | None = None) -> None:
        self.last_cursor = None
        self.cursor_x.set("--")
        self.cursor_y.set("--")
        self._update_cursor_overlay()

    def _reset_zoom(self) -> None:
        self.range = self.default_range
        self.status.set("View reset to the full visible range.")
        self._log_info("Viewer zoom reset to default range.")
        self._apply_range_to_axes()

    def _confirm(self) -> None:
        self._apply_selection_from_fields(show_error=False, reason="confirm")
        self._finalize_result(show_error=True)

    def _all_targets_assigned(self) -> bool:
        return all(target in self.assignments for target in self.targets)

    def _finalize_result(self, *, show_error: bool) -> bool:
        selections: dict[str, SelectionRecord] = {}
        for target in self.targets:
            record = self.assignments.get(target)
            if record is None:
                if show_error:
                    messagebox.showerror(
                        "Missing selection",
                        (
                            f"Selection for '{'global' if target == GLOBAL_TARGET else target}' is missing. "
                            "Each target needs both a through wavelength and an extinction wavelength."
                        ),
                    )
                return False
            selections[target] = record
        self.result = ViewerSelectionResult(
            selections=selections,
            visible_groups=tuple(name for name, var in self.visible_vars.items() if var.get()),
            x_limits_nm=(self.range.xmin, self.range.xmax),
            y_limits_db=(self.range.ymin, self.range.ymax),
        )
        self._log_info(
            "Viewer confirmation completed: visible_groups=%s x_limits=%s y_limits=%s",
            self.result.visible_groups,
            self.result.x_limits_nm,
            self.result.y_limits_db,
        )
        self.root.destroy()
        return True

    def cancel(self) -> None:
        self.cancelled = True
        self.root.destroy()

    def _log_info(self, message: str, *args: Any) -> None:
        if self.logger is not None:
            self.logger.info(message, *args)

    def _log_exception(self, message: str) -> None:
        if self.logger is not None:
            self.logger.exception(message)

    def _axes_limits_changed(self, _axes: Any) -> None:
        if self.axes is None or self._syncing_limits:
            return
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        self.range = _PlotRange(
            xmin=float(min(xlim)),
            xmax=float(max(xlim)),
            ymin=float(min(ylim)),
            ymax=float(max(ylim)),
        )

    def _apply_range_to_axes(self, *, draw: bool = True) -> None:
        if self.axes is None:
            return
        self._syncing_limits = True
        try:
            self.axes.set_xlim(self.range.xmin, self.range.xmax)
            self.axes.set_ylim(self.range.ymin, self.range.ymax)
        finally:
            self._syncing_limits = False
        self._update_cursor_overlay(draw=draw)

    def _install_cursor_overlay(self) -> None:
        if self.axes is None:
            return
        self.cursor_vline = self.axes.axvline(
            0.0,
            color="#666666",
            linestyle="--",
            linewidth=0.9,
            alpha=0.8,
            visible=False,
        )
        self.cursor_hline = self.axes.axhline(
            0.0,
            color="#666666",
            linestyle="--",
            linewidth=0.9,
            alpha=0.8,
            visible=False,
        )
        self.cursor_annotation = self.axes.annotate(
            "",
            xy=(0.0, 0.0),
            xytext=(12, 12),
            textcoords="offset points",
            fontfamily="Consolas",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.25",
                "fc": "white",
                "ec": "#909090",
                "alpha": 0.94,
            },
        )
        self.cursor_annotation.set_visible(False)
        self._update_cursor_overlay(draw=False)

    def _update_cursor_overlay(self, *, draw: bool = True) -> None:
        if (
            self.axes is None
            or self.figure_canvas is None
            or self.cursor_vline is None
            or self.cursor_hline is None
            or self.cursor_annotation is None
        ):
            return

        if self.last_cursor is None:
            self.cursor_vline.set_visible(False)
            self.cursor_hline.set_visible(False)
            self.cursor_annotation.set_visible(False)
            if draw:
                self.figure_canvas.draw_idle()
            return

        xdata, ydata = self.last_cursor
        xmin, xmax = self.axes.get_xlim()
        ymin, ymax = self.axes.get_ylim()
        if not (xmin <= xdata <= xmax and ymin <= ydata <= ymax):
            self.cursor_vline.set_visible(False)
            self.cursor_hline.set_visible(False)
            self.cursor_annotation.set_visible(False)
            if draw:
                self.figure_canvas.draw_idle()
            return

        xoffset = 12 if xdata <= xmin + 0.7 * (xmax - xmin) else -112
        yoffset = 12 if ydata <= ymin + 0.7 * (ymax - ymin) else -44
        self.cursor_vline.set_xdata([xdata, xdata])
        self.cursor_hline.set_ydata([ydata, ydata])
        self.cursor_vline.set_visible(True)
        self.cursor_hline.set_visible(True)
        self.cursor_annotation.xy = (xdata, ydata)
        self.cursor_annotation.set_position((xoffset, yoffset))
        self.cursor_annotation.set_text(f"x={xdata:.6f} nm\ny={ydata:.4f} dB")
        self.cursor_annotation.set_visible(True)
        if draw:
            self.figure_canvas.draw_idle()

    def _redraw(self) -> None:
        if self.axes is None or self.figure_canvas is None:
            return

        preserved_range = self.range
        self._syncing_limits = True
        try:
            self.axes.clear()
            self.axes.set_facecolor("white")
            self.axes.grid(True, color="#d7d7d7", linewidth=0.8, alpha=0.65)
            self.axes.set_xlabel("Wavelength (nm)")
            self.axes.set_ylabel("20*log10(|E|) (dB)")

            visible = [group for group in self.groups if self.visible_vars[group.name].get()]
            legend_handles: list[Any] = []
            legend_labels: list[str] = []
            if not visible:
                self.axes.text(
                    0.5,
                    0.5,
                    "No curves visible.",
                    transform=self.axes.transAxes,
                    ha="center",
                    va="center",
                )
            else:
                for group in visible:
                    group_line: Any | None = None
                    for index, curve in enumerate(group.curves):
                        order = np.argsort(curve.wavelength_nm)
                        x = curve.wavelength_nm[order]
                        y = curve.magnitude_db()[order]
                        if x.size < 2:
                            continue
                        color = _shade(self.colors[group.name], index, len(group.curves))
                        (line,) = self.axes.plot(x, y, color=color, linewidth=2.0, alpha=0.95)
                        if group_line is None:
                            group_line = line
                    if group_line is not None:
                        legend_handles.append(group_line)
                        legend_labels.append(group.name)

                if legend_handles:
                    self.axes.legend(
                        legend_handles,
                        legend_labels,
                        loc="upper right",
                        frameon=True,
                        framealpha=0.92,
                    )

            active_target = self.active_target.get()
            for target, record in self.assignments.items():
                if target != active_target:
                    continue
                color = "#111111" if target == GLOBAL_TARGET else self.colors[target]
                label = "global" if target == GLOBAL_TARGET else target
                self.axes.axvline(
                    record.through_wavelength_nm,
                    color=color,
                    linestyle="-",
                    linewidth=1.1,
                    alpha=0.9,
                )
                self.axes.axvline(
                    record.extinction_wavelength_nm,
                    color=color,
                    linestyle="--",
                    linewidth=1.1,
                    alpha=0.9,
                )
                self.axes.annotate(
                    f"{label} through={record.through_wavelength_nm:.4f}",
                    xy=(record.through_wavelength_nm, 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(6, -8),
                    textcoords="offset points",
                    ha="left",
                    va="top",
                    fontsize=9,
                    fontweight="bold",
                    color=color,
                    bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": color, "alpha": 0.88},
                )
                self.axes.annotate(
                    f"{label} extinction={record.extinction_wavelength_nm:.4f}",
                    xy=(record.extinction_wavelength_nm, 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(6, -26),
                    textcoords="offset points",
                    ha="left",
                    va="top",
                    fontsize=9,
                    fontweight="bold",
                    color=color,
                    bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": color, "alpha": 0.88},
                )

            self.axes.set_xlim(preserved_range.xmin, preserved_range.xmax)
            self.axes.set_ylim(preserved_range.ymin, preserved_range.ymax)
        finally:
            self._syncing_limits = False

        self._install_cursor_overlay()
        self.figure_canvas.draw_idle()
        self._log_plot_state("after-redraw")

    def _log_plot_state(self, reason: str) -> None:
        if self.logger is None or self.axes is None or self.figure_canvas is None:
            return

        self.figure_canvas.draw()
        lines = self.axes.get_lines()
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        self.logger.info(
            "Calibration figure state [%s]: canvas=%dx%d axes_lines=%d collections=%d texts=%d visible_groups=%s xlim=(%.6f, %.6f) ylim=(%.6f, %.6f)",
            reason,
            self.figure_canvas.get_tk_widget().winfo_width(),
            self.figure_canvas.get_tk_widget().winfo_height(),
            len(lines),
            len(self.axes.collections),
            len(self.axes.texts),
            [name for name, var in self.visible_vars.items() if var.get()],
            float(min(xlim)),
            float(max(xlim)),
            float(min(ylim)),
            float(max(ylim)),
        )


def _run_viewer(
    groups: Sequence[CurveGroup] | Mapping[str, CurveGroup],
    *,
    mode: str,
    shared_baseline: bool,
    title: str,
    logger: logging.Logger | None = None,
) -> ViewerSelectionResult:
    sequence = list(groups.values()) if isinstance(groups, Mapping) else list(groups)
    return _MatplotlibViewer(
        sequence,
        mode=mode,
        shared_baseline=shared_baseline,
        title=title,
        logger=logger,
    ).show()


def _normalize_groups(groups: Sequence[CurveGroup]) -> list[CurveGroup]:
    if not groups:
        raise ValueError("At least one curve group is required.")
    result = []
    seen: set[str] = set()
    for group in groups:
        if group.name in seen:
            raise ValueError(f"Duplicate curve group name: {group.name}")
        seen.add(group.name)
        if not group.curves:
            raise ValueError(f"Curve group '{group.name}' must contain at least one curve.")
        curves = []
        for curve in group.curves:
            wavelength = _as_float_1d(curve.wavelength_nm, "curve.wavelength_nm")
            amplitude = np.asarray(curve.complex_amplitude, dtype=np.complex128)
            if amplitude.ndim != 1 or amplitude.size != wavelength.size:
                raise ValueError("Each curve must be 1D and match wavelength length.")
            curves.append(
                ComplexCurve(
                    wavelength_nm=wavelength,
                    complex_amplitude=amplitude,
                    label=curve.label,
                    sweep_value=curve.sweep_value,
                )
            )
        result.append(CurveGroup(group.name, tuple(curves), group.color, group.metadata))
    return result


def _default_range(groups: Sequence[CurveGroup]) -> _PlotRange:
    wavelengths = np.concatenate([curve.wavelength_nm for group in groups for curve in group.curves])
    magnitudes = np.concatenate([curve.magnitude_db() for group in groups for curve in group.curves])
    return _PlotRange(
        xmin=float(np.min(wavelengths)),
        xmax=float(np.max(wavelengths)),
        ymin=float(np.min(magnitudes)),
        ymax=float(np.max(magnitudes)),
    ).padded()


def _as_float_1d(values: ArrayLike, field_name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 2:
        raise ValueError(f"{field_name} must be a 1D array with at least two samples.")
    return array.astype(np.float64, copy=False)


def _shade(color: str, index: int, total: int) -> str:
    if total <= 1 or not color.startswith("#") or len(color) != 7:
        return color
    red = int(color[1:3], 16) / 255.0
    green = int(color[3:5], 16) / 255.0
    blue = int(color[5:7], 16) / 255.0
    hue, lightness, saturation = colorsys.rgb_to_hls(red, green, blue)
    factor = 0.82 + 0.18 * (index / max(total - 1, 1))
    lightness = max(0.0, min(1.0, lightness * factor))
    red, green, blue = colorsys.hls_to_rgb(hue, lightness, saturation)
    return "#{:02x}{:02x}{:02x}".format(
        int(round(red * 255)),
        int(round(green * 255)),
        int(round(blue * 255)),
    )


def _demo_groups() -> list[CurveGroup]:
    wavelength = np.linspace(1549.92, 1550.22, 700)
    groups: list[CurveGroup] = []
    for index, (name, center, sigma) in enumerate(
        (("theta_i", 1550.03, 0.0045), ("theta_o", 1550.08, 0.0050), ("v3", 1550.14, 0.0060))
    ):
        curves = []
        for sweep in np.linspace(-1.0, 1.0, 5):
            shifted = center + 0.012 * sweep
            phase = 0.6 * sweep + 0.4 * np.sin((wavelength - shifted) * 60.0)
            magnitude = 1.0 - 0.75 * np.exp(-((wavelength - shifted) ** 2) / (2 * sigma**2))
            magnitude = np.clip(magnitude, 0.03, None)
            curves.append(
                ComplexCurve(
                    wavelength_nm=wavelength,
                    complex_amplitude=magnitude * np.exp(1j * phase),
                    label=f"{name}={sweep:.2f}",
                    sweep_value=float(sweep),
                )
            )
        groups.append(CurveGroup(name=name, curves=tuple(curves), color=PALETTE[index]))
    return groups


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Interactive viewer for calibration complex-response curves.")
    parser.add_argument("--mode", choices=("single", "per-variable"), default="single")
    parser.add_argument("--shared-baseline", action="store_true")
    parser.add_argument("--title", default="Calibration Curve Viewer")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.demo:
        parser.error("Only --demo is supported from the CLI. Import the module for real data.")

    try:
        if args.mode == "single":
            result = select_extinction_reference(_demo_groups(), title=args.title)
            print(
                {
                    "through_wavelength_nm": result.through_wavelength_nm,
                    "extinction_wavelength_nm": result.extinction_wavelength_nm,
                }
            )
        else:
            result = select_variable_targets(
                _demo_groups(),
                title=args.title,
                shared_baseline=args.shared_baseline,
            )
            print(
                {
                    "visible_groups": list(result.visible_groups),
                    "selections": {
                        key: {
                            "through_wavelength_nm": value.through_wavelength_nm,
                            "extinction_wavelength_nm": value.extinction_wavelength_nm,
                        }
                        for key, value in result.selections.items()
                    },
                }
            )
    except SelectionCancelledError:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
