from __future__ import annotations

import argparse
import colorsys
import math
import tkinter as tk
from dataclasses import dataclass, field
from tkinter import messagebox, ttk
from typing import Any, Mapping, Sequence

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
    wavelength_nm: float
    baseline_db: float


@dataclass(frozen=True, slots=True)
class ViewerSelectionResult:
    selections: dict[str, SelectionRecord]
    visible_groups: tuple[str, ...]
    x_limits_nm: tuple[float, float]
    y_limits_db: tuple[float, float]
    shared_baseline_db: float | None = None


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
) -> SelectionRecord:
    result = _run_viewer(groups, mode="single", shared_baseline=False, title=title)
    return result.selections[GLOBAL_TARGET]


def select_variable_targets(
    groups: Sequence[CurveGroup] | Mapping[str, CurveGroup],
    *,
    title: str = "Select Calibration Targets",
    shared_baseline: bool = True,
) -> ViewerSelectionResult:
    return _run_viewer(groups, mode="per-variable", shared_baseline=shared_baseline, title=title)


class _Viewer:
    def __init__(
        self,
        groups: Sequence[CurveGroup],
        *,
        mode: str,
        shared_baseline: bool,
        title: str,
    ) -> None:
        self.groups = _normalize_groups(groups)
        self.mode = mode
        self.shared_baseline = shared_baseline
        self.colors = {
            group.name: group.color or PALETTE[index % len(PALETTE)]
            for index, group in enumerate(self.groups)
        }
        self.targets = [GLOBAL_TARGET] if mode == "single" else [group.name for group in self.groups]
        self.default_range = _default_range(self.groups)
        self.range = self.default_range
        self.result: ViewerSelectionResult | None = None
        self.cancelled = False

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
        self.cursor_x = tk.StringVar(value="--")
        self.cursor_y = tk.StringVar(value="--")
        self.wavelength_var = tk.StringVar()
        self.baseline_var = tk.StringVar()
        self.shared_baseline_var = tk.StringVar()
        self.status = tk.StringVar(
            value="Move for crosshair. Drag with left mouse to zoom. Right click resets zoom."
        )

        self.assignments: dict[str, SelectionRecord] = {}
        self.last_cursor: tuple[float, float] | None = None
        self.zoom_start: tuple[int, int] | None = None
        self.zoom_box: int | None = None

        self.left_margin = 84
        self.top_margin = 24
        self.right_margin = 24
        self.bottom_margin = 58

        self.canvas = tk.Canvas(self.root, bg="white", highlightthickness=0)
        self.tree: ttk.Treeview
        self._build_ui()
        self._bind_canvas()
        self.active_target.trace_add("write", lambda *_args: self._load_active_selection())
        self.shared_baseline_var.trace_add("write", lambda *_args: self._shared_baseline_changed())
        self._refresh_tree()
        self._redraw()

    def show(self) -> ViewerSelectionResult:
        self.root.mainloop()
        if self.result is not None:
            return self.result
        if self.cancelled:
            raise SelectionCancelledError("Calibration selection was cancelled.")
        raise RuntimeError("Calibration viewer closed without a result.")

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        controls = ttk.Frame(outer, width=330)
        controls.pack(side="left", fill="y")
        controls.pack_propagate(False)

        plot_wrap = ttk.Frame(outer)
        plot_wrap.pack(side="left", fill="both", expand=True, padx=(12, 0))

        ttk.Label(
            controls,
            text=(
                "Input curves are complex amplitudes. The viewer plots 20*log10(|E|) in dB so "
                "you can pick wavelength positions and extinction-ratio baselines directly."
            ),
            wraplength=300,
            justify="left",
        ).pack(fill="x", pady=(0, 12))

        group_frame = ttk.LabelFrame(controls, text="Visible Variables", padding=10)
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
        ttk.Button(cursor_buttons, text="Use X", command=self._use_cursor_x).pack(
            side="left", fill="x", expand=True
        )
        ttk.Button(cursor_buttons, text="Use Y", command=self._use_cursor_y).pack(
            side="left", fill="x", expand=True, padx=(8, 0)
        )
        ttk.Button(cursor_buttons, text="Reset Zoom", command=self._reset_zoom).pack(
            side="left", fill="x", expand=True, padx=(8, 0)
        )

        select_frame = ttk.LabelFrame(controls, text="Selections", padding=10)
        select_frame.pack(fill="both", expand=True)
        if self.mode == "per-variable":
            ttk.Label(select_frame, text="Active variable").pack(anchor="w")
            ttk.Combobox(
                select_frame,
                textvariable=self.active_target,
                state="readonly",
                values=self.targets,
            ).pack(fill="x", pady=(0, 8))

        ttk.Label(select_frame, text="Wavelength (nm)").pack(anchor="w")
        ttk.Entry(select_frame, textvariable=self.wavelength_var).pack(fill="x", pady=(0, 8))
        if self.shared_baseline:
            ttk.Label(select_frame, text="Shared baseline (dB)").pack(anchor="w")
            ttk.Entry(select_frame, textvariable=self.shared_baseline_var).pack(fill="x", pady=(0, 8))
        else:
            ttk.Label(select_frame, text="Baseline (dB)").pack(anchor="w")
            ttk.Entry(select_frame, textvariable=self.baseline_var).pack(fill="x", pady=(0, 8))

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
            columns=("target", "wavelength", "baseline"),
            show="headings",
            height=max(5, min(10, len(self.targets))),
        )
        for column, title, width, anchor in (
            ("target", "Target", 120, "w"),
            ("wavelength", "Wavelength (nm)", 120, "e"),
            ("baseline", "Baseline (dB)", 120, "e"),
        ):
            self.tree.heading(column, text=title)
            self.tree.column(column, width=width, anchor=anchor)
        self.tree.pack(fill="both", expand=True)

        footer = ttk.Frame(controls)
        footer.pack(fill="x", pady=(12, 0))
        ttk.Button(footer, text="Confirm", command=self._confirm).pack(
            side="left", fill="x", expand=True
        )
        ttk.Button(footer, text="Cancel", command=self.cancel).pack(
            side="left", fill="x", expand=True, padx=(8, 0)
        )

        self.canvas.pack(in_=plot_wrap, fill="both", expand=True)
        ttk.Label(plot_wrap, textvariable=self.status, anchor="w").pack(fill="x", pady=(8, 0))

    def _bind_canvas(self) -> None:
        self.canvas.bind("<Configure>", lambda _e: self._redraw())
        self.canvas.bind("<Motion>", self._motion)
        self.canvas.bind("<Leave>", self._leave)
        self.canvas.bind("<ButtonPress-1>", self._zoom_press)
        self.canvas.bind("<B1-Motion>", self._zoom_drag)
        self.canvas.bind("<ButtonRelease-1>", self._zoom_release)
        self.canvas.bind("<Button-3>", lambda _e: self._reset_zoom())

    def _show_all(self) -> None:
        for value in self.visible_vars.values():
            value.set(True)
        self._redraw()

    def _hide_all(self) -> None:
        for value in self.visible_vars.values():
            value.set(False)
        self._redraw()

    def _show_only_active(self) -> None:
        active = self.active_target.get()
        for name, value in self.visible_vars.items():
            value.set(active == GLOBAL_TARGET or name == active)
        self._redraw()

    def _use_cursor_x(self) -> None:
        if self.last_cursor is not None:
            self.wavelength_var.set(f"{self.last_cursor[0]:.6f}")

    def _use_cursor_y(self) -> None:
        if self.last_cursor is None:
            return
        if self.shared_baseline:
            self.shared_baseline_var.set(f"{self.last_cursor[1]:.4f}")
        else:
            self.baseline_var.set(f"{self.last_cursor[1]:.4f}")

    def _apply_selection(self) -> None:
        try:
            wavelength = float(self.wavelength_var.get())
        except ValueError:
            messagebox.showerror("Invalid wavelength", "Please enter a valid wavelength in nm.")
            return

        try:
            baseline = (
                float(self.shared_baseline_var.get())
                if self.shared_baseline
                else float(self.baseline_var.get())
            )
        except ValueError:
            messagebox.showerror("Invalid baseline", "Please enter a valid baseline in dB.")
            return

        self.assignments[self.active_target.get()] = SelectionRecord(wavelength, baseline)
        self._refresh_tree()
        self._redraw()

    def _clear_selection(self) -> None:
        self.assignments.pop(self.active_target.get(), None)
        self.wavelength_var.set("")
        if not self.shared_baseline:
            self.baseline_var.set("")
        self._refresh_tree()
        self._redraw()

    def _load_active_selection(self) -> None:
        record = self.assignments.get(self.active_target.get())
        if record is None:
            self.wavelength_var.set("")
            if not self.shared_baseline:
                self.baseline_var.set("")
            return
        self.wavelength_var.set(f"{record.wavelength_nm:.6f}")
        if not self.shared_baseline:
            self.baseline_var.set(f"{record.baseline_db:.4f}")

    def _shared_baseline_changed(self) -> None:
        if self.shared_baseline:
            self._refresh_tree()
            self._redraw()

    def _refresh_tree(self) -> None:
        self.tree.delete(*self.tree.get_children())
        shared = self._shared_baseline()
        for target in self.targets:
            record = self.assignments.get(target)
            wavelength_text = "--" if record is None else f"{record.wavelength_nm:.6f}"
            baseline_text = "--"
            if record is not None:
                baseline_text = f"{(shared if shared is not None else record.baseline_db):.4f}"
            elif shared is not None:
                baseline_text = f"{shared:.4f}"
            self.tree.insert(
                "",
                "end",
                values=("global" if target == GLOBAL_TARGET else target, wavelength_text, baseline_text),
            )

    def _motion(self, event: tk.Event[tk.Misc]) -> None:
        if not self._in_plot(event.x, event.y):
            self.last_cursor = None
            self.cursor_x.set("--")
            self.cursor_y.set("--")
            self._redraw()
            return
        xdata, ydata = self._to_data(event.x, event.y)
        self.last_cursor = (xdata, ydata)
        self.cursor_x.set(f"{xdata:.6f}")
        self.cursor_y.set(f"{ydata:.4f}")
        self._redraw()

    def _leave(self, _event: tk.Event[tk.Misc]) -> None:
        self.last_cursor = None
        self.cursor_x.set("--")
        self.cursor_y.set("--")
        self._redraw()

    def _zoom_press(self, event: tk.Event[tk.Misc]) -> None:
        if self._in_plot(event.x, event.y):
            self.zoom_start = (event.x, event.y)

    def _zoom_drag(self, event: tk.Event[tk.Misc]) -> None:
        if self.zoom_start is None:
            return
        if self.zoom_box is not None:
            self.canvas.delete(self.zoom_box)
        x0, y0 = self.zoom_start
        self.zoom_box = self.canvas.create_rectangle(x0, y0, event.x, event.y, dash=(4, 2))

    def _zoom_release(self, event: tk.Event[tk.Misc]) -> None:
        if self.zoom_start is None:
            return
        x0, y0 = self.zoom_start
        self.zoom_start = None
        if self.zoom_box is not None:
            self.canvas.delete(self.zoom_box)
            self.zoom_box = None
        if abs(event.x - x0) < 10 or abs(event.y - y0) < 10:
            return
        if not (self._in_plot(x0, y0) and self._in_plot(event.x, event.y)):
            return
        d0 = self._to_data(x0, y0)
        d1 = self._to_data(event.x, event.y)
        self.range = _PlotRange(
            xmin=min(d0[0], d1[0]),
            xmax=max(d0[0], d1[0]),
            ymin=min(d0[1], d1[1]),
            ymax=max(d0[1], d1[1]),
        )
        self.status.set("Zoom applied. Move for crosshair. Right click resets zoom.")
        self._redraw()

    def _reset_zoom(self) -> None:
        self.range = self.default_range
        self.status.set("Move for crosshair. Drag with left mouse to zoom. Right click resets zoom.")
        self._redraw()

    def _confirm(self) -> None:
        shared = self._shared_baseline()
        selections: dict[str, SelectionRecord] = {}
        for target in self.targets:
            record = self.assignments.get(target)
            if record is None:
                messagebox.showerror(
                    "Missing selection",
                    f"Selection for '{'global' if target == GLOBAL_TARGET else target}' is missing.",
                )
                return
            selections[target] = SelectionRecord(
                wavelength_nm=record.wavelength_nm,
                baseline_db=shared if shared is not None else record.baseline_db,
            )
        self.result = ViewerSelectionResult(
            selections=selections,
            visible_groups=tuple(name for name, var in self.visible_vars.items() if var.get()),
            x_limits_nm=(self.range.xmin, self.range.xmax),
            y_limits_db=(self.range.ymin, self.range.ymax),
            shared_baseline_db=shared,
        )
        self.root.destroy()

    def cancel(self) -> None:
        self.cancelled = True
        self.root.destroy()

    def _shared_baseline(self) -> float | None:
        if not self.shared_baseline:
            return None
        try:
            return float(self.shared_baseline_var.get())
        except ValueError:
            return None

    def _plot_box(self) -> tuple[int, int, int, int]:
        width = max(self.canvas.winfo_width(), 400)
        height = max(self.canvas.winfo_height(), 300)
        left = self.left_margin
        top = self.top_margin
        right = width - self.right_margin
        bottom = height - self.bottom_margin
        return left, top, right, bottom

    def _in_plot(self, x: int, y: int) -> bool:
        left, top, right, bottom = self._plot_box()
        return left <= x <= right and top <= y <= bottom

    def _to_canvas(self, xdata: float, ydata: float) -> tuple[float, float]:
        left, top, right, bottom = self._plot_box()
        xspan = self.range.xmax - self.range.xmin
        yspan = self.range.ymax - self.range.ymin
        xratio = 0.0 if xspan <= 0 else (xdata - self.range.xmin) / xspan
        yratio = 0.0 if yspan <= 0 else (self.range.ymax - ydata) / yspan
        return left + xratio * (right - left), top + yratio * (bottom - top)

    def _to_data(self, x: int, y: int) -> tuple[float, float]:
        left, top, right, bottom = self._plot_box()
        xratio = (x - left) / max(right - left, 1)
        yratio = (y - top) / max(bottom - top, 1)
        return (
            self.range.xmin + xratio * (self.range.xmax - self.range.xmin),
            self.range.ymax - yratio * (self.range.ymax - self.range.ymin),
        )

    def _redraw(self) -> None:
        self.canvas.delete("all")
        left, top, right, bottom = self._plot_box()
        self.canvas.create_rectangle(left, top, right, bottom, outline="#444444")

        for tick in _ticks(self.range.xmin, self.range.xmax):
            x, _ = self._to_canvas(tick, self.range.ymin)
            self.canvas.create_line(x, bottom, x, bottom + 5, fill="#444444")
            self.canvas.create_text(x, bottom + 18, text=f"{tick:.4f}", font=("Segoe UI", 9))
        for tick in _ticks(self.range.ymin, self.range.ymax):
            _, y = self._to_canvas(self.range.xmin, tick)
            self.canvas.create_line(left - 5, y, left, y, fill="#444444")
            self.canvas.create_text(left - 10, y, text=f"{tick:.1f}", anchor="e", font=("Segoe UI", 9))
        self.canvas.create_text((left + right) / 2, bottom + 40, text="Wavelength (nm)")
        self.canvas.create_text(24, (top + bottom) / 2, text="20*log10(|E|) (dB)", angle=90)

        visible = [group for group in self.groups if self.visible_vars[group.name].get()]
        if not visible:
            self.canvas.create_text((left + right) / 2, (top + bottom) / 2, text="No variables visible.")
        for group in visible:
            for index, curve in enumerate(group.curves):
                order = np.argsort(curve.wavelength_nm)
                x = curve.wavelength_nm[order]
                y = curve.magnitude_db()[order]
                mask = (x >= self.range.xmin) & (x <= self.range.xmax)
                if not np.any(mask):
                    continue
                x = x[mask]
                y = y[mask]
                if x.size < 2:
                    continue
                step = max(1, math.ceil(x.size / max(right - left, 1)))
                points: list[float] = []
                for xpoint, ypoint in zip(x[::step], y[::step], strict=True):
                    points.extend(self._to_canvas(float(xpoint), float(ypoint)))
                if x[-1] != x[::step][-1]:
                    points.extend(self._to_canvas(float(x[-1]), float(y[-1])))
                self.canvas.create_line(
                    *points,
                    fill=_shade(self.colors[group.name], index, len(group.curves)),
                    width=2,
                )

        shared = self._shared_baseline()
        for target, record in self.assignments.items():
            color = "#111111" if target == GLOBAL_TARGET else self.colors[target]
            if self.range.xmin <= record.wavelength_nm <= self.range.xmax:
                xpos, _ = self._to_canvas(record.wavelength_nm, self.range.ymin)
                self.canvas.create_line(xpos, top, xpos, bottom, fill=color, dash=(3, 3))
                label = "global" if target == GLOBAL_TARGET else target
                self.canvas.create_text(
                    xpos + 6,
                    top + 12,
                    text=f"{label} x={record.wavelength_nm:.4f}",
                    anchor="w",
                    font=("Segoe UI", 9, "bold"),
                    fill=color,
                )
            baseline = shared if shared is not None else record.baseline_db
            if self.range.ymin <= baseline <= self.range.ymax:
                _, ypos = self._to_canvas(self.range.xmin, baseline)
                self.canvas.create_line(left, ypos, right, ypos, fill=color, dash=(5, 3))

        if self.last_cursor is not None:
            xdata, ydata = self.last_cursor
            if self.range.xmin <= xdata <= self.range.xmax and self.range.ymin <= ydata <= self.range.ymax:
                xpos, ypos = self._to_canvas(xdata, ydata)
                self.canvas.create_line(xpos, top, xpos, bottom, fill="#777777", dash=(2, 2))
                self.canvas.create_line(left, ypos, right, ypos, fill="#777777", dash=(2, 2))
                info_x = min(xpos + 8, right - 120)
                info_y = max(ypos - 12, top + 28)
                self.canvas.create_rectangle(info_x, info_y - 30, info_x + 120, info_y + 8, fill="white")
                self.canvas.create_text(
                    info_x + 6,
                    info_y - 10,
                    anchor="w",
                    font=("Consolas", 9),
                    text=f"x={xdata:.6f} nm\ny={ydata:.4f} dB",
                )

        if visible:
            box_left = max(right - 210, 160)
            box_top = top + 8
            self.canvas.create_rectangle(box_left, box_top, right - 8, box_top + 26 + 22 * len(visible))
            self.canvas.create_text(box_left + 10, box_top + 12, anchor="w", text="Visible variables")
            for index, group in enumerate(visible, start=1):
                y = box_top + 10 + 20 * index
                self.canvas.create_rectangle(box_left + 10, y - 5, box_left + 22, y + 7, fill=self.colors[group.name])
                self.canvas.create_text(box_left + 30, y + 1, anchor="w", text=group.name)


def _run_viewer(
    groups: Sequence[CurveGroup] | Mapping[str, CurveGroup],
    *,
    mode: str,
    shared_baseline: bool,
    title: str,
) -> ViewerSelectionResult:
    sequence = list(groups.values()) if isinstance(groups, Mapping) else list(groups)
    return _Viewer(sequence, mode=mode, shared_baseline=shared_baseline, title=title).show()


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


def _ticks(lower: float, upper: float, count: int = 5) -> list[float]:
    if not math.isfinite(lower) or not math.isfinite(upper):
        return [0.0]
    if math.isclose(lower, upper):
        return [lower]
    return [float(value) for value in np.linspace(lower, upper, count)]


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
            print({"wavelength_nm": result.wavelength_nm, "baseline_db": result.baseline_db})
        else:
            result = select_variable_targets(
                _demo_groups(),
                title=args.title,
                shared_baseline=args.shared_baseline,
            )
            print(
                {
                    "shared_baseline_db": result.shared_baseline_db,
                    "visible_groups": list(result.visible_groups),
                    "selections": {
                        key: {"wavelength_nm": value.wavelength_nm, "baseline_db": value.baseline_db}
                        for key, value in result.selections.items()
                    },
                }
            )
    except SelectionCancelledError:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
