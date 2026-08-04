"""Tkinter controls for dose NN saved-scene render settings."""

from __future__ import annotations

import re
from typing import Any

from .dose_nn_render_controls import DOSE_NN_DOSE_COLOR_SCALE_MODES
from .dose_nn_render_controls import DOSE_NN_COLORWASH_STYLES
from .dose_nn_render_controls import DEFAULT_DOSE_NN_REFERENCE_TRIAL_NUMBER
from .dose_nn_render_controls import DoseNNRenderControlSelection
from .dose_nn_render_controls import normalize_dose_nn_render_control_selection


class TkDoseNNRenderControlSelectionAdapter:
    """Collect dose-specific render controls without extending the generic broker."""

    def __init__(self, default_geometry: str = "980x840", min_size: tuple[int, int] = (760, 580)):
        self.default_geometry = str(default_geometry)
        self.min_size = tuple(int(value) for value in min_size)

    def collect_control_selection(
        self,
        option: Any,
        initial_selection: DoseNNRenderControlSelection | None = None,
    ) -> DoseNNRenderControlSelection | None:
        """Return selected dose controls, or None if the dialog is cancelled."""
        import tkinter as tk
        from tkinter import messagebox, ttk

        resolved_initial_selection = initial_selection or DoseNNRenderControlSelection()
        available_trials = tuple(int(trial_number) for trial_number in tuple(getattr(option, "available_trials", ())))
        lattice_dose_range = getattr(option, "lattice_dose_range", None)
        result_ref: dict[str, DoseNNRenderControlSelection | None] = {"value": None}

        root = tk.Tk()
        root.title("Dose NN render controls")
        root.geometry(self.default_geometry)
        root.minsize(int(self.min_size[0]), int(self.min_size[1]))
        root.attributes("-topmost", True)
        root.after(250, lambda: root.attributes("-topmost", False))

        canvas = tk.Canvas(root, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
        frame = ttk.Frame(canvas, padding=14)
        frame.bind(
            "<Configure>",
            lambda event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        frame_window_id = canvas.create_window((0, 0), window=frame, anchor="nw")
        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(frame_window_id, width=event.width),
        )
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        frame.columnconfigure(1, weight=1)

        scene_label = str(getattr(option, "display_label", getattr(option, "scene_id", "saved scene")))
        ttk.Label(frame, text="Scene", font=("TkDefaultFont", 10, "bold")).grid(
            row=0,
            column=0,
            sticky="w",
            pady=(0, 4),
        )
        ttk.Label(frame, text=scene_label, wraplength=720, justify="left").grid(
            row=0,
            column=1,
            sticky="ew",
            pady=(0, 4),
        )

        trial_summary = _format_available_trials(available_trials)
        if lattice_dose_range is not None:
            trial_summary = "{} | dose range {:.4g}-{:.4g}".format(trial_summary, *lattice_dose_range)
        ttk.Label(frame, text=trial_summary, wraplength=720, justify="left").grid(
            row=1,
            column=1,
            sticky="ew",
            pady=(0, 12),
        )

        selected_trials_var = tk.StringVar(value=_format_trial_numbers(resolved_initial_selection.selected_trials))
        reference_trials_var = tk.StringVar(
            value=_format_trial_numbers(
                resolved_initial_selection.reference_trial_numbers
                or ((DEFAULT_DOSE_NN_REFERENCE_TRIAL_NUMBER,) if available_trials else None)
            )
        )
        dose_min_var = tk.StringVar(value=_format_optional_number(resolved_initial_selection.dose_threshold_min))
        dose_max_var = tk.StringVar(value=_format_optional_number(resolved_initial_selection.dose_threshold_max))
        max_lattice_var = tk.StringVar(value=_format_optional_number(resolved_initial_selection.max_lattice_points))
        radius_var = tk.StringVar(value=_format_optional_number(resolved_initial_selection.spatial_radius_mm))
        biopsy_stride_var = tk.StringVar(value=str(resolved_initial_selection.biopsy_point_stride))
        vector_stride_var = tk.StringVar(value=str(resolved_initial_selection.vector_stride))
        color_scale_mode_var = tk.StringVar(value=str(resolved_initial_selection.dose_color_scale_mode))
        color_scale_min_var = tk.StringVar(value=_format_optional_number(resolved_initial_selection.dose_color_scale_min))
        color_scale_max_var = tk.StringVar(value=_format_optional_number(resolved_initial_selection.dose_color_scale_max))
        opacity_var = tk.StringVar(value=str(resolved_initial_selection.dose_colorwash_opacity))
        colorwash_point_size_var = tk.StringVar(value=str(resolved_initial_selection.dose_colorwash_point_size))
        colorwash_style_var = tk.StringVar(value=str(resolved_initial_selection.dose_colorwash_style))
        dose_scalar_bar_title_var = tk.StringVar(value=str(resolved_initial_selection.dose_scalar_bar_title))
        x_axis_label_var = tk.StringVar(value=str(resolved_initial_selection.x_axis_label))
        y_axis_label_var = tk.StringVar(value=str(resolved_initial_selection.y_axis_label))
        z_axis_label_var = tk.StringVar(value=str(resolved_initial_selection.z_axis_label))

        show_biopsy_var = tk.BooleanVar(value=bool(resolved_initial_selection.show_biopsy_points))
        show_reference_var = tk.BooleanVar(value=bool(resolved_initial_selection.show_reference_biopsy_points))
        show_lattice_var = tk.BooleanVar(value=bool(resolved_initial_selection.show_lattice_points))
        show_colorwash_var = tk.BooleanVar(value=bool(resolved_initial_selection.show_dose_colorwash))
        show_nn_points_var = tk.BooleanVar(value=bool(resolved_initial_selection.show_nearest_neighbour_points))
        show_vectors_var = tk.BooleanVar(value=bool(resolved_initial_selection.show_nearest_neighbour_vectors))
        show_axes_var = tk.BooleanVar(value=bool(resolved_initial_selection.show_axes))
        show_scalar_bar_var = tk.BooleanVar(value=bool(resolved_initial_selection.show_scalar_bar))

        current_row = 2
        current_row = _add_labeled_entry(frame, current_row, "Trials", selected_trials_var, "blank = all; use 0-100 or 0,50,100")
        current_row = _add_labeled_entry(frame, current_row, "Nominal trial", reference_trials_var, "usually 0 = non-transformed position")
        current_row = _add_labeled_entry(frame, current_row, "Dose min", dose_min_var, "blank = no lower cutoff")
        current_row = _add_labeled_entry(frame, current_row, "Dose max", dose_max_var, "blank = no upper cutoff")
        ttk.Label(frame, text="Dose color scaling").grid(row=current_row, column=0, sticky="w", pady=(4, 4))
        ttk.Combobox(
            frame,
            textvariable=color_scale_mode_var,
            values=DOSE_NN_DOSE_COLOR_SCALE_MODES,
            state="readonly",
        ).grid(row=current_row, column=1, sticky="ew", pady=(4, 4))
        current_row += 1
        current_row = _add_labeled_entry(frame, current_row, "Color scale min", color_scale_min_var, "blank = use dose data minimum")
        current_row = _add_labeled_entry(frame, current_row, "Color scale max", color_scale_max_var, "blank = use dose data maximum")
        current_row = _add_labeled_entry(frame, current_row, "Max lattice points", max_lattice_var, "blank = no cap")
        current_row = _add_labeled_entry(frame, current_row, "Spatial radius mm", radius_var, "blank = no radius filter")
        current_row = _add_labeled_entry(frame, current_row, "Biopsy stride", biopsy_stride_var, "positive integer")
        current_row = _add_labeled_entry(frame, current_row, "Vector stride", vector_stride_var, "positive integer")

        ttk.Label(frame, text="Colorwash style").grid(row=current_row, column=0, sticky="w", pady=(4, 4))
        ttk.Combobox(
            frame,
            textvariable=colorwash_style_var,
            values=DOSE_NN_COLORWASH_STYLES,
            state="readonly",
        ).grid(row=current_row, column=1, sticky="ew", pady=(4, 4))
        current_row += 1
        current_row = _add_labeled_entry(frame, current_row, "Colorwash opacity", opacity_var, "0 to 1")
        current_row = _add_labeled_entry(
            frame,
            current_row,
            "Colorwash point size",
            colorwash_point_size_var,
            "positive number",
        )
        current_row = _add_labeled_entry(frame, current_row, "Dose colorbar title", dose_scalar_bar_title_var, "example: Dose (Gy)")
        current_row = _add_labeled_entry(frame, current_row, "X axis label", x_axis_label_var, "physical patient-space x")
        current_row = _add_labeled_entry(frame, current_row, "Y axis label", y_axis_label_var, "physical patient-space y")
        current_row = _add_labeled_entry(frame, current_row, "Z axis label", z_axis_label_var, "physical patient-space z")

        layers_frame = ttk.LabelFrame(frame, text="Layers", padding=8)
        layers_frame.grid(row=current_row, column=0, columnspan=2, sticky="ew", pady=(10, 8))
        layers_frame.columnconfigure(0, weight=1)
        layers_frame.columnconfigure(1, weight=1)
        _add_checkbox(layers_frame, 0, 0, "Biopsy points", show_biopsy_var)
        _add_checkbox(layers_frame, 0, 1, "Nominal biopsy position", show_reference_var)
        _add_checkbox(layers_frame, 1, 0, "Dose lattice points", show_lattice_var)
        _add_checkbox(layers_frame, 1, 1, "Dose colorwash", show_colorwash_var)
        _add_checkbox(layers_frame, 2, 0, "NN dose points", show_nn_points_var)
        _add_checkbox(layers_frame, 2, 1, "NN vectors", show_vectors_var)
        _add_checkbox(layers_frame, 3, 0, "Axes", show_axes_var)
        _add_checkbox(layers_frame, 3, 1, "Scalar bar", show_scalar_bar_var)
        current_row += 1

        button_frame = ttk.Frame(frame)
        button_frame.grid(row=current_row, column=0, columnspan=2, sticky="e", pady=(8, 0))

        def _submit() -> None:
            try:
                selection = DoseNNRenderControlSelection(
                    selected_trials=_parse_trial_numbers(selected_trials_var.get()),
                    reference_trial_numbers=_parse_trial_numbers(reference_trials_var.get()),
                    dose_threshold_min=_parse_optional_float(dose_min_var.get()),
                    dose_threshold_max=_parse_optional_float(dose_max_var.get()),
                    max_lattice_points=_parse_optional_int(max_lattice_var.get()),
                    spatial_radius_mm=_parse_optional_float(radius_var.get()),
                    biopsy_point_stride=int(biopsy_stride_var.get()),
                    vector_stride=int(vector_stride_var.get()),
                    show_biopsy_points=bool(show_biopsy_var.get()),
                    show_reference_biopsy_points=bool(show_reference_var.get()),
                    show_lattice_points=bool(show_lattice_var.get()),
                    show_dose_colorwash=bool(show_colorwash_var.get()),
                    dose_colorwash_style=str(colorwash_style_var.get()),
                    dose_color_scale_mode=str(color_scale_mode_var.get()),
                    dose_color_scale_min=_parse_optional_float(color_scale_min_var.get()),
                    dose_color_scale_max=_parse_optional_float(color_scale_max_var.get()),
                    dose_colorwash_opacity=float(opacity_var.get()),
                    dose_colorwash_point_size=float(colorwash_point_size_var.get()),
                    show_nearest_neighbour_points=bool(show_nn_points_var.get()),
                    show_nearest_neighbour_vectors=bool(show_vectors_var.get()),
                    show_axes=bool(show_axes_var.get()),
                    show_scalar_bar=bool(show_scalar_bar_var.get()),
                    dose_scalar_bar_title=str(dose_scalar_bar_title_var.get()),
                    x_axis_label=str(x_axis_label_var.get()),
                    y_axis_label=str(y_axis_label_var.get()),
                    z_axis_label=str(z_axis_label_var.get()),
                )
                result_ref["value"] = normalize_dose_nn_render_control_selection(
                    selection,
                    available_trials=available_trials,
                )
            except Exception as exc:
                messagebox.showwarning("Invalid dose render controls", str(exc), parent=root)
                return
            root.destroy()

        def _cancel() -> None:
            result_ref["value"] = None
            root.destroy()

        ttk.Button(button_frame, text="Cancel", command=_cancel).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(button_frame, text="Render", command=_submit).grid(row=0, column=1)
        root.protocol("WM_DELETE_WINDOW", _cancel)
        root.mainloop()
        return result_ref["value"]

    def notify_render_result(self, results: tuple[Any, ...]) -> None:
        """Show a completion message for GUI-launched renders."""
        import tkinter as tk
        from tkinter import messagebox

        if len(results) == 0:
            return
        root = tk.Tk()
        root.withdraw()
        try:
            screenshot_paths = [str(getattr(result, "screenshot_path", result)) for result in tuple(results)]
            messagebox.showinfo(
                "Dose NN render complete",
                "Wrote render output:\n{}".format("\n".join(screenshot_paths)),
                parent=root,
            )
        finally:
            root.destroy()

    def notify_render_error(self, error: BaseException) -> None:
        """Show render errors without losing the selector loop."""
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        try:
            messagebox.showerror("Dose NN render failed", str(error), parent=root)
        finally:
            root.destroy()


def _add_labeled_entry(frame: Any, row: int, label: str, variable: Any, hint: str) -> int:
    from tkinter import ttk

    ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=(4, 4))
    entry = ttk.Entry(frame, textvariable=variable)
    entry.grid(row=row, column=1, sticky="ew", pady=(4, 4))
    ttk.Label(frame, text=hint, foreground="#526174").grid(row=row + 1, column=1, sticky="w")
    return row + 2


def _add_checkbox(frame: Any, row: int, column: int, label: str, variable: Any) -> None:
    from tkinter import ttk

    ttk.Checkbutton(frame, text=label, variable=variable).grid(row=row, column=column, sticky="w", padx=(0, 12))


def _format_available_trials(available_trials: tuple[int, ...]) -> str:
    if len(available_trials) == 0:
        return "Available trials: unavailable from manifest"
    if len(available_trials) <= 12:
        return "Available trials: {}".format(_format_trial_numbers(available_trials))
    return "Available trials: {} to {} ({} total)".format(
        int(min(available_trials)),
        int(max(available_trials)),
        len(available_trials),
    )


def _format_trial_numbers(trial_numbers: tuple[int, ...] | None) -> str:
    if trial_numbers is None:
        return ""
    return ",".join(str(int(trial_number)) for trial_number in tuple(trial_numbers))


def _format_optional_number(value: float | int | None) -> str:
    return "" if value is None else str(value)


def _parse_trial_numbers(value: str) -> tuple[int, ...] | None:
    text = str(value).strip()
    if text == "" or text.lower() == "all":
        return None
    normalized_text = re.sub(r"\s*-\s*", "-", text.replace(";", ","))
    fragments = [fragment.strip() for fragment in re.split(r"[\s,]+", normalized_text) if fragment.strip() != ""]
    trial_numbers_list: list[int] = []
    for fragment in fragments:
        range_match = re.fullmatch(r"(\d+)-(\d+)", fragment)
        if range_match is not None:
            start_trial = int(range_match.group(1))
            stop_trial = int(range_match.group(2))
            if stop_trial < start_trial:
                raise ValueError("trial ranges must be ascending")
            trial_numbers_list.extend(range(start_trial, stop_trial + 1))
        else:
            trial_numbers_list.append(int(fragment))
    trial_numbers = tuple(trial_numbers_list)
    return None if len(trial_numbers) == 0 else trial_numbers


def _parse_optional_float(value: str) -> float | None:
    text = str(value).strip()
    return None if text == "" else float(text)


def _parse_optional_int(value: str) -> int | None:
    text = str(value).strip()
    return None if text == "" else int(text)