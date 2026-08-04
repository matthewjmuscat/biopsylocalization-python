"""Dose-owned render controls for saved dose NN scene GUIs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from .dose_nn_pyvista import DoseNNPyVistaRenderSettings
from .dose_nn_scene import DoseNNRenderConfig


DOSE_NN_COLORWASH_STYLES = ("points", "volume", "auto")
DOSE_NN_DOSE_COLOR_SCALE_MODES = ("linear", "log")
DEFAULT_DOSE_NN_REFERENCE_TRIAL_NUMBER = 0


@dataclass(frozen=True, slots=True)
class DoseNNRenderControlSelection:
    """Dose-specific GUI control values independent of any GUI toolkit."""

    selected_trials: tuple[int, ...] | None = None
    reference_trial_numbers: tuple[int, ...] | None = None
    dose_threshold_min: float | None = None
    dose_threshold_max: float | None = None
    max_lattice_points: int | None = None
    spatial_radius_mm: float | None = None
    biopsy_point_stride: int = 1
    vector_stride: int = 1
    show_biopsy_points: bool = True
    show_reference_biopsy_points: bool = False
    show_lattice_points: bool = True
    show_dose_colorwash: bool = False
    dose_colorwash_style: str = "points"
    dose_color_scale_mode: str = "linear"
    dose_color_scale_min: float | None = None
    dose_color_scale_max: float | None = None
    dose_colorwash_opacity: float = 0.28
    dose_colorwash_point_size: float = 12.0
    show_nearest_neighbour_points: bool = True
    show_nearest_neighbour_vectors: bool = True
    show_axes: bool = True
    show_scalar_bar: bool = True
    dose_scalar_bar_title: str = "Dose"
    x_axis_label: str = "Left-Right x (mm)"
    y_axis_label: str = "Posterior-Anterior y (mm)"
    z_axis_label: str = "Inferior-Superior z (mm)"


def normalize_dose_nn_render_control_selection(
    selection: DoseNNRenderControlSelection | None = None,
    *,
    available_trials: Sequence[int] = (),
) -> DoseNNRenderControlSelection:
    """Normalize and validate dose render controls from a GUI or CLI surface."""
    resolved_selection = selection or DoseNNRenderControlSelection()
    resolved_available_trials = _normalize_trial_numbers(tuple(available_trials)) or ()
    selected_trials = _normalize_trial_numbers(resolved_selection.selected_trials)
    reference_trial_numbers = _normalize_trial_numbers(resolved_selection.reference_trial_numbers)

    if bool(resolved_selection.show_reference_biopsy_points) and reference_trial_numbers is None:
        reference_trial_numbers = (DEFAULT_DOSE_NN_REFERENCE_TRIAL_NUMBER,)

    _validate_requested_trials("selected_trials", selected_trials, resolved_available_trials)
    _validate_requested_trials("reference_trial_numbers", reference_trial_numbers, resolved_available_trials)

    dose_threshold_min = _optional_float(resolved_selection.dose_threshold_min)
    dose_threshold_max = _optional_float(resolved_selection.dose_threshold_max)
    if dose_threshold_min is not None and dose_threshold_max is not None:
        if dose_threshold_min > dose_threshold_max:
            raise ValueError("dose_threshold_min cannot exceed dose_threshold_max")

    max_lattice_points = _optional_positive_int("max_lattice_points", resolved_selection.max_lattice_points)
    spatial_radius_mm = _optional_positive_float("spatial_radius_mm", resolved_selection.spatial_radius_mm)
    biopsy_point_stride = _positive_int("biopsy_point_stride", resolved_selection.biopsy_point_stride)
    vector_stride = _positive_int("vector_stride", resolved_selection.vector_stride)
    dose_colorwash_style = _normalize_colorwash_style(resolved_selection.dose_colorwash_style)
    dose_color_scale_mode = _normalize_dose_color_scale_mode(resolved_selection.dose_color_scale_mode)
    dose_color_scale_min = _optional_float(resolved_selection.dose_color_scale_min)
    dose_color_scale_max = _optional_float(resolved_selection.dose_color_scale_max)
    if (dose_color_scale_min is None) != (dose_color_scale_max is None):
        raise ValueError("dose_color_scale_min and dose_color_scale_max must both be set, or both blank")
    if dose_color_scale_min is not None and dose_color_scale_min >= dose_color_scale_max:
        raise ValueError("dose_color_scale_min must be less than dose_color_scale_max")
    if dose_color_scale_mode == "log" and dose_color_scale_min is not None and dose_color_scale_min <= 0.0:
        raise ValueError("log dose color scaling requires dose_color_scale_min > 0")
    dose_colorwash_opacity = float(resolved_selection.dose_colorwash_opacity)
    if dose_colorwash_opacity < 0.0 or dose_colorwash_opacity > 1.0:
        raise ValueError("dose_colorwash_opacity must be between 0 and 1")
    dose_colorwash_point_size = _positive_float(
        "dose_colorwash_point_size",
        resolved_selection.dose_colorwash_point_size,
    )

    return DoseNNRenderControlSelection(
        selected_trials=selected_trials,
        reference_trial_numbers=reference_trial_numbers,
        dose_threshold_min=dose_threshold_min,
        dose_threshold_max=dose_threshold_max,
        max_lattice_points=max_lattice_points,
        spatial_radius_mm=spatial_radius_mm,
        biopsy_point_stride=biopsy_point_stride,
        vector_stride=vector_stride,
        show_biopsy_points=bool(resolved_selection.show_biopsy_points),
        show_reference_biopsy_points=bool(resolved_selection.show_reference_biopsy_points),
        show_lattice_points=bool(resolved_selection.show_lattice_points),
        show_dose_colorwash=bool(resolved_selection.show_dose_colorwash),
        dose_colorwash_style=dose_colorwash_style,
        dose_color_scale_mode=dose_color_scale_mode,
        dose_color_scale_min=dose_color_scale_min,
        dose_color_scale_max=dose_color_scale_max,
        dose_colorwash_opacity=dose_colorwash_opacity,
        dose_colorwash_point_size=dose_colorwash_point_size,
        show_nearest_neighbour_points=bool(resolved_selection.show_nearest_neighbour_points),
        show_nearest_neighbour_vectors=bool(resolved_selection.show_nearest_neighbour_vectors),
        show_axes=bool(resolved_selection.show_axes),
        show_scalar_bar=bool(resolved_selection.show_scalar_bar),
        dose_scalar_bar_title=str(resolved_selection.dose_scalar_bar_title),
        x_axis_label=str(resolved_selection.x_axis_label),
        y_axis_label=str(resolved_selection.y_axis_label),
        z_axis_label=str(resolved_selection.z_axis_label),
    )


def dose_nn_render_config_from_control_selection(
    selection: DoseNNRenderControlSelection | None = None,
    *,
    available_trials: Sequence[int] = (),
) -> DoseNNRenderConfig:
    """Translate dose-owned GUI controls into renderer-neutral config."""
    resolved_selection = normalize_dose_nn_render_control_selection(
        selection,
        available_trials=available_trials,
    )
    return DoseNNRenderConfig(
        selected_trials=resolved_selection.selected_trials,
        reference_trial_numbers=resolved_selection.reference_trial_numbers,
        dose_threshold_min=resolved_selection.dose_threshold_min,
        dose_threshold_max=resolved_selection.dose_threshold_max,
        max_lattice_points=resolved_selection.max_lattice_points,
        spatial_radius_mm=resolved_selection.spatial_radius_mm,
        biopsy_point_stride=resolved_selection.biopsy_point_stride,
        vector_stride=resolved_selection.vector_stride,
        show_biopsy_points=resolved_selection.show_biopsy_points,
        show_reference_biopsy_points=resolved_selection.show_reference_biopsy_points,
        show_lattice_points=resolved_selection.show_lattice_points,
        show_dose_colorwash=resolved_selection.show_dose_colorwash,
        show_nearest_neighbour_points=resolved_selection.show_nearest_neighbour_points,
        show_nearest_neighbour_vectors=resolved_selection.show_nearest_neighbour_vectors,
    )


def dose_nn_pyvista_settings_from_control_selection(
    selection: DoseNNRenderControlSelection | None = None,
    *,
    available_trials: Sequence[int] = (),
    base_settings: DoseNNPyVistaRenderSettings | None = None,
) -> DoseNNPyVistaRenderSettings:
    """Translate dose-owned GUI controls into PyVista render settings."""
    resolved_selection = normalize_dose_nn_render_control_selection(
        selection,
        available_trials=available_trials,
    )
    resolved_base_settings = base_settings or DoseNNPyVistaRenderSettings()
    return replace(
        resolved_base_settings,
        dose_colorwash_style=resolved_selection.dose_colorwash_style,
        dose_color_scale_mode=resolved_selection.dose_color_scale_mode,
        dose_color_scale_min=resolved_selection.dose_color_scale_min,
        dose_color_scale_max=resolved_selection.dose_color_scale_max,
        dose_colorwash_point_size=resolved_selection.dose_colorwash_point_size,
        dose_colorwash_opacity=resolved_selection.dose_colorwash_opacity,
        show_axes=resolved_selection.show_axes,
        show_scalar_bar=resolved_selection.show_scalar_bar,
        dose_scalar_bar_title=resolved_selection.dose_scalar_bar_title,
        x_axis_label=resolved_selection.x_axis_label,
        y_axis_label=resolved_selection.y_axis_label,
        z_axis_label=resolved_selection.z_axis_label,
    )


def _normalize_trial_numbers(trial_numbers: Sequence[int] | None) -> tuple[int, ...] | None:
    if trial_numbers is None:
        return None
    resolved_trial_numbers = tuple(int(trial_number) for trial_number in tuple(trial_numbers))
    if len(resolved_trial_numbers) == 0:
        return None
    return resolved_trial_numbers


def _validate_requested_trials(
    field_name: str,
    requested_trials: tuple[int, ...] | None,
    available_trials: tuple[int, ...],
) -> None:
    if requested_trials is None or len(available_trials) == 0:
        return
    missing_trials = sorted(set(requested_trials).difference(set(available_trials)))
    if missing_trials:
        raise ValueError("{} requested unavailable trials: {}".format(field_name, missing_trials))


def _normalize_colorwash_style(value: str) -> str:
    resolved_value = str(value).strip().lower()
    if resolved_value == "point":
        resolved_value = "points"
    if resolved_value not in DOSE_NN_COLORWASH_STYLES:
        raise ValueError("unsupported dose colorwash style: {}".format(value))
    return resolved_value


def _normalize_dose_color_scale_mode(value: str) -> str:
    resolved_value = str(value).strip().lower()
    if resolved_value in ("lin", "linear"):
        return "linear"
    if resolved_value in ("log", "log10", "logarithmic"):
        return "log"
    raise ValueError("unsupported dose color scale mode: {}".format(value))


def _optional_float(value: float | None) -> float | None:
    return None if value is None else float(value)


def _optional_positive_int(field_name: str, value: int | None) -> int | None:
    if value is None:
        return None
    return _positive_int(field_name, value)


def _optional_positive_float(field_name: str, value: float | None) -> float | None:
    if value is None:
        return None
    return _positive_float(field_name, value)


def _positive_int(field_name: str, value: int) -> int:
    resolved_value = int(value)
    if resolved_value <= 0:
        raise ValueError("{} must be positive".format(field_name))
    return resolved_value


def _positive_float(field_name: str, value: float) -> float:
    resolved_value = float(value)
    if resolved_value <= 0:
        raise ValueError("{} must be positive".format(field_name))
    return resolved_value