"""PyVista backend for dosimetric nearest-neighbour render scenes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .dose_nn_scene import DoseNNPreparedScene, DoseNNRenderConfig, DoseNNRenderScene, prepare_dose_nn_render_scene


PYVISTA_DOSE_NN_BACKEND_KEY = "pyvista"


@dataclass(frozen=True, slots=True)
class DoseNNPyVistaRenderSettings:
    """Display and export settings for the PyVista dose NN backend."""

    off_screen: bool = True
    window_size: tuple[int, int] = (1200, 900)
    background_color: str = "white"
    dose_colormap: str = "coolwarm"
    dose_color_scale_mode: str = "linear"
    dose_color_scale_min: float | None = None
    dose_color_scale_max: float | None = None
    lattice_point_size: float = 5.0
    dose_colorwash_style: str = "points"
    dose_colorwash_volume_max_voxels: int | None = 250_000
    dose_colorwash_point_size: float = 12.0
    dose_colorwash_opacity: float = 0.08
    dose_colorwash_point_opacity_mode: str = "constant"
    dose_colorwash_point_opacity_min: float = 0.02
    biopsy_point_size: float = 12.0
    reference_biopsy_point_size: float = 10.0
    nearest_point_size: float = 8.0
    vector_line_width: float = 2.0
    biopsy_color: str = "crimson"
    reference_biopsy_color: str = "royalblue"
    nearest_point_color: str = "black"
    vector_color: str = "dimgray"
    show_axes: bool = True
    show_scalar_bar: bool = True
    dose_scalar_bar_title: str = "Dose (Gy)"
    dose_scalar_bar_num_labels: int = 5
    dose_scalar_bar_label_format: str = "%.3g"
    dose_scalar_bar_font_family: str = "arial"
    dose_scalar_bar_title_font_size: int = 18
    dose_scalar_bar_label_font_size: int = 14
    dose_scalar_bar_title_separation: int = 12
    dose_scalar_bar_vertical: bool = True
    dose_scalar_bar_use_opacity: bool = False
    dose_scalar_bar_show_background: bool = True
    x_axis_label: str = "Left-Right x (mm)"
    y_axis_label: str = "Posterior-Anterior y (mm)"
    z_axis_label: str = "Inferior-Superior z (mm)"
    axes_font_family: str = "arial"
    axes_title_font_size: int = 18
    axes_tick_label_font_size: int = 14
    camera_position: Any | None = None


@dataclass(frozen=True, slots=True)
class DoseNNPyVistaExportResult:
    """Paths written by a PyVista dose NN export."""

    screenshot_path: Path
    provenance_path: Path


@dataclass(frozen=True, slots=True)
class DoseNNPyVistaFrameSequenceExportResult:
    """Paths written by a PyVista per-trial frame-sequence export."""

    frame_paths: tuple[Path, ...]
    manifest_path: Path


def is_pyvista_available() -> bool:
    """Return whether PyVista can be imported in the current environment."""
    try:
        _import_pyvista()
    except RuntimeError:
        return False
    return True


def render_dose_nn_scene_pyvista(
    scene: DoseNNRenderScene,
    config: DoseNNRenderConfig | None = None,
    settings: DoseNNPyVistaRenderSettings | None = None,
) -> Any:
    """Prepare a dose NN scene and return a configured PyVista plotter."""
    prepared_scene = prepare_dose_nn_render_scene(scene, config)
    return build_pyvista_dose_nn_plotter(prepared_scene, settings=settings)


def build_pyvista_dose_nn_plotter(
    prepared_scene: DoseNNPreparedScene,
    settings: DoseNNPyVistaRenderSettings | None = None,
) -> Any:
    """Build a PyVista plotter from a prepared renderer-neutral dose NN scene."""
    pv = _import_pyvista()
    resolved_settings = settings or DoseNNPyVistaRenderSettings()
    plotter = pv.Plotter(
        off_screen=bool(resolved_settings.off_screen),
        window_size=tuple(int(value) for value in resolved_settings.window_size),
    )
    plotter.set_background(resolved_settings.background_color)

    if prepared_scene.config.show_lattice_points:
        _add_lattice_points(pv, plotter, prepared_scene, resolved_settings)
    if prepared_scene.config.show_dose_colorwash:
        _add_dose_colorwash(pv, plotter, prepared_scene, resolved_settings)
    if prepared_scene.config.show_reference_biopsy_points:
        _add_reference_biopsy_points(pv, plotter, prepared_scene, resolved_settings)
    if prepared_scene.config.show_biopsy_points:
        _add_biopsy_points(pv, plotter, prepared_scene, resolved_settings)
    if prepared_scene.config.show_nearest_neighbour_points:
        _add_nearest_neighbour_points(pv, plotter, prepared_scene, resolved_settings)
    if prepared_scene.config.show_nearest_neighbour_vectors:
        _add_nearest_neighbour_vectors(pv, plotter, prepared_scene, resolved_settings)

    if resolved_settings.show_axes:
        _add_axes(plotter, resolved_settings)
    if resolved_settings.show_scalar_bar:
        _configure_scalar_bar_actors(plotter, resolved_settings)
    if resolved_settings.show_scalar_bar and resolved_settings.dose_scalar_bar_show_background:
        _add_scalar_bar_background_box(pv, plotter, resolved_settings)
    if resolved_settings.camera_position is not None:
        plotter.camera_position = resolved_settings.camera_position
    else:
        plotter.reset_camera()
    return plotter


def export_dose_nn_scene_pyvista(
    scene: DoseNNRenderScene,
    output_path: Path | str,
    config: DoseNNRenderConfig | None = None,
    settings: DoseNNPyVistaRenderSettings | None = None,
    *,
    provenance_path: Path | str | None = None,
    close_plotter: bool = True,
) -> DoseNNPyVistaExportResult:
    """Render a dose NN scene to a screenshot and provenance sidecar."""
    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_provenance_path = Path(provenance_path) if provenance_path is not None else _default_provenance_path(
        resolved_output_path
    )
    resolved_provenance_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_config = config or DoseNNRenderConfig()
    resolved_settings = settings or DoseNNPyVistaRenderSettings()
    prepared_scene = prepare_dose_nn_render_scene(scene, resolved_config)
    plotter = build_pyvista_dose_nn_plotter(prepared_scene, settings=resolved_settings)
    try:
        if bool(resolved_settings.off_screen):
            plotter.screenshot(str(resolved_output_path))
        else:
            plotter.show(screenshot=str(resolved_output_path), auto_close=bool(close_plotter))
    finally:
        if close_plotter and bool(resolved_settings.off_screen):
            plotter.close()

    _write_json(
        resolved_provenance_path,
        build_pyvista_dose_nn_export_provenance(
            prepared_scene,
            settings=resolved_settings,
            screenshot_path=resolved_output_path,
        ),
    )
    return DoseNNPyVistaExportResult(
        screenshot_path=resolved_output_path,
        provenance_path=resolved_provenance_path,
    )


def export_dose_nn_trial_frame_sequence_pyvista(
    scene: DoseNNRenderScene,
    output_dir: Path | str,
    *,
    selected_trials: tuple[int, ...] | None = None,
    max_frames: int | None = None,
    frames_per_second: float = 12.0,
    base_config: DoseNNRenderConfig | None = None,
    settings: DoseNNPyVistaRenderSettings | None = None,
    frame_name_prefix: str = "dose_nn_trial",
    manifest_path: Path | str | None = None,
    overwrite: bool = False,
) -> DoseNNPyVistaFrameSequenceExportResult:
    """Export one screenshot per selected trial plus a frame manifest."""
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    resolved_manifest_path = Path(manifest_path) if manifest_path is not None else resolved_output_dir.joinpath(
        "frame_sequence_manifest.json"
    )
    if not overwrite and resolved_manifest_path.exists():
        raise FileExistsError("dose NN frame manifest already exists: {}".format(resolved_manifest_path))

    trial_numbers = _resolve_frame_trial_numbers(scene, selected_trials, max_frames=max_frames)
    resolved_base_config = base_config or DoseNNRenderConfig()
    resolved_settings = settings or DoseNNPyVistaRenderSettings()
    frame_paths: list[Path] = []
    frame_records: list[dict[str, Any]] = []
    for frame_index, trial_number in enumerate(trial_numbers):
        frame_path = resolved_output_dir.joinpath(
            "{}_{:04d}_trial_{:06d}.png".format(
                _sanitize_output_name(frame_name_prefix),
                int(frame_index),
                int(trial_number),
            )
        )
        if not overwrite and frame_path.exists():
            raise FileExistsError("dose NN frame already exists: {}".format(frame_path))
        export_dose_nn_scene_pyvista(
            scene,
            frame_path,
            config=_config_for_trial(resolved_base_config, int(trial_number)),
            settings=resolved_settings,
            provenance_path=frame_path.with_suffix(".png.provenance.json"),
        )
        frame_paths.append(frame_path)
        frame_records.append(
            {
                "frame_index": int(frame_index),
                "trial_number": int(trial_number),
                "frame_path": str(frame_path),
                "provenance_path": str(frame_path.with_suffix(".png.provenance.json")),
            }
        )

    _write_json(
        resolved_manifest_path,
        {
            "schema_version": "dose_nn_pyvista_frame_sequence_v1",
            "backend": PYVISTA_DOSE_NN_BACKEND_KEY,
            "frames_per_second": float(frames_per_second),
            "frame_count": len(frame_records),
            "scene_metadata": _scene_metadata_from_scene(scene),
            "base_render_config": _json_ready(asdict(resolved_base_config)),
            "render_settings": _json_ready(asdict(resolved_settings)),
            "frames": frame_records,
        },
    )
    return DoseNNPyVistaFrameSequenceExportResult(
        frame_paths=tuple(frame_paths),
        manifest_path=resolved_manifest_path,
    )


def build_pyvista_dose_nn_export_provenance(
    prepared_scene: DoseNNPreparedScene,
    *,
    settings: DoseNNPyVistaRenderSettings | None = None,
    screenshot_path: Path | str | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable provenance payload for a PyVista export."""
    resolved_settings = settings or DoseNNPyVistaRenderSettings()
    payload = {
        "backend": PYVISTA_DOSE_NN_BACKEND_KEY,
        "screenshot_path": "" if screenshot_path is None else str(screenshot_path),
        "scene_metadata": _scene_metadata_dict(prepared_scene),
        "render_config": _json_ready(asdict(prepared_scene.config)),
        "render_settings": _json_ready(asdict(resolved_settings)),
        "prepared_scene_summary": {
            "lattice_point_count": int(prepared_scene.lattice_points.shape[0]),
            "biopsy_point_count": int(prepared_scene.biopsy_points.shape[0]),
            "reference_biopsy_point_count": int(prepared_scene.reference_biopsy_points.shape[0]),
            "nearest_neighbour_point_count": int(np.reshape(prepared_scene.nearest_lattice_points, (-1, 3)).shape[0]),
            "vector_count": int(prepared_scene.num_vectors),
            "selected_trials": [int(trial_number) for trial_number in np.unique(prepared_scene.trial_numbers)],
            "reference_trials": [
                int(trial_number) for trial_number in np.unique(prepared_scene.reference_trial_numbers)
            ],
        },
    }
    json.dumps(payload)
    return payload


def _add_lattice_points(
    pv: Any,
    plotter: Any,
    prepared_scene: DoseNNPreparedScene,
    settings: DoseNNPyVistaRenderSettings,
) -> None:
    if prepared_scene.lattice_points.shape[0] == 0:
        return
    point_cloud = pv.PolyData(prepared_scene.lattice_points)
    point_cloud["dose"] = prepared_scene.lattice_doses
    plotter.add_mesh(
        point_cloud,
        name="dose_lattice_points",
        scalars="dose",
        cmap=settings.dose_colormap,
        point_size=float(settings.lattice_point_size),
        render_points_as_spheres=True,
        show_scalar_bar=bool(settings.show_scalar_bar),
        scalar_bar_args=_dose_scalar_bar_args(settings),
        **_dose_scalar_kwargs(settings),
    )


def _add_dose_colorwash(
    pv: Any,
    plotter: Any,
    prepared_scene: DoseNNPreparedScene,
    settings: DoseNNPyVistaRenderSettings,
) -> None:
    if prepared_scene.colorwash_lattice_points.shape[0] == 0:
        return
    resolved_style = str(settings.dose_colorwash_style).strip().lower()
    if resolved_style in ("point", "points"):
        _add_dose_point_colorwash(pv, plotter, prepared_scene, settings)
        return
    if resolved_style == "volume":
        _add_dose_volume_colorwash(pv, plotter, prepared_scene, settings)
        return
    if resolved_style == "auto":
        try:
            _add_dose_volume_colorwash(pv, plotter, prepared_scene, settings)
        except ValueError:
            _add_dose_point_colorwash(pv, plotter, prepared_scene, settings)
        return
    raise ValueError("unsupported dose colorwash style: {}".format(settings.dose_colorwash_style))


def _add_dose_point_colorwash(
    pv: Any,
    plotter: Any,
    prepared_scene: DoseNNPreparedScene,
    settings: DoseNNPyVistaRenderSettings,
) -> None:
    if prepared_scene.lattice_points.shape[0] == 0:
        return
    point_cloud = pv.PolyData(prepared_scene.lattice_points)
    point_cloud["dose"] = prepared_scene.lattice_doses
    opacity_arg: float | str
    if _normalize_colorwash_point_opacity_mode(settings.dose_colorwash_point_opacity_mode) == "center_fade":
        point_cloud["dose_colorwash_opacity"] = _center_fade_point_opacity(prepared_scene, settings)
        opacity_arg = "dose_colorwash_opacity"
    else:
        opacity_arg = float(settings.dose_colorwash_opacity)
    plotter.add_mesh(
        point_cloud,
        name="dose_colorwash_points",
        scalars="dose",
        cmap=settings.dose_colormap,
        point_size=float(settings.dose_colorwash_point_size),
        opacity=opacity_arg,
        render_points_as_spheres=False,
        show_scalar_bar=(bool(settings.show_scalar_bar) and not bool(prepared_scene.config.show_lattice_points)),
        scalar_bar_args=_dose_scalar_bar_args(settings),
        **_dose_scalar_kwargs(settings),
    )


def _add_dose_volume_colorwash(
    pv: Any,
    plotter: Any,
    prepared_scene: DoseNNPreparedScene,
    settings: DoseNNPyVistaRenderSettings,
) -> None:
    if not bool(np.any(prepared_scene.colorwash_lattice_visibility_mask)):
        return
    dose_grid = _rectilinear_dose_grid_from_lattice(
        pv,
        prepared_scene.colorwash_lattice_points,
        prepared_scene.colorwash_lattice_doses,
        visible_mask=prepared_scene.colorwash_lattice_visibility_mask,
        max_voxels=settings.dose_colorwash_volume_max_voxels,
    )
    plotter.add_volume(
        dose_grid,
        scalars="dose",
        name="dose_colorwash_volume",
        cmap=settings.dose_colormap,
        opacity=float(settings.dose_colorwash_opacity),
        show_scalar_bar=(bool(settings.show_scalar_bar) and not bool(prepared_scene.config.show_lattice_points)),
        scalar_bar_args=_dose_scalar_bar_args(settings),
        **_dose_scalar_kwargs(settings),
    )


def _dose_scalar_bar_args(settings: DoseNNPyVistaRenderSettings) -> dict[str, Any]:
    args: dict[str, Any] = {
        "title": settings.dose_scalar_bar_title,
        "n_labels": int(settings.dose_scalar_bar_num_labels),
        "fmt": settings.dose_scalar_bar_label_format,
        "font_family": settings.dose_scalar_bar_font_family,
        "title_font_size": int(settings.dose_scalar_bar_title_font_size),
        "label_font_size": int(settings.dose_scalar_bar_label_font_size),
        "color": "black",
        "vertical": bool(settings.dose_scalar_bar_vertical),
        "use_opacity": bool(settings.dose_scalar_bar_use_opacity),
    }
    if bool(settings.dose_scalar_bar_vertical):
        args.update({"position_x": 0.88, "position_y": 0.18, "width": 0.06, "height": 0.62})
    else:
        args.update({"position_x": 0.24, "position_y": 0.06, "width": 0.58, "height": 0.08})
    return args


def _dose_scalar_bar_clipped_range_labels(settings: DoseNNPyVistaRenderSettings) -> tuple[str, str] | None:
    if settings.dose_color_scale_min is None or settings.dose_color_scale_max is None:
        return None
    return (
        "<= {}".format(_format_scalar_bar_boundary_value(settings.dose_color_scale_min, settings)),
        ">= {}".format(_format_scalar_bar_boundary_value(settings.dose_color_scale_max, settings)),
    )


def _format_scalar_bar_boundary_value(value: float, settings: DoseNNPyVistaRenderSettings) -> str:
    resolved_value = float(value)
    try:
        return str(settings.dose_scalar_bar_label_format % resolved_value).strip()
    except (TypeError, ValueError):
        return "{:.3g}".format(resolved_value)


def _dose_scalar_bar_tick_values(settings: DoseNNPyVistaRenderSettings) -> np.ndarray | None:
    if settings.dose_color_scale_min is None or settings.dose_color_scale_max is None:
        return None
    label_count = int(settings.dose_scalar_bar_num_labels)
    if label_count <= 2:
        return np.asarray([], dtype=float)
    minimum = float(settings.dose_color_scale_min)
    maximum = float(settings.dose_color_scale_max)
    if _normalize_dose_color_scale_mode(settings.dose_color_scale_mode) == "log":
        return np.geomspace(minimum, maximum, label_count, dtype=float)[1:-1]
    return np.linspace(minimum, maximum, label_count, dtype=float)[1:-1]


def _dose_scalar_kwargs(settings: DoseNNPyVistaRenderSettings) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    dose_color_scale_mode = _normalize_dose_color_scale_mode(settings.dose_color_scale_mode)
    dose_color_scale_min = settings.dose_color_scale_min
    dose_color_scale_max = settings.dose_color_scale_max
    if dose_color_scale_min is not None or dose_color_scale_max is not None:
        if dose_color_scale_min is None or dose_color_scale_max is None:
            raise ValueError("dose color scale min and max must both be set, or both blank")
        resolved_min = float(dose_color_scale_min)
        resolved_max = float(dose_color_scale_max)
        if resolved_min >= resolved_max:
            raise ValueError("dose color scale min must be less than max")
        if dose_color_scale_mode == "log" and resolved_min <= 0.0:
            raise ValueError("log dose color scaling requires dose color scale min > 0")
        kwargs["clim"] = (resolved_min, resolved_max)
    if dose_color_scale_mode == "log":
        kwargs["log_scale"] = True
    return kwargs


def _normalize_dose_color_scale_mode(value: str) -> str:
    resolved_value = str(value).strip().lower()
    if resolved_value in ("lin", "linear"):
        return "linear"
    if resolved_value in ("log", "log10", "logarithmic"):
        return "log"
    raise ValueError("unsupported dose color scale mode: {}".format(value))


def _normalize_colorwash_point_opacity_mode(value: str) -> str:
    resolved_value = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if resolved_value in ("constant", "flat"):
        return "constant"
    if resolved_value in ("center", "center_fade", "central", "central_fade"):
        return "center_fade"
    raise ValueError("unsupported dose colorwash point opacity mode: {}".format(value))


def _center_fade_point_opacity(
    prepared_scene: DoseNNPreparedScene,
    settings: DoseNNPyVistaRenderSettings,
) -> np.ndarray:
    points = prepared_scene.lattice_points
    if points.shape[0] == 0:
        return np.asarray([], dtype=float)
    centers = prepared_scene.biopsy_points
    if centers.shape[0] == 0:
        return np.full(points.shape[0], float(settings.dose_colorwash_opacity), dtype=float)

    max_opacity = _clamp(float(settings.dose_colorwash_opacity), 0.0, 1.0)
    min_opacity = _clamp(float(settings.dose_colorwash_point_opacity_min), 0.0, max_opacity)
    distances = _nearest_center_distances(points, centers)
    if prepared_scene.config.spatial_radius_mm is not None:
        fade_radius = max(float(prepared_scene.config.spatial_radius_mm), np.finfo(float).eps)
    else:
        positive_distances = distances[distances > 0.0]
        fade_radius = float(np.percentile(positive_distances, 95)) if positive_distances.shape[0] else 1.0
    normalized_distance = np.clip(distances / fade_radius, 0.0, 1.0)
    return max_opacity - (max_opacity - min_opacity) * normalized_distance


def _nearest_center_distances(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return np.asarray([], dtype=float)
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return _nearest_center_distances_chunked(points, centers)

    distances, _ = cKDTree(centers).query(points, k=1)
    return np.asarray(distances, dtype=float)


def _nearest_center_distances_chunked(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distances = np.full(points.shape[0], np.inf, dtype=float)
    point_chunk_size = 4096
    center_chunk_size = 512
    for point_start_index in range(0, points.shape[0], point_chunk_size):
        point_stop_index = min(point_start_index + point_chunk_size, points.shape[0])
        point_chunk = points[point_start_index:point_stop_index]
        chunk_distances_squared = np.full(point_chunk.shape[0], np.inf, dtype=float)
        for center_start_index in range(0, centers.shape[0], center_chunk_size):
            center_stop_index = min(center_start_index + center_chunk_size, centers.shape[0])
            deltas = point_chunk[:, None, :] - centers[None, center_start_index:center_stop_index, :]
            distances_squared = np.sum(deltas * deltas, axis=2)
            chunk_distances_squared = np.minimum(chunk_distances_squared, np.min(distances_squared, axis=1))
        distances[point_start_index:point_stop_index] = np.sqrt(chunk_distances_squared)
    return distances


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(float(minimum), min(float(maximum), float(value)))


def _rectilinear_dose_grid_from_lattice(
    pv: Any,
    lattice_points: np.ndarray,
    lattice_doses: np.ndarray,
    *,
    visible_mask: np.ndarray | None = None,
    max_voxels: int | None = None,
) -> Any:
    points = np.asarray(lattice_points, dtype=float)
    doses = np.asarray(lattice_doses, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("dose volume colorwash requires lattice_points with shape (n, 3)")
    if doses.shape != (points.shape[0],):
        raise ValueError("dose volume colorwash requires one dose value per lattice point")
    if visible_mask is not None:
        resolved_visible_mask = np.asarray(visible_mask, dtype=bool)
        if resolved_visible_mask.shape != (points.shape[0],):
            raise ValueError("dose volume colorwash visible mask must match lattice point count")
        if not bool(np.any(resolved_visible_mask)):
            raise ValueError("dose volume colorwash has no visible lattice points after filters")
        points, doses = _crop_lattice_to_visible_rectilinear_bounds(points, doses, resolved_visible_mask)

    x_values = np.unique(points[:, 0])
    y_values = np.unique(points[:, 1])
    z_values = np.unique(points[:, 2])
    if min(len(x_values), len(y_values), len(z_values)) < 2:
        raise ValueError("dose volume colorwash requires at least two lattice coordinates along each axis")
    expected_point_count = len(x_values) * len(y_values) * len(z_values)
    if expected_point_count != points.shape[0]:
        raise ValueError("dose volume colorwash requires a complete rectilinear dose lattice")

    x_indices = np.searchsorted(x_values, points[:, 0])
    y_indices = np.searchsorted(y_values, points[:, 1])
    z_indices = np.searchsorted(z_values, points[:, 2])
    dose_grid_values = np.empty((len(x_values), len(y_values), len(z_values)), dtype=float)
    visited_grid_points = np.zeros(dose_grid_values.shape, dtype=bool)
    for point_index, dose_value in enumerate(doses):
        grid_index = (int(x_indices[point_index]), int(y_indices[point_index]), int(z_indices[point_index]))
        if visited_grid_points[grid_index]:
            raise ValueError("dose volume colorwash requires unique rectilinear lattice points")
        dose_grid_values[grid_index] = float(dose_value)
        visited_grid_points[grid_index] = True
    if not bool(np.all(visited_grid_points)):
        raise ValueError("dose volume colorwash requires a complete rectilinear dose lattice")

    dose_grid = pv.RectilinearGrid(x_values, y_values, z_values)
    x_values, y_values, z_values, dose_grid_values = _downsample_rectilinear_grid_values(
        x_values,
        y_values,
        z_values,
        dose_grid_values,
        max_voxels=max_voxels,
    )
    dose_grid = pv.RectilinearGrid(x_values, y_values, z_values)
    dose_grid.point_data["dose"] = dose_grid_values.ravel(order="F")
    return dose_grid


def _crop_lattice_to_visible_rectilinear_bounds(
    points: np.ndarray,
    doses: np.ndarray,
    visible_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    selected_points = points[np.asarray(visible_mask, dtype=bool)]
    if selected_points.shape[0] == 0:
        raise ValueError("dose volume colorwash has no visible lattice points after filters")

    axis_values = (np.unique(points[:, 0]), np.unique(points[:, 1]), np.unique(points[:, 2]))
    selected_axis_values = (
        np.unique(selected_points[:, 0]),
        np.unique(selected_points[:, 1]),
        np.unique(selected_points[:, 2]),
    )
    axis_bounds: list[tuple[float, float]] = []
    for full_values, visible_values in zip(axis_values, selected_axis_values):
        min_index = int(np.searchsorted(full_values, float(np.min(visible_values))))
        max_index = int(np.searchsorted(full_values, float(np.max(visible_values))))
        if min_index == max_index and full_values.shape[0] > 1:
            if max_index < full_values.shape[0] - 1:
                max_index += 1
            else:
                min_index -= 1
        axis_bounds.append((float(full_values[min_index]), float(full_values[max_index])))

    crop_mask = (
        (points[:, 0] >= axis_bounds[0][0])
        & (points[:, 0] <= axis_bounds[0][1])
        & (points[:, 1] >= axis_bounds[1][0])
        & (points[:, 1] <= axis_bounds[1][1])
        & (points[:, 2] >= axis_bounds[2][0])
        & (points[:, 2] <= axis_bounds[2][1])
    )
    return points[crop_mask], doses[crop_mask]


def _downsample_rectilinear_grid_values(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    dose_grid_values: np.ndarray,
    *,
    max_voxels: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    resolved_max_voxels = _normalize_volume_max_voxels(max_voxels)
    if resolved_max_voxels is None:
        return x_values, y_values, z_values, dose_grid_values
    if int(np.prod(dose_grid_values.shape)) <= resolved_max_voxels:
        return x_values, y_values, z_values, dose_grid_values

    axis_lengths = [len(x_values), len(y_values), len(z_values)]
    strides = [1, 1, 1]
    while _sampled_voxel_count(axis_lengths, strides) > resolved_max_voxels:
        sampled_lengths = [_sampled_axis_length(axis_length, stride) for axis_length, stride in zip(axis_lengths, strides)]
        candidate_axes = [axis_index for axis_index, sampled_length in enumerate(sampled_lengths) if sampled_length > 2]
        if len(candidate_axes) == 0:
            break
        axis_index = max(candidate_axes, key=lambda index: sampled_lengths[index])
        strides[axis_index] += 1

    x_indices = _sample_axis_indices(len(x_values), strides[0])
    y_indices = _sample_axis_indices(len(y_values), strides[1])
    z_indices = _sample_axis_indices(len(z_values), strides[2])
    return (
        x_values[x_indices],
        y_values[y_indices],
        z_values[z_indices],
        dose_grid_values[np.ix_(x_indices, y_indices, z_indices)],
    )


def _normalize_volume_max_voxels(value: int | None) -> int | None:
    if value is None:
        return None
    resolved_value = int(value)
    if resolved_value <= 0:
        return None
    return max(8, resolved_value)


def _sampled_voxel_count(axis_lengths: list[int], strides: list[int]) -> int:
    sampled_lengths = [_sampled_axis_length(axis_length, stride) for axis_length, stride in zip(axis_lengths, strides)]
    return int(np.prod(sampled_lengths))


def _sampled_axis_length(axis_length: int, stride: int) -> int:
    return int(_sample_axis_indices(axis_length, stride).shape[0])


def _sample_axis_indices(axis_length: int, stride: int) -> np.ndarray:
    indices = np.arange(0, axis_length, max(1, int(stride)), dtype=int)
    if indices[-1] != axis_length - 1:
        indices = np.append(indices, axis_length - 1)
    return np.unique(indices)


def _add_axes(plotter: Any, settings: DoseNNPyVistaRenderSettings) -> None:
    plotter.add_axes(
        xlabel="X",
        ylabel="Y",
        zlabel="Z",
        line_width=2,
        viewport=(0.0, 0.0, 0.14, 0.14),
    )
    plotter.show_bounds(
        xtitle=settings.x_axis_label,
        ytitle=settings.y_axis_label,
        ztitle=settings.z_axis_label,
        font_size=int(settings.axes_tick_label_font_size),
        font_family=settings.axes_font_family,
        color="black",
        location="outer",
        ticks="outside",
        fmt="%.0f",
        use_2d=False,
    )
    cube_axes_actor = getattr(plotter.renderer, "cube_axes_actor", None)
    if cube_axes_actor is not None:
        for axis_index in range(3):
            cube_axes_actor.GetTitleTextProperty(axis_index).SetFontSize(int(settings.axes_title_font_size))
            cube_axes_actor.GetLabelTextProperty(axis_index).SetFontSize(int(settings.axes_tick_label_font_size))


def _configure_scalar_bar_actors(plotter: Any, settings: DoseNNPyVistaRenderSettings) -> None:
    scalar_bar_found = False
    view_props = plotter.renderer.GetViewProps()
    view_props.InitTraversal()
    for _ in range(view_props.GetNumberOfItems()):
        prop = view_props.GetNextProp()
        if prop.GetClassName() != "vtkScalarBarActor":
            continue
        scalar_bar_found = True
        prop.DrawFrameOff()
        prop.DrawBackgroundOff()
        prop.SetVerticalTitleSeparation(max(0, int(settings.dose_scalar_bar_title_separation)))
        _configure_clipped_scalar_bar_ticks(prop, settings)
    if scalar_bar_found:
        _add_scalar_bar_clipped_endpoint_labels(plotter, settings)


def _configure_clipped_scalar_bar_ticks(scalar_bar_actor: Any, settings: DoseNNPyVistaRenderSettings) -> None:
    tick_values = _dose_scalar_bar_tick_values(settings)
    if tick_values is None:
        return
    try:
        import vtk
    except ImportError:
        return

    custom_labels = vtk.vtkDoubleArray()
    for tick_value in tick_values:
        custom_labels.InsertNextValue(float(tick_value))
    scalar_bar_actor.SetCustomLabels(custom_labels)
    scalar_bar_actor.UseCustomLabelsOn()


def _add_scalar_bar_clipped_endpoint_labels(plotter: Any, settings: DoseNNPyVistaRenderSettings) -> None:
    clipped_range_labels = _dose_scalar_bar_clipped_range_labels(settings)
    if clipped_range_labels is None:
        return
    try:
        import vtk
    except ImportError:
        return

    below_label, above_label = clipped_range_labels
    if bool(settings.dose_scalar_bar_vertical):
        label_specs = (
            (below_label, 0.905, 0.18),
            (above_label, 0.905, 0.735),
        )
    else:
        label_specs = (
            (below_label, 0.24, 0.145),
            (above_label, 0.82, 0.145),
        )
    for label_text, x_value, y_value in label_specs:
        actor = vtk.vtkTextActor()
        actor.SetInput(str(label_text))
        actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        actor.SetPosition(float(x_value), float(y_value))
        actor.GetTextProperty().SetFontFamilyToArial()
        actor.GetTextProperty().SetFontSize(int(settings.dose_scalar_bar_label_font_size))
        actor.GetTextProperty().SetColor(0.0, 0.0, 0.0)
        actor.GetTextProperty().SetJustificationToLeft()
        actor.GetTextProperty().SetVerticalJustificationToCentered()
        actor.GetProperty().SetDisplayLocationToForeground()
        plotter.renderer.AddActor2D(actor)


def _add_scalar_bar_background_box(pv: Any, plotter: Any, settings: DoseNNPyVistaRenderSettings) -> None:
    try:
        import vtk
    except ImportError:
        return

    if bool(settings.dose_scalar_bar_vertical):
        x_min, y_min, width, height = 0.872, 0.155, 0.095, 0.725
    else:
        x_min, y_min, width, height = 0.225, 0.045, 0.61, 0.12

    points = vtk.vtkPoints()
    for x_value, y_value in (
        (x_min, y_min),
        (x_min + width, y_min),
        (x_min + width, y_min + height),
        (x_min, y_min + height),
    ):
        points.InsertNextPoint(float(x_value), float(y_value), 0.0)
    polygon = vtk.vtkPolygon()
    polygon.GetPointIds().SetNumberOfIds(4)
    for index in range(4):
        polygon.GetPointIds().SetId(index, index)

    cells = vtk.vtkCellArray()
    cells.InsertNextCell(polygon)
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(points)
    poly_data.SetPolys(cells)

    coordinate = vtk.vtkCoordinate()
    coordinate.SetCoordinateSystemToNormalizedViewport()
    mapper = vtk.vtkPolyDataMapper2D()
    mapper.SetInputData(poly_data)
    mapper.SetTransformCoordinate(coordinate)

    actor = vtk.vtkActor2D()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1.0, 1.0, 1.0)
    actor.GetProperty().SetOpacity(0.92)
    actor.GetProperty().SetDisplayLocationToBackground()
    actor.SetLayerNumber(0)
    plotter.renderer.AddActor2D(actor)


def _add_biopsy_points(
    pv: Any,
    plotter: Any,
    prepared_scene: DoseNNPreparedScene,
    settings: DoseNNPyVistaRenderSettings,
) -> None:
    if prepared_scene.biopsy_points.shape[0] == 0:
        return
    point_cloud = pv.PolyData(prepared_scene.biopsy_points)
    point_cloud["interpolated_dose"] = prepared_scene.interpolated_biopsy_doses
    plotter.add_mesh(
        point_cloud,
        name="biopsy_query_points",
        color=settings.biopsy_color,
        point_size=float(settings.biopsy_point_size),
        render_points_as_spheres=True,
    )


def _add_reference_biopsy_points(
    pv: Any,
    plotter: Any,
    prepared_scene: DoseNNPreparedScene,
    settings: DoseNNPyVistaRenderSettings,
) -> None:
    if prepared_scene.reference_biopsy_points.shape[0] == 0:
        return
    point_cloud = pv.PolyData(prepared_scene.reference_biopsy_points)
    plotter.add_mesh(
        point_cloud,
        name="reference_biopsy_points",
        color=settings.reference_biopsy_color,
        point_size=float(settings.reference_biopsy_point_size),
        render_points_as_spheres=True,
    )


def _add_nearest_neighbour_points(
    pv: Any,
    plotter: Any,
    prepared_scene: DoseNNPreparedScene,
    settings: DoseNNPyVistaRenderSettings,
) -> None:
    nearest_points = np.reshape(prepared_scene.nearest_lattice_points, (-1, 3))
    if nearest_points.shape[0] == 0:
        return
    point_cloud = pv.PolyData(nearest_points)
    point_cloud["nearest_dose"] = np.reshape(prepared_scene.nearest_lattice_doses, (-1,))
    plotter.add_mesh(
        point_cloud,
        name="dose_nn_nearest_points",
        color=settings.nearest_point_color,
        point_size=float(settings.nearest_point_size),
        render_points_as_spheres=True,
    )


def _add_nearest_neighbour_vectors(
    pv: Any,
    plotter: Any,
    prepared_scene: DoseNNPreparedScene,
    settings: DoseNNPyVistaRenderSettings,
) -> None:
    if prepared_scene.vector_start_points.shape[0] == 0:
        return
    line_mesh = _line_mesh_from_start_end_points(
        pv,
        prepared_scene.vector_start_points,
        prepared_scene.vector_end_points,
    )
    plotter.add_mesh(
        line_mesh,
        name="dose_nn_vectors",
        color=settings.vector_color,
        line_width=float(settings.vector_line_width),
    )


def _line_mesh_from_start_end_points(pv: Any, start_points: np.ndarray, end_points: np.ndarray) -> Any:
    if start_points.shape != end_points.shape:
        raise ValueError("vector start and end points must have matching shapes")
    if start_points.ndim != 2 or start_points.shape[1] != 3:
        raise ValueError("vector start and end points must have shape (n, 3)")
    points = np.empty((start_points.shape[0] * 2, 3), dtype=float)
    points[0::2] = start_points
    points[1::2] = end_points
    line_indices = np.arange(points.shape[0], dtype=np.int64).reshape((-1, 2))
    lines = np.column_stack((np.full(line_indices.shape[0], 2, dtype=np.int64), line_indices)).ravel()
    mesh = pv.PolyData(points)
    mesh.lines = lines
    return mesh


def _scene_metadata_dict(prepared_scene: DoseNNPreparedScene) -> dict[str, Any]:
    return _scene_metadata_from_scene(prepared_scene)


def _scene_metadata_from_scene(scene: DoseNNPreparedScene | DoseNNRenderScene) -> dict[str, Any]:
    metadata = scene.metadata
    return {
        "patient_uid": str(metadata.patient_uid),
        "biopsy_roi": str(metadata.biopsy_roi),
        "biopsy_index": metadata.biopsy_index,
        "localization_kind": str(metadata.localization_kind),
        "result_column": str(metadata.result_column),
        "source_label": str(metadata.source_label),
        "extra": _json_ready(dict(metadata.extra)),
    }


def _resolve_frame_trial_numbers(
    scene: DoseNNRenderScene,
    selected_trials: tuple[int, ...] | None,
    *,
    max_frames: int | None,
) -> tuple[int, ...]:
    available_trials = scene.available_trials
    if selected_trials is None:
        resolved_trials = available_trials
        if max_frames is not None:
            resolved_trials = resolved_trials[: int(max_frames)]
    else:
        resolved_trials = tuple(int(trial_number) for trial_number in tuple(selected_trials))
    if len(resolved_trials) == 0:
        raise ValueError("dose NN frame export requires at least one selected trial")
    missing_trials = sorted(set(resolved_trials).difference(set(available_trials)))
    if missing_trials:
        raise ValueError("dose NN frame export requested unavailable trials: {}".format(missing_trials))
    return resolved_trials


def _config_for_trial(base_config: DoseNNRenderConfig, trial_number: int) -> DoseNNRenderConfig:
    return DoseNNRenderConfig(
        selected_trials=(int(trial_number),),
        reference_trial_numbers=base_config.reference_trial_numbers,
        dose_threshold_min=base_config.dose_threshold_min,
        dose_threshold_max=base_config.dose_threshold_max,
        max_lattice_points=base_config.max_lattice_points,
        spatial_radius_mm=base_config.spatial_radius_mm,
        biopsy_point_stride=base_config.biopsy_point_stride,
        vector_stride=base_config.vector_stride,
        show_biopsy_points=base_config.show_biopsy_points,
        show_reference_biopsy_points=base_config.show_reference_biopsy_points,
        show_lattice_points=base_config.show_lattice_points,
        show_dose_colorwash=base_config.show_dose_colorwash,
        show_nearest_neighbour_points=base_config.show_nearest_neighbour_points,
        show_nearest_neighbour_vectors=base_config.show_nearest_neighbour_vectors,
    )


def _sanitize_output_name(value: str) -> str:
    sanitized_value = "".join(
        character if character.isalnum() or character in ("-", "_") else "_" for character in str(value)
    )
    sanitized_value = sanitized_value.strip("_")
    return sanitized_value or "dose_nn_trial"


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def _default_provenance_path(output_path: Path) -> Path:
    if output_path.suffix:
        return output_path.with_suffix(output_path.suffix + ".provenance.json")
    return output_path.with_name(output_path.name + ".provenance.json")


def _write_json(output_path: Path, payload: Mapping[str, Any]) -> None:
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def _import_pyvista() -> Any:
    try:
        import pyvista as pv
    except Exception as exc:  # pragma: no cover - exercised only when dependency is absent/broken.
        raise RuntimeError(
            "PyVista is required for the dose NN PyVista backend; install pyvista to use this renderer"
        ) from exc
    return pv