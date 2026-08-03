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
    dose_colormap: str = "viridis"
    lattice_point_size: float = 5.0
    biopsy_point_size: float = 12.0
    nearest_point_size: float = 8.0
    vector_line_width: float = 2.0
    biopsy_color: str = "crimson"
    nearest_point_color: str = "black"
    vector_color: str = "dimgray"
    show_axes: bool = True
    show_scalar_bar: bool = True
    dose_scalar_bar_title: str = "Dose"
    camera_position: Any | None = None


@dataclass(frozen=True, slots=True)
class DoseNNPyVistaExportResult:
    """Paths written by a PyVista dose NN export."""

    screenshot_path: Path
    provenance_path: Path


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
    if prepared_scene.config.show_biopsy_points:
        _add_biopsy_points(pv, plotter, prepared_scene, resolved_settings)
    if prepared_scene.config.show_nearest_neighbour_points:
        _add_nearest_neighbour_points(pv, plotter, prepared_scene, resolved_settings)
    if prepared_scene.config.show_nearest_neighbour_vectors:
        _add_nearest_neighbour_vectors(pv, plotter, prepared_scene, resolved_settings)

    if resolved_settings.show_axes:
        plotter.add_axes()
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
        plotter.screenshot(str(resolved_output_path))
    finally:
        if close_plotter:
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
            "nearest_neighbour_point_count": int(np.reshape(prepared_scene.nearest_lattice_points, (-1, 3)).shape[0]),
            "vector_count": int(prepared_scene.num_vectors),
            "selected_trials": [int(trial_number) for trial_number in np.unique(prepared_scene.trial_numbers)],
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
        scalar_bar_args={"title": settings.dose_scalar_bar_title},
    )


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
    metadata = prepared_scene.metadata
    return {
        "patient_uid": str(metadata.patient_uid),
        "biopsy_roi": str(metadata.biopsy_roi),
        "biopsy_index": metadata.biopsy_index,
        "localization_kind": str(metadata.localization_kind),
        "result_column": str(metadata.result_column),
        "source_label": str(metadata.source_label),
        "extra": _json_ready(dict(metadata.extra)),
    }


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