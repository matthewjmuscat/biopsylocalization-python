"""Stage-boundary debug render surfaces for optimizer v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from biopsy_optimizer.v2.contracts import OptimizerV2CandidatePool, OptimizerV2SearchRunResult


@dataclass(frozen=True)
class OptimizerV2RenderLayer:
    """One replayable render layer for a scene."""

    layer_name: str
    layer_kind: str
    points: Optional[np.ndarray] = None
    color: Optional[np.ndarray] = None
    geometry: Optional[Any] = None


@dataclass(frozen=True)
class OptimizerV2RenderCameraConfig:
    """Optional camera/view contract for replayable optimizer-v2 scenes."""

    lookat: np.ndarray
    up: np.ndarray
    front: np.ndarray
    zoom: float


@dataclass(frozen=True)
class OptimizerV2StageBoundaryRenderJob:
    """Geometry-free render job description for one optimizer stage boundary."""

    scene_name: str
    stage_name: str
    input_candidate_points: np.ndarray
    survivor_candidate_points: np.ndarray
    target_points: np.ndarray
    render_layers: Tuple[OptimizerV2RenderLayer, ...]
    camera_config: Optional[OptimizerV2RenderCameraConfig] = None
    nominal_biopsy_centroid: Optional[np.ndarray] = None
    winner_candidate_points: Optional[np.ndarray] = None


def build_stage_boundary_render_jobs(
    search_result: OptimizerV2SearchRunResult,
    candidate_pool: OptimizerV2CandidatePool,
    target_points_array: np.ndarray,
    nominal_biopsy_centroid: Optional[np.ndarray] = None,
    stage_names_to_render: Optional[Sequence[str]] = None,
    include_final_winner: bool = True,
    additional_render_layers: Optional[Sequence[OptimizerV2RenderLayer]] = None,
    additional_point_clouds: Optional[Sequence[Any]] = None,
    camera_config: Optional[OptimizerV2RenderCameraConfig] = None,
    scene_name_prefix: Optional[str] = None,
) -> Tuple[OptimizerV2StageBoundaryRenderJob, ...]:
    """Build one render-job description per selected stage boundary."""
    normalized_candidate_points = _validate_xyz_points_array(
        candidate_pool.candidate_points,
        "candidate_pool.candidate_points",
    )
    normalized_target_points = _validate_xyz_points_array(target_points_array, "target_points_array")
    normalized_nominal_biopsy_centroid = None
    if nominal_biopsy_centroid is not None:
        normalized_nominal_biopsy_centroid = _validate_single_xyz_point(
            nominal_biopsy_centroid,
            "nominal_biopsy_centroid",
        )

    resolved_stage_names_to_render = _resolve_stage_names_to_render(search_result, stage_names_to_render)
    resolved_additional_render_layers = tuple(additional_render_layers or ()) + tuple(
        build_geometry_render_layer(
            layer_name="additional_geometry_{}".format(geometry_index),
            geometry=geometry,
        )
        for geometry_index, geometry in enumerate(additional_point_clouds or ())
    )
    winner_candidate_points = None
    if include_final_winner and search_result.operational_winner_candidate_index_global is not None:
        winner_candidate_points = normalized_candidate_points[
            [int(search_result.operational_winner_candidate_index_global)]
        ]

    stage_boundary_render_jobs = []
    for stage_result in search_result.stage_results:
        if stage_result.stage_name not in resolved_stage_names_to_render:
            continue

        render_layers = [
            build_point_cloud_render_layer(
                layer_name="stage_input_candidates",
                points=normalized_candidate_points[
                    np.asarray(stage_result.input_candidate_indices_global, dtype=np.int32)
                ],
                color=np.array([1.0, 0.6, 0.0]),
            ),
            build_point_cloud_render_layer(
                layer_name="stage_survivors",
                points=normalized_candidate_points[
                    np.asarray(stage_result.survivor_candidate_indices_global, dtype=np.int32)
                ],
                color=np.array([0.0, 1.0, 0.0]),
            ),
            build_point_cloud_render_layer(
                layer_name="target_points",
                points=normalized_target_points,
                color=np.array([0.0, 0.0, 1.0]),
            ),
        ]
        if normalized_nominal_biopsy_centroid is not None:
            render_layers.append(
                build_point_cloud_render_layer(
                    layer_name="nominal_biopsy_centroid",
                    points=normalized_nominal_biopsy_centroid[np.newaxis, :],
                    color=np.array([1.0, 0.0, 0.0]),
                )
            )
        if winner_candidate_points is not None:
            render_layers.append(
                build_point_cloud_render_layer(
                    layer_name="operational_winner",
                    points=winner_candidate_points,
                    color=np.array([1.0, 0.0, 1.0]),
                )
            )
        render_layers.extend(resolved_additional_render_layers)

        stage_boundary_render_jobs.append(
            OptimizerV2StageBoundaryRenderJob(
                scene_name=_build_stage_boundary_scene_name(
                    stage_result.stage_name,
                    scene_name_prefix=scene_name_prefix,
                ),
                stage_name=stage_result.stage_name,
                input_candidate_points=normalized_candidate_points[
                    np.asarray(stage_result.input_candidate_indices_global, dtype=np.int32)
                ],
                survivor_candidate_points=normalized_candidate_points[
                    np.asarray(stage_result.survivor_candidate_indices_global, dtype=np.int32)
                ],
                target_points=normalized_target_points,
                render_layers=tuple(render_layers),
                camera_config=_validate_camera_config(camera_config),
                nominal_biopsy_centroid=normalized_nominal_biopsy_centroid,
                winner_candidate_points=winner_candidate_points,
            )
        )

    return tuple(stage_boundary_render_jobs)


def render_stage_boundary_candidate_clouds(
    search_result: OptimizerV2SearchRunResult,
    candidate_pool: OptimizerV2CandidatePool,
    target_points_array: np.ndarray,
    nominal_biopsy_centroid: Optional[np.ndarray] = None,
    stage_names_to_render: Optional[Sequence[str]] = None,
    include_final_winner: bool = True,
    additional_render_layers: Optional[Sequence[OptimizerV2RenderLayer]] = None,
    additional_point_clouds: Optional[Sequence[Any]] = None,
    camera_config: Optional[OptimizerV2RenderCameraConfig] = None,
    scene_name_prefix: Optional[str] = None,
) -> Tuple[OptimizerV2StageBoundaryRenderJob, ...]:
    """Build and render one Open3D scene per selected stage boundary."""
    stage_boundary_render_jobs = build_stage_boundary_render_jobs(
        search_result=search_result,
        candidate_pool=candidate_pool,
        target_points_array=target_points_array,
        nominal_biopsy_centroid=nominal_biopsy_centroid,
        stage_names_to_render=stage_names_to_render,
        include_final_winner=include_final_winner,
        additional_render_layers=additional_render_layers,
        additional_point_clouds=additional_point_clouds,
        camera_config=camera_config,
        scene_name_prefix=scene_name_prefix,
    )
    return render_scene_render_jobs(stage_boundary_render_jobs)


def _build_stage_boundary_scene_name(
    stage_name: str,
    scene_name_prefix: Optional[str] = None,
) -> str:
    resolved_scene_name = "optimizer_v2_{}".format(stage_name)
    if scene_name_prefix:
        resolved_scene_name = "{}__{}".format(scene_name_prefix, resolved_scene_name)
    return resolved_scene_name


def render_scene_render_jobs(
    render_jobs: Sequence[OptimizerV2StageBoundaryRenderJob],
) -> Tuple[OptimizerV2StageBoundaryRenderJob, ...]:
    """Render prebuilt scene jobs so they can be replayed or exported later."""
    import open3d as o3d
    import point_containment_tools

    resolved_render_jobs = tuple(render_jobs)
    if len(resolved_render_jobs) == 0:
        return resolved_render_jobs

    stage_geometries_by_name = {}
    layer_visibility_by_name = {}
    ordered_stage_names = []
    for render_job in resolved_render_jobs:
        ordered_stage_names.append(render_job.stage_name)
        stage_geometries = {}
        for render_layer in render_job.render_layers:
            if render_layer.layer_kind == "point_cloud":
                geometry = point_containment_tools.create_point_cloud(
                    render_layer.points,
                    np.asarray(render_layer.color, dtype=float),
                )
            elif render_layer.layer_kind == "geometry":
                geometry = render_layer.geometry
            else:
                raise ValueError("unsupported render layer kind: {}".format(render_layer.layer_kind))

            stage_geometries[render_layer.layer_name] = geometry
            layer_visibility_by_name.setdefault(render_layer.layer_name, True)

        stage_geometries_by_name[render_job.stage_name] = stage_geometries

    visualizer = o3d.visualization.VisualizerWithKeyCallback()
    visualizer.create_window(window_name=_build_multistage_scene_name(resolved_render_jobs))
    render_option = visualizer.get_render_option()
    render_option.show_coordinate_frame = True
    render_option.background_color = np.asarray([1.0, 1.0, 1.0])

    active_stage_name_ref = {"value": ordered_stage_names[0]}
    _sync_stage_geometries_to_viewer(
        visualizer,
        stage_geometries_by_name[active_stage_name_ref["value"]],
        layer_visibility_by_name,
        add_visible_layers=True,
    )

    initial_camera_config = resolved_render_jobs[0].camera_config
    if initial_camera_config is not None:
        view_control = visualizer.get_view_control()
        view_control.set_lookat(np.asarray(initial_camera_config.lookat, dtype=float))
        view_control.set_up(np.asarray(initial_camera_config.up, dtype=float))
        view_control.set_front(np.asarray(initial_camera_config.front, dtype=float))
        view_control.set_zoom(float(initial_camera_config.zoom))
    else:
        visualizer.reset_view_point(True)

    _register_render_layer_toggle_callbacks(
        visualizer,
        stage_geometries_by_name,
        active_stage_name_ref,
        layer_visibility_by_name,
        ordered_stage_names,
    )
    _print_render_layer_toggle_help(
        _build_multistage_scene_name(resolved_render_jobs),
        ordered_stage_names,
    )
    visualizer.run()
    visualizer.destroy_window()

    return resolved_render_jobs


def _register_render_layer_toggle_callbacks(
    visualizer,
    stage_geometries_by_name,
    active_stage_name_ref,
    layer_visibility_by_name,
    ordered_stage_names,
):
    key_to_layer_names = {
        ord("I"): ("stage_input_candidates",),
        ord("S"): ("stage_survivors",),
        ord("T"): ("target_points",),
        ord("N"): ("nominal_biopsy_centroid",),
        ord("W"): ("operational_winner",),
        ord("P"): ("planned_sampled_points",),
        ord("C"): ("planned_core_structure",),
        ord("L"): ("planned_centroid_line",),
        ord("O"): ("prostate_structure",),
        ord("U"): ("urethra_structure",),
        ord("R"): ("rectum_structure",),
        ord("D"): ("target_structure_surface",),
    }

    for key_code, layer_names in key_to_layer_names.items():
        visualizer.register_key_callback(
            key_code,
            _build_toggle_callback(
                stage_geometries_by_name,
                active_stage_name_ref,
                layer_visibility_by_name,
                layer_names,
            ),
        )

    for stage_index, stage_name in enumerate(ordered_stage_names[:9], start=1):
        visualizer.register_key_callback(
            ord(str(stage_index)),
            _build_stage_switch_callback(
                stage_geometries_by_name,
                active_stage_name_ref,
                layer_visibility_by_name,
                stage_name,
            ),
        )

    visualizer.register_key_callback(
        ord("A"),
        _build_set_all_visible_callback(
            stage_geometries_by_name,
            active_stage_name_ref,
            layer_visibility_by_name,
            visible=True,
        ),
    )
    visualizer.register_key_callback(
        ord("X"),
        _build_set_all_visible_callback(
            stage_geometries_by_name,
            active_stage_name_ref,
            layer_visibility_by_name,
            visible=False,
        ),
    )
    visualizer.register_key_callback(
        ord("H"),
        _build_help_callback(ordered_stage_names),
    )


def _build_toggle_callback(
    stage_geometries_by_name,
    active_stage_name_ref,
    layer_visibility_by_name,
    layer_names,
):
    resolved_layer_names = tuple(layer_names)

    def _toggle_callback(visualizer):
        did_toggle = False
        active_stage_geometries = stage_geometries_by_name[active_stage_name_ref["value"]]
        for layer_name in resolved_layer_names:
            geometry = active_stage_geometries.get(layer_name)
            if geometry is None:
                continue

            if layer_visibility_by_name.get(layer_name, True):
                visualizer.remove_geometry(geometry, reset_bounding_box=False)
                layer_visibility_by_name[layer_name] = False
            else:
                visualizer.add_geometry(geometry, reset_bounding_box=False)
                layer_visibility_by_name[layer_name] = True
            did_toggle = True

        if did_toggle:
            visualizer.update_renderer()
        return False

    return _toggle_callback


def _build_stage_switch_callback(
    stage_geometries_by_name,
    active_stage_name_ref,
    layer_visibility_by_name,
    target_stage_name,
):
    def _stage_switch_callback(visualizer):
        current_stage_name = active_stage_name_ref["value"]
        if target_stage_name == current_stage_name:
            return False

        _sync_stage_geometries_to_viewer(
            visualizer,
            stage_geometries_by_name[current_stage_name],
            layer_visibility_by_name,
            add_visible_layers=False,
        )
        _sync_stage_geometries_to_viewer(
            visualizer,
            stage_geometries_by_name[target_stage_name],
            layer_visibility_by_name,
            add_visible_layers=True,
        )
        active_stage_name_ref["value"] = target_stage_name
        visualizer.update_renderer()
        print("[optimizer-v2 render] switched to {}".format(target_stage_name))
        return False

    return _stage_switch_callback


def _build_set_all_visible_callback(
    stage_geometries_by_name,
    active_stage_name_ref,
    layer_visibility_by_name,
    visible,
):
    def _set_all_visible_callback(visualizer):
        active_stage_geometries = stage_geometries_by_name[active_stage_name_ref["value"]]
        for layer_name in tuple(layer_visibility_by_name.keys()):
            geometry = active_stage_geometries.get(layer_name)
            currently_visible = layer_visibility_by_name[layer_name]
            if visible and not currently_visible:
                if geometry is not None:
                    visualizer.add_geometry(geometry, reset_bounding_box=False)
                layer_visibility_by_name[layer_name] = True
            elif not visible and currently_visible:
                if geometry is not None:
                    visualizer.remove_geometry(geometry, reset_bounding_box=False)
                layer_visibility_by_name[layer_name] = False

        visualizer.update_renderer()
        return False

    return _set_all_visible_callback


def _build_help_callback(ordered_stage_names):
    def _help_callback(_visualizer):
        _print_render_layer_toggle_help("Current optimizer-v2 scene", ordered_stage_names)
        return False

    return _help_callback


def _print_render_layer_toggle_help(
    scene_name: str,
    ordered_stage_names: Sequence[str],
) -> None:
    print("[optimizer-v2 render] {} controls:".format(scene_name))
    if ordered_stage_names:
        print(
            "  stage keys: {}".format(
                ", ".join(
                    "{}={}".format(stage_index, stage_name)
                    for stage_index, stage_name in enumerate(ordered_stage_names[:9], start=1)
                )
            )
        )
    print("  I=input candidates, S=survivors, T=target points, N=nominal centroid, W=winner")
    print("  P=planned sampled points, C=planned core structure, L=planned centroid line")
    print("  O=prostate, U=urethra, R=rectum, D=target surface cloud")
    print("  A=show all, X=hide all, H=print help, Q/Esc=close window")


def _sync_stage_geometries_to_viewer(
    visualizer,
    stage_geometries,
    layer_visibility_by_name,
    add_visible_layers,
):
    for layer_name, geometry in stage_geometries.items():
        if not layer_visibility_by_name.get(layer_name, True):
            continue

        if add_visible_layers:
            visualizer.add_geometry(geometry, reset_bounding_box=False)
        else:
            visualizer.remove_geometry(geometry, reset_bounding_box=False)


def _build_multistage_scene_name(
    render_jobs: Sequence[OptimizerV2StageBoundaryRenderJob],
) -> str:
    first_scene_name = render_jobs[0].scene_name
    stage_suffix = "__optimizer_v2_{}".format(render_jobs[0].stage_name)
    if first_scene_name.endswith(stage_suffix):
        return first_scene_name[: -len(stage_suffix)] + "__optimizer_v2"
    return first_scene_name


def build_point_cloud_render_layer(
    layer_name: str,
    points: np.ndarray,
    color: np.ndarray,
) -> OptimizerV2RenderLayer:
    return OptimizerV2RenderLayer(
        layer_name=layer_name,
        layer_kind="point_cloud",
        points=_validate_xyz_points_array(points, layer_name),
        color=_validate_color_vector(color, "{}.color".format(layer_name)),
    )


def build_geometry_render_layer(
    layer_name: str,
    geometry: Any,
) -> OptimizerV2RenderLayer:
    return OptimizerV2RenderLayer(
        layer_name=layer_name,
        layer_kind="geometry",
        geometry=geometry,
    )


def build_success_failure_render_layers_from_chunk_score_result(
    chunk_score_result,
    candidate_local_chunk_index: int = 0,
    include_nominal_slice: bool = False,
    success_color: np.ndarray = np.array([0.0, 1.0, 0.0]),
    failure_color: np.ndarray = np.array([1.0, 0.0, 0.0]),
) -> Tuple[OptimizerV2RenderLayer, ...]:
    """Build success/failure point-cloud layers from one scored candidate chunk."""
    if chunk_score_result.relative_structure_localized_points is None:
        raise ValueError(
            "chunk_score_result.relative_structure_localized_points is missing; rerun scoring with include_relative_structure_localized_points_for_debug=True"
        )

    localized_points = _coerce_points_to_numpy(chunk_score_result.relative_structure_localized_points)
    containment_result = _coerce_points_to_numpy(chunk_score_result.structured_containment_result).astype(bool)
    if candidate_local_chunk_index < 0 or candidate_local_chunk_index >= localized_points.shape[0]:
        raise ValueError("candidate_local_chunk_index is out of range for this chunk score result")

    candidate_localized_points = localized_points[candidate_local_chunk_index]
    candidate_containment_result = containment_result[candidate_local_chunk_index]
    if chunk_score_result.chunk_layout.include_nominal and not include_nominal_slice:
        candidate_localized_points = candidate_localized_points[1:]
        candidate_containment_result = candidate_containment_result[1:]

    successful_points = candidate_localized_points[candidate_containment_result]
    failed_points = candidate_localized_points[~candidate_containment_result]
    render_layers = []
    candidate_index_global = int(chunk_score_result.candidate_indices_global[candidate_local_chunk_index])

    if successful_points.size > 0:
        render_layers.append(
            build_point_cloud_render_layer(
                layer_name="candidate_{}_success_points".format(candidate_index_global),
                points=successful_points.reshape(-1, 3),
                color=success_color,
            )
        )
    if failed_points.size > 0:
        render_layers.append(
            build_point_cloud_render_layer(
                layer_name="candidate_{}_failure_points".format(candidate_index_global),
                points=failed_points.reshape(-1, 3),
                color=failure_color,
            )
        )

    return tuple(render_layers)


def _resolve_stage_names_to_render(
    search_result: OptimizerV2SearchRunResult,
    stage_names_to_render: Optional[Sequence[str]],
) -> Tuple[str, ...]:
    available_stage_names = tuple(stage_result.stage_name for stage_result in search_result.stage_results)
    if stage_names_to_render is None:
        return available_stage_names

    normalized_stage_names_to_render = tuple(stage_names_to_render)
    invalid_stage_names = sorted(set(normalized_stage_names_to_render) - set(available_stage_names))
    if invalid_stage_names:
        raise ValueError(
            "unknown stage_names_to_render: {}".format(", ".join(invalid_stage_names))
        )
    return normalized_stage_names_to_render


def _validate_xyz_points_array(points_array: np.ndarray, array_name: str) -> np.ndarray:
    normalized_points_array = np.asarray(points_array, dtype=float)
    if normalized_points_array.ndim != 2 or normalized_points_array.shape[1] != 3:
        raise ValueError("{} must have shape (num_points, 3)".format(array_name))
    if normalized_points_array.shape[0] == 0:
        raise ValueError("{} cannot be empty".format(array_name))
    return normalized_points_array


def _validate_single_xyz_point(point: np.ndarray, point_name: str) -> np.ndarray:
    normalized_point = np.asarray(point, dtype=float).reshape(-1)
    if normalized_point.shape != (3,):
        raise ValueError("{} must have shape (3,)".format(point_name))
    return normalized_point


def _validate_color_vector(color: np.ndarray, color_name: str) -> np.ndarray:
    normalized_color = np.asarray(color, dtype=float).reshape(-1)
    if normalized_color.shape != (3,):
        raise ValueError("{} must have shape (3,)".format(color_name))
    return normalized_color


def _validate_camera_config(
    camera_config: Optional[OptimizerV2RenderCameraConfig],
) -> Optional[OptimizerV2RenderCameraConfig]:
    if camera_config is None:
        return None

    return OptimizerV2RenderCameraConfig(
        lookat=_validate_single_xyz_point(camera_config.lookat, "camera_config.lookat"),
        up=_validate_single_xyz_point(camera_config.up, "camera_config.up"),
        front=_validate_single_xyz_point(camera_config.front, "camera_config.front"),
        zoom=float(camera_config.zoom),
    )


def _coerce_points_to_numpy(points):
    if hasattr(points, "get"):
        return points.get()
    return np.asarray(points)


__all__ = [
    "OptimizerV2RenderCameraConfig",
    "OptimizerV2RenderLayer",
    "OptimizerV2StageBoundaryRenderJob",
    "build_geometry_render_layer",
    "build_point_cloud_render_layer",
    "build_success_failure_render_layers_from_chunk_score_result",
    "build_stage_boundary_render_jobs",
    "render_scene_render_jobs",
    "render_stage_boundary_candidate_clouds",
]