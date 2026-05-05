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
    import plotting_funcs
    import point_containment_tools

    resolved_render_jobs = tuple(render_jobs)
    for render_job in resolved_render_jobs:
        geometries_to_plot = []
        for render_layer in render_job.render_layers:
            if render_layer.layer_kind == "point_cloud":
                geometries_to_plot.append(
                    point_containment_tools.create_point_cloud(
                        render_layer.points,
                        np.asarray(render_layer.color, dtype=float),
                    )
                )
            elif render_layer.layer_kind == "geometry":
                geometries_to_plot.append(render_layer.geometry)
            else:
                raise ValueError("unsupported render layer kind: {}".format(render_layer.layer_kind))

        plotting_funcs.plot_geometries(
            *geometries_to_plot,
            label=render_job.scene_name,
            lookat_inp=None if render_job.camera_config is None else render_job.camera_config.lookat,
            up_inp=None if render_job.camera_config is None else render_job.camera_config.up,
            front_inp=None if render_job.camera_config is None else render_job.camera_config.front,
            zoom_inp=None if render_job.camera_config is None else render_job.camera_config.zoom,
        )

    return resolved_render_jobs


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