"""Stage-boundary debug render surfaces for optimizer v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from biopsy_optimizer.v2.contracts import OptimizerV2CandidatePool, OptimizerV2SearchRunResult


@dataclass(frozen=True)
class OptimizerV2StageBoundaryRenderJob:
    """Geometry-free render job description for one optimizer stage boundary."""

    stage_name: str
    input_candidate_points: np.ndarray
    survivor_candidate_points: np.ndarray
    target_points: np.ndarray
    nominal_biopsy_centroid: Optional[np.ndarray] = None
    winner_candidate_points: Optional[np.ndarray] = None


def build_stage_boundary_render_jobs(
    search_result: OptimizerV2SearchRunResult,
    candidate_pool: OptimizerV2CandidatePool,
    target_points_array: np.ndarray,
    nominal_biopsy_centroid: Optional[np.ndarray] = None,
    stage_names_to_render: Optional[Sequence[str]] = None,
    include_final_winner: bool = True,
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
    winner_candidate_points = None
    if include_final_winner and search_result.operational_winner_candidate_index_global is not None:
        winner_candidate_points = normalized_candidate_points[
            [int(search_result.operational_winner_candidate_index_global)]
        ]

    stage_boundary_render_jobs = []
    for stage_result in search_result.stage_results:
        if stage_result.stage_name not in resolved_stage_names_to_render:
            continue

        stage_boundary_render_jobs.append(
            OptimizerV2StageBoundaryRenderJob(
                stage_name=stage_result.stage_name,
                input_candidate_points=normalized_candidate_points[
                    np.asarray(stage_result.input_candidate_indices_global, dtype=np.int32)
                ],
                survivor_candidate_points=normalized_candidate_points[
                    np.asarray(stage_result.survivor_candidate_indices_global, dtype=np.int32)
                ],
                target_points=normalized_target_points,
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
    additional_point_clouds: Optional[Sequence[Any]] = None,
) -> Tuple[OptimizerV2StageBoundaryRenderJob, ...]:
    """Render one Open3D scene per selected stage boundary."""
    import plotting_funcs
    import point_containment_tools

    stage_boundary_render_jobs = build_stage_boundary_render_jobs(
        search_result=search_result,
        candidate_pool=candidate_pool,
        target_points_array=target_points_array,
        nominal_biopsy_centroid=nominal_biopsy_centroid,
        stage_names_to_render=stage_names_to_render,
        include_final_winner=include_final_winner,
    )
    additional_point_clouds_list = list(additional_point_clouds or [])

    for render_job in stage_boundary_render_jobs:
        stage_input_point_cloud = point_containment_tools.create_point_cloud(
            render_job.input_candidate_points,
            np.array([1.0, 0.6, 0.0]),
        )
        survivor_point_cloud = point_containment_tools.create_point_cloud(
            render_job.survivor_candidate_points,
            np.array([0.0, 1.0, 0.0]),
        )
        target_point_cloud = point_containment_tools.create_point_cloud(
            render_job.target_points,
            np.array([0.0, 0.0, 1.0]),
        )
        geometries_to_plot = [
            stage_input_point_cloud,
            survivor_point_cloud,
            target_point_cloud,
            *additional_point_clouds_list,
        ]

        if render_job.nominal_biopsy_centroid is not None:
            geometries_to_plot.append(
                point_containment_tools.create_point_cloud(
                    render_job.nominal_biopsy_centroid[np.newaxis, :],
                    np.array([1.0, 0.0, 0.0]),
                )
            )
        if render_job.winner_candidate_points is not None:
            geometries_to_plot.append(
                point_containment_tools.create_point_cloud(
                    render_job.winner_candidate_points,
                    np.array([1.0, 0.0, 1.0]),
                )
            )

        plotting_funcs.plot_geometries(
            *geometries_to_plot,
            label="optimizer_v2_{}".format(render_job.stage_name),
        )

    return stage_boundary_render_jobs


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


__all__ = [
    "OptimizerV2StageBoundaryRenderJob",
    "build_stage_boundary_render_jobs",
    "render_stage_boundary_candidate_clouds",
]