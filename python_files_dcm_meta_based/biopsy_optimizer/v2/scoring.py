"""Target-only scoring surfaces for optimizer v2.

This module intentionally stops at one scored candidate chunk. Stage policy,
survivor pruning, tie-break escalation, and downstream winner transport remain
outside this module.
"""

from __future__ import annotations

import time
from typing import Any, Optional, Sequence

import cupy as cp
import numpy as np

from biopsy_optimizer.v2.contracts import (
    OptimizerV2CandidatePool,
    OptimizerV2ChunkLayout,
    OptimizerV2ChunkScoreResult,
)
from preprocessing import containment_runner, localization_transformer
from preprocessing.transform_bank import SharedTransformBankPrefix


DEFAULT_CONTAINMENT_KERNEL_TYPE = (
    "one_to_one_pip_kernel_advanced_reparameterized_version_gpu_memory_performance_optimized"
)


def score_target_candidate_chunk(
    candidate_pool: OptimizerV2CandidatePool,
    chunk_layout: OptimizerV2ChunkLayout,
    nominal_biopsy_points: np.ndarray,
    nominal_biopsy_centroid: np.ndarray,
    nominal_biopsy_centroid_line: np.ndarray,
    biopsy_transform_bank_prefix: SharedTransformBankPrefix,
    target_relative_structures_nominal_plus_trials: Sequence[Sequence[np.ndarray]],
    target_structure_centroid: np.ndarray,
    target_transform_bank_prefix: SharedTransformBankPrefix,
    prepared_relative_structures_pack: Optional[Any] = None,
    objective_reducer_name: str = "mean_pd",
    max_test_structures_per_call: Optional[int] = None,
    validate_nearest_z_helper_against_ver5: bool = True,
    create_tested_candidate_dataframe: bool = True,
    include_relative_structure_localized_points_for_debug: bool = False,
    containment_log_sub_dirs_list: Optional[Sequence[str]] = None,
    containment_log_file_name: Optional[str] = None,
    include_edges_in_log: bool = False,
    kernel_type: str = DEFAULT_CONTAINMENT_KERNEL_TYPE,
    return_array_as: str = "numpy",
) -> OptimizerV2ChunkScoreResult:
    """Score one target-only candidate chunk under one shared trial prefix.

    The selection score is computed from stochastic trials only. If nominal rows
    are present, their reducer value is emitted separately as metadata.
    """
    chunk_total_start_time = time.perf_counter()
    normalized_candidate_points = _select_chunk_candidate_points(candidate_pool, chunk_layout)
    _validate_objective_reducer_name(objective_reducer_name)
    _validate_return_array_as(return_array_as)
    _validate_trial_alignment(chunk_layout, biopsy_transform_bank_prefix, target_transform_bank_prefix)
    _validate_target_relative_structure_pack(target_relative_structures_nominal_plus_trials, chunk_layout)

    biopsy_self_transform_start_time = time.perf_counter()
    candidate_biopsy_self_transform_batch = localization_transformer.build_candidate_biopsy_self_transform_batch(
        nominal_biopsy_points=nominal_biopsy_points,
        nominal_biopsy_centroid=nominal_biopsy_centroid,
        nominal_biopsy_centroid_line=nominal_biopsy_centroid_line,
        candidate_centroids=normalized_candidate_points,
        biopsy_transform_bank_prefix=biopsy_transform_bank_prefix,
        include_nominal=chunk_layout.include_nominal,
        return_array_as="cupy",
    )
    biopsy_self_transform_elapsed_seconds = time.perf_counter() - biopsy_self_transform_start_time

    relative_structure_localization_start_time = time.perf_counter()
    relative_structure_localized_biopsy_batch = localization_transformer.build_relative_structure_localized_biopsy_batch(
        candidate_biopsy_self_transform_batch=candidate_biopsy_self_transform_batch,
        relative_structure_centroid=target_structure_centroid,
        relative_structure_transform_bank_prefix=target_transform_bank_prefix,
        return_array_as="cupy",
    )
    relative_structure_localization_elapsed_seconds = (
        time.perf_counter() - relative_structure_localization_start_time
    )

    flatten_for_containment_start_time = time.perf_counter()
    aligned_containment_test_batch = localization_transformer.flatten_relative_structure_localized_batch_for_containment(
        relative_structure_localized_biopsy_batch=relative_structure_localized_biopsy_batch,
        nominal_relative_structure_index=chunk_layout.nominal_relative_structure_index,
        trial_relative_structure_start_index=chunk_layout.trial_relative_structure_start_index,
        return_array_as="numpy",
    )
    flatten_for_containment_elapsed_seconds = time.perf_counter() - flatten_for_containment_start_time

    containment_start_time = time.perf_counter()
    aligned_containment_run_result = containment_runner.run_aligned_containment_batch(
        list_of_relative_structures_containting_list_of_constant_zslices_arrays=target_relative_structures_nominal_plus_trials,
        aligned_containment_test_batch=aligned_containment_test_batch,
        prepared_relative_structures_pack=prepared_relative_structures_pack,
        log_sub_dirs_list=containment_log_sub_dirs_list,
        log_file_name=containment_log_file_name,
        include_edges_in_log=include_edges_in_log,
        kernel_type=kernel_type,
        return_array_as="cupy",
        max_test_structures_per_call=max_test_structures_per_call,
        validate_nearest_z_helper_against_ver5=validate_nearest_z_helper_against_ver5,
    )
    containment_elapsed_seconds = time.perf_counter() - containment_start_time
    containment_grandmother_elapsed_seconds = float(
        aligned_containment_run_result.grandmother_elapsed_seconds
    )
    containment_reshape_elapsed_seconds = float(
        aligned_containment_run_result.reshape_elapsed_seconds
    )
    containment_grandmother_mother_call_elapsed_seconds = float(
        aligned_containment_run_result.grandmother_mother_call_elapsed_seconds
    )
    containment_grandmother_mother_nearest_z_helper_name = str(
        aligned_containment_run_result.grandmother_mother_nearest_z_helper_name
    )
    containment_grandmother_mother_nearest_z_helper_elapsed_seconds = float(
        aligned_containment_run_result.grandmother_mother_nearest_z_helper_elapsed_seconds
    )
    containment_grandmother_mother_nearest_z_helper_validation_enabled = bool(
        aligned_containment_run_result.grandmother_mother_nearest_z_helper_validation_enabled
    )
    containment_grandmother_mother_nearest_z_helper_validation_elapsed_seconds = float(
        aligned_containment_run_result.grandmother_mother_nearest_z_helper_validation_elapsed_seconds
    )
    containment_grandmother_mother_nearest_z_helper_validation_match = bool(
        aligned_containment_run_result.grandmother_mother_nearest_z_helper_validation_match
    )
    containment_grandmother_mother_prepper_elapsed_seconds = float(
        aligned_containment_run_result.grandmother_mother_prepper_elapsed_seconds
    )
    containment_grandmother_mother_containment_execution_elapsed_seconds = float(
        aligned_containment_run_result.grandmother_mother_containment_execution_elapsed_seconds
    )
    containment_grandmother_mother_valid_point_compaction_elapsed_seconds = float(
        aligned_containment_run_result.grandmother_mother_valid_point_compaction_elapsed_seconds
    )
    containment_grandmother_mother_valid_point_upload_elapsed_seconds = float(
        aligned_containment_run_result.grandmother_mother_valid_point_upload_elapsed_seconds
    )
    containment_grandmother_mother_kernel_input_prepare_elapsed_seconds = float(
        aligned_containment_run_result.grandmother_mother_kernel_input_prepare_elapsed_seconds
    )
    containment_grandmother_mother_kernel_execution_elapsed_seconds = float(
        aligned_containment_run_result.grandmother_mother_kernel_execution_elapsed_seconds
    )
    containment_grandmother_mother_result_writeback_elapsed_seconds = float(
        aligned_containment_run_result.grandmother_mother_result_writeback_elapsed_seconds
    )
    containment_grandmother_chunk_slicing_elapsed_seconds = float(
        aligned_containment_run_result.grandmother_chunk_slicing_elapsed_seconds
    )
    containment_grandmother_chunk_concatenation_elapsed_seconds = float(
        aligned_containment_run_result.grandmother_chunk_concatenation_elapsed_seconds
    )
    containment_grandmother_chunk_count = int(
        aligned_containment_run_result.grandmother_chunk_count
    )
    containment_grandmother_used_chunking = bool(
        aligned_containment_run_result.grandmother_used_chunking
    )

    score_reduction_start_time = time.perf_counter()
    structured_containment_result_cp_arr = cp.asarray(aligned_containment_run_result.structured_containment_result)
    stochastic_containment_result_cp_arr, nominal_containment_result_cp_arr = _split_nominal_and_stochastic_results(
        structured_containment_result_cp_arr,
        chunk_layout,
    )

    candidate_trial_mean_point_scores_np_arr = None
    if objective_reducer_name == "mean_pd":
        candidate_trial_mean_point_scores_cp_arr = stochastic_containment_result_cp_arr.astype(cp.float32).mean(axis=2)
        candidate_trial_mean_point_scores_np_arr = cp.asnumpy(candidate_trial_mean_point_scores_cp_arr).astype(
            np.float32
        )

    point_probabilities_cp_arr = stochastic_containment_result_cp_arr.astype(cp.float32).mean(axis=1)
    stochastic_success_counts_cp_arr = stochastic_containment_result_cp_arr.sum(axis=1, dtype=cp.int32)
    candidate_scores_cp_arr = _reduce_point_probabilities(point_probabilities_cp_arr, objective_reducer_name)
    if nominal_containment_result_cp_arr is None:
        candidate_nominal_scores_np_arr = np.full(chunk_layout.num_candidates, np.nan, dtype=np.float32)
    else:
        candidate_nominal_scores_cp_arr = _reduce_point_probabilities(
            nominal_containment_result_cp_arr.astype(cp.float32),
            objective_reducer_name,
        )
        candidate_nominal_scores_np_arr = cp.asnumpy(candidate_nominal_scores_cp_arr).astype(np.float32)

    point_probabilities_np_arr = cp.asnumpy(point_probabilities_cp_arr).astype(np.float32)
    stochastic_success_counts_np_arr = cp.asnumpy(stochastic_success_counts_cp_arr).astype(np.int32)
    candidate_scores_np_arr = cp.asnumpy(candidate_scores_cp_arr).astype(np.float32)
    distance_to_target_centroid_mm = np.linalg.norm(
        normalized_candidate_points - np.asarray(target_structure_centroid, dtype=float).reshape(1, 3),
        axis=1,
    ).astype(np.float32)
    score_reduction_elapsed_seconds = time.perf_counter() - score_reduction_start_time

    tested_candidate_dataframe = None
    tested_candidate_dataframe_elapsed_seconds = 0.0
    if create_tested_candidate_dataframe:
        tested_candidate_dataframe_start_time = time.perf_counter()
        tested_candidate_dataframe = build_tested_candidate_dataframe_from_chunk_score_result(
            chunk_layout=chunk_layout,
            candidate_centroids=normalized_candidate_points,
            objective_reducer_name=objective_reducer_name,
            stochastic_success_counts=stochastic_success_counts_np_arr,
            point_probabilities=point_probabilities_np_arr,
            candidate_scores=candidate_scores_np_arr,
            candidate_nominal_scores=candidate_nominal_scores_np_arr,
            distance_to_target_centroid_mm=distance_to_target_centroid_mm,
        )
        tested_candidate_dataframe_elapsed_seconds = (
            time.perf_counter() - tested_candidate_dataframe_start_time
        )

    total_elapsed_seconds = time.perf_counter() - chunk_total_start_time

    return OptimizerV2ChunkScoreResult(
        chunk_layout=chunk_layout,
        candidate_indices_global=np.asarray(chunk_layout.candidate_indices_global, dtype=np.int32),
        candidate_centroids=normalized_candidate_points,
        objective_reducer_name=objective_reducer_name,
        structured_containment_result=_coerce_output_array(structured_containment_result_cp_arr, return_array_as),
        stochastic_success_counts=stochastic_success_counts_np_arr,
        point_probabilities=point_probabilities_np_arr,
        candidate_trial_mean_point_scores=candidate_trial_mean_point_scores_np_arr,
        candidate_scores=candidate_scores_np_arr,
        candidate_nominal_scores=candidate_nominal_scores_np_arr,
        distance_to_target_centroid_mm=distance_to_target_centroid_mm,
        biopsy_self_transform_elapsed_seconds=float(biopsy_self_transform_elapsed_seconds),
        relative_structure_localization_elapsed_seconds=float(
            relative_structure_localization_elapsed_seconds
        ),
        flatten_for_containment_elapsed_seconds=float(flatten_for_containment_elapsed_seconds),
        containment_elapsed_seconds=float(containment_elapsed_seconds),
        containment_grandmother_elapsed_seconds=float(
            containment_grandmother_elapsed_seconds
        ),
        containment_reshape_elapsed_seconds=float(containment_reshape_elapsed_seconds),
        containment_grandmother_mother_call_elapsed_seconds=float(
            containment_grandmother_mother_call_elapsed_seconds
        ),
        containment_grandmother_mother_nearest_z_helper_name=(
            containment_grandmother_mother_nearest_z_helper_name
        ),
        containment_grandmother_mother_nearest_z_helper_elapsed_seconds=float(
            containment_grandmother_mother_nearest_z_helper_elapsed_seconds
        ),
        containment_grandmother_mother_nearest_z_helper_validation_enabled=bool(
            containment_grandmother_mother_nearest_z_helper_validation_enabled
        ),
        containment_grandmother_mother_nearest_z_helper_validation_elapsed_seconds=float(
            containment_grandmother_mother_nearest_z_helper_validation_elapsed_seconds
        ),
        containment_grandmother_mother_nearest_z_helper_validation_match=bool(
            containment_grandmother_mother_nearest_z_helper_validation_match
        ),
        containment_grandmother_mother_prepper_elapsed_seconds=float(
            containment_grandmother_mother_prepper_elapsed_seconds
        ),
        containment_grandmother_mother_containment_execution_elapsed_seconds=float(
            containment_grandmother_mother_containment_execution_elapsed_seconds
        ),
        containment_grandmother_mother_valid_point_compaction_elapsed_seconds=float(
            containment_grandmother_mother_valid_point_compaction_elapsed_seconds
        ),
        containment_grandmother_mother_valid_point_upload_elapsed_seconds=float(
            containment_grandmother_mother_valid_point_upload_elapsed_seconds
        ),
        containment_grandmother_mother_kernel_input_prepare_elapsed_seconds=float(
            containment_grandmother_mother_kernel_input_prepare_elapsed_seconds
        ),
        containment_grandmother_mother_kernel_execution_elapsed_seconds=float(
            containment_grandmother_mother_kernel_execution_elapsed_seconds
        ),
        containment_grandmother_mother_result_writeback_elapsed_seconds=float(
            containment_grandmother_mother_result_writeback_elapsed_seconds
        ),
        containment_grandmother_chunk_slicing_elapsed_seconds=float(
            containment_grandmother_chunk_slicing_elapsed_seconds
        ),
        containment_grandmother_chunk_concatenation_elapsed_seconds=float(
            containment_grandmother_chunk_concatenation_elapsed_seconds
        ),
        containment_grandmother_chunk_count=int(containment_grandmother_chunk_count),
        containment_grandmother_used_chunking=bool(
            containment_grandmother_used_chunking
        ),
        score_reduction_elapsed_seconds=float(score_reduction_elapsed_seconds),
        tested_candidate_dataframe_elapsed_seconds=float(tested_candidate_dataframe_elapsed_seconds),
        total_elapsed_seconds=float(total_elapsed_seconds),
        relative_structure_localized_points=(
            _coerce_output_array(relative_structure_localized_biopsy_batch.transformed_points, return_array_as)
            if include_relative_structure_localized_points_for_debug
            else None
        ),
        tested_candidate_dataframe=tested_candidate_dataframe,
    )


def build_tested_candidate_dataframe_from_chunk_score_result(
    chunk_layout: OptimizerV2ChunkLayout,
    candidate_centroids: np.ndarray,
    objective_reducer_name: str,
    stochastic_success_counts: np.ndarray,
    point_probabilities: np.ndarray,
    candidate_scores: np.ndarray,
    candidate_nominal_scores: np.ndarray,
    distance_to_target_centroid_mm: np.ndarray,
):
    """Build one candidate-summary dataframe for a scored chunk."""
    import pandas

    normalized_candidate_centroids = np.asarray(candidate_centroids, dtype=np.float32)
    normalized_stochastic_success_counts = np.asarray(stochastic_success_counts, dtype=np.int32)
    normalized_point_probabilities = np.asarray(point_probabilities, dtype=np.float32)
    normalized_candidate_scores = np.asarray(candidate_scores, dtype=np.float32)
    normalized_candidate_nominal_scores = np.asarray(candidate_nominal_scores, dtype=np.float32)
    normalized_distance_to_target_centroid_mm = np.asarray(distance_to_target_centroid_mm, dtype=np.float32)

    return pandas.DataFrame(
        {
            "Candidate global index": np.asarray(chunk_layout.candidate_indices_global, dtype=np.int32),
            "Candidate local chunk index": np.arange(chunk_layout.num_candidates, dtype=np.int32),
            "Candidate X": normalized_candidate_centroids[:, 0],
            "Candidate Y": normalized_candidate_centroids[:, 1],
            "Candidate Z": normalized_candidate_centroids[:, 2],
            "Objective value": normalized_candidate_scores,
            "Objective reducer name": [objective_reducer_name] * chunk_layout.num_candidates,
            "Nominal objective value": normalized_candidate_nominal_scores,
            "Distance to target centroid mm": normalized_distance_to_target_centroid_mm,
            "Num trials used": np.full(chunk_layout.num_candidates, chunk_layout.num_trials, dtype=np.int32),
            "Num biopsy sample points": np.full(
                chunk_layout.num_candidates,
                normalized_point_probabilities.shape[1],
                dtype=np.int32,
            ),
            "Mean point probability": normalized_point_probabilities.mean(axis=1, dtype=np.float32),
            "Max point probability": normalized_point_probabilities.max(axis=1),
            "Min point probability": normalized_point_probabilities.min(axis=1),
            "Total successes all points": normalized_stochastic_success_counts.sum(axis=1, dtype=np.int32),
        }
    )


def _select_chunk_candidate_points(
    candidate_pool: OptimizerV2CandidatePool,
    chunk_layout: OptimizerV2ChunkLayout,
) -> np.ndarray:
    normalized_candidate_points = np.asarray(candidate_pool.candidate_points, dtype=float)
    if normalized_candidate_points.ndim != 2 or normalized_candidate_points.shape[1] != 3:
        raise ValueError("candidate_pool.candidate_points must have shape (num_candidates, 3)")

    if chunk_layout.num_candidates == 0:
        raise ValueError("chunk_layout must contain at least one candidate")

    max_candidate_index = max(chunk_layout.candidate_indices_global)
    if max_candidate_index >= normalized_candidate_points.shape[0]:
        raise ValueError(
            "chunk_layout references candidate index {} but only {} candidates are available".format(
                max_candidate_index,
                normalized_candidate_points.shape[0],
            )
        )

    return normalized_candidate_points[np.asarray(chunk_layout.candidate_indices_global, dtype=np.int32)]


def _validate_target_relative_structure_pack(
    target_relative_structures_nominal_plus_trials: Sequence[Sequence[np.ndarray]],
    chunk_layout: OptimizerV2ChunkLayout,
) -> None:
    required_num_relative_structures = chunk_layout.trial_relative_structure_start_index + chunk_layout.num_trials
    if chunk_layout.include_nominal:
        required_num_relative_structures = max(
            required_num_relative_structures,
            chunk_layout.nominal_relative_structure_index + 1,
        )

    if len(target_relative_structures_nominal_plus_trials) < required_num_relative_structures:
        raise ValueError(
            "target_relative_structures_nominal_plus_trials must contain at least {} structures for this chunk layout".format(
                required_num_relative_structures,
            )
        )


def _validate_trial_alignment(
    chunk_layout: OptimizerV2ChunkLayout,
    biopsy_transform_bank_prefix: SharedTransformBankPrefix,
    target_transform_bank_prefix: SharedTransformBankPrefix,
) -> None:
    if chunk_layout.num_trials <= 0:
        raise ValueError("chunk_layout.num_trials must be positive for stochastic scoring")
    if biopsy_transform_bank_prefix.requested_num_trials != chunk_layout.num_trials:
        raise ValueError(
            "biopsy_transform_bank_prefix.requested_num_trials must equal chunk_layout.num_trials"
        )
    if target_transform_bank_prefix.requested_num_trials != chunk_layout.num_trials:
        raise ValueError(
            "target_transform_bank_prefix.requested_num_trials must equal chunk_layout.num_trials"
        )


def _split_nominal_and_stochastic_results(
    structured_containment_result_cp_arr: cp.ndarray,
    chunk_layout: OptimizerV2ChunkLayout,
):
    if chunk_layout.include_nominal:
        nominal_containment_result_cp_arr = structured_containment_result_cp_arr[:, 0, :]
        stochastic_containment_result_cp_arr = structured_containment_result_cp_arr[:, 1:, :]
    else:
        nominal_containment_result_cp_arr = None
        stochastic_containment_result_cp_arr = structured_containment_result_cp_arr

    if stochastic_containment_result_cp_arr.shape[1] != chunk_layout.num_trials:
        raise ValueError(
            "structured stochastic trial dimension {} does not match chunk_layout.num_trials {}".format(
                stochastic_containment_result_cp_arr.shape[1],
                chunk_layout.num_trials,
            )
        )
    return stochastic_containment_result_cp_arr, nominal_containment_result_cp_arr


def _reduce_point_probabilities(point_probabilities_cp_arr: cp.ndarray, objective_reducer_name: str) -> cp.ndarray:
    if objective_reducer_name == "mean_pd":
        return point_probabilities_cp_arr.mean(axis=1)
    if objective_reducer_name == "max_pd":
        return point_probabilities_cp_arr.max(axis=1)
    if objective_reducer_name == "min_pd":
        return point_probabilities_cp_arr.min(axis=1)
    raise ValueError("unsupported objective_reducer_name: {}".format(objective_reducer_name))


def _validate_objective_reducer_name(objective_reducer_name: str) -> None:
    if objective_reducer_name not in {"mean_pd", "max_pd", "min_pd"}:
        raise ValueError("objective_reducer_name must be one of 'mean_pd', 'max_pd', or 'min_pd'")


def _validate_return_array_as(return_array_as: str) -> None:
    if return_array_as not in {"cupy", "numpy"}:
        raise ValueError("return_array_as must be 'cupy' or 'numpy'")


def _coerce_output_array(output_array: Any, return_array_as: str):
    if return_array_as == "cupy":
        return output_array
    return cp.asnumpy(output_array)


__all__ = [
    "DEFAULT_CONTAINMENT_KERNEL_TYPE",
    "build_tested_candidate_dataframe_from_chunk_score_result",
    "score_target_candidate_chunk",
]