"""Stage orchestration for optimizer v2 target-only ranking."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import pandas

from biopsy_optimizer.v2.config import OptimizerV2SearchConfig, OptimizerV2StageConfig
from biopsy_optimizer.v2.contracts import (
    OptimizerV2CandidatePool,
    OptimizerV2ChunkLayout,
    OptimizerV2ChunkScoreResult,
    OptimizerV2SearchRunResult,
    OptimizerV2StageRunResult,
    OptimizerV2WinnerValidationResult,
)
from biopsy_optimizer.v2.scoring import (
    DEFAULT_CONTAINMENT_KERNEL_TYPE,
    score_target_candidate_chunk,
)
from preprocessing.transform_bank import SharedTransformBankPrefix


DEFAULT_STAGE_PROVISIONAL_TIE_BREAK_METHOD = "score_desc_distance_candidate_index__provisional"


def run_target_staged_candidate_search(
    candidate_pool: OptimizerV2CandidatePool,
    search_config: OptimizerV2SearchConfig,
    nominal_biopsy_points: np.ndarray,
    nominal_biopsy_centroid: np.ndarray,
    nominal_biopsy_centroid_line: np.ndarray,
    biopsy_transform_bank_prefix_provider: Callable[[int], SharedTransformBankPrefix],
    target_relative_structures_nominal_plus_trials_provider: Callable[[int], Sequence[Sequence[np.ndarray]]],
    target_structure_centroid: np.ndarray,
    target_transform_bank_prefix_provider: Callable[[int], SharedTransformBankPrefix],
    initial_candidate_indices_global: Optional[Sequence[int]] = None,
    objective_reducer_name: str = "mean_pd",
    max_candidates_per_chunk: Optional[int] = None,
    include_nominal: bool = True,
    nominal_relative_structure_index: int = 0,
    trial_relative_structure_start_index: int = 1,
    downstream_comparable_trial_count: Optional[int] = None,
    max_test_structures_per_call: Optional[int] = None,
    containment_log_sub_dirs_list: Optional[Sequence[str]] = None,
    containment_log_file_name: Optional[str] = None,
    include_edges_in_log: bool = False,
    kernel_type: str = DEFAULT_CONTAINMENT_KERNEL_TYPE,
    return_array_as: str = "numpy",
) -> OptimizerV2SearchRunResult:
    """Run the staged A -> B -> C target-only optimizer-v2 search.

    The runner remains orchestration-only. Stage-specific target structure packs
    and transform-bank prefixes are supplied by providers so geometry generation
    stays outside optimizer-v2.
    """
    normalized_candidate_indices_global = _resolve_initial_candidate_indices_global(
        candidate_pool,
        initial_candidate_indices_global,
    )
    resolved_max_candidates_per_chunk = _resolve_max_candidates_per_chunk(
        normalized_candidate_indices_global.size,
        max_candidates_per_chunk,
    )
    if normalized_candidate_indices_global.size == 0:
        empty_dataframe = pandas.DataFrame()
        return OptimizerV2SearchRunResult(
            stage_results=tuple(),
            tested_candidate_dataframe=empty_dataframe,
            ranked_candidate_dataframe=empty_dataframe.copy(),
            operational_winner_candidate_index_global=None,
        )

    stage_results = []
    stage_tested_candidate_dataframes = []
    current_candidate_indices_global = normalized_candidate_indices_global

    for stage_config in search_config.stage_configs:
        stage_result = _run_target_candidate_stage(
            candidate_pool=candidate_pool,
            stage_config=stage_config,
            candidate_indices_global=current_candidate_indices_global,
            nominal_biopsy_points=nominal_biopsy_points,
            nominal_biopsy_centroid=nominal_biopsy_centroid,
            nominal_biopsy_centroid_line=nominal_biopsy_centroid_line,
            biopsy_transform_bank_prefix_provider=biopsy_transform_bank_prefix_provider,
            target_relative_structures_nominal_plus_trials_provider=target_relative_structures_nominal_plus_trials_provider,
            target_structure_centroid=target_structure_centroid,
            target_transform_bank_prefix_provider=target_transform_bank_prefix_provider,
            objective_reducer_name=objective_reducer_name,
            max_candidates_per_chunk=resolved_max_candidates_per_chunk,
            include_nominal=include_nominal,
            nominal_relative_structure_index=nominal_relative_structure_index,
            trial_relative_structure_start_index=trial_relative_structure_start_index,
            max_test_structures_per_call=max_test_structures_per_call,
            containment_log_sub_dirs_list=containment_log_sub_dirs_list,
            containment_log_file_name=containment_log_file_name,
            include_edges_in_log=include_edges_in_log,
            kernel_type=kernel_type,
            return_array_as=return_array_as,
        )
        stage_results.append(stage_result)
        stage_tested_candidate_dataframes.append(stage_result.tested_candidate_dataframe)
        current_candidate_indices_global = stage_result.survivor_candidate_indices_global

    combined_tested_candidate_dataframe = pandas.concat(stage_tested_candidate_dataframes, ignore_index=True)
    final_ranked_candidate_dataframe = stage_results[-1].ranked_candidate_dataframe.copy()
    operational_winner_candidate_index_global = None
    if not final_ranked_candidate_dataframe.empty:
        operational_winner_candidate_index_global = int(final_ranked_candidate_dataframe.iloc[0]["Candidate global index"])

    winner_validation_result = _build_winner_validation_result(
        candidate_pool=candidate_pool,
        final_ranked_candidate_dataframe=final_ranked_candidate_dataframe,
        final_stage_result=stage_results[-1],
        nominal_biopsy_points=nominal_biopsy_points,
        nominal_biopsy_centroid=nominal_biopsy_centroid,
        nominal_biopsy_centroid_line=nominal_biopsy_centroid_line,
        biopsy_transform_bank_prefix_provider=biopsy_transform_bank_prefix_provider,
        target_relative_structures_nominal_plus_trials_provider=target_relative_structures_nominal_plus_trials_provider,
        target_structure_centroid=target_structure_centroid,
        target_transform_bank_prefix_provider=target_transform_bank_prefix_provider,
        objective_reducer_name=objective_reducer_name,
        include_nominal=include_nominal,
        nominal_relative_structure_index=nominal_relative_structure_index,
        trial_relative_structure_start_index=trial_relative_structure_start_index,
        downstream_comparable_trial_count=downstream_comparable_trial_count,
        max_test_structures_per_call=max_test_structures_per_call,
        containment_log_sub_dirs_list=containment_log_sub_dirs_list,
        containment_log_file_name=containment_log_file_name,
        include_edges_in_log=include_edges_in_log,
        kernel_type=kernel_type,
        return_array_as=return_array_as,
    )
    if winner_validation_result is not None:
        final_ranked_candidate_dataframe = _apply_winner_validation_to_candidate_dataframe(
            final_ranked_candidate_dataframe,
            stage_name=stage_results[-1].stage_name,
            winner_validation_result=winner_validation_result,
        )
        stage_results[-1].ranked_candidate_dataframe = final_ranked_candidate_dataframe.copy()
        combined_tested_candidate_dataframe = _apply_winner_validation_to_candidate_dataframe(
            combined_tested_candidate_dataframe,
            stage_name=stage_results[-1].stage_name,
            winner_validation_result=winner_validation_result,
        )
        stage_results[-1].tested_candidate_dataframe = _apply_winner_validation_to_candidate_dataframe(
            stage_results[-1].tested_candidate_dataframe,
            stage_name=stage_results[-1].stage_name,
            winner_validation_result=winner_validation_result,
        )

    return OptimizerV2SearchRunResult(
        stage_results=tuple(stage_results),
        tested_candidate_dataframe=combined_tested_candidate_dataframe,
        ranked_candidate_dataframe=final_ranked_candidate_dataframe,
        operational_winner_candidate_index_global=operational_winner_candidate_index_global,
        winner_validation_result=winner_validation_result,
    )


def _run_target_candidate_stage(
    candidate_pool: OptimizerV2CandidatePool,
    stage_config: OptimizerV2StageConfig,
    candidate_indices_global: np.ndarray,
    nominal_biopsy_points: np.ndarray,
    nominal_biopsy_centroid: np.ndarray,
    nominal_biopsy_centroid_line: np.ndarray,
    biopsy_transform_bank_prefix_provider: Callable[[int], SharedTransformBankPrefix],
    target_relative_structures_nominal_plus_trials_provider: Callable[[int], Sequence[Sequence[np.ndarray]]],
    target_structure_centroid: np.ndarray,
    target_transform_bank_prefix_provider: Callable[[int], SharedTransformBankPrefix],
    objective_reducer_name: str,
    max_candidates_per_chunk: int,
    include_nominal: bool,
    nominal_relative_structure_index: int,
    trial_relative_structure_start_index: int,
    max_test_structures_per_call: Optional[int],
    containment_log_sub_dirs_list: Optional[Sequence[str]],
    containment_log_file_name: Optional[str],
    include_edges_in_log: bool,
    kernel_type: str,
    return_array_as: str,
) -> OptimizerV2StageRunResult:
    biopsy_transform_bank_prefix = biopsy_transform_bank_prefix_provider(stage_config.num_trials)
    target_transform_bank_prefix = target_transform_bank_prefix_provider(stage_config.num_trials)
    _validate_stage_transform_bank_prefix(
        stage_config=stage_config,
        transform_bank_prefix=biopsy_transform_bank_prefix,
        provider_name="biopsy_transform_bank_prefix_provider",
    )
    _validate_stage_transform_bank_prefix(
        stage_config=stage_config,
        transform_bank_prefix=target_transform_bank_prefix,
        provider_name="target_transform_bank_prefix_provider",
    )
    target_relative_structures_nominal_plus_trials = target_relative_structures_nominal_plus_trials_provider(
        stage_config.num_trials
    )

    chunk_score_results = []
    stage_tested_candidate_frames = []
    for chunk_candidate_indices_global in _yield_candidate_index_chunks(candidate_indices_global, max_candidates_per_chunk):
        chunk_layout = OptimizerV2ChunkLayout(
            candidate_indices_global=tuple(int(candidate_index) for candidate_index in chunk_candidate_indices_global),
            num_trials=stage_config.num_trials,
            include_nominal=include_nominal,
            nominal_relative_structure_index=nominal_relative_structure_index,
            trial_relative_structure_start_index=trial_relative_structure_start_index,
        )
        chunk_score_result = score_target_candidate_chunk(
            candidate_pool=candidate_pool,
            chunk_layout=chunk_layout,
            nominal_biopsy_points=nominal_biopsy_points,
            nominal_biopsy_centroid=nominal_biopsy_centroid,
            nominal_biopsy_centroid_line=nominal_biopsy_centroid_line,
            biopsy_transform_bank_prefix=biopsy_transform_bank_prefix,
            target_relative_structures_nominal_plus_trials=target_relative_structures_nominal_plus_trials,
            target_structure_centroid=target_structure_centroid,
            target_transform_bank_prefix=target_transform_bank_prefix,
            objective_reducer_name=objective_reducer_name,
            max_test_structures_per_call=max_test_structures_per_call,
            create_tested_candidate_dataframe=True,
            containment_log_sub_dirs_list=_resolve_stage_containment_log_sub_dirs(
                containment_log_sub_dirs_list,
                stage_config.stage_name,
            ),
            containment_log_file_name=containment_log_file_name,
            include_edges_in_log=include_edges_in_log,
            kernel_type=kernel_type,
            return_array_as=return_array_as,
        )
        chunk_score_results.append(chunk_score_result)
        stage_tested_candidate_frames.append(
            _annotate_stage_tested_candidate_dataframe(
                chunk_score_result.tested_candidate_dataframe,
                stage_config=stage_config,
                stage_input_candidate_count=candidate_indices_global.size,
                biopsy_transform_bank_prefix=biopsy_transform_bank_prefix,
                target_transform_bank_prefix=target_transform_bank_prefix,
            )
        )

    stage_tested_candidate_dataframe = pandas.concat(stage_tested_candidate_frames, ignore_index=True)
    stage_ranked_candidate_dataframe, survivor_candidate_indices_global = _build_stage_ranked_candidate_dataframe(
        stage_tested_candidate_dataframe,
        stage_config,
    )
    return OptimizerV2StageRunResult(
        stage_name=stage_config.stage_name,
        num_trials=stage_config.num_trials,
        input_candidate_indices_global=np.asarray(candidate_indices_global, dtype=np.int32),
        survivor_candidate_indices_global=survivor_candidate_indices_global,
        chunk_score_results=tuple(chunk_score_results),
        tested_candidate_dataframe=stage_tested_candidate_dataframe,
        ranked_candidate_dataframe=stage_ranked_candidate_dataframe,
    )


def _annotate_stage_tested_candidate_dataframe(
    tested_candidate_dataframe,
    stage_config: OptimizerV2StageConfig,
    stage_input_candidate_count: int,
    biopsy_transform_bank_prefix: SharedTransformBankPrefix,
    target_transform_bank_prefix: SharedTransformBankPrefix,
):
    annotated_dataframe = tested_candidate_dataframe.copy()
    annotated_dataframe["Stage name"] = stage_config.stage_name
    annotated_dataframe["Stage input candidate count"] = np.int32(stage_input_candidate_count)
    annotated_dataframe["Biopsy transform bank prefix size used"] = np.int32(
        biopsy_transform_bank_prefix.requested_num_trials
    )
    annotated_dataframe["Target transform bank prefix size used"] = np.int32(
        target_transform_bank_prefix.requested_num_trials
    )
    annotated_dataframe["Available shared biopsy transform samples"] = np.int32(
        biopsy_transform_bank_prefix.available_num_trials
    )
    annotated_dataframe["Available shared target transform samples"] = np.int32(
        target_transform_bank_prefix.available_num_trials
    )
    annotated_dataframe["Tie-break resolution method"] = DEFAULT_STAGE_PROVISIONAL_TIE_BREAK_METHOD
    annotated_dataframe["Tie-break warning flag"] = False
    annotated_dataframe["Tie-break fallback flag"] = False
    annotated_dataframe["Winning-candidate downstream-comparable target score"] = np.nan
    annotated_dataframe["Downstream-comparable score trial count"] = np.nan
    return annotated_dataframe


def _build_stage_ranked_candidate_dataframe(
    stage_tested_candidate_dataframe,
    stage_config: OptimizerV2StageConfig,
):
    ranked_candidate_dataframe = stage_tested_candidate_dataframe.sort_values(
        by=["Objective value", "Distance to target centroid mm", "Candidate global index"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked_candidate_dataframe["Candidate rank"] = np.arange(1, len(ranked_candidate_dataframe) + 1, dtype=np.int32)

    survivor_count = stage_config.resolve_survivor_count(len(ranked_candidate_dataframe))
    ranked_candidate_dataframe["Stage output survivor count"] = np.int32(survivor_count)
    ranked_candidate_dataframe["Is survivor"] = False
    if survivor_count > 0:
        ranked_candidate_dataframe.loc[: survivor_count - 1, "Is survivor"] = True

    survivor_candidate_indices_global = ranked_candidate_dataframe.loc[
        ranked_candidate_dataframe["Is survivor"],
        "Candidate global index",
    ].to_numpy(dtype=np.int32)
    return ranked_candidate_dataframe, survivor_candidate_indices_global


def _resolve_initial_candidate_indices_global(
    candidate_pool: OptimizerV2CandidatePool,
    initial_candidate_indices_global: Optional[Sequence[int]],
) -> np.ndarray:
    if initial_candidate_indices_global is None:
        return np.arange(candidate_pool.candidate_points.shape[0], dtype=np.int32)
    normalized_candidate_indices_global = np.asarray(initial_candidate_indices_global, dtype=np.int32)
    if normalized_candidate_indices_global.ndim != 1:
        raise ValueError("initial_candidate_indices_global must be one-dimensional")
    return normalized_candidate_indices_global


def _resolve_max_candidates_per_chunk(num_candidates: int, max_candidates_per_chunk: Optional[int]) -> int:
    if max_candidates_per_chunk is None:
        return max(1, int(num_candidates))
    if max_candidates_per_chunk <= 0:
        raise ValueError("max_candidates_per_chunk must be positive when provided")
    return int(max_candidates_per_chunk)


def _yield_candidate_index_chunks(candidate_indices_global: np.ndarray, max_candidates_per_chunk: int):
    for start_index in range(0, candidate_indices_global.size, max_candidates_per_chunk):
        yield candidate_indices_global[start_index : start_index + max_candidates_per_chunk]


def _validate_stage_transform_bank_prefix(
    stage_config: OptimizerV2StageConfig,
    transform_bank_prefix: SharedTransformBankPrefix,
    provider_name: str,
) -> None:
    if transform_bank_prefix.requested_num_trials != stage_config.num_trials:
        raise ValueError(
            "{} returned requested_num_trials {}, expected {} for {}".format(
                provider_name,
                transform_bank_prefix.requested_num_trials,
                stage_config.num_trials,
                stage_config.stage_name,
            )
        )


def _resolve_stage_containment_log_sub_dirs(
    containment_log_sub_dirs_list: Optional[Sequence[str]],
    stage_name: str,
):
    if containment_log_sub_dirs_list is None:
        return None
    return [*containment_log_sub_dirs_list, stage_name]


def _build_winner_validation_result(
    candidate_pool: OptimizerV2CandidatePool,
    final_ranked_candidate_dataframe,
    final_stage_result: OptimizerV2StageRunResult,
    nominal_biopsy_points: np.ndarray,
    nominal_biopsy_centroid: np.ndarray,
    nominal_biopsy_centroid_line: np.ndarray,
    biopsy_transform_bank_prefix_provider: Callable[[int], SharedTransformBankPrefix],
    target_relative_structures_nominal_plus_trials_provider: Callable[[int], Sequence[Sequence[np.ndarray]]],
    target_structure_centroid: np.ndarray,
    target_transform_bank_prefix_provider: Callable[[int], SharedTransformBankPrefix],
    objective_reducer_name: str,
    include_nominal: bool,
    nominal_relative_structure_index: int,
    trial_relative_structure_start_index: int,
    downstream_comparable_trial_count: Optional[int],
    max_test_structures_per_call: Optional[int],
    containment_log_sub_dirs_list: Optional[Sequence[str]],
    containment_log_file_name: Optional[str],
    include_edges_in_log: bool,
    kernel_type: str,
    return_array_as: str,
) -> Optional[OptimizerV2WinnerValidationResult]:
    if final_ranked_candidate_dataframe.empty or downstream_comparable_trial_count is None:
        return None
    if downstream_comparable_trial_count <= 0:
        raise ValueError("downstream_comparable_trial_count must be positive when provided")

    winner_row = final_ranked_candidate_dataframe.iloc[0]
    winner_candidate_index_global = int(winner_row["Candidate global index"])
    optimizer_selection_score = float(winner_row["Objective value"])
    optimizer_selection_trial_count = int(final_stage_result.num_trials)
    if downstream_comparable_trial_count == optimizer_selection_trial_count:
        return OptimizerV2WinnerValidationResult(
            candidate_index_global=winner_candidate_index_global,
            objective_reducer_name=objective_reducer_name,
            optimizer_selection_score=optimizer_selection_score,
            optimizer_selection_trial_count=optimizer_selection_trial_count,
            downstream_comparable_target_score=optimizer_selection_score,
            downstream_comparable_trial_count=optimizer_selection_trial_count,
            downstream_comparable_nominal_target_score=float(winner_row["Nominal objective value"]),
            used_additional_rescore=False,
            chunk_score_result=None,
        )

    biopsy_transform_bank_prefix = biopsy_transform_bank_prefix_provider(downstream_comparable_trial_count)
    target_transform_bank_prefix = target_transform_bank_prefix_provider(downstream_comparable_trial_count)
    target_relative_structures_nominal_plus_trials = target_relative_structures_nominal_plus_trials_provider(
        downstream_comparable_trial_count
    )
    winner_chunk_layout = OptimizerV2ChunkLayout(
        candidate_indices_global=(winner_candidate_index_global,),
        num_trials=downstream_comparable_trial_count,
        include_nominal=include_nominal,
        nominal_relative_structure_index=nominal_relative_structure_index,
        trial_relative_structure_start_index=trial_relative_structure_start_index,
    )
    winner_chunk_score_result = score_target_candidate_chunk(
        candidate_pool=candidate_pool,
        chunk_layout=winner_chunk_layout,
        nominal_biopsy_points=nominal_biopsy_points,
        nominal_biopsy_centroid=nominal_biopsy_centroid,
        nominal_biopsy_centroid_line=nominal_biopsy_centroid_line,
        biopsy_transform_bank_prefix=biopsy_transform_bank_prefix,
        target_relative_structures_nominal_plus_trials=target_relative_structures_nominal_plus_trials,
        target_structure_centroid=target_structure_centroid,
        target_transform_bank_prefix=target_transform_bank_prefix,
        objective_reducer_name=objective_reducer_name,
        max_test_structures_per_call=max_test_structures_per_call,
        create_tested_candidate_dataframe=True,
        containment_log_sub_dirs_list=_resolve_stage_containment_log_sub_dirs(
            containment_log_sub_dirs_list,
            "winner_rescore",
        ),
        containment_log_file_name=containment_log_file_name,
        include_edges_in_log=include_edges_in_log,
        kernel_type=kernel_type,
        return_array_as=return_array_as,
    )
    return OptimizerV2WinnerValidationResult(
        candidate_index_global=winner_candidate_index_global,
        objective_reducer_name=objective_reducer_name,
        optimizer_selection_score=optimizer_selection_score,
        optimizer_selection_trial_count=optimizer_selection_trial_count,
        downstream_comparable_target_score=float(winner_chunk_score_result.candidate_scores[0]),
        downstream_comparable_trial_count=downstream_comparable_trial_count,
        downstream_comparable_nominal_target_score=float(winner_chunk_score_result.candidate_nominal_scores[0]),
        used_additional_rescore=True,
        chunk_score_result=winner_chunk_score_result,
    )


def _apply_winner_validation_to_candidate_dataframe(
    candidate_dataframe,
    stage_name: str,
    winner_validation_result: OptimizerV2WinnerValidationResult,
):
    updated_candidate_dataframe = candidate_dataframe.copy()
    winner_mask = updated_candidate_dataframe["Candidate global index"] == winner_validation_result.candidate_index_global
    if "Stage name" in updated_candidate_dataframe.columns:
        winner_mask &= updated_candidate_dataframe["Stage name"] == stage_name
    updated_candidate_dataframe.loc[
        winner_mask,
        "Winning-candidate downstream-comparable target score",
    ] = winner_validation_result.downstream_comparable_target_score
    updated_candidate_dataframe.loc[
        winner_mask,
        "Downstream-comparable score trial count",
    ] = np.int32(winner_validation_result.downstream_comparable_trial_count)
    return updated_candidate_dataframe


__all__ = [
    "DEFAULT_STAGE_PROVISIONAL_TIE_BREAK_METHOD",
    "run_target_staged_candidate_search",
]