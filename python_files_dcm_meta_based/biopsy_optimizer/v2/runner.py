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
    OptimizerV2WinnerResolutionResult,
    OptimizerV2WinnerValidationResult,
)
from biopsy_optimizer.v2.scoring import (
    DEFAULT_CONTAINMENT_KERNEL_TYPE,
    score_target_candidate_chunk,
)
from preprocessing.transform_bank import SharedTransformBankPrefix


DEFAULT_STAGE_PROVISIONAL_TIE_BREAK_METHOD = "score_desc_distance_candidate_index__provisional"
DEFAULT_STAGE_STATISTICAL_PRUNE_STD_DEV_THRESHOLD = 1.0
STAGE_PRUNE_METHOD_LEGACY_SURVIVOR_CUTOFF = "legacy_survivor_cutoff"
STAGE_PRUNE_METHOD_PAIRED_MEAN_PD_LEADER_1SIGMA = "paired_mean_pd_leader_1sigma"
STAGE_PRUNE_REASON_PAIRED_MEAN_PD_DOMINATED = "paired_mean_pd_dominance_1sigma"
STAGE_PRUNE_REASON_STATISTICALLY_COMPETITIVE = "statistically_competitive"
FINAL_WINNER_METHOD_SCORE_UNIQUE = "score_unique_stage_c"
FINAL_WINNER_METHOD_SCORE_RESCORE = "score_rescore_prefix_unique"
FINAL_WINNER_METHOD_NEAREST_TARGET_CENTROID_FALLBACK = "nearest_target_centroid_fallback"


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
    winner_resolution_result = _resolve_final_winner(
        candidate_pool=candidate_pool,
        search_config=search_config,
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
        max_test_structures_per_call=max_test_structures_per_call,
        containment_log_sub_dirs_list=containment_log_sub_dirs_list,
        containment_log_file_name=containment_log_file_name,
        include_edges_in_log=include_edges_in_log,
        kernel_type=kernel_type,
        return_array_as=return_array_as,
    )
    if winner_resolution_result is not None:
        final_ranked_candidate_dataframe = _move_operational_winner_to_top(
            final_ranked_candidate_dataframe,
            winner_resolution_result.candidate_index_global,
        )
        final_ranked_candidate_dataframe = _apply_winner_resolution_to_candidate_dataframe(
            final_ranked_candidate_dataframe,
            stage_name=None,
            winner_resolution_result=winner_resolution_result,
        )
        stage_results[-1].ranked_candidate_dataframe = final_ranked_candidate_dataframe.copy()
        stage_results[-1].tested_candidate_dataframe = _apply_winner_resolution_to_candidate_dataframe(
            stage_results[-1].tested_candidate_dataframe,
            stage_name=stage_results[-1].stage_name,
            winner_resolution_result=winner_resolution_result,
        )
        combined_tested_candidate_dataframe = _apply_winner_resolution_to_candidate_dataframe(
            combined_tested_candidate_dataframe,
            stage_name=stage_results[-1].stage_name,
            winner_resolution_result=winner_resolution_result,
        )
    operational_winner_candidate_index_global = None
    if winner_resolution_result is not None:
        operational_winner_candidate_index_global = winner_resolution_result.candidate_index_global
    elif not final_ranked_candidate_dataframe.empty:
        operational_winner_candidate_index_global = int(final_ranked_candidate_dataframe.iloc[0]["Candidate global index"])

    winner_validation_result = _build_winner_validation_result(
        final_ranked_candidate_dataframe=final_ranked_candidate_dataframe,
        winner_resolution_result=winner_resolution_result,
        nominal_biopsy_points=nominal_biopsy_points,
        nominal_biopsy_centroid=nominal_biopsy_centroid,
        nominal_biopsy_centroid_line=nominal_biopsy_centroid_line,
        candidate_pool=candidate_pool,
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
        winner_resolution_result=winner_resolution_result,
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
        chunk_score_results=chunk_score_results,
        objective_reducer_name=objective_reducer_name,
    )
    return OptimizerV2StageRunResult(
        stage_name=stage_config.stage_name,
        num_trials=stage_config.num_trials,
        input_candidate_indices_global=np.asarray(candidate_indices_global, dtype=np.int32),
        survivor_candidate_indices_global=survivor_candidate_indices_global,
        chunk_score_results=tuple(chunk_score_results),
        tested_candidate_dataframe=stage_ranked_candidate_dataframe.copy(),
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
    chunk_score_results: Sequence[OptimizerV2ChunkScoreResult],
    objective_reducer_name: str,
):
    ranked_candidate_dataframe = stage_tested_candidate_dataframe.sort_values(
        by=["Objective value", "Distance to target centroid mm", "Candidate global index"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked_candidate_dataframe["Candidate rank"] = np.arange(1, len(ranked_candidate_dataframe) + 1, dtype=np.int32)

    configured_survivor_count = stage_config.resolve_survivor_count(len(ranked_candidate_dataframe))
    statistical_prune_result = _resolve_stage_statistical_prune_result(
        ranked_candidate_dataframe=ranked_candidate_dataframe,
        chunk_score_results=chunk_score_results,
        objective_reducer_name=objective_reducer_name,
    )
    if statistical_prune_result is None:
        survivor_mask = np.zeros(len(ranked_candidate_dataframe), dtype=bool)
        if configured_survivor_count > 0:
            survivor_mask[:configured_survivor_count] = True
        prune_method = pandas.Series(
            STAGE_PRUNE_METHOD_LEGACY_SURVIVOR_CUTOFF,
            index=ranked_candidate_dataframe.index,
            dtype=object,
        )
        statistical_leader_candidate_index = pandas.Series(np.nan, index=ranked_candidate_dataframe.index, dtype=float)
        paired_mean_deficit = pandas.Series(np.nan, index=ranked_candidate_dataframe.index, dtype=float)
        paired_standard_error = pandas.Series(np.nan, index=ranked_candidate_dataframe.index, dtype=float)
        paired_z_score = pandas.Series(np.nan, index=ranked_candidate_dataframe.index, dtype=float)
        dominance_prune_flag = np.zeros(len(ranked_candidate_dataframe), dtype=bool)
        prune_reason = pandas.Series("survived_stage", index=ranked_candidate_dataframe.index, dtype=object)
        prune_reason.loc[~survivor_mask] = "stage_survivor_cutoff"
    else:
        survivor_mask = statistical_prune_result["survivor_mask"]
        prune_method = pandas.Series(
            STAGE_PRUNE_METHOD_PAIRED_MEAN_PD_LEADER_1SIGMA,
            index=ranked_candidate_dataframe.index,
            dtype=object,
        )
        statistical_leader_candidate_index = pandas.Series(
            np.int32(statistical_prune_result["leader_candidate_index_global"]),
            index=ranked_candidate_dataframe.index,
        )
        paired_mean_deficit = pandas.Series(statistical_prune_result["paired_mean_deficit"], index=ranked_candidate_dataframe.index)
        paired_standard_error = pandas.Series(
            statistical_prune_result["paired_standard_error"],
            index=ranked_candidate_dataframe.index,
        )
        paired_z_score = pandas.Series(statistical_prune_result["paired_z_score"], index=ranked_candidate_dataframe.index)
        dominance_prune_flag = statistical_prune_result["dominance_prune_flag"]
        prune_reason = pandas.Series(
            STAGE_PRUNE_REASON_STATISTICALLY_COMPETITIVE,
            index=ranked_candidate_dataframe.index,
            dtype=object,
        )
        prune_reason.loc[dominance_prune_flag] = STAGE_PRUNE_REASON_PAIRED_MEAN_PD_DOMINATED

    survivor_count = int(np.count_nonzero(survivor_mask))
    ranked_candidate_dataframe["Stage configured survivor count target"] = np.int32(configured_survivor_count)
    ranked_candidate_dataframe["Stage output survivor count"] = np.int32(survivor_count)
    ranked_candidate_dataframe["Stage prune method"] = prune_method
    ranked_candidate_dataframe["Stage statistical leader candidate global index"] = statistical_leader_candidate_index
    ranked_candidate_dataframe["Stage paired mean deficit vs leader"] = paired_mean_deficit
    ranked_candidate_dataframe["Stage paired standard error vs leader"] = paired_standard_error
    ranked_candidate_dataframe["Stage paired z score vs leader"] = paired_z_score
    ranked_candidate_dataframe["Stage statistical dominance prune flag"] = dominance_prune_flag.astype(bool)
    ranked_candidate_dataframe["Is survivor"] = survivor_mask.astype(bool)
    ranked_candidate_dataframe["Stage active candidate count before prune"] = ranked_candidate_dataframe.get(
        "Stage input candidate count",
        np.int32(len(ranked_candidate_dataframe)),
    )
    ranked_candidate_dataframe["Stage active candidate count after prune"] = np.int32(survivor_count)
    ranked_candidate_dataframe["Stage prune flag"] = (~ranked_candidate_dataframe["Is survivor"]).astype(bool)
    prune_stage_name = pandas.Series(np.nan, index=ranked_candidate_dataframe.index, dtype=object)
    prune_stage_name.loc[ranked_candidate_dataframe["Stage prune flag"]] = str(stage_config.stage_name)
    ranked_candidate_dataframe["Pruned at stage"] = prune_stage_name
    ranked_candidate_dataframe["Stage prune reason"] = prune_reason

    survivor_candidate_indices_global = ranked_candidate_dataframe.loc[
        ranked_candidate_dataframe["Is survivor"],
        "Candidate global index",
    ].to_numpy(dtype=np.int32)
    return ranked_candidate_dataframe, survivor_candidate_indices_global


def _resolve_stage_statistical_prune_result(
    ranked_candidate_dataframe,
    chunk_score_results: Sequence[OptimizerV2ChunkScoreResult],
    objective_reducer_name: str,
):
    if objective_reducer_name != "mean_pd":
        return None
    if len(ranked_candidate_dataframe) == 0:
        return None

    ranked_trial_score_matrix = _build_ranked_candidate_trial_score_matrix(
        ranked_candidate_dataframe,
        chunk_score_results,
    )
    if ranked_trial_score_matrix is None:
        return None
    if ranked_trial_score_matrix.shape[1] < 2:
        return None

    leader_trial_scores = ranked_trial_score_matrix[0]
    paired_trial_deficits = leader_trial_scores.reshape(1, -1) - ranked_trial_score_matrix
    paired_mean_deficit = paired_trial_deficits.mean(axis=1).astype(np.float32)
    paired_standard_error = (
        paired_trial_deficits.std(axis=1, ddof=1).astype(np.float32) / np.float32(np.sqrt(ranked_trial_score_matrix.shape[1]))
    )
    paired_standard_error = np.nan_to_num(paired_standard_error, nan=0.0, posinf=np.inf, neginf=np.inf)

    paired_z_score = np.zeros(len(ranked_candidate_dataframe), dtype=np.float32)
    nonzero_standard_error_mask = paired_standard_error > 0.0
    paired_z_score[nonzero_standard_error_mask] = (
        paired_mean_deficit[nonzero_standard_error_mask] / paired_standard_error[nonzero_standard_error_mask]
    )
    zero_standard_error_positive_deficit_mask = (~nonzero_standard_error_mask) & (paired_mean_deficit > 0.0)
    paired_z_score[zero_standard_error_positive_deficit_mask] = np.inf

    dominance_prune_flag = paired_mean_deficit > (
        np.float32(DEFAULT_STAGE_STATISTICAL_PRUNE_STD_DEV_THRESHOLD) * paired_standard_error
    )
    dominance_prune_flag[0] = False

    return {
        "leader_candidate_index_global": int(ranked_candidate_dataframe.iloc[0]["Candidate global index"]),
        "survivor_mask": (~dominance_prune_flag).astype(bool),
        "paired_mean_deficit": paired_mean_deficit,
        "paired_standard_error": paired_standard_error,
        "paired_z_score": paired_z_score,
        "dominance_prune_flag": dominance_prune_flag.astype(bool),
    }


def _build_ranked_candidate_trial_score_matrix(
    ranked_candidate_dataframe,
    chunk_score_results: Sequence[OptimizerV2ChunkScoreResult],
):
    candidate_trial_scores_by_candidate_index = {}
    num_trials = None

    for chunk_score_result in chunk_score_results:
        candidate_trial_mean_point_scores = chunk_score_result.candidate_trial_mean_point_scores
        if candidate_trial_mean_point_scores is None:
            return None

        candidate_trial_mean_point_scores = np.asarray(candidate_trial_mean_point_scores, dtype=np.float32)
        if candidate_trial_mean_point_scores.ndim != 2:
            return None
        if candidate_trial_mean_point_scores.shape[0] != len(chunk_score_result.candidate_indices_global):
            return None
        if candidate_trial_mean_point_scores.shape[1] != chunk_score_result.chunk_layout.num_trials:
            return None
        if not np.all(np.isfinite(candidate_trial_mean_point_scores)):
            return None

        if num_trials is None:
            num_trials = candidate_trial_mean_point_scores.shape[1]
        elif candidate_trial_mean_point_scores.shape[1] != num_trials:
            return None

        for candidate_index_global, candidate_trial_scores in zip(
            chunk_score_result.candidate_indices_global,
            candidate_trial_mean_point_scores,
        ):
            candidate_trial_scores_by_candidate_index[int(candidate_index_global)] = candidate_trial_scores

    ranked_candidate_indices_global = ranked_candidate_dataframe["Candidate global index"].to_numpy(dtype=np.int32)
    if any(int(candidate_index_global) not in candidate_trial_scores_by_candidate_index for candidate_index_global in ranked_candidate_indices_global):
        return None

    return np.vstack(
        [
            candidate_trial_scores_by_candidate_index[int(candidate_index_global)]
            for candidate_index_global in ranked_candidate_indices_global
        ]
    ).astype(np.float32)


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
    final_ranked_candidate_dataframe,
    winner_resolution_result: Optional[OptimizerV2WinnerResolutionResult],
    nominal_biopsy_points: np.ndarray,
    nominal_biopsy_centroid: np.ndarray,
    nominal_biopsy_centroid_line: np.ndarray,
    candidate_pool: OptimizerV2CandidatePool,
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
    if winner_resolution_result is None or final_ranked_candidate_dataframe.empty or downstream_comparable_trial_count is None:
        return None
    if downstream_comparable_trial_count <= 0:
        raise ValueError("downstream_comparable_trial_count must be positive when provided")

    winner_candidate_index_global = int(winner_resolution_result.candidate_index_global)
    optimizer_selection_score = float(winner_resolution_result.resolved_objective_value)
    optimizer_selection_trial_count = int(winner_resolution_result.final_resolution_trial_count)
    if downstream_comparable_trial_count == optimizer_selection_trial_count:
        return OptimizerV2WinnerValidationResult(
            candidate_index_global=winner_candidate_index_global,
            objective_reducer_name=objective_reducer_name,
            optimizer_selection_score=optimizer_selection_score,
            optimizer_selection_trial_count=optimizer_selection_trial_count,
            downstream_comparable_target_score=optimizer_selection_score,
            downstream_comparable_trial_count=optimizer_selection_trial_count,
            downstream_comparable_nominal_target_score=float(winner_resolution_result.resolved_nominal_objective_value),
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


def _resolve_final_winner(
    candidate_pool: OptimizerV2CandidatePool,
    search_config: OptimizerV2SearchConfig,
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
    max_test_structures_per_call: Optional[int],
    containment_log_sub_dirs_list: Optional[Sequence[str]],
    containment_log_file_name: Optional[str],
    include_edges_in_log: bool,
    kernel_type: str,
    return_array_as: str,
) -> Optional[OptimizerV2WinnerResolutionResult]:
    if final_ranked_candidate_dataframe.empty:
        return None

    score_tolerance = search_config.tie_break_config.score_tolerance
    tied_candidate_dataframe = _select_tied_final_candidates(
        final_ranked_candidate_dataframe,
        score_tolerance,
    )
    winner_row = final_ranked_candidate_dataframe.iloc[0]
    if len(tied_candidate_dataframe) == 1:
        return OptimizerV2WinnerResolutionResult(
            candidate_index_global=int(winner_row["Candidate global index"]),
            objective_reducer_name=objective_reducer_name,
            resolution_method=FINAL_WINNER_METHOD_SCORE_UNIQUE,
            tie_warning_flag=False,
            tie_break_fallback_flag=False,
            num_tied_candidates_at_stage_c=1,
            num_additional_rescore_attempts_used=0,
            final_resolution_trial_count=int(final_stage_result.num_trials),
            resolved_objective_value=float(winner_row["Objective value"]),
            resolved_nominal_objective_value=float(winner_row["Nominal objective value"]),
            chunk_score_result=None,
            tied_candidate_dataframe=tied_candidate_dataframe.copy(),
        )

    current_tied_candidate_dataframe = tied_candidate_dataframe.copy()
    current_trial_count = int(final_stage_result.num_trials)
    last_rescore_chunk_score_result = None
    num_additional_rescore_attempts_used = 0

    for _ in range(search_config.tie_break_config.max_additional_rescore_attempts):
        next_trial_count = int(
            np.ceil(
                current_trial_count * search_config.tie_break_config.rescore_trial_count_multiplier
            )
        )
        rescore_chunk_score_result, rescored_ranked_candidate_dataframe = _score_candidate_subset_for_winner_resolution(
            candidate_pool=candidate_pool,
            candidate_indices_global=current_tied_candidate_dataframe["Candidate global index"].to_numpy(dtype=np.int32),
            trial_count=next_trial_count,
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
            max_test_structures_per_call=max_test_structures_per_call,
            containment_log_sub_dirs_list=_resolve_stage_containment_log_sub_dirs(
                containment_log_sub_dirs_list,
                "winner_tie_break",
            ),
            containment_log_file_name=containment_log_file_name,
            include_edges_in_log=include_edges_in_log,
            kernel_type=kernel_type,
            return_array_as=return_array_as,
        )
        last_rescore_chunk_score_result = rescore_chunk_score_result
        num_additional_rescore_attempts_used += 1
        current_trial_count = next_trial_count
        current_tied_candidate_dataframe = _select_tied_final_candidates(
            rescored_ranked_candidate_dataframe,
            score_tolerance,
        )

        if len(current_tied_candidate_dataframe) == 1:
            winner_row = rescored_ranked_candidate_dataframe.iloc[0]
            return OptimizerV2WinnerResolutionResult(
                candidate_index_global=int(winner_row["Candidate global index"]),
                objective_reducer_name=objective_reducer_name,
                resolution_method=FINAL_WINNER_METHOD_SCORE_RESCORE,
                tie_warning_flag=True,
                tie_break_fallback_flag=False,
                num_tied_candidates_at_stage_c=len(tied_candidate_dataframe),
                num_additional_rescore_attempts_used=num_additional_rescore_attempts_used,
                final_resolution_trial_count=current_trial_count,
                resolved_objective_value=float(winner_row["Objective value"]),
                resolved_nominal_objective_value=float(winner_row["Nominal objective value"]),
                chunk_score_result=rescore_chunk_score_result,
                tied_candidate_dataframe=rescored_ranked_candidate_dataframe.copy(),
            )

    fallback_ranked_candidate_dataframe = current_tied_candidate_dataframe.sort_values(
        by=["Distance to target centroid mm", "Candidate global index"],
        ascending=[True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    winner_row = fallback_ranked_candidate_dataframe.iloc[0]
    return OptimizerV2WinnerResolutionResult(
        candidate_index_global=int(winner_row["Candidate global index"]),
        objective_reducer_name=objective_reducer_name,
        resolution_method=FINAL_WINNER_METHOD_NEAREST_TARGET_CENTROID_FALLBACK,
        tie_warning_flag=True,
        tie_break_fallback_flag=True,
        num_tied_candidates_at_stage_c=len(tied_candidate_dataframe),
        num_additional_rescore_attempts_used=num_additional_rescore_attempts_used,
        final_resolution_trial_count=current_trial_count,
        resolved_objective_value=float(winner_row["Objective value"]),
        resolved_nominal_objective_value=float(winner_row["Nominal objective value"]),
        chunk_score_result=last_rescore_chunk_score_result,
        tied_candidate_dataframe=fallback_ranked_candidate_dataframe.copy(),
    )


def _score_candidate_subset_for_winner_resolution(
    candidate_pool: OptimizerV2CandidatePool,
    candidate_indices_global: np.ndarray,
    trial_count: int,
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
    max_test_structures_per_call: Optional[int],
    containment_log_sub_dirs_list: Optional[Sequence[str]],
    containment_log_file_name: Optional[str],
    include_edges_in_log: bool,
    kernel_type: str,
    return_array_as: str,
):
    chunk_layout = OptimizerV2ChunkLayout(
        candidate_indices_global=tuple(int(candidate_index) for candidate_index in candidate_indices_global.tolist()),
        num_trials=trial_count,
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
        biopsy_transform_bank_prefix=biopsy_transform_bank_prefix_provider(trial_count),
        target_relative_structures_nominal_plus_trials=target_relative_structures_nominal_plus_trials_provider(trial_count),
        target_structure_centroid=target_structure_centroid,
        target_transform_bank_prefix=target_transform_bank_prefix_provider(trial_count),
        objective_reducer_name=objective_reducer_name,
        max_test_structures_per_call=max_test_structures_per_call,
        create_tested_candidate_dataframe=True,
        containment_log_sub_dirs_list=containment_log_sub_dirs_list,
        containment_log_file_name=containment_log_file_name,
        include_edges_in_log=include_edges_in_log,
        kernel_type=kernel_type,
        return_array_as=return_array_as,
    )
    ranked_candidate_dataframe = chunk_score_result.tested_candidate_dataframe.sort_values(
        by=["Objective value", "Distance to target centroid mm", "Candidate global index"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return chunk_score_result, ranked_candidate_dataframe


def _select_tied_final_candidates(candidate_dataframe, score_tolerance: float):
    best_objective_value = float(candidate_dataframe.iloc[0]["Objective value"])
    tied_mask = np.abs(candidate_dataframe["Objective value"].to_numpy(dtype=float) - best_objective_value) <= score_tolerance
    return candidate_dataframe.loc[tied_mask].copy().reset_index(drop=True)


def _move_operational_winner_to_top(candidate_dataframe, winner_candidate_index_global: int):
    winner_mask = candidate_dataframe["Candidate global index"] == winner_candidate_index_global
    if winner_mask.sum() != 1:
        return candidate_dataframe.copy()

    updated_candidate_dataframe = pandas.concat(
        [candidate_dataframe.loc[winner_mask], candidate_dataframe.loc[~winner_mask]],
        ignore_index=True,
    )
    if "Candidate rank" in updated_candidate_dataframe.columns:
        updated_candidate_dataframe["Candidate rank"] = np.arange(
            1,
            len(updated_candidate_dataframe) + 1,
            dtype=np.int32,
        )
    return updated_candidate_dataframe


def _apply_winner_resolution_to_candidate_dataframe(
    candidate_dataframe,
    stage_name: Optional[str],
    winner_resolution_result: OptimizerV2WinnerResolutionResult,
):
    updated_candidate_dataframe = candidate_dataframe.copy()
    resolution_mask = np.ones(len(updated_candidate_dataframe), dtype=bool)
    if stage_name is not None and "Stage name" in updated_candidate_dataframe.columns:
        resolution_mask &= updated_candidate_dataframe["Stage name"] == stage_name

    updated_candidate_dataframe.loc[resolution_mask, "Tie-break resolution method"] = winner_resolution_result.resolution_method
    updated_candidate_dataframe.loc[resolution_mask, "Winner determination method"] = winner_resolution_result.resolution_method
    updated_candidate_dataframe.loc[resolution_mask, "Tie-break warning flag"] = winner_resolution_result.tie_warning_flag
    updated_candidate_dataframe.loc[resolution_mask, "Tie-break fallback flag"] = winner_resolution_result.tie_break_fallback_flag
    updated_candidate_dataframe.loc[
        resolution_mask,
        "Final winner additional rescore attempts used",
    ] = np.int32(winner_resolution_result.num_additional_rescore_attempts_used)
    updated_candidate_dataframe.loc[
        resolution_mask,
        "Final winner resolution trial count",
    ] = np.int32(winner_resolution_result.final_resolution_trial_count)
    updated_candidate_dataframe.loc[
        resolution_mask,
        "Final winner tie candidate count",
    ] = np.int32(winner_resolution_result.num_tied_candidates_at_stage_c)
    updated_candidate_dataframe.loc[
        resolution_mask,
        "Final winner resolved objective value",
    ] = winner_resolution_result.resolved_objective_value
    updated_candidate_dataframe.loc[
        resolution_mask,
        "Final winner resolved nominal objective value",
    ] = winner_resolution_result.resolved_nominal_objective_value
    updated_candidate_dataframe.loc[
        resolution_mask,
        "Operational winner candidate index",
    ] = np.int32(winner_resolution_result.candidate_index_global)

    operational_winner_mask = updated_candidate_dataframe["Candidate global index"] == winner_resolution_result.candidate_index_global
    if stage_name is not None and "Stage name" in updated_candidate_dataframe.columns:
        operational_winner_mask &= updated_candidate_dataframe["Stage name"] == stage_name
    updated_candidate_dataframe["Is operational winner"] = False
    updated_candidate_dataframe.loc[operational_winner_mask, "Is operational winner"] = True
    return updated_candidate_dataframe


__all__ = [
    "DEFAULT_STAGE_PROVISIONAL_TIE_BREAK_METHOD",
    "FINAL_WINNER_METHOD_NEAREST_TARGET_CENTROID_FALLBACK",
    "FINAL_WINNER_METHOD_SCORE_RESCORE",
    "FINAL_WINNER_METHOD_SCORE_UNIQUE",
    "run_target_staged_candidate_search",
]