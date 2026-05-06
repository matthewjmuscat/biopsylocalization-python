"""Carry-down dataframe builders for optimizer v2 outputs."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import pandas

from biopsy_optimizer.v2.contracts import OptimizerV2SearchRunResult


DOWNSTREAM_MC_SCORE_MERGE_COLUMN = "__downstream_mc_target_score"
DOWNSTREAM_MC_TRIAL_COUNT_MERGE_COLUMN = "__downstream_mc_trial_count"
BIOPSY_SAMPLE_COUNT_PLANNED_MERGE_COLUMN = "__planned_biopsy_sample_count"
BIOPSY_SAMPLE_COUNT_FINALIZED_MERGE_COLUMN = "__finalized_biopsy_sample_count"
DOWNSTREAM_MC_JOIN_COLUMNS = (
    "Patient ID",
    "Biopsy ROI",
    "Biopsy ref #",
    "Biopsy index",
    "Target structure ID",
    "Target structure type",
    "Target structure index",
)
BIOPSY_SAMPLE_COUNT_JOIN_COLUMNS = (
    "Patient ID",
    "Biopsy ROI",
    "Biopsy ref #",
    "Biopsy index",
)
PLANNED_BIOPSY_SAMPLED_POINT_COUNT_COLUMN = "Target optimizer planned biopsy sampled point count"
FINALIZED_BIOPSY_SAMPLED_POINT_COUNT_COLUMN = "Target optimizer finalized biopsy sampled point count"
BIOPSY_SAMPLED_POINT_COUNT_DELTA_COLUMN = "Target optimizer biopsy sampled point count delta"
BIOPSY_SAMPLED_POINT_COUNT_RATIO_COLUMN = "Target optimizer biopsy sampled point count ratio"
BIOPSY_SAMPLED_POINT_COUNT_MISMATCH_FLAG_COLUMN = "Target optimizer biopsy sampled point count mismatch flag"


def build_target_dil_optimization_summary_dataframe(
    search_result: OptimizerV2SearchRunResult,
    metadata: Optional[Mapping[str, Any]] = None,
) -> pandas.DataFrame:
    """Build one carry-down winner summary row for the target-DIL optimizer lane."""
    if search_result.ranked_candidate_dataframe.empty:
        return _apply_metadata_to_dataframe(
            _initialize_optimizer_output_placeholder_columns(pandas.DataFrame()),
            metadata,
        )

    winner_row = search_result.ranked_candidate_dataframe.iloc[0]
    summary_row = {
        "Target optimizer lane": "target_dil_optimizer_v2",
        "Target optimizer final stage name": _resolve_final_stage_name(search_result),
        "Target optimizer num stages": len(search_result.stage_results),
        "Target optimizer num tested candidate rows": len(search_result.tested_candidate_dataframe),
        "Target optimizer num final ranked candidates": len(search_result.ranked_candidate_dataframe),
        "Target optimizer operational winner candidate index": int(
            search_result.operational_winner_candidate_index_global
        )
        if search_result.operational_winner_candidate_index_global is not None
        else np.nan,
        "Target optimizer selected X": float(winner_row["Candidate X"]),
        "Target optimizer selected Y": float(winner_row["Candidate Y"]),
        "Target optimizer selected Z": float(winner_row["Candidate Z"]),
        "Target optimizer retained score": float(winner_row["Objective value"]),
        "Target optimizer retained nominal score": float(winner_row["Nominal objective value"]),
        "Target optimizer objective reducer name": winner_row["Objective reducer name"],
        "Target optimizer distance to target centroid mm": float(winner_row["Distance to target centroid mm"]),
        "Target optimizer winner determination method": winner_row.get("Winner determination method", np.nan),
        "Target optimizer tie-break warning flag": winner_row.get("Tie-break warning flag", np.nan),
        "Target optimizer centroid fallback flag": winner_row.get("Tie-break fallback flag", np.nan),
        "Target optimizer final resolution trial count": winner_row.get("Final winner resolution trial count", np.nan),
        "Target optimizer selection score trial count": winner_row.get(
            "Final winner resolution trial count",
            np.nan,
        ),
        "Target optimizer additional rescore attempts used": winner_row.get(
            "Final winner additional rescore attempts used",
            np.nan,
        ),
        "Target optimizer final tie candidate count": winner_row.get("Final winner tie candidate count", np.nan),
        "Target optimizer resolved objective value": winner_row.get("Final winner resolved objective value", np.nan),
        "Target optimizer resolved nominal objective value": winner_row.get(
            "Final winner resolved nominal objective value",
            np.nan,
        ),
        "Target optimizer selected winner optimizer-side target score": winner_row.get(
            "Final winner resolved objective value",
            np.nan,
        ),
        "Target optimizer selected winner optimizer-side trial count": winner_row.get(
            "Final winner resolution trial count",
            np.nan,
        ),
        "Target optimizer downstream comparable target score": winner_row.get(
            "Winning-candidate downstream-comparable target score",
            np.nan,
        ),
        "Target optimizer downstream comparable trial count": winner_row.get(
            "Downstream-comparable score trial count",
            np.nan,
        ),
        "Target optimizer selected winner downstream-comparable target score": winner_row.get(
            "Winning-candidate downstream-comparable target score",
            np.nan,
        ),
        "Target optimizer selected winner downstream-comparable trial count": winner_row.get(
            "Downstream-comparable score trial count",
            np.nan,
        ),
        "Target optimizer selected winner downstream MC target score": np.nan,
        "Target optimizer selected winner downstream MC trial count": np.nan,
        "Target optimizer selected winner downstream MC agreement delta": np.nan,
        "Target optimizer downstream comparable score trial count": winner_row.get(
            "Downstream-comparable score trial count",
            np.nan,
        ),
        "Target optimizer selected winner score-surface delta": _resolve_selected_winner_score_surface_delta(
            winner_row
        ),
        "Target optimizer agreement delta": _resolve_selected_winner_score_surface_delta(winner_row),
    }
    summary_dataframe = _initialize_optimizer_output_placeholder_columns(
        pandas.DataFrame([summary_row])
    )
    return _apply_metadata_to_dataframe(summary_dataframe, metadata)


def build_target_dil_ranked_candidate_output_dataframe(
    search_result: OptimizerV2SearchRunResult,
    metadata: Optional[Mapping[str, Any]] = None,
) -> pandas.DataFrame:
    """Build the ranked optimizer-candidate table with carry-down metadata stamped on it."""
    ranked_candidate_dataframe = search_result.ranked_candidate_dataframe.copy()
    if ranked_candidate_dataframe.empty:
        return _apply_metadata_to_dataframe(
            _initialize_optimizer_output_placeholder_columns(ranked_candidate_dataframe),
            metadata,
        )

    ranked_candidate_dataframe["Target optimizer lane"] = "target_dil_optimizer_v2"
    ranked_candidate_dataframe["Target optimizer final stage name"] = _resolve_final_stage_name(search_result)
    ranked_candidate_dataframe["Target optimizer num stages"] = np.int32(len(search_result.stage_results))
    ranked_candidate_dataframe["Target optimizer num tested candidate rows"] = np.int32(
        len(search_result.tested_candidate_dataframe)
    )
    ranked_candidate_dataframe["Target optimizer num final ranked candidates"] = np.int32(
        len(search_result.ranked_candidate_dataframe)
    )
    ranked_candidate_dataframe["Target optimizer selection score trial count"] = ranked_candidate_dataframe.get(
        "Final winner resolution trial count",
        np.nan,
    )
    ranked_candidate_dataframe.rename(
        columns={
            "Objective value": "Target optimizer retained score",
            "Nominal objective value": "Target optimizer retained nominal score",
            "Objective reducer name": "Target optimizer objective reducer name",
            "Distance to target centroid mm": "Target optimizer distance to target centroid mm",
            "Winning-candidate downstream-comparable target score": "Target optimizer downstream comparable target score",
            "Downstream-comparable score trial count": "Target optimizer downstream comparable trial count",
            "Winner determination method": "Target optimizer winner determination method",
        },
        inplace=True,
    )
    ranked_candidate_dataframe[
        "Target optimizer downstream comparable score trial count"
    ] = ranked_candidate_dataframe.get(
        "Target optimizer downstream comparable trial count",
        np.nan,
    )
    ranked_candidate_dataframe[
        "Target optimizer selected winner optimizer-side target score"
    ] = ranked_candidate_dataframe.get(
        "Final winner resolved objective value",
        np.nan,
    )
    ranked_candidate_dataframe[
        "Target optimizer selected winner optimizer-side trial count"
    ] = ranked_candidate_dataframe.get(
        "Final winner resolution trial count",
        np.nan,
    )
    ranked_candidate_dataframe[
        "Target optimizer selected winner downstream-comparable target score"
    ] = ranked_candidate_dataframe.get(
        "Target optimizer downstream comparable target score",
        np.nan,
    )
    ranked_candidate_dataframe[
        "Target optimizer selected winner downstream-comparable trial count"
    ] = ranked_candidate_dataframe.get(
        "Target optimizer downstream comparable trial count",
        np.nan,
    )
    ranked_candidate_dataframe[
        "Target optimizer selected winner downstream MC target score"
    ] = np.nan
    ranked_candidate_dataframe[
        "Target optimizer selected winner downstream MC trial count"
    ] = np.nan
    ranked_candidate_dataframe[
        "Target optimizer selected winner downstream MC agreement delta"
    ] = np.nan
    ranked_candidate_dataframe[
        "Target optimizer selected winner score-surface delta"
    ] = ranked_candidate_dataframe.apply(
        _resolve_selected_winner_score_surface_delta,
        axis=1,
    )
    ranked_candidate_dataframe["Target optimizer agreement delta"] = ranked_candidate_dataframe[
        "Target optimizer selected winner score-surface delta"
    ]
    ranked_candidate_dataframe = _initialize_optimizer_output_placeholder_columns(
        ranked_candidate_dataframe
    )
    return _apply_metadata_to_dataframe(ranked_candidate_dataframe, metadata)


def build_target_dil_tested_candidate_output_dataframe(
    search_result: OptimizerV2SearchRunResult,
    metadata: Optional[Mapping[str, Any]] = None,
) -> pandas.DataFrame:
    """Build the all-stages tested-candidate audit table with carry-down metadata stamped on it."""
    tested_candidate_dataframe = search_result.tested_candidate_dataframe.copy()
    if tested_candidate_dataframe.empty:
        return _apply_metadata_to_dataframe(
            _initialize_optimizer_output_placeholder_columns(tested_candidate_dataframe),
            metadata,
        )

    tested_candidate_dataframe["Target optimizer lane"] = "target_dil_optimizer_v2"
    tested_candidate_dataframe["Target optimizer final stage name"] = _resolve_final_stage_name(search_result)
    tested_candidate_dataframe["Target optimizer num stages"] = np.int32(len(search_result.stage_results))
    tested_candidate_dataframe["Target optimizer num tested candidate rows"] = np.int32(
        len(search_result.tested_candidate_dataframe)
    )
    tested_candidate_dataframe["Target optimizer num final ranked candidates"] = np.int32(
        len(search_result.ranked_candidate_dataframe)
    )
    tested_candidate_dataframe.rename(
        columns={
            "Objective value": "Target optimizer tested score",
            "Nominal objective value": "Target optimizer tested nominal score",
            "Objective reducer name": "Target optimizer objective reducer name",
            "Distance to target centroid mm": "Target optimizer distance to target centroid mm",
            "Stage name": "Target optimizer stage name",
            "Candidate rank": "Target optimizer stage candidate rank",
            "Is survivor": "Target optimizer stage survivor flag",
            "Stage input candidate count": "Target optimizer stage input candidate count",
            "Stage configured survivor count target": "Target optimizer stage configured survivor count target",
            "Stage output survivor count": "Target optimizer stage output survivor count",
            "Stage active candidate count before prune": "Target optimizer stage active candidate count before prune",
            "Stage active candidate count after prune": "Target optimizer stage active candidate count after prune",
            "Stage prune method": "Target optimizer stage prune method",
            "Stage prune flag": "Target optimizer stage prune flag",
            "Stage prune reason": "Target optimizer stage prune reason",
            "Stage statistical leader candidate global index": "Target optimizer stage statistical leader candidate global index",
            "Stage statistical prune std dev threshold": "Target optimizer stage statistical prune std dev threshold",
            "Stage paired mean deficit vs leader": "Target optimizer stage paired mean deficit vs leader",
            "Stage paired standard error vs leader": "Target optimizer stage paired standard error vs leader",
            "Stage paired z score vs leader": "Target optimizer stage paired z score vs leader",
            "Stage statistical dominance prune flag": "Target optimizer stage statistical dominance prune flag",
            "Pruned at stage": "Target optimizer pruned at stage",
            "Winning-candidate downstream-comparable target score": "Target optimizer downstream comparable target score",
            "Downstream-comparable score trial count": "Target optimizer downstream comparable trial count",
            "Winner determination method": "Target optimizer winner determination method",
            "Is operational winner": "Target optimizer operational winner flag",
        },
        inplace=True,
    )
    tested_candidate_dataframe["Target optimizer downstream comparable score trial count"] = tested_candidate_dataframe.get(
        "Target optimizer downstream comparable trial count",
        np.nan,
    )
    tested_candidate_dataframe = _initialize_optimizer_output_placeholder_columns(
        tested_candidate_dataframe
    )
    return _apply_metadata_to_dataframe(tested_candidate_dataframe, metadata)


def annotate_target_dil_optimizer_dataframe_with_biopsy_sampling_audit(
    dataframe: Optional[pandas.DataFrame],
    biopsy_sampling_audit_dataframe: Optional[pandas.DataFrame],
) -> Optional[pandas.DataFrame]:
    if dataframe is None:
        return None

    updated_dataframe = _initialize_optimizer_output_placeholder_columns(dataframe)
    if updated_dataframe.empty:
        return updated_dataframe
    if biopsy_sampling_audit_dataframe is None or biopsy_sampling_audit_dataframe.empty:
        return updated_dataframe
    if any(join_column not in updated_dataframe.columns for join_column in BIOPSY_SAMPLE_COUNT_JOIN_COLUMNS):
        return updated_dataframe

    selected_columns = [
        *BIOPSY_SAMPLE_COUNT_JOIN_COLUMNS,
        PLANNED_BIOPSY_SAMPLED_POINT_COUNT_COLUMN,
        FINALIZED_BIOPSY_SAMPLED_POINT_COUNT_COLUMN,
    ]
    missing_columns = [
        column_name
        for column_name in selected_columns
        if column_name not in biopsy_sampling_audit_dataframe.columns
    ]
    if missing_columns:
        return updated_dataframe

    normalized_dataframe = _normalize_downstream_mc_join_columns(updated_dataframe)
    normalized_audit_dataframe = _normalize_downstream_mc_join_columns(
        biopsy_sampling_audit_dataframe[selected_columns].copy()
    )
    normalized_audit_dataframe.rename(
        columns={
            PLANNED_BIOPSY_SAMPLED_POINT_COUNT_COLUMN: BIOPSY_SAMPLE_COUNT_PLANNED_MERGE_COLUMN,
            FINALIZED_BIOPSY_SAMPLED_POINT_COUNT_COLUMN: BIOPSY_SAMPLE_COUNT_FINALIZED_MERGE_COLUMN,
        },
        inplace=True,
    )
    normalized_audit_dataframe = normalized_audit_dataframe.drop_duplicates(
        subset=list(BIOPSY_SAMPLE_COUNT_JOIN_COLUMNS),
        keep="last",
    )

    merged_dataframe = normalized_dataframe.merge(
        normalized_audit_dataframe,
        how="left",
        on=list(BIOPSY_SAMPLE_COUNT_JOIN_COLUMNS),
        sort=False,
    )
    merged_dataframe[PLANNED_BIOPSY_SAMPLED_POINT_COUNT_COLUMN] = merged_dataframe[
        BIOPSY_SAMPLE_COUNT_PLANNED_MERGE_COLUMN
    ].combine_first(merged_dataframe[PLANNED_BIOPSY_SAMPLED_POINT_COUNT_COLUMN])
    merged_dataframe[FINALIZED_BIOPSY_SAMPLED_POINT_COUNT_COLUMN] = merged_dataframe[
        BIOPSY_SAMPLE_COUNT_FINALIZED_MERGE_COLUMN
    ]

    valid_count_mask = merged_dataframe[PLANNED_BIOPSY_SAMPLED_POINT_COUNT_COLUMN].notna() & merged_dataframe[
        FINALIZED_BIOPSY_SAMPLED_POINT_COUNT_COLUMN
    ].notna()
    point_count_delta = pandas.Series(np.nan, index=merged_dataframe.index, dtype=float)
    point_count_delta.loc[valid_count_mask] = (
        merged_dataframe.loc[valid_count_mask, FINALIZED_BIOPSY_SAMPLED_POINT_COUNT_COLUMN]
        - merged_dataframe.loc[valid_count_mask, PLANNED_BIOPSY_SAMPLED_POINT_COUNT_COLUMN]
    )
    merged_dataframe[BIOPSY_SAMPLED_POINT_COUNT_DELTA_COLUMN] = point_count_delta

    ratio_mask = valid_count_mask & (
        merged_dataframe[PLANNED_BIOPSY_SAMPLED_POINT_COUNT_COLUMN] != 0
    )
    point_count_ratio = pandas.Series(np.nan, index=merged_dataframe.index, dtype=float)
    point_count_ratio.loc[ratio_mask] = (
        merged_dataframe.loc[ratio_mask, FINALIZED_BIOPSY_SAMPLED_POINT_COUNT_COLUMN]
        / merged_dataframe.loc[ratio_mask, PLANNED_BIOPSY_SAMPLED_POINT_COUNT_COLUMN]
    )
    merged_dataframe[BIOPSY_SAMPLED_POINT_COUNT_RATIO_COLUMN] = point_count_ratio

    mismatch_flag = pandas.Series(np.nan, index=merged_dataframe.index, dtype=object)
    mismatch_flag.loc[valid_count_mask] = (
        merged_dataframe.loc[valid_count_mask, FINALIZED_BIOPSY_SAMPLED_POINT_COUNT_COLUMN]
        != merged_dataframe.loc[valid_count_mask, PLANNED_BIOPSY_SAMPLED_POINT_COUNT_COLUMN]
    ).astype(bool)
    merged_dataframe[BIOPSY_SAMPLED_POINT_COUNT_MISMATCH_FLAG_COLUMN] = mismatch_flag

    return merged_dataframe.drop(
        columns=[
            BIOPSY_SAMPLE_COUNT_PLANNED_MERGE_COLUMN,
            BIOPSY_SAMPLE_COUNT_FINALIZED_MERGE_COLUMN,
        ],
        errors="ignore",
    )


def annotate_target_dil_optimizer_dataframe_with_downstream_mc(
    dataframe: Optional[pandas.DataFrame],
    downstream_structure_score_dataframe: Optional[pandas.DataFrame],
    downstream_trial_count: Optional[int],
) -> Optional[pandas.DataFrame]:
    if dataframe is None:
        return None

    updated_dataframe = _initialize_optimizer_output_placeholder_columns(dataframe)
    if updated_dataframe.empty:
        return updated_dataframe
    if downstream_structure_score_dataframe is None or downstream_structure_score_dataframe.empty:
        return updated_dataframe
    if downstream_trial_count is None or int(downstream_trial_count) <= 0:
        return updated_dataframe
    if any(join_column not in updated_dataframe.columns for join_column in DOWNSTREAM_MC_JOIN_COLUMNS):
        return updated_dataframe

    normalized_dataframe = _normalize_downstream_mc_join_columns(updated_dataframe)
    annotation_source_dataframe = _build_downstream_mc_annotation_source_dataframe(
        downstream_structure_score_dataframe,
        int(downstream_trial_count),
    )
    if annotation_source_dataframe.empty:
        return updated_dataframe

    merged_dataframe = normalized_dataframe.merge(
        annotation_source_dataframe,
        how="left",
        on=list(DOWNSTREAM_MC_JOIN_COLUMNS),
        sort=False,
    )
    merged_dataframe[
        "Target optimizer selected winner downstream MC target score"
    ] = merged_dataframe[DOWNSTREAM_MC_SCORE_MERGE_COLUMN]
    merged_dataframe[
        "Target optimizer selected winner downstream MC trial count"
    ] = np.int32(int(downstream_trial_count))

    downstream_comparable_score = merged_dataframe[
        "Target optimizer selected winner downstream-comparable target score"
    ].combine_first(
        merged_dataframe.get("Target optimizer downstream comparable target score")
    )
    merged_dataframe[
        "Target optimizer selected winner downstream MC agreement delta"
    ] = merged_dataframe[
        "Target optimizer selected winner downstream MC target score"
    ] - downstream_comparable_score

    return merged_dataframe.drop(
        columns=[DOWNSTREAM_MC_SCORE_MERGE_COLUMN, DOWNSTREAM_MC_TRIAL_COUNT_MERGE_COLUMN],
        errors="ignore",
    )


def _resolve_final_stage_name(search_result: OptimizerV2SearchRunResult) -> str:
    if not search_result.stage_results:
        return ""
    return str(search_result.stage_results[-1].stage_name)


def _build_downstream_mc_annotation_source_dataframe(
    downstream_structure_score_dataframe: pandas.DataFrame,
    downstream_trial_count: int,
) -> pandas.DataFrame:
    selected_columns = [
        "Patient ID",
        "Bx ID",
        "Bx refnum",
        "Bx index",
        "Relative structure ROI",
        "Relative structure type",
        "Relative structure index",
        "Global mean binom est",
    ]
    missing_columns = [
        column_name
        for column_name in selected_columns
        if column_name not in downstream_structure_score_dataframe.columns
    ]
    if missing_columns:
        return pandas.DataFrame(columns=[*DOWNSTREAM_MC_JOIN_COLUMNS, DOWNSTREAM_MC_SCORE_MERGE_COLUMN])

    annotation_source_dataframe = downstream_structure_score_dataframe[selected_columns].copy()
    annotation_source_dataframe.rename(
        columns={
            "Bx ID": "Biopsy ROI",
            "Bx refnum": "Biopsy ref #",
            "Bx index": "Biopsy index",
            "Relative structure ROI": "Target structure ID",
            "Relative structure type": "Target structure type",
            "Relative structure index": "Target structure index",
        },
        inplace=True,
    )
    annotation_source_dataframe = _normalize_downstream_mc_join_columns(annotation_source_dataframe)
    annotation_source_dataframe[DOWNSTREAM_MC_SCORE_MERGE_COLUMN] = pandas.to_numeric(
        annotation_source_dataframe["Global mean binom est"],
        errors="coerce",
    )
    annotation_source_dataframe[DOWNSTREAM_MC_TRIAL_COUNT_MERGE_COLUMN] = np.int32(downstream_trial_count)
    annotation_source_dataframe = annotation_source_dataframe[
        [
            *DOWNSTREAM_MC_JOIN_COLUMNS,
            DOWNSTREAM_MC_SCORE_MERGE_COLUMN,
            DOWNSTREAM_MC_TRIAL_COUNT_MERGE_COLUMN,
        ]
    ]
    return annotation_source_dataframe.drop_duplicates(
        subset=list(DOWNSTREAM_MC_JOIN_COLUMNS),
        keep="last",
    ).reset_index(drop=True)


def _normalize_downstream_mc_join_columns(dataframe: pandas.DataFrame) -> pandas.DataFrame:
    updated_dataframe = dataframe.copy()
    string_join_columns = (
        "Patient ID",
        "Biopsy ROI",
        "Target structure ID",
        "Target structure type",
    )
    numeric_join_columns = (
        "Biopsy ref #",
        "Biopsy index",
        "Target structure index",
    )
    for column_name in string_join_columns:
        if column_name in updated_dataframe.columns:
            updated_dataframe[column_name] = updated_dataframe[column_name].astype(str)
    for column_name in numeric_join_columns:
        if column_name in updated_dataframe.columns:
            updated_dataframe[column_name] = pandas.to_numeric(
                updated_dataframe[column_name],
                errors="coerce",
            )
    return updated_dataframe


def _initialize_downstream_mc_placeholder_columns(
    dataframe: pandas.DataFrame,
) -> pandas.DataFrame:
    updated_dataframe = dataframe.copy()
    placeholder_columns = (
        "Target optimizer selected winner downstream MC target score",
        "Target optimizer selected winner downstream MC trial count",
        "Target optimizer selected winner downstream MC agreement delta",
    )
    for column_name in placeholder_columns:
        if column_name not in updated_dataframe.columns:
            updated_dataframe[column_name] = np.nan
    return updated_dataframe


def _initialize_biopsy_sampling_audit_placeholder_columns(
    dataframe: pandas.DataFrame,
) -> pandas.DataFrame:
    updated_dataframe = dataframe.copy()
    placeholder_columns = (
        PLANNED_BIOPSY_SAMPLED_POINT_COUNT_COLUMN,
        FINALIZED_BIOPSY_SAMPLED_POINT_COUNT_COLUMN,
        BIOPSY_SAMPLED_POINT_COUNT_DELTA_COLUMN,
        BIOPSY_SAMPLED_POINT_COUNT_RATIO_COLUMN,
        BIOPSY_SAMPLED_POINT_COUNT_MISMATCH_FLAG_COLUMN,
    )
    for column_name in placeholder_columns:
        if column_name not in updated_dataframe.columns:
            updated_dataframe[column_name] = np.nan
    return updated_dataframe


def _initialize_optimizer_output_placeholder_columns(
    dataframe: pandas.DataFrame,
) -> pandas.DataFrame:
    updated_dataframe = _initialize_biopsy_sampling_audit_placeholder_columns(dataframe)
    return _initialize_downstream_mc_placeholder_columns(updated_dataframe)


def _resolve_selected_winner_score_surface_delta(row) -> float:
    optimizer_side_score = row.get(
        "Final winner resolved objective value",
        row.get(
            "Target optimizer selected winner optimizer-side target score",
            row.get("Target optimizer resolved objective value", np.nan),
        ),
    )
    if pandas.isna(optimizer_side_score):
        optimizer_side_score = row.get("Objective value", row.get("Target optimizer retained score", np.nan))

    downstream_score = row.get(
        "Winning-candidate downstream-comparable target score",
        row.get(
            "Target optimizer selected winner downstream-comparable target score",
            row.get("Target optimizer downstream comparable target score", np.nan),
        ),
    )
    if pandas.isna(optimizer_side_score) or pandas.isna(downstream_score):
        return np.nan
    return float(downstream_score) - float(optimizer_side_score)


def _apply_metadata_to_dataframe(
    dataframe: pandas.DataFrame,
    metadata: Optional[Mapping[str, Any]],
) -> pandas.DataFrame:
    if not metadata:
        return dataframe

    updated_dataframe = dataframe.copy()
    for key, value in metadata.items():
        updated_dataframe[key] = value
    return updated_dataframe


__all__ = [
    "annotate_target_dil_optimizer_dataframe_with_biopsy_sampling_audit",
    "annotate_target_dil_optimizer_dataframe_with_downstream_mc",
    "build_target_dil_optimization_summary_dataframe",
    "build_target_dil_ranked_candidate_output_dataframe",
    "build_target_dil_tested_candidate_output_dataframe",
]