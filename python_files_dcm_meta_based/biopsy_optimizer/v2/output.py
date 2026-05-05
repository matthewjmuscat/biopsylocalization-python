"""Carry-down dataframe builders for optimizer v2 outputs."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import pandas

from biopsy_optimizer.v2.contracts import OptimizerV2SearchRunResult


def build_target_dil_optimization_summary_dataframe(
    search_result: OptimizerV2SearchRunResult,
    metadata: Optional[Mapping[str, Any]] = None,
) -> pandas.DataFrame:
    """Build one carry-down winner summary row for the target-DIL optimizer lane."""
    if search_result.ranked_candidate_dataframe.empty:
        return _apply_metadata_to_dataframe(pandas.DataFrame(), metadata)

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
        "Target optimizer downstream comparable target score": winner_row.get(
            "Winning-candidate downstream-comparable target score",
            np.nan,
        ),
        "Target optimizer downstream comparable trial count": winner_row.get(
            "Downstream-comparable score trial count",
            np.nan,
        ),
        "Target optimizer agreement delta": _resolve_agreement_delta(winner_row),
    }
    summary_dataframe = pandas.DataFrame([summary_row])
    return _apply_metadata_to_dataframe(summary_dataframe, metadata)


def build_target_dil_ranked_candidate_output_dataframe(
    search_result: OptimizerV2SearchRunResult,
    metadata: Optional[Mapping[str, Any]] = None,
) -> pandas.DataFrame:
    """Build the ranked optimizer-candidate table with carry-down metadata stamped on it."""
    ranked_candidate_dataframe = search_result.ranked_candidate_dataframe.copy()
    if ranked_candidate_dataframe.empty:
        return _apply_metadata_to_dataframe(ranked_candidate_dataframe, metadata)

    ranked_candidate_dataframe["Target optimizer lane"] = "target_dil_optimizer_v2"
    ranked_candidate_dataframe["Target optimizer final stage name"] = _resolve_final_stage_name(search_result)
    ranked_candidate_dataframe["Target optimizer num stages"] = np.int32(len(search_result.stage_results))
    ranked_candidate_dataframe["Target optimizer num tested candidate rows"] = np.int32(
        len(search_result.tested_candidate_dataframe)
    )
    ranked_candidate_dataframe["Target optimizer num final ranked candidates"] = np.int32(
        len(search_result.ranked_candidate_dataframe)
    )
    ranked_candidate_dataframe["Target optimizer agreement delta"] = ranked_candidate_dataframe.apply(
        _resolve_agreement_delta,
        axis=1,
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
    return _apply_metadata_to_dataframe(ranked_candidate_dataframe, metadata)


def _resolve_final_stage_name(search_result: OptimizerV2SearchRunResult) -> str:
    if not search_result.stage_results:
        return ""
    return str(search_result.stage_results[-1].stage_name)


def _resolve_agreement_delta(row) -> float:
    retained_score = row.get("Objective value", row.get("Target optimizer retained score", np.nan))
    downstream_score = row.get(
        "Winning-candidate downstream-comparable target score",
        row.get("Target optimizer downstream comparable target score", np.nan),
    )
    if pandas.isna(retained_score) or pandas.isna(downstream_score):
        return np.nan
    return float(downstream_score) - float(retained_score)


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
    "build_target_dil_optimization_summary_dataframe",
    "build_target_dil_ranked_candidate_output_dataframe",
]