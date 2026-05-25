from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd

from legacy_data_keys import legacy_data_keys

from .contracts import normalize_legacy_table_name


PATIENT_ARTIFACT_MANIFEST_SCHEMA_VERSION = "phase3a_patient_artifact_manifest_v1"
PATIENT_STITCH_PLAN_SCHEMA_VERSION = "phase3a_patient_stitch_plan_v1"
LEGACY_ARTIFACT_KEYS = legacy_data_keys.artifacts
CURRENT_DTYPE_POLICY_NOTE = (
    "Shadow manifest only; does not convert dataframe dtypes. Current runtime memory policy remains "
    "dataframe_builders.convert_columns_to_categorical_and_downcast(...)."
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_path_name(value: str) -> str:
    safe = str(value).strip().replace("/", "_").replace("\\", "_")
    for old, new in ((" ", "_"), (":", "_"), (",", "_")):
        safe = safe.replace(old, new)
    return safe or "unknown"


def _patient_scope(patient_uid: str) -> str:
    if patient_uid == "":
        return "run_or_cohort"
    if patient_uid == LEGACY_ARTIFACT_KEYS.global_patient_uid:
        return "global"
    return "patient"


def _load_contract_lookup(contracts_df: pd.DataFrame) -> dict[tuple[str, str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for contract_row in contracts_df.to_dict(orient="records"):
        key = (
            str(contract_row.get("normalized_table_name", "")),
            str(contract_row.get("output_section", "")),
            str(contract_row.get("file_extension", "")),
        )
        lookup[key] = contract_row
    return lookup


def build_patient_artifact_manifest(inventory_df: pd.DataFrame,
                                    contracts_df: pd.DataFrame) -> pd.DataFrame:
    """Build a shadow manifest of current patient-scoped artifacts.

    This reads the current file inventory only. It does not write pipeline outputs,
    load dataframes, or apply dtype conversions.
    """
    if inventory_df.empty:
        return pd.DataFrame()

    inventory_df = inventory_df.copy()
    table_mask = inventory_df["artifact_kind"].eq("table")
    inventory_df.loc[table_mask, "normalized_table_name"] = inventory_df.loc[table_mask].apply(
        normalize_legacy_table_name,
        axis=1,
    )
    inventory_df.loc[~table_mask, "normalized_table_name"] = inventory_df.loc[
        ~table_mask,
        "legacy_dataframe_name",
    ]

    contract_lookup = _load_contract_lookup(contracts_df)
    rows: list[dict[str, Any]] = []
    for artifact_row in inventory_df.to_dict(orient="records"):
        patient_uid = str(artifact_row.get("patient_uid", "") or "")
        scope = _patient_scope(patient_uid)
        if scope != "patient":
            continue

        key = (
            str(artifact_row.get("normalized_table_name", "")),
            str(artifact_row.get("output_section", "")),
            str(artifact_row.get("file_extension", "")),
        )
        contract_row = contract_lookup.get(key, {})
        rows.append({
            "schema_version": PATIENT_ARTIFACT_MANIFEST_SCHEMA_VERSION,
            "patient_uid": patient_uid,
            "artifact_scope": scope,
            "relative_path": artifact_row.get("relative_path", ""),
            "output_section": artifact_row.get("output_section", ""),
            "artifact_kind": artifact_row.get("artifact_kind", ""),
            "file_extension": artifact_row.get("file_extension", ""),
            "file_size_bytes": artifact_row.get("file_size_bytes", 0),
            "normalized_table_name": artifact_row.get("normalized_table_name", ""),
            "current_output_class": artifact_row.get("output_class", ""),
            "proposed_lifetime_class": contract_row.get("proposed_lifetime_class", "not_applicable"),
            "source_stage": contract_row.get("source_stage", "not_applicable"),
            "builder_or_source": contract_row.get("builder_or_source", "not_applicable"),
            "canonical_primary_key": contract_row.get("canonical_primary_key", "not_applicable"),
            "stitch_key": contract_row.get("stitch_key", "not_applicable"),
            "pruning_assessment": contract_row.get("pruning_assessment", "not_applicable"),
            "dtype_policy_note": CURRENT_DTYPE_POLICY_NOTE,
            "shadow_manifest_status": "existing_current_output_artifact",
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["patient_uid", "relative_path"]).reset_index(drop=True)


def build_patient_stitch_plan(contracts_df: pd.DataFrame) -> pd.DataFrame:
    """Build the shadow plan for reconstructing final artifacts from patient fragments."""
    rows: list[dict[str, Any]] = []
    for contract_row in contracts_df.to_dict(orient="records"):
        proposed_lifetime_class = str(contract_row.get("proposed_lifetime_class", ""))
        if proposed_lifetime_class == "run_metadata":
            stitch_readiness = "run_metadata_not_patient_stitched"
        elif proposed_lifetime_class in {
            "patient_appendable",
            "patient_or_biopsy_fragment_stitchable",
            "patient_derived_stitchable",
        }:
            stitch_readiness = "patient_fragment_ready"
        elif proposed_lifetime_class == "cohort_named_but_likely_stitchable":
            stitch_readiness = "needs_patient_fragment_builder"
        elif proposed_lifetime_class == "final_stage_after_patient_fragments":
            stitch_readiness = "needs_final_aggregation_builder"
        elif proposed_lifetime_class == "downstream_calculable_or_optional_derived":
            stitch_readiness = "defer_until_pruning_decision"
        else:
            stitch_readiness = "needs_review"

        rows.append({
            "schema_version": PATIENT_STITCH_PLAN_SCHEMA_VERSION,
            "normalized_table_name": contract_row.get("normalized_table_name", ""),
            "output_section": contract_row.get("output_section", ""),
            "file_extension": contract_row.get("file_extension", ""),
            "current_output_class": contract_row.get("current_output_class", ""),
            "proposed_lifetime_class": proposed_lifetime_class,
            "stitch_readiness": stitch_readiness,
            "source_stage": contract_row.get("source_stage", ""),
            "builder_or_source": contract_row.get("builder_or_source", ""),
            "canonical_primary_key": contract_row.get("canonical_primary_key", ""),
            "stitch_key": contract_row.get("stitch_key", ""),
            "pruning_assessment": contract_row.get("pruning_assessment", ""),
            "pruning_notes": contract_row.get("pruning_notes", ""),
            "dtype_policy_note": CURRENT_DTYPE_POLICY_NOTE,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["stitch_readiness", "output_section", "normalized_table_name"]).reset_index(drop=True)


def summarize_patient_artifacts(patient_manifest_df: pd.DataFrame,
                                stitch_plan_df: pd.DataFrame) -> dict[str, Any]:
    summary = {
        "schema_version": PATIENT_ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "generated_utc": _utc_now_iso(),
        "dtype_policy_note": CURRENT_DTYPE_POLICY_NOTE,
        "patient_artifact_count": int(len(patient_manifest_df)),
        "patient_count": 0,
        "artifact_scope_counts": {},
        "patient_artifact_counts_by_patient": {},
        "stitch_plan_contract_count": int(len(stitch_plan_df)),
        "stitch_readiness_counts": {},
    }
    if not patient_manifest_df.empty:
        patient_rows = patient_manifest_df[patient_manifest_df["artifact_scope"].eq("patient")]
        summary["patient_count"] = int(patient_rows["patient_uid"].nunique())
        summary["artifact_scope_counts"] = dict(Counter(patient_manifest_df["artifact_scope"]))
        summary["patient_artifact_counts_by_patient"] = dict(Counter(patient_rows["patient_uid"]))
    if not stitch_plan_df.empty:
        summary["stitch_readiness_counts"] = dict(Counter(stitch_plan_df["stitch_readiness"]))
    return summary


def write_patient_artifact_outputs(patient_manifest_df: pd.DataFrame,
                                   stitch_plan_df: pd.DataFrame,
                                   output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir.joinpath("patient_artifact_manifest.csv")
    stitch_plan_path = output_dir.joinpath("patient_stitch_plan.csv")
    summary_path = output_dir.joinpath("patient_artifact_manifest_summary.json")

    patient_manifest_df.to_csv(manifest_path, index=False)
    stitch_plan_df.to_csv(stitch_plan_path, index=False)
    with summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(
            summarize_patient_artifacts(patient_manifest_df, stitch_plan_df),
            file_obj,
            indent=2,
            sort_keys=True,
        )
        file_obj.write("\n")

    per_patient_dir = output_dir.joinpath("per_patient")
    per_patient_dir.mkdir(parents=True, exist_ok=True)
    if not patient_manifest_df.empty:
        for patient_uid, patient_df in patient_manifest_df.groupby("patient_uid"):
            patient_path = per_patient_dir.joinpath(f"{_safe_path_name(patient_uid)}_artifact_manifest.csv")
            patient_df.sort_values("relative_path").to_csv(patient_path, index=False)

    return manifest_path, stitch_plan_path, summary_path