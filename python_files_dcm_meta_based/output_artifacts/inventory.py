from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd

from legacy_data_keys import legacy_data_keys


OUTPUT_ARTIFACT_INVENTORY_SCHEMA_VERSION = "phase2_output_artifact_inventory_v1"
INVENTORY_EXTENSIONS = {".csv", ".parquet", ".json", ".jsonl", ".log", ".svg", ".pdf", ".html", ".png"}
LEGACY_ARTIFACT_KEYS = legacy_data_keys.artifacts


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _path_parts(relative_path: Path) -> tuple[str, ...]:
    return tuple(part for part in relative_path.parts if part not in ("", "."))


def _artifact_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".parquet"}:
        return "table"
    if suffix in {".svg", ".pdf", ".html", ".png"}:
        return "figure_or_render_asset"
    if suffix in {".json", ".jsonl", ".log"}:
        return "manifest_or_runtime_metadata"
    return "other"


def _strip_patient_prefix(filename_stem: str, patient_uid: str | None) -> str:
    if patient_uid and filename_stem.startswith(f"{patient_uid}-"):
        return filename_stem[len(patient_uid) + 1:]
    return filename_stem


def _classify_relative_path(relative_path: Path) -> dict[str, Any]:
    parts = _path_parts(relative_path)
    suffix = relative_path.suffix.lower()
    artifact_kind = _artifact_kind(relative_path)
    output_section = parts[0] if parts else "unknown"
    patient_uid = ""
    dataframe_name = relative_path.stem
    output_class = "unclassified"
    lifetime_recommendation = "needs_review"
    classification_reason = "No path rule matched."

    if parts[:1] == ("manifests",):
        output_class = "manifest_or_runtime_metadata"
        lifetime_recommendation = "run_start_or_input_stage"
        classification_reason = "Run/input manifest artifact."
    elif len(parts) == 1 and relative_path.name.startswith("uncertainties_file_auto_generated"):
        output_class = "manifest_or_runtime_metadata"
        lifetime_recommendation = "run_start_or_input_stage"
        classification_reason = "Run-level generated uncertainty configuration sidecar."
    elif parts[:1] == ("logs",):
        output_class = "manifest_or_runtime_metadata"
        lifetime_recommendation = "runtime_logging"
        classification_reason = "Runtime log/status artifact."
    elif len(parts) >= 3 and parts[0] == "Output CSVs" and parts[1] == "Preprocessing":
        output_section = "Output CSVs/Preprocessing"
        patient_uid = parts[2]
        dataframe_name = _strip_patient_prefix(relative_path.stem, patient_uid)
        if patient_uid == LEGACY_ARTIFACT_KEYS.global_patient_uid:
            output_class = "true_cohort_post_aggregation"
            lifetime_recommendation = "cohort_post_aggregation"
            classification_reason = "Preprocessing global folder artifact."
        else:
            output_class = "patient_append_safe_table"
            lifetime_recommendation = "flush_after_patient_preprocessing"
            classification_reason = "Patient preprocessing dataframe exported from the per-patient dataframe dict."
    elif len(parts) >= 3 and parts[0] == "Output CSVs" and parts[1] == "MC simulation":
        output_section = "Output CSVs/MC simulation"
        patient_uid = parts[2]
        dataframe_name = _strip_patient_prefix(relative_path.stem, patient_uid)
        if patient_uid == LEGACY_ARTIFACT_KEYS.global_patient_uid:
            output_class = "true_cohort_post_aggregation"
            lifetime_recommendation = "cohort_post_aggregation"
            classification_reason = "MC global folder artifact."
        elif len(parts) >= 4:
            output_class = "patient_derived_table_requiring_stitch_only"
            lifetime_recommendation = "flush_after_biopsy_or_patient_mc"
            classification_reason = "Per-biopsy MC table nested under a patient and biopsy directory."
        else:
            output_class = "patient_append_safe_table"
            lifetime_recommendation = "flush_after_patient_mc"
            classification_reason = "Patient-level MC dataframe exported from the per-patient dataframe dict."
    elif len(parts) >= 2 and parts[0] == "Output CSVs" and parts[1] == "Cohort":
        output_section = "Output CSVs/Cohort"
        output_class = "true_cohort_post_aggregation"
        lifetime_recommendation = "cohort_post_aggregation"
        classification_reason = "Cohort dataframe exported from master_cohort_patient_data_and_dataframes."
    elif len(parts) >= 2 and parts[0] == "Output CSVs" and parts[1] == "FANOVA simulation":
        output_section = "Output CSVs/FANOVA simulation"
        patient_uid = parts[2] if len(parts) >= 3 else ""
        output_class = "deprecated_or_analysis_only_output"
        lifetime_recommendation = "analysis_optional_or_deprecated"
        classification_reason = "FANOVA/Sobol analysis output is optional analysis output in the current pipeline."
    elif parts[:1] == ("Output figures",):
        output_section = "Output figures"
        output_class = "figure_or_render_asset"
        lifetime_recommendation = "render_after_required_tables_exist"
        classification_reason = "Rendered figure or validation asset."
    elif artifact_kind == "figure_or_render_asset":
        output_class = "figure_or_render_asset"
        lifetime_recommendation = "render_after_required_tables_exist"
        classification_reason = "Figure/render file extension."
    elif artifact_kind == "manifest_or_runtime_metadata":
        output_class = "manifest_or_runtime_metadata"
        lifetime_recommendation = "runtime_or_stage_metadata"
        classification_reason = "Manifest or runtime metadata extension."

    return {
        "artifact_kind": artifact_kind,
        "output_section": output_section,
        "patient_uid": patient_uid,
        "legacy_dataframe_name": dataframe_name,
        "output_class": output_class,
        "lifetime_recommendation": lifetime_recommendation,
        "classification_reason": classification_reason,
        "primary_key_status": "needs_phase2_review" if suffix in {".csv", ".parquet"} else "not_applicable",
        "builder_mapping_status": "needs_phase2_review" if suffix in {".csv", ".parquet"} else "not_applicable",
    }


def build_output_artifact_inventory(run_dir: Path, *, include_other_files: bool = False) -> pd.DataFrame:
    run_dir = Path(run_dir).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if not include_other_files and suffix not in INVENTORY_EXTENSIONS:
            continue
        relative_path = path.relative_to(run_dir)
        classification = _classify_relative_path(relative_path)
        rows.append({
            "schema_version": OUTPUT_ARTIFACT_INVENTORY_SCHEMA_VERSION,
            "run_dir": str(run_dir),
            "relative_path": relative_path.as_posix(),
            "file_name": path.name,
            "file_extension": suffix,
            "file_size_bytes": int(path.stat().st_size),
            **classification,
        })

    if not rows:
        return pd.DataFrame(columns=[
            "schema_version",
            "run_dir",
            "relative_path",
            "file_name",
            "file_extension",
            "file_size_bytes",
            "artifact_kind",
            "output_section",
            "patient_uid",
            "legacy_dataframe_name",
            "output_class",
            "lifetime_recommendation",
            "classification_reason",
            "primary_key_status",
            "builder_mapping_status",
        ])

    return pd.DataFrame(rows).sort_values("relative_path").reset_index(drop=True)


def summarize_output_artifact_inventory(inventory_df: pd.DataFrame) -> dict[str, Any]:
    if inventory_df.empty:
        return {
            "schema_version": OUTPUT_ARTIFACT_INVENTORY_SCHEMA_VERSION,
            "generated_utc": _utc_now_iso(),
            "artifact_count": 0,
            "artifact_kind_counts": {},
            "output_class_counts": {},
            "output_section_counts": {},
        }
    return {
        "schema_version": OUTPUT_ARTIFACT_INVENTORY_SCHEMA_VERSION,
        "generated_utc": _utc_now_iso(),
        "run_dir": str(inventory_df["run_dir"].iloc[0]),
        "artifact_count": int(len(inventory_df)),
        "artifact_kind_counts": dict(Counter(inventory_df["artifact_kind"])),
        "output_class_counts": dict(Counter(inventory_df["output_class"])),
        "output_section_counts": dict(Counter(inventory_df["output_section"])),
        "table_count": int((inventory_df["artifact_kind"] == "table").sum()),
    }


def write_output_artifact_inventory(inventory_df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = output_dir.joinpath("output_artifact_inventory.csv")
    summary_path = output_dir.joinpath("output_artifact_inventory_summary.json")
    inventory_df.to_csv(inventory_path, index=False)
    with summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(summarize_output_artifact_inventory(inventory_df), file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")
    return inventory_path, summary_path