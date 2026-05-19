from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .contracts import CANONICAL_KEY_POLICY_NOTE
from .exporters import DataframeArtifact
from .exporters import iter_biopsy_mc_artifacts
from .exporters import iter_patient_mc_artifacts
from .exporters import iter_patient_preprocessing_artifacts
from .exporters import write_dataframe_artifact
from .in_memory_stitching import build_in_memory_stitch_validation
from .in_memory_stitching import summarize_in_memory_stitch_validation
from .schema_registry import write_output_schema_coverage_report


PHASE3C_OUTPUT_SURFACE_SCHEMA_VERSION = "phase3c_patient_fragment_output_surface_v1"
PHASE3C_OUTPUT_DIR_NAME = "phase3c_patient_fragment_output_surface"


@dataclass(frozen=True)
class Phase3COutputSurfaceResult:
    output_dir: Path
    manifest_path: Path
    summary_path: Path
    stitch_validation_path: Path
    stitch_validation_summary_path: Path
    schema_coverage_path: Path
    schema_coverage_summary_path: Path
    schema_unmatched_manifest_path: Path
    artifact_count: int
    summary: dict[str, Any]
    stitch_validation_summary: dict[str, Any]
    schema_coverage_summary: dict[str, Any]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _cohort_relative_path(dataframe_name: str) -> Path:
    return Path("Output CSVs").joinpath("Cohort", f"{dataframe_name}.csv")


def _artifact_row(artifact: DataframeArtifact,
                  artifact_path: Path,
                  output_dir: Path,
                  phase3c_role: str,
                  generated_utc: str) -> dict[str, Any]:
    dataframe = artifact.dataframe
    relative_path = artifact_path.relative_to(output_dir)
    return {
        "schema_version": PHASE3C_OUTPUT_SURFACE_SCHEMA_VERSION,
        "generated_utc": generated_utc,
        "phase3c_role": phase3c_role,
        "source_scope": artifact.source_scope,
        "patient_uid": artifact.patient_uid or "",
        "biopsy_index": "" if artifact.biopsy_index is None else artifact.biopsy_index,
        "dataframe_name": artifact.dataframe_name,
        "relative_path": relative_path.as_posix(),
        "file_extension": artifact.file_extension,
        "file_size_bytes": int(artifact_path.stat().st_size) if artifact_path.exists() else 0,
        "row_count": int(len(dataframe)),
        "column_count": int(len(dataframe.columns)),
        "has_multiindex_columns": bool(isinstance(dataframe.columns, pd.MultiIndex)),
        "column_nlevels": int(dataframe.columns.nlevels),
        "index_nlevels": int(dataframe.index.nlevels),
        "csv_index_written": False if artifact.file_extension == ".csv" else "not_applicable",
        "canonical_key_policy_note": CANONICAL_KEY_POLICY_NOTE,
    }


def _stitched_artifacts_from_tables(stitched_tables: dict[str, pd.DataFrame]) -> list[DataframeArtifact]:
    artifacts: list[DataframeArtifact] = []
    for dataframe_name, dataframe in stitched_tables.items():
        artifacts.append(
            DataframeArtifact(
                source_scope="stitched_cohort",
                dataframe_name=dataframe_name,
                dataframe=dataframe,
                relative_path=_cohort_relative_path(dataframe_name),
                file_extension=".csv",
            )
        )
    return artifacts


def collect_phase3c_output_artifacts(master_structure_reference_dict: dict,
                                     master_cohort_patient_data_and_dataframes: dict,
                                     all_ref_key: str,
                                     bx_ref: str) -> tuple[list[DataframeArtifact], pd.DataFrame, dict[str, pd.DataFrame]]:
    patient_fragment_artifacts = [
        *iter_patient_preprocessing_artifacts(master_structure_reference_dict, all_ref_key),
        *iter_patient_mc_artifacts(master_structure_reference_dict, all_ref_key),
        *iter_biopsy_mc_artifacts(master_structure_reference_dict, bx_ref),
    ]
    validation_df, stitched_tables = build_in_memory_stitch_validation(
        master_structure_reference_dict=master_structure_reference_dict,
        master_cohort_patient_data_and_dataframes=master_cohort_patient_data_and_dataframes,
        all_ref_key=all_ref_key,
        bx_ref=bx_ref,
        return_stitched_tables=True,
    )
    stitched_artifacts = _stitched_artifacts_from_tables(stitched_tables)
    return [*patient_fragment_artifacts, *stitched_artifacts], validation_df, stitched_tables


def summarize_phase3c_artifact_manifest(manifest_df: pd.DataFrame,
                                         stitch_validation_df: pd.DataFrame,
                                         generated_utc: str | None = None) -> dict[str, Any]:
    if generated_utc is None:
        generated_utc = _utc_now_iso()

    if manifest_df.empty:
        return {
            "schema_version": PHASE3C_OUTPUT_SURFACE_SCHEMA_VERSION,
            "generated_utc": generated_utc,
            "artifact_count": 0,
            "phase3c_role_counts": {},
            "source_scope_counts": {},
            "file_extension_counts": {},
            "multiindex_artifact_count": 0,
            "stitch_validation": summarize_in_memory_stitch_validation(stitch_validation_df),
        }

    patient_rows = manifest_df[manifest_df["patient_uid"].astype(str).ne("")]
    return {
        "schema_version": PHASE3C_OUTPUT_SURFACE_SCHEMA_VERSION,
        "generated_utc": generated_utc,
        "artifact_count": int(len(manifest_df)),
        "patient_count": int(patient_rows["patient_uid"].nunique()),
        "phase3c_role_counts": dict(Counter(manifest_df["phase3c_role"])),
        "source_scope_counts": dict(Counter(manifest_df["source_scope"])),
        "file_extension_counts": dict(Counter(manifest_df["file_extension"])),
        "multiindex_artifact_count": int(manifest_df["has_multiindex_columns"].astype(bool).sum()),
        "stitch_validation": summarize_in_memory_stitch_validation(stitch_validation_df),
    }


def write_phase3c_output_surface(master_structure_reference_dict: dict,
                                 master_cohort_patient_data_and_dataframes: dict,
                                 all_ref_key: str,
                                 bx_ref: str,
                                 output_dir: Path,
                                 *,
                                 write_stitched_tables: bool = True) -> Phase3COutputSurfaceResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts, stitch_validation_df, _stitched_tables = collect_phase3c_output_artifacts(
        master_structure_reference_dict=master_structure_reference_dict,
        master_cohort_patient_data_and_dataframes=master_cohort_patient_data_and_dataframes,
        all_ref_key=all_ref_key,
        bx_ref=bx_ref,
    )

    generated_utc = _utc_now_iso()
    manifest_rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        if artifact.source_scope == "stitched_cohort" and not write_stitched_tables:
            continue
        artifact_path = write_dataframe_artifact(
            artifact.dataframe,
            output_dir.joinpath(artifact.relative_path),
            csv_index=False,
            parquet_index=False,
        )
        phase3c_role = (
            "stitched_final_artifact"
            if artifact.source_scope == "stitched_cohort"
            else "patient_fragment_artifact"
        )
        manifest_rows.append(_artifact_row(artifact, artifact_path, output_dir, phase3c_role, generated_utc))

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = output_dir.joinpath("phase3c_artifact_manifest.csv")
    summary_path = output_dir.joinpath("phase3c_artifact_manifest_summary.json")
    stitch_validation_path = output_dir.joinpath("phase3c_stitch_validation.csv")
    stitch_validation_summary_path = output_dir.joinpath("phase3c_stitch_validation_summary.json")

    manifest_df.to_csv(manifest_path, index=False)
    stitch_validation_df.to_csv(stitch_validation_path, index=False)
    schema_coverage_path, schema_unmatched_manifest_path, schema_coverage_summary_path, schema_coverage_summary = write_output_schema_coverage_report(
        manifest_df,
        stitch_validation_df,
        output_dir,
    )
    summary = summarize_phase3c_artifact_manifest(manifest_df, stitch_validation_df, generated_utc)
    summary["schema_coverage"] = schema_coverage_summary
    stitch_validation_summary = summarize_in_memory_stitch_validation(stitch_validation_df)

    with summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")
    with stitch_validation_summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(stitch_validation_summary, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")

    return Phase3COutputSurfaceResult(
        output_dir=output_dir,
        manifest_path=manifest_path,
        summary_path=summary_path,
        stitch_validation_path=stitch_validation_path,
        stitch_validation_summary_path=stitch_validation_summary_path,
        schema_coverage_path=schema_coverage_path,
        schema_coverage_summary_path=schema_coverage_summary_path,
        schema_unmatched_manifest_path=schema_unmatched_manifest_path,
        artifact_count=int(len(manifest_df)),
        summary=summary,
        stitch_validation_summary=stitch_validation_summary,
        schema_coverage_summary=schema_coverage_summary,
    )