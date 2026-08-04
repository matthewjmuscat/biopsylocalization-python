"""Run manifest writers for patient-runner results."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping

from output_artifacts.manifest_contracts import manifest_contract

from .contracts import PatientBatchRunResult
from .contracts import PatientRunResult
from .contracts import PatientStageResult


PATIENT_RUN_MANIFEST_SCHEMA_VERSION = "patient_run_manifest_v1"
PATIENT_BATCH_RUN_MANIFEST_SCHEMA_VERSION = "patient_batch_run_manifest_v1"

MANIFEST_CONTRACTS = (
    manifest_contract(
        "patient_run_manifest",
        "Patient run manifest",
        "patient",
        "manifest",
        "current_durable",
        ("patients/<patient_uid>/patient_run_manifest.json", "patient_run_manifest.json"),
        "json",
        "patient_runner.manifests.PATIENT_RUN_MANIFEST_SCHEMA_VERSION",
        "patient_runner.manifests.write_patient_run_manifest",
        "Record status, stage results, artifacts, and lightweight identity for one patient run.",
        (
            "patient UID and label",
            "source run and input manifest IDs",
            "patient metadata",
            "overall patient status",
            "output root",
            "elapsed time",
            "stage statuses and warnings",
            "artifact paths",
        ),
        reader="post_run.cohort_assembly.manifest_loader.load_patient_batch_result_from_manifest",
    ),
    manifest_contract(
        "patient_batch_run_manifest",
        "Patient batch run manifest",
        "batch_run",
        "manifest",
        "current_durable",
        ("patient_batch_run_manifest.json", "patient_scientific_runner/patient_batch_run_manifest.json"),
        "json",
        "patient_runner.manifests.PATIENT_BATCH_RUN_MANIFEST_SCHEMA_VERSION",
        "patient_runner.manifests.write_patient_batch_run_manifest",
        "Aggregate patient-run results and provide the primary entry point for post-run assembly.",
        (
            "batch status",
            "output root",
            "elapsed time",
            "patient count and failed-patient count",
            "batch artifact paths",
            "per-patient status summaries",
            "run metadata",
        ),
        reader="post_run.cohort_assembly.manifest_loader.load_patient_batch_result_from_manifest",
    ),
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _relative_paths(paths: tuple[Path, ...], root: Path) -> list[str]:
    relative_paths: list[str] = []
    for path in paths:
        try:
            relative_paths.append(path.relative_to(root).as_posix())
        except ValueError:
            relative_paths.append("")
    return relative_paths


def _stage_result_manifest(stage_result: PatientStageResult) -> dict[str, Any]:
    return {
        "stage_name": stage_result.stage_name,
        "status": stage_result.status.value,
        "elapsed_seconds": stage_result.elapsed_seconds,
        "artifact_count": stage_result.artifact_count,
        "output_paths": [path.as_posix() for path in stage_result.output_paths],
        "warnings": list(stage_result.warnings),
        "metadata": _json_safe(stage_result.metadata),
    }


def patient_run_result_manifest(patient_result: PatientRunResult) -> dict[str, Any]:
    """Return a JSON-ready manifest for one patient run result."""
    patient_case = patient_result.patient_case
    return {
        "schema_version": PATIENT_RUN_MANIFEST_SCHEMA_VERSION,
        "generated_utc": _utc_now_iso(),
        "patient_uid": patient_case.patient_uid,
        "patient_label": patient_case.patient_label,
        "source_run_id": patient_case.source_run_id,
        "input_manifest_id": patient_case.input_manifest_id,
        "patient_metadata": _json_safe(patient_case.metadata),
        "status": patient_result.status.value,
        "succeeded": patient_result.succeeded,
        "output_root": patient_result.output_root.as_posix(),
        "elapsed_seconds": patient_result.elapsed_seconds,
        "stage_count": len(patient_result.stage_results),
        "failed_stage_count": len(patient_result.failed_stage_results),
        "artifact_count": len(patient_result.artifact_paths),
        "artifact_paths": [path.as_posix() for path in patient_result.artifact_paths],
        "artifact_paths_relative_to_output_root": _relative_paths(patient_result.artifact_paths, patient_result.output_root),
        "metadata": _json_safe(patient_result.metadata),
        "stages": [_stage_result_manifest(stage_result) for stage_result in patient_result.stage_results],
    }


def patient_batch_run_result_manifest(batch_result: PatientBatchRunResult) -> dict[str, Any]:
    """Return a JSON-ready manifest for a patient batch run result."""
    return {
        "schema_version": PATIENT_BATCH_RUN_MANIFEST_SCHEMA_VERSION,
        "generated_utc": _utc_now_iso(),
        "status": batch_result.status.value,
        "succeeded": batch_result.succeeded,
        "output_root": batch_result.output_root.as_posix(),
        "elapsed_seconds": batch_result.elapsed_seconds,
        "patient_count": batch_result.patient_count,
        "failed_patient_count": len(batch_result.failed_patient_results),
        "artifact_count": len(batch_result.artifact_paths),
        "artifact_paths": [path.as_posix() for path in batch_result.artifact_paths],
        "artifact_paths_relative_to_output_root": _relative_paths(batch_result.artifact_paths, batch_result.output_root),
        "metadata": _json_safe(batch_result.metadata),
        "patients": [
            {
                "patient_uid": patient_result.patient_case.patient_uid,
                "patient_label": patient_result.patient_case.patient_label,
                "status": patient_result.status.value,
                "succeeded": patient_result.succeeded,
                "output_root": patient_result.output_root.as_posix(),
                "output_root_relative_to_batch_root": _relative_paths(
                    (patient_result.output_root,),
                    batch_result.output_root,
                )[0],
                "elapsed_seconds": patient_result.elapsed_seconds,
                "stage_count": len(patient_result.stage_results),
                "failed_stage_count": len(patient_result.failed_stage_results),
                "artifact_count": len(patient_result.artifact_paths),
                "metadata": _json_safe(patient_result.metadata),
            }
            for patient_result in batch_result.patient_results
        ],
    }


def write_patient_run_manifest(patient_result: PatientRunResult, output_path: Path | None = None) -> Path:
    """Write a JSON manifest beside one patient's output artifacts."""
    resolved_output_path = Path(output_path) if output_path is not None else patient_result.output_root.joinpath(
        "patient_run_manifest.json",
    )
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(patient_run_result_manifest(patient_result), file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")
    return resolved_output_path


def write_patient_batch_run_manifest(batch_result: PatientBatchRunResult, output_path: Path | None = None) -> Path:
    """Write a JSON manifest for one batch run."""
    resolved_output_path = Path(output_path) if output_path is not None else batch_result.output_root.joinpath(
        "patient_batch_run_manifest.json",
    )
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(patient_batch_run_result_manifest(batch_result), file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")
    return resolved_output_path
