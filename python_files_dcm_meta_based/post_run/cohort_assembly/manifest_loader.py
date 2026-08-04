"""Load completed patient-runner manifests for post-run utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from patient_runner.contracts import PatientBatchRunResult
from patient_runner.contracts import PatientCase
from patient_runner.contracts import PatientRunResult
from patient_runner.contracts import PatientStageResult
from patient_runner.contracts import PatientStageStatus
from patient_runner.manifests import PATIENT_BATCH_RUN_MANIFEST_SCHEMA_VERSION
from patient_runner.manifests import PATIENT_RUN_MANIFEST_SCHEMA_VERSION


BATCH_MANIFEST_FILE_NAME = "patient_batch_run_manifest.json"
PATIENT_MANIFEST_FILE_NAME = "patient_run_manifest.json"


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise TypeError(f"Manifest root must be a JSON object: {path}")
    return payload


def _resolve_existing_dir_or_prefix(requested_path: Path) -> Path:
    requested_path = requested_path.expanduser()
    if requested_path.is_dir() or requested_path.is_file():
        return requested_path
    parent = requested_path.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"Parent directory does not exist: {parent}")
    candidates = sorted(
        path
        for path in parent.glob(f"{requested_path.name}*")
        if path.is_dir() or path.is_file()
    )
    if not candidates:
        raise FileNotFoundError(f"Could not resolve manifest path or output prefix: {requested_path}")
    if len(candidates) > 1:
        raise FileNotFoundError(
            f"Ambiguous manifest path or output prefix {requested_path!s}: {[str(path) for path in candidates]}"
        )
    return candidates[0]


def resolve_patient_batch_manifest_path(manifest_path_or_output_dir: str | Path) -> Path:
    """Resolve a batch manifest from a manifest path, runner dir, or run-output dir."""
    resolved_path = _resolve_existing_dir_or_prefix(Path(manifest_path_or_output_dir))
    if resolved_path.is_file():
        if resolved_path.name != BATCH_MANIFEST_FILE_NAME:
            raise FileNotFoundError(f"Expected {BATCH_MANIFEST_FILE_NAME}, got: {resolved_path}")
        return resolved_path

    direct_manifest = resolved_path / BATCH_MANIFEST_FILE_NAME
    if direct_manifest.is_file():
        return direct_manifest
    scientific_runner_manifest = resolved_path / "patient_scientific_runner" / BATCH_MANIFEST_FILE_NAME
    if scientific_runner_manifest.is_file():
        return scientific_runner_manifest
    raise FileNotFoundError(
        "Could not find patient batch manifest at "
        f"{direct_manifest} or {scientific_runner_manifest}"
    )


def _validate_schema(manifest: Mapping[str, Any], expected_schema_version: str, source_path: Path) -> None:
    schema_version = str(manifest.get("schema_version", "")).strip()
    if schema_version and schema_version != expected_schema_version:
        raise ValueError(
            f"Unsupported manifest schema_version {schema_version!r} in {source_path}; "
            f"expected {expected_schema_version!r}"
        )


def _path_from_manifest(value: Any, *, fallback_root: Path) -> Path:
    path_text = str(value or "").strip()
    if not path_text:
        return fallback_root
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return fallback_root / path


def _paths_from_manifest(manifest: Mapping[str, Any], root: Path) -> tuple[Path, ...]:
    absolute_paths = tuple(Path(path).expanduser() for path in manifest.get("artifact_paths", ()) if str(path).strip())
    if absolute_paths:
        return tuple(path if path.is_absolute() else root / path for path in absolute_paths)
    return tuple(
        root / str(path)
        for path in manifest.get("artifact_paths_relative_to_output_root", ())
        if str(path).strip()
    )


def _status_from_manifest(value: Any) -> PatientStageStatus:
    status_text = str(value or PatientStageStatus.SKIPPED.value).strip() or PatientStageStatus.SKIPPED.value
    return PatientStageStatus(status_text)


def _stage_results_from_manifest(patient_manifest: Mapping[str, Any]) -> tuple[PatientStageResult, ...]:
    stage_results: list[PatientStageResult] = []
    for raw_stage in patient_manifest.get("stages", ()):
        if not isinstance(raw_stage, Mapping):
            continue
        stage_results.append(
            PatientStageResult(
                stage_name=str(raw_stage.get("stage_name", "unknown")),
                status=_status_from_manifest(raw_stage.get("status")),
                elapsed_seconds=float(raw_stage.get("elapsed_seconds", 0.0) or 0.0),
                artifact_count=int(raw_stage.get("artifact_count", 0) or 0),
                output_paths=tuple(Path(path) for path in raw_stage.get("output_paths", ()) if str(path).strip()),
                warnings=tuple(str(warning) for warning in raw_stage.get("warnings", ())),
                metadata=dict(raw_stage.get("metadata", {})),
            )
        )
    return tuple(stage_results)


def _fallback_artifacts_for_patient(batch_artifact_paths: Sequence[Path], patient_output_root: Path) -> tuple[Path, ...]:
    artifacts: list[Path] = []
    for artifact_path in batch_artifact_paths:
        try:
            artifact_path.relative_to(patient_output_root)
        except ValueError:
            continue
        artifacts.append(artifact_path)
    return tuple(artifacts)


def _load_patient_manifest(patient_output_root: Path) -> tuple[Path | None, dict[str, Any]]:
    patient_manifest_path = patient_output_root / PATIENT_MANIFEST_FILE_NAME
    if not patient_manifest_path.is_file():
        return None, {}
    patient_manifest = _read_json_object(patient_manifest_path)
    _validate_schema(patient_manifest, PATIENT_RUN_MANIFEST_SCHEMA_VERSION, patient_manifest_path)
    return patient_manifest_path, patient_manifest


def _patient_result_from_manifest_record(record: Mapping[str, Any],
                                         *,
                                         batch_output_root: Path,
                                         batch_metadata: Mapping[str, Any],
                                         batch_artifact_paths: Sequence[Path]) -> PatientRunResult:
    patient_output_root = _path_from_manifest(
        record.get("output_root") or record.get("output_root_relative_to_batch_root"),
        fallback_root=batch_output_root,
    )
    _, patient_manifest = _load_patient_manifest(patient_output_root)
    patient_uid = str(patient_manifest.get("patient_uid") or record.get("patient_uid") or "").strip()
    if not patient_uid:
        raise ValueError(f"Patient manifest record is missing patient_uid under {patient_output_root}")

    artifact_paths = _paths_from_manifest(patient_manifest, patient_output_root) if patient_manifest else ()
    if not artifact_paths:
        artifact_paths = _fallback_artifacts_for_patient(batch_artifact_paths, patient_output_root)

    patient_case = PatientCase(
        patient_uid=patient_uid,
        patient_label=str(patient_manifest.get("patient_label") or record.get("patient_label") or ""),
        source_run_id=str(patient_manifest.get("source_run_id") or batch_metadata.get("source_run_id") or ""),
        input_manifest_id=str(patient_manifest.get("input_manifest_id") or batch_metadata.get("input_manifest_id") or ""),
        metadata=dict(patient_manifest.get("patient_metadata", {})),
    )
    return PatientRunResult(
        patient_case=patient_case,
        status=_status_from_manifest(patient_manifest.get("status") or record.get("status")),
        output_root=patient_output_root,
        elapsed_seconds=float(patient_manifest.get("elapsed_seconds") or record.get("elapsed_seconds") or 0.0),
        stage_results=_stage_results_from_manifest(patient_manifest),
        artifact_paths=artifact_paths,
        metadata=dict(patient_manifest.get("metadata", {})),
    )


def load_patient_batch_result_from_manifest(manifest_path_or_output_dir: str | Path) -> PatientBatchRunResult:
    """Reconstruct the lightweight batch result needed by post-run assembly."""
    manifest_path = resolve_patient_batch_manifest_path(manifest_path_or_output_dir)
    batch_manifest = _read_json_object(manifest_path)
    _validate_schema(batch_manifest, PATIENT_BATCH_RUN_MANIFEST_SCHEMA_VERSION, manifest_path)
    batch_output_root = _path_from_manifest(batch_manifest.get("output_root"), fallback_root=manifest_path.parent)
    batch_metadata = dict(batch_manifest.get("metadata", {}))
    batch_artifact_paths = _paths_from_manifest(batch_manifest, batch_output_root)
    patient_records = batch_manifest.get("patients", ())
    if not isinstance(patient_records, list):
        raise TypeError(f"patients must be a list in {manifest_path}")
    patient_results = tuple(
        _patient_result_from_manifest_record(
            record,
            batch_output_root=batch_output_root,
            batch_metadata=batch_metadata,
            batch_artifact_paths=batch_artifact_paths,
        )
        for record in patient_records
        if isinstance(record, Mapping)
    )
    return PatientBatchRunResult(
        status=_status_from_manifest(batch_manifest.get("status")),
        output_root=batch_output_root,
        patient_results=patient_results,
        elapsed_seconds=float(batch_manifest.get("elapsed_seconds", 0.0) or 0.0),
        metadata={
            **batch_metadata,
            "loaded_from_manifest_path": manifest_path.as_posix(),
        },
    )