"""Standalone parent/worker scaffold for patient-scientific execution."""

from __future__ import annotations

import csv
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any, Mapping, Sequence

from .contracts import PatientCase
from .contracts import PatientStageStatus
from .contracts import validate_patient_uids
from .inputs import PatientInputPaths
from config.snapshots import read_pipeline_config_snapshot
from output_artifacts.run_compatibility import RUN_COMPATIBILITY_METADATA_KEY
from output_artifacts.run_compatibility import read_run_compatibility_identity


PATIENT_PROCESS_RUN_PLAN_SCHEMA_VERSION = "patient_process_run_plan_v3"
PATIENT_WORKER_JOB_SCHEMA_VERSION = "patient_worker_job_v3"
PATIENT_WORKER_RESULT_SCHEMA_VERSION = "patient_worker_result_v2"
LEGACY_PATIENT_WORKER_JOB_SCHEMA_VERSIONS = frozenset({"patient_worker_job_v1", "patient_worker_job_v2"})
DEFAULT_PATIENT_PROCESS_RUNNER_DIR_NAME = "patient_process_runner"
STANDALONE_PATIENT_RUNNER_JOB_NAME = "standalone_patient_runner"
PATIENT_PROCESS_REQUESTED_JOB_NAMES = frozenset(
    {
        STANDALONE_PATIENT_RUNNER_JOB_NAME,
        "legacy_oracle",
        "post_run_assembly",
        "validation",
    }
)
PATIENT_ARTIFACT_RETENTION_LEVELS = frozenset({"minimal", "context", "diagnostic", "full_debug"})
PATIENT_PROCESS_EXECUTION_MODES = frozenset({"plan_only", "dry_run_workers", "live_workers"})
SCIENTIFIC_CONFIG_SNAPSHOT_FINGERPRINT_METADATA_KEY = "scientific_config_snapshot_fingerprint_sha256"
SCIENTIFIC_CONFIG_SNAPSHOT_FILE_SHA256_METADATA_KEY = "scientific_config_snapshot_file_sha256"
RUN_COMPATIBILITY_IDENTITY_FILE_SHA256_METADATA_KEY = "run_compatibility_identity_file_sha256"


class PatientProcessFailurePolicy(str, Enum):
    """Failure policies for the standalone parent orchestrator."""

    STOP_ON_FAILURE = "stop_on_failure"
    CONTINUE_ON_FAILURE = "continue_on_failure"


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


def _read_json_object(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_object(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(_json_safe(payload), file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")
    return path


def _non_empty_string(value: Any, field_name: str) -> str:
    resolved_value = str(value).strip()
    if resolved_value == "":
        raise ValueError(f"{field_name} cannot be empty")
    return resolved_value


def _optional_path(value: Any) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return Path(str(value))


def _normalize_requested_jobs(values: Sequence[str]) -> tuple[str, ...]:
    requested_jobs = tuple(_non_empty_string(value, "requested_jobs item") for value in values)
    if len(requested_jobs) == 0:
        raise ValueError("requested_jobs cannot be empty")
    if len(set(requested_jobs)) != len(requested_jobs):
        raise ValueError("requested_jobs cannot contain duplicates")
    unsupported_jobs = sorted(set(requested_jobs).difference(PATIENT_PROCESS_REQUESTED_JOB_NAMES))
    if unsupported_jobs:
        raise ValueError("unsupported requested_jobs: {}".format(unsupported_jobs))
    return requested_jobs


def _normalize_retention_level(value: str) -> str:
    retention_level = _non_empty_string(value, "retention_level").lower()
    if retention_level not in PATIENT_ARTIFACT_RETENTION_LEVELS:
        raise ValueError(
            "retention_level must be one of: {}".format(", ".join(sorted(PATIENT_ARTIFACT_RETENTION_LEVELS)))
        )
    return retention_level


def _normalize_execution_mode(value: str) -> str:
    execution_mode = _non_empty_string(value, "execution_mode").lower()
    if execution_mode not in PATIENT_PROCESS_EXECUTION_MODES:
        raise ValueError(
            "execution_mode must be one of: {}".format(", ".join(sorted(PATIENT_PROCESS_EXECUTION_MODES)))
        )
    return execution_mode


def _case_row_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _patient_cases_and_inputs_from_manifest(
    input_case_manifest_path: Path,
    patient_uids: Sequence[str] = (),
) -> tuple[tuple[PatientCase, PatientInputPaths], ...]:
    requested_patient_uids = validate_patient_uids(patient_uids, "patient_uids")
    requested_set = set(requested_patient_uids)
    rows_by_patient_uid: dict[str, Mapping[str, str]] = {}
    with Path(input_case_manifest_path).open("r", encoding="utf-8", newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            patient_uid = str(row.get("Patient UID (generated)", ""))
            if patient_uid.strip() == "":
                continue
            if patient_uid in rows_by_patient_uid:
                raise ValueError("input case manifest contains duplicate patient UID: {!r}".format(patient_uid))
            rows_by_patient_uid[patient_uid] = dict(row)

    if requested_patient_uids:
        missing_patient_uids = tuple(
            patient_uid for patient_uid in requested_patient_uids if patient_uid not in rows_by_patient_uid
        )
        if missing_patient_uids:
            raise KeyError(f"patient_uids not found in input case manifest: {missing_patient_uids}")
        ordered_patient_uids = requested_patient_uids
    else:
        ordered_patient_uids = tuple(rows_by_patient_uid.keys())

    patient_cases_and_inputs: list[tuple[PatientCase, PatientInputPaths]] = []
    for patient_uid in ordered_patient_uids:
        if requested_set and patient_uid not in requested_set:
            continue
        row = rows_by_patient_uid[patient_uid]
        patient_inputs = PatientInputPaths.from_case_manifest_row(row)
        patient_cases_and_inputs.append(
            (
                PatientCase(
                patient_uid=patient_uid,
                patient_label=patient_uid,
                input_manifest_id=Path(input_case_manifest_path).as_posix(),
                metadata={
                    "patient_name": row.get("Patient Name", ""),
                    "patient_id_from_dicom": row.get("Patient ID (from dicom)", ""),
                    "fraction_number_legacy_parsed": row.get("Fraction number (legacy parsed)", ""),
                    "core_rt_complete": _case_row_bool(row.get("Core RTSTRUCT/RTDOSE/RTPLAN complete", False)),
                    "has_rtstruct": _case_row_bool(row.get("Has RTSTRUCT", False)),
                    "has_rtdose": _case_row_bool(row.get("Has RTDOSE", False)),
                    "has_rtplan": _case_row_bool(row.get("Has RTPLAN", False)),
                    "num_us_files": row.get("Num US files", ""),
                    "num_mr_t2_files": row.get("Num MR T2 files", ""),
                    "num_mr_adc_files": row.get("Num MR ADC files", ""),
                    "patient_input_manifest_identity_sha256": patient_inputs.manifest_identity_sha256,
                },
                ),
                patient_inputs,
            )
        )
    return tuple(patient_cases_and_inputs)


@dataclass(frozen=True, slots=True)
class PatientWorkerJob:
    """Serializable job packet for one patient worker process."""

    job_id: str
    patient_case: PatientCase
    patient_inputs: PatientInputPaths
    input_case_manifest_path: Path
    output_root: Path
    pathway_name: str
    checkpoint_name: str
    attempt_number: int = 1
    run_id: str = ""
    scientific_config_snapshot_path: Path | None = None
    run_compatibility_identity_path: Path | None = None
    retention_level: str = "minimal"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.patient_case, PatientCase):
            raise TypeError("patient_case must be a PatientCase instance")
        if not isinstance(self.patient_inputs, PatientInputPaths):
            raise TypeError("patient_inputs must be a PatientInputPaths instance")
        if self.patient_inputs.patient_uid != self.patient_case.patient_uid:
            raise ValueError("patient_inputs.patient_uid must match patient_case.patient_uid")
        object.__setattr__(self, "job_id", _non_empty_string(self.job_id, "job_id"))
        object.__setattr__(self, "input_case_manifest_path", Path(self.input_case_manifest_path))
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "pathway_name", _non_empty_string(self.pathway_name, "pathway_name"))
        object.__setattr__(self, "checkpoint_name", _non_empty_string(self.checkpoint_name, "checkpoint_name"))
        attempt_number = int(self.attempt_number)
        if attempt_number < 1:
            raise ValueError("attempt_number must be at least 1")
        object.__setattr__(self, "attempt_number", attempt_number)
        object.__setattr__(self, "run_id", str(self.run_id).strip())
        object.__setattr__(self, "scientific_config_snapshot_path", _optional_path(self.scientific_config_snapshot_path))
        object.__setattr__(self, "run_compatibility_identity_path", _optional_path(self.run_compatibility_identity_path))
        object.__setattr__(self, "retention_level", _normalize_retention_level(self.retention_level))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def patient_output_root(self) -> Path:
        return self.output_root.joinpath("patients", self.patient_case.safe_patient_uid)

    @property
    def result_path(self) -> Path:
        return self.output_root.joinpath("worker_results", f"{self.job_id}_attempt_{self.attempt_number}.json")

    @property
    def job_path(self) -> Path:
        return self.output_root.joinpath("worker_jobs", f"{self.job_id}.json")

    def as_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": PATIENT_WORKER_JOB_SCHEMA_VERSION,
            "job_id": self.job_id,
            "attempt_number": self.attempt_number,
            "run_id": self.run_id,
            "patient_case": {
                "patient_uid": self.patient_case.patient_uid,
                "patient_label": self.patient_case.patient_label,
                "source_run_id": self.patient_case.source_run_id,
                "input_manifest_id": self.patient_case.input_manifest_id,
                "metadata": dict(self.patient_case.metadata),
            },
            "patient_inputs": self.patient_inputs.to_dict(),
            "input_case_manifest_path": self.input_case_manifest_path.as_posix(),
            "output_root": self.output_root.as_posix(),
            "patient_output_root": self.patient_output_root.as_posix(),
            "pathway_name": self.pathway_name,
            "checkpoint_name": self.checkpoint_name,
            "scientific_config_snapshot_path": (
                None
                if self.scientific_config_snapshot_path is None
                else self.scientific_config_snapshot_path.as_posix()
            ),
            "run_compatibility_identity_path": (
                None
                if self.run_compatibility_identity_path is None
                else self.run_compatibility_identity_path.as_posix()
            ),
            "retention_level": self.retention_level,
            "result_path": self.result_path.as_posix(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PatientWorkerJob":
        schema_version = payload.get("schema_version")
        supported_schema_versions = {PATIENT_WORKER_JOB_SCHEMA_VERSION, *LEGACY_PATIENT_WORKER_JOB_SCHEMA_VERSIONS}
        if schema_version not in supported_schema_versions:
            raise ValueError(
                f"Unsupported worker job schema_version {schema_version!r}; "
                f"expected one of {sorted(supported_schema_versions)!r}"
            )
        patient_case_payload = payload.get("patient_case", {})
        if not isinstance(patient_case_payload, Mapping):
            raise TypeError("patient_case must be an object")
        patient_uid = str(patient_case_payload.get("patient_uid", ""))
        patient_metadata = patient_case_payload.get("metadata", {})
        patient_inputs_payload = payload.get("patient_inputs")
        if isinstance(patient_inputs_payload, Mapping):
            patient_inputs = PatientInputPaths.from_dict(patient_inputs_payload)
        else:
            legacy_core_paths = patient_metadata.get("core_input_paths", {}) if isinstance(patient_metadata, Mapping) else {}
            patient_inputs = PatientInputPaths(
                patient_uid=patient_uid,
                rtstruct=_optional_path(legacy_core_paths.get("rtstruct")) if isinstance(legacy_core_paths, Mapping) else None,
                rtdose=_optional_path(legacy_core_paths.get("rtdose")) if isinstance(legacy_core_paths, Mapping) else None,
                rtplan=_optional_path(legacy_core_paths.get("rtplan")) if isinstance(legacy_core_paths, Mapping) else None,
            )
        return cls(
            job_id=str(payload.get("job_id", "")),
            attempt_number=int(payload.get("attempt_number", 1)),
            run_id=str(payload.get("run_id", "")),
            patient_case=PatientCase(
                patient_uid=patient_uid,
                patient_label=str(patient_case_payload.get("patient_label", "")),
                source_run_id=str(patient_case_payload.get("source_run_id", "")),
                input_manifest_id=str(patient_case_payload.get("input_manifest_id", "")),
                metadata=patient_metadata,
            ),
            patient_inputs=patient_inputs,
            input_case_manifest_path=Path(str(payload.get("input_case_manifest_path", ""))),
            output_root=Path(str(payload.get("output_root", ""))),
            pathway_name=str(payload.get("pathway_name", "")),
            checkpoint_name=str(payload.get("checkpoint_name", "")),
            scientific_config_snapshot_path=_optional_path(payload.get("scientific_config_snapshot_path")),
            run_compatibility_identity_path=_optional_path(payload.get("run_compatibility_identity_path")),
            retention_level=str(payload.get("retention_level", "minimal")),
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class PatientProcessRunPlan:
    """Parent-orchestrator plan for standalone patient-worker execution."""

    output_root: Path
    input_case_manifest_path: Path
    worker_jobs: tuple[PatientWorkerJob, ...]
    pathway_name: str
    checkpoint_name: str
    run_id: str = "patient-process-runner"
    failure_policy: PatientProcessFailurePolicy | str = PatientProcessFailurePolicy.STOP_ON_FAILURE
    max_workers: int = 1
    timeout_seconds: float | None = None
    execution_mode: str = "plan_only"
    requested_jobs: Sequence[str] = (STANDALONE_PATIENT_RUNNER_JOB_NAME,)
    scientific_config_snapshot_path: Path | None = None
    run_compatibility_identity_path: Path | None = None
    retention_level: str = "minimal"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "input_case_manifest_path", Path(self.input_case_manifest_path))
        object.__setattr__(self, "worker_jobs", tuple(self.worker_jobs))
        if any(not isinstance(worker_job, PatientWorkerJob) for worker_job in self.worker_jobs):
            raise TypeError("worker_jobs entries must be PatientWorkerJob instances")
        object.__setattr__(self, "pathway_name", _non_empty_string(self.pathway_name, "pathway_name"))
        object.__setattr__(self, "checkpoint_name", _non_empty_string(self.checkpoint_name, "checkpoint_name"))
        object.__setattr__(self, "run_id", _non_empty_string(self.run_id, "run_id"))
        object.__setattr__(self, "failure_policy", PatientProcessFailurePolicy(self.failure_policy))
        max_workers = int(self.max_workers)
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        object.__setattr__(self, "max_workers", max_workers)
        timeout_seconds = self.timeout_seconds
        if timeout_seconds is not None:
            timeout_seconds = float(timeout_seconds)
            if timeout_seconds <= 0:
                raise ValueError("timeout_seconds must be positive when provided")
        object.__setattr__(self, "timeout_seconds", timeout_seconds)
        object.__setattr__(self, "execution_mode", _normalize_execution_mode(self.execution_mode))
        if self.execution_mode == "live_workers" and (
            self.pathway_name != "anatomical_qa" or self.checkpoint_name != "anatomical_qa"
        ):
            raise ValueError("live_workers currently supports anatomical_qa pathway/checkpoint only")
        object.__setattr__(self, "requested_jobs", _normalize_requested_jobs(self.requested_jobs))
        object.__setattr__(self, "scientific_config_snapshot_path", _optional_path(self.scientific_config_snapshot_path))
        object.__setattr__(self, "run_compatibility_identity_path", _optional_path(self.run_compatibility_identity_path))
        if self.execution_mode == "live_workers" and self.scientific_config_snapshot_path is None:
            raise ValueError("live_workers requires a scientific config snapshot")
        if self.execution_mode == "live_workers" and self.run_compatibility_identity_path is None:
            raise ValueError("live_workers requires a run compatibility identity")
        object.__setattr__(self, "retention_level", _normalize_retention_level(self.retention_level))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def plan_path(self) -> Path:
        return self.output_root.joinpath("patient_process_run_plan.json")

    def as_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": PATIENT_PROCESS_RUN_PLAN_SCHEMA_VERSION,
            "generated_utc": _utc_now_iso(),
            "runner_boundary": "standalone_patient_process_runner",
            "execution_policy": "parent_writes_worker_jobs_and_launches_patient_processes",
            "run_id": self.run_id,
            "output_root": self.output_root.as_posix(),
            "input_case_manifest_path": self.input_case_manifest_path.as_posix(),
            "pathway_name": self.pathway_name,
            "checkpoint_name": self.checkpoint_name,
            "failure_policy": self.failure_policy.value,
            "max_workers": self.max_workers,
            "timeout_seconds": self.timeout_seconds,
            "execution_mode": self.execution_mode,
            "requested_jobs": list(self.requested_jobs),
            "scientific_config_snapshot_path": (
                None
                if self.scientific_config_snapshot_path is None
                else self.scientific_config_snapshot_path.as_posix()
            ),
            "run_compatibility_identity_path": (
                None
                if self.run_compatibility_identity_path is None
                else self.run_compatibility_identity_path.as_posix()
            ),
            "retention_level": self.retention_level,
            "patient_count": len(self.worker_jobs),
            "patient_uids": [worker_job.patient_case.patient_uid for worker_job in self.worker_jobs],
            "worker_job_paths": [worker_job.job_path.as_posix() for worker_job in self.worker_jobs],
            "worker_commands": [
                patient_worker_command(
                    worker_job.job_path,
                    dry_run=self.execution_mode == "dry_run_workers",
                )
                for worker_job in self.worker_jobs
            ],
            "worker_jobs": [worker_job.as_mapping() for worker_job in self.worker_jobs],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class PatientWorkerResult:
    """Serializable result returned by one patient worker process."""

    worker_job: PatientWorkerJob
    status: PatientStageStatus | str
    elapsed_seconds: float
    exit_code: int
    dry_run: bool = False
    timed_out: bool = False
    warnings: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.worker_job, PatientWorkerJob):
            raise TypeError("worker_job must be a PatientWorkerJob instance")
        object.__setattr__(self, "status", PatientStageStatus(self.status))
        object.__setattr__(self, "elapsed_seconds", float(self.elapsed_seconds))
        object.__setattr__(self, "exit_code", int(self.exit_code))
        object.__setattr__(self, "dry_run", bool(self.dry_run))
        object.__setattr__(self, "timed_out", bool(self.timed_out))
        object.__setattr__(self, "warnings", tuple(str(warning) for warning in self.warnings))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0 and self.status in {PatientStageStatus.SUCCEEDED, PatientStageStatus.SKIPPED}

    def as_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": PATIENT_WORKER_RESULT_SCHEMA_VERSION,
            "generated_utc": _utc_now_iso(),
            "job": self.worker_job.as_mapping(),
            "patient_uid": self.worker_job.patient_case.patient_uid,
            "status": self.status.value,
            "succeeded": self.succeeded,
            "exit_code": self.exit_code,
            "dry_run": self.dry_run,
            "timed_out": self.timed_out,
            "elapsed_seconds": self.elapsed_seconds,
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


def build_patient_process_run_plan(
    *,
    input_case_manifest_path: Path,
    output_root: Path,
    pathway_name: str,
    checkpoint_name: str,
    patient_uids: Sequence[str] = (),
    run_id: str = "patient-process-runner",
    failure_policy: PatientProcessFailurePolicy | str = PatientProcessFailurePolicy.STOP_ON_FAILURE,
    max_workers: int = 1,
    timeout_seconds: float | None = None,
    execution_mode: str = "plan_only",
    requested_jobs: Sequence[str] = (STANDALONE_PATIENT_RUNNER_JOB_NAME,),
    scientific_config_snapshot_path: Path | None = None,
    run_compatibility_identity_path: Path | None = None,
    retention_level: str = "minimal",
    capture_input_content: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> PatientProcessRunPlan:
    """Build a lightweight process plan with optional patient byte identities.

    ``capture_input_content`` streams every declared file before execution and
    binds its role/path/bytes to the job. It decodes no DICOM and changes no
    scientific defaults. Missing files or hashing failures reject planning.
    """
    if not isinstance(capture_input_content, bool):
        raise TypeError("capture_input_content must be boolean")
    resolved_output_root = Path(output_root).expanduser().resolve()
    resolved_metadata = dict(metadata or {})
    resolved_snapshot_path = _optional_path(scientific_config_snapshot_path)
    resolved_compatibility_path = _optional_path(run_compatibility_identity_path)
    resolved_execution_mode = _normalize_execution_mode(execution_mode)
    if resolved_execution_mode == "live_workers":
        if pathway_name != "anatomical_qa" or checkpoint_name != "anatomical_qa":
            raise ValueError("live_workers currently supports anatomical_qa pathway/checkpoint only")
        if resolved_snapshot_path is None or not resolved_snapshot_path.is_file():
            raise FileNotFoundError("live_workers scientific config snapshot does not exist: {}".format(resolved_snapshot_path))
        if resolved_compatibility_path is None or not resolved_compatibility_path.is_file():
            raise FileNotFoundError(
                "live_workers run compatibility identity does not exist: {}".format(resolved_compatibility_path)
            )
    resolved_snapshot = None
    if resolved_snapshot_path is not None and resolved_snapshot_path.is_file():
        resolved_snapshot = read_pipeline_config_snapshot(resolved_snapshot_path)
        resolved_metadata.update(
            {
                SCIENTIFIC_CONFIG_SNAPSHOT_FINGERPRINT_METADATA_KEY: resolved_snapshot.config_sha256,
                SCIENTIFIC_CONFIG_SNAPSHOT_FILE_SHA256_METADATA_KEY: _sha256_file(resolved_snapshot_path),
            }
        )
    if resolved_compatibility_path is not None and resolved_compatibility_path.is_file():
        compatibility_identity = read_run_compatibility_identity(resolved_compatibility_path)
        if resolved_snapshot is None:
            raise ValueError("run compatibility identity requires an available scientific config snapshot")
        if compatibility_identity.scientific_config_sha256 != resolved_snapshot.config_sha256:
            raise ValueError("run compatibility identity scientific config SHA does not match snapshot")
        resolved_metadata.update(
            {
                RUN_COMPATIBILITY_METADATA_KEY: compatibility_identity.to_dict(),
                RUN_COMPATIBILITY_IDENTITY_FILE_SHA256_METADATA_KEY: _sha256_file(resolved_compatibility_path),
            }
        )
    resolved_metadata.update(
        {
            "pathway_name": pathway_name,
            "checkpoint_name": checkpoint_name,
            "planned_stage_names": [
                "grid_preprocessing",
                "anatomical_preprocessing",
            ] if pathway_name == "anatomical_qa" else [],
        }
    )
    if resolved_snapshot is not None:
        random_seed_config = resolved_snapshot.config.get("random_seeds", {})
        if isinstance(random_seed_config, Mapping):
            from random_seed_policy import random_seed_policy_metadata

            resolved_metadata["random_seed_policy"] = random_seed_policy_metadata(
                transform_generation_random_seed=random_seed_config.get("transform_generation_random_seed"),
                optimizer_v1_random_seed=random_seed_config.get("optimizer_v1_random_seed"),
            )
    patient_cases_and_inputs = _patient_cases_and_inputs_from_manifest(Path(input_case_manifest_path), patient_uids)
    worker_jobs = tuple(
        PatientWorkerJob(
            job_id=f"patient_{index:04d}_{patient_case.safe_patient_uid}",
            patient_case=patient_case,
            patient_inputs=patient_inputs,
            input_case_manifest_path=Path(input_case_manifest_path),
            output_root=resolved_output_root,
            pathway_name=pathway_name,
            checkpoint_name=checkpoint_name,
            run_id=run_id,
            scientific_config_snapshot_path=resolved_snapshot_path,
            run_compatibility_identity_path=resolved_compatibility_path,
            retention_level=retention_level,
            metadata={**resolved_metadata, "patient_index": index},
        )
        for index, (patient_case, patient_inputs) in enumerate(patient_cases_and_inputs, start=1)
    )
    if capture_input_content:
        from dataclasses import replace
        from input_data.content_identity import capture_patient_input_content, INPUT_CONTENT_KEY

        worker_jobs = tuple(replace(job, metadata={
            **job.metadata, INPUT_CONTENT_KEY: capture_patient_input_content(job.patient_inputs),
        }) for job in worker_jobs)
    return PatientProcessRunPlan(
        output_root=resolved_output_root,
        input_case_manifest_path=Path(input_case_manifest_path),
        worker_jobs=worker_jobs,
        pathway_name=pathway_name,
        checkpoint_name=checkpoint_name,
        run_id=run_id,
        failure_policy=failure_policy,
        max_workers=max_workers,
        timeout_seconds=timeout_seconds,
        execution_mode=resolved_execution_mode,
        requested_jobs=requested_jobs,
        scientific_config_snapshot_path=resolved_snapshot_path,
        run_compatibility_identity_path=resolved_compatibility_path,
        retention_level=retention_level,
        metadata=resolved_metadata,
    )


def write_patient_worker_job_packets(plan: PatientProcessRunPlan) -> tuple[Path, ...]:
    """Write one JSON job packet per patient worker."""
    return tuple(_write_json_object(job.job_path, job.as_mapping()) for job in plan.worker_jobs)


def write_patient_process_run_plan(plan: PatientProcessRunPlan) -> Path:
    """Write the parent plan JSON for a standalone process run."""
    return _write_json_object(plan.plan_path, plan.as_mapping())


def load_patient_worker_job(job_path: Path) -> PatientWorkerJob:
    """Load one worker job packet."""
    return PatientWorkerJob.from_mapping(_read_json_object(Path(job_path)))


def write_patient_worker_result(result: PatientWorkerResult, output_path: Path | None = None) -> Path:
    """Write one worker result JSON object."""
    resolved_output_path = Path(output_path) if output_path is not None else result.worker_job.result_path
    return _write_json_object(resolved_output_path, result.as_mapping())


def _worker_setup_failure_result(
    job: PatientWorkerJob,
    *,
    start_time: float,
    exit_code: int,
    warning: str,
    failed_boundary: str,
    input_preflight_metadata: Mapping[str, Any],
) -> PatientWorkerResult:
    elapsed_seconds = perf_counter() - start_time
    warnings = [warning]
    metadata: dict[str, Any] = {
        **job.metadata,
        "worker_boundary": "standalone_patient_process_runner",
        "failed_boundary": failed_boundary,
        "input_preflight": dict(input_preflight_metadata),
    }
    try:
        from .contracts import PatientRunResult
        from .contracts import PatientStageName
        from .contracts import PatientStageResult
        from .manifests import write_patient_run_manifest

        stage_result = PatientStageResult.failure(
            PatientStageName.LEGACY_BRIDGE,
            elapsed_seconds=elapsed_seconds,
            warnings=(warning,),
            metadata={"patient_uid": job.patient_case.patient_uid, "failed_boundary": failed_boundary},
        )
        patient_result = PatientRunResult.from_stage_results(
            job.patient_case,
            job.patient_output_root,
            (stage_result,),
            elapsed_seconds=elapsed_seconds,
            metadata=metadata,
        )
        metadata["patient_run_manifest_path"] = write_patient_run_manifest(patient_result).as_posix()
    except Exception as exc:
        warnings.append("failed to write patient setup-failure manifest: {}".format(exc))
        metadata["patient_run_manifest_error"] = str(exc)
    return PatientWorkerResult(
        worker_job=job,
        status=PatientStageStatus.FAILED,
        elapsed_seconds=elapsed_seconds,
        exit_code=exit_code,
        dry_run=False,
        warnings=tuple(warnings),
        metadata=metadata,
    )


def _validate_worker_compatibility_identity(
    job: PatientWorkerJob,
    *,
    scientific_config_sha256: str,
) -> dict[str, Any]:
    if job.run_compatibility_identity_path is None or not job.run_compatibility_identity_path.is_file():
        raise FileNotFoundError("standalone live worker requires a run compatibility identity")
    compatibility_identity = read_run_compatibility_identity(job.run_compatibility_identity_path)
    expected_payload = job.metadata.get(RUN_COMPATIBILITY_METADATA_KEY)
    if not isinstance(expected_payload, Mapping):
        raise ValueError("worker job is missing embedded run compatibility identity")
    expected_identity = type(compatibility_identity).from_dict(expected_payload)
    if compatibility_identity != expected_identity:
        raise ValueError("run compatibility identity file differs from planned worker identity")
    expected_file_sha256 = str(
        job.metadata.get(RUN_COMPATIBILITY_IDENTITY_FILE_SHA256_METADATA_KEY, "")
    )
    current_file_sha256 = _sha256_file(job.run_compatibility_identity_path)
    if expected_file_sha256 == "" or current_file_sha256 != expected_file_sha256:
        raise ValueError("run compatibility identity file SHA differs from planned worker identity")
    if compatibility_identity.scientific_config_sha256 != scientific_config_sha256:
        raise ValueError("run compatibility identity scientific config SHA differs from worker snapshot")

    from output_artifacts.schema_registry import OUTPUT_SCHEMA_REGISTRY_VERSION
    from startup.code_identity import capture_code_identity
    from startup.runtime_environment import capture_runtime_environment_identity

    repository_root = Path(__file__).resolve().parents[2]
    current_code_identity = capture_code_identity(repository_root)
    current_environment_identity = capture_runtime_environment_identity(repository_root)
    mismatches = {}
    for field_name, current_value in (
        ("code_source_sha256", current_code_identity.source_tree_sha256),
        ("runtime_environment_sha256", current_environment_identity.identity_sha256),
        ("output_schema_registry_version", OUTPUT_SCHEMA_REGISTRY_VERSION),
    ):
        expected_value = str(getattr(compatibility_identity, field_name))
        if str(current_value) != expected_value:
            mismatches[field_name] = {"planned": expected_value, "worker": str(current_value)}
    if mismatches:
        raise ValueError("worker runtime differs from run compatibility identity: {}".format(mismatches))
    return {
        "run_compatibility_identity_path": job.run_compatibility_identity_path.as_posix(),
        RUN_COMPATIBILITY_IDENTITY_FILE_SHA256_METADATA_KEY: current_file_sha256,
        "run_compatibility_identity_sha256": compatibility_identity.identity_sha256,
        "code_source_sha256": current_code_identity.source_tree_sha256,
        "runtime_environment_sha256": current_environment_identity.identity_sha256,
        "output_schema_registry_version": OUTPUT_SCHEMA_REGISTRY_VERSION,
    }


def run_patient_worker_job(job: PatientWorkerJob, *, dry_run: bool = False, runtime_builder=None) -> PatientWorkerResult:
    """Run one patient worker job.

    Live execution is currently gated to the first ``anatomical_qa`` checkpoint.
    Later pathways fail closed until their standalone input/resource boundaries
    have independent parity evidence. ``runtime_builder`` is a Python-only
    validation injection; serialized jobs and the normal worker CLI cannot select
    executable code. The default always constructs fresh standalone input state.
    """
    start_time = perf_counter()
    missing_core_input_roles = job.patient_inputs.missing_core_roles
    config_snapshot_available = (
        job.scientific_config_snapshot_path is not None
        and job.scientific_config_snapshot_path.is_file()
    )
    compatibility_identity_available = (
        job.run_compatibility_identity_path is not None
        and job.run_compatibility_identity_path.is_file()
    )
    input_preflight_metadata = {
        "core_input_paths_all_present": not missing_core_input_roles,
        "missing_core_input_roles": missing_core_input_roles,
        "scientific_config_snapshot_path": (
            ""
            if job.scientific_config_snapshot_path is None
            else job.scientific_config_snapshot_path.as_posix()
        ),
        "scientific_config_snapshot_available": config_snapshot_available,
        "run_compatibility_identity_path": (
            ""
            if job.run_compatibility_identity_path is None
            else job.run_compatibility_identity_path.as_posix()
        ),
        "run_compatibility_identity_available": compatibility_identity_available,
    }
    if dry_run:
        warnings = ["dry-run worker did not build runtime state or execute scientific stages"]
        if missing_core_input_roles:
            warnings.append(
                "input preflight found missing core input files: " + ", ".join(missing_core_input_roles)
            )
        if not config_snapshot_available:
            warnings.append("scientific config snapshot is not available")
        if not compatibility_identity_available:
            warnings.append("run compatibility identity is not available")
        return PatientWorkerResult(
            worker_job=job,
            status=PatientStageStatus.SKIPPED,
            elapsed_seconds=perf_counter() - start_time,
            exit_code=0,
            dry_run=True,
            warnings=tuple(warnings),
            metadata={
                "worker_boundary": "standalone_patient_process_runner",
                "input_preflight": input_preflight_metadata,
            },
        )

    if missing_core_input_roles:
        return _worker_setup_failure_result(
            job,
            start_time=start_time,
            exit_code=2,
            warning="input preflight found missing core input files: " + ", ".join(missing_core_input_roles),
            failed_boundary="core_input_path_preflight",
            input_preflight_metadata=input_preflight_metadata,
        )

    if not config_snapshot_available:
        return _worker_setup_failure_result(
            job,
            start_time=start_time,
            exit_code=2,
            warning="standalone live worker requires a scientific config snapshot",
            failed_boundary="scientific_config_snapshot_preflight",
            input_preflight_metadata=input_preflight_metadata,
        )

    try:
        config_snapshot = read_pipeline_config_snapshot(job.scientific_config_snapshot_path)
    except Exception as exc:
        return _worker_setup_failure_result(
            job,
            start_time=start_time,
            exit_code=2,
            warning="invalid scientific config snapshot: {}".format(exc),
            failed_boundary="scientific_config_snapshot_preflight",
            input_preflight_metadata=input_preflight_metadata,
        )
    current_snapshot_file_sha256 = _sha256_file(job.scientific_config_snapshot_path)
    expected_snapshot_fingerprint = str(
        job.metadata.get(SCIENTIFIC_CONFIG_SNAPSHOT_FINGERPRINT_METADATA_KEY, "")
    )
    expected_snapshot_file_sha256 = str(
        job.metadata.get(SCIENTIFIC_CONFIG_SNAPSHOT_FILE_SHA256_METADATA_KEY, "")
    )
    input_preflight_metadata.update(
        {
            "scientific_config_sha256": config_snapshot.config_sha256,
            SCIENTIFIC_CONFIG_SNAPSHOT_FINGERPRINT_METADATA_KEY: expected_snapshot_fingerprint,
            SCIENTIFIC_CONFIG_SNAPSHOT_FILE_SHA256_METADATA_KEY: expected_snapshot_file_sha256,
            "scientific_config_snapshot_current_file_sha256": current_snapshot_file_sha256,
        }
    )
    if (
        expected_snapshot_fingerprint == ""
        or expected_snapshot_file_sha256 == ""
        or config_snapshot.config_sha256 != expected_snapshot_fingerprint
        or current_snapshot_file_sha256 != expected_snapshot_file_sha256
    ):
        return _worker_setup_failure_result(
            job,
            start_time=start_time,
            exit_code=2,
            warning="scientific config snapshot differs from the planned worker identity",
            failed_boundary="scientific_config_snapshot_identity_preflight",
            input_preflight_metadata=input_preflight_metadata,
        )

    try:
        compatibility_preflight = _validate_worker_compatibility_identity(
            job,
            scientific_config_sha256=config_snapshot.config_sha256,
        )
        input_preflight_metadata.update(compatibility_preflight)
    except Exception as exc:
        return _worker_setup_failure_result(
            job,
            start_time=start_time,
            exit_code=2,
            warning="run compatibility preflight failed: {}".format(exc),
            failed_boundary="run_compatibility_identity_preflight",
            input_preflight_metadata=input_preflight_metadata,
        )

    if job.pathway_name != "anatomical_qa" or job.checkpoint_name != "anatomical_qa":
        return _worker_setup_failure_result(
            job,
            start_time=start_time,
            exit_code=2,
            warning="standalone live worker currently supports anatomical_qa only",
            failed_boundary="standalone_pathway_support",
            input_preflight_metadata=input_preflight_metadata,
        )

    try:
        from config.rehydration import rehydrate_pipeline_scientific_config_snapshot

        pipeline_config = rehydrate_pipeline_scientific_config_snapshot(config_snapshot)
    except Exception as exc:
        return _worker_setup_failure_result(
            job,
            start_time=start_time,
            exit_code=2,
            warning="scientific config rehydration failed: {}".format(exc),
            failed_boundary="scientific_config_rehydration",
            input_preflight_metadata=input_preflight_metadata,
        )

    runtime_metadata = {
        **job.metadata,
        "worker_boundary": "standalone_patient_process_runner",
        "worker_job_id": job.job_id,
        "run_id": job.run_id,
        "retention_level": job.retention_level,
        "scientific_config_sha256": config_snapshot.config_sha256,
        "patient_input_manifest_identity_sha256": job.patient_inputs.manifest_identity_sha256,
    }
    from input_data.content_identity import INPUT_CONTENT_KEY, verify_patient_input_content

    if INPUT_CONTENT_KEY in job.metadata:
        try:
            verify_patient_input_content(job.metadata[INPUT_CONTENT_KEY], job.patient_inputs)
            runtime_metadata["input_content_verified_before"] = True
        except Exception as exc:
            return _worker_setup_failure_result(
                job, start_time=start_time, exit_code=2,
                warning="input content preflight failed: {}".format(exc),
                failed_boundary="input_content_preflight", input_preflight_metadata=input_preflight_metadata,
            )
    try:
        from .runtime_builder import build_standalone_patient_runtime

        build_runtime = build_standalone_patient_runtime if runtime_builder is None else runtime_builder
        standalone_runtime = build_runtime(
            patient_case=job.patient_case,
            patient_inputs=job.patient_inputs,
            pipeline_config=pipeline_config,
            metadata=runtime_metadata,
        )
    except Exception as exc:
        return _worker_setup_failure_result(
            job,
            start_time=start_time,
            exit_code=2,
            warning="one-patient runtime build failed: {}".format(exc),
            failed_boundary="one_patient_runtime_state_builder",
            input_preflight_metadata=input_preflight_metadata,
        )

    try:
        from .runner import run_patient_case
        from .scientific_runner import build_patient_scientific_run_config_from_pipeline
        from .scientific_runner import build_patient_scientific_runner_stages

        scientific_run_config = build_patient_scientific_run_config_from_pipeline(
            pipeline_config,
            standalone_runtime.config_build_context,
            output_root=job.output_root,
            pathway_name=job.pathway_name,
            checkpoint_name=job.checkpoint_name,
            patient_uids=(job.patient_case.patient_uid,),
            run_id=job.run_id or job.job_id,
            max_workers=1,
            execution_backend="sequential",
            metadata=runtime_metadata,
        )
        standalone_runtime.runtime_state.metadata.update(
            {
                **scientific_run_config.batch_config.metadata,
                **scientific_run_config.metadata,
                "pathway_name": scientific_run_config.pathway_name.value,
                "planned_stage_names": tuple(
                    stage_name.value for stage_name in scientific_run_config.planned_stage_names
                ),
            }
        )
        stages = build_patient_scientific_runner_stages(scientific_run_config)
        capture_checkpoint = job.metadata.get("capture_anatomical_checkpoint", False)
        if not isinstance(capture_checkpoint, bool):
            raise TypeError("capture_anatomical_checkpoint must be a boolean")
        if capture_checkpoint:
            from validation.anatomical_execution import with_anatomical_checkpoint

            stages = with_anatomical_checkpoint(stages, pipeline_config)
        if INPUT_CONTENT_KEY in job.metadata:
            stages = _with_input_content_verification(stages, job)
        patient_result = run_patient_case(
            standalone_runtime.runtime_state,
            scientific_run_config.batch_config.patient_config,
            stages=stages,
        )
    except Exception as exc:
        return _worker_setup_failure_result(
            job,
            start_time=start_time,
            exit_code=1,
            warning="standalone scientific execution setup failed: {}".format(exc),
            failed_boundary="scientific_execution_setup",
            input_preflight_metadata=input_preflight_metadata,
        )

    stage_statuses = {
        stage_result.stage_name: stage_result.status.value
        for stage_result in patient_result.stage_results
    }
    stage_warnings = tuple(
        warning
        for stage_result in patient_result.stage_results
        for warning in stage_result.warnings
    )
    return PatientWorkerResult(
        worker_job=job,
        status=patient_result.status,
        elapsed_seconds=perf_counter() - start_time,
        exit_code=0 if patient_result.succeeded else 1,
        dry_run=False,
        warnings=stage_warnings,
        metadata={
            "worker_boundary": "standalone_patient_process_runner",
            "executed_boundary": "anatomical_qa",
            "input_preflight": input_preflight_metadata,
            "patient_output_root": patient_result.output_root.as_posix(),
            "patient_run_manifest_path": patient_result.output_root.joinpath("patient_run_manifest.json").as_posix(),
            "artifact_paths": tuple(path.as_posix() for path in patient_result.artifact_paths),
            "stage_statuses": stage_statuses,
        },
    )


def _with_input_content_verification(stages: tuple, job: PatientWorkerJob) -> tuple:
    """Verify immutable inputs after computation/export, before success is sealed."""
    from .runner import PatientStage
    from input_data.content_identity import INPUT_CONTENT_KEY, verify_patient_input_content

    if not stages or stages[-1].stage_name != "patient_artifact_writing":
        raise ValueError("input verification requires final patient artifact writing stage")
    original = stages[-1].runner

    def verify_after(runtime_state, config):
        result = original(runtime_state, config)
        if result.succeeded:
            verify_patient_input_content(job.metadata[INPUT_CONTENT_KEY], job.patient_inputs)
            runtime_state.metadata["input_content_verified_after"] = True
        return result

    return (*stages[:-1], PatientStage(stages[-1].stage_name, verify_after))


def run_worker_job_file(job_path: Path, *, dry_run: bool = False) -> PatientWorkerResult:
    """Load, run, and write one worker job file."""
    worker_job = load_patient_worker_job(job_path)
    result = run_patient_worker_job(worker_job, dry_run=dry_run)
    write_patient_worker_result(result)
    return result


def patient_worker_command(job_path: Path | str, *, dry_run: bool = False, worker_script_path: Path | None = None) -> list[str]:
    """Return the exact command used to launch one worker job."""
    command = [
        sys.executable,
        str(worker_script_path or (Path(__file__).resolve().parents[1] / "run_patient_scientific_worker.py")),
        str(job_path),
    ]
    if dry_run:
        command.append("--dry-run")
    return command


def launch_worker_job_file(job_path: Path, *, dry_run: bool = False, timeout_seconds: float | None = None, worker_script_path: Path | None = None, log_path: Path | None = None) -> PatientWorkerResult:
    """Launch one worker job in a subprocess and load its result JSON."""
    command = patient_worker_command(job_path, dry_run=dry_run, worker_script_path=worker_script_path)
    worker_job = load_patient_worker_job(job_path)
    launch_start_time = perf_counter()
    if worker_job.result_path.is_file():
        worker_job.result_path.unlink()
    try:
        with ExitStack() as resources:
            output_options = {}
            if log_path is not None:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_file = resources.enter_context(log_path.open("x", encoding="utf-8"))
                output_options = {"stdout": log_file, "stderr": subprocess.STDOUT}
            completed = subprocess.run(command, check=False, timeout=timeout_seconds, **output_options)
    except subprocess.TimeoutExpired:
        return PatientWorkerResult(
            worker_job=worker_job,
            status=PatientStageStatus.FAILED,
            elapsed_seconds=perf_counter() - launch_start_time,
            exit_code=124,
            dry_run=dry_run,
            timed_out=True,
            warnings=("worker exceeded timeout_seconds",),
            metadata={"worker_command": command, "timeout_seconds": timeout_seconds},
        )
    except OSError as exc:
        return PatientWorkerResult(
            worker_job=worker_job, status=PatientStageStatus.FAILED,
            elapsed_seconds=perf_counter() - launch_start_time, exit_code=1, dry_run=dry_run,
            warnings=("worker launch failed: {}".format(exc),),
            metadata={"worker_command": command, "failed_boundary": "worker_launch"},
        )
    if worker_job.result_path.is_file():
        try:
            result_payload = _read_json_object(worker_job.result_path)
            if result_payload.get("schema_version") != PATIENT_WORKER_RESULT_SCHEMA_VERSION:
                raise ValueError("worker result schema_version is unsupported")
            result_job_payload = result_payload.get("job", {})
            if not isinstance(result_job_payload, Mapping):
                raise TypeError("worker result job must be an object")
            result_worker_job = PatientWorkerJob.from_mapping(result_job_payload)
            if result_worker_job != worker_job:
                raise ValueError("worker result identity does not match launched job")
            result_exit_code = int(result_payload.get("exit_code", completed.returncode))
            if result_exit_code != int(completed.returncode):
                raise ValueError("worker result exit_code does not match subprocess exit code")
            status = PatientStageStatus(result_payload.get("status", PatientStageStatus.FAILED.value))
            elapsed_seconds = float(result_payload.get("elapsed_seconds", 0.0) or 0.0)
            warnings_payload = result_payload.get("warnings", ())
            if isinstance(warnings_payload, (str, bytes)) or not isinstance(warnings_payload, Sequence):
                raise TypeError("worker result warnings must be an array")
            metadata_payload = result_payload.get("metadata", {})
            if not isinstance(metadata_payload, Mapping):
                raise TypeError("worker result metadata must be an object")
            if str(result_payload.get("patient_uid", "")) != worker_job.patient_case.patient_uid:
                raise ValueError("worker result patient_uid does not match launched job")
            if bool(result_payload.get("dry_run", False)) != bool(dry_run):
                raise ValueError("worker result dry_run does not match launched command")
            expected_succeeded = result_exit_code == 0 and status in {
                PatientStageStatus.SUCCEEDED,
                PatientStageStatus.SKIPPED,
            }
            if bool(result_payload.get("succeeded", False)) != expected_succeeded:
                raise ValueError("worker result succeeded flag is inconsistent")
            warnings = tuple(str(warning) for warning in warnings_payload)
            metadata = dict(metadata_payload)
        except Exception as exc:
            status = PatientStageStatus.FAILED
            elapsed_seconds = perf_counter() - launch_start_time
            warnings = ("worker wrote an invalid result JSON: {}".format(exc),)
            metadata = {"failed_boundary": "worker_result_validation"}
            result_exit_code = int(completed.returncode) or 1
    else:
        status = PatientStageStatus.FAILED
        elapsed_seconds = 0.0
        warnings = ("worker did not write a result JSON",)
        metadata = {"failed_boundary": "worker_result_missing"}
        result_exit_code = int(completed.returncode) or 1
    return PatientWorkerResult(
        worker_job=worker_job,
        status=status,
        elapsed_seconds=elapsed_seconds,
        exit_code=result_exit_code,
        dry_run=dry_run,
        warnings=warnings,
        metadata={"worker_command": command, **metadata},
    )


def run_patient_process_plan(
    plan: PatientProcessRunPlan,
    *,
    dry_run_workers: bool = False,
    timeout_seconds: float | None = None,
) -> tuple[PatientWorkerResult, ...]:
    """Run a plan through sequential subprocess workers."""
    if STANDALONE_PATIENT_RUNNER_JOB_NAME not in plan.requested_jobs:
        raise ValueError("patient process plan does not request standalone_patient_runner")
    unsupported_executable_jobs = sorted(set(plan.requested_jobs).difference({STANDALONE_PATIENT_RUNNER_JOB_NAME}))
    if unsupported_executable_jobs:
        raise NotImplementedError(
            "patient process execution does not yet support requested jobs: {}".format(unsupported_executable_jobs)
        )
    if plan.execution_mode == "plan_only":
        raise ValueError("plan_only process plans cannot launch workers")
    if dry_run_workers != (plan.execution_mode == "dry_run_workers"):
        raise ValueError("dry_run_workers must agree with plan.execution_mode")
    patient_root = plan.output_root / "patients"
    from output_artifacts.manifest_index import default_run_manifest_index_path

    if default_run_manifest_index_path(plan.output_root).exists() or (plan.output_root / "patient_batch_run_manifest.json").exists() or (patient_root.exists() and any(patient_root.iterdir())):
        raise FileExistsError("process execution requires a fresh patient output root; resume is not implemented")
    safe_names = [job.patient_case.safe_patient_uid for job in plan.worker_jobs]
    if len(safe_names) != len(set(safe_names)):
        raise ValueError("patient UIDs collide after filesystem normalization")
    job_paths = write_patient_worker_job_packets(plan)
    write_patient_process_run_plan(plan)
    resolved_timeout_seconds = plan.timeout_seconds if timeout_seconds is None else timeout_seconds
    results: list[PatientWorkerResult] = []
    for job, job_path in zip(plan.worker_jobs, job_paths):
        result = launch_worker_job_file(
            job_path,
            dry_run=dry_run_workers,
            timeout_seconds=resolved_timeout_seconds,
            log_path=plan.output_root / "worker_logs" / f"{job.job_id}_attempt_{job.attempt_number}.log",
        )
        results.append(result)
        if not result.succeeded and plan.failure_policy == PatientProcessFailurePolicy.STOP_ON_FAILURE:
            break
    from .process_finalization import finalize_patient_process_run

    return finalize_patient_process_run(plan, tuple(results)).worker_results
