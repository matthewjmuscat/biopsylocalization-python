"""Standalone parent/worker scaffold for patient-scientific execution."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any, Mapping, Sequence

from .contracts import PatientCase
from .contracts import PatientStageStatus
from .contracts import validate_patient_uids


PATIENT_PROCESS_RUN_PLAN_SCHEMA_VERSION = "patient_process_run_plan_v1"
PATIENT_WORKER_JOB_SCHEMA_VERSION = "patient_worker_job_v1"
PATIENT_WORKER_RESULT_SCHEMA_VERSION = "patient_worker_result_v1"
DEFAULT_PATIENT_PROCESS_RUNNER_DIR_NAME = "patient_process_runner"


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


def _case_row_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _case_row_path(value: Any) -> str:
    return str(value or "").strip()


def _case_row_path_exists(value: Any) -> bool:
    path_text = _case_row_path(value)
    return bool(path_text) and Path(path_text).expanduser().is_file()


def _core_input_path_metadata(row: Mapping[str, str]) -> dict[str, Any]:
    core_input_paths = {
        "rtstruct": _case_row_path(row.get("RTSTRUCT path")),
        "rtdose": _case_row_path(row.get("RTDOSE path")),
        "rtplan": _case_row_path(row.get("RTPLAN path")),
    }
    core_input_paths_exist = {
        role: _case_row_path_exists(path)
        for role, path in core_input_paths.items()
    }
    return {
        "core_input_paths": core_input_paths,
        "core_input_paths_exist": core_input_paths_exist,
        "core_input_paths_all_present": all(core_input_paths_exist.values()),
    }


def _missing_core_input_roles(patient_case: PatientCase) -> tuple[str, ...]:
    if "core_input_paths_exist" not in patient_case.metadata:
        return ()
    paths_exist = patient_case.metadata.get("core_input_paths_exist", {})
    if not isinstance(paths_exist, Mapping):
        return ()
    return tuple(
        role
        for role in ("rtstruct", "rtdose", "rtplan")
        if not bool(paths_exist.get(role, False))
    )


def _patient_cases_from_input_case_manifest(
    input_case_manifest_path: Path,
    patient_uids: Sequence[str] = (),
) -> tuple[PatientCase, ...]:
    requested_patient_uids = validate_patient_uids(patient_uids, "patient_uids")
    requested_set = set(requested_patient_uids)
    rows_by_patient_uid: dict[str, Mapping[str, str]] = {}
    with Path(input_case_manifest_path).open("r", encoding="utf-8", newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            patient_uid = str(row.get("Patient UID (generated)", "")).strip()
            if patient_uid == "":
                continue
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

    patient_cases: list[PatientCase] = []
    for patient_uid in ordered_patient_uids:
        if requested_set and patient_uid not in requested_set:
            continue
        row = rows_by_patient_uid[patient_uid]
        patient_cases.append(
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
                    **_core_input_path_metadata(row),
                },
            )
        )
    return tuple(patient_cases)


@dataclass(frozen=True, slots=True)
class PatientWorkerJob:
    """Serializable job packet for one patient worker process."""

    job_id: str
    patient_case: PatientCase
    input_case_manifest_path: Path
    output_root: Path
    pathway_name: str
    checkpoint_name: str
    attempt_number: int = 1
    run_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.patient_case, PatientCase):
            raise TypeError("patient_case must be a PatientCase instance")
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
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def patient_output_root(self) -> Path:
        return self.output_root.joinpath("patients", self.patient_case.safe_patient_uid)

    @property
    def result_path(self) -> Path:
        return self.output_root.joinpath("worker_results", f"{self.job_id}_attempt_{self.attempt_number}.json")

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
            "input_case_manifest_path": self.input_case_manifest_path.as_posix(),
            "output_root": self.output_root.as_posix(),
            "patient_output_root": self.patient_output_root.as_posix(),
            "pathway_name": self.pathway_name,
            "checkpoint_name": self.checkpoint_name,
            "result_path": self.result_path.as_posix(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PatientWorkerJob":
        schema_version = payload.get("schema_version")
        if schema_version != PATIENT_WORKER_JOB_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported worker job schema_version {schema_version!r}; "
                f"expected {PATIENT_WORKER_JOB_SCHEMA_VERSION!r}"
            )
        patient_case_payload = payload.get("patient_case", {})
        if not isinstance(patient_case_payload, Mapping):
            raise TypeError("patient_case must be an object")
        return cls(
            job_id=str(payload.get("job_id", "")),
            attempt_number=int(payload.get("attempt_number", 1)),
            run_id=str(payload.get("run_id", "")),
            patient_case=PatientCase(
                patient_uid=str(patient_case_payload.get("patient_uid", "")),
                patient_label=str(patient_case_payload.get("patient_label", "")),
                source_run_id=str(patient_case_payload.get("source_run_id", "")),
                input_manifest_id=str(patient_case_payload.get("input_manifest_id", "")),
                metadata=patient_case_payload.get("metadata", {}),
            ),
            input_case_manifest_path=Path(str(payload.get("input_case_manifest_path", ""))),
            output_root=Path(str(payload.get("output_root", ""))),
            pathway_name=str(payload.get("pathway_name", "")),
            checkpoint_name=str(payload.get("checkpoint_name", "")),
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
            "patient_count": len(self.worker_jobs),
            "patient_uids": [worker_job.patient_case.patient_uid for worker_job in self.worker_jobs],
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
    metadata: Mapping[str, Any] | None = None,
) -> PatientProcessRunPlan:
    """Build a standalone process plan from the DICOM input case manifest."""
    resolved_output_root = Path(output_root)
    patient_cases = _patient_cases_from_input_case_manifest(Path(input_case_manifest_path), patient_uids)
    worker_jobs = tuple(
        PatientWorkerJob(
            job_id=f"patient_{index:04d}_{patient_case.safe_patient_uid}",
            patient_case=patient_case,
            input_case_manifest_path=Path(input_case_manifest_path),
            output_root=resolved_output_root,
            pathway_name=pathway_name,
            checkpoint_name=checkpoint_name,
            run_id=run_id,
            metadata={"patient_index": index},
        )
        for index, patient_case in enumerate(patient_cases, start=1)
    )
    return PatientProcessRunPlan(
        output_root=resolved_output_root,
        input_case_manifest_path=Path(input_case_manifest_path),
        worker_jobs=worker_jobs,
        pathway_name=pathway_name,
        checkpoint_name=checkpoint_name,
        run_id=run_id,
        failure_policy=failure_policy,
        max_workers=max_workers,
        metadata=dict(metadata or {}),
    )


def write_patient_worker_job_packets(plan: PatientProcessRunPlan) -> tuple[Path, ...]:
    """Write one JSON job packet per patient worker."""
    job_dir = plan.output_root.joinpath("worker_jobs")
    return tuple(_write_json_object(job_dir.joinpath(f"{job.job_id}.json"), job.as_mapping()) for job in plan.worker_jobs)


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


def run_patient_worker_job(job: PatientWorkerJob, *, dry_run: bool = False) -> PatientWorkerResult:
    """Run one patient worker job.

    Scientific execution is intentionally not wired here yet. The current
    standalone scaffold can prove process/job/result wiring in dry-run mode while
    keeping the one-patient runtime builder explicit.
    """
    start_time = perf_counter()
    missing_core_input_roles = _missing_core_input_roles(job.patient_case)
    input_preflight_metadata = {
        "core_input_paths_all_present": not missing_core_input_roles,
        "missing_core_input_roles": missing_core_input_roles,
    }
    if dry_run:
        warnings = ["dry-run worker did not build runtime state or execute scientific stages"]
        if missing_core_input_roles:
            warnings.append(
                "input preflight found missing core input files: " + ", ".join(missing_core_input_roles)
            )
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
        return PatientWorkerResult(
            worker_job=job,
            status=PatientStageStatus.FAILED,
            elapsed_seconds=perf_counter() - start_time,
            exit_code=2,
            dry_run=False,
            warnings=(
                "input preflight found missing core input files: " + ", ".join(missing_core_input_roles),
            ),
            metadata={
                "worker_boundary": "standalone_patient_process_runner",
                "failed_boundary": "core_input_path_preflight",
                "input_preflight": input_preflight_metadata,
            },
        )

    return PatientWorkerResult(
        worker_job=job,
        status=PatientStageStatus.FAILED,
        elapsed_seconds=perf_counter() - start_time,
        exit_code=2,
        dry_run=False,
        warnings=("standalone one-patient runtime builder is not implemented yet",),
        metadata={
            "worker_boundary": "standalone_patient_process_runner",
            "missing_boundary": "one_patient_runtime_state_builder",
            "input_preflight": input_preflight_metadata,
        },
    )


def run_worker_job_file(job_path: Path, *, dry_run: bool = False) -> PatientWorkerResult:
    """Load, run, and write one worker job file."""
    worker_job = load_patient_worker_job(job_path)
    result = run_patient_worker_job(worker_job, dry_run=dry_run)
    write_patient_worker_result(result)
    return result


def launch_worker_job_file(job_path: Path, *, dry_run: bool = False, timeout_seconds: float | None = None) -> PatientWorkerResult:
    """Launch one worker job in a subprocess and load its result JSON."""
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "run_patient_scientific_worker.py"),
        str(job_path),
    ]
    if dry_run:
        command.append("--dry-run")
    completed = subprocess.run(command, check=False, timeout=timeout_seconds)
    worker_job = load_patient_worker_job(job_path)
    if worker_job.result_path.is_file():
        result_payload = _read_json_object(worker_job.result_path)
        status = PatientStageStatus(result_payload.get("status", PatientStageStatus.FAILED.value))
        elapsed_seconds = float(result_payload.get("elapsed_seconds", 0.0) or 0.0)
        warnings = result_payload.get("warnings", ())
        metadata = result_payload.get("metadata", {})
    else:
        status = PatientStageStatus.FAILED
        elapsed_seconds = 0.0
        warnings = ("worker did not write a result JSON",)
        metadata = {}
    return PatientWorkerResult(
        worker_job=worker_job,
        status=status,
        elapsed_seconds=elapsed_seconds,
        exit_code=int(completed.returncode),
        dry_run=dry_run,
        warnings=warnings,
        metadata=metadata,
    )


def run_patient_process_plan(
    plan: PatientProcessRunPlan,
    *,
    dry_run_workers: bool = False,
    timeout_seconds: float | None = None,
) -> tuple[PatientWorkerResult, ...]:
    """Run a plan through sequential subprocess workers."""
    job_paths = write_patient_worker_job_packets(plan)
    write_patient_process_run_plan(plan)
    results: list[PatientWorkerResult] = []
    for job_path in job_paths:
        result = launch_worker_job_file(job_path, dry_run=dry_run_workers, timeout_seconds=timeout_seconds)
        results.append(result)
        if not result.succeeded and plan.failure_policy == PatientProcessFailurePolicy.STOP_ON_FAILURE:
            break
    return tuple(results)