"""Finalize standalone process results for existing post-run readers.

The parent owns this disk-only boundary. It validates worker evidence and writes
batch/index manifests; it never loads DICOM, scientific arrays, or cohort state.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from output_artifacts.manifest_index import ManifestIndexRecorder
from output_artifacts.run_compatibility import RUN_COMPATIBILITY_METADATA_KEY
from post_run.cohort_assembly.manifest_loader import load_patient_result_from_manifest

from .contracts import PatientBatchRunResult, PatientRunResult, PatientStageResult, PatientStageStatus
from .manifests import PATIENT_BATCH_RUN_MANIFEST_SCHEMA_VERSION, PATIENT_RUN_MANIFEST_SCHEMA_VERSION
from .manifests import write_patient_batch_run_manifest, write_patient_run_manifest

if TYPE_CHECKING:
    from .process_runner import PatientProcessRunPlan, PatientWorkerJob, PatientWorkerResult


@dataclass(frozen=True)
class PatientProcessFinalization:
    """Persisted batch result and validated worker summaries in launch order."""

    batch_result: PatientBatchRunResult
    worker_results: tuple[PatientWorkerResult, ...]
    batch_manifest_path: Path
    index_path: Path


def read_validated_worker_patient_result(job: PatientWorkerJob, worker: PatientWorkerResult) -> PatientRunResult:
    """Validate a successful worker's identity, ordered stages, and artifact files."""
    patient = load_patient_result_from_manifest(job.patient_output_root / "patient_run_manifest.json")
    if patient.patient_case.patient_uid != job.patient_case.patient_uid:
        raise ValueError("patient manifest UID differs from worker job")
    if not patient.succeeded or not patient.stage_results:
        raise ValueError("successful worker requires successful patient stage evidence")
    if patient.metadata.get("worker_job_id") != job.job_id or patient.metadata.get("run_id") != job.run_id:
        raise ValueError("patient manifest run/job identity differs from worker job")
    if patient.metadata.get(RUN_COMPATIBILITY_METADATA_KEY) != job.metadata.get(RUN_COMPATIBILITY_METADATA_KEY):
        raise ValueError("patient manifest compatibility identity differs from worker job")
    if RUN_COMPATIBILITY_METADATA_KEY not in patient.metadata:
        raise ValueError("successful live patient manifest lacks strict compatibility identity")
    expected_stages = (*job.metadata["planned_stage_names"], "patient_artifact_writing")
    if tuple(stage.stage_name for stage in patient.stage_results) != expected_stages:
        raise ValueError("patient manifest stages differ from the planned pathway")
    if any(not stage.succeeded for stage in patient.stage_results):
        raise ValueError("successful worker contains skipped or failed required stages")
    if "artifact_paths" not in worker.metadata:
        raise ValueError("successful worker must report its artifact inventory explicitly")
    expected_artifacts = tuple(Path(path).resolve() for path in worker.metadata["artifact_paths"])
    if tuple(path.resolve() for path in patient.artifact_paths) != expected_artifacts:
        raise ValueError("worker artifact inventory differs from patient manifest")
    patient_root = job.patient_output_root.resolve()
    for path in patient.artifact_paths:
        if not path.resolve().is_relative_to(patient_root) or not path.is_file():
            raise ValueError("patient artifact is missing or outside its patient directory: {}".format(path))
    return patient


def finalize_patient_process_run(
    plan: PatientProcessRunPlan,
    worker_results: tuple[PatientWorkerResult, ...],
) -> PatientProcessFinalization:
    """Write assembly-compatible manifests for every planned patient.

    Successful live workers must have complete matching patient evidence. Missing
    or inconsistent evidence converts the worker to failure. Timeout/unlaunched
    patients receive explicit result records; failed outputs are never advertised
    as usable artifacts. Use a fresh output root for each process run.
    """
    from .process_runner import write_patient_worker_result

    if tuple(worker.worker_job for worker in worker_results) != plan.worker_jobs[:len(worker_results)]:
        raise ValueError("worker results must be an ordered prefix of the process plan")
    patients = []
    validated_workers = []
    for index, job in enumerate(plan.worker_jobs):
        worker = worker_results[index] if index < len(worker_results) else None
        patient = None
        if worker is not None and worker.succeeded and not worker.dry_run:
            try:
                patient = read_validated_worker_patient_result(job, worker)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                worker = replace(
                    worker,
                    status=PatientStageStatus.FAILED,
                    exit_code=1,
                    warnings=(*worker.warnings, "patient evidence validation failed: {}".format(exc)),
                    metadata={**worker.metadata, "failed_boundary": "patient_manifest_validation"},
                )
        if patient is None:
            if worker is not None and not worker.dry_run:
                try:
                    failed_patient = load_patient_result_from_manifest(job.patient_output_root / "patient_run_manifest.json")
                    if failed_patient.patient_case.patient_uid == job.patient_case.patient_uid and not failed_patient.succeeded:
                        patient = replace(failed_patient, status=PatientStageStatus.FAILED, artifact_paths=())
                except (OSError, ValueError, TypeError, KeyError):
                    pass
        if patient is None:
            status = (
                PatientStageStatus.NOT_STARTED if worker is None
                else PatientStageStatus.SKIPPED if worker.dry_run and worker.succeeded
                else PatientStageStatus.FAILED
            )
            reason = "not_launched" if worker is None else "dry_run" if worker.dry_run else "worker_failed"
            patient = PatientRunResult(
                patient_case=job.patient_case,
                output_root=job.patient_output_root,
                status=status,
                elapsed_seconds=0.0 if worker is None else worker.elapsed_seconds,
                stage_results=(PatientStageResult(
                    stage_name="legacy_bridge",
                    status=status,
                    warnings=() if worker is None else tuple(worker.warnings),
                    metadata={"reason": reason},
                ),),
                metadata={
                    **job.metadata,
                    "run_id": job.run_id,
                    "worker_job_id": job.job_id,
                    "execution_status": reason,
                    "scientific_execution": False,
                },
            )
        if worker is None or worker.dry_run or not worker.succeeded:
            write_patient_run_manifest(patient)
        patients.append(patient)
        if worker is not None:
            write_patient_worker_result(worker)
            validated_workers.append(worker)

    all_succeeded = bool(patients) and all(patient.succeeded for patient in patients)
    dry_run = plan.execution_mode == "dry_run_workers"
    batch = PatientBatchRunResult(
        output_root=plan.output_root,
        patient_results=tuple(patients),
        status=(
            PatientStageStatus.FAILED if any(not worker.succeeded for worker in validated_workers)
            else PatientStageStatus.SKIPPED if dry_run
            else PatientStageStatus.SUCCEEDED if all_succeeded
            else PatientStageStatus.FAILED
        ),
        elapsed_seconds=sum(worker.elapsed_seconds for worker in validated_workers),
        metadata={
            **plan.metadata,
            "run_id": plan.run_id,
            "execution_backend": "sequential_subprocess",
            "execution_mode": plan.execution_mode,
            "scientific_execution": not dry_run,
            "assembly_ready": all_succeeded and not dry_run,
            "planned_patient_count": len(plan.worker_jobs),
            "launched_patient_count": len(worker_results),
            "unlaunched_patient_uids": [job.patient_case.patient_uid for job in plan.worker_jobs[len(worker_results):]],
        },
    )
    batch_path = write_patient_batch_run_manifest(batch)
    recorder = ManifestIndexRecorder(plan.output_root, run_id=plan.run_id, metadata=dict(batch.metadata))
    for patient in patients:
        recorder.record_written_manifest(
            "patient_run_manifest", patient.output_root / "patient_run_manifest.json",
            manifest_schema_version=PATIENT_RUN_MANIFEST_SCHEMA_VERSION,
            patient_uid=patient.patient_case.patient_uid, stage_name="patient_runner",
        )
    recorder.record_written_manifest(
        "patient_batch_run_manifest", batch_path,
        manifest_schema_version=PATIENT_BATCH_RUN_MANIFEST_SCHEMA_VERSION, stage_name="patient_batch_runner",
    )
    return PatientProcessFinalization(batch, tuple(validated_workers), batch_path, recorder.write(overwrite=False))