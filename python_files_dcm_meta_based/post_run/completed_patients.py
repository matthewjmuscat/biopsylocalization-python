"""Read and match completed patient executions without loading scientific arrays.

Reusable post-run boundary: exact jobs/results/manifests, strict run identities,
and patient input identity precede any checkpoint-specific comparison. No legacy
missing-provenance mode, execution, numerical comparison, or cohort state lives
here. Explicit job paths also support accepted singleton validation attempts.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

from input_data.content_identity import INPUT_CONTENT_KEY, validate_patient_input_content
from output_artifacts.run_compatibility import RunCompatibilityIdentity, require_compatible_run_identities
from patient_runner.contracts import PatientRunResult
from patient_runner.process_finalization import read_validated_worker_patient_result
from patient_runner.process_runner import (
    PatientWorkerJob, PatientWorkerResult, load_patient_worker_job,
    PATIENT_WORKER_RESULT_SCHEMA_VERSION,
)


@dataclass(frozen=True)
class CompletedPatient:
    """Verified lightweight job and patient manifest; contains no patient arrays."""

    job: PatientWorkerJob
    patient: PatientRunResult


def read_completed_patient(job_path: Path, *, require_content_identity: bool = True) -> CompletedPatient:
    """Read an explicit successful attempt and verify all durable bindings.

    The original inputs need not remain online. Content verification must have
    succeeded in the worker, and its ledger is checked structurally here.
    Missing/corrupt/mismatched records fail; discovery never selects a retry.
    """
    job = load_patient_worker_job(job_path)
    if job.job_path.resolve() != Path(job_path).resolve():
        raise ValueError("job file is outside its declared execution root")
    payload = json.loads(job.result_path.read_text())
    if (payload.get("schema_version") != PATIENT_WORKER_RESULT_SCHEMA_VERSION
            or payload.get("status") != "succeeded" or payload.get("succeeded") is not True
            or payload.get("exit_code") != 0 or payload.get("dry_run") is not False
            or payload.get("timed_out") is not False
            or payload.get("patient_uid") != job.patient_case.patient_uid
            or PatientWorkerJob.from_mapping(payload["job"]) != job):
        raise ValueError("worker result is incomplete or differs from the requested attempt")
    worker = PatientWorkerResult(job, "succeeded", payload["elapsed_seconds"], 0, metadata=payload["metadata"])
    patient = read_validated_worker_patient_result(job, worker)
    identity = job.metadata.get(INPUT_CONTENT_KEY)
    if identity is None and require_content_identity:
        raise ValueError("completed patient lacks input content identity")
    if identity is not None:
        validate_patient_input_content(identity, job.patient_inputs)
        if (patient.metadata.get(INPUT_CONTENT_KEY) != identity
                or patient.metadata.get("input_content_verified_before") is not True
                or patient.metadata.get("input_content_verified_after") is not True):
            raise ValueError("patient lacks completed input content verification")
    if patient.metadata.get("patient_input_manifest_identity_sha256") != job.patient_inputs.manifest_identity_sha256:
        raise ValueError("patient manifest input role identity differs from its job")
    RunCompatibilityIdentity.from_dict(job.metadata["run_compatibility_identity"])
    return CompletedPatient(job, patient)


def match_completed_patients(
    reference_jobs: Sequence[Path], candidate_jobs: Sequence[Path],
) -> tuple[tuple[CompletedPatient, CompletedPatient], ...]:
    """Require compatible, complete, identical patient sets; match by exact UID.

    Scientific pathway and stage sequence must agree. Paths/run IDs/order may
    differ. Duplicate patients within either surface fail, including retries.
    This boundary can serve later checkpoint comparators without knowing anatomy.
    """
    surfaces = []
    identities = []
    for paths in (reference_jobs, candidate_jobs):
        patients = {}
        if not paths:
            raise ValueError("completed patient surface must not be empty")
        for path in paths:
            completed = read_completed_patient(Path(path))
            uid = completed.job.patient_case.patient_uid
            if uid in patients:
                raise ValueError("duplicate patient in completed surface: " + uid)
            patients[uid] = completed
            identities.append(RunCompatibilityIdentity.from_dict(completed.job.metadata["run_compatibility_identity"]))
        surfaces.append(patients)
    require_compatible_run_identities(identities)
    reference, candidate = surfaces
    if reference.keys() != candidate.keys():
        raise ValueError("completed patient sets differ")
    matches = []
    for uid in sorted(reference):
        left, right = reference[uid], candidate[uid]
        for attribute in ("pathway_name", "checkpoint_name", "patient_inputs"):
            if getattr(left.job, attribute) != getattr(right.job, attribute):
                raise ValueError("patient execution differs in " + attribute + ": " + uid)
        if left.job.metadata[INPUT_CONTENT_KEY] != right.job.metadata[INPUT_CONTENT_KEY]:
            raise ValueError("patient input bytes differ: " + uid)
        if left.job.metadata["planned_stage_names"] != right.job.metadata["planned_stage_names"]:
            raise ValueError("patient stage plans differ: " + uid)
        matches.append((left, right))
    return tuple(matches)
