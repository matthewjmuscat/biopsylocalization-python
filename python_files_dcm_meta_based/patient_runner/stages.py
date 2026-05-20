"""Initial patient-runner stage implementations."""

from __future__ import annotations

from .artifacts import PatientArtifactStore
from .contracts import LegacyPatientRuntimeState
from .contracts import PatientRunConfig
from .contracts import PatientStageName
from .contracts import PatientStageResult


def write_patient_artifacts_stage(runtime_state: LegacyPatientRuntimeState,
                                  config: PatientRunConfig) -> PatientStageResult:
    """Write the patient-local dataframe artifacts currently present in memory."""
    artifact_store = PatientArtifactStore.from_config(config, runtime_state)
    artifacts = artifact_store.collect(runtime_state)
    written_paths = artifact_store.write(artifacts)
    return PatientStageResult.success(
        PatientStageName.PATIENT_ARTIFACT_WRITING,
        artifact_count=len(written_paths),
        output_paths=written_paths,
        metadata={
            "patient_uid": runtime_state.patient_uid,
            "patient_output_dir": artifact_store.output_root.as_posix(),
        },
    )