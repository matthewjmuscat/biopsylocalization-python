"""Patient-local runner scaffold for the localization pipeline.

This package is the Phase C.0 boundary around existing scientific code. It keeps
the legacy all-patient pipeline available as the validation oracle while giving
new work a typed, patient-local surface.
"""

from .artifacts import PatientArtifactStore
from .artifacts import collect_patient_dataframe_artifacts
from .artifacts import write_patient_dataframe_artifacts
from .batch import resolve_patient_uids
from .batch import run_patient_batch
from .batch import run_patient_batch_from_legacy
from .cohort_assembly import PatientBatchCohortAssemblyResult
from .cohort_assembly import assemble_patient_batch_cohort_tables
from .cohort_assembly import build_patient_batch_artifact_inventory
from .cohort_assembly import summarize_patient_batch_cohort_assembly
from .cohort_assembly import summarize_patient_batch_cohort_validation
from .cohort_assembly import validate_patient_batch_cohort_assembly
from .cohort_assembly import write_patient_batch_cohort_assembly_outputs
from .contracts import LegacyCohortRuntimeState
from .contracts import LegacyPatientRuntimeState
from .contracts import LegacyRuntimeKeys
from .contracts import PatientBatchExecutionBackend
from .contracts import PatientBatchRunConfig
from .contracts import PatientBatchRunResult
from .contracts import PatientCase
from .contracts import PatientRunConfig
from .contracts import PatientRunResult
from .contracts import PatientStageName
from .contracts import PatientStageResult
from .contracts import PatientStageStatus
from .contracts import resolve_legacy_patient_uids
from .contracts import validate_patient_uids
from .legacy_bridge import build_patient_case_from_legacy
from .legacy_bridge import carve_patient_runtime_state
from .legacy_bridge import carve_patient_runtime_state_by_uid
from .runner import PatientStage
from .runner import PatientStageRunner
from .runner import default_patient_stages
from .runner import run_patient_case
from .runner import run_patient_stages
from .stages import write_patient_artifacts_stage

__all__ = [
    "LegacyCohortRuntimeState",
    "LegacyPatientRuntimeState",
    "LegacyRuntimeKeys",
    "PatientArtifactStore",
    "PatientBatchCohortAssemblyResult",
    "PatientBatchExecutionBackend",
    "PatientBatchRunConfig",
    "PatientBatchRunResult",
    "PatientCase",
    "PatientRunConfig",
    "PatientRunResult",
    "PatientStage",
    "PatientStageName",
    "PatientStageResult",
    "PatientStageRunner",
    "PatientStageStatus",
    "assemble_patient_batch_cohort_tables",
    "build_patient_case_from_legacy",
    "build_patient_batch_artifact_inventory",
    "carve_patient_runtime_state",
    "carve_patient_runtime_state_by_uid",
    "collect_patient_dataframe_artifacts",
    "default_patient_stages",
    "resolve_legacy_patient_uids",
    "resolve_patient_uids",
    "run_patient_batch",
    "run_patient_batch_from_legacy",
    "run_patient_case",
    "run_patient_stages",
    "summarize_patient_batch_cohort_assembly",
    "summarize_patient_batch_cohort_validation",
    "validate_patient_uids",
    "validate_patient_batch_cohort_assembly",
    "write_patient_batch_cohort_assembly_outputs",
    "write_patient_artifacts_stage",
    "write_patient_dataframe_artifacts",
]
