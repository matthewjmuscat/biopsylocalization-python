"""Patient-local runner scaffold for the localization pipeline.

This package is the Phase C.0 boundary around existing scientific code. It keeps
the legacy all-patient pipeline available as the validation oracle while giving
new work a typed, patient-local surface.
"""

from .artifacts import PatientArtifactStore
from .artifacts import collect_patient_dataframe_artifacts
from .artifacts import write_patient_dataframe_artifacts
from .contracts import LegacyPatientRuntimeState
from .contracts import PatientCase
from .contracts import PatientRunConfig
from .contracts import PatientRunResult
from .contracts import PatientStageName
from .contracts import PatientStageResult
from .contracts import PatientStageStatus
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
    "LegacyPatientRuntimeState",
    "PatientArtifactStore",
    "PatientCase",
    "PatientRunConfig",
    "PatientRunResult",
    "PatientStage",
    "PatientStageName",
    "PatientStageResult",
    "PatientStageRunner",
    "PatientStageStatus",
    "build_patient_case_from_legacy",
    "carve_patient_runtime_state",
    "carve_patient_runtime_state_by_uid",
    "collect_patient_dataframe_artifacts",
    "default_patient_stages",
    "run_patient_case",
    "run_patient_stages",
    "write_patient_artifacts_stage",
    "write_patient_dataframe_artifacts",
]