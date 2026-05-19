"""Output artifact inventory helpers for patient-scoped refactors."""

from .contracts import OUTPUT_TABLE_CONTRACT_SCHEMA_VERSION
from .contracts import build_output_table_contracts
from .contracts import normalize_legacy_table_name
from .contracts import summarize_output_table_contracts
from .contracts import write_output_table_contracts
from .exporters import PHASE3B_DATAFRAME_EXPORT_SCHEMA_VERSION
from .exporters import DataframeArtifact
from .exporters import iter_biopsy_mc_artifacts
from .exporters import iter_cohort_artifacts
from .exporters import iter_patient_mc_artifacts
from .exporters import iter_patient_preprocessing_artifacts
from .exporters import write_dataframe_artifact
from .exporters import write_dataframe_artifacts
from .in_memory_stitching import IN_MEMORY_STITCH_VALIDATION_SCHEMA_VERSION
from .in_memory_stitching import build_in_memory_stitch_validation
from .in_memory_stitching import collect_patient_fragment_dataframes
from .in_memory_stitching import summarize_in_memory_stitch_validation
from .in_memory_stitching import write_in_memory_stitch_validation_outputs
from .inventory import OUTPUT_ARTIFACT_INVENTORY_SCHEMA_VERSION
from .inventory import build_output_artifact_inventory
from .inventory import summarize_output_artifact_inventory
from .inventory import write_output_artifact_inventory
from .patient_artifacts import PATIENT_ARTIFACT_MANIFEST_SCHEMA_VERSION
from .patient_artifacts import PATIENT_STITCH_PLAN_SCHEMA_VERSION
from .patient_artifacts import build_patient_artifact_manifest
from .patient_artifacts import build_patient_stitch_plan
from .patient_artifacts import summarize_patient_artifacts
from .patient_artifacts import write_patient_artifact_outputs
from .phase3c_surface import PHASE3C_OUTPUT_DIR_NAME
from .phase3c_surface import PHASE3C_OUTPUT_SURFACE_SCHEMA_VERSION
from .phase3c_surface import Phase3COutputSurfaceResult
from .phase3c_surface import collect_phase3c_output_artifacts
from .phase3c_surface import summarize_phase3c_artifact_manifest
from .phase3c_surface import write_phase3c_output_surface
from .stitch_validation import SHADOW_STITCH_VALIDATION_SCHEMA_VERSION
from .stitch_validation import run_shadow_stitch_validation
from .stitch_validation import summarize_shadow_stitch_validation
from .stitch_validation import write_shadow_stitch_validation

__all__ = [
    "OUTPUT_ARTIFACT_INVENTORY_SCHEMA_VERSION",
    "PATIENT_ARTIFACT_MANIFEST_SCHEMA_VERSION",
    "PATIENT_STITCH_PLAN_SCHEMA_VERSION",
    "OUTPUT_TABLE_CONTRACT_SCHEMA_VERSION",
    "PHASE3B_DATAFRAME_EXPORT_SCHEMA_VERSION",
    "PHASE3C_OUTPUT_DIR_NAME",
    "PHASE3C_OUTPUT_SURFACE_SCHEMA_VERSION",
    "SHADOW_STITCH_VALIDATION_SCHEMA_VERSION",
    "IN_MEMORY_STITCH_VALIDATION_SCHEMA_VERSION",
    "DataframeArtifact",
    "Phase3COutputSurfaceResult",
    "build_output_artifact_inventory",
    "build_in_memory_stitch_validation",
    "build_patient_artifact_manifest",
    "build_patient_stitch_plan",
    "build_output_table_contracts",
    "collect_phase3c_output_artifacts",
    "collect_patient_fragment_dataframes",
    "iter_biopsy_mc_artifacts",
    "iter_cohort_artifacts",
    "iter_patient_mc_artifacts",
    "iter_patient_preprocessing_artifacts",
    "normalize_legacy_table_name",
    "run_shadow_stitch_validation",
    "summarize_in_memory_stitch_validation",
    "summarize_output_artifact_inventory",
    "summarize_patient_artifacts",
    "summarize_phase3c_artifact_manifest",
    "summarize_output_table_contracts",
    "summarize_shadow_stitch_validation",
    "write_dataframe_artifact",
    "write_dataframe_artifacts",
    "write_in_memory_stitch_validation_outputs",
    "write_output_artifact_inventory",
    "write_patient_artifact_outputs",
    "write_output_table_contracts",
    "write_phase3c_output_surface",
    "write_shadow_stitch_validation",
]
