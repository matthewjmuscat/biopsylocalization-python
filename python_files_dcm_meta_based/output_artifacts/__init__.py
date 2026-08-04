"""Output artifact inventory helpers for patient-scoped refactors."""

from .assembly_planner import OUTPUT_ASSEMBLY_PLANNER_SCHEMA_VERSION
from .assembly_planner import OutputAssemblyPlan
from .assembly_planner import OutputRowOrderPolicy
from .assembly_planner import build_output_assembly_plans
from .assembly_planner import build_shadow_stitch_pairs_from_output_assembly_plans
from .assembly_planner import output_assembly_plan_rows
from .contracts import OUTPUT_TABLE_CONTRACT_SCHEMA_VERSION
from .contracts import build_output_table_contracts
from .contracts import normalize_legacy_table_name
from .contracts import summarize_output_table_contracts
from .contracts import write_output_table_contracts
from .context_contracts import SCIENTIFIC_CONTEXT_CONTRACTS_SCHEMA_VERSION
from .context_contracts import ArtifactRef
from .context_contracts import ArrayArtifactSpec
from .context_contracts import PatientArtifactIndex
from .context_contracts import TableArtifactSpec
from .context_contracts import artifact_ref_from_dict
from .context_contracts import patient_artifact_index_from_dict
from .context_contracts import read_patient_artifact_index
from .context_contracts import write_patient_artifact_index
from .exporters import PHASE3B_DATAFRAME_EXPORT_SCHEMA_VERSION
from .exporters import DataframeArtifact
from .exporters import iter_biopsy_mc_artifacts
from .exporters import iter_cohort_artifacts
from .exporters import iter_patient_mc_artifacts
from .exporters import iter_patient_preprocessing_artifacts
from .exporters import write_dataframe_artifact
from .exporters import write_dataframe_artifacts
from .expected_artifacts import EXPECTED_ARTIFACT_POLICY_SCHEMA_VERSION
from .expected_artifacts import ExpectedArtifactDecision
from .expected_artifacts import ExpectedArtifactStatus
from .expected_artifacts import MissingArtifactSeverity
from .expected_artifacts import classify_expected_artifact
from .expected_artifacts import classify_expected_assembly_plan
from .expected_artifacts import classify_expected_table_spec
from .expected_artifacts import expected_artifact_decision_report_fields
from .in_memory_stitching import IN_MEMORY_STITCH_VALIDATION_SCHEMA_VERSION
from .in_memory_stitching import build_in_memory_stitch_validation
from .in_memory_stitching import collect_patient_fragment_dataframes
from .in_memory_stitching import summarize_in_memory_stitch_validation
from .in_memory_stitching import write_in_memory_stitch_validation_outputs
from .inventory import OUTPUT_ARTIFACT_INVENTORY_SCHEMA_VERSION
from .inventory import build_output_artifact_inventory
from .inventory import summarize_output_artifact_inventory
from .inventory import write_output_artifact_inventory
from .manifest_index import RUN_MANIFEST_INDEX_SCHEMA_VERSION
from .manifest_index import ManifestIndexEntry
from .manifest_index import ManifestIndexRecorder
from .manifest_index import build_run_manifest_index
from .manifest_index import default_run_manifest_index_path
from .manifest_index import manifest_index_entry
from .manifest_index import manifest_index_rows
from .manifest_index import read_run_manifest_index
from .manifest_index import summarize_manifest_index_entries
from .manifest_index import write_run_manifest_index
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
from .schema_registry import EXPECTED_CURRENT_REGISTRY_COUNT
from .schema_registry import OUTPUT_SCHEMA_COVERAGE_SCHEMA_VERSION
from .schema_registry import OUTPUT_SCHEMA_REGISTRY_VERSION
from .schema_registry import CanonicalKeySpec
from .schema_registry import OutputSchemaRegistry
from .schema_registry import OutputTableSpec
from .schema_registry import build_output_schema_data_dictionary
from .schema_registry import build_output_schema_coverage_report
from .schema_registry import render_output_schema_data_dictionary_markdown
from .schema_registry import summarize_output_schema_coverage
from .schema_registry import write_output_schema_coverage_report
from .schema_registry import write_output_schema_data_dictionary
from .stitch_validation import SHADOW_STITCH_VALIDATION_SCHEMA_VERSION
from .stitch_validation import run_shadow_stitch_validation
from .stitch_validation import summarize_shadow_stitch_validation
from .stitch_validation import write_shadow_stitch_validation

__all__ = [
    "OUTPUT_ARTIFACT_INVENTORY_SCHEMA_VERSION",
    "PATIENT_ARTIFACT_MANIFEST_SCHEMA_VERSION",
    "PATIENT_STITCH_PLAN_SCHEMA_VERSION",
    "OUTPUT_TABLE_CONTRACT_SCHEMA_VERSION",
    "SCIENTIFIC_CONTEXT_CONTRACTS_SCHEMA_VERSION",
    "OUTPUT_ASSEMBLY_PLANNER_SCHEMA_VERSION",
    "PHASE3B_DATAFRAME_EXPORT_SCHEMA_VERSION",
    "PHASE3C_OUTPUT_DIR_NAME",
    "PHASE3C_OUTPUT_SURFACE_SCHEMA_VERSION",
    "OUTPUT_SCHEMA_COVERAGE_SCHEMA_VERSION",
    "OUTPUT_SCHEMA_REGISTRY_VERSION",
    "EXPECTED_ARTIFACT_POLICY_SCHEMA_VERSION",
    "EXPECTED_CURRENT_REGISTRY_COUNT",
    "SHADOW_STITCH_VALIDATION_SCHEMA_VERSION",
    "IN_MEMORY_STITCH_VALIDATION_SCHEMA_VERSION",
    "RUN_MANIFEST_INDEX_SCHEMA_VERSION",
    "DataframeArtifact",
    "ArtifactRef",
    "ArrayArtifactSpec",
    "PatientArtifactIndex",
    "TableArtifactSpec",
    "CanonicalKeySpec",
    "OutputAssemblyPlan",
    "OutputRowOrderPolicy",
    "OutputSchemaRegistry",
    "OutputTableSpec",
    "ExpectedArtifactDecision",
    "ExpectedArtifactStatus",
    "MissingArtifactSeverity",
    "ManifestIndexEntry",
    "ManifestIndexRecorder",
    "Phase3COutputSurfaceResult",
    "build_output_artifact_inventory",
    "build_output_assembly_plans",
    "build_in_memory_stitch_validation",
    "build_patient_artifact_manifest",
    "build_patient_stitch_plan",
    "build_output_table_contracts",
    "build_output_schema_data_dictionary",
    "build_output_schema_coverage_report",
    "build_run_manifest_index",
    "build_shadow_stitch_pairs_from_output_assembly_plans",
    "collect_phase3c_output_artifacts",
    "collect_patient_fragment_dataframes",
    "classify_expected_artifact",
    "artifact_ref_from_dict",
    "classify_expected_assembly_plan",
    "classify_expected_table_spec",
    "expected_artifact_decision_report_fields",
    "iter_biopsy_mc_artifacts",
    "iter_cohort_artifacts",
    "iter_patient_mc_artifacts",
    "iter_patient_preprocessing_artifacts",
    "default_run_manifest_index_path",
    "manifest_index_entry",
    "manifest_index_rows",
    "normalize_legacy_table_name",
    "output_assembly_plan_rows",
    "read_run_manifest_index",
    "patient_artifact_index_from_dict",
    "read_patient_artifact_index",
    "run_shadow_stitch_validation",
    "render_output_schema_data_dictionary_markdown",
    "summarize_manifest_index_entries",
    "summarize_in_memory_stitch_validation",
    "summarize_output_artifact_inventory",
    "summarize_output_schema_coverage",
    "summarize_patient_artifacts",
    "summarize_phase3c_artifact_manifest",
    "summarize_output_table_contracts",
    "summarize_shadow_stitch_validation",
    "write_dataframe_artifact",
    "write_dataframe_artifacts",
    "write_in_memory_stitch_validation_outputs",
    "write_output_artifact_inventory",
    "write_output_schema_coverage_report",
    "write_output_schema_data_dictionary",
    "write_patient_artifact_outputs",
    "write_output_table_contracts",
    "write_run_manifest_index",
    "write_patient_artifact_index",
    "write_phase3c_output_surface",
    "write_shadow_stitch_validation",
]
