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
from .cohort_assembly import PatientBatchCohortAssemblyConfig
from .cohort_assembly import assemble_patient_batch_cohort_tables
from .cohort_assembly import build_patient_batch_artifact_inventory
from .cohort_assembly import run_patient_batch_cohort_assembly
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
from .manifests import PATIENT_BATCH_RUN_MANIFEST_SCHEMA_VERSION
from .manifests import PATIENT_RUN_MANIFEST_SCHEMA_VERSION
from .manifests import patient_batch_run_result_manifest
from .manifests import patient_run_result_manifest
from .manifests import write_patient_batch_run_manifest
from .manifests import write_patient_run_manifest
from .main_validation import DEFAULT_PATIENT_RUNNER_SHADOW_OUTPUT_DIR_NAME
from .main_validation import PATIENT_RUNNER_MAIN_VALIDATION_SCHEMA_VERSION
from .main_validation import PatientRunnerMainValidationConfig
from .main_validation import PatientRunnerMainValidationMode
from .main_validation import PatientRunnerMainValidationResult
from .main_validation import PatientRunnerMainValidationSkippedResult
from .main_validation import run_patient_runner_main_validation
from .main_validation import summarize_patient_runner_main_validation
from .main_validation import write_patient_runner_main_validation_summary
from .parity import DEFAULT_PATIENT_RUNNER_PARITY_DIR_NAME
from .parity import PATIENT_RUNNER_POST_RUN_PARITY_SCHEMA_VERSION
from .parity import PatientRunnerParitySurface
from .parity import PatientRunnerParitySurfaceResult
from .parity import PatientRunnerPostRunParityConfig
from .parity import PatientRunnerPostRunParityResult
from .parity import compare_patient_runner_assembled_cohort_tables
from .parity import compare_patient_runner_recursive_csvs
from .parity import default_patient_runner_post_run_parity_output_dir
from .parity import format_patient_runner_post_run_parity_summary
from .parity import run_patient_runner_post_run_parity
from .parity import summarize_patient_runner_parity_surface
from .parity import summarize_patient_runner_post_run_parity
from .parity import summarize_patient_runner_post_run_parity_surfaces
from .runner import PatientStage
from .runner import PatientStageRunner
from .runner import default_patient_stages
from .runner import run_patient_case
from .runner import run_patient_stages
from .scientific_config import PatientAnatomicalPreprocessingScientificConfig
from .scientific_config import PatientGridPreprocessingScientificConfig
from .scientific_config import PatientGuidanceScientificConfig
from .scientific_config import PatientMCPrepScientificConfig
from .scientific_config import PatientMCSimulationScientificConfig
from .scientific_config import PatientMRADCInputNormalizationStageConfig
from .scientific_config import PatientOptimizationScientificConfig
from .scientific_config import PatientPreprocessingScientificConfig
from .scientific_config import PatientProstateOnlyMRADCStageConfig
from .scientific_config import PatientRawContourPullingStageConfig
from .scientific_config import PatientRealBiopsyProcessingStageConfig
from .scientific_config import PatientRealizedBiopsyTargetingStageConfig
from .scientific_config import PatientRunnerScientificConfig
from .scientific_config import PatientSampledBiopsyProcessingStageConfig
from .scientific_config import PatientScientificStageResources
from .scientific_config import PatientSimulatedBiopsyFinalizationStageConfig
from .scientific_config import PatientSimulatedBiopsyPlanningStageConfig
from .scientific_config import PatientSimulatedBiopsyPreparationStageConfig
from .scientific_config import PatientStandardNonBiopsyStructureProcessingStageConfig
from .scientific_config import PatientStructureSelectionStageConfig
from .scientific_config import PatientUncertaintyAttachmentStageConfig
from .scientific_dependencies import DEFAULT_PATIENT_SCIENTIFIC_EXECUTABLE_STAGE_ORDER
from .scientific_dependencies import DEFAULT_PATIENT_SCIENTIFIC_GRAPH_ORDER
from .scientific_dependencies import DEFAULT_PATIENT_SCIENTIFIC_GRAPH_PATHWAYS
from .scientific_dependencies import DEFAULT_PATIENT_SCIENTIFIC_GRAPH_STAGE_DEPENDENCIES
from .scientific_dependencies import DEFAULT_PATIENT_SCIENTIFIC_PATHWAYS
from .scientific_dependencies import DEFAULT_PATIENT_SCIENTIFIC_STAGE_DEPENDENCIES
from .scientific_dependencies import PatientScientificPathwayName
from .scientific_dependencies import PatientScientificStageDependency
from .scientific_dependencies import executable_patient_scientific_pathway_stage_names
from .scientific_dependencies import patient_scientific_pathway_graph_stage_names
from .scientific_dependencies import patient_scientific_pathway_stage_names
from .scientific_dependencies import resolve_patient_scientific_pathway_name
from .scientific_dependencies import resolve_patient_scientific_stage_names
from .scientific_dependencies import summarize_patient_scientific_dependency_graph
from .scientific_dependencies import summarize_patient_scientific_pathways
from .scientific_dependencies import validate_patient_scientific_graph_dependencies
from .scientific_dependencies import validate_patient_scientific_pathway_dependencies
from .scientific_dependencies import validate_patient_scientific_pathway_graph_dependencies
from .scientific_dependencies import validate_patient_scientific_stage_dependencies
from .scientific_stages import DEFAULT_SCIENTIFIC_STAGE_ORDER
from .scientific_stages import build_patient_scientific_stages
from .scientific_stages import build_patient_scientific_stages_for_pathway
from .scientific_stages import run_patient_anatomical_preprocessing_scientific_stage
from .scientific_stages import run_patient_grid_preprocessing_scientific_stage
from .scientific_stages import run_patient_guidance_scientific_stage
from .scientific_stages import run_patient_mc_prep_scientific_stage
from .scientific_stages import run_patient_mc_simulation_scientific_stage
from .scientific_stages import run_patient_optimization_scientific_stage
from .scientific_stages import run_patient_preprocessing_scientific_stage
from .scientific_stages import run_patient_simulated_biopsy_finalization_scientific_stage
from .scientific_stages import run_patient_transform_generation_scientific_stage
from .scientific_tranches import DEFAULT_PATIENT_SCIENTIFIC_TRANCHE_ORDER
from .scientific_tranches import DEFAULT_PATIENT_SCIENTIFIC_TRANCHES
from .scientific_tranches import PatientScientificTranche
from .scientific_tranches import PatientScientificTrancheName
from .scientific_tranches import build_patient_scientific_stages_for_tranches
from .scientific_tranches import default_patient_scientific_tranches
from .scientific_tranches import get_patient_scientific_tranche
from .scientific_tranches import iter_patient_scientific_tranches
from .scientific_tranches import patient_scientific_tranche_stage_names
from .scientific_tranches import resolve_patient_scientific_tranche_names
from .scientific_tranches import summarize_patient_scientific_tranches
from .stages import write_patient_artifacts_stage

__all__ = [
    "LegacyCohortRuntimeState",
    "LegacyPatientRuntimeState",
    "LegacyRuntimeKeys",
    "PatientArtifactStore",
    "PatientBatchCohortAssemblyConfig",
    "PatientBatchCohortAssemblyResult",
    "PatientBatchExecutionBackend",
    "PatientBatchRunConfig",
    "PatientBatchRunResult",
    "PatientCase",
    "PatientRunConfig",
    "PATIENT_BATCH_RUN_MANIFEST_SCHEMA_VERSION",
    "PATIENT_RUNNER_MAIN_VALIDATION_SCHEMA_VERSION",
    "PATIENT_RUNNER_POST_RUN_PARITY_SCHEMA_VERSION",
    "PATIENT_RUN_MANIFEST_SCHEMA_VERSION",
    "DEFAULT_PATIENT_RUNNER_PARITY_DIR_NAME",
    "DEFAULT_PATIENT_RUNNER_SHADOW_OUTPUT_DIR_NAME",
    "DEFAULT_PATIENT_SCIENTIFIC_EXECUTABLE_STAGE_ORDER",
    "DEFAULT_PATIENT_SCIENTIFIC_GRAPH_ORDER",
    "DEFAULT_PATIENT_SCIENTIFIC_GRAPH_PATHWAYS",
    "DEFAULT_PATIENT_SCIENTIFIC_GRAPH_STAGE_DEPENDENCIES",
    "DEFAULT_PATIENT_SCIENTIFIC_PATHWAYS",
    "DEFAULT_PATIENT_SCIENTIFIC_STAGE_DEPENDENCIES",
    "DEFAULT_PATIENT_SCIENTIFIC_TRANCHE_ORDER",
    "DEFAULT_PATIENT_SCIENTIFIC_TRANCHES",
    "DEFAULT_SCIENTIFIC_STAGE_ORDER",
    "PatientAnatomicalPreprocessingScientificConfig",
    "PatientGridPreprocessingScientificConfig",
    "PatientGuidanceScientificConfig",
    "PatientMCPrepScientificConfig",
    "PatientMCSimulationScientificConfig",
    "PatientMRADCInputNormalizationStageConfig",
    "PatientOptimizationScientificConfig",
    "PatientPreprocessingScientificConfig",
    "PatientProstateOnlyMRADCStageConfig",
    "PatientRawContourPullingStageConfig",
    "PatientRealBiopsyProcessingStageConfig",
    "PatientRealizedBiopsyTargetingStageConfig",
    "PatientRunResult",
    "PatientRunnerMainValidationConfig",
    "PatientRunnerMainValidationMode",
    "PatientRunnerMainValidationResult",
    "PatientRunnerMainValidationSkippedResult",
    "PatientRunnerParitySurface",
    "PatientRunnerParitySurfaceResult",
    "PatientRunnerPostRunParityConfig",
    "PatientRunnerPostRunParityResult",
    "PatientRunnerScientificConfig",
    "PatientSampledBiopsyProcessingStageConfig",
    "PatientScientificPathwayName",
    "PatientScientificStageDependency",
    "PatientScientificTranche",
    "PatientScientificTrancheName",
    "PatientScientificStageResources",
    "PatientStage",
    "PatientStageName",
    "PatientStageResult",
    "PatientStageRunner",
    "PatientStageStatus",
    "PatientSimulatedBiopsyFinalizationStageConfig",
    "PatientSimulatedBiopsyPlanningStageConfig",
    "PatientSimulatedBiopsyPreparationStageConfig",
    "PatientStandardNonBiopsyStructureProcessingStageConfig",
    "PatientStructureSelectionStageConfig",
    "PatientUncertaintyAttachmentStageConfig",
    "assemble_patient_batch_cohort_tables",
    "build_patient_case_from_legacy",
    "build_patient_batch_artifact_inventory",
    "build_patient_scientific_stages",
    "build_patient_scientific_stages_for_pathway",
    "build_patient_scientific_stages_for_tranches",
    "carve_patient_runtime_state",
    "carve_patient_runtime_state_by_uid",
    "collect_patient_dataframe_artifacts",
    "compare_patient_runner_assembled_cohort_tables",
    "compare_patient_runner_recursive_csvs",
    "default_patient_stages",
    "default_patient_scientific_tranches",
    "default_patient_runner_post_run_parity_output_dir",
    "executable_patient_scientific_pathway_stage_names",
    "format_patient_runner_post_run_parity_summary",
    "get_patient_scientific_tranche",
    "iter_patient_scientific_tranches",
    "patient_batch_run_result_manifest",
    "patient_run_result_manifest",
    "patient_scientific_pathway_graph_stage_names",
    "patient_scientific_pathway_stage_names",
    "patient_scientific_tranche_stage_names",
    "resolve_legacy_patient_uids",
    "resolve_patient_scientific_pathway_name",
    "resolve_patient_scientific_stage_names",
    "resolve_patient_scientific_tranche_names",
    "resolve_patient_uids",
    "run_patient_batch",
    "run_patient_batch_cohort_assembly",
    "run_patient_batch_from_legacy",
    "run_patient_case",
    "run_patient_anatomical_preprocessing_scientific_stage",
    "run_patient_grid_preprocessing_scientific_stage",
    "run_patient_guidance_scientific_stage",
    "run_patient_mc_prep_scientific_stage",
    "run_patient_mc_simulation_scientific_stage",
    "run_patient_optimization_scientific_stage",
    "run_patient_preprocessing_scientific_stage",
    "run_patient_simulated_biopsy_finalization_scientific_stage",
    "run_patient_transform_generation_scientific_stage",
    "run_patient_runner_post_run_parity",
    "run_patient_runner_main_validation",
    "run_patient_stages",
    "summarize_patient_runner_main_validation",
    "summarize_patient_runner_parity_surface",
    "summarize_patient_runner_post_run_parity",
    "summarize_patient_runner_post_run_parity_surfaces",
    "summarize_patient_scientific_dependency_graph",
    "summarize_patient_scientific_pathways",
    "summarize_patient_scientific_tranches",
    "summarize_patient_batch_cohort_assembly",
    "summarize_patient_batch_cohort_validation",
    "validate_patient_uids",
    "validate_patient_batch_cohort_assembly",
    "validate_patient_scientific_graph_dependencies",
    "validate_patient_scientific_pathway_dependencies",
    "validate_patient_scientific_pathway_graph_dependencies",
    "validate_patient_scientific_stage_dependencies",
    "write_patient_batch_run_manifest",
    "write_patient_batch_cohort_assembly_outputs",
    "write_patient_artifacts_stage",
    "write_patient_dataframe_artifacts",
    "write_patient_runner_main_validation_summary",
    "write_patient_run_manifest",
]
