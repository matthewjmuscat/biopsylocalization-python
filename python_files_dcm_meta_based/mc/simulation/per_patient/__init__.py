"""Patient-local Monte Carlo simulation entrypoints."""

from .containment import (
    MC_CONTAINMENT_BIOPSY_OUTPUT_KEYS,
    MC_STRUCTURE_SPECIFIC_RESULT_KEYS,
    PatientContainmentDilatedStructureBank,
    PatientContainmentOutputs,
    PatientContainmentStructureInventory,
    build_mutual_structure_specific_results_template,
    build_patient_containment_dilated_structure_bank,
    build_patient_relative_structure_inventory,
    build_structure_specific_results_template,
    collect_patient_containment_outputs,
)
from .contracts import (
    MCContainmentSimulationConfig,
    MCConvexPatientRunResult,
    MCConvexSimulationConfig,
    MCDoseSimulationConfig,
    MCReferenceKeys,
    MCSimulationRuntimeConfig,
)
from .convex_legacy_adapter import (
    build_single_patient_mc_master_info,
    collect_mc_patient_outputs,
    run_patient_mc_convex_legacy_adapter,
)
from .dose import MC_DOSE_BIOPSY_OUTPUT_KEYS, PatientDoseOutputs, collect_patient_dose_outputs
from .legacy_keys import (
    LegacyMCBiopsyIdentityKeys,
    LegacyMCBiopsyOutputKeys,
    LegacyMCContainmentIntermediateKeys,
    LegacyMCKeyBundle,
    LegacyMCMasterInfoKeys,
    legacy_mc_keys,
)

__all__ = [
    "MCContainmentSimulationConfig",
    "MCConvexPatientRunResult",
    "MCConvexSimulationConfig",
    "MCDoseSimulationConfig",
    "MC_CONTAINMENT_BIOPSY_OUTPUT_KEYS",
    "MC_DOSE_BIOPSY_OUTPUT_KEYS",
    "MCReferenceKeys",
    "MC_STRUCTURE_SPECIFIC_RESULT_KEYS",
    "MCSimulationRuntimeConfig",
    "LegacyMCBiopsyIdentityKeys",
    "LegacyMCBiopsyOutputKeys",
    "LegacyMCContainmentIntermediateKeys",
    "LegacyMCKeyBundle",
    "LegacyMCMasterInfoKeys",
    "PatientContainmentDilatedStructureBank",
    "PatientContainmentOutputs",
    "PatientContainmentStructureInventory",
    "PatientDoseOutputs",
    "build_single_patient_mc_master_info",
    "build_mutual_structure_specific_results_template",
    "build_patient_containment_dilated_structure_bank",
    "build_patient_relative_structure_inventory",
    "build_structure_specific_results_template",
    "collect_mc_patient_outputs",
    "collect_patient_containment_outputs",
    "collect_patient_dose_outputs",
    "legacy_mc_keys",
    "run_patient_mc_convex_legacy_adapter",
]
