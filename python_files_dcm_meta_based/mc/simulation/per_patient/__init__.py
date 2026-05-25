"""Patient-local Monte Carlo simulation entrypoints."""

from .containment import (
    MC_CONTAINMENT_BIOPSY_OUTPUT_KEYS,
    PatientContainmentOutputs,
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
    "MCSimulationRuntimeConfig",
    "LegacyMCBiopsyIdentityKeys",
    "LegacyMCBiopsyOutputKeys",
    "LegacyMCKeyBundle",
    "LegacyMCMasterInfoKeys",
    "PatientContainmentOutputs",
    "PatientDoseOutputs",
    "build_single_patient_mc_master_info",
    "collect_mc_patient_outputs",
    "collect_patient_containment_outputs",
    "collect_patient_dose_outputs",
    "legacy_mc_keys",
    "run_patient_mc_convex_legacy_adapter",
]
