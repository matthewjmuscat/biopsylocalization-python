"""Shared legacy dictionary key contracts for additive migration modules."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class LegacyMasterInfoKeys:
    """Stable names for legacy master-structure-info dictionary fields."""

    global_key: str = "Global"
    by_patient_key: str = "By patient"
    num_cases_key: str = "Num cases"
    num_structures_key: str = "Num structures"
    num_biopsies_key: str = "Num biopsies"
    num_unique_patient_names_key: str = "Num unique patient names"
    num_biopsies_by_type_key: str = "Num biopsies by bx type dict"
    num_dils_key: str = "Num DILs"
    bx_types_list_key: str = "Bx types list"
    preprocessing_info_key: str = "Preprocessing info"
    mc_info_key: str = "MC info"
    specific_output_dir_key: str = "Specific output dir"
    raw_mc_output_dir_key: str = "Raw MC output dir"
    run_output_folder_label_key: str = "Run output folder label"
    run_output_metadata_key: str = "Run output metadata"


@dataclass(frozen=True, slots=True)
class LegacyPatientReferenceKeys:
    """Stable names for patient-level legacy reference/info dictionaries."""

    patient_uid_generated_key: str = "Patient UID (generated)"
    patient_id_from_dicom_key: str = "Patient ID (from dicom)"
    patient_name_key: str = "Patient Name"
    fraction_number_key: str = "Fraction number"
    ready_to_plot_data_list_key: str = "Ready to plot data list"


@dataclass(frozen=True, slots=True)
class LegacyStructureRecordKeys:
    """Stable names for legacy per-structure record fields."""

    roi_key: str = "ROI"
    ref_number_key: str = "Ref #"
    index_number_key: str = "Index number"
    simulated_bool_key: str = "Simulated bool"
    simulated_type_key: str = "Simulated type"


@dataclass(frozen=True, slots=True)
class LegacyStructureInfoKeys:
    """Stable names for legacy per-family structure-count dictionaries."""

    num_structs_key: str = "Num structs"
    num_sim_structs_key: str = "Num sim structs"
    num_real_structs_key: str = "Num real structs"
    biopsy_type_counts_key: str = "Biopsy type counts"
    total_num_structs_key: str = "Total num structs"


@dataclass(frozen=True, slots=True)
class LegacyPatientAllReferenceKeys:
    """Stable names for legacy all-reference nested stores."""

    multi_structure_information_key: str = "Multi-structure information dict (not for csv output)"
    preprocessing_output_dataframes_key: str = "Multi-structure pre-processing output dataframes dict"
    mc_output_dataframes_key: str = "Multi-structure MC simulation output dataframes dict"


@dataclass(frozen=True, slots=True)
class LegacyBiopsyRuntimeKeys:
    """Stable names for legacy biopsy runtime sidecar fields."""

    output_dataframes_key: str = "Output data frames"
    simulated_biopsy_transport_request_key: str = "Simulated biopsy transport request dict"


@dataclass(frozen=True, slots=True)
class LegacyArtifactKeys:
    """Stable names for legacy output artifact sentinels."""

    global_patient_uid: str = "Global"


@dataclass(frozen=True, slots=True)
class LegacyDataKeyBundle:
    """Default key bundle for additive legacy-data adapters."""

    master_info: LegacyMasterInfoKeys = field(default_factory=LegacyMasterInfoKeys)
    patient_reference: LegacyPatientReferenceKeys = field(default_factory=LegacyPatientReferenceKeys)
    structure_record: LegacyStructureRecordKeys = field(default_factory=LegacyStructureRecordKeys)
    structure_info: LegacyStructureInfoKeys = field(default_factory=LegacyStructureInfoKeys)
    patient_all_reference: LegacyPatientAllReferenceKeys = field(default_factory=LegacyPatientAllReferenceKeys)
    biopsy_runtime: LegacyBiopsyRuntimeKeys = field(default_factory=LegacyBiopsyRuntimeKeys)
    artifacts: LegacyArtifactKeys = field(default_factory=LegacyArtifactKeys)


legacy_data_keys = LegacyDataKeyBundle()