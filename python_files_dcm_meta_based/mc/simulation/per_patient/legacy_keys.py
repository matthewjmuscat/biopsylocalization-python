"""Legacy dictionary key contracts for patient-level MC simulation adapters."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class LegacyMCMasterInfoKeys:
    """Stable names for legacy master-info dictionary fields."""

    global_key: str = "Global"
    by_patient_key: str = "By patient"
    num_cases_key: str = "Num cases"
    mc_info_key: str = "MC info"
    containment_performed_key: str = "MC containment sim performed"
    dose_performed_key: str = "MC dose sim performed"
    sim_performed_key: str = "MC sim performed"


@dataclass(frozen=True, slots=True)
class LegacyMCBiopsyIdentityKeys:
    """Stable names for legacy biopsy identity/source fields."""

    roi_key: str = "ROI"
    ref_number_key: str = "Ref #"
    index_number_key: str = "Index number"
    simulated_bool_key: str = "Simulated bool"
    simulated_type_key: str = "Simulated type"


@dataclass(frozen=True, slots=True)
class LegacyMCBiopsyOutputKeys:
    """Stable names for MC artifacts stored on legacy biopsy dictionaries."""

    containment_compiled_results_dataframe_key: str = "MC data: compiled sim results dataframe"
    containment_sum_to_one_results_dataframe_key: str = "MC data: compiled sim sum-to-one results dataframe"
    containment_compiled_results_key: str = "MC data: compiled sim results"
    containment_distances_global_dataframe_key: str = "MC data: MC sim compiled distances global dataframe"
    containment_distances_point_wise_dataframe_key: str = "MC data: MC sim compiled distances point-wise dataframe"
    containment_distances_voxel_wise_dataframe_key: str = "MC data: MC sim compiled distances voxel-wise dataframe"
    containment_light_trials_dataframe_key: str = "MC data: MC sim containment and distance all trials dataframe (light)"
    dose_values_nominal_and_trials_array_key: str = "MC data: Dose vals for each sampled bx pt arr (nominal & all MC trials)"
    dose_gradient_values_nominal_and_trials_array_key: str = (
        "MC data: Dose gradient vals for each sampled bx pt arr (nominal & all MC trials)"
    )
    differential_dvh_dict_key: str = "MC data: Differential DVH dict"
    cumulative_dvh_dict_key: str = "MC data: Cumulative DVH dict"
    dose_volume_metrics_dict_key: str = "MC data: dose volume metrics dict"
    dose_statistics_mle_key: str = "MC data: Dose statistics (MLE) for each sampled bx pt list (mean, std)"
    dose_statistics_basic_key: str = "MC data: Dose statistics for each sampled bx pt list (mean, std, quantiles)"
    voxelized_dose_results_list_key: str = "MC data: voxelized dose results list"
    voxelized_dose_results_dict_key: str = "MC data: voxelized dose results dict (dict of lists)"

    @property
    def containment_output_keys(self) -> tuple[str, ...]:
        return (
            self.containment_compiled_results_dataframe_key,
            self.containment_sum_to_one_results_dataframe_key,
            self.containment_compiled_results_key,
            self.containment_distances_global_dataframe_key,
            self.containment_distances_point_wise_dataframe_key,
            self.containment_distances_voxel_wise_dataframe_key,
            self.containment_light_trials_dataframe_key,
        )

    @property
    def dose_output_keys(self) -> tuple[str, ...]:
        return (
            self.dose_values_nominal_and_trials_array_key,
            self.dose_gradient_values_nominal_and_trials_array_key,
            self.differential_dvh_dict_key,
            self.cumulative_dvh_dict_key,
            self.dose_volume_metrics_dict_key,
            self.dose_statistics_mle_key,
            self.dose_statistics_basic_key,
            self.voxelized_dose_results_list_key,
            self.voxelized_dose_results_dict_key,
        )


@dataclass(frozen=True, slots=True)
class LegacyMCDoseReferenceKeys:
    """Stable names for legacy dose-grid and plan-dose records."""

    dose_and_gradient_map_array_key: str = "Dose and gradient phys space and pixel 3d arr"
    dose_grid_point_cloud_key: str = "Dose grid point cloud"
    dose_grid_point_cloud_thresholded_key: str = "Dose grid point cloud thresholded"
    dose_grid_gradient_point_cloud_key: str = "Dose grid gradient point cloud"
    dose_grid_gradient_point_cloud_thresholded_key: str = "Dose grid gradient point cloud thresholded"
    dose_kdtree_key: str = "KDtree"
    dose_gradient_kdtree_key: str = "KDtree gradient"
    prescription_doses_dict_key: str = "Prescription doses dict"
    target_prescription_key: str = "TARGET"


@dataclass(frozen=True, slots=True)
class LegacyMCContainmentIntermediateKeys:
    """Stable names for intermediate containment state stored on legacy records."""

    bx_only_shifted_points_array_key: str = "MC data: bx only shifted 3darr"
    bx_and_structure_shifted_dict_key: str = "MC data: bx and structure shifted dict"
    normal_dist_dilations_samples_array_key: str = "MC data: Generated normal dist random samples dilations arr"
    nominal_containment_raw_dataframe_key: str = "MC data: Nominal containment raw results dataframe"
    containment_raw_dataframe_key: str = "MC data: MC sim containment raw results dataframe"


@dataclass(frozen=True, slots=True)
class LegacyMCKeyBundle:
    """Default key bundle for legacy MC adapters and output collectors."""

    master_info: LegacyMCMasterInfoKeys = field(default_factory=LegacyMCMasterInfoKeys)
    biopsy_identity: LegacyMCBiopsyIdentityKeys = field(default_factory=LegacyMCBiopsyIdentityKeys)
    biopsy_outputs: LegacyMCBiopsyOutputKeys = field(default_factory=LegacyMCBiopsyOutputKeys)
    dose_reference: LegacyMCDoseReferenceKeys = field(default_factory=LegacyMCDoseReferenceKeys)
    containment_intermediates: LegacyMCContainmentIntermediateKeys = field(
        default_factory=LegacyMCContainmentIntermediateKeys
    )


legacy_mc_keys = LegacyMCKeyBundle()
