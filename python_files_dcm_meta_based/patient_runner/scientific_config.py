"""Typed configuration boundary for patient-runner scientific stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from biopsy_optimizer.v1.per_patient import OptimizerV1LegacyConfig
    from biopsy_optimizer.v2.per_patient import OptimizerV2LiveConfig
    from guidance_maps.config import GuidanceMapPlanningConfig
    from mc.simulation.per_patient import MCConvexSimulationConfig, MCMRSimulationConfig
    from preprocessing.dose_grid_processing import DoseGridProcessingConfig
    from preprocessing.mr_adc_grid_processing import MRADCGridProcessingConfig
    from preprocessing.structure_processing.non_biopsy_structure_processing import NonBiopsyStructurePreprocessingConfig


@dataclass(frozen=True, slots=True)
class PatientScientificStageResources:
    """Runtime resources shared by opt-in scientific patient stages."""

    parallel_pool: Any = None
    rng: Any = None
    runtime_logger: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class PatientMRADCInputNormalizationStageConfig:
    """Config for patient-local MR ADC input series normalization."""

    mr_adc_ref: str
    previous_mr_adc_units: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "mr_adc_ref", _non_empty_string(self.mr_adc_ref, "mr_adc_ref"))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class PatientGridPreprocessingScientificConfig:
    """Opt-in grid preprocessing slices that run before anatomical preprocessing."""

    dose_grid_config: DoseGridProcessingConfig | None = None
    mr_adc_input_normalization: PatientMRADCInputNormalizationStageConfig | None = None
    mr_adc_grid_config: MRADCGridProcessingConfig | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _reject_grid_render_options(self.dose_grid_config)
        _reject_grid_render_options(self.mr_adc_grid_config)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def enabled(self) -> bool:
        return any(
            step is not None
            for step in (
                self.dose_grid_config,
                self.mr_adc_input_normalization,
                self.mr_adc_grid_config,
            )
        )


@dataclass(frozen=True, slots=True)
class PatientRawContourPullingStageConfig:
    """Config for patient-local RTSTRUCT raw contour pulling."""

    rtstruct_dicom_paths_by_patient_uid: Mapping[str, Any]
    structs_referenced_list_generalized: Sequence[str]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        rtstruct_paths = dict(self.rtstruct_dicom_paths_by_patient_uid)
        if not rtstruct_paths:
            raise ValueError("rtstruct_dicom_paths_by_patient_uid cannot be empty")
        object.__setattr__(self, "rtstruct_dicom_paths_by_patient_uid", rtstruct_paths)
        object.__setattr__(
            self,
            "structs_referenced_list_generalized",
            _non_empty_string_tuple(
                self.structs_referenced_list_generalized,
                "structs_referenced_list_generalized",
            ),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class PatientStructureSelectionStageConfig:
    """Config for patient-local selected/unique anatomical structure selection."""

    structs_referenced_dict: Mapping[str, Any]
    structs_referenced_list_generalized: Sequence[str]
    structs_referenced_list_generalized_unique_structs: Sequence[str]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "structs_referenced_dict", dict(self.structs_referenced_dict))
        object.__setattr__(
            self,
            "structs_referenced_list_generalized",
            _non_empty_string_tuple(
                self.structs_referenced_list_generalized,
                "structs_referenced_list_generalized",
            ),
        )
        object.__setattr__(
            self,
            "structs_referenced_list_generalized_unique_structs",
            _non_empty_string_tuple(
                self.structs_referenced_list_generalized_unique_structs,
                "structs_referenced_list_generalized_unique_structs",
            ),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class PatientStandardNonBiopsyStructureProcessingStageConfig:
    """Config for patient-local prostate/rectum/urethra/DIL preprocessing."""

    oar_ref: str
    rectum_ref_key: str
    urethra_ref_key: str
    dil_ref: str
    structs_referenced_dict: Mapping[str, Any]
    preprocessing_config: NonBiopsyStructurePreprocessingConfig
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("oar_ref", "rectum_ref_key", "urethra_ref_key", "dil_ref"):
            object.__setattr__(self, field_name, _non_empty_string(getattr(self, field_name), field_name))
        object.__setattr__(self, "structs_referenced_dict", dict(self.structs_referenced_dict))
        _reject_anatomical_presentation_options(self.preprocessing_config)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def structure_ref_types(self) -> tuple[str, str, str, str]:
        return (self.oar_ref, self.rectum_ref_key, self.urethra_ref_key, self.dil_ref)


@dataclass(frozen=True, slots=True)
class PatientProstateOnlyMRADCStageConfig:
    """Config for patient-local prostate-only MR ADC summary finalization."""

    dil_ref: str
    mr_adc_ref: str
    demonstrate_mr_adc_pcd_containment_correctness_prostate_only_all_other_structures_removed_bool: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dil_ref", _non_empty_string(self.dil_ref, "dil_ref"))
        object.__setattr__(self, "mr_adc_ref", _non_empty_string(self.mr_adc_ref, "mr_adc_ref"))
        demonstrate_flag = bool(
            self.demonstrate_mr_adc_pcd_containment_correctness_prostate_only_all_other_structures_removed_bool
        )
        if demonstrate_flag:
            raise ValueError(
                "patient-runner anatomical preprocessing does not support prostate-only MR ADC plot side effects"
            )
        object.__setattr__(
            self,
            "demonstrate_mr_adc_pcd_containment_correctness_prostate_only_all_other_structures_removed_bool",
            demonstrate_flag,
        )
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class PatientAnatomicalPreprocessingScientificConfig:
    """Opt-in anatomical preprocessing slices that run after grid preprocessing."""

    raw_contour_pulling: PatientRawContourPullingStageConfig | None = None
    structure_selection: PatientStructureSelectionStageConfig | None = None
    standard_non_biopsy_structure_processing: PatientStandardNonBiopsyStructureProcessingStageConfig | None = None
    prostate_only_mr_adc: PatientProstateOnlyMRADCStageConfig | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def enabled(self) -> bool:
        return any(
            step is not None
            for step in (
                self.raw_contour_pulling,
                self.structure_selection,
                self.standard_non_biopsy_structure_processing,
                self.prostate_only_mr_adc,
            )
        )


@dataclass(frozen=True, slots=True)
class PatientSimulatedBiopsyPreparationStageConfig:
    """Config for patient-local simulated-biopsy target, multiplicity, and length prep."""

    dil_ref: str
    simulated_biopsy_length_method: str
    biopsy_needle_compartment_length: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "dil_ref", _non_empty_string(self.dil_ref, "dil_ref"))
        object.__setattr__(
            self,
            "simulated_biopsy_length_method",
            _non_empty_string(self.simulated_biopsy_length_method, "simulated_biopsy_length_method"),
        )
        object.__setattr__(self, "biopsy_needle_compartment_length", float(self.biopsy_needle_compartment_length))


@dataclass(frozen=True, slots=True)
class PatientRealBiopsyProcessingStageConfig:
    """Config for patient-local real-biopsy geometry finalization."""

    structs_referenced_dict: Mapping[str, Any]
    interp_inter_slice_dist: float
    interp_intra_slice_dist: float
    interp_dist_caps: float
    biopsy_radius: float
    display_pca_fit_variation_for_biopsies_bool: bool
    voxel_size_for_structure_volume_calc_non_bx: float
    factor_for_voxel_size: float
    cupy_array_upper_limit_nxn_size_input: Any
    nearest_zslice_vals_and_indices_cupy_generic_max_size: Any
    generate_cuda_log_files_volume_calculation: bool
    constant_z_slice_polygons_handler_option: Any
    remove_consecutive_duplicate_points_in_polygons: bool
    include_edges_in_log_files: bool
    custom_cuda_kernel_type: Any
    demonstrate_volume_calculation_correctness_bool_1: bool
    plot_volume_calculation_containment_result_bool_1_old: bool
    plot_binary_mask_bool: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "structs_referenced_dict", dict(self.structs_referenced_dict))
        for field_name in (
            "interp_inter_slice_dist",
            "interp_intra_slice_dist",
            "interp_dist_caps",
            "biopsy_radius",
            "voxel_size_for_structure_volume_calc_non_bx",
            "factor_for_voxel_size",
        ):
            object.__setattr__(self, field_name, _positive_float(getattr(self, field_name), field_name))
        for field_name in (
            "display_pca_fit_variation_for_biopsies_bool",
            "generate_cuda_log_files_volume_calculation",
            "remove_consecutive_duplicate_points_in_polygons",
            "include_edges_in_log_files",
            "demonstrate_volume_calculation_correctness_bool_1",
            "plot_volume_calculation_containment_result_bool_1_old",
            "plot_binary_mask_bool",
        ):
            object.__setattr__(self, field_name, bool(getattr(self, field_name)))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_preprocessing_config(
        cls,
        preprocessing_config: Any,
        *,
        structs_referenced_dict: Mapping[str, Any],
        biopsy_radius: float,
        display_pca_fit_variation_for_biopsies_bool: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> "PatientRealBiopsyProcessingStageConfig":
        """Build the real-biopsy adapter config from the existing pipeline config slice."""
        return cls(
            structs_referenced_dict=structs_referenced_dict,
            interp_inter_slice_dist=preprocessing_config.interp_inter_slice_dist,
            interp_intra_slice_dist=preprocessing_config.interp_intra_slice_dist,
            interp_dist_caps=preprocessing_config.interp_dist_caps,
            biopsy_radius=biopsy_radius,
            display_pca_fit_variation_for_biopsies_bool=display_pca_fit_variation_for_biopsies_bool,
            voxel_size_for_structure_volume_calc_non_bx=(
                preprocessing_config.voxel_size_for_structure_volume_calc_non_bx
            ),
            factor_for_voxel_size=preprocessing_config.factor_for_voxel_size,
            cupy_array_upper_limit_nxn_size_input=preprocessing_config.cupy_array_upper_limit_nxn_size_input,
            nearest_zslice_vals_and_indices_cupy_generic_max_size=(
                preprocessing_config.nearest_zslice_vals_and_indices_cupy_generic_max_size
            ),
            generate_cuda_log_files_volume_calculation=(
                preprocessing_config.generate_cuda_log_files_volume_calculation
            ),
            constant_z_slice_polygons_handler_option=preprocessing_config.constant_z_slice_polygons_handler_option,
            remove_consecutive_duplicate_points_in_polygons=(
                preprocessing_config.remove_consecutive_duplicate_points_in_polygons
            ),
            include_edges_in_log_files=preprocessing_config.include_edges_in_log_files,
            custom_cuda_kernel_type=preprocessing_config.custom_cuda_kernel_type,
            demonstrate_volume_calculation_correctness_bool_1=(
                preprocessing_config.demonstrate_volume_calculation_correctness_bool_1
            ),
            plot_volume_calculation_containment_result_bool_1_old=(
                preprocessing_config.plot_volume_calculation_containment_result_bool_1_old
            ),
            plot_binary_mask_bool=preprocessing_config.plot_binary_mask_bool,
            metadata=dict(metadata or {}),
        )


@dataclass(frozen=True, slots=True)
class PatientSimulatedBiopsyFinalizationStageConfig:
    """Config for patient-local post-optimizer simulated-biopsy geometry finalization."""

    structs_referenced_dict: Mapping[str, Any]
    interp_inter_slice_dist: float
    interp_intra_slice_dist: float
    interp_dist_caps: float
    biopsy_radius: float
    voxel_size_for_structure_volume_calc_non_bx: float
    factor_for_voxel_size: float
    cupy_array_upper_limit_nxn_size_input: Any
    nearest_zslice_vals_and_indices_cupy_generic_max_size: Any
    generate_cuda_log_files_volume_calculation: bool
    constant_z_slice_polygons_handler_option: Any
    remove_consecutive_duplicate_points_in_polygons: bool
    include_edges_in_log_files: bool
    custom_cuda_kernel_type: Any
    demonstrate_volume_calculation_correctness_bool_1: bool
    plot_volume_calculation_containment_result_bool_1_old: bool
    plot_binary_mask_bool: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "structs_referenced_dict", dict(self.structs_referenced_dict))
        for field_name in (
            "interp_inter_slice_dist",
            "interp_intra_slice_dist",
            "interp_dist_caps",
            "biopsy_radius",
            "voxel_size_for_structure_volume_calc_non_bx",
            "factor_for_voxel_size",
        ):
            object.__setattr__(self, field_name, _positive_float(getattr(self, field_name), field_name))
        for field_name in (
            "generate_cuda_log_files_volume_calculation",
            "remove_consecutive_duplicate_points_in_polygons",
            "include_edges_in_log_files",
            "demonstrate_volume_calculation_correctness_bool_1",
            "plot_volume_calculation_containment_result_bool_1_old",
            "plot_binary_mask_bool",
        ):
            object.__setattr__(self, field_name, bool(getattr(self, field_name)))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_preprocessing_config(
        cls,
        preprocessing_config: Any,
        *,
        structs_referenced_dict: Mapping[str, Any],
        biopsy_radius: float,
        metadata: Mapping[str, Any] | None = None,
    ) -> "PatientSimulatedBiopsyFinalizationStageConfig":
        """Build the simulated-biopsy finalization config from the existing pipeline config slice."""
        return cls(
            structs_referenced_dict=structs_referenced_dict,
            interp_inter_slice_dist=preprocessing_config.interp_inter_slice_dist,
            interp_intra_slice_dist=preprocessing_config.interp_intra_slice_dist,
            interp_dist_caps=preprocessing_config.interp_dist_caps,
            biopsy_radius=biopsy_radius,
            voxel_size_for_structure_volume_calc_non_bx=(
                preprocessing_config.voxel_size_for_structure_volume_calc_non_bx
            ),
            factor_for_voxel_size=preprocessing_config.factor_for_voxel_size,
            cupy_array_upper_limit_nxn_size_input=preprocessing_config.cupy_array_upper_limit_nxn_size_input,
            nearest_zslice_vals_and_indices_cupy_generic_max_size=(
                preprocessing_config.nearest_zslice_vals_and_indices_cupy_generic_max_size
            ),
            generate_cuda_log_files_volume_calculation=(
                preprocessing_config.generate_cuda_log_files_volume_calculation
            ),
            constant_z_slice_polygons_handler_option=preprocessing_config.constant_z_slice_polygons_handler_option,
            remove_consecutive_duplicate_points_in_polygons=(
                preprocessing_config.remove_consecutive_duplicate_points_in_polygons
            ),
            include_edges_in_log_files=preprocessing_config.include_edges_in_log_files,
            custom_cuda_kernel_type=preprocessing_config.custom_cuda_kernel_type,
            demonstrate_volume_calculation_correctness_bool_1=(
                preprocessing_config.demonstrate_volume_calculation_correctness_bool_1
            ),
            plot_volume_calculation_containment_result_bool_1_old=(
                preprocessing_config.plot_volume_calculation_containment_result_bool_1_old
            ),
            plot_binary_mask_bool=preprocessing_config.plot_binary_mask_bool,
            metadata=dict(metadata or {}),
        )

    @property
    def enabled(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class PatientSimulatedBiopsyPlanningStageConfig:
    """Config for patient-local simulated-biopsy planning and planning samples."""

    bx_sample_pts_lattice_spacing: float
    simulated_bx_rad: float
    centroid_line_vec_sim_list: Sequence[float] = (0.0, 0.0, 1.0)
    centroid_first_pos_sim_list: Sequence[float] = (0.0, 0.0, 0.0)
    num_centroids_for_sim_bxs: int = 10
    plot_simulated_cores_immediately: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "bx_sample_pts_lattice_spacing",
            _positive_float(self.bx_sample_pts_lattice_spacing, "bx_sample_pts_lattice_spacing"),
        )
        object.__setattr__(self, "simulated_bx_rad", _positive_float(self.simulated_bx_rad, "simulated_bx_rad"))
        object.__setattr__(
            self,
            "centroid_line_vec_sim_list",
            _three_float_tuple(self.centroid_line_vec_sim_list, "centroid_line_vec_sim_list"),
        )
        object.__setattr__(
            self,
            "centroid_first_pos_sim_list",
            _three_float_tuple(self.centroid_first_pos_sim_list, "centroid_first_pos_sim_list"),
        )
        object.__setattr__(self, "num_centroids_for_sim_bxs", _positive_int(self.num_centroids_for_sim_bxs, "num_centroids_for_sim_bxs"))
        object.__setattr__(self, "plot_simulated_cores_immediately", bool(self.plot_simulated_cores_immediately))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class PatientRealizedBiopsyTargetingStageConfig:
    """Config for patient-local realized biopsy target annotation."""

    oar_ref: str
    dil_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "oar_ref", _non_empty_string(self.oar_ref, "oar_ref"))
        object.__setattr__(self, "dil_ref", _non_empty_string(self.dil_ref, "dil_ref"))


@dataclass(frozen=True, slots=True)
class PatientSampledBiopsyProcessingStageConfig:
    """Config for patient-local biopsy sample-point storage and biopsy-frame coordinates."""

    bx_sample_pts_lattice_spacing: float
    show_reconstructed_biopsy_in_biopsy_coord_sys_tr_and_rot: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "bx_sample_pts_lattice_spacing", float(self.bx_sample_pts_lattice_spacing))
        object.__setattr__(
            self,
            "show_reconstructed_biopsy_in_biopsy_coord_sys_tr_and_rot",
            bool(self.show_reconstructed_biopsy_in_biopsy_coord_sys_tr_and_rot),
        )


@dataclass(frozen=True, slots=True)
class PatientUncertaintyAttachmentStageConfig:
    """Config for attaching a resolved uncertainty dataframe to one patient."""

    read_uncertainties_dataframe: Any
    uncertainty_data_cls: type[Any]


@dataclass(frozen=True, slots=True)
class PatientPreprocessingScientificConfig:
    """Opt-in preprocessing slices that already have patient-local entrypoints."""

    real_biopsy_processing: PatientRealBiopsyProcessingStageConfig | None = None
    simulated_biopsy_preparation: PatientSimulatedBiopsyPreparationStageConfig | None = None
    simulated_biopsy_planning: PatientSimulatedBiopsyPlanningStageConfig | None = None
    realized_biopsy_targeting: PatientRealizedBiopsyTargetingStageConfig | None = None
    sampled_biopsy_processing: PatientSampledBiopsyProcessingStageConfig | None = None
    uncertainty_attachment: PatientUncertaintyAttachmentStageConfig | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def enabled(self) -> bool:
        return any(
            step is not None
            for step in (
                self.real_biopsy_processing,
                self.uncertainty_attachment,
                self.simulated_biopsy_preparation,
                self.simulated_biopsy_planning,
                self.realized_biopsy_targeting,
                self.sampled_biopsy_processing,
            )
        )


@dataclass(frozen=True, slots=True)
class PatientMCPrepScientificConfig:
    """Config for patient-local MC transform generation and application."""

    structs_referenced_list: Sequence[str]
    simulate_uniform_bx_shifts_due_to_bx_needle_compartment: bool
    biopsy_needle_compartment_length: float
    run_transform_generation: bool = True
    run_biopsy_self_transforms: bool = True
    run_relative_structure_transforms: bool = True
    num_generated_transform_samples: int | None = None
    max_simulations: int | None = None
    num_mc_containment_simulations: int | None = None
    inspect_self_biopsy_dilate_bool: bool = False
    inspect_self_biopsy_dilate_and_rotate_bool: bool = False
    inspect_self_biopsy_dilate_and_rotate_and_translate_bool: bool = False
    inspect_relative_structure_rotate_and_shift_number: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.structs_referenced_list, str):
            raise TypeError("structs_referenced_list must be a sequence of structure keys, not a string")
        object.__setattr__(
            self,
            "structs_referenced_list",
            tuple(_non_empty_string(value, "structs_referenced_list item") for value in self.structs_referenced_list),
        )
        if not self.structs_referenced_list:
            raise ValueError("structs_referenced_list cannot be empty")
        object.__setattr__(
            self,
            "simulate_uniform_bx_shifts_due_to_bx_needle_compartment",
            bool(self.simulate_uniform_bx_shifts_due_to_bx_needle_compartment),
        )
        object.__setattr__(self, "biopsy_needle_compartment_length", float(self.biopsy_needle_compartment_length))
        for field_name in (
            "run_transform_generation",
            "run_biopsy_self_transforms",
            "run_relative_structure_transforms",
            "inspect_self_biopsy_dilate_bool",
            "inspect_self_biopsy_dilate_and_rotate_bool",
            "inspect_self_biopsy_dilate_and_rotate_and_translate_bool",
        ):
            object.__setattr__(self, field_name, bool(getattr(self, field_name)))
        for field_name in (
            "num_generated_transform_samples",
            "max_simulations",
            "num_mc_containment_simulations",
        ):
            field_value = getattr(self, field_name)
            if field_value is not None:
                object.__setattr__(self, field_name, _positive_int(field_value, field_name))
        inspect_count = int(self.inspect_relative_structure_rotate_and_shift_number)
        if inspect_count < 0:
            raise ValueError("inspect_relative_structure_rotate_and_shift_number cannot be negative")
        object.__setattr__(self, "inspect_relative_structure_rotate_and_shift_number", inspect_count)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def enabled(self) -> bool:
        return self.transform_generation_enabled or self.transform_application_enabled

    @property
    def transform_generation_enabled(self) -> bool:
        return self.run_transform_generation

    @property
    def transform_application_enabled(self) -> bool:
        return any(
            (
                self.run_biopsy_self_transforms,
                self.run_relative_structure_transforms,
            )
        )


@dataclass(frozen=True, slots=True)
class PatientMCSimulationScientificConfig:
    """Config for patient-local convex MC and MR ADC simulation stages."""

    convex_config: MCConvexSimulationConfig | None = None
    mr_config: MCMRSimulationConfig | None = None
    mr_adc_ref: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mr_config is not None:
            object.__setattr__(self, "mr_adc_ref", _non_empty_string(self.mr_adc_ref, "mr_adc_ref"))
        else:
            object.__setattr__(self, "mr_adc_ref", str(self.mr_adc_ref).strip())
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def enabled(self) -> bool:
        return self.convex_config is not None or self.mr_config is not None


@dataclass(frozen=True, slots=True)
class PatientOptimizationScientificConfig:
    """Config for patient-local optimizer stages."""

    optimizer_v1_config: OptimizerV1LegacyConfig | None = None
    optimizer_v2_config: OptimizerV2LiveConfig | None = None
    optimizer_v2_resolved_max_test_structures_per_call: int | None = None
    optimizer_v2_resolved_max_candidates_per_chunk: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "optimizer_v2_resolved_max_test_structures_per_call",
            "optimizer_v2_resolved_max_candidates_per_chunk",
        ):
            field_value = getattr(self, field_name)
            if field_value is not None:
                object.__setattr__(self, field_name, _positive_int(field_value, field_name))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def enabled(self) -> bool:
        return self.optimizer_v1_config is not None or self.optimizer_v2_config is not None


@dataclass(frozen=True, slots=True)
class PatientGuidanceScientificConfig:
    """Config for patient-local guidance-map firing-depth precompute."""

    dil_ref: str
    oar_ref: str
    rectum_ref: str
    biopsy_fire_travel_distances: Any
    biopsy_needle_compartment_length: float
    interp_inter_slice_dist: float
    interp_intra_slice_dist: float
    radius_for_normals_estimation: float
    max_nn_for_normals_estimation: int
    biopsy_needle_tip_length: float
    planning_config: GuidanceMapPlanningConfig | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("dil_ref", "oar_ref", "rectum_ref"):
            object.__setattr__(self, field_name, _non_empty_string(getattr(self, field_name), field_name))
        for field_name in (
            "biopsy_needle_compartment_length",
            "interp_inter_slice_dist",
            "interp_intra_slice_dist",
            "radius_for_normals_estimation",
            "biopsy_needle_tip_length",
        ):
            object.__setattr__(self, field_name, float(getattr(self, field_name)))
        object.__setattr__(
            self,
            "max_nn_for_normals_estimation",
            _positive_int(self.max_nn_for_normals_estimation, "max_nn_for_normals_estimation"),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def enabled(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class PatientRunnerScientificConfig:
    """Top-level opt-in scientific stage config for patient-runner execution."""

    resources: PatientScientificStageResources = field(default_factory=PatientScientificStageResources)
    grid_preprocessing: PatientGridPreprocessingScientificConfig | None = None
    anatomical_preprocessing: PatientAnatomicalPreprocessingScientificConfig | None = None
    preprocessing: PatientPreprocessingScientificConfig | None = None
    mc_prep: PatientMCPrepScientificConfig | None = None
    mc_simulation: PatientMCSimulationScientificConfig | None = None
    optimization: PatientOptimizationScientificConfig | None = None
    simulated_biopsy_finalization: PatientSimulatedBiopsyFinalizationStageConfig | None = None
    guidance: PatientGuidanceScientificConfig | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.resources, PatientScientificStageResources):
            raise TypeError("resources must be a PatientScientificStageResources instance")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def enabled_stage_names(self) -> tuple[str, ...]:
        enabled_names = []
        for name, stage_config in (
            ("grid_preprocessing", self.grid_preprocessing),
            ("anatomical_preprocessing", self.anatomical_preprocessing),
            ("preprocessing", self.preprocessing),
            ("mc_prep", self.mc_prep),
            ("mc_simulation", self.mc_simulation),
            ("optimization", self.optimization),
            ("simulated_biopsy_finalization", self.simulated_biopsy_finalization),
            ("guidance", self.guidance),
        ):
            if stage_config is not None and stage_config.enabled:
                enabled_names.append(name)
        return tuple(enabled_names)


def _non_empty_string(value: Any, field_name: str) -> str:
    resolved_value = str(value).strip()
    if resolved_value == "":
        raise ValueError(f"{field_name} cannot be empty")
    return resolved_value


def _non_empty_string_tuple(values: Sequence[Any], field_name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{field_name} must be a sequence of strings, not a string")
    resolved_values = tuple(_non_empty_string(value, f"{field_name} item") for value in values)
    if not resolved_values:
        raise ValueError(f"{field_name} cannot be empty")
    return resolved_values


def _positive_int(value: Any, field_name: str) -> int:
    resolved_value = int(value)
    if resolved_value < 1:
        raise ValueError(f"{field_name} must be at least 1")
    return resolved_value


def _reject_grid_render_options(config: Any) -> None:
    if config is None:
        return
    render_fields = (
        "show_3d_dose_renderings",
        "show_3d_dose_renderings_thresholded",
        "show_3d_mr_adc_renderings",
        "show_3d_mr_adc_renderings_thresholded",
    )
    enabled_fields = tuple(
        field_name
        for field_name in render_fields
        if bool(getattr(config, field_name, False))
    )
    if enabled_fields:
        raise ValueError(
            "patient-runner grid preprocessing does not support render side effects: "
            + ", ".join(enabled_fields)
        )


def _reject_anatomical_presentation_options(config: Any) -> None:
    presentation_fields = (
        "demonstrate_volume_calculation_correctness_bool_1",
        "plot_volume_calculation_containment_result_bool_1_old",
        "plot_binary_mask_bool",
        "demonstrate_structure_dimension_calculation_correctness_bool_1",
        "demonstrate_structure_dimension_calculation_correctness_bool_1_old",
        "demonstrate_mr_adc_pcd_containment_correctness_bool",
        "display_structure_surface_mesh_bool",
        "show_equivalent_ellipsoid_from_pca_bool",
    )
    enabled_fields = tuple(
        field_name
        for field_name in presentation_fields
        if bool(getattr(config, field_name, False))
    )
    if enabled_fields:
        raise ValueError(
            "patient-runner anatomical preprocessing does not support presentation/debug side effects: "
            + ", ".join(enabled_fields)
        )


def _positive_float(value: Any, field_name: str) -> float:
    resolved_value = float(value)
    if resolved_value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return resolved_value


def _three_float_tuple(values: Sequence[Any], field_name: str) -> tuple[float, float, float]:
    resolved_values = tuple(float(value) for value in values)
    if len(resolved_values) != 3:
        raise ValueError(f"{field_name} must contain exactly three values")
    return resolved_values