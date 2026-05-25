"""Typed contracts for patient-local Monte Carlo simulation modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from presentation import LegacyPresentationContext


@dataclass(frozen=True, slots=True)
class MCReferenceKeys:
    """Legacy reference keys required by convex MC simulation."""

    structs_referenced_list: Sequence[str]
    structs_referenced_dict: Mapping[str, Any]
    bx_ref: str
    oar_ref: str
    dil_ref: str
    rectum_ref: str
    urethra_ref: str
    dose_ref: str
    plan_ref: str
    all_ref_key: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "structs_referenced_list", tuple(self.structs_referenced_list))
        object.__setattr__(self, "structs_referenced_dict", dict(self.structs_referenced_dict))
        for field_name in (
            "bx_ref",
            "oar_ref",
            "dil_ref",
            "rectum_ref",
            "urethra_ref",
            "dose_ref",
            "plan_ref",
            "all_ref_key",
        ):
            field_value = str(getattr(self, field_name)).strip()
            if field_value == "":
                raise ValueError(f"{field_name} cannot be empty")
            object.__setattr__(self, field_name, field_value)


@dataclass(frozen=True, slots=True)
class MCSimulationRuntimeConfig:
    """Runtime and diagnostic options shared across convex MC sub-stages."""

    biopsy_needle_compartment_length: float
    simulate_uniform_bx_shifts_due_to_bx_needle_compartment: bool
    plot_uniform_shifts_to_check_plotly: bool
    plot_translation_vectors_pointclouds: bool
    plot_shifted_biopsies: bool
    spinner_type: str
    cupy_array_upper_limit_NxN_size_input: int
    nearest_zslice_vals_and_indices_cupy_generic_max_size: int
    custom_cuda_kernel_type: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "spinner_type", str(self.spinner_type))
        object.__setattr__(self, "custom_cuda_kernel_type", str(self.custom_cuda_kernel_type))


@dataclass(frozen=True, slots=True)
class MCContainmentSimulationConfig:
    """Configuration for patient-local containment simulation."""

    containment_views_jsons_paths_list: Sequence[Any]
    show_num_containment_demonstration_plots: int
    containment_results_structure_types_to_show_per_trial: Sequence[str]
    show_num_nearest_neighbour_surface_boundary_demonstration: int
    show_num_relative_structure_centroid_demonstration: int
    plot_cupy_containment_distribution_results: bool
    structure_miss_probability_roi: str
    cancer_tissue_label: str
    default_exterior_tissue: str
    miss_structure_complement_label: str
    tissue_length_above_probability_threshold_list: Sequence[float]
    n_bootstraps_for_tissue_length_above_threshold: int
    perform_mc_containment_sim: bool
    raw_data_mc_containment_dump_bool: bool
    keep_light_containment_and_distances_to_relative_structures_dataframe_bool: bool
    show_non_bx_relative_structure_z_dilation_bool: bool
    show_non_bx_relative_structure_xy_dilation_bool: bool
    generate_cuda_log_files_MC_containment_sim: bool
    constant_z_slice_polygons_handler_option: str
    remove_consecutive_duplicate_points_in_polygons: bool
    interp_dist_caps: float
    cuml_NN_algo: str
    check_if_end_caps_filled_proper_NN_num: int
    nn_search_end_cap_grid_factor: float
    tissue_volume_operator_dictionary: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "containment_views_jsons_paths_list", tuple(self.containment_views_jsons_paths_list))
        object.__setattr__(
            self,
            "containment_results_structure_types_to_show_per_trial",
            tuple(self.containment_results_structure_types_to_show_per_trial),
        )
        object.__setattr__(
            self,
            "tissue_length_above_probability_threshold_list",
            tuple(self.tissue_length_above_probability_threshold_list),
        )
        object.__setattr__(self, "tissue_volume_operator_dictionary", dict(self.tissue_volume_operator_dictionary))
        for field_name in (
            "structure_miss_probability_roi",
            "cancer_tissue_label",
            "default_exterior_tissue",
            "miss_structure_complement_label",
            "constant_z_slice_polygons_handler_option",
            "cuml_NN_algo",
        ):
            object.__setattr__(self, field_name, str(getattr(self, field_name)))


@dataclass(frozen=True, slots=True)
class MCDoseSimulationConfig:
    """Configuration for patient-local dose and dose-gradient localization."""

    biopsy_z_voxel_length: float
    num_dose_calc_NN: int
    num_dose_NN_to_show_for_animation_plotting: int
    dose_views_jsons_paths_list: Sequence[Any]
    show_NN_dose_demonstration_plots: bool
    show_NN_dose_demonstration_plots_all_trials_at_once: bool
    differential_dvh_resolution: float
    cumulative_dvh_resolution: float
    v_percent_DVH_to_calc_list: Sequence[float]
    volume_DVH_quantiles_to_calculate: Sequence[float]
    perform_mc_dose_sim: bool
    idw_power: float
    raw_data_mc_dosimetry_dump_bool: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "dose_views_jsons_paths_list", tuple(self.dose_views_jsons_paths_list))
        object.__setattr__(self, "v_percent_DVH_to_calc_list", tuple(self.v_percent_DVH_to_calc_list))
        object.__setattr__(self, "volume_DVH_quantiles_to_calculate", tuple(self.volume_DVH_quantiles_to_calculate))


@dataclass(frozen=True, slots=True)
class MCConvexSimulationConfig:
    """Complete typed boundary for the current convex MC simulator oracle."""

    keys: MCReferenceKeys
    runtime: MCSimulationRuntimeConfig
    containment: MCContainmentSimulationConfig
    dose: MCDoseSimulationConfig


@dataclass(slots=True)
class MCConvexPatientRunResult:
    """Output bundle from running convex MC simulation against one patient."""

    patient_uid: str
    patient_reference_dict: dict[str, Any]
    master_structure_reference_dict: dict[str, dict[str, Any]]
    master_structure_info_dict: dict[str, Any]
    containment_outputs: Any
    dose_outputs: Any
    presentation_context: LegacyPresentationContext
    live_display: Any = None
    performed_flags: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.patient_uid = str(self.patient_uid)
        self.performed_flags = dict(self.performed_flags or {})
        self.metadata = dict(self.metadata or {})
