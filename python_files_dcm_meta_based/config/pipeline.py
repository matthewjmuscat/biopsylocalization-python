from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from biopsy_optimizer.v2.config import OptimizerV2SearchConfig
from guidance_maps.config import GuidanceMapPlanningConfig
from preprocessing.structure_processing.non_biopsy_structure_processing import (
    NonBiopsyStructurePreprocessingConfig,
)
from startup.guidance_map_workflow import GuidanceMapRenderConfig


def _non_empty_string(value: Any, field_name: str) -> str:
    resolved_value = str(value).strip()
    if resolved_value == "":
        raise ValueError(f"{field_name} cannot be empty")
    return resolved_value


def _tuple_from_sequence(values: Sequence[Any], field_name: str) -> tuple[Any, ...]:
    if isinstance(values, str):
        raise TypeError(f"{field_name} must be a sequence, not a string")
    return tuple(values)


@dataclass(frozen=True)
class RuntimeUIConfig:
    spinner_type: str = "moon"
    rich_live_display_bool: bool = True


@dataclass(frozen=True)
class LegacyReferenceConfig:
    all_ref_key: str = "All ref"
    bx_ref: str = "Bx ref"
    by_patient_key: str = "By patient"
    global_key: str = "Global"
    global_num_cases_key: str = "Num cases"
    oar_ref: str = "OAR ref"
    dil_ref: str = "DIL ref"
    rectum_ref_key: str = "Rectum ref"
    urethra_ref_key: str = "Urethra ref"
    dose_ref: str = "Dose ref"
    plan_ref: str = "Plan ref"
    mr_adc_ref: str = "MR ADC ref"
    mr_t2_ref: str = "MR T2 ref"
    us_ref: str = "US ref"

    def __post_init__(self) -> None:
        for field_name in (
            "all_ref_key",
            "bx_ref",
            "by_patient_key",
            "global_key",
            "global_num_cases_key",
            "oar_ref",
            "dil_ref",
            "rectum_ref_key",
            "urethra_ref_key",
            "dose_ref",
            "plan_ref",
            "mr_adc_ref",
            "mr_t2_ref",
            "us_ref",
        ):
            object.__setattr__(self, field_name, _non_empty_string(getattr(self, field_name), field_name))


@dataclass(frozen=True)
class StructureRegistryConfig:
    structs_referenced_dict: Mapping[str, Any] = field(default_factory=dict)
    structs_referenced_list: Sequence[str] = ()
    structs_referenced_list_generalized: Sequence[str] = ()
    structs_referenced_list_generalized_unique_structs: Sequence[str] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "structs_referenced_dict", dict(self.structs_referenced_dict))
        for field_name in (
            "structs_referenced_list",
            "structs_referenced_list_generalized",
            "structs_referenced_list_generalized_unique_structs",
        ):
            values = _tuple_from_sequence(getattr(self, field_name), field_name)
            object.__setattr__(self, field_name, tuple(str(value) for value in values))


@dataclass(frozen=True)
class GridPreprocessingConfig:
    show_3d_dose_renderings: bool = False
    show_3d_dose_renderings_thresholded: bool = False
    show_3d_mr_adc_renderings: bool = False
    show_3d_mr_adc_renderings_thresholded: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "show_3d_dose_renderings",
            "show_3d_dose_renderings_thresholded",
            "show_3d_mr_adc_renderings",
            "show_3d_mr_adc_renderings_thresholded",
        ):
            object.__setattr__(self, field_name, bool(getattr(self, field_name)))


@dataclass(frozen=True)
class BiopsyGeometryConfig:
    biopsy_radius: float = 0.5
    simulated_biopsy_planning_radius_mm: float = 0.5
    biopsy_fire_travel_distances: Sequence[float] = (15, 22)
    biopsy_needle_tip_length: float = 6
    display_pca_fit_variation_for_biopsies_bool: bool = False

    def __post_init__(self) -> None:
        if self.biopsy_radius <= 0:
            raise ValueError("biopsy_radius must be positive")
        if self.simulated_biopsy_planning_radius_mm <= 0:
            raise ValueError("simulated_biopsy_planning_radius_mm must be positive")
        if self.biopsy_needle_tip_length <= 0:
            raise ValueError("biopsy_needle_tip_length must be positive")
        object.__setattr__(self, "biopsy_fire_travel_distances", tuple(self.biopsy_fire_travel_distances))
        object.__setattr__(
            self,
            "display_pca_fit_variation_for_biopsies_bool",
            bool(self.display_pca_fit_variation_for_biopsies_bool),
        )


@dataclass(frozen=True)
class SimulatedBiopsyConfig:
    optimizer_simulated_type: str = "Target DIL v2"
    simulated_biopsy_length_method: str = "match real"
    centroid_line_vec_sim_list: Sequence[float] = (0, 0, 1)
    centroid_first_pos_sim_list: Sequence[float] = (0, 0, 0)
    num_centroids_for_sim_bxs: int = 10
    plot_simulated_cores_immediately: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "optimizer_simulated_type", _non_empty_string(self.optimizer_simulated_type, "optimizer_simulated_type"))
        object.__setattr__(
            self,
            "simulated_biopsy_length_method",
            _non_empty_string(self.simulated_biopsy_length_method, "simulated_biopsy_length_method"),
        )
        object.__setattr__(self, "centroid_line_vec_sim_list", tuple(self.centroid_line_vec_sim_list))
        object.__setattr__(self, "centroid_first_pos_sim_list", tuple(self.centroid_first_pos_sim_list))
        if len(self.centroid_line_vec_sim_list) != 3:
            raise ValueError("centroid_line_vec_sim_list must contain exactly three values")
        if len(self.centroid_first_pos_sim_list) != 3:
            raise ValueError("centroid_first_pos_sim_list must contain exactly three values")
        num_centroids = int(self.num_centroids_for_sim_bxs)
        if num_centroids < 1:
            raise ValueError("num_centroids_for_sim_bxs must be at least 1")
        object.__setattr__(self, "num_centroids_for_sim_bxs", num_centroids)
        object.__setattr__(self, "plot_simulated_cores_immediately", bool(self.plot_simulated_cores_immediately))


@dataclass(frozen=True)
class SamplingClassificationConfig:
    show_reconstructed_biopsy_in_biopsy_coord_sys_tr_and_rot: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "show_reconstructed_biopsy_in_biopsy_coord_sys_tr_and_rot",
            bool(self.show_reconstructed_biopsy_in_biopsy_coord_sys_tr_and_rot),
        )


@dataclass(frozen=True)
class BiopsyRuntimeConfig:
    geometry: BiopsyGeometryConfig = field(default_factory=BiopsyGeometryConfig)
    simulated: SimulatedBiopsyConfig = field(default_factory=SimulatedBiopsyConfig)
    sampling: SamplingClassificationConfig = field(default_factory=SamplingClassificationConfig)


@dataclass(frozen=True)
class ArtifactConfig:
    output_folder_name: str = "Output data"
    preprocessed_data_folder_name: str = "Preprocessed data"
    preprocessed_reference_dict_filename: str = "master_structure_reference_dict"
    preprocessed_info_dict_filename: str = "master_structure_info_dict"
    export_pickled_preprocessed_data: bool = False
    skip_preprocessing: bool = False


FROZEN_PREPROCESSED_BUNDLE_CONFIG_METADATA_KEY = "Frozen preprocessed bundle config"
_FROZEN_PREPROCESSED_BUNDLE_CONFIG_FIELD_NAMES = (
    "interp_inter_slice_dist",
    "interp_intra_slice_dist",
    "radius_for_normals_estimation",
    "max_nn_for_normals_estimation",
)


@dataclass(frozen=True)
class FrozenPreprocessedBundleConfig:
    interp_inter_slice_dist: float
    interp_intra_slice_dist: float
    radius_for_normals_estimation: float
    max_nn_for_normals_estimation: int

    def __post_init__(self) -> None:
        if self.interp_inter_slice_dist <= 0:
            raise ValueError("interp_inter_slice_dist must be positive")
        if self.interp_intra_slice_dist <= 0:
            raise ValueError("interp_intra_slice_dist must be positive")
        if self.radius_for_normals_estimation <= 0:
            raise ValueError("radius_for_normals_estimation must be positive")
        if self.max_nn_for_normals_estimation <= 0:
            raise ValueError("max_nn_for_normals_estimation must be positive")

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            field_name: getattr(self, field_name)
            for field_name in _FROZEN_PREPROCESSED_BUNDLE_CONFIG_FIELD_NAMES
        }

    @classmethod
    def from_metadata_dict(cls, config_dict: dict[str, Any]) -> "FrozenPreprocessedBundleConfig":
        missing_field_names = [
            field_name
            for field_name in _FROZEN_PREPROCESSED_BUNDLE_CONFIG_FIELD_NAMES
            if field_name not in config_dict
        ]
        if missing_field_names:
            raise ValueError(
                "Frozen preprocessed bundle config metadata is missing fields: "
                + ", ".join(missing_field_names)
            )

        return cls(
            **{
                field_name: config_dict[field_name]
                for field_name in _FROZEN_PREPROCESSED_BUNDLE_CONFIG_FIELD_NAMES
            }
        )

    def diff(self, other: "FrozenPreprocessedBundleConfig") -> dict[str, dict[str, Any]]:
        mismatch_dict = {}
        for field_name in _FROZEN_PREPROCESSED_BUNDLE_CONFIG_FIELD_NAMES:
            bundle_value = getattr(self, field_name)
            runtime_value = getattr(other, field_name)
            if bundle_value == runtime_value:
                continue
            mismatch_dict[field_name] = {
                "bundle_value": bundle_value,
                "runtime_value": runtime_value,
            }
        return mismatch_dict


@dataclass(frozen=True)
class RuntimeReplayConfig:
    lower_bound_dose_value: Optional[float]
    lower_bound_dose_gradient_value: float
    lower_bound_mr_adc_value: Optional[float]
    upper_bound_mr_adc_value: Optional[float]
    color_flattening_deg_mr: float

    def __post_init__(self) -> None:
        if self.lower_bound_dose_gradient_value < 0:
            raise ValueError("lower_bound_dose_gradient_value cannot be negative")
        if self.color_flattening_deg_mr <= 0:
            raise ValueError("color_flattening_deg_mr must be positive")
        if (
            self.lower_bound_mr_adc_value is not None
            and self.upper_bound_mr_adc_value is not None
            and self.lower_bound_mr_adc_value > self.upper_bound_mr_adc_value
        ):
            raise ValueError("lower_bound_mr_adc_value cannot exceed upper_bound_mr_adc_value")


@dataclass(frozen=True)
class PreprocessingInterpolationConfig:
    interp_inter_slice_dist: float
    interp_intra_slice_dist: float
    interp_dist_caps: float


@dataclass(frozen=True)
class PreprocessingGeometryConfig:
    radius_for_normals_estimation: float
    max_nn_for_normals_estimation: int
    voxel_size_for_structure_volume_calc_non_bx: float
    voxel_size_for_structure_dimension_calc: float
    factor_for_voxel_size: float


@dataclass(frozen=True)
class PreprocessingKernelExecutionConfig:
    cupy_array_upper_limit_nxn_size_input: Any
    nearest_zslice_vals_and_indices_cupy_generic_max_size: Any
    constant_z_slice_polygons_handler_option: Any
    remove_consecutive_duplicate_points_in_polygons: bool
    include_edges_in_log_files: bool
    custom_cuda_kernel_type: Any


@dataclass(frozen=True)
class PreprocessingDebugConfig:
    generate_cuda_log_files_volume_calculation: bool = False
    demonstrate_volume_calculation_correctness_bool_1: bool = False
    plot_volume_calculation_containment_result_bool_1_old: bool = False
    plot_binary_mask_bool: bool = False
    generate_cuda_log_files_structure_dimension_calculation: bool = False
    demonstrate_structure_dimension_calculation_correctness_bool_1: bool = False
    demonstrate_structure_dimension_calculation_correctness_bool_1_old: bool = False
    demonstrate_mr_adc_pcd_containment_correctness_bool: bool = False
    demonstrate_mr_adc_pcd_containment_correctness_prostate_only_all_other_structures_removed_bool: bool = False
    display_structure_surface_mesh_bool: bool = False
    show_equivalent_ellipsoid_from_pca_bool: bool = False


@dataclass(frozen=True)
class PreprocessingConfig:
    interpolation: PreprocessingInterpolationConfig
    geometry: PreprocessingGeometryConfig
    kernel_execution: PreprocessingKernelExecutionConfig
    debug: PreprocessingDebugConfig = field(default_factory=PreprocessingDebugConfig)

    @property
    def interp_inter_slice_dist(self) -> float:
        return self.interpolation.interp_inter_slice_dist

    @property
    def interp_intra_slice_dist(self) -> float:
        return self.interpolation.interp_intra_slice_dist

    @property
    def interp_dist_caps(self) -> float:
        return self.interpolation.interp_dist_caps

    @property
    def radius_for_normals_estimation(self) -> float:
        return self.geometry.radius_for_normals_estimation

    @property
    def max_nn_for_normals_estimation(self) -> int:
        return self.geometry.max_nn_for_normals_estimation

    @property
    def voxel_size_for_structure_volume_calc_non_bx(self) -> float:
        return self.geometry.voxel_size_for_structure_volume_calc_non_bx

    @property
    def voxel_size_for_structure_dimension_calc(self) -> float:
        return self.geometry.voxel_size_for_structure_dimension_calc

    @property
    def factor_for_voxel_size(self) -> float:
        return self.geometry.factor_for_voxel_size

    @property
    def cupy_array_upper_limit_nxn_size_input(self) -> Any:
        return self.kernel_execution.cupy_array_upper_limit_nxn_size_input

    @property
    def nearest_zslice_vals_and_indices_cupy_generic_max_size(self) -> Any:
        return self.kernel_execution.nearest_zslice_vals_and_indices_cupy_generic_max_size

    @property
    def constant_z_slice_polygons_handler_option(self) -> Any:
        return self.kernel_execution.constant_z_slice_polygons_handler_option

    @property
    def remove_consecutive_duplicate_points_in_polygons(self) -> bool:
        return self.kernel_execution.remove_consecutive_duplicate_points_in_polygons

    @property
    def include_edges_in_log_files(self) -> bool:
        return self.kernel_execution.include_edges_in_log_files

    @property
    def custom_cuda_kernel_type(self) -> Any:
        return self.kernel_execution.custom_cuda_kernel_type

    @property
    def generate_cuda_log_files_volume_calculation(self) -> bool:
        return self.debug.generate_cuda_log_files_volume_calculation

    @property
    def demonstrate_volume_calculation_correctness_bool_1(self) -> bool:
        return self.debug.demonstrate_volume_calculation_correctness_bool_1

    @property
    def plot_volume_calculation_containment_result_bool_1_old(self) -> bool:
        return self.debug.plot_volume_calculation_containment_result_bool_1_old

    @property
    def plot_binary_mask_bool(self) -> bool:
        return self.debug.plot_binary_mask_bool

    @property
    def generate_cuda_log_files_structure_dimension_calculation(self) -> bool:
        return self.debug.generate_cuda_log_files_structure_dimension_calculation

    @property
    def demonstrate_structure_dimension_calculation_correctness_bool_1(self) -> bool:
        return self.debug.demonstrate_structure_dimension_calculation_correctness_bool_1

    @property
    def demonstrate_structure_dimension_calculation_correctness_bool_1_old(self) -> bool:
        return self.debug.demonstrate_structure_dimension_calculation_correctness_bool_1_old

    @property
    def demonstrate_mr_adc_pcd_containment_correctness_bool(self) -> bool:
        return self.debug.demonstrate_mr_adc_pcd_containment_correctness_bool

    @property
    def demonstrate_mr_adc_pcd_containment_correctness_prostate_only_all_other_structures_removed_bool(self) -> bool:
        return self.debug.demonstrate_mr_adc_pcd_containment_correctness_prostate_only_all_other_structures_removed_bool

    @property
    def display_structure_surface_mesh_bool(self) -> bool:
        return self.debug.display_structure_surface_mesh_bool

    @property
    def show_equivalent_ellipsoid_from_pca_bool(self) -> bool:
        return self.debug.show_equivalent_ellipsoid_from_pca_bool

    def build_non_biopsy_structure_preprocessing_config(
        self,
        *,
        all_ref_key: str,
        oar_ref: str,
        dil_ref: str,
        mr_adc_ref: str,
    ) -> NonBiopsyStructurePreprocessingConfig:
        return NonBiopsyStructurePreprocessingConfig(
            all_ref_key=all_ref_key,
            oar_ref=oar_ref,
            dil_ref=dil_ref,
            mr_adc_ref=mr_adc_ref,
            interp_inter_slice_dist=self.interp_inter_slice_dist,
            interp_intra_slice_dist=self.interp_intra_slice_dist,
            interp_dist_caps=self.interp_dist_caps,
            radius_for_normals_estimation=self.radius_for_normals_estimation,
            max_nn_for_normals_estimation=self.max_nn_for_normals_estimation,
            voxel_size_for_structure_volume_calc_non_bx=self.voxel_size_for_structure_volume_calc_non_bx,
            voxel_size_for_structure_dimension_calc=self.voxel_size_for_structure_dimension_calc,
            factor_for_voxel_size=self.factor_for_voxel_size,
            cupy_array_upper_limit_NxN_size_input=self.cupy_array_upper_limit_nxn_size_input,
            nearest_zslice_vals_and_indices_cupy_generic_max_size=(
                self.nearest_zslice_vals_and_indices_cupy_generic_max_size
            ),
            generate_cuda_log_files_volume_calculation=self.generate_cuda_log_files_volume_calculation,
            constant_z_slice_polygons_handler_option=self.constant_z_slice_polygons_handler_option,
            remove_consecutive_duplicate_points_in_polygons=self.remove_consecutive_duplicate_points_in_polygons,
            include_edges_in_log_files=self.include_edges_in_log_files,
            custom_cuda_kernel_type=self.custom_cuda_kernel_type,
            demonstrate_volume_calculation_correctness_bool_1=(
                self.demonstrate_volume_calculation_correctness_bool_1
            ),
            plot_volume_calculation_containment_result_bool_1_old=(
                self.plot_volume_calculation_containment_result_bool_1_old
            ),
            plot_binary_mask_bool=self.plot_binary_mask_bool,
            generate_cuda_log_files_structure_dimension_calculation=(
                self.generate_cuda_log_files_structure_dimension_calculation
            ),
            demonstrate_structure_dimension_calculation_correctness_bool_1=(
                self.demonstrate_structure_dimension_calculation_correctness_bool_1
            ),
            demonstrate_structure_dimension_calculation_correctness_bool_1_old=(
                self.demonstrate_structure_dimension_calculation_correctness_bool_1_old
            ),
            demonstrate_mr_adc_pcd_containment_correctness_bool=(
                self.demonstrate_mr_adc_pcd_containment_correctness_bool
            ),
            display_structure_surface_mesh_bool=self.display_structure_surface_mesh_bool,
            show_equivalent_ellipsoid_from_pca_bool=self.show_equivalent_ellipsoid_from_pca_bool,
        )

    def build_frozen_preprocessed_bundle_config(self) -> FrozenPreprocessedBundleConfig:
        return FrozenPreprocessedBundleConfig(
            interp_inter_slice_dist=self.interp_inter_slice_dist,
            interp_intra_slice_dist=self.interp_intra_slice_dist,
            radius_for_normals_estimation=self.radius_for_normals_estimation,
            max_nn_for_normals_estimation=self.max_nn_for_normals_estimation,
        )


@dataclass(frozen=True)
class GuidanceMapConfig:
    planning_config: GuidanceMapPlanningConfig = field(default_factory=GuidanceMapPlanningConfig)
    render_config: GuidanceMapRenderConfig = field(default_factory=GuidanceMapRenderConfig)


@dataclass(frozen=True)
class OptimizerV2CapacityConfig:
    max_candidates_per_chunk: Optional[int] = None
    max_test_structures_per_call: Optional[int] = None
    fallback_max_test_structures_per_call: Optional[int] = 4_000_000
    auto_calibrate_max_test_structures_per_call: bool = True
    verify_calibrated_max_test_structures_per_call: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "max_candidates_per_chunk",
            "max_test_structures_per_call",
            "fallback_max_test_structures_per_call",
        ):
            field_value = getattr(self, field_name)
            if field_value is not None and int(field_value) < 1:
                raise ValueError(f"{field_name} must be positive when provided")
            if field_value is not None:
                object.__setattr__(self, field_name, int(field_value))
        object.__setattr__(
            self,
            "auto_calibrate_max_test_structures_per_call",
            bool(self.auto_calibrate_max_test_structures_per_call),
        )
        object.__setattr__(
            self,
            "verify_calibrated_max_test_structures_per_call",
            bool(self.verify_calibrated_max_test_structures_per_call),
        )


@dataclass(frozen=True)
class OptimizerV2DiagnosticsConfig:
    validate_nearest_z_helper_against_ver5: bool = True
    benchmark_isolated_winner_validation_bool: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "validate_nearest_z_helper_against_ver5",
            bool(self.validate_nearest_z_helper_against_ver5),
        )
        object.__setattr__(
            self,
            "benchmark_isolated_winner_validation_bool",
            bool(self.benchmark_isolated_winner_validation_bool),
        )


@dataclass(frozen=True)
class OptimizerV2PlotlyExportConfig:
    enabled: bool = False
    formats: Sequence[str] = ("svg", "pdf")
    width: int = 1920
    height: int = 1080
    scale: float = 1.0
    camera_eye: Sequence[float] = (1.45, -1.45, 2.25)
    camera_center: Sequence[float] = (0.0, 0.0, 0.0)
    camera_up: Sequence[float] = (0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "formats", tuple(str(value) for value in self.formats))
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "height", int(self.height))
        object.__setattr__(self, "scale", float(self.scale))
        object.__setattr__(self, "camera_eye", tuple(float(value) for value in self.camera_eye))
        object.__setattr__(self, "camera_center", tuple(float(value) for value in self.camera_center))
        object.__setattr__(self, "camera_up", tuple(float(value) for value in self.camera_up))
        if self.width < 1:
            raise ValueError("width must be positive")
        if self.height < 1:
            raise ValueError("height must be positive")
        if self.scale <= 0:
            raise ValueError("scale must be positive")


@dataclass(frozen=True)
class OptimizerV2RenderConfig:
    render_stage_boundary_candidate_clouds_bool: bool = False
    render_stage_names_to_render: Optional[Sequence[str]] = None
    render_backend: str = "both"
    render_layer_style_by_name: Optional[Mapping[str, Any]] = None
    plotly_export: OptimizerV2PlotlyExportConfig = field(default_factory=OptimizerV2PlotlyExportConfig)
    render_dialog_timeout_seconds: Optional[float] = None
    render_dialog_timeout_extend_seconds: float = 300.0
    render_winner_containment_debug_bool: bool = False
    render_winner_containment_backend: Optional[str] = "both"
    render_include_target_points_bool: bool = False
    render_include_target_surface_bool: bool = True
    render_include_planned_sampled_points_bool: bool = True
    render_include_planned_core_structure_bool: bool = True
    render_include_planned_centroid_line_bool: bool = True
    render_include_selected_anatomy_bool: bool = True
    render_patient_whitelist: Optional[Sequence[str]] = None
    render_roi_whitelist: Optional[Sequence[str]] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "render_stage_boundary_candidate_clouds_bool",
            bool(self.render_stage_boundary_candidate_clouds_bool),
        )
        if self.render_stage_names_to_render is not None:
            object.__setattr__(
                self,
                "render_stage_names_to_render",
                tuple(str(value) for value in self.render_stage_names_to_render),
            )
        object.__setattr__(self, "render_backend", str(self.render_backend))
        if self.render_layer_style_by_name is not None:
            object.__setattr__(self, "render_layer_style_by_name", dict(self.render_layer_style_by_name))
        object.__setattr__(self, "render_dialog_timeout_extend_seconds", float(self.render_dialog_timeout_extend_seconds))
        if self.render_dialog_timeout_seconds is not None:
            object.__setattr__(self, "render_dialog_timeout_seconds", float(self.render_dialog_timeout_seconds))
        object.__setattr__(self, "render_winner_containment_debug_bool", bool(self.render_winner_containment_debug_bool))
        if self.render_winner_containment_backend is not None:
            object.__setattr__(self, "render_winner_containment_backend", str(self.render_winner_containment_backend))
        for field_name in (
            "render_include_target_points_bool",
            "render_include_target_surface_bool",
            "render_include_planned_sampled_points_bool",
            "render_include_planned_core_structure_bool",
            "render_include_planned_centroid_line_bool",
            "render_include_selected_anatomy_bool",
        ):
            object.__setattr__(self, field_name, bool(getattr(self, field_name)))
        if self.render_patient_whitelist is not None:
            object.__setattr__(self, "render_patient_whitelist", tuple(str(value) for value in self.render_patient_whitelist))
        if self.render_roi_whitelist is not None:
            object.__setattr__(self, "render_roi_whitelist", tuple(str(value) for value in self.render_roi_whitelist))
        if self.render_dialog_timeout_extend_seconds <= 0:
            raise ValueError("render_dialog_timeout_extend_seconds must be positive")
        if self.render_dialog_timeout_seconds is not None and self.render_dialog_timeout_seconds <= 0:
            raise ValueError("render_dialog_timeout_seconds must be positive when provided")


@dataclass(frozen=True)
class OptimizerV2RuntimeConfig:
    search_config: OptimizerV2SearchConfig
    capacity: OptimizerV2CapacityConfig = field(default_factory=OptimizerV2CapacityConfig)
    diagnostics: OptimizerV2DiagnosticsConfig = field(default_factory=OptimizerV2DiagnosticsConfig)
    rendering: OptimizerV2RenderConfig = field(default_factory=OptimizerV2RenderConfig)
    num_stochastic_targeting_transform_samples_input: int = 0

    def __post_init__(self) -> None:
        if self.num_stochastic_targeting_transform_samples_input < 0:
            raise ValueError("num_stochastic_targeting_transform_samples_input cannot be negative")


@dataclass(frozen=True)
class OptimizerV1RuntimeConfig:
    voxel_size_for_dil_optimizer_grid: float = 1
    optimal_normal_dist_option: str = "dil dimension driven"
    bias_LR_multiplier: float = 1
    bias_AP_multiplier: float = 1
    bias_SI_multiplier: float = 1.5
    num_normal_dist_points_for_biopsy_optimizer: int = 10000
    normal_dist_sigma_factor_biopsy_optimizer: float = 0.25
    plot_each_normal_dist_containment_result_bool: bool = False
    plot_optimization_point_lattice_bool: bool = False
    show_optimization_point_bool: bool = False
    cupy_array_upper_limit_nxn_size_input: Any = 1e9
    numpy_array_upper_limit_nxn_size_input: Any = 1e9
    nearest_zslice_vals_and_indices_cupy_generic_max_size: Any = 5e7
    nearest_zslice_vals_and_indices_numpy_generic_max_size: Any = 1e9
    constant_z_slice_polygons_handler_option: Any = "auto-close-if-open"
    remove_consecutive_duplicate_points_in_polygons: bool = True
    include_edges_in_log_files: bool = False
    custom_cuda_kernel_type: Any = "one_to_one_pip_kernel_advanced_reparameterized_version_gpu_memory_performance_optimized"
    demonstrate_dil_optimization_points_inside_correctness_bool_1: bool = False
    demonstrate_dil_optimization_points_inside_correctness_bool_2: bool = False
    demonstrate_dil_optimization_points_inside_correctness_num_3: int = 0
    generate_cuda_log_files_biopsy_optimizer: bool = False
    display_optimization_contour_plots_bool: bool = False

    def __post_init__(self) -> None:
        if self.voxel_size_for_dil_optimizer_grid <= 0:
            raise ValueError("voxel_size_for_dil_optimizer_grid must be positive")
        object.__setattr__(
            self,
            "optimal_normal_dist_option",
            _non_empty_string(self.optimal_normal_dist_option, "optimal_normal_dist_option"),
        )
        if self.num_normal_dist_points_for_biopsy_optimizer < 1:
            raise ValueError("num_normal_dist_points_for_biopsy_optimizer must be at least 1")
        if self.normal_dist_sigma_factor_biopsy_optimizer <= 0:
            raise ValueError("normal_dist_sigma_factor_biopsy_optimizer must be positive")
        for field_name in (
            "plot_each_normal_dist_containment_result_bool",
            "plot_optimization_point_lattice_bool",
            "show_optimization_point_bool",
            "remove_consecutive_duplicate_points_in_polygons",
            "include_edges_in_log_files",
            "demonstrate_dil_optimization_points_inside_correctness_bool_1",
            "demonstrate_dil_optimization_points_inside_correctness_bool_2",
            "generate_cuda_log_files_biopsy_optimizer",
            "display_optimization_contour_plots_bool",
        ):
            object.__setattr__(self, field_name, bool(getattr(self, field_name)))
        demonstrate_count = int(self.demonstrate_dil_optimization_points_inside_correctness_num_3)
        if demonstrate_count < 0:
            raise ValueError("demonstrate_dil_optimization_points_inside_correctness_num_3 cannot be negative")
        object.__setattr__(self, "demonstrate_dil_optimization_points_inside_correctness_num_3", demonstrate_count)


@dataclass(frozen=True)
class OptimizerRuntimeConfig:
    optimizer_v2_search_config: OptimizerV2SearchConfig
    num_stochastic_targeting_transform_samples_input: int = 0
    optimizer_v2: OptimizerV2RuntimeConfig | None = None
    optimizer_v1: OptimizerV1RuntimeConfig = field(default_factory=OptimizerV1RuntimeConfig)

    def __post_init__(self) -> None:
        if self.num_stochastic_targeting_transform_samples_input < 0:
            raise ValueError("num_stochastic_targeting_transform_samples_input cannot be negative")
        if self.optimizer_v2 is None:
            object.__setattr__(
                self,
                "optimizer_v2",
                OptimizerV2RuntimeConfig(
                    search_config=self.optimizer_v2_search_config,
                    num_stochastic_targeting_transform_samples_input=(
                        self.num_stochastic_targeting_transform_samples_input
                    ),
                ),
            )
        elif self.optimizer_v2.search_config != self.optimizer_v2_search_config:
            raise ValueError("optimizer_v2.search_config must match optimizer_v2_search_config")


@dataclass(frozen=True)
class RandomSeedConfig:
    transform_generation_random_seed: Optional[int]
    optimizer_v1_random_seed: Optional[int]


@dataclass(frozen=True)
class MCCountsConfig:
    num_mc_containment_simulations_input: int = 10000
    num_mc_dose_simulations_input: int = 10000
    num_mc_mr_simulations_input: int = 10000

    def __post_init__(self) -> None:
        for field_name in (
            "num_mc_containment_simulations_input",
            "num_mc_dose_simulations_input",
            "num_mc_mr_simulations_input",
        ):
            field_value = int(getattr(self, field_name))
            if field_value < 0:
                raise ValueError(f"{field_name} cannot be negative")
            object.__setattr__(self, field_name, field_value)

    @property
    def perform_mc_containment_sim(self) -> bool:
        return bool(self.num_mc_containment_simulations_input)

    @property
    def perform_mc_dose_sim(self) -> bool:
        return bool(self.num_mc_dose_simulations_input)

    @property
    def perform_mc_mr_sim(self) -> bool:
        return bool(self.num_mc_mr_simulations_input)

    @property
    def perform_mc_sim(self) -> bool:
        return self.perform_mc_containment_sim or self.perform_mc_dose_sim or self.perform_mc_mr_sim

    @property
    def max_num_mc_simulations(self) -> int:
        return max(
            self.num_mc_containment_simulations_input,
            self.num_mc_dose_simulations_input,
            self.num_mc_mr_simulations_input,
        )


@dataclass(frozen=True)
class MCPrepConfig:
    simulate_uniform_bx_shifts_due_to_bx_needle_compartment: bool = True
    biopsy_needle_compartment_length: float = 19.0
    bx_sample_pts_lattice_spacing: float = 1.0
    run_transform_generation: bool = True
    run_biopsy_self_transforms: bool = True
    run_relative_structure_transforms: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "simulate_uniform_bx_shifts_due_to_bx_needle_compartment",
            bool(self.simulate_uniform_bx_shifts_due_to_bx_needle_compartment),
        )
        for field_name in (
            "run_transform_generation",
            "run_biopsy_self_transforms",
            "run_relative_structure_transforms",
        ):
            object.__setattr__(self, field_name, bool(getattr(self, field_name)))
        if self.biopsy_needle_compartment_length <= 0:
            raise ValueError("biopsy_needle_compartment_length must be positive")
        if self.bx_sample_pts_lattice_spacing <= 0:
            raise ValueError("bx_sample_pts_lattice_spacing must be positive")


@dataclass(frozen=True)
class MCSimulationCoreConfig:
    biopsy_z_voxel_length: float = 1.0
    num_dose_calc_nn: int = 4
    num_mr_calc_nn: int = 4
    idw_power: float = 1.0
    tissue_length_above_probability_threshold_list: Sequence[float] = (0.95, 0.75, 0.5, 0.25)
    n_bootstraps_for_tissue_length_above_threshold: int = 1000
    differential_dvh_resolution: int = 100
    cumulative_dvh_resolution: int = 100
    v_percent_dvh_to_calc_list: Sequence[float] = (100, 125, 150, 200, 300)
    volume_dvh_quantiles_to_calculate: Sequence[float] = (5, 25, 50, 75, 95)
    cuml_nn_algo: str = "brute"
    nn_search_end_cap_grid_factor: float = 0.1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tissue_length_above_probability_threshold_list",
            tuple(self.tissue_length_above_probability_threshold_list),
        )
        object.__setattr__(self, "v_percent_dvh_to_calc_list", tuple(self.v_percent_dvh_to_calc_list))
        object.__setattr__(
            self,
            "volume_dvh_quantiles_to_calculate",
            tuple(self.volume_dvh_quantiles_to_calculate),
        )


@dataclass(frozen=True)
class MCDebugConfig:
    inspect_self_biopsy_dilate_bool: bool = False
    inspect_self_biopsy_dilate_and_rotate_bool: bool = False
    inspect_self_biopsy_dilate_and_rotate_and_translate_bool: bool = False
    inspect_relative_structure_rotate_and_shift_number: int = 0
    plot_uniform_shifts_to_check_plotly: bool = False
    plot_translation_vectors_pointclouds: bool = False
    plot_shifted_biopsies: bool = False
    show_nn_dose_demonstration_plots: bool = False
    show_nn_dose_demonstration_plots_all_trials_at_once: bool = False
    show_num_containment_demonstration_plots: int = 0
    plot_cupy_containment_distribution_results: bool = False
    show_num_nearest_neighbour_surface_boundary_demonstration: int = 0
    show_num_relative_structure_centroid_demonstration: int = 0
    show_nn_mr_adc_demonstration_plots: bool = False
    show_nn_mr_adc_demonstration_plots_all_trials_at_once: bool = False
    generate_cuda_log_files_mc_containment_sim: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "inspect_self_biopsy_dilate_bool",
            "inspect_self_biopsy_dilate_and_rotate_bool",
            "inspect_self_biopsy_dilate_and_rotate_and_translate_bool",
            "plot_uniform_shifts_to_check_plotly",
            "plot_translation_vectors_pointclouds",
            "plot_shifted_biopsies",
            "show_nn_dose_demonstration_plots",
            "show_nn_dose_demonstration_plots_all_trials_at_once",
            "plot_cupy_containment_distribution_results",
            "show_nn_mr_adc_demonstration_plots",
            "show_nn_mr_adc_demonstration_plots_all_trials_at_once",
            "generate_cuda_log_files_mc_containment_sim",
        ):
            object.__setattr__(self, field_name, bool(getattr(self, field_name)))
        for field_name in (
            "inspect_relative_structure_rotate_and_shift_number",
            "show_num_containment_demonstration_plots",
            "show_num_nearest_neighbour_surface_boundary_demonstration",
            "show_num_relative_structure_centroid_demonstration",
        ):
            field_value = int(getattr(self, field_name))
            if field_value < 0:
                raise ValueError(f"{field_name} cannot be negative")
            object.__setattr__(self, field_name, field_value)


@dataclass(frozen=True)
class MCOutputDumpConfig:
    raw_data_mc_dosimetry_dump_bool: bool = False
    raw_data_mc_containment_dump_bool: bool = False
    raw_data_mc_mr_dump_bool: bool = False
    keep_light_containment_and_distances_to_relative_structures_dataframe_bool: bool = True

    def __post_init__(self) -> None:
        for field_name in (
            "raw_data_mc_dosimetry_dump_bool",
            "raw_data_mc_containment_dump_bool",
            "raw_data_mc_mr_dump_bool",
            "keep_light_containment_and_distances_to_relative_structures_dataframe_bool",
        ):
            object.__setattr__(self, field_name, bool(getattr(self, field_name)))


@dataclass(frozen=True)
class MCTissueClassificationConfig:
    structure_miss_probability_roi: str = "Prostate"
    cancer_tissue_label: str = "DIL"
    default_exterior_tissue: str = "Periprostatic"
    miss_structure_complement_label: str = "Prostate complement"
    tissue_volume_operator_dictionary: Mapping[str, Any] = field(
        default_factory=lambda: {
            "DIL": "greater",
            "Prostatic": "greater",
            "Rectal": "less",
            "Urethral": "less",
            "Periprostatic": "less",
        }
    )

    def __post_init__(self) -> None:
        for field_name in (
            "structure_miss_probability_roi",
            "cancer_tissue_label",
            "default_exterior_tissue",
            "miss_structure_complement_label",
        ):
            object.__setattr__(self, field_name, _non_empty_string(getattr(self, field_name), field_name))
        object.__setattr__(self, "tissue_volume_operator_dictionary", dict(self.tissue_volume_operator_dictionary))


@dataclass(frozen=True)
class MCVisualizationConfig:
    num_dose_nn_to_show_for_animation_plotting: int = 100
    containment_results_structure_types_to_show_per_trial: Sequence[str] = ("OAR ref", "DIL ref")
    show_non_bx_relative_structure_z_dilation_bool: bool = False
    show_non_bx_relative_structure_xy_dilation_bool: bool = False
    check_if_end_caps_filled_proper_nn_num: int = 0

    def __post_init__(self) -> None:
        num_dose_nn = int(self.num_dose_nn_to_show_for_animation_plotting)
        if num_dose_nn < 0:
            raise ValueError("num_dose_nn_to_show_for_animation_plotting cannot be negative")
        object.__setattr__(self, "num_dose_nn_to_show_for_animation_plotting", num_dose_nn)
        object.__setattr__(
            self,
            "containment_results_structure_types_to_show_per_trial",
            tuple(str(value) for value in self.containment_results_structure_types_to_show_per_trial),
        )
        for field_name in (
            "show_non_bx_relative_structure_z_dilation_bool",
            "show_non_bx_relative_structure_xy_dilation_bool",
        ):
            object.__setattr__(self, field_name, bool(getattr(self, field_name)))
        end_cap_count = int(self.check_if_end_caps_filled_proper_nn_num)
        if end_cap_count < 0:
            raise ValueError("check_if_end_caps_filled_proper_nn_num cannot be negative")
        object.__setattr__(self, "check_if_end_caps_filled_proper_nn_num", end_cap_count)


@dataclass(frozen=True)
class MonteCarloConfig:
    counts: MCCountsConfig = field(default_factory=MCCountsConfig)
    prep: MCPrepConfig = field(default_factory=MCPrepConfig)
    simulation: MCSimulationCoreConfig = field(default_factory=MCSimulationCoreConfig)
    debug: MCDebugConfig = field(default_factory=MCDebugConfig)
    output_dumps: MCOutputDumpConfig = field(default_factory=MCOutputDumpConfig)
    tissue: MCTissueClassificationConfig = field(default_factory=MCTissueClassificationConfig)
    visualization: MCVisualizationConfig = field(default_factory=MCVisualizationConfig)


@dataclass(frozen=True)
class ValidationSidecarConfig:
    selected_structures_against_legacy: bool = False
    non_biopsy_structures_against_legacy: bool = False
    prostate_only_mr_adc_against_legacy: bool = False

    def active_validation_labels(self) -> tuple[str, ...]:
        labels = []
        if self.selected_structures_against_legacy:
            labels.append("selected-structures-legacy")
        if self.non_biopsy_structures_against_legacy:
            labels.append("non-biopsy-structures-legacy")
        if self.prostate_only_mr_adc_against_legacy:
            labels.append("prostate-only-mr-adc-legacy")
        return tuple(labels)

    def validation_run_type_string(self) -> str:
        active_labels = self.active_validation_labels()
        if len(active_labels) == 0:
            return "standard-run"
        return "validation-" + "_".join(active_labels)

    def build_run_folder_suffix(self, input_data_counts: Optional[dict[str, int]] = None) -> str:
        suffix_parts = [self.validation_run_type_string()]
        if input_data_counts:
            count_parts = [
                f"{count_name}-{input_data_counts[count_name]}"
                for count_name in sorted(input_data_counts)
            ]
            suffix_parts.append("inputs-" + "_".join(count_parts))
        return " - ".join(suffix_parts)


@dataclass(frozen=True)
class PipelineConfig:
    ui: RuntimeUIConfig
    artifacts: ArtifactConfig
    preprocessing: PreprocessingConfig
    replay: RuntimeReplayConfig
    guidance_maps: GuidanceMapConfig
    optimizer: OptimizerRuntimeConfig
    random_seeds: RandomSeedConfig
    validation_sidecars: ValidationSidecarConfig = field(default_factory=ValidationSidecarConfig)
    mc: MonteCarloConfig = field(default_factory=MonteCarloConfig)
    legacy_refs: LegacyReferenceConfig = field(default_factory=LegacyReferenceConfig)
    structure_registry: StructureRegistryConfig = field(default_factory=StructureRegistryConfig)
    grid_preprocessing: GridPreprocessingConfig = field(default_factory=GridPreprocessingConfig)
    biopsy: BiopsyRuntimeConfig = field(default_factory=BiopsyRuntimeConfig)