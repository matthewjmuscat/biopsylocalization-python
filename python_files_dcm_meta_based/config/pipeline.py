from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from biopsy_optimizer.v2.config import OptimizerV2SearchConfig
from preprocessing.structure_processing.non_biopsy_structure_processing import (
    NonBiopsyStructurePreprocessingConfig,
)
from startup.guidance_map_workflow import GuidanceMapRenderConfig


@dataclass(frozen=True)
class RuntimeUIConfig:
    spinner_type: str = "moon"
    rich_live_display_bool: bool = True


@dataclass(frozen=True)
class ArtifactConfig:
    output_folder_name: str = "Output data"
    preprocessed_data_folder_name: str = "Preprocessed data"
    preprocessed_reference_dict_filename: str = "master_structure_reference_dict"
    preprocessed_info_dict_filename: str = "master_structure_info_dict"
    export_pickled_preprocessed_data: bool = False
    skip_preprocessing: bool = False


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
class PreprocessingConfig:
    interp_inter_slice_dist: float
    interp_intra_slice_dist: float
    interp_dist_caps: float
    radius_for_normals_estimation: float
    max_nn_for_normals_estimation: int
    voxel_size_for_structure_volume_calc_non_bx: float
    voxel_size_for_structure_dimension_calc: float
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
    generate_cuda_log_files_structure_dimension_calculation: bool
    demonstrate_structure_dimension_calculation_correctness_bool_1: bool
    demonstrate_structure_dimension_calculation_correctness_bool_1_old: bool
    demonstrate_mr_adc_pcd_containment_correctness_bool: bool
    display_structure_surface_mesh_bool: bool
    show_equivalent_ellipsoid_from_pca_bool: bool

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
    render_config: GuidanceMapRenderConfig = field(default_factory=GuidanceMapRenderConfig)


@dataclass(frozen=True)
class OptimizerRuntimeConfig:
    optimizer_v2_search_config: OptimizerV2SearchConfig
    num_stochastic_targeting_transform_samples_input: int = 0

    def __post_init__(self) -> None:
        if self.num_stochastic_targeting_transform_samples_input < 0:
            raise ValueError("num_stochastic_targeting_transform_samples_input cannot be negative")


@dataclass(frozen=True)
class RandomSeedConfig:
    transform_generation_random_seed: Optional[int]
    optimizer_v1_random_seed: Optional[int]


@dataclass(frozen=True)
class PipelineConfig:
    ui: RuntimeUIConfig
    artifacts: ArtifactConfig
    preprocessing: PreprocessingConfig
    replay: RuntimeReplayConfig
    guidance_maps: GuidanceMapConfig
    optimizer: OptimizerRuntimeConfig
    random_seeds: RandomSeedConfig