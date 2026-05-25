"""Patient-level MC containment output contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

from legacy_data_keys import legacy_data_keys

from .contracts import MCContainmentSimulationConfig, MCSimulationRuntimeConfig
from .legacy_keys import legacy_mc_keys
from .relative_structure_inventory import (
    PatientRelativeStructureInventory,
    RelativeStructureInfo,
    build_patient_relative_structure_inventory,
)

MC_CONTAINMENT_BIOPSY_OUTPUT_KEYS = legacy_mc_keys.biopsy_outputs.containment_output_keys
MC_STRUCTURE_SPECIFIC_RESULT_KEYS = (
    "Total successes (containment) list",
    "Binomial estimator list",
    "Confidence interval 95 (containment) list",
    "Standard error (containment) list",
    "Nominal containment list",
)
MC_CONTAINMENT_DISTANCE_MERGE_COLUMNS = (
    "Trial num",
    "Original pt index",
    "Relative struct input index",
)
MC_CONTAINMENT_POINT_CONTAINED_COLUMN = "Pt contained bool"
MC_CONTAINMENT_NN_DISTANCE_COLUMN = "Struct. boundary NN dist."
MC_CONTAINMENT_TRIAL_COLUMN = "Trial num"
MC_CONTAINMENT_ORIGINAL_POINT_INDEX_COLUMN = "Original pt index"
MC_CONTAINMENT_TOTAL_SUCCESSES_COLUMN = "Total successes"
MC_CONTAINMENT_BINOMIAL_ESTIMATOR_COLUMN = "Binomial estimator"
MC_CONTAINMENT_NOMINAL_COLUMN = "Nominal"
MC_CONTAINMENT_BINOMIAL_STD_ERR_COLUMN = "Binom est STD err"
MC_CONTAINMENT_CI_LOWER_COLUMN = "CI lower vals"
MC_CONTAINMENT_CI_UPPER_COLUMN = "CI upper vals"
MC_CONTAINMENT_TISSUE_CLASS_COLUMN = "Tissue class"
MC_CONTAINMENT_PATIENT_ID_COLUMN = "Patient ID"
MC_CONTAINMENT_BX_ID_COLUMN = "Bx ID"
MC_CONTAINMENT_BX_REFNUM_COLUMN = "Bx refnum"
MC_CONTAINMENT_BX_INDEX_COLUMN = "Bx index"
MC_CONTAINMENT_SIMULATED_BOOL_COLUMN = legacy_data_keys.structure_record.simulated_bool_key
MC_CONTAINMENT_SIMULATED_TYPE_COLUMN = legacy_data_keys.structure_record.simulated_type_key
MC_CONTAINMENT_RELATIVE_ROI_COLUMN = "Relative structure ROI"
MC_CONTAINMENT_RELATIVE_TYPE_COLUMN = "Relative structure type"
MC_CONTAINMENT_RELATIVE_INDEX_COLUMN = "Relative structure index"
MC_CONTAINMENT_BX_FRAME_X_COLUMN = "X (Bx frame)"
MC_CONTAINMENT_BX_FRAME_Y_COLUMN = "Y (Bx frame)"
MC_CONTAINMENT_BX_FRAME_Z_COLUMN = "Z (Bx frame)"
MC_CONTAINMENT_VOXEL_REFERENCE_COLUMN = MC_CONTAINMENT_BX_FRAME_Z_COLUMN
MC_CONTAINMENT_DISTANCE_VALUE_COLUMNS = (
    MC_CONTAINMENT_NN_DISTANCE_COLUMN,
    "Dist. from struct. centroid",
    "Dist. from struct. centroid X",
    "Dist. from struct. centroid Y",
    "Dist. from struct. centroid Z",
)
MC_CONTAINMENT_LIGHT_DROP_COLUMNS = (
    "Nearest zslice zval",
    "Nearest zslice index",
    "Pt clr R",
    "Pt clr G",
    "Pt clr B",
    "Test pt X",
    "Test pt Y",
    "Test pt Z",
    "Struct. boundary NN relative index (all pts stacked)",
)
PatientContainmentStructureInventory = PatientRelativeStructureInventory


@dataclass(slots=True)
class PatientContainmentDilatedStructureBank:
    """Patient-local relative-structure dilation state reused for each biopsy."""

    patient_uid: str
    dilated_structures_by_structure: dict[RelativeStructureInfo, list[Any]] = field(default_factory=dict)
    centroids_by_structure: dict[RelativeStructureInfo, Any] = field(default_factory=dict)
    relative_structure_mapping_by_structure: dict[RelativeStructureInfo, Any] = field(default_factory=dict)

    @property
    def relative_structure_infos(self) -> tuple[RelativeStructureInfo, ...]:
        return tuple(self.dilated_structures_by_structure.keys())


@dataclass(slots=True)
class PatientContainmentBiopsyContext:
    """Patient-local biopsy inputs for the containment loop."""

    patient_uid: str
    biopsy_index: int
    num_sample_points: int
    roi: Any
    ref_number: Any
    simulated_bool: Any
    simulated_type: Any
    unshifted_sampled_points: Any
    sampled_points_bx_coord_sys: Any
    shifted_points_by_relative_structure: Mapping[RelativeStructureInfo, Any]
    biopsy_structure_info: dict[str, Any]


@dataclass(slots=True)
class PatientContainmentRelativeStructureInput:
    """Prepared one-biopsy/one-relative-structure containment inputs."""

    structure_info: RelativeStructureInfo
    shifted_biopsy_points: Any
    shifted_biopsy_points_cutoff: Any
    combined_nominal_and_shifted_biopsy_points: Any
    combined_nominal_and_shifted_biopsy_points_reshaped: Any
    nominal_and_dilated_structures: list[Any]
    centroids_of_nominal_and_dilated_structures: Any
    trial_to_relative_structure_mapping: Any

    @property
    def roi(self) -> Any:
        return self.structure_info[0]

    @property
    def structure_type(self) -> str:
        return self.structure_info[1]

    @property
    def ref_number(self) -> Any:
        return self.structure_info[2]

    @property
    def structure_index(self) -> int:
        return self.structure_info[3]


@dataclass(slots=True)
class PatientContainmentCentroidDistanceResult:
    """Relative-structure centroid distance outputs for one containment comparison."""

    dataframe: Any
    centroid_distances: Any
    centroid_distances_x: Any
    centroid_distances_y: Any
    centroid_distances_z: Any
    sorted_centroids: Any


@dataclass(slots=True)
class PatientContainmentNearestNeighbourResult:
    """Nearest-neighbour boundary distance outputs for one containment comparison."""

    dataframe: Any
    distances: Any
    relative_indices: Any
    candidates_stacked: Any
    candidates_indices: Any
    absolute_indices: Any


@dataclass(slots=True)
class PatientContainmentKernelResult:
    """Point-containment kernel outputs for one containment comparison."""

    dataframe: Any
    containment_result_array: Any
    prepper_output_tuple: tuple[Any, ...]


@dataclass(slots=True)
class PatientContainmentDistanceSummaryOutputs:
    """Distance summary dataframes compiled for one biopsy."""

    global_dataframe: Any
    point_wise_dataframe: Any
    voxel_wise_dataframe: Any


@dataclass(slots=True)
class PatientContainmentBiopsyStatisticsOutputs:
    """Per-biopsy containment statistics and legacy output payloads."""

    compiled_results_dataframe: Any
    sum_to_one_results_dataframe: Any
    compiled_results_dict: dict[RelativeStructureInfo, dict[str, Any]]
    distance_outputs: PatientContainmentDistanceSummaryOutputs
    light_trials_dataframe: Any | None = None

    def legacy_biopsy_updates(self) -> dict[str, Any]:
        output_keys = legacy_mc_keys.biopsy_outputs
        updates = {
            output_keys.containment_compiled_results_dataframe_key: self.compiled_results_dataframe,
            output_keys.containment_sum_to_one_results_dataframe_key: self.sum_to_one_results_dataframe,
            output_keys.containment_compiled_results_key: self.compiled_results_dict,
            output_keys.containment_distances_global_dataframe_key: self.distance_outputs.global_dataframe,
            output_keys.containment_distances_point_wise_dataframe_key: self.distance_outputs.point_wise_dataframe,
            output_keys.containment_distances_voxel_wise_dataframe_key: self.distance_outputs.voxel_wise_dataframe,
        }
        if self.light_trials_dataframe is not None:
            updates[output_keys.containment_light_trials_dataframe_key] = self.light_trials_dataframe
        return updates


@dataclass(slots=True)
class PatientContainmentOutputs:
    """Containment outputs collected from one patient's biopsy records."""

    patient_uid: str
    biopsy_outputs: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "patient_uid": self.patient_uid,
            "biopsy_outputs": self.biopsy_outputs,
        }


def build_structure_specific_results_template() -> dict[str, None]:
    """Return the legacy per-structure containment result template."""
    return {key: None for key in MC_STRUCTURE_SPECIFIC_RESULT_KEYS}


def build_mutual_structure_specific_results_template() -> dict[str, None]:
    """Return the legacy mutual-containment result template."""
    return {key: None for key in MC_STRUCTURE_SPECIFIC_RESULT_KEYS}


def _structure_zslices_for_containment(patient_reference_dict: Mapping[str, Any],
                                       *,
                                       non_bx_structure_type: str,
                                       structure_index: int,
                                       oar_ref: str,
                                       rectum_ref: str,
                                       urethra_ref: str) -> Any:
    geometry_keys = legacy_data_keys.structure_geometry
    if non_bx_structure_type in {oar_ref, rectum_ref, urethra_ref}:
        return patient_reference_dict[non_bx_structure_type][structure_index][
            geometry_keys.equal_num_zslice_contour_points_key
        ]
    interslice_information = patient_reference_dict[non_bx_structure_type][structure_index][
        geometry_keys.interslice_interpolation_information_key
    ]
    return interslice_information.interpolated_pts_list


def _build_legacy_biopsy_structure_info(biopsy_structure: Mapping[str, Any]) -> dict[str, Any]:
    structure_record_keys = legacy_data_keys.structure_record
    structure_metadata_keys = legacy_data_keys.structure_metadata
    return {
        structure_metadata_keys.structure_id_key: biopsy_structure[structure_record_keys.roi_key],
        structure_metadata_keys.struct_ref_type_key: biopsy_structure[structure_record_keys.struct_type_key],
        structure_metadata_keys.dicom_ref_number_key: biopsy_structure[structure_record_keys.ref_number_key],
        structure_record_keys.index_number_key: biopsy_structure[structure_record_keys.index_number_key],
        structure_record_keys.simulated_bool_key: biopsy_structure[structure_record_keys.simulated_bool_key],
        structure_record_keys.simulated_type_key: biopsy_structure[structure_record_keys.simulated_type_key],
    }


def build_patient_containment_biopsy_context(patient_uid: str,
                                             biopsy_index: int,
                                             biopsy_structure: Mapping[str, Any]) -> PatientContainmentBiopsyContext:
    """Build the per-biopsy context used by the containment loop."""
    identity_keys = legacy_mc_keys.biopsy_identity
    biopsy_runtime_keys = legacy_data_keys.biopsy_runtime
    intermediate_keys = legacy_mc_keys.containment_intermediates
    return PatientContainmentBiopsyContext(
        patient_uid=str(patient_uid),
        biopsy_index=int(biopsy_index),
        num_sample_points=int(biopsy_structure[biopsy_runtime_keys.num_sampled_bx_points_key]),
        roi=biopsy_structure[identity_keys.roi_key],
        ref_number=biopsy_structure[identity_keys.ref_number_key],
        simulated_bool=biopsy_structure[identity_keys.simulated_bool_key],
        simulated_type=biopsy_structure[identity_keys.simulated_type_key],
        unshifted_sampled_points=biopsy_structure[
            biopsy_runtime_keys.random_uniformly_sampled_volume_points_array_key
        ],
        sampled_points_bx_coord_sys=biopsy_structure[
            biopsy_runtime_keys.random_uniformly_sampled_volume_points_bx_coord_sys_array_key
        ],
        shifted_points_by_relative_structure=biopsy_structure[
            intermediate_keys.bx_and_structure_shifted_dict_key
        ],
        biopsy_structure_info=_build_legacy_biopsy_structure_info(biopsy_structure),
    )


def build_patient_containment_relative_structure_input(
    biopsy_context: PatientContainmentBiopsyContext,
    dilated_structure_bank: PatientContainmentDilatedStructureBank,
    structure_info: RelativeStructureInfo,
    shifted_biopsy_points_source: Any,
    *,
    num_mc_containment_simulations: int,
) -> PatientContainmentRelativeStructureInput:
    """Build the one-biopsy/one-relative-structure arrays consumed by containment kernels."""
    import cupy as cp
    import numpy as np

    shifted_biopsy_points = cp.asnumpy(shifted_biopsy_points_source)
    shifted_biopsy_points_cutoff = shifted_biopsy_points[0:int(num_mc_containment_simulations)]
    combined_biopsy_points = np.concatenate(
        [
            biopsy_context.unshifted_sampled_points[np.newaxis, :, :],
            shifted_biopsy_points_cutoff,
        ],
        axis=0,
    )
    return PatientContainmentRelativeStructureInput(
        structure_info=structure_info,
        shifted_biopsy_points=shifted_biopsy_points,
        shifted_biopsy_points_cutoff=shifted_biopsy_points_cutoff,
        combined_nominal_and_shifted_biopsy_points=combined_biopsy_points,
        combined_nominal_and_shifted_biopsy_points_reshaped=np.reshape(combined_biopsy_points, (-1, 3)),
        nominal_and_dilated_structures=dilated_structure_bank.dilated_structures_by_structure[structure_info],
        centroids_of_nominal_and_dilated_structures=dilated_structure_bank.centroids_by_structure[structure_info],
        trial_to_relative_structure_mapping=dilated_structure_bank.relative_structure_mapping_by_structure[structure_info],
    )


def iter_patient_containment_relative_structure_inputs(
    biopsy_context: PatientContainmentBiopsyContext,
    dilated_structure_bank: PatientContainmentDilatedStructureBank,
    *,
    num_mc_containment_simulations: int,
) -> Iterator[PatientContainmentRelativeStructureInput]:
    """Yield prepared containment inputs for each relative structure for one biopsy."""
    for structure_info, shifted_biopsy_points_source in biopsy_context.shifted_points_by_relative_structure.items():
        yield build_patient_containment_relative_structure_input(
            biopsy_context,
            dilated_structure_bank,
            structure_info,
            shifted_biopsy_points_source,
            num_mc_containment_simulations=num_mc_containment_simulations,
        )


def calculate_patient_relative_structure_centroid_distances(
    relative_structure_input: PatientContainmentRelativeStructureInput,
    biopsy_context: PatientContainmentBiopsyContext,
    *,
    num_mc_containment_simulations: int,
) -> PatientContainmentCentroidDistanceResult:
    """Calculate centroid distances for one biopsy/relative-structure containment comparison."""
    import numpy as np
    import relative_structure_centroid_calc

    (
        centroid_distances,
        centroid_distances_x,
        centroid_distances_y,
        centroid_distances_z,
        sorted_centroids,
    ) = relative_structure_centroid_calc.relative_structure_centroid_calculation_function(
        relative_structure_input.centroids_of_nominal_and_dilated_structures,
        relative_structure_input.trial_to_relative_structure_mapping,
        relative_structure_input.combined_nominal_and_shifted_biopsy_points,
    )
    dataframe = relative_structure_centroid_calc.relative_structure_centroid_df(
        centroid_distances,
        centroid_distances_x,
        centroid_distances_y,
        centroid_distances_z,
        int(num_mc_containment_simulations),
        biopsy_context.num_sample_points,
        relative_structure_input.trial_to_relative_structure_mapping,
        convert_to_categorical_and_downcast=False,
        do_not_convert_column_names_to_categorical=[],
        float_dtype=np.float32,
        int_dtype=np.int32,
    )
    return PatientContainmentCentroidDistanceResult(
        dataframe=dataframe,
        centroid_distances=centroid_distances,
        centroid_distances_x=centroid_distances_x,
        centroid_distances_y=centroid_distances_y,
        centroid_distances_z=centroid_distances_z,
        sorted_centroids=sorted_centroids,
    )


def calculate_patient_relative_structure_nearest_neighbour_distances(
    relative_structure_input: PatientContainmentRelativeStructureInput,
    biopsy_context: PatientContainmentBiopsyContext,
    *,
    num_mc_containment_simulations: int,
    containment_config: MCContainmentSimulationConfig,
    runtime_config: MCSimulationRuntimeConfig,
) -> PatientContainmentNearestNeighbourResult:
    """Calculate nearest-neighbour boundary distances using the existing kernel helper API."""
    import numpy as np
    import custom_raw_kernel_cuda_nearest_neighbour

    (
        distances,
        relative_indices,
        candidates_stacked,
        candidates_indices,
        absolute_indices,
    ) = custom_raw_kernel_cuda_nearest_neighbour.custom_gpu_kernel_NN_search_mother_function(
        relative_structure_input.nominal_and_dilated_structures,
        relative_structure_input.trial_to_relative_structure_mapping,
        relative_structure_input.combined_nominal_and_shifted_biopsy_points,
        biopsy_context.num_sample_points,
        int(num_mc_containment_simulations),
        grid_factor=containment_config.nn_search_end_cap_grid_factor,
        kernel_type=runtime_config.custom_cuda_kernel_type,
        check_if_end_caps_filled_proper_NN_num=containment_config.check_if_end_caps_filled_proper_NN_num,
        block_size=256,
    )
    dataframe = custom_raw_kernel_cuda_nearest_neighbour.build_results_df(
        distances,
        relative_indices,
        int(num_mc_containment_simulations),
        biopsy_context.num_sample_points,
        relative_structure_input.trial_to_relative_structure_mapping,
        convert_to_categorical_and_downcast=False,
        do_not_convert_column_names_to_categorical=[],
        float_dtype=np.float32,
        int_dtype=np.int32,
    )
    return PatientContainmentNearestNeighbourResult(
        dataframe=dataframe,
        distances=distances,
        relative_indices=relative_indices,
        candidates_stacked=candidates_stacked,
        candidates_indices=candidates_indices,
        absolute_indices=absolute_indices,
    )


def run_patient_relative_structure_containment_kernel(
    relative_structure_input: PatientContainmentRelativeStructureInput,
    biopsy_context: PatientContainmentBiopsyContext,
    *,
    num_mc_containment_simulations: int,
    containment_config: MCContainmentSimulationConfig,
    runtime_config: MCSimulationRuntimeConfig,
    include_edges_in_log: bool = False,
) -> PatientContainmentKernelResult:
    """Run the existing point-containment kernel helper API for one comparison."""
    import numpy as np
    import custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p

    structure_metadata_keys = legacy_data_keys.structure_metadata
    structure_record_keys = legacy_data_keys.structure_record
    log_sub_dirs_list = [
        biopsy_context.patient_uid,
        biopsy_context.roi,
        relative_structure_input.structure_type,
    ]
    if containment_config.generate_cuda_log_files_MC_containment_sim:
        custom_cuda_log_file_name = (
            f"{biopsy_context.patient_uid}_{biopsy_context.roi}_"
            f"{relative_structure_input.structure_type}_N-{int(num_mc_containment_simulations)}_containment_log.txt"
        )
    else:
        custom_cuda_log_file_name = None

    containment_result_array, prepper_output_tuple = (
        custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p.custom_point_containment_mother_function(
            relative_structure_input.nominal_and_dilated_structures,
            relative_structure_input.combined_nominal_and_shifted_biopsy_points,
            relative_structure_input.trial_to_relative_structure_mapping,
            constant_z_slice_polygons_handler_option=containment_config.constant_z_slice_polygons_handler_option,
            remove_consecutive_duplicate_points_in_polygons=(
                containment_config.remove_consecutive_duplicate_points_in_polygons
            ),
            log_sub_dirs_list=log_sub_dirs_list,
            log_file_name=custom_cuda_log_file_name,
            include_edges_in_log=include_edges_in_log,
            kernel_type=runtime_config.custom_cuda_kernel_type,
        )
    )
    structure_info_dict = {
        structure_metadata_keys.structure_id_key: relative_structure_input.roi,
        structure_metadata_keys.struct_ref_type_key: relative_structure_input.structure_type,
        structure_record_keys.index_number_key: relative_structure_input.structure_index,
    }
    dataframe = custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p.create_containment_results_dataframe_type_II(
        biopsy_context.patient_uid,
        biopsy_context.biopsy_structure_info,
        structure_info_dict,
        prepper_output_tuple[0],
        relative_structure_input.combined_nominal_and_shifted_biopsy_points,
        containment_result_array,
        relative_structure_input.trial_to_relative_structure_mapping,
        convert_to_categorical_and_downcast=False,
        do_not_convert_column_names_to_categorical=[MC_CONTAINMENT_POINT_CONTAINED_COLUMN],
        float_dtype=np.float64,
        int_dtype=np.int64,
        df_type="mc containment simulator",
    )
    return PatientContainmentKernelResult(
        dataframe=dataframe,
        containment_result_array=containment_result_array,
        prepper_output_tuple=prepper_output_tuple,
    )


def merge_patient_relative_structure_containment_dataframes(
    containment_dataframe: Any,
    nearest_neighbour_dataframe: Any,
    centroid_dataframe: Any,
) -> Any:
    """Merge containment, NN-boundary, and centroid-distance dataframes like the oracle."""
    import numpy as np
    import pandas
    import dataframe_builders

    merged_dataframe = pandas.merge(
        containment_dataframe,
        nearest_neighbour_dataframe,
        on=list(MC_CONTAINMENT_DISTANCE_MERGE_COLUMNS),
        how="left",
    )
    merged_dataframe[MC_CONTAINMENT_NN_DISTANCE_COLUMN] = np.where(
        merged_dataframe[MC_CONTAINMENT_POINT_CONTAINED_COLUMN],
        -abs(merged_dataframe[MC_CONTAINMENT_NN_DISTANCE_COLUMN]),
        abs(merged_dataframe[MC_CONTAINMENT_NN_DISTANCE_COLUMN]),
    )
    merged_dataframe = pandas.merge(
        merged_dataframe,
        centroid_dataframe,
        on=list(MC_CONTAINMENT_DISTANCE_MERGE_COLUMNS),
        how="left",
    )
    return dataframe_builders.convert_columns_to_categorical_and_downcast(
        merged_dataframe,
        threshold=0.25,
        do_not_convert_column_names_to_categorical=[
            MC_CONTAINMENT_POINT_CONTAINED_COLUMN,
            "Original pt index",
        ],
    )


def run_patient_relative_structure_containment_core(
    relative_structure_input: PatientContainmentRelativeStructureInput,
    biopsy_context: PatientContainmentBiopsyContext,
    *,
    num_mc_containment_simulations: int,
    containment_config: MCContainmentSimulationConfig,
    runtime_config: MCSimulationRuntimeConfig,
) -> Any:
    """Run the core one-biopsy/one-relative-structure containment computation."""
    centroid_result = calculate_patient_relative_structure_centroid_distances(
        relative_structure_input,
        biopsy_context,
        num_mc_containment_simulations=num_mc_containment_simulations,
    )
    nearest_neighbour_result = calculate_patient_relative_structure_nearest_neighbour_distances(
        relative_structure_input,
        biopsy_context,
        num_mc_containment_simulations=num_mc_containment_simulations,
        containment_config=containment_config,
        runtime_config=runtime_config,
    )
    containment_result = run_patient_relative_structure_containment_kernel(
        relative_structure_input,
        biopsy_context,
        num_mc_containment_simulations=num_mc_containment_simulations,
        containment_config=containment_config,
        runtime_config=runtime_config,
    )
    return merge_patient_relative_structure_containment_dataframes(
        containment_result.dataframe,
        nearest_neighbour_result.dataframe,
        centroid_result.dataframe,
    )


def _insert_patient_biopsy_identity_columns(dataframe: Any,
                                            biopsy_context: PatientContainmentBiopsyContext) -> Any:
    dataframe.insert(0, MC_CONTAINMENT_SIMULATED_TYPE_COLUMN, biopsy_context.simulated_type)
    dataframe.insert(0, MC_CONTAINMENT_SIMULATED_BOOL_COLUMN, biopsy_context.simulated_bool)
    dataframe.insert(0, MC_CONTAINMENT_BX_INDEX_COLUMN, biopsy_context.biopsy_index)
    dataframe.insert(0, MC_CONTAINMENT_BX_REFNUM_COLUMN, str(biopsy_context.ref_number))
    dataframe.insert(0, MC_CONTAINMENT_BX_ID_COLUMN, biopsy_context.roi)
    dataframe.insert(0, MC_CONTAINMENT_PATIENT_ID_COLUMN, biopsy_context.patient_uid)
    return dataframe


def _add_biopsy_frame_and_voxel_columns(dataframe: Any,
                                        biopsy_context: PatientContainmentBiopsyContext,
                                        *,
                                        biopsy_z_voxel_length: float,
                                        in_place: bool = True) -> Any:
    import dataframe_builders
    import misc_tools

    dataframe_with_vectors = misc_tools.include_vector_columns_in_dataframe(
        dataframe,
        biopsy_context.sampled_points_bx_coord_sys,
        reference_column_name=MC_CONTAINMENT_ORIGINAL_POINT_INDEX_COLUMN,
        new_column_name_x=MC_CONTAINMENT_BX_FRAME_X_COLUMN,
        new_column_name_y=MC_CONTAINMENT_BX_FRAME_Y_COLUMN,
        new_column_name_z=MC_CONTAINMENT_BX_FRAME_Z_COLUMN,
        in_place=in_place,
    )
    return dataframe_builders.add_voxel_columns_helper_func(
        dataframe_with_vectors,
        biopsy_z_voxel_length,
        MC_CONTAINMENT_VOXEL_REFERENCE_COLUMN,
        in_place=in_place,
    )


def _add_binomial_error_columns(dataframe: Any, *, num_mc_containment_simulations: int) -> Any:
    import math_funcs as mf

    dataframe[MC_CONTAINMENT_BINOMIAL_STD_ERR_COLUMN] = dataframe.apply(
        lambda row: mf.binomial_se_estimator(
            row[MC_CONTAINMENT_BINOMIAL_ESTIMATOR_COLUMN],
            int(num_mc_containment_simulations),
            row[MC_CONTAINMENT_TOTAL_SUCCESSES_COLUMN],
        ),
        axis=1,
    )
    confidence_interval_results = dataframe.apply(
        lambda row: mf.binomial_CI_estimator_general(
            row[MC_CONTAINMENT_BINOMIAL_ESTIMATOR_COLUMN],
            int(num_mc_containment_simulations),
            confidence_level=0.95,
        ),
        axis=1,
    )
    dataframe[MC_CONTAINMENT_CI_LOWER_COLUMN] = confidence_interval_results.apply(lambda x: x[0])
    dataframe[MC_CONTAINMENT_CI_UPPER_COLUMN] = confidence_interval_results.apply(lambda x: x[1])
    return dataframe


def compile_patient_containment_independent_probabilities(
    containment_dataframe: Any,
    biopsy_context: PatientContainmentBiopsyContext,
    *,
    num_mc_containment_simulations: int,
    biopsy_z_voxel_length: float,
) -> Any:
    """Compile independent per-structure containment probabilities for one biopsy."""
    shifted_group_columns = [
        MC_CONTAINMENT_RELATIVE_ROI_COLUMN,
        MC_CONTAINMENT_RELATIVE_TYPE_COLUMN,
        MC_CONTAINMENT_POINT_CONTAINED_COLUMN,
        MC_CONTAINMENT_ORIGINAL_POINT_INDEX_COLUMN,
        MC_CONTAINMENT_RELATIVE_INDEX_COLUMN,
    ]
    groupby_columns = [
        MC_CONTAINMENT_RELATIVE_ROI_COLUMN,
        MC_CONTAINMENT_RELATIVE_TYPE_COLUMN,
        MC_CONTAINMENT_RELATIVE_INDEX_COLUMN,
        MC_CONTAINMENT_ORIGINAL_POINT_INDEX_COLUMN,
    ]
    compiled_dataframe = (
        containment_dataframe[containment_dataframe[MC_CONTAINMENT_TRIAL_COLUMN] != 0][shifted_group_columns]
        .groupby(groupby_columns)
        .sum()
        .sort_index()
        .reset_index()
        .rename(columns={MC_CONTAINMENT_POINT_CONTAINED_COLUMN: MC_CONTAINMENT_TOTAL_SUCCESSES_COLUMN})
    )
    compiled_dataframe[MC_CONTAINMENT_BINOMIAL_ESTIMATOR_COLUMN] = (
        compiled_dataframe[MC_CONTAINMENT_TOTAL_SUCCESSES_COLUMN] / int(num_mc_containment_simulations)
    )
    nominal_dataframe = (
        containment_dataframe[containment_dataframe[MC_CONTAINMENT_TRIAL_COLUMN] == 0][
            groupby_columns + [MC_CONTAINMENT_POINT_CONTAINED_COLUMN]
        ]
        .reset_index(drop=True)
        .rename(columns={MC_CONTAINMENT_POINT_CONTAINED_COLUMN: MC_CONTAINMENT_NOMINAL_COLUMN})
    )
    nominal_dataframe = nominal_dataframe.astype({MC_CONTAINMENT_NOMINAL_COLUMN: "uint8"})
    compiled_dataframe = compiled_dataframe.merge(nominal_dataframe, how="inner", on=groupby_columns)
    compiled_dataframe = compiled_dataframe.sort_values(
        [
            MC_CONTAINMENT_RELATIVE_ROI_COLUMN,
            MC_CONTAINMENT_RELATIVE_TYPE_COLUMN,
            MC_CONTAINMENT_RELATIVE_INDEX_COLUMN,
            MC_CONTAINMENT_ORIGINAL_POINT_INDEX_COLUMN,
        ],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    _insert_patient_biopsy_identity_columns(compiled_dataframe, biopsy_context)
    compiled_dataframe = _add_biopsy_frame_and_voxel_columns(
        compiled_dataframe,
        biopsy_context,
        biopsy_z_voxel_length=biopsy_z_voxel_length,
    )
    return _add_binomial_error_columns(
        compiled_dataframe,
        num_mc_containment_simulations=num_mc_containment_simulations,
    )


def compute_patient_containment_sum_to_one_probabilities(containment_dataframe: Any,
                                                         structs_referenced_dict: Mapping[str, Any],
                                                         *,
                                                         default_exterior_tissue: str) -> Any:
    """Compile hierarchy-based sum-to-one tissue probabilities for one biopsy."""
    import pandas
    import misc_tools

    tissue_hierarchy_list = misc_tools.tissue_heirarchy_list_creator_func(
        structs_referenced_dict,
        append_default_exterior_tissue=False,
    )
    shifted_dataframe = containment_dataframe[containment_dataframe[MC_CONTAINMENT_TRIAL_COLUMN] != 0].copy()
    shifted_dataframe["structure_priority"] = pandas.Categorical(
        shifted_dataframe[MC_CONTAINMENT_RELATIVE_TYPE_COLUMN],
        categories=tissue_hierarchy_list,
        ordered=True,
    )
    sorted_shifted_dataframe = shifted_dataframe.sort_values(
        by=[MC_CONTAINMENT_ORIGINAL_POINT_INDEX_COLUMN, MC_CONTAINMENT_TRIAL_COLUMN, "structure_priority"]
    )
    sorted_shifted_dataframe["contained_flag"] = sorted_shifted_dataframe[
        MC_CONTAINMENT_POINT_CONTAINED_COLUMN
    ].astype(int)
    sorted_shifted_dataframe["max_contained_flag"] = sorted_shifted_dataframe.groupby(
        [MC_CONTAINMENT_ORIGINAL_POINT_INDEX_COLUMN, MC_CONTAINMENT_TRIAL_COLUMN, MC_CONTAINMENT_RELATIVE_TYPE_COLUMN]
    )["contained_flag"].transform("max")
    sorted_shifted_dataframe["cumulative_sum"] = sorted_shifted_dataframe.groupby(
        [MC_CONTAINMENT_ORIGINAL_POINT_INDEX_COLUMN, MC_CONTAINMENT_TRIAL_COLUMN]
    )["max_contained_flag"].cumsum()
    filtered_shifted_dataframe = sorted_shifted_dataframe[sorted_shifted_dataframe["cumulative_sum"] <= 1]
    result_dataframe = filtered_shifted_dataframe.groupby(
        [MC_CONTAINMENT_ORIGINAL_POINT_INDEX_COLUMN, MC_CONTAINMENT_RELATIVE_TYPE_COLUMN]
    ).agg({"max_contained_flag": "sum"}).reset_index()

    all_combinations = pandas.MultiIndex.from_product(
        [shifted_dataframe[MC_CONTAINMENT_ORIGINAL_POINT_INDEX_COLUMN].unique(), tissue_hierarchy_list],
        names=[MC_CONTAINMENT_ORIGINAL_POINT_INDEX_COLUMN, MC_CONTAINMENT_RELATIVE_TYPE_COLUMN],
    ).to_frame(index=False)
    final_result = pandas.merge(
        all_combinations,
        result_dataframe,
        on=[MC_CONTAINMENT_ORIGINAL_POINT_INDEX_COLUMN, MC_CONTAINMENT_RELATIVE_TYPE_COLUMN],
        how="left",
    )
    final_result["max_contained_flag"] = final_result["max_contained_flag"].fillna(0).astype(int)

    default_tissue_rows = []
    for point_index in final_result[MC_CONTAINMENT_ORIGINAL_POINT_INDEX_COLUMN].unique():
        num_trials = shifted_dataframe[
            shifted_dataframe[MC_CONTAINMENT_ORIGINAL_POINT_INDEX_COLUMN] == point_index
        ][MC_CONTAINMENT_TRIAL_COLUMN].nunique()
        total_successes = final_result[
            final_result[MC_CONTAINMENT_ORIGINAL_POINT_INDEX_COLUMN] == point_index
        ]["max_contained_flag"].sum()
        default_tissue_rows.append(
            {
                MC_CONTAINMENT_ORIGINAL_POINT_INDEX_COLUMN: point_index,
                MC_CONTAINMENT_RELATIVE_TYPE_COLUMN: default_exterior_tissue,
                "max_contained_flag": num_trials - total_successes,
            }
        )
    final_result = pandas.concat([final_result, pandas.DataFrame(default_tissue_rows)], ignore_index=True)
    final_result.rename(columns={"max_contained_flag": MC_CONTAINMENT_TOTAL_SUCCESSES_COLUMN}, inplace=True)
    final_result[MC_CONTAINMENT_TOTAL_SUCCESSES_COLUMN] = (
        final_result[MC_CONTAINMENT_TOTAL_SUCCESSES_COLUMN].fillna(0).astype(int)
    )

    tissue_class_mapping = {key: value["Tissue class name"] for key, value in structs_referenced_dict.items()}
    tissue_class_mapping[default_exterior_tissue] = default_exterior_tissue
    final_result[MC_CONTAINMENT_RELATIVE_TYPE_COLUMN] = final_result[MC_CONTAINMENT_RELATIVE_TYPE_COLUMN].map(
        tissue_class_mapping
    )
    final_result.rename(columns={MC_CONTAINMENT_RELATIVE_TYPE_COLUMN: MC_CONTAINMENT_TISSUE_CLASS_COLUMN}, inplace=True)

    nominal_dataframe = containment_dataframe[containment_dataframe[MC_CONTAINMENT_TRIAL_COLUMN] == 0].copy()
    nominal_dataframe["structure_priority"] = pandas.Categorical(
        nominal_dataframe[MC_CONTAINMENT_RELATIVE_TYPE_COLUMN],
        categories=tissue_hierarchy_list,
        ordered=True,
    )
    nominal_dataframe = nominal_dataframe.sort_values(
        by=[MC_CONTAINMENT_ORIGINAL_POINT_INDEX_COLUMN, "structure_priority"]
    )
    nominal_dataframe["contained_flag"] = nominal_dataframe[MC_CONTAINMENT_POINT_CONTAINED_COLUMN].astype(int)
    nominal_dataframe["max_contained_flag"] = nominal_dataframe.groupby(
        [MC_CONTAINMENT_ORIGINAL_POINT_INDEX_COLUMN, MC_CONTAINMENT_RELATIVE_TYPE_COLUMN]
    )["contained_flag"].transform("max")
    nominal_dataframe["cumulative_sum"] = nominal_dataframe.groupby(
        [MC_CONTAINMENT_ORIGINAL_POINT_INDEX_COLUMN]
    )["max_contained_flag"].cumsum()
    nominal_filtered = nominal_dataframe[nominal_dataframe["cumulative_sum"] <= 1]
    nominal_result = nominal_filtered.groupby(
        [MC_CONTAINMENT_ORIGINAL_POINT_INDEX_COLUMN, MC_CONTAINMENT_RELATIVE_TYPE_COLUMN]
    ).agg({"max_contained_flag": "sum"}).reset_index()
    nominal_result = pandas.merge(
        all_combinations,
        nominal_result,
        on=[MC_CONTAINMENT_ORIGINAL_POINT_INDEX_COLUMN, MC_CONTAINMENT_RELATIVE_TYPE_COLUMN],
        how="left",
    )
    nominal_result["max_contained_flag"] = nominal_result["max_contained_flag"].fillna(0).astype(int)

    default_nominal_rows = []
    for point_index in nominal_dataframe[MC_CONTAINMENT_ORIGINAL_POINT_INDEX_COLUMN].unique():
        num_trials_nominal = nominal_dataframe[
            nominal_dataframe[MC_CONTAINMENT_ORIGINAL_POINT_INDEX_COLUMN] == point_index
        ][MC_CONTAINMENT_TRIAL_COLUMN].nunique()
        total_successes_nominal = nominal_result[
            nominal_result[MC_CONTAINMENT_ORIGINAL_POINT_INDEX_COLUMN] == point_index
        ]["max_contained_flag"].sum()
        default_nominal_rows.append(
            {
                MC_CONTAINMENT_ORIGINAL_POINT_INDEX_COLUMN: point_index,
                MC_CONTAINMENT_RELATIVE_TYPE_COLUMN: default_exterior_tissue,
                "max_contained_flag": num_trials_nominal - total_successes_nominal,
            }
        )
    nominal_result = pandas.concat([nominal_result, pandas.DataFrame(default_nominal_rows)], ignore_index=True)
    nominal_result.rename(columns={"max_contained_flag": MC_CONTAINMENT_NOMINAL_COLUMN}, inplace=True)
    nominal_result[MC_CONTAINMENT_RELATIVE_TYPE_COLUMN] = nominal_result[MC_CONTAINMENT_RELATIVE_TYPE_COLUMN].map(
        tissue_class_mapping
    )
    nominal_result = nominal_result.sort_values(
        by=[MC_CONTAINMENT_ORIGINAL_POINT_INDEX_COLUMN, MC_CONTAINMENT_RELATIVE_TYPE_COLUMN]
    )
    nominal_result.rename(columns={MC_CONTAINMENT_RELATIVE_TYPE_COLUMN: MC_CONTAINMENT_TISSUE_CLASS_COLUMN}, inplace=True)

    final_result_with_nominal = pandas.merge(
        final_result,
        nominal_result[
            [MC_CONTAINMENT_ORIGINAL_POINT_INDEX_COLUMN, MC_CONTAINMENT_TISSUE_CLASS_COLUMN, MC_CONTAINMENT_NOMINAL_COLUMN]
        ],
        on=[MC_CONTAINMENT_ORIGINAL_POINT_INDEX_COLUMN, MC_CONTAINMENT_TISSUE_CLASS_COLUMN],
        how="left",
    )
    final_result_with_nominal = final_result_with_nominal[
        [
            MC_CONTAINMENT_TISSUE_CLASS_COLUMN,
            MC_CONTAINMENT_ORIGINAL_POINT_INDEX_COLUMN,
            MC_CONTAINMENT_TOTAL_SUCCESSES_COLUMN,
            MC_CONTAINMENT_NOMINAL_COLUMN,
        ]
    ]
    return final_result_with_nominal.sort_values(
        by=[MC_CONTAINMENT_ORIGINAL_POINT_INDEX_COLUMN, MC_CONTAINMENT_TISSUE_CLASS_COLUMN]
    ).reset_index(drop=True)


def compile_patient_containment_sum_to_one_probabilities(
    containment_dataframe: Any,
    biopsy_context: PatientContainmentBiopsyContext,
    structs_referenced_dict: Mapping[str, Any],
    *,
    num_mc_containment_simulations: int,
    biopsy_z_voxel_length: float,
    default_exterior_tissue: str,
) -> Any:
    """Compile hierarchy-based tissue probabilities plus biopsy metadata for one biopsy."""
    sum_to_one_dataframe = compute_patient_containment_sum_to_one_probabilities(
        containment_dataframe,
        structs_referenced_dict,
        default_exterior_tissue=default_exterior_tissue,
    )
    sum_to_one_dataframe[MC_CONTAINMENT_BINOMIAL_ESTIMATOR_COLUMN] = (
        sum_to_one_dataframe[MC_CONTAINMENT_TOTAL_SUCCESSES_COLUMN] / int(num_mc_containment_simulations)
    )
    sum_to_one_dataframe = sum_to_one_dataframe.astype({MC_CONTAINMENT_NOMINAL_COLUMN: "uint8"})
    _insert_patient_biopsy_identity_columns(sum_to_one_dataframe, biopsy_context)
    sum_to_one_dataframe = _add_biopsy_frame_and_voxel_columns(
        sum_to_one_dataframe,
        biopsy_context,
        biopsy_z_voxel_length=biopsy_z_voxel_length,
    )
    return _add_binomial_error_columns(
        sum_to_one_dataframe,
        num_mc_containment_simulations=num_mc_containment_simulations,
    )


def compile_patient_containment_distance_summaries(containment_dataframe: Any,
                                                   biopsy_context: PatientContainmentBiopsyContext,
                                                   *,
                                                   biopsy_z_voxel_length: float) -> PatientContainmentDistanceSummaryOutputs:
    """Compile global, point-wise, and voxel-wise distance summary dataframes."""
    import dataframe_builders

    distance_value_columns = list(MC_CONTAINMENT_DISTANCE_VALUE_COLUMNS)
    global_group_columns = [
        MC_CONTAINMENT_PATIENT_ID_COLUMN,
        MC_CONTAINMENT_BX_ID_COLUMN,
        MC_CONTAINMENT_BX_INDEX_COLUMN,
        MC_CONTAINMENT_RELATIVE_ROI_COLUMN,
        MC_CONTAINMENT_RELATIVE_TYPE_COLUMN,
        MC_CONTAINMENT_RELATIVE_INDEX_COLUMN,
    ]
    global_dataframe = containment_dataframe.groupby(global_group_columns)[distance_value_columns].describe(
        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    )
    global_dataframe = dataframe_builders.convert_columns_to_categorical_and_downcast(
        global_dataframe,
        threshold=0.25,
    )
    global_dataframe.reset_index(inplace=True)

    dataframe_with_vectors = _add_biopsy_frame_and_voxel_columns(
        containment_dataframe,
        biopsy_context,
        biopsy_z_voxel_length=biopsy_z_voxel_length,
        in_place=False,
    )
    point_wise_group_columns = [
        MC_CONTAINMENT_PATIENT_ID_COLUMN,
        MC_CONTAINMENT_BX_ID_COLUMN,
        MC_CONTAINMENT_BX_INDEX_COLUMN,
        MC_CONTAINMENT_SIMULATED_BOOL_COLUMN,
        MC_CONTAINMENT_SIMULATED_TYPE_COLUMN,
        MC_CONTAINMENT_RELATIVE_ROI_COLUMN,
        MC_CONTAINMENT_RELATIVE_TYPE_COLUMN,
        MC_CONTAINMENT_RELATIVE_INDEX_COLUMN,
        MC_CONTAINMENT_ORIGINAL_POINT_INDEX_COLUMN,
        MC_CONTAINMENT_BX_FRAME_X_COLUMN,
        MC_CONTAINMENT_BX_FRAME_Y_COLUMN,
        MC_CONTAINMENT_BX_FRAME_Z_COLUMN,
        "Voxel index",
        "Voxel begin (Z)",
        "Voxel end (Z)",
    ]
    point_wise_dataframe = dataframe_with_vectors.groupby(point_wise_group_columns)[distance_value_columns].describe(
        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    )
    point_wise_dataframe.reset_index(inplace=True)

    voxel_wise_group_columns = [
        MC_CONTAINMENT_PATIENT_ID_COLUMN,
        MC_CONTAINMENT_BX_ID_COLUMN,
        MC_CONTAINMENT_BX_INDEX_COLUMN,
        MC_CONTAINMENT_SIMULATED_BOOL_COLUMN,
        MC_CONTAINMENT_SIMULATED_TYPE_COLUMN,
        MC_CONTAINMENT_RELATIVE_ROI_COLUMN,
        MC_CONTAINMENT_RELATIVE_TYPE_COLUMN,
        MC_CONTAINMENT_RELATIVE_INDEX_COLUMN,
        "Voxel index",
        "Voxel begin (Z)",
        "Voxel end (Z)",
    ]
    voxel_wise_dataframe = dataframe_with_vectors.groupby(voxel_wise_group_columns)[distance_value_columns].describe(
        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    )
    voxel_wise_dataframe.reset_index(inplace=True)
    return PatientContainmentDistanceSummaryOutputs(
        global_dataframe=global_dataframe,
        point_wise_dataframe=point_wise_dataframe,
        voxel_wise_dataframe=voxel_wise_dataframe,
    )


def build_patient_containment_light_trials_dataframe(containment_dataframe: Any,
                                                     biopsy_context: PatientContainmentBiopsyContext,
                                                     *,
                                                     biopsy_z_voxel_length: float) -> Any:
    """Build the optional lighter all-trials containment/distance dataframe."""
    import dataframe_builders

    light_dataframe = containment_dataframe.drop(columns=list(MC_CONTAINMENT_LIGHT_DROP_COLUMNS))
    light_dataframe = _add_biopsy_frame_and_voxel_columns(
        light_dataframe,
        biopsy_context,
        biopsy_z_voxel_length=biopsy_z_voxel_length,
        in_place=False,
    )
    return dataframe_builders.convert_columns_to_categorical_and_downcast(
        light_dataframe,
        threshold=0.25,
    )


def build_patient_containment_legacy_structure_results(
    inventory: PatientRelativeStructureInventory,
    compiled_results_dataframe: Any,
) -> dict[RelativeStructureInfo, dict[str, Any]]:
    """Build the legacy per-structure containment-results dictionary for one biopsy."""
    compiled_results_dict = inventory.relative_structure_template.copy()
    for structure_info in compiled_results_dict.keys():
        structure_roi = structure_info[0]
        non_bx_structure_type = structure_info[1]
        structure_index = structure_info[3]
        structure_dataframe = compiled_results_dataframe[
            (compiled_results_dataframe[MC_CONTAINMENT_RELATIVE_ROI_COLUMN] == structure_roi)
            & (compiled_results_dataframe[MC_CONTAINMENT_RELATIVE_TYPE_COLUMN] == non_bx_structure_type)
            & (compiled_results_dataframe[MC_CONTAINMENT_RELATIVE_INDEX_COLUMN] == structure_index)
        ].sort_values(by=[MC_CONTAINMENT_ORIGINAL_POINT_INDEX_COLUMN])
        structure_specific_results_dict = build_structure_specific_results_template()
        structure_specific_results_dict["Total successes (containment) list"] = structure_dataframe[
            MC_CONTAINMENT_TOTAL_SUCCESSES_COLUMN
        ].to_list()
        structure_specific_results_dict["Binomial estimator list"] = structure_dataframe[
            MC_CONTAINMENT_BINOMIAL_ESTIMATOR_COLUMN
        ].to_list()
        structure_specific_results_dict["Nominal containment list"] = structure_dataframe[
            MC_CONTAINMENT_NOMINAL_COLUMN
        ].to_list()
        compiled_results_dict[structure_info] = structure_specific_results_dict
    return compiled_results_dict


def add_binomial_statistics_to_patient_containment_legacy_results(
    compiled_results_dict: dict[RelativeStructureInfo, dict[str, Any]],
    *,
    num_mc_containment_simulations: int,
    parallel_pool: Any | None = None,
) -> dict[RelativeStructureInfo, dict[str, Any]]:
    """Add legacy confidence intervals and standard errors to per-structure result lists."""
    import math_funcs as mf

    for structure_specific_results_dict in compiled_results_dict.values():
        probability_estimator_list = structure_specific_results_dict["Binomial estimator list"]
        num_successes_list = structure_specific_results_dict["Total successes (containment) list"]
        args_list = [
            (probability_estimator_list[index], int(num_mc_containment_simulations), num_successes_list[index])
            for index in range(len(probability_estimator_list))
        ]
        if parallel_pool is None:
            confidence_interval_list = [mf.binomial_CI_estimator(*args) for args in args_list]
            standard_err_list = [mf.binomial_se_estimator(*args) for args in args_list]
        else:
            confidence_interval_list = parallel_pool.starmap(mf.binomial_CI_estimator, args_list)
            standard_err_list = parallel_pool.starmap(mf.binomial_se_estimator, args_list)
        structure_specific_results_dict["Confidence interval 95 (containment) list"] = confidence_interval_list
        structure_specific_results_dict["Standard error (containment) list"] = standard_err_list
    return compiled_results_dict


def compile_patient_containment_biopsy_statistics(
    containment_dataframe: Any,
    biopsy_context: PatientContainmentBiopsyContext,
    inventory: PatientRelativeStructureInventory,
    structs_referenced_dict: Mapping[str, Any],
    *,
    num_mc_containment_simulations: int,
    biopsy_z_voxel_length: float,
    default_exterior_tissue: str,
    keep_light_containment_and_distances_dataframe: bool,
    parallel_pool: Any | None = None,
) -> PatientContainmentBiopsyStatisticsOutputs:
    """Compile all per-biopsy containment statistics produced by the oracle loop."""
    import dataframe_builders

    compiled_results_dataframe = compile_patient_containment_independent_probabilities(
        containment_dataframe,
        biopsy_context,
        num_mc_containment_simulations=num_mc_containment_simulations,
        biopsy_z_voxel_length=biopsy_z_voxel_length,
    )
    sum_to_one_results_dataframe = compile_patient_containment_sum_to_one_probabilities(
        containment_dataframe,
        biopsy_context,
        structs_referenced_dict,
        num_mc_containment_simulations=num_mc_containment_simulations,
        biopsy_z_voxel_length=biopsy_z_voxel_length,
        default_exterior_tissue=default_exterior_tissue,
    )
    distance_outputs = compile_patient_containment_distance_summaries(
        containment_dataframe,
        biopsy_context,
        biopsy_z_voxel_length=biopsy_z_voxel_length,
    )
    light_trials_dataframe = None
    if keep_light_containment_and_distances_dataframe:
        light_trials_dataframe = build_patient_containment_light_trials_dataframe(
            containment_dataframe,
            biopsy_context,
            biopsy_z_voxel_length=biopsy_z_voxel_length,
        )
    compiled_results_dict = build_patient_containment_legacy_structure_results(
        inventory,
        compiled_results_dataframe,
    )
    compiled_results_dict = add_binomial_statistics_to_patient_containment_legacy_results(
        compiled_results_dict,
        num_mc_containment_simulations=num_mc_containment_simulations,
        parallel_pool=parallel_pool,
    )
    compiled_results_dataframe = dataframe_builders.convert_columns_to_categorical_and_downcast(
        compiled_results_dataframe,
        threshold=0.25,
    )
    sum_to_one_results_dataframe = dataframe_builders.convert_columns_to_categorical_and_downcast(
        sum_to_one_results_dataframe,
        threshold=0.25,
    )
    return PatientContainmentBiopsyStatisticsOutputs(
        compiled_results_dataframe=compiled_results_dataframe,
        sum_to_one_results_dataframe=sum_to_one_results_dataframe,
        compiled_results_dict=compiled_results_dict,
        distance_outputs=distance_outputs,
        light_trials_dataframe=light_trials_dataframe,
    )


def write_patient_containment_biopsy_statistics_to_legacy_record(
    biopsy_structure: dict[str, Any],
    statistics_outputs: PatientContainmentBiopsyStatisticsOutputs,
) -> dict[str, Any]:
    """Write compiled containment statistics to one legacy biopsy record."""
    biopsy_structure.update(statistics_outputs.legacy_biopsy_updates())
    return biopsy_structure


def build_patient_containment_dilated_structure_bank(
    patient_uid: str,
    patient_reference_dict: Mapping[str, Any],
    inventory: PatientRelativeStructureInventory,
    *,
    num_mc_containment_simulations: int,
    oar_ref: str,
    rectum_ref: str,
    urethra_ref: str,
    containment_config: MCContainmentSimulationConfig,
    parallel_pool: Any,
) -> PatientContainmentDilatedStructureBank:
    """Build the patient-level dilated relative-structure bank used by containment."""
    import cupy as cp
    import numpy as np
    import polygon_dilation_helpers_numpy

    intermediate_keys = legacy_mc_keys.containment_intermediates
    geometry_keys = legacy_data_keys.structure_geometry
    dilated_structures_by_structure: dict[RelativeStructureInfo, list[Any]] = {}
    centroids_by_structure: dict[RelativeStructureInfo, Any] = {}
    relative_structure_mapping_by_structure: dict[RelativeStructureInfo, Any] = {}

    for structure_info in inventory.relative_structure_infos:
        non_bx_structure_type = structure_info[1]
        structure_index = structure_info[3]
        non_bx_struct_zslices_list = _structure_zslices_for_containment(
            patient_reference_dict,
            non_bx_structure_type=non_bx_structure_type,
            structure_index=structure_index,
            oar_ref=oar_ref,
            rectum_ref=rectum_ref,
            urethra_ref=urethra_ref,
        )
        dilation_samples = cp.asnumpy(
            patient_reference_dict[non_bx_structure_type][structure_index][
                intermediate_keys.normal_dist_dilations_samples_array_key
            ]
        )
        nominal_centroid = patient_reference_dict[non_bx_structure_type][structure_index][
            geometry_keys.structure_global_centroid_key
        ].copy()

        if not dilation_samples.any():
            dilated_structures_by_structure[structure_info] = [non_bx_struct_zslices_list]
            centroids_by_structure[structure_info] = np.reshape(nominal_centroid, (1, 3))
            relative_structure_mapping_by_structure[structure_info] = np.zeros(
                int(num_mc_containment_simulations) + 1,
                dtype=int,
            )
            continue

        org_config_2d_arr, org_config_indices_slices_arr = (
            polygon_dilation_helpers_numpy.convert_to_2d_array_and_indices_numpy(non_bx_struct_zslices_list)
        )
        dilated_structures_list, dilated_structures_slices_indices_list = (
            polygon_dilation_helpers_numpy.generate_dilated_structures_parallelized(
                org_config_2d_arr,
                org_config_indices_slices_arr,
                dilation_samples,
                containment_config.show_non_bx_relative_structure_z_dilation_bool,
                containment_config.show_non_bx_relative_structure_xy_dilation_bool,
                parallel_pool,
            )
        )

        reconstructed_dilated_structures = []
        centroids_of_each_dilated_structure = np.empty([len(dilated_structures_list), 3])
        for dilated_structure_index, dilated_structure_2d_arr in enumerate(dilated_structures_list):
            reconstructed_dilated_structures.append(
                polygon_dilation_helpers_numpy.reconstruct_list_from_2d_array(
                    dilated_structure_2d_arr,
                    dilated_structures_slices_indices_list[dilated_structure_index],
                )
            )
            centroids_of_each_dilated_structure[dilated_structure_index, :] = np.mean(
                dilated_structure_2d_arr,
                axis=0,
            )

        nominal_and_dilated_structures = [non_bx_struct_zslices_list] + reconstructed_dilated_structures
        dilated_structures_by_structure[structure_info] = nominal_and_dilated_structures
        centroids_by_structure[structure_info] = np.vstack((nominal_centroid, centroids_of_each_dilated_structure))
        relative_structure_mapping_by_structure[structure_info] = np.arange(0, len(nominal_and_dilated_structures))

        del dilated_structures_list
        del dilated_structures_slices_indices_list

    return PatientContainmentDilatedStructureBank(
        patient_uid=str(patient_uid),
        dilated_structures_by_structure=dilated_structures_by_structure,
        centroids_by_structure=centroids_by_structure,
        relative_structure_mapping_by_structure=relative_structure_mapping_by_structure,
    )


def collect_patient_containment_outputs(patient_uid: str,
                                        patient_reference_dict: Mapping[str, Any],
                                        *,
                                        bx_ref: str) -> PatientContainmentOutputs:
    """Collect containment artifacts written into one patient dictionary."""
    identity_keys = legacy_mc_keys.biopsy_identity
    biopsy_outputs: list[dict[str, Any]] = []
    for biopsy_index, biopsy_structure in enumerate(patient_reference_dict.get(bx_ref, ())):
        outputs = {
            output_key: biopsy_structure.get(output_key)
            for output_key in MC_CONTAINMENT_BIOPSY_OUTPUT_KEYS
            if output_key in biopsy_structure
        }
        if outputs:
            biopsy_outputs.append(
                {
                    identity_keys.roi_key: biopsy_structure.get(identity_keys.roi_key),
                    identity_keys.ref_number_key: biopsy_structure.get(identity_keys.ref_number_key),
                    identity_keys.index_number_key: biopsy_structure.get(identity_keys.index_number_key, biopsy_index),
                    identity_keys.simulated_bool_key: biopsy_structure.get(identity_keys.simulated_bool_key),
                    identity_keys.simulated_type_key: biopsy_structure.get(identity_keys.simulated_type_key),
                    "outputs": outputs,
                }
            )
    return PatientContainmentOutputs(
        patient_uid=str(patient_uid),
        biopsy_outputs=biopsy_outputs,
    )
