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
