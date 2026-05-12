from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any
from typing import Optional

import anatomy_reconstructor_tools
import dataframe_builders
import misc_tools
import mr_localizers
import numpy as np
import open3d as o3d
import pandas
import plotting_funcs
import point_containment_tools

from preprocessing.interpolation.interpolation import interpolation_information_obj
from startup.runtime_logging import get_active_runtime_logger


STRUCTURE_PREPROCESSING_TIMINGS_DF_KEY = "Structure preprocessing timings"


@dataclass(frozen=True)
class NonBiopsyStructurePreprocessingConfig:
    all_ref_key: str
    oar_ref: str
    dil_ref: str
    mr_adc_ref: str
    interp_inter_slice_dist: float
    interp_intra_slice_dist: float
    interp_dist_caps: float
    radius_for_normals_estimation: float
    max_nn_for_normals_estimation: int
    voxel_size_for_structure_volume_calc_non_bx: float
    voxel_size_for_structure_dimension_calc: float
    factor_for_voxel_size: float
    cupy_array_upper_limit_NxN_size_input: Any
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


def _hide_indeterminate_task(indeterminate_progress_sub, task_id) -> None:
    if indeterminate_progress_sub is None or task_id is None:
        return
    indeterminate_progress_sub.update(task_id, visible=False)


def _append_or_initialize_dataframe(patient_output_dataframes_dict, dataframe_key, dataframe_to_add):
    existing_dataframe = patient_output_dataframes_dict.get(dataframe_key)
    if existing_dataframe is None:
        patient_output_dataframes_dict[dataframe_key] = dataframe_to_add
        return
    patient_output_dataframes_dict[dataframe_key] = pandas.concat(
        [existing_dataframe, dataframe_to_add],
        ignore_index=True,
    )


def _log_phase_timing(
    runtime_logger,
    phase_name,
    elapsed_seconds,
    *,
    patient_uid,
    structure_id,
    structure_ref_type,
    structure_index,
    details=None,
) -> None:
    if runtime_logger is None:
        return
    resolved_details = {"elapsed_seconds": round(float(elapsed_seconds), 6)}
    if details is not None:
        resolved_details.update(details)
    runtime_logger.checkpoint(
        phase_name,
        "Completed structure preprocessing phase.",
        patient_uid=patient_uid,
        structure_id=structure_id,
        structure_ref_type=structure_ref_type,
        structure_index=structure_index,
        details=resolved_details,
    )


def _build_dil_shape_feature_overrides(
    *,
    pydicom_item,
    specific_structure,
    sp_patient_selected_structure_info_dataframe,
    config,
):
    default_overrides = {
        "DIL centroid (X, prostate frame)": [None],
        "DIL centroid (Y, prostate frame)": [None],
        "DIL centroid (Z, prostate frame)": [None],
        "DIL centroid distance (prostate frame)": [None],
        "DIL prostate sextant (LR)": [None],
        "DIL prostate sextant (AP)": [None],
        "DIL prostate sextant (SI)": [None],
    }
    if sp_patient_selected_structure_info_dataframe is None or len(sp_patient_selected_structure_info_dataframe) == 0:
        return default_overrides

    selected_prostate_df = sp_patient_selected_structure_info_dataframe[
        sp_patient_selected_structure_info_dataframe["Struct ref type"] == config.oar_ref
    ]
    if len(selected_prostate_df) == 0:
        return default_overrides

    selected_prostate_info = selected_prostate_df.to_dict("records")[0]
    if selected_prostate_info.get("Struct found bool") is not True:
        return default_overrides

    prostate_structure_index = selected_prostate_info.get("Index number")
    if prostate_structure_index is None:
        return default_overrides

    prostate_structure = pydicom_item[config.oar_ref][prostate_structure_index]
    prostate_structure_global_centroid = np.array(
        prostate_structure["Structure global centroid"]
    ).reshape(3)
    prostate_dimension_at_centroid_dict = prostate_structure.get("Structure dimension at centroid dict") or {}
    prostate_z_dimension_length_at_centroid = prostate_dimension_at_centroid_dict.get(
        "Z dimension length at centroid"
    )
    if prostate_z_dimension_length_at_centroid is None:
        return default_overrides

    distance_to_mid_gland_threshold = abs(prostate_z_dimension_length_at_centroid / 6)
    specific_structure_global_centroid = np.array(
        specific_structure["Structure global centroid"]
    ).reshape(3)
    specific_structure_global_centroid_in_prostate_frame = (
        specific_structure_global_centroid - prostate_structure_global_centroid
    )
    dil_prostate_position_dict = misc_tools.bx_position_classifier_in_prostate_frame_sextant(
        specific_structure_global_centroid_in_prostate_frame,
        distance_to_mid_gland_threshold,
    )
    return {
        "DIL centroid (X, prostate frame)": [specific_structure_global_centroid_in_prostate_frame[0]],
        "DIL centroid (Y, prostate frame)": [specific_structure_global_centroid_in_prostate_frame[1]],
        "DIL centroid (Z, prostate frame)": [specific_structure_global_centroid_in_prostate_frame[2]],
        "DIL centroid distance (prostate frame)": [
            np.linalg.norm(specific_structure_global_centroid_in_prostate_frame)
        ],
        "DIL prostate sextant (LR)": [dil_prostate_position_dict["LR"]],
        "DIL prostate sextant (AP)": [dil_prostate_position_dict["AP"]],
        "DIL prostate sextant (SI)": [dil_prostate_position_dict["SI"]],
    }


def preprocess_non_biopsy_structure(
    *,
    patient_uid,
    pydicom_item,
    master_structure_reference_dict,
    struct_ref_type,
    specific_structure_index,
    structs_referenced_dict,
    config,
    parallel_pool,
    layout_groups,
    structures_progress,
    indeterminate_progress_sub,
    important_info,
    live_display,
    runtime_logger=None,
    sp_patient_selected_structure_info_dataframe=None,
):
    specific_structure = master_structure_reference_dict[patient_uid][struct_ref_type][specific_structure_index]
    structure_id = specific_structure["ROI"]
    structure_reference_number = specific_structure["Ref #"]
    patient_output_dataframes_dict = master_structure_reference_dict[patient_uid][config.all_ref_key][
        "Multi-structure pre-processing output dataframes dict"
    ]
    runtime_logger = runtime_logger or get_active_runtime_logger()
    phase_timings = {}
    structure_start_time = time.perf_counter()

    if runtime_logger is not None:
        runtime_logger.phase_start(
            "preprocessing.structure",
            "Starting non-biopsy structure preprocessing.",
            patient_uid=patient_uid,
            structure_id=structure_id,
            structure_ref_type=struct_ref_type,
            structure_index=specific_structure_index,
        )

    def run_phase(phase_key, task_description, phase_callable, *, refresh_live_display=False):
        nonlocal live_display
        task_id = None
        if indeterminate_progress_sub is not None:
            task_id = indeterminate_progress_sub.add_task(task_description, total=None)
        if refresh_live_display and live_display is not None:
            live_display.refresh()
        phase_start_time = time.perf_counter()
        try:
            result = phase_callable()
        finally:
            _hide_indeterminate_task(indeterminate_progress_sub, task_id)
        elapsed_seconds = time.perf_counter() - phase_start_time
        phase_timings[f"{phase_key}_elapsed_seconds"] = elapsed_seconds
        _log_phase_timing(
            runtime_logger,
            f"preprocessing.structure.{phase_key}",
            elapsed_seconds,
            patient_uid=patient_uid,
            structure_id=structure_id,
            structure_ref_type=struct_ref_type,
            structure_index=specific_structure_index,
        )
        return result

    def build_raw_data():
        threeDdata_zslice_list = specific_structure["Raw contour pts zslice list"].copy()
        total_structure_points = sum(np.shape(x)[0] for x in threeDdata_zslice_list)
        threeDdata_array = np.empty([total_structure_points, 3])
        lower_bound_index = 0
        for threeDdata_zslice in threeDdata_zslice_list:
            current_zslice_num_points = np.size(threeDdata_zslice, 0)
            threeDdata_array[
                lower_bound_index:lower_bound_index + current_zslice_num_points
            ] = threeDdata_zslice
            lower_bound_index = lower_bound_index + current_zslice_num_points
        return threeDdata_zslice_list, threeDdata_array

    threeDdata_zslice_list, threeDdata_array = run_phase(
        "raw_data",
        "[cyan]~~Build raw data",
        build_raw_data,
    )

    def interpolate_structure():
        interslice_interpolation_information, threeDdata_equal_pt_zslice_list = (
            anatomy_reconstructor_tools.inter_zslice_interpolator(
                parallel_pool,
                threeDdata_zslice_list,
                config.interp_inter_slice_dist,
            )
        )
        threeDdata_to_intra_zslice_interpolate_zslice_list = (
            interslice_interpolation_information.interpolated_pts_list
        )
        num_z_slices_data_to_intra_slice_interpolate = len(
            threeDdata_to_intra_zslice_interpolate_zslice_list
        )
        interpolation_information = interpolation_information_obj(
            num_z_slices_data_to_intra_slice_interpolate
        )
        interpolation_information.serial_analyze(
            threeDdata_to_intra_zslice_interpolate_zslice_list,
            config.interp_intra_slice_dist,
        )

        first_zslice = threeDdata_to_intra_zslice_interpolate_zslice_list[0]
        last_zslice = threeDdata_to_intra_zslice_interpolate_zslice_list[-1]
        interpolation_information.create_fill_new_v2(
            first_zslice,
            config.interp_dist_caps,
            kernel_type=config.custom_cuda_kernel_type,
        )
        interpolation_information.create_fill_new_v2(
            last_zslice,
            config.interp_dist_caps,
            kernel_type=config.custom_cuda_kernel_type,
        )

        pcd_color = structs_referenced_dict[struct_ref_type]["PCD color"]
        threeDdata_point_cloud = point_containment_tools.create_point_cloud(
            threeDdata_array,
            pcd_color,
        )
        threeDdata_array_fully_interpolated = interpolation_information.interpolated_pts_np_arr
        threeDdata_array_fully_interpolated_with_end_caps = (
            interpolation_information.interpolated_pts_with_end_caps_np_arr
        )
        threeDdata_array_interslice_interpolation = np.vstack(
            interslice_interpolation_information.interpolated_pts_list
        )
        interslice_interp_pcd = point_containment_tools.create_point_cloud(
            threeDdata_array_interslice_interpolation,
            pcd_color,
        )
        inter_and_intra_interp_pcd = point_containment_tools.create_point_cloud(
            threeDdata_array_fully_interpolated,
            pcd_color,
        )
        inter_and_intra_and_end_caps_interp_pcd = point_containment_tools.create_point_cloud(
            threeDdata_array_fully_interpolated_with_end_caps,
            pcd_color,
        )
        interpolated_pcd_dict = {
            "Interslice": interslice_interp_pcd,
            "Full": inter_and_intra_interp_pcd,
            "Full with end caps": inter_and_intra_and_end_caps_interp_pcd,
        }
        return (
            interslice_interpolation_information,
            threeDdata_equal_pt_zslice_list,
            interpolation_information,
            threeDdata_point_cloud,
            interpolated_pcd_dict,
            threeDdata_array_fully_interpolated_with_end_caps,
        )

    (
        interslice_interpolation_information,
        threeDdata_equal_pt_zslice_list,
        interpolation_information,
        threeDdata_point_cloud,
        interpolated_pcd_dict,
        threeDdata_array_fully_interpolated_with_end_caps,
    ) = run_phase(
        "interpolate",
        "[cyan]~~Interpolate structure",
        interpolate_structure,
    )

    structure_info = misc_tools.specific_structure_info_dict_creator(
        "given",
        specific_structure=specific_structure,
    )
    zslices_list = interslice_interpolation_information.interpolated_pts_list
    mr_adc_value_column_name_str = "MR ADC value"
    mr_adc_enabled = config.mr_adc_ref in pydicom_item

    if mr_adc_enabled:
        adc_mr_phys_space_arr = mr_localizers.grab_mr_adc_2d_arr(
            pydicom_item,
            config.mr_adc_ref,
            filter_out_negatives=True,
        )

        def determine_mr_containment():
            containment_info_for_all_lattice_points_grand_pandas_dataframe = (
                mr_localizers.test_points_of_given_2d_lattice_from_within_given_structure_and_return_dataframe_type_2III(
                    adc_mr_phys_space_arr,
                    zslices_list,
                    structure_info,
                    config.constant_z_slice_polygons_handler_option,
                    config.remove_consecutive_duplicate_points_in_polygons,
                    config.custom_cuda_kernel_type,
                    associated_value_str=mr_adc_value_column_name_str,
                )
            )
            if config.demonstrate_mr_adc_pcd_containment_correctness_bool:
                plotting_funcs.plot_containment_info_dataframe_to_point_cloud_plus_other_clouds(
                    containment_info_for_all_lattice_points_grand_pandas_dataframe,
                    "Test pt X",
                    "Test pt Y",
                    "Test pt Z",
                    "Pt clr R",
                    "Pt clr G",
                    "Pt clr B",
                    additional_point_clouds=[interpolated_pcd_dict["Full with end caps"]],
                )
            return containment_info_for_all_lattice_points_grand_pandas_dataframe

        containment_info_for_all_lattice_points_grand_pandas_dataframe = run_phase(
            "mr_adc_determine_containment",
            "[cyan]~~Calculating MR statistics (determining containment)",
            determine_mr_containment,
        )

        def compute_mr_statistics():
            return dataframe_builders.dataframe_mr_summary_statistics(
                containment_info_for_all_lattice_points_grand_pandas_dataframe,
                mr_adc_value_column_name_str,
                filter_column="Pt contained bool",
                filter_value=True,
            )

        mr_adc_value_summary_statistics_specific_structure = run_phase(
            "mr_adc_compute_statistics",
            "[cyan]~~Calculating MR statistics (computing statistics)",
            compute_mr_statistics,
        )

        def update_prostate_only_points():
            if struct_ref_type == config.oar_ref:
                patient_output_dataframes_dict[
                    "Prostate only points MR ADC dataframe (temporary for pre-processing)"
                ] = containment_info_for_all_lattice_points_grand_pandas_dataframe[
                    containment_info_for_all_lattice_points_grand_pandas_dataframe["Pt contained bool"] == True
                ]
                return

            containment_info_for_all_lattice_points_grand_pandas_dataframe_prostate_only = (
                patient_output_dataframes_dict.get(
                    "Prostate only points MR ADC dataframe (temporary for pre-processing)"
                )
            )
            if containment_info_for_all_lattice_points_grand_pandas_dataframe_prostate_only is None:
                return
            containment_info_for_all_lattice_points_grand_pandas_dataframe_prostate_only = (
                dataframe_builders.drop_rows_where_b_is_true(
                    containment_info_for_all_lattice_points_grand_pandas_dataframe_prostate_only,
                    containment_info_for_all_lattice_points_grand_pandas_dataframe,
                    index_col="Test pt index",
                    flag_col="Pt contained bool",
                    keep_unmatched=True,
                )
            )
            patient_output_dataframes_dict[
                "Prostate only points MR ADC dataframe (temporary for pre-processing)"
            ] = containment_info_for_all_lattice_points_grand_pandas_dataframe_prostate_only

        run_phase(
            "mr_adc_update_prostate_only",
            "[cyan]~~Keeping track of prostate only MR ADC values",
            update_prostate_only_points,
        )
        del containment_info_for_all_lattice_points_grand_pandas_dataframe
        _append_or_initialize_dataframe(
            patient_output_dataframes_dict,
            "MR - ADC - summary statistics by structure dataframe",
            mr_adc_value_summary_statistics_specific_structure,
        )

    def compute_structure_volume():
        interpolated_pts_np_arr = interslice_interpolation_information.interpolated_pts_np_arr
        interpolated_zvals_list = (
            interslice_interpolation_information.zslice_vals_after_interpolation_list
        )
        return misc_tools.structure_volume_calculator(
            interpolated_pts_np_arr,
            interpolated_zvals_list,
            zslices_list,
            structure_info,
            patient_uid,
            config.voxel_size_for_structure_volume_calc_non_bx,
            config.factor_for_voxel_size,
            config.cupy_array_upper_limit_NxN_size_input,
            layout_groups,
            config.nearest_zslice_vals_and_indices_cupy_generic_max_size,
            structures_progress,
            live_display,
            generate_cuda_log_files_volume_calculation=config.generate_cuda_log_files_volume_calculation,
            constant_z_slice_polygons_handler_option=config.constant_z_slice_polygons_handler_option,
            remove_consecutive_duplicate_points_in_polygons=(
                config.remove_consecutive_duplicate_points_in_polygons
            ),
            include_edges_in_log_files=config.include_edges_in_log_files,
            custom_cuda_kernel_type=config.custom_cuda_kernel_type,
            demonstrate_volume_calculation_correctness_bool_1=(
                config.demonstrate_volume_calculation_correctness_bool_1
            ),
            plot_volume_calculation_containment_result_bool_1_old=(
                config.plot_volume_calculation_containment_result_bool_1_old
            ),
            plot_binary_mask_bool=config.plot_binary_mask_bool,
            other_pcds_to_plot_list=[interpolated_pcd_dict["Full with end caps"]],
        )

    (
        structure_volume,
        maximum_distance,
        voxel_size_for_structure_volume_calc,
        binary_mask_arr,
        live_display,
    ) = run_phase(
        "volume",
        "[cyan]~~Calculating structure volume",
        compute_structure_volume,
    )

    def compute_structure_dimensions():
        interpolated_pts_np_arr = interslice_interpolation_information.interpolated_pts_np_arr
        interpolated_zvals_list = (
            interslice_interpolation_information.zslice_vals_after_interpolation_list
        )
        non_bx_structure_global_centroid = np.array(
            specific_structure["Structure global centroid"]
        ).reshape(3)
        return misc_tools.structure_dimensions_calculator(
            interpolated_pts_np_arr,
            interpolated_zvals_list,
            zslices_list,
            non_bx_structure_global_centroid,
            structure_info,
            patient_uid,
            config.voxel_size_for_structure_dimension_calc,
            config.factor_for_voxel_size,
            config.cupy_array_upper_limit_NxN_size_input,
            layout_groups,
            config.nearest_zslice_vals_and_indices_cupy_generic_max_size,
            structures_progress,
            live_display,
            generate_cuda_log_files_structure_dimension_calculation=(
                config.generate_cuda_log_files_structure_dimension_calculation
            ),
            constant_z_slice_polygons_handler_option=config.constant_z_slice_polygons_handler_option,
            remove_consecutive_duplicate_points_in_polygons=(
                config.remove_consecutive_duplicate_points_in_polygons
            ),
            include_edges_in_log_files=config.include_edges_in_log_files,
            custom_cuda_kernel_type=config.custom_cuda_kernel_type,
            demonstrate_structure_dimension_calculation_correctness_bool_1=(
                config.demonstrate_structure_dimension_calculation_correctness_bool_1
            ),
            demonstrate_structure_dimension_calculation_correctness_bool_1_old=(
                config.demonstrate_structure_dimension_calculation_correctness_bool_1_old
            ),
            other_pcds_to_plot_list=[interpolated_pcd_dict["Full with end caps"]],
        )

    (
        structure_dimension_at_centroid_dict,
        voxel_size_for_structure_dimension_calc,
        live_display,
    ) = run_phase(
        "dimensions",
        "[cyan]~~Calculating structure dimensions",
        compute_structure_dimensions,
    )

    def compute_triangle_mesh():
        return misc_tools.compute_structure_triangle_mesh(
            config.interp_inter_slice_dist,
            config.interp_intra_slice_dist,
            threeDdata_array_fully_interpolated_with_end_caps,
            config.radius_for_normals_estimation,
            config.max_nn_for_normals_estimation,
        )

    (
        fully_interp_with_end_caps_structure_triangle_mesh,
        water_tight_bool,
    ) = run_phase(
        "triangle_mesh",
        "[cyan]~~Calculating structure triangle mesh",
        compute_triangle_mesh,
        refresh_live_display=True,
    )

    if water_tight_bool is False:
        important_info.add_text_line(
            (
                f"WARNING! Patient: {patient_uid}, Structure: {structure_id}, "
                f"({struct_ref_type}) is not water tight! Surface area may be inaccurate!"
            ),
            live_display,
        )

    def compute_structure_surface_area():
        if config.display_structure_surface_mesh_bool:
            o3d.visualization.draw_geometries(
                [fully_interp_with_end_caps_structure_triangle_mesh],
                mesh_show_back_face=True,
            )
        return misc_tools.compute_surface_area(
            fully_interp_with_end_caps_structure_triangle_mesh
        )

    structure_fully_interp_with_end_caps_surface_area = run_phase(
        "surface_area",
        "[cyan]~~Calculating surface area",
        compute_structure_surface_area,
    )

    def compute_shape_features():
        surface_volume_ratio = (
            structure_fully_interp_with_end_caps_surface_area / structure_volume
        )
        sphericity = misc_tools.calculate_sphericity(
            structure_volume,
            structure_fully_interp_with_end_caps_surface_area,
        )
        compactness_1 = misc_tools.calculate_compactness_1(
            structure_volume,
            structure_fully_interp_with_end_caps_surface_area,
        )
        compactness_2 = misc_tools.calculate_compactness_2(
            structure_volume,
            structure_fully_interp_with_end_caps_surface_area,
        )
        spherical_disproportion = misc_tools.spherical_disproportion(
            structure_volume,
            structure_fully_interp_with_end_caps_surface_area,
        )
        maximum_3D_diameter = maximum_distance
        si_arclength = misc_tools.compute_arc_length_from_centroids(
            specific_structure["Structure centroid pts"]
        )
        pca_lengths_of_structure_dict, pca_eigenvectors_of_structure_arr = misc_tools.pca_lengths(
            binary_mask_arr
        )
        equivalent_ellipse_dimensions = {
            "Major axis": 4 * math.sqrt(pca_lengths_of_structure_dict["Major"]),
            "Minor axis": 4 * math.sqrt(pca_lengths_of_structure_dict["Minor"]),
            "Least axis": 4 * math.sqrt(pca_lengths_of_structure_dict["Least"]),
        }
        if config.show_equivalent_ellipsoid_from_pca_bool:
            axis_diameters = list(equivalent_ellipse_dimensions.values())
            misc_tools.draw_oriented_ellipse_point_cloud(
                threeDdata_array_fully_interpolated_with_end_caps,
                axis_diameters,
                pca_eigenvectors_of_structure_arr,
            )
        elongation = math.sqrt(
            pca_lengths_of_structure_dict["Minor"]
            / pca_lengths_of_structure_dict["Major"]
        )
        flatness = math.sqrt(
            pca_lengths_of_structure_dict["Least"]
            / pca_lengths_of_structure_dict["Major"]
        )
        shape_features_3d_dictionary = {
            "Patient ID": [patient_uid],
            "Structure ID": [structure_id],
            "Structure index": [specific_structure_index],
            "Structure type": [struct_ref_type],
            "Structure refnum": [structure_reference_number],
            "Volume": [structure_volume],
            "Surface area": [structure_fully_interp_with_end_caps_surface_area],
            "Surface area to volume ratio": [surface_volume_ratio],
            "Sphericity": [sphericity],
            "Compactness 1": [compactness_1],
            "Compactness 2": [compactness_2],
            "Spherical disproportion": [spherical_disproportion],
            "Maximum 3D diameter": [maximum_3D_diameter],
            "PCA major": [pca_lengths_of_structure_dict["Major"]],
            "PCA minor": [pca_lengths_of_structure_dict["Minor"]],
            "PCA least": [pca_lengths_of_structure_dict["Least"]],
            "PCA eigenvector major": [tuple(pca_eigenvectors_of_structure_arr[0, :])],
            "PCA eigenvector minor": [tuple(pca_eigenvectors_of_structure_arr[1, :])],
            "PCA eigenvector least": [tuple(pca_eigenvectors_of_structure_arr[2, :])],
            "Major axis (equivalent ellipse)": [equivalent_ellipse_dimensions["Major axis"]],
            "Minor axis (equivalent ellipse)": [equivalent_ellipse_dimensions["Minor axis"]],
            "Least axis (equivalent ellipse)": [equivalent_ellipse_dimensions["Least axis"]],
            "Elongation": [elongation],
            "Flatness": [flatness],
            "L/R dimension at centroid": structure_dimension_at_centroid_dict[
                "X dimension length at centroid"
            ],
            "A/P dimension at centroid": structure_dimension_at_centroid_dict[
                "Y dimension length at centroid"
            ],
            "S/I dimension at centroid": structure_dimension_at_centroid_dict[
                "Z dimension length at centroid"
            ],
            "S/I arclength": [si_arclength],
        }
        if struct_ref_type == config.dil_ref:
            shape_features_3d_dictionary.update(
                _build_dil_shape_feature_overrides(
                    pydicom_item=pydicom_item,
                    specific_structure=specific_structure,
                    sp_patient_selected_structure_info_dataframe=(
                        sp_patient_selected_structure_info_dataframe
                    ),
                    config=config,
                )
            )
        shape_features_dataframe = pandas.DataFrame(shape_features_3d_dictionary)
        shape_features_dataframe = dataframe_builders.convert_columns_to_categorical_and_downcast(
            shape_features_dataframe,
            threshold=0.25,
        )
        return shape_features_dataframe, maximum_3D_diameter

    shape_features_dataframe, maximum_3D_diameter = run_phase(
        "shape_features",
        "[cyan]~~Calculating structure shape features",
        compute_shape_features,
    )

    def store_structure_outputs():
        specific_structure["Raw contour pts"] = threeDdata_array
        specific_structure["Equal num zslice contour pts"] = threeDdata_equal_pt_zslice_list
        specific_structure["Inter-slice interpolation information"] = interslice_interpolation_information
        specific_structure["Intra-slice interpolation information"] = interpolation_information
        specific_structure["Maximum pairwise distance"] = maximum_3D_diameter
        specific_structure["Structure volume"] = structure_volume
        specific_structure["Voxel size for structure volume calc"] = voxel_size_for_structure_volume_calc
        specific_structure["Structure dimension at centroid dict"] = structure_dimension_at_centroid_dict
        specific_structure["Voxel size for structure dimension calc"] = voxel_size_for_structure_dimension_calc
        specific_structure["Structure surface area"] = structure_fully_interp_with_end_caps_surface_area
        specific_structure["Structure features dataframe"] = shape_features_dataframe
        specific_structure["Point cloud raw"] = threeDdata_point_cloud
        specific_structure["Interpolated structure point cloud dict"] = interpolated_pcd_dict
        specific_structure["Structure OPEN3D triangle mesh object"] = (
            fully_interp_with_end_caps_structure_triangle_mesh
        )
        return None

    run_phase(
        "store_outputs",
        "[cyan]~~Storing structure outputs",
        store_structure_outputs,
    )

    structure_total_elapsed_seconds = time.perf_counter() - structure_start_time
    phase_timings["total_elapsed_seconds"] = structure_total_elapsed_seconds
    specific_structure["Structure preprocessing phase timing dict"] = {
        key: round(float(value), 6)
        for key, value in phase_timings.items()
    }
    specific_structure["Structure preprocessing elapsed seconds"] = round(
        float(structure_total_elapsed_seconds),
        6,
    )

    timing_row_dict = {
        "Patient ID": [patient_uid],
        "Structure ID": [structure_id],
        "Structure index": [specific_structure_index],
        "Structure type": [struct_ref_type],
        "Structure refnum": [structure_reference_number],
        "MR ADC enabled": [mr_adc_enabled],
        "Mesh watertight": [water_tight_bool],
        "Raw contour point count": [int(len(threeDdata_array))],
        "Interpolated point count": [
            int(len(interpolation_information.interpolated_pts_np_arr))
        ],
        "Interpolated point count with end caps": [
            int(len(interpolation_information.interpolated_pts_with_end_caps_np_arr))
        ],
    }
    for timing_key, timing_value in phase_timings.items():
        timing_row_dict[timing_key] = [timing_value]
    timing_dataframe = pandas.DataFrame(timing_row_dict)
    _append_or_initialize_dataframe(
        patient_output_dataframes_dict,
        STRUCTURE_PREPROCESSING_TIMINGS_DF_KEY,
        timing_dataframe,
    )

    _log_phase_timing(
        runtime_logger,
        "preprocessing.structure.total",
        structure_total_elapsed_seconds,
        patient_uid=patient_uid,
        structure_id=structure_id,
        structure_ref_type=struct_ref_type,
        structure_index=specific_structure_index,
        details={
            "raw_contour_point_count": int(len(threeDdata_array)),
            "interpolated_point_count": int(len(interpolation_information.interpolated_pts_np_arr)),
            "interpolated_with_end_caps_point_count": int(
                len(interpolation_information.interpolated_pts_with_end_caps_np_arr)
            ),
            "mesh_watertight": bool(water_tight_bool),
        },
    )
    if runtime_logger is not None:
        runtime_logger.phase_end(
            "preprocessing.structure",
            "Completed non-biopsy structure preprocessing.",
            patient_uid=patient_uid,
            structure_id=structure_id,
            structure_ref_type=struct_ref_type,
            structure_index=specific_structure_index,
            details={
                "total_elapsed_seconds": round(float(structure_total_elapsed_seconds), 6),
                "mesh_watertight": bool(water_tight_bool),
            },
            clear_phase=True,
        )

    return live_display