import math

import numpy as np

import anatomy_reconstructor_tools
import biopsy_creator
import biopsy_transporter
import centroid_finder
import math_funcs as mf
import misc_tools
import pca
import point_containment_tools
from preprocessing.interpolation.interpolation import interpolation_information_obj
from preprocessing.biopsy_processing.biopsy_geometry_helper import finalize_biopsy_geometry_from_zslice_list
from preprocessing.biopsy_processing.simulated_biopsy_planner import build_simulated_biopsy_planning_state
from preprocessing.biopsy_processing.simulated_biopsy_planner import get_planned_simulated_biopsy_zslice_list


def _transport_planned_simulated_biopsy(pydicom_item,
                                        specific_structure,
                                        planned_threeDdata_zslice_list,
                                        centroid_dil_sim_key,
                                        optimal_dil_sim_key
                                        ):
    simulated_type = specific_structure["Simulated type"]

    if simulated_type == centroid_dil_sim_key:
        return biopsy_transporter.biopsy_transporter_centroid(
            pydicom_item,
            specific_structure,
            planned_threeDdata_zslice_list,
        )

    if simulated_type == optimal_dil_sim_key:
        return biopsy_transporter.biopsy_transporter_optimal(
            pydicom_item,
            specific_structure,
            planned_threeDdata_zslice_list,
        )

    return planned_threeDdata_zslice_list


def _finalize_simulated_biopsy_geometry(master_structure_reference_dict,
                                        patientUID,
                                        bx_ref,
                                        specific_structure_index,
                                        specific_structure,
                                        threeDdata_zslice_list,
                                        structs_referenced_dict,
                                        parallel_pool,
                                        interp_inter_slice_dist,
                                        interp_intra_slice_dist,
                                        interp_dist_caps,
                                        biopsy_radius,
                                        voxel_size_for_structure_volume_calc_non_bx,
                                        factor_for_voxel_size,
                                        cupy_array_upper_limit_NxN_size_input,
                                        layout_groups,
                                        nearest_zslice_vals_and_indices_cupy_generic_max_size,
                                        generate_cuda_log_files_volume_calculation,
                                        constant_z_slice_polygons_handler_option,
                                        remove_consecutive_duplicate_points_in_polygons,
                                        include_edges_in_log_files,
                                        custom_cuda_kernel_type,
                                        demonstrate_volume_calculation_correctness_bool_1,
                                        plot_volume_calculation_containment_result_bool_1_old,
                                        plot_binary_mask_bool,
                                        structures_progress,
                                        indeterminate_progress_sub,
                                        live_display
                                        ):
    return finalize_biopsy_geometry_from_zslice_list(master_structure_reference_dict,
                                                     patientUID,
                                                     bx_ref,
                                                     specific_structure_index,
                                                     specific_structure,
                                                     threeDdata_zslice_list,
                                                     structs_referenced_dict,
                                                     parallel_pool,
                                                     interp_inter_slice_dist,
                                                     interp_intra_slice_dist,
                                                     interp_dist_caps,
                                                     biopsy_radius,
                                                     voxel_size_for_structure_volume_calc_non_bx,
                                                     factor_for_voxel_size,
                                                     cupy_array_upper_limit_NxN_size_input,
                                                     layout_groups,
                                                     nearest_zslice_vals_and_indices_cupy_generic_max_size,
                                                     generate_cuda_log_files_volume_calculation,
                                                     constant_z_slice_polygons_handler_option,
                                                     remove_consecutive_duplicate_points_in_polygons,
                                                     include_edges_in_log_files,
                                                     custom_cuda_kernel_type,
                                                     demonstrate_volume_calculation_correctness_bool_1,
                                                     plot_volume_calculation_containment_result_bool_1_old,
                                                     plot_binary_mask_bool,
                                                     structures_progress,
                                                     indeterminate_progress_sub,
                                                     live_display,
                                                     display_pca_fit_variation_for_biopsies_bool=False,
                                                     store_raw_contour_pts_zslice_list_bool=True)


def simulated_biopsy_processer(master_structure_reference_dict,
                               master_structure_info_dict,
                               structs_referenced_dict,
                               bx_ref,
                               centroid_dil_sim_key,
                               optimal_dil_sim_key,
                               parallel_pool,
                               interp_inter_slice_dist,
                               interp_intra_slice_dist,
                               interp_dist_caps,
                               biopsy_radius,
                               voxel_size_for_structure_volume_calc_non_bx,
                               factor_for_voxel_size,
                               cupy_array_upper_limit_NxN_size_input,
                               layout_groups,
                               nearest_zslice_vals_and_indices_cupy_generic_max_size,
                               generate_cuda_log_files_volume_calculation,
                               constant_z_slice_polygons_handler_option,
                               remove_consecutive_duplicate_points_in_polygons,
                               include_edges_in_log_files,
                               custom_cuda_kernel_type,
                               demonstrate_volume_calculation_correctness_bool_1,
                               plot_volume_calculation_containment_result_bool_1_old,
                               plot_binary_mask_bool,
                               patients_progress,
                               structures_progress,
                               completed_progress,
                               indeterminate_progress_sub,
                               live_display,
                               centroid_line_vec_sim_list,
                               centroid_first_pos_sim_list,
                               num_centroids_for_sim_bxs,
                               simulated_bx_rad,
                               plot_simulated_cores_immediately
                               ):

    patientUID_default = "Initializing"
    processing_patients_task_main_description = "[red]Processing patient sim-bx data [{}]...".format(patientUID_default)
    processing_patients_task_completed_main_description = "[green]Processing patient sim-bx data"
    processing_patients_task = patients_progress.add_task(processing_patients_task_main_description, total=master_structure_info_dict["Global"]["Num cases"])
    processing_patients_task_completed = completed_progress.add_task(processing_patients_task_completed_main_description, total=master_structure_info_dict["Global"]["Num cases"], visible=False)

    for patientUID, pydicom_item in master_structure_reference_dict.items():
        processing_patients_task_main_description = "[red]Processing patient sim-bx data [{}]...".format(patientUID)
        patients_progress.update(processing_patients_task, description=processing_patients_task_main_description)

        structureID_default = "Initializing"
        num_sim_bx_structs_patient_specific = master_structure_info_dict["By patient"][patientUID][bx_ref]["Num sim structs"]
        processing_structures_task_main_description = "[cyan]Processing structures [{},{}]...".format(patientUID, structureID_default)
        processing_structures_task = structures_progress.add_task(processing_structures_task_main_description, total=num_sim_bx_structs_patient_specific)

        for specific_structure_index, specific_structure in enumerate(pydicom_item[bx_ref]):
            structureID = specific_structure["ROI"]
            simulated_bool = specific_structure["Simulated bool"]
            sim_type = specific_structure["Simulated type"]

            if simulated_bool == False:
                continue

            processing_structures_task_main_description = "[cyan]Processing structures [{},{}]...".format(patientUID, structureID)
            structures_progress.update(processing_structures_task, description=processing_structures_task_main_description)

            build_simulated_biopsy_planning_state(
                specific_structure,
                centroid_line_vec_sim_list,
                centroid_first_pos_sim_list,
                num_centroids_for_sim_bxs,
                simulated_bx_rad,
                plot_simulated_cores_immediately,
            )
            threeDdata_zslice_list = get_planned_simulated_biopsy_zslice_list(specific_structure)
            threeDdata_zslice_list = _transport_planned_simulated_biopsy(
                pydicom_item,
                specific_structure,
                threeDdata_zslice_list,
                centroid_dil_sim_key,
                optimal_dil_sim_key,
            )
            live_display = _finalize_simulated_biopsy_geometry(master_structure_reference_dict,
                                                               patientUID,
                                                               bx_ref,
                                                               specific_structure_index,
                                                               specific_structure,
                                                               threeDdata_zslice_list,
                                                               structs_referenced_dict,
                                                               parallel_pool,
                                                               interp_inter_slice_dist,
                                                               interp_intra_slice_dist,
                                                               interp_dist_caps,
                                                               biopsy_radius,
                                                               voxel_size_for_structure_volume_calc_non_bx,
                                                               factor_for_voxel_size,
                                                               cupy_array_upper_limit_NxN_size_input,
                                                               layout_groups,
                                                               nearest_zslice_vals_and_indices_cupy_generic_max_size,
                                                               generate_cuda_log_files_volume_calculation,
                                                               constant_z_slice_polygons_handler_option,
                                                               remove_consecutive_duplicate_points_in_polygons,
                                                               include_edges_in_log_files,
                                                               custom_cuda_kernel_type,
                                                               demonstrate_volume_calculation_correctness_bool_1,
                                                               plot_volume_calculation_containment_result_bool_1_old,
                                                               plot_binary_mask_bool,
                                                               structures_progress,
                                                               indeterminate_progress_sub,
                                                               live_display)

            if False:
                total_structure_points = sum([np.shape(x)[0] for x in threeDdata_zslice_list])
                threeDdata_array = np.empty([total_structure_points, 3])

                structure_centroids_array = np.empty([len(threeDdata_zslice_list), 3])

                lower_bound_index = 0
                for index, threeDdata_zslice in enumerate(threeDdata_zslice_list):
                    current_zslice_num_points = np.size(threeDdata_zslice, 0)
                    threeDdata_array[lower_bound_index:lower_bound_index + current_zslice_num_points] = threeDdata_zslice
                    lower_bound_index = lower_bound_index + current_zslice_num_points

                    structure_zslice_centroid = np.mean(threeDdata_zslice, axis=0)
                    structure_centroids_array[index] = structure_zslice_centroid

                structure_global_centroid = centroid_finder.centeroidfinder_numpy_3D(structure_centroids_array)

                interslice_interpolation_information, threeDdata_equal_pt_zslice_list = anatomy_reconstructor_tools.inter_zslice_interpolator(
                    parallel_pool,
                    threeDdata_zslice_list,
                    interp_inter_slice_dist,
                )

                threeDdata_to_intra_zslice_interpolate_zslice_list = interslice_interpolation_information.interpolated_pts_list
                num_z_slices_data_to_intra_slice_interpolate = len(threeDdata_to_intra_zslice_interpolate_zslice_list)

                interpolation_information = interpolation_information_obj(num_z_slices_data_to_intra_slice_interpolate)
                interpolation_information.serial_analyze(threeDdata_to_intra_zslice_interpolate_zslice_list, interp_intra_slice_dist)

                first_zslice = threeDdata_to_intra_zslice_interpolate_zslice_list[0]
                last_zslice = threeDdata_to_intra_zslice_interpolate_zslice_list[-1]
                interpolation_information.create_fill_new_v2(first_zslice, interp_dist_caps, kernel_type=custom_cuda_kernel_type)
                interpolation_information.create_fill_new_v2(last_zslice, interp_dist_caps, kernel_type=custom_cuda_kernel_type)

                pcd_color = structs_referenced_dict[bx_ref]["PCD color dict"][sim_type]
                threeDdata_point_cloud = point_containment_tools.create_point_cloud(threeDdata_array, pcd_color)

                threeDdata_array_fully_interpolated = interpolation_information.interpolated_pts_np_arr
                threeDdata_array_fully_interpolated_with_end_caps = interpolation_information.interpolated_pts_with_end_caps_np_arr
                threeDdata_array_interslice_interpolation = np.vstack(interslice_interpolation_information.interpolated_pts_list)
                interslice_interp_pcd = point_containment_tools.create_point_cloud(threeDdata_array_interslice_interpolation, pcd_color)
                inter_and_intra_interp_pcd = point_containment_tools.create_point_cloud(threeDdata_array_fully_interpolated, pcd_color)
                inter_and_intra_and_end_caps_interp_pcd = point_containment_tools.create_point_cloud(threeDdata_array_fully_interpolated_with_end_caps, pcd_color)
                interpolated_pcd_dict = {"Interslice": interslice_interp_pcd, "Full": inter_and_intra_interp_pcd, "Full with end caps": inter_and_intra_and_end_caps_interp_pcd}

                centroid_line = pca.linear_fitter(structure_centroids_array.T)
                centroid_line_length = np.linalg.norm(centroid_line[0, :] - centroid_line[1, :])
                slice_reconstruction_max_distance = 0.1
                num_centroid_samples_of_centroid_line = int(math.ceil(centroid_line_length / slice_reconstruction_max_distance))
                centroid_line_sample = np.empty((num_centroid_samples_of_centroid_line, 3), dtype=float)
                centroid_line_sample[0, :] = centroid_line[0, :]
                travel_vec = np.array([centroid_line[1] - centroid_line[0]]) * 1 / num_centroid_samples_of_centroid_line
                for i in range(1, num_centroid_samples_of_centroid_line):
                    init_point = centroid_line_sample[-1]
                    new_point = init_point + travel_vec
                    centroid_line_sample[i] = new_point

                line_start = centroid_line[0, :]
                line_end = centroid_line[1, :]
                variation_distance_arr = np.empty(structure_centroids_array.shape[0])
                for index, point in enumerate(structure_centroids_array):
                    distance, _ = biopsy_creator.point_to_line_segment_distance(point, line_start, line_end)
                    variation_distance_arr[index] = distance
                mean_variation = np.mean(variation_distance_arr)

                maximum_2d_distance_between_centroids = biopsy_creator.distance_of_most_distant_points_2d_projection(structure_centroids_array, travel_vec)

                list_travel_vec = np.squeeze(travel_vec).tolist()
                list_centroid_line_first_point = np.squeeze(centroid_line_sample[0]).tolist()
                biopsy_reconstructed_cyl_z_length_from_contour_data = centroid_line_length
                drawn_biopsy_array_transpose = biopsy_creator.biopsy_points_creater_by_transport(
                    list_travel_vec,
                    list_centroid_line_first_point,
                    num_centroid_samples_of_centroid_line,
                    np.linalg.norm(travel_vec),
                    biopsy_radius,
                    False,
                )
                drawn_biopsy_array = drawn_biopsy_array_transpose.T
                reconstructed_biopsy_point_cloud = point_containment_tools.create_point_cloud(drawn_biopsy_array, pcd_color)
                reconstructed_bx_delaunay_global_convex_structure_obj = point_containment_tools.delaunay_obj(drawn_biopsy_array, pcd_color)
                reconstructed_bx_delaunay_global_convex_structure_obj.generate_lineset()

                vec_with_largest_z_val_index = centroid_line[:, 2].argmax()
                vec_with_largest_z_val = centroid_line[vec_with_largest_z_val_index, :]
                base_sup_vec_bx_centroid_arr = vec_with_largest_z_val

                vec_with_smallest_z_val_index = centroid_line[:, 2].argmin()
                vec_with_smallest_z_val = centroid_line[vec_with_smallest_z_val_index, :]
                apex_inf_vec_bx_centroid_arr = vec_with_smallest_z_val

                apex_to_base_bx_best_fit_vec = base_sup_vec_bx_centroid_arr - apex_inf_vec_bx_centroid_arr
                apex_to_base_bx_best_fit_vec_length = np.linalg.norm(apex_to_base_bx_best_fit_vec)
                apex_to_base_bx_best_fit_unit_vec = apex_to_base_bx_best_fit_vec / apex_to_base_bx_best_fit_vec_length

                structure_info = misc_tools.specific_structure_info_dict_creator('given', specific_structure=specific_structure)
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Calculating structure volume", total=None)

                z_axis_np_vec = np.array([0, 0, 1], dtype=float)
                centroid_line_to_z_axis_rotation_matrix_other = mf.rotation_matrix_from_vectors(apex_to_base_bx_best_fit_vec, z_axis_np_vec)
                rotated_reconstructed_bx_arr = (centroid_line_to_z_axis_rotation_matrix_other @ drawn_biopsy_array.T).T
                rotated_reconstructed_bx_arr_rounded = np.copy(rotated_reconstructed_bx_arr)

                distance_between_rings = np.linalg.norm(travel_vec)
                sci_not_dist_bet_rings = '%e' % distance_between_rings
                num_zeros_before_first_dig_after_decimal = int(sci_not_dist_bet_rings.partition('-')[2]) - 1
                num_decimals_for_rounding = num_zeros_before_first_dig_after_decimal + 2
                rotated_reconstructed_bx_arr_rounded[:, 2] = np.round(rotated_reconstructed_bx_arr[:, 2], decimals=num_decimals_for_rounding)

                zvals_list = np.unique(rotated_reconstructed_bx_arr_rounded[:, 2]).tolist()
                zslices_list = [rotated_reconstructed_bx_arr_rounded[rotated_reconstructed_bx_arr_rounded[:, 2] == z_val] for z_val in zvals_list]

                structure_volume, maximum_distance, voxel_size_for_structure_volume_calc, binary_mask_arr, live_display = misc_tools.structure_volume_calculator(
                    rotated_reconstructed_bx_arr_rounded,
                    zvals_list,
                    zslices_list,
                    structure_info,
                    patientUID,
                    voxel_size_for_structure_volume_calc_non_bx,
                    factor_for_voxel_size,
                    cupy_array_upper_limit_NxN_size_input,
                    layout_groups,
                    nearest_zslice_vals_and_indices_cupy_generic_max_size,
                    structures_progress,
                    live_display,
                    generate_cuda_log_files_volume_calculation=generate_cuda_log_files_volume_calculation,
                    constant_z_slice_polygons_handler_option=constant_z_slice_polygons_handler_option,
                    remove_consecutive_duplicate_points_in_polygons=remove_consecutive_duplicate_points_in_polygons,
                    include_edges_in_log_files=include_edges_in_log_files,
                    custom_cuda_kernel_type=custom_cuda_kernel_type,
                    demonstrate_volume_calculation_correctness_bool_1=demonstrate_volume_calculation_correctness_bool_1,
                    plot_volume_calculation_containment_result_bool_1_old=plot_volume_calculation_containment_result_bool_1_old,
                    plot_binary_mask_bool=plot_binary_mask_bool,
                    other_pcds_to_plot_list=[interpolated_pcd_dict['Full with end caps']],
                )

                indeterminate_progress_sub.update(indeterminate_task, visible=False)

                master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Raw contour pts zslice list"] = threeDdata_zslice_list
                master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Raw contour pts"] = threeDdata_array
                master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Equal num zslice contour pts"] = threeDdata_equal_pt_zslice_list
                master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Inter-slice interpolation information"] = interslice_interpolation_information
                master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Intra-slice interpolation information"] = interpolation_information
                master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Maximum pairwise distance"] = maximum_distance
                master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Structure volume"] = structure_volume
                master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Voxel size for structure volume calc"] = voxel_size_for_structure_volume_calc
                master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Point cloud raw"] = threeDdata_point_cloud
                master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Interpolated structure point cloud dict"] = interpolated_pcd_dict
                master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Centroid variation arr"] = variation_distance_arr
                master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Mean centroid variation"] = mean_variation
                master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Maximum projected distance between original centroids"] = maximum_2d_distance_between_centroids
                master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Structure centroid pts"] = structure_centroids_array
                master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Reconstructed biopsy cylinder length (from contour data)"] = biopsy_reconstructed_cyl_z_length_from_contour_data
                master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Best fit line of centroid pts"] = centroid_line
                master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Centroid line unit vec (bx needle base to bx needle tip)"] = apex_to_base_bx_best_fit_unit_vec
                master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Centroid line vec (bx needle base to bx needle tip)"] = apex_to_base_bx_best_fit_vec
                master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Centroid line vec length (bx needle base to bx needle tip)"] = apex_to_base_bx_best_fit_vec_length
                master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Centroid line sample pts"] = centroid_line_sample
                master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Reconstructed structure pts arr"] = drawn_biopsy_array
                master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Reconstructed structure point cloud"] = reconstructed_biopsy_point_cloud
                master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Reconstructed structure delaunay global"] = reconstructed_bx_delaunay_global_convex_structure_obj
                master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Structure global centroid"] = structure_global_centroid

            structures_progress.update(processing_structures_task, advance=1)

        structures_progress.remove_task(processing_structures_task)
        patients_progress.update(processing_patients_task, advance=1)
        completed_progress.update(processing_patients_task_completed, advance=1)

    patients_progress.update(processing_patients_task, visible=False)
    completed_progress.update(processing_patients_task_completed, visible=True)

    return live_display