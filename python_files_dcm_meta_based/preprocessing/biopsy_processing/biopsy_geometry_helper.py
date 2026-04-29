import math

import numpy as np
import open3d as o3d

import biopsy_creator
import centroid_finder
import math_funcs as mf
import misc_tools
import pca
import plotting_funcs
import point_containment_tools


def build_reconstructed_biopsy_model_for_sampling_from_zslice_list(threeDdata_zslice_list,
                                                                   biopsy_radius,
                                                                   pcd_color=None
                                                                   ):
    if pcd_color is None:
        pcd_color = np.array([0, 0, 0], dtype=float)

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

    centroid_line = pca.linear_fitter(structure_centroids_array.T)
    centroid_line_length = np.linalg.norm(centroid_line[0, :] - centroid_line[1, :])
    slice_reconstruction_max_distance = 0.1
    num_centroid_samples_of_centroid_line = int(math.ceil(centroid_line_length / slice_reconstruction_max_distance))
    centroid_line_sample = np.empty((num_centroid_samples_of_centroid_line, 3), dtype=float)
    centroid_line_sample[0, :] = centroid_line[0, :]
    travel_vec = np.array([centroid_line[1] - centroid_line[0]]) * 1 / num_centroid_samples_of_centroid_line
    for index in range(1, num_centroid_samples_of_centroid_line):
        init_point = centroid_line_sample[-1]
        new_point = init_point + travel_vec
        centroid_line_sample[index] = new_point

    line_start = centroid_line[0, :]
    line_end = centroid_line[1, :]
    variation_distance_arr = np.empty(structure_centroids_array.shape[0])
    for index, point in enumerate(structure_centroids_array):
        distance, _closest_point_on_line = biopsy_creator.point_to_line_segment_distance(point, line_start, line_end)
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

    translation_vec_bx_coord_sys_origin = -apex_inf_vec_bx_centroid_arr
    apex_to_base_bx_best_fit_vec = base_sup_vec_bx_centroid_arr - apex_inf_vec_bx_centroid_arr
    apex_to_base_bx_best_fit_vec_length = np.linalg.norm(apex_to_base_bx_best_fit_vec)
    apex_to_base_bx_best_fit_unit_vec = apex_to_base_bx_best_fit_vec / apex_to_base_bx_best_fit_vec_length

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

    return {
        "Raw contour pts": threeDdata_array,
        "Structure centroid pts": structure_centroids_array,
        "Structure global centroid": structure_global_centroid,
        "Centroid variation arr": variation_distance_arr,
        "Mean centroid variation": mean_variation,
        "Maximum projected distance between original centroids": maximum_2d_distance_between_centroids,
        "Reconstructed biopsy cylinder length (from contour data)": biopsy_reconstructed_cyl_z_length_from_contour_data,
        "Best fit line of centroid pts": centroid_line,
        "Centroid line unit vec (bx needle base to bx needle tip)": apex_to_base_bx_best_fit_unit_vec,
        "Centroid line vec (bx needle base to bx needle tip)": apex_to_base_bx_best_fit_vec,
        "Centroid line vec length (bx needle base to bx needle tip)": apex_to_base_bx_best_fit_vec_length,
        "Centroid line sample pts": centroid_line_sample,
        "Reconstructed structure pts arr": drawn_biopsy_array,
        "Reconstructed structure point cloud": reconstructed_biopsy_point_cloud,
        "Reconstructed structure delaunay global": reconstructed_bx_delaunay_global_convex_structure_obj,
        "Distance between centroid sample rings": distance_between_rings,
        "Rotated reconstructed structure pts arr rounded": rotated_reconstructed_bx_arr_rounded,
        "Rotated reconstructed structure z values": zvals_list,
        "Rotated reconstructed structure zslice list": zslices_list,
        "Biopsy coord sys origin translation vec": translation_vec_bx_coord_sys_origin,
        "Centroid line to z axis rotation matrix": centroid_line_to_z_axis_rotation_matrix_other,
    }


def finalize_biopsy_geometry_from_zslice_list(master_structure_reference_dict,
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
                                              store_raw_contour_pts_zslice_list_bool=False
                                              ):
    import anatomy_reconstructor_tools
    from preprocessing.interpolation.interpolation import interpolation_information_obj

    structureID = specific_structure["ROI"]
    sim_type = specific_structure["Simulated type"]
    pcd_color = structs_referenced_dict[bx_ref]['PCD color dict'][sim_type]
    reconstructed_biopsy_model_dict = build_reconstructed_biopsy_model_for_sampling_from_zslice_list(
        threeDdata_zslice_list,
        biopsy_radius,
        pcd_color,
    )

    threeDdata_array = reconstructed_biopsy_model_dict["Raw contour pts"]
    structure_centroids_array = reconstructed_biopsy_model_dict["Structure centroid pts"]
    structure_global_centroid = reconstructed_biopsy_model_dict["Structure global centroid"]

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

    threeDdata_point_cloud = point_containment_tools.create_point_cloud(threeDdata_array, pcd_color)

    threeDdata_array_fully_interpolated = interpolation_information.interpolated_pts_np_arr
    threeDdata_array_fully_interpolated_with_end_caps = interpolation_information.interpolated_pts_with_end_caps_np_arr
    threeDdata_array_interslice_interpolation = np.vstack(interslice_interpolation_information.interpolated_pts_list)
    interslice_interp_pcd = point_containment_tools.create_point_cloud(threeDdata_array_interslice_interpolation, pcd_color)
    inter_and_intra_interp_pcd = point_containment_tools.create_point_cloud(threeDdata_array_fully_interpolated, pcd_color)
    inter_and_intra_and_end_caps_interp_pcd = point_containment_tools.create_point_cloud(threeDdata_array_fully_interpolated_with_end_caps, pcd_color)
    interpolated_pcd_dict = {"Interslice": interslice_interp_pcd, "Full": inter_and_intra_interp_pcd, "Full with end caps": inter_and_intra_and_end_caps_interp_pcd}

    centroid_line = reconstructed_biopsy_model_dict["Best fit line of centroid pts"]
    variation_distance_arr = reconstructed_biopsy_model_dict["Centroid variation arr"]
    mean_variation = reconstructed_biopsy_model_dict["Mean centroid variation"]
    maximum_2d_distance_between_centroids = reconstructed_biopsy_model_dict["Maximum projected distance between original centroids"]
    centroid_line_sample = reconstructed_biopsy_model_dict["Centroid line sample pts"]
    closest_point_on_line_arr = None
    if display_pca_fit_variation_for_biopsies_bool == True:
        line_start = centroid_line[0, :]
        line_end = centroid_line[1, :]
        closest_point_on_line_arr = np.empty_like(structure_centroids_array)

        for index, point in enumerate(structure_centroids_array):
            _distance, closest_point_on_line = biopsy_creator.point_to_line_segment_distance(point, line_start, line_end)
            closest_point_on_line_arr[index] = closest_point_on_line

    if display_pca_fit_variation_for_biopsies_bool == True:
        live_display.stop()

        biopsy_centroids_pcd = point_containment_tools.create_point_cloud(structure_centroids_array, np.array([0, 0, 1]))

        line_color = np.array([0, 1, 1])
        centroid_line_set = o3d.geometry.LineSet()
        centroid_line_set.points = o3d.utility.Vector3dVector(centroid_line)
        centroid_line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
        centroid_line_set.paint_uniform_color(line_color)

        nearest_points_line_color = np.array([1, 0, 1])
        nearest_points_pca_to_centroids_line_set = o3d.geometry.LineSet()
        nearest_points_pca_to_centroids_line_set.points = o3d.utility.Vector3dVector(np.vstack((structure_centroids_array, closest_point_on_line_arr)))
        num_centroids_temp = structure_centroids_array.shape[0]
        lines_connections = [[index, index + num_centroids_temp] for index in range(0, num_centroids_temp)]
        nearest_points_pca_to_centroids_line_set.lines = o3d.utility.Vector2iVector(lines_connections)
        nearest_points_pca_to_centroids_line_set.paint_uniform_color(nearest_points_line_color)

        print(f"Pt: {patientUID}, Bx: {structureID}, Mean variation: {mean_variation}, Max dinstance between centroids: {maximum_2d_distance_between_centroids}")

        plotting_funcs.plot_geometries(biopsy_centroids_pcd, centroid_line_set, nearest_points_pca_to_centroids_line_set)

        del biopsy_centroids_pcd, centroid_line_set, nearest_points_pca_to_centroids_line_set, nearest_points_line_color, line_color, lines_connections, num_centroids_temp
        live_display.start()

    structure_info = misc_tools.specific_structure_info_dict_creator('given', specific_structure=specific_structure)
    indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Calculating structure volume", total=None)

    rotated_reconstructed_bx_arr_rounded = reconstructed_biopsy_model_dict["Rotated reconstructed structure pts arr rounded"]
    zvals_list = reconstructed_biopsy_model_dict["Rotated reconstructed structure z values"]
    zslices_list = reconstructed_biopsy_model_dict["Rotated reconstructed structure zslice list"]

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

    if store_raw_contour_pts_zslice_list_bool == True:
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
    master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Reconstructed biopsy cylinder length (from contour data)"] = reconstructed_biopsy_model_dict["Reconstructed biopsy cylinder length (from contour data)"]
    master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Best fit line of centroid pts"] = reconstructed_biopsy_model_dict["Best fit line of centroid pts"]
    master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Centroid line unit vec (bx needle base to bx needle tip)"] = reconstructed_biopsy_model_dict["Centroid line unit vec (bx needle base to bx needle tip)"]
    master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Centroid line vec (bx needle base to bx needle tip)"] = reconstructed_biopsy_model_dict["Centroid line vec (bx needle base to bx needle tip)"]
    master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Centroid line vec length (bx needle base to bx needle tip)"] = reconstructed_biopsy_model_dict["Centroid line vec length (bx needle base to bx needle tip)"]
    master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Centroid line sample pts"] = centroid_line_sample
    master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Reconstructed structure pts arr"] = reconstructed_biopsy_model_dict["Reconstructed structure pts arr"]
    master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Reconstructed structure point cloud"] = reconstructed_biopsy_model_dict["Reconstructed structure point cloud"]
    master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Reconstructed structure delaunay global"] = reconstructed_biopsy_model_dict["Reconstructed structure delaunay global"]
    master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Structure global centroid"] = structure_global_centroid

    return live_display