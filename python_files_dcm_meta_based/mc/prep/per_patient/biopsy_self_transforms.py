import cupy as cp
import numpy as np

import pca
from MC_prepper_funcs import biopsy_dilator_step_1
from MC_prepper_funcs import biopsy_rotator_step_2_vectorized_version
from MC_prepper_funcs import biopsy_translator_step_3


def _load_plotting_helpers():
    import plotting_funcs
    import point_containment_tools

    return plotting_funcs, point_containment_tools


def apply_patient_biopsy_self_transforms(*,
                                         patient_uid,
                                         pydicom_item,
                                         bx_ref,
                                         max_simulations,
                                         simulate_uniform_bx_shifts_due_to_bx_needle_compartment,
                                         inspect_self_biopsy_dilate_bool,
                                         inspect_self_biopsy_dilate_and_rotate_bool,
                                         inspect_self_biopsy_dilate_and_rotate_and_translate_bool):
    """Apply the legacy BX-only MC transform loop body for one patient."""
    transformed_biopsy_count = 0

    for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
        randomly_sampled_bx_pts_arr = specific_bx_structure["Random uniformly sampled volume pts arr"]
        randomly_sampled_bx_pts_cp_arr = cp.array(randomly_sampled_bx_pts_arr)

        bx_normal_dist_dilations_samples_arr = specific_bx_structure["MC data: Generated normal dist random samples dilations arr"]
        bx_normal_dist_dilations_samples_cp_arr = cp.array(bx_normal_dist_dilations_samples_arr)

        centroid_line = specific_bx_structure["Best fit line of centroid pts"]
        centroid_line_cp_arr = cp.array(centroid_line)

        bx_global_centroid = specific_bx_structure["Structure global centroid"].reshape((1, 3))
        bx_global_centroid_cp_arr = cp.array(bx_global_centroid)

        randomly_sampled_bx_pts_cp_arr_dilated_max_simulations = biopsy_dilator_step_1(
            randomly_sampled_bx_pts_cp_arr,
            bx_normal_dist_dilations_samples_cp_arr,
            centroid_line_cp_arr,
            bx_global_centroid_cp_arr,
            max_simulations,
        )

        if inspect_self_biopsy_dilate_bool == True:
            plotting_funcs, point_containment_tools = _load_plotting_helpers()
            for trial_index in np.arange(max_simulations):
                nominal_bx_pcd_color = np.array([1, 0, 1])
                nominal_bx_pcd = point_containment_tools.create_point_cloud(randomly_sampled_bx_pts_arr, nominal_bx_pcd_color)

                pcd_color_bx_self_dilation = np.array([0, 1, 1])
                self_bx_dilation_step_point_cloud = point_containment_tools.create_point_cloud(cp.asnumpy(randomly_sampled_bx_pts_cp_arr_dilated_max_simulations[trial_index]), pcd_color_bx_self_dilation)

                bx_global_centroid_pcd = point_containment_tools.create_point_cloud(bx_global_centroid, np.array([0, 0, 0]))
                plotting_funcs.plot_geometries(nominal_bx_pcd, self_bx_dilation_step_point_cloud, bx_global_centroid_pcd)

        specific_structure_normal_dist_dilations_samples_arr = specific_bx_structure["MC data: Generated normal dist random samples rotations arr"]
        specific_structure_normal_dist_dilations_samples_cp_arr = cp.array(specific_structure_normal_dist_dilations_samples_arr)

        randomly_sampled_bx_pts_cp_arr_dilated_and_rotated_max_simulations = biopsy_rotator_step_2_vectorized_version(
            randomly_sampled_bx_pts_cp_arr_dilated_max_simulations,
            specific_structure_normal_dist_dilations_samples_cp_arr,
            bx_global_centroid_cp_arr,
            max_simulations,
        )

        if inspect_self_biopsy_dilate_and_rotate_bool == True:
            plotting_funcs, point_containment_tools = _load_plotting_helpers()
            for trial_index in np.arange(max_simulations):
                nominal_bx_pcd_color = np.array([1, 0, 1])
                nominal_bx_pcd = point_containment_tools.create_point_cloud(randomly_sampled_bx_pts_arr, nominal_bx_pcd_color)

                pcd_color_bx_self_dilation = np.array([0, 1, 1])
                self_bx_dilation_step_point_cloud = point_containment_tools.create_point_cloud(cp.asnumpy(randomly_sampled_bx_pts_cp_arr_dilated_max_simulations[trial_index]), pcd_color_bx_self_dilation)

                pcd_color_bx_self_dilate_and_rotate = np.array([0, 1, 0.8])
                self_bx_dilate_and_rotation_step_point_cloud = point_containment_tools.create_point_cloud(cp.asnumpy(randomly_sampled_bx_pts_cp_arr_dilated_and_rotated_max_simulations[trial_index]), pcd_color_bx_self_dilate_and_rotate)

                bx_global_centroid_pcd = point_containment_tools.create_point_cloud(bx_global_centroid, np.array([0, 0, 0]))
                plotting_funcs.plot_geometries(nominal_bx_pcd, self_bx_dilation_step_point_cloud, self_bx_dilate_and_rotation_step_point_cloud, bx_global_centroid_pcd)

        if simulate_uniform_bx_shifts_due_to_bx_needle_compartment == True:
            random_uniformly_sampled_bx_shifts_cp_arr = specific_bx_structure["MC data: Generated uniform dist (biopsy needle compartment) random distance (z_needle) samples arr"]

            lines = pca.vectorized_linear_fitter(randomly_sampled_bx_pts_cp_arr_dilated_and_rotated_max_simulations)

            point1 = lines[:, 0, :]
            point2 = lines[:, 1, :]

            mask = point1[:, 2] > point2[:, 2]
            point_sup = cp.where(mask[:, None], point1, point2)
            point_inf = cp.where(mask[:, None], point2, point1)

            vec = point_sup - point_inf
            biopsy_vec_handle_to_tip_unit = vec / cp.linalg.norm(vec, axis=1, keepdims=True)
            biopsy_vec_tip_to_handle_unit = -biopsy_vec_handle_to_tip_unit
            all_trials_bx_unit_vecs_tip_to_handle = biopsy_vec_tip_to_handle_unit

            bx_needle_uniform_compartment_shift_vectors_cp_array = cp.multiply(all_trials_bx_unit_vecs_tip_to_handle, random_uniformly_sampled_bx_shifts_cp_arr[..., None])
            bx_needle_uniform_compartment_shift_vectors_np_array = cp.asnumpy(bx_needle_uniform_compartment_shift_vectors_cp_array)
            specific_bx_structure["MC data: Generated uniform (biopsy needle compartment) random vectors (z_needle) samples arr"] = bx_needle_uniform_compartment_shift_vectors_np_array

            bx_normal_translation_arr = specific_bx_structure["MC data: Generated normal dist random samples arr"]
            bx_total_only_translation_arr = bx_needle_uniform_compartment_shift_vectors_cp_array + cp.array(bx_normal_translation_arr)

            bx_only_shifted_needle_compartment_shifted_only_randomly_sampled_bx_pts_3Darr = biopsy_translator_step_3(randomly_sampled_bx_pts_cp_arr_dilated_and_rotated_max_simulations, bx_needle_uniform_compartment_shift_vectors_cp_array)
            bx_only_shifted_randomly_sampled_bx_pts_3Darr = biopsy_translator_step_3(bx_only_shifted_needle_compartment_shifted_only_randomly_sampled_bx_pts_3Darr, cp.array(bx_normal_translation_arr))
        else:
            bx_total_only_translation_arr = specific_bx_structure["MC data: Generated normal dist random samples arr"]

            bx_only_shifted_needle_compartment_shifted_only_randomly_sampled_bx_pts_3Darr = randomly_sampled_bx_pts_cp_arr_dilated_and_rotated_max_simulations
            bx_only_shifted_randomly_sampled_bx_pts_3Darr = biopsy_translator_step_3(randomly_sampled_bx_pts_cp_arr_dilated_and_rotated_max_simulations, cp.array(bx_total_only_translation_arr))

        if inspect_self_biopsy_dilate_and_rotate_and_translate_bool == True:
            plotting_funcs, point_containment_tools = _load_plotting_helpers()
            for trial_index in np.arange(max_simulations):
                nominal_bx_pcd_color = np.array([1, 0, 1])
                nominal_bx_pcd = point_containment_tools.create_point_cloud(randomly_sampled_bx_pts_arr, nominal_bx_pcd_color)

                pcd_color_bx_self_dilation = np.array([0, 1, 1])
                self_bx_dilation_step_point_cloud = point_containment_tools.create_point_cloud(cp.asnumpy(randomly_sampled_bx_pts_cp_arr_dilated_max_simulations[trial_index]), pcd_color_bx_self_dilation)

                pcd_color_bx_self_dilate_and_rotate = np.array([0, 1, 0])
                self_bx_dilate_and_rotation_step_point_cloud = point_containment_tools.create_point_cloud(cp.asnumpy(randomly_sampled_bx_pts_cp_arr_dilated_and_rotated_max_simulations[trial_index]), pcd_color_bx_self_dilate_and_rotate)

                pcd_color_bx_self_dilate_and_rotate_and_needle_compartment_shift = np.array([1, 0, 0])
                self_bx_dilate_and_rotation_and_needle_compartment_shift_step_point_cloud = point_containment_tools.create_point_cloud(cp.asnumpy(bx_only_shifted_needle_compartment_shifted_only_randomly_sampled_bx_pts_3Darr[trial_index]), pcd_color_bx_self_dilate_and_rotate_and_needle_compartment_shift)

                pcd_color_bx_self_dilate_and_rotate_and_needle_compartment_shift_and_general_shift = np.array([1, 0.5, 0])
                self_bx_dilate_and_rotation_and_needle_compartment_shift_and_general_shift_step_point_cloud = point_containment_tools.create_point_cloud(cp.asnumpy(bx_only_shifted_randomly_sampled_bx_pts_3Darr[trial_index]), pcd_color_bx_self_dilate_and_rotate_and_needle_compartment_shift_and_general_shift)

                bx_global_centroid_pcd = point_containment_tools.create_point_cloud(bx_global_centroid, np.array([0, 0, 0]))
                plotting_funcs.plot_geometries(
                    nominal_bx_pcd,
                    self_bx_dilation_step_point_cloud,
                    self_bx_dilate_and_rotation_step_point_cloud,
                    self_bx_dilate_and_rotation_and_needle_compartment_shift_step_point_cloud,
                    self_bx_dilate_and_rotation_and_needle_compartment_shift_and_general_shift_step_point_cloud,
                    bx_global_centroid_pcd,
                    show_axes=True,
                    axes_origin=bx_global_centroid,
                )

        specific_bx_structure["MC data: bx only shifted 3darr"] = bx_only_shifted_randomly_sampled_bx_pts_3Darr.get()
        transformed_biopsy_count += 1

    return {
        "patient_uid": patient_uid,
        "num_transformed_biopsies": transformed_biopsy_count,
    }