import cupy as cp
import numpy as np

import plotting_funcs
import point_containment_tools
from MC_prepper_funcs import rotate_biopsy_to_relative_structure_points_vectorized
from MC_prepper_funcs import translate_biopsy_to_relative_structure_points


def _create_patient_specific_structure_dict_for_data(pydicom_item,
                                                     structs_referenced_list):
    structure_organized_for_bx_data_blank_dict = {}
    for non_bx_struct_type in structs_referenced_list[1:]:
        for specific_non_bx_structure_index, specific_non_bx_structure in enumerate(pydicom_item[non_bx_struct_type]):
            specific_non_bx_struct_roi = specific_non_bx_structure["ROI"]
            specific_non_bx_struct_refnum = specific_non_bx_structure["Ref #"]
            structure_organized_for_bx_data_blank_dict[
                specific_non_bx_struct_roi,
                non_bx_struct_type,
                specific_non_bx_struct_refnum,
                specific_non_bx_structure_index,
            ] = None

    return structure_organized_for_bx_data_blank_dict


def apply_patient_relative_structure_transforms(*,
                                                patient_uid,
                                                pydicom_item,
                                                structs_referenced_list,
                                                bx_ref,
                                                num_MC_containment_simulations,
                                                inspect_relative_structure_rotate_and_shift_number):
    """Apply the legacy relative-structure MC transform loop body for one patient."""
    transformed_biopsy_count = 0

    structure_organized_for_bx_data_blank_dict = _create_patient_specific_structure_dict_for_data(
        pydicom_item,
        structs_referenced_list,
    )

    for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
        structure_shifted_bx_data_dict = structure_organized_for_bx_data_blank_dict.copy()

        bx_global_centroid = specific_bx_structure["Structure global centroid"].reshape((1, 3))
        randomly_sampled_bx_pts_arr = specific_bx_structure["Random uniformly sampled volume pts arr"]

        bx_only_shifted_randomly_sampled_bx_pts_3Darr = cp.asarray(specific_bx_structure["MC data: bx only shifted 3darr"])

        for non_bx_struct_type in structs_referenced_list[1:]:
            for specific_non_bx_struct_index, specific_non_bx_struct in enumerate(pydicom_item[non_bx_struct_type]):
                specific_non_bx_struct_roi = specific_non_bx_struct["ROI"]
                specific_non_bx_struct_refnum = specific_non_bx_struct["Ref #"]

                relative_structure_global_centroid = specific_non_bx_struct["Structure global centroid"].reshape((1, 3))

                relative_structure_normal_dist_rotations_samples_arr = specific_non_bx_struct["MC data: Generated normal dist random samples rotations arr"]
                relative_structure_normal_dist_translations_samples_arr = specific_non_bx_struct["MC data: Generated normal dist random samples arr"]

                bx_rotated_relative_structure = rotate_biopsy_to_relative_structure_points_vectorized(
                    bx_only_shifted_randomly_sampled_bx_pts_3Darr,
                    relative_structure_normal_dist_rotations_samples_arr,
                    relative_structure_global_centroid,
                    num_MC_containment_simulations,
                )

                bx_rotated_and_translated_relative_structure = translate_biopsy_to_relative_structure_points(
                    bx_rotated_relative_structure,
                    relative_structure_normal_dist_translations_samples_arr,
                    num_MC_containment_simulations,
                )

                structure_shifted_bx_data_dict[
                    specific_non_bx_struct_roi,
                    non_bx_struct_type,
                    specific_non_bx_struct_refnum,
                    specific_non_bx_struct_index,
                ] = bx_rotated_and_translated_relative_structure.get()

                for trial_index in np.arange(inspect_relative_structure_rotate_and_shift_number):
                    nominal_bx_pcd_color = np.array([1, 0, 1])
                    nominal_bx_pcd = point_containment_tools.create_point_cloud(randomly_sampled_bx_pts_arr, nominal_bx_pcd_color)

                    pcd_color_self_bx_transformations_step_point_cloud = np.array([0, 1, 1])
                    self_bx_transformations_step_point_cloud = point_containment_tools.create_point_cloud(cp.asnumpy(bx_only_shifted_randomly_sampled_bx_pts_3Darr[trial_index]), pcd_color_self_bx_transformations_step_point_cloud)

                    pcd_color_bx_to_relative_structure_transformations_step_point_cloud = np.array([0, 0, 1])
                    bx_to_relative_structure_rotation_only_step_point_cloud = point_containment_tools.create_point_cloud(cp.asnumpy(bx_rotated_relative_structure[trial_index]), pcd_color_bx_to_relative_structure_transformations_step_point_cloud)

                    pcd_color_bx_to_relative_structure_transformations_step_point_cloud = np.array([1, 0, 0])
                    bx_to_relative_structure_transformations_step_point_cloud = point_containment_tools.create_point_cloud(cp.asnumpy(bx_rotated_and_translated_relative_structure[trial_index]), pcd_color_bx_to_relative_structure_transformations_step_point_cloud)

                    interpolated_relative_structure_pcd = specific_non_bx_struct["Interpolated structure point cloud dict"]["Full"]

                    bx_global_centroid_pcd = point_containment_tools.create_point_cloud(bx_global_centroid, np.array([0, 0, 0]))
                    relative_structure_global_centroid_pcd = point_containment_tools.create_point_cloud(relative_structure_global_centroid, np.array([0, 0, 0]))

                    plotting_funcs.plot_geometries(
                        nominal_bx_pcd,
                        self_bx_transformations_step_point_cloud,
                        bx_to_relative_structure_rotation_only_step_point_cloud,
                        bx_to_relative_structure_transformations_step_point_cloud,
                        bx_global_centroid_pcd,
                        interpolated_relative_structure_pcd,
                        relative_structure_global_centroid_pcd,
                        show_axes=True,
                        axes_origin=relative_structure_global_centroid,
                    )

        pydicom_item[bx_ref][specific_bx_structure_index]["MC data: bx and structure shifted dict"] = structure_shifted_bx_data_dict
        transformed_biopsy_count += 1

    return {
        "patient_uid": patient_uid,
        "num_transformed_biopsies": transformed_biopsy_count,
    }