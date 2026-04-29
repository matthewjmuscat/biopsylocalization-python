import numpy as np
import scipy.spatial

import misc_tools


def _build_bx_location_in_prostate_dict(pydicom_item,
                                        specific_bx_structure,
                                        selected_prostate_info,
                                        prostate_found_bool,
                                        oar_ref
                                        ):
    bx_structure_global_centroid = specific_bx_structure["Structure global centroid"].copy().reshape((3))

    if prostate_found_bool is True:
        prostate_structure_index = selected_prostate_info["Index number"]
        prostate_structure = pydicom_item[oar_ref][prostate_structure_index]
        prostate_structure_global_centroid = prostate_structure["Structure global centroid"].copy().reshape((3))
        prostate_dimension_at_centroid_dict = prostate_structure["Structure dimension at centroid dict"]
        prostate_z_dimension_length_at_centroid = prostate_dimension_at_centroid_dict["Z dimension length at centroid"]

        distance_to_mid_gland_threshold = abs(prostate_z_dimension_length_at_centroid / 6)
        bx_centroid_vec_rel_to_prostate_centroid = bx_structure_global_centroid - prostate_structure_global_centroid
        bx_prostate_position_dict = misc_tools.bx_position_classifier_in_prostate_frame_sextant(
            bx_centroid_vec_rel_to_prostate_centroid,
            distance_to_mid_gland_threshold,
        )
    else:
        bx_prostate_position_dict = {"LR": None, "AP": None, "SI": None}

    return {
        "Relative prostate info": selected_prostate_info,
        "Bx position in prostate": bx_prostate_position_dict,
    }


def _build_target_dil_dicts(pydicom_item,
                            specific_bx_structure,
                            dil_ref
                            ):
    bx_structure_global_centroid = specific_bx_structure["Structure global centroid"].copy().reshape((3))
    bx_structure_reconstructed_pts = specific_bx_structure["Reconstructed structure pts arr"].copy()

    dil_distance_dict = {}
    closest_dil_centroid_info = [None, None]
    closest_dil_surface_info = [None, None]

    for specific_dil_structure_index, specific_dil_structure in enumerate(pydicom_item[dil_ref]):
        dil_structure_id = specific_dil_structure["ROI"]
        dil_structure_reference_number = specific_dil_structure["Ref #"]

        dil_structure_info = (
            dil_structure_id,
            dil_ref,
            dil_structure_reference_number,
            specific_dil_structure_index,
        )

        dil_structure_global_centroid = specific_dil_structure["Structure global centroid"].copy().reshape((3))
        vector_between_bx_centroid_and_dil_centroid = dil_structure_global_centroid - bx_structure_global_centroid
        distance_between_bx_centroid_and_dil_centroid = np.linalg.norm(vector_between_bx_centroid_and_dil_centroid)

        if closest_dil_centroid_info[1] is None:
            closest_dil_centroid_info[0] = dil_structure_info
            closest_dil_centroid_info[1] = distance_between_bx_centroid_and_dil_centroid
        elif distance_between_bx_centroid_and_dil_centroid < closest_dil_centroid_info[1]:
            closest_dil_centroid_info[0] = dil_structure_info
            closest_dil_centroid_info[1] = distance_between_bx_centroid_and_dil_centroid

        dil_interslice_interpolation_information = specific_dil_structure["Inter-slice interpolation information"]
        dil_structure_interpolated_points = dil_interslice_interpolation_information.interpolated_pts_np_arr
        dil_kd_tree_scipy = scipy.spatial.KDTree(dil_structure_interpolated_points)
        nn_distances, _ = dil_kd_tree_scipy.query(bx_structure_reconstructed_pts, k=1)
        closest_surface_to_surface_distance_bx_to_dil = np.amin(nn_distances)

        if closest_dil_surface_info[1] is None:
            closest_dil_surface_info[0] = dil_structure_info
            closest_dil_surface_info[1] = closest_surface_to_surface_distance_bx_to_dil
        elif closest_surface_to_surface_distance_bx_to_dil < closest_dil_surface_info[1]:
            closest_dil_surface_info[0] = dil_structure_info
            closest_dil_surface_info[1] = closest_surface_to_surface_distance_bx_to_dil

        dil_distance_dict[dil_structure_info] = {
            "DIL centroid vector": dil_structure_global_centroid,
            "Bx centroid vector": bx_structure_global_centroid,
            "Vector DIL centroid - BX centroid": vector_between_bx_centroid_and_dil_centroid,
            "X to DIL centroid": vector_between_bx_centroid_and_dil_centroid[0],
            "Y to DIL centroid": vector_between_bx_centroid_and_dil_centroid[1],
            "Z to DIL centroid": vector_between_bx_centroid_and_dil_centroid[2],
            "Distance DIL centroid - BX centroid": distance_between_bx_centroid_and_dil_centroid,
            "Shortest distance from BX surface to DIL surface": closest_surface_to_surface_distance_bx_to_dil,
        }

    target_dil_by_centroids_dict = {closest_dil_centroid_info[0]: dil_distance_dict[closest_dil_centroid_info[0]]}
    target_dil_by_surfaces_dict = {closest_dil_surface_info[0]: dil_distance_dict[closest_dil_surface_info[0]]}

    return target_dil_by_centroids_dict, target_dil_by_surfaces_dict, dil_distance_dict


def get_legacy_realized_biopsy_targeting_fields(pydicom_item,
                                                specific_bx_structure,
                                                selected_prostate_info,
                                                prostate_found_bool,
                                                oar_ref,
                                                dil_ref
                                                ):
    bx_location_in_prostate_ref_frame_dict = _build_bx_location_in_prostate_dict(
        pydicom_item,
        specific_bx_structure,
        selected_prostate_info,
        prostate_found_bool,
        oar_ref,
    )
    target_dil_by_centroids_dict, target_dil_by_surfaces_dict, dil_distance_dict = _build_target_dil_dicts(
        pydicom_item,
        specific_bx_structure,
        dil_ref,
    )

    return {
        "Bx location in prostate dict": bx_location_in_prostate_ref_frame_dict,
        "Target DIL by centroid dict": target_dil_by_centroids_dict,
        "Target DIL by surfaces dict": target_dil_by_surfaces_dict,
        "Nearest DILs info dict": dil_distance_dict,
    }