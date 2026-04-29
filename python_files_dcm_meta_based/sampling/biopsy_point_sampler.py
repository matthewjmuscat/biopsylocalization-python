import MC_simulator_convex
import numpy as np


def sample_biopsy_points_from_reconstructed_global_delaunay_convex_structure(
        grid_separation_distance,
        delaunay_global_convex_structure_tri,
        reconstructed_bx_arr,
        patientUID,
        structure_type,
        specific_structure_index,
        z_axis_to_centroid_vec_rotation_matrix,
):
    return MC_simulator_convex.grid_point_sampler_rotated_from_global_delaunay_convex_structure_parallel_repaired(
        grid_separation_distance,
        delaunay_global_convex_structure_tri,
        reconstructed_bx_arr,
        patientUID,
        structure_type,
        specific_structure_index,
        z_axis_to_centroid_vec_rotation_matrix,
    )


def sample_biopsy_points_from_reconstructed_biopsy_model_dict(
        grid_separation_distance,
        reconstructed_biopsy_model_dict,
        patientUID,
        structure_type,
        specific_structure_index,
):
    reconstructed_delaunay_global_convex_structure_obj = reconstructed_biopsy_model_dict["Reconstructed structure delaunay global"]
    if reconstructed_delaunay_global_convex_structure_obj is None:
        raise ValueError("Reconstructed biopsy model delaunay object is missing")

    reconstructed_bx_arr = reconstructed_biopsy_model_dict["Reconstructed structure pts arr"]
    z_axis_to_centroid_vec_rotation_matrix = np.asarray(
        reconstructed_biopsy_model_dict["Centroid line to z axis rotation matrix"],
        dtype=float,
    ).T

    return sample_biopsy_points_from_reconstructed_global_delaunay_convex_structure(
        grid_separation_distance,
        reconstructed_delaunay_global_convex_structure_obj.delaunay_triangulation,
        reconstructed_bx_arr,
        patientUID,
        structure_type,
        specific_structure_index,
        z_axis_to_centroid_vec_rotation_matrix,
    )