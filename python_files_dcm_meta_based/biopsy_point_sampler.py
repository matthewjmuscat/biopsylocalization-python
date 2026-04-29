import MC_simulator_convex


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