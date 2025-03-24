import numpy as np
import pandas
import dataframe_builders

def relative_structure_centroid_calculation_function(centroids_of_nominal_and_each_dilated_structure_2darr,
                                     test_struct_to_relative_struct_1d_mapping_array,
                                     combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff):
    # New version incoroporating dilations
    # resort the centroids array according to the 1d mapping
    centroids_of_nominal_and_each_dilated_structure_2darr_resorted = centroids_of_nominal_and_each_dilated_structure_2darr[test_struct_to_relative_struct_1d_mapping_array]

    # Ensure centroids have shape (num_slices, 1, 3) for broadcasting
    centroids_of_nominal_and_each_dilated_structure_reshaped = centroids_of_nominal_and_each_dilated_structure_2darr_resorted[:, np.newaxis, :]

    # Compute centroid-to-point vectors
    non_bx_structure_centroid_to_bx_points_vectors_all_trials = combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff - centroids_of_nominal_and_each_dilated_structure_reshaped

    # Compute Euclidean distances (L2 norm)                    
    non_bx_structure_centroid_to_bx_points_vectors_all_trials_distances = np.linalg.norm(non_bx_structure_centroid_to_bx_points_vectors_all_trials, axis=2)

    # Flatten the distances
    non_bx_structure_centroid_to_bx_points_vectors_all_trials_distances_flattened = non_bx_structure_centroid_to_bx_points_vectors_all_trials_distances.flatten()

    # Extract X, Y, Z coordinates and flatten them
    non_bx_structure_centroid_to_bx_points_vectors_all_trials_distances_flattened_X = non_bx_structure_centroid_to_bx_points_vectors_all_trials[:, :, 0].flatten()
    non_bx_structure_centroid_to_bx_points_vectors_all_trials_distances_flattened_Y = non_bx_structure_centroid_to_bx_points_vectors_all_trials[:, :, 1].flatten()
    non_bx_structure_centroid_to_bx_points_vectors_all_trials_distances_flattened_Z = non_bx_structure_centroid_to_bx_points_vectors_all_trials[:, :, 2].flatten()

    return non_bx_structure_centroid_to_bx_points_vectors_all_trials_distances_flattened, non_bx_structure_centroid_to_bx_points_vectors_all_trials_distances_flattened_X, non_bx_structure_centroid_to_bx_points_vectors_all_trials_distances_flattened_Y, non_bx_structure_centroid_to_bx_points_vectors_all_trials_distances_flattened_Z, centroids_of_nominal_and_each_dilated_structure_2darr_resorted


def relative_structure_centroid_df(non_bx_structure_centroid_to_bx_points_vectors_all_trials_distances_flattened,
                                   non_bx_structure_centroid_to_bx_points_vectors_all_trials_distances_flattened_X,
                                   non_bx_structure_centroid_to_bx_points_vectors_all_trials_distances_flattened_Y,
                                   non_bx_structure_centroid_to_bx_points_vectors_all_trials_distances_flattened_Z,
                                   num_MC_containment_simulations,
                                   num_sample_pts_in_bx,
                                   test_struct_to_relative_struct_1d_mapping_array,
                                   convert_to_categorical_and_downcast = True,
                                   do_not_convert_column_names_to_categorical = [],
                                   float_dtype = np.float32,
                                   int_dtype = np.int32):
    
    structure_centroid_distances_df = pandas.DataFrame({"Trial num": np.repeat(np.arange(num_MC_containment_simulations+1), num_sample_pts_in_bx).astype(int_dtype),
                                      "Original pt index": np.tile(np.arange(num_sample_pts_in_bx), num_MC_containment_simulations+1).astype(int_dtype),
                                      "Relative struct input index": np.repeat(test_struct_to_relative_struct_1d_mapping_array, num_sample_pts_in_bx).astype(int_dtype),
                                      'Dist. from struct. centroid': non_bx_structure_centroid_to_bx_points_vectors_all_trials_distances_flattened.astype(float_dtype),
                                      'Dist. from struct. centroid X': non_bx_structure_centroid_to_bx_points_vectors_all_trials_distances_flattened_X.astype(float_dtype),
                                      'Dist. from struct. centroid Y': non_bx_structure_centroid_to_bx_points_vectors_all_trials_distances_flattened_Y.astype(float_dtype),
                                      'Dist. from struct. centroid Z': non_bx_structure_centroid_to_bx_points_vectors_all_trials_distances_flattened_Z.astype(float_dtype)})
    
    if convert_to_categorical_and_downcast:
        structure_centroid_distances_df = dataframe_builders.convert_columns_to_categorical_and_downcast(structure_centroid_distances_df, 
                                                                                            threshold=0.25, 
                                                                                            do_not_convert_column_names_to_categorical = do_not_convert_column_names_to_categorical)
        
    return structure_centroid_distances_df