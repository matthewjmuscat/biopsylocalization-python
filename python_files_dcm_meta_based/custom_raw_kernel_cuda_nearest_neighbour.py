import cupy as cp
import polygon_dilation_helpers_numpy
import numpy as np
import pandas as pd
import point_containment_tools
import plotting_funcs
import dataframe_builders

# Define a custom kernel for NN search (brute-force)
nn_kernel = cp.RawKernel(r'''
extern "C" __global__
void nn_search_kernel(const float* points_x,
                        const float* points_y,
                        const float* points_z,
                        const float* candidates_x,
                        const float* candidates_y,
                        const float* candidates_z,
                        const long long int* trial_candidate_starts,
                        const long long int* trial_candidate_lengths, 
                        long long int num_queries, 
                        float* out_distances, 
                        long long int* out_indices_absolute,
                        long long int* out_indices_relative) {

    // Define FLT_MAX for CUDA (manually) for float32
    #define FLT_MAX 3.40e+38f

                                    
    // Each thread handles one query point.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_queries) return;
    
    // Get the query point (assume 3D points stored contiguously)
    float qx = points_x[idx];
    float qy = points_y[idx];
    float qz = points_z[idx];
                         
    // Get the candidate structure for this query
    long long int start_idx = trial_candidate_starts[idx];
    long long int num_candidates = trial_candidate_lengths[idx];
                        
    // Initialize the best distance and index
    float best_dist = FLT_MAX;
    long long int best_idx_absolute = -1;
    long long int best_idx_relative = -1;
    
    // Loop over all candidate points
    for (long long int i = 0; i < num_candidates; i++) {
        // Get the candidate point
        float cx = candidates_x[start_idx + i];
        float cy = candidates_y[start_idx + i];
        float cz = candidates_z[start_idx + i];

        // Compute the Euclidean distance
        float dx = qx - cx;
        float dy = qy - cy;
        float dz = qz - cz;
        float dist = dx * dx + dy * dy + dz * dz;
                         
        // Update the best distance and index
        if (dist < best_dist) {
            best_dist = dist;
            best_idx_absolute = start_idx + i;
            best_idx_relative = i;
        }
    }
                         
    // Write out the Euclidean distance (take the square root) and index.
    out_distances[idx] = sqrtf(best_dist);
    out_indices_absolute[idx] = best_idx_absolute;
    out_indices_relative[idx] = best_idx_relative;
                         
}
''', 'nn_search_kernel')








def custom_gpu_kernel_prepper_function_NN(
    nominal_and_dilated_structures_list_of_lists_of_const_zslice_arr,
    test_struct_to_relative_struct_1d_mapping_array,
    combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff,
    num_sample_pts_in_bx,
    num_MC_containment_simulations,
    grid_factor = 0.1,
    kernel_type = "one_to_one_pip_kernel_advanced_reparameterized_version_gpu_memory_performance_optimized",
    check_if_end_caps_filled_proper_NN_num = 0
    ):
    """
    Prepares data for a custom GPU kernel nearest neighbor search that uses a per–trial candidate structure.
    
    Parameters:
      - nominal_and_dilated_structures_list_of_lists_of_const_zslice_arr: a list where each element is a 2D numpy array 
          of shape (N_i, 3) representing candidate points for one structure (N_i can vary).
      - test_struct_to_relative_struct_1d_mapping_array: 1D numpy array mapping each trial to a candidate structure index.
      - combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff: numpy array of shape 
          (num_trials, num_sample_pts_in_bx, 3) containing the query points for each trial.
      - interp_dist_caps: parameter for end–cap generation (passed to your helper).
      - num_sample_pts_in_bx: number of query points per trial.
      - num_MC_containment_simulations: total number of trials minus one (or as appropriate).
    
    Returns a dictionary with the following keys:
      - 'candidates_stacked': CuPy array, shape (total_candidate_points, 3), the vertically stacked candidate points.
      - 'candidates_indices': CuPy array, shape (num_structures, 2), where each row gives the [start, end) indices 
            into candidates_stacked for that candidate structure.
      - 'query_points': CuPy array, shape (total_queries, 3) containing all query points (flattened over trials).
      - 'trial_candidate_starts': CuPy array, shape (total_queries,), which for each query gives the starting index 
            in candidates_stacked for its candidate structure.
      - 'trial_candidate_lengths': CuPy array, shape (total_queries,), which gives the number of candidate points 
            for the candidate structure corresponding to that query.
      - 'trial_offsets': CuPy array, shape (num_trials,), where each element is the starting index in query_points for that trial.
      - (Other simulation parameters as needed)
    """
    # ------------------------------
    # STEP 1. Build closed candidate structures.
    # For each candidate structure (each element in the list), generate an end–capped version.
    num_structures = len(nominal_and_dilated_structures_list_of_lists_of_const_zslice_arr)

    nominal_and_dilated_structures_list_of_lists_of_const_zslice_arr_end_capped = polygon_dilation_helpers_numpy.create_end_caps_for_zslices_ver2(nominal_and_dilated_structures_list_of_lists_of_const_zslice_arr, 
                                                                                                                                                grid_factor = grid_factor,
                                                                                                                                                kernel_type= kernel_type)

    nominal_and_dilated_structures_end_capped_2d_arrs = [None] * num_structures
    for idx, sp_struct_zslices_list in enumerate(nominal_and_dilated_structures_list_of_lists_of_const_zslice_arr_end_capped):
        candidate_closed_2d, _ = polygon_dilation_helpers_numpy.convert_to_2d_array_and_indices_numpy(sp_struct_zslices_list)
        nominal_and_dilated_structures_end_capped_2d_arrs[idx] = candidate_closed_2d

    # ------------------------------
    # STEP 2. Stack all candidate arrays into one large contiguous array.
    # Also get an indices array that gives, for each candidate structure, its start and end index.
    # (Assume your helper does this; typically it returns (stacked_array, indices) where indices[i] = [start, end] for structure i.)
    candidates_stacked, candidates_indices = polygon_dilation_helpers_numpy.convert_to_2d_array_and_indices_numpy(nominal_and_dilated_structures_end_capped_2d_arrs)
    # candidates_stacked: shape (total_candidate_points, 3)
    # candidates_indices: shape (num_structures, 2)

    if check_if_end_caps_filled_proper_NN_num > 0:
        random_choice_array = np.random.choice(np.arange(0,num_structures), check_if_end_caps_filled_proper_NN_num, replace = False)
        random_choice_array = np.append(0,random_choice_array)
        for struct_idx in random_choice_array:
            struct_pcd = point_containment_tools.create_point_cloud(nominal_and_dilated_structures_end_capped_2d_arrs[struct_idx], color = np.array([0,0,0]))
            plotting_funcs.plot_geometries(struct_pcd)
    
    # ------------------------------
    # STEP 3. Prepare the query points.
    # combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff has shape (num_trials, num_sample_pts_in_bx, 3).
    num_trials = combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff.shape[0]
    total_queries = num_trials * num_sample_pts_in_bx
    # Flatten the queries to shape (total_queries, 3)
    query_points = combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff.reshape(-1, 3)
    
    # ------------------------------
    # STEP 4. Build per–trial candidate lookup.
    # For each trial (i.e. each row in the original query points), get the candidate structure index from the mapping.
    # Then, for that trial, the candidate structure starts at:
    #    candidates_indices[m][0]  and has length = candidates_indices[m][1] - candidates_indices[m][0]
    mapping = test_struct_to_relative_struct_1d_mapping_array  # assumed numpy 1D array, shape (num_trials,)
    
    # For each trial, get the start and length.
    trial_candidate_starts = np.empty(num_trials, dtype=np.int64)
    trial_candidate_lengths = np.empty(num_trials, dtype=np.int64)
    for t in range(num_trials):
        m = mapping[t]  # candidate structure index for trial t
        start, end = candidates_indices[m]
        trial_candidate_starts[t] = start
        trial_candidate_lengths[t] = end - start

    # Now we need to expand these per–trial values to each query point.
    # Each trial has num_sample_pts_in_bx query points, so repeat the candidate lookup for each trial.
    trial_candidate_starts_expanded = np.repeat(trial_candidate_starts, num_sample_pts_in_bx)
    trial_candidate_lengths_expanded = np.repeat(trial_candidate_lengths, num_sample_pts_in_bx)
    
    # Also, compute trial_offsets: where in the flattened query_points each trial begins.
    #trial_offsets = np.arange(0, total_queries, num_sample_pts_in_bx, dtype=np.int64)

    # ------------------------------
    # Convert everything to CuPy contiguous arrays.
    data_dict = {
        'candidates_stacked': cp.asarray(candidates_stacked),
        'candidates_indices': cp.asarray(candidates_indices),
        'query_points': cp.asarray(query_points),
        'trial_candidate_starts': cp.asarray(trial_candidate_starts_expanded),
        'trial_candidate_lengths': cp.asarray(trial_candidate_lengths_expanded),
        #'trial_offsets': cp.ascontiguousarray(cp.asarray(trial_offsets, dtype=cp.int64)),
        'num_trials': num_trials,
        'num_sample_pts_in_bx': num_sample_pts_in_bx,
        'num_MC_containment_simulations': num_MC_containment_simulations,
        'total queries': total_queries
    }
    
    return data_dict




def custom_gpu_kernel_NN_search_mother_function(
    nominal_and_dilated_structures_list_of_lists_of_const_zslice_arr,
    test_struct_to_relative_struct_1d_mapping_array,
    combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff,
    num_sample_pts_in_bx,
    num_MC_containment_simulations,
    grid_factor = 0.1,
    kernel_type = "one_to_one_pip_kernel_advanced_reparameterized_version_gpu_memory_performance_optimized",
    check_if_end_caps_filled_proper_NN_num = 0,
    block_size=256):

    data_dict = custom_gpu_kernel_prepper_function_NN(
                nominal_and_dilated_structures_list_of_lists_of_const_zslice_arr,
                test_struct_to_relative_struct_1d_mapping_array,
                combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff,
                num_sample_pts_in_bx,
                num_MC_containment_simulations,
                grid_factor = grid_factor,
                kernel_type = kernel_type,
                check_if_end_caps_filled_proper_NN_num = check_if_end_caps_filled_proper_NN_num)
    
    # Extract the data from the dictionary
    points_x = cp.ascontiguousarray(data_dict['query_points'][:, 0], dtype=cp.float32)
    points_y = cp.ascontiguousarray(data_dict['query_points'][:, 1], dtype=cp.float32)
    points_z = cp.ascontiguousarray(data_dict['query_points'][:, 2], dtype=cp.float32)
    candidates_x = cp.ascontiguousarray(data_dict['candidates_stacked'][:, 0], dtype=cp.float32)
    candidates_y = cp.ascontiguousarray(data_dict['candidates_stacked'][:, 1], dtype=cp.float32)
    candidates_z = cp.ascontiguousarray(data_dict['candidates_stacked'][:, 2], dtype=cp.float32)
    trial_candidate_starts = cp.ascontiguousarray(data_dict['trial_candidate_starts'], dtype=cp.int64)
    trial_candidate_lengths = cp.ascontiguousarray(data_dict['trial_candidate_lengths'], dtype=cp.int64)      
    
    num_queries = cp.int64(data_dict['total queries'])

    # Initialize the results array
    results_distances = cp.zeros(num_queries, dtype=cp.float32)
    results_indices_absolute = cp.zeros(num_queries, dtype=cp.int64)
    results_indices_relative = cp.zeros(num_queries, dtype=cp.int64)

    results_distances = cp.ascontiguousarray(results_distances, dtype=cp.float32)
    results_indices_absolute = cp.ascontiguousarray(results_indices_absolute, dtype=cp.int64)
    results_indices_relative = cp.ascontiguousarray(results_indices_relative, dtype=cp.int64)

    
    # Compute the grid size
    grid_size = (num_queries.item() + block_size - 1) // block_size


    ### IMPORTANT NOTE: WHEN CALLING THIS KERNEL, ALL DATA STORED ON GPU MUST BE CONTIGUOUSLY STORED IN MEMORY

    nn_kernel(
            (grid_size,), (block_size,),
            (points_x, 
             points_y, 
             points_z, 
             candidates_x, 
             candidates_y, 
             candidates_z,
             trial_candidate_starts, 
             trial_candidate_lengths,
             num_queries,
             results_distances,
             results_indices_absolute,
             results_indices_relative
             )
        )
    
    # Return this to check results with results_indices_absolute
    candidates_stacked = data_dict['candidates_stacked']
    candidates_indices = data_dict['candidates_indices']

    return results_distances, results_indices_relative, candidates_stacked, candidates_indices, results_indices_absolute


def build_results_df(results_distances,
                      results_indices_relative,
                      num_MC_containment_simulations,
                      num_sample_pts_in_bx,
                      test_struct_to_relative_struct_1d_mapping_array,
                      convert_to_categorical_and_downcast = True,
                      do_not_convert_column_names_to_categorical = [],
                      float_dtype = np.float32,
                      int_dtype = np.int32):
    """
    Builds a DataFrame from the results of the custom GPU kernel nearest neighbor search."
    """
    nearest_distance = cp.asnumpy(results_distances)

    result_df = pd.DataFrame({
        "Trial num": np.repeat(np.arange(num_MC_containment_simulations + 1), num_sample_pts_in_bx).astype(int_dtype),
        "Original pt index": np.tile(np.arange(num_sample_pts_in_bx), num_MC_containment_simulations + 1).astype(int_dtype),
        "Struct. boundary NN dist.": nearest_distance.astype(float_dtype),
        "Struct. boundary NN relative index (all pts stacked)": cp.asnumpy(results_indices_relative).astype(int_dtype),
        "Relative struct input index": np.repeat(test_struct_to_relative_struct_1d_mapping_array, num_sample_pts_in_bx).astype(int_dtype)
    })

    if convert_to_categorical_and_downcast:
        result_df = dataframe_builders.convert_columns_to_categorical_and_downcast(result_df, 
                                                                                            threshold=0.25, 
                                                                                            do_not_convert_column_names_to_categorical = do_not_convert_column_names_to_categorical)

    return result_df