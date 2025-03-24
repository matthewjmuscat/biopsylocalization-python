import cupy as cp
import numpy as np
import pandas as pd
from cuml.neighbors import NearestNeighbors
import polygon_dilation_helpers_numpy  # assumed to be your module

def nearest_neighbor_cupy(structure_points, query_points):
    """
    Performs nearest neighbor search using CuPy (GPU acceleration).

    Parameters:
        structure_points (np.ndarray): (M,3) array of structure boundary points.
        query_points (np.ndarray): (N,3) array of query points.

    Returns:
        nearest_distances (np.ndarray): (N,) array of nearest distances.
        nearest_indices (np.ndarray): (N,) array of indices of the nearest points.
    """
    # Convert to CuPy arrays
    structure_points_cp = cp.asarray(structure_points)
    query_points_cp = cp.asarray(query_points)

    # Compute squared Euclidean distances using broadcasting
    distances_cp = cp.linalg.norm(structure_points_cp[:, None, :] - query_points_cp[None, :, :], axis=2)

    # Find the nearest neighbor (min distance and index)
    nearest_indices_cp = cp.argmin(distances_cp, axis=0)
    nearest_distances_cp = cp.min(distances_cp, axis=0)

    # Convert back to NumPy
    return cp.asnumpy(nearest_distances_cp), cp.asnumpy(nearest_indices_cp)







# verrryyy slow
def gpu_dilation_structure_boundary_biopsy_nearest_neighbor_search_cuml(
    nominal_and_dilated_structures_list_of_lists_of_const_zslice_arr,
    test_struct_to_relative_struct_1d_mapping_array,
    combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff,
    interp_dist_caps,
    num_sample_pts_in_bx,
    num_MC_containment_simulations,
    show_num_nearest_neighbour_surface_boundary_demonstration = 0,
    algorithm_NN = 'brute'
):
    """
    For each trial, creates an end-cap filled structure (if needed), builds a cuML NearestNeighbors model,
    and queries it using the GPU. Returns a DataFrame of nearest neighbor distances along with other info.
    
    Parameters:
      - nominal_and_dilated_structures_list_of_lists_of_const_zslice_arr: List of lists of constant z-slice arrays.
      - test_struct_to_relative_struct_1d_mapping_array: 1D array mapping trial indices to structure indices.
      - combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff: Array with shape 
            (num_trials, num_sample_pts_in_bx, 3) containing query points.
      - interp_dist_caps: Parameter passed to create_end_caps_for_zslices.
      - num_sample_pts_in_bx: Number of sample points per trial.
      - num_MC_containment_simulations: Total number of trials minus one.
    
    Returns:
      - nearest_neighbour_boundary_distances_df: Pandas DataFrame with results.
      - nn_cache: Dictionary caching the NearestNeighbors model for each structure.
      - nominal_and_dilated_structures_with_end_caps_list_of_2d_arr: List of 2D arrays for each structure.
    """
    
    # Total number of query points (flattened)
    total_num_test_points = (
        combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff.shape[0] *
        combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff.shape[1]
    )
    
    nearest_distance_to_structure_boundary = np.empty(total_num_test_points, dtype=np.float32)
    nearest_point_index_to_structure_boundary = np.empty(total_num_test_points, dtype=np.int32)
    
    # Preallocate a list for the processed structures (with end caps)
    if show_num_nearest_neighbour_surface_boundary_demonstration > 0:
        nominal_and_dilated_structures_with_end_caps_list_of_2d_arr = [None] * len(
            nominal_and_dilated_structures_list_of_lists_of_const_zslice_arr
        )
    
    # Cache the NearestNeighbors model per unique structure index.
    nn_cache = {}
    
    for trial_structure_index, sp_struct_index in enumerate(test_struct_to_relative_struct_1d_mapping_array):
        # Check if we have already built a model for this structure.
        if sp_struct_index not in nn_cache:
            # Create a filled version of the dilated structure.
            structure_trial_zslices_list = nominal_and_dilated_structures_list_of_lists_of_const_zslice_arr[sp_struct_index]
            structure_trial_zslices_with_end_caps_list = polygon_dilation_helpers_numpy.create_end_caps_for_zslices(
                structure_trial_zslices_list, interp_dist_caps
            )
            structure_trial_zslices_with_end_caps_2darr, _ = polygon_dilation_helpers_numpy.convert_to_2d_array_and_indices_numpy(
                structure_trial_zslices_with_end_caps_list
            )
            if show_num_nearest_neighbour_surface_boundary_demonstration > 0:
                nominal_and_dilated_structures_with_end_caps_list_of_2d_arr[sp_struct_index] = structure_trial_zslices_with_end_caps_2darr
            
            # Build the cuML NearestNeighbors model.
            # Use algorithm='brute' to guarantee exact results.
            nn_model = NearestNeighbors(n_neighbors=1, algorithm=algorithm_NN)
            nn_model.fit(structure_trial_zslices_with_end_caps_2darr)
            nn_cache[sp_struct_index] = nn_model
        else:
            nn_model = nn_cache[sp_struct_index]
        
        # Get query points for the current trial.
        query_points = combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff[trial_structure_index]
        
        # Query using cuML. The results will be exact when using 'brute'.
        distances, indices = nn_model.kneighbors(query_points)
        
        # Assign the results into our grand arrays.
        start_idx = trial_structure_index * num_sample_pts_in_bx
        end_idx = (trial_structure_index + 1) * num_sample_pts_in_bx
        nearest_distance_to_structure_boundary[start_idx:end_idx] = distances.flatten()
        nearest_point_index_to_structure_boundary[start_idx:end_idx] = indices.flatten()
    
    # Build the final DataFrame.
    nearest_neighbour_boundary_distances_df = pd.DataFrame({
        "Trial num": np.repeat(np.arange(num_MC_containment_simulations + 1), num_sample_pts_in_bx),
        "Original pt index": np.tile(np.arange(num_sample_pts_in_bx), num_MC_containment_simulations + 1),
        "Struct. boundary NN dist.": nearest_distance_to_structure_boundary,
        "Relative struct input index": np.repeat(test_struct_to_relative_struct_1d_mapping_array, num_sample_pts_in_bx)
    })
    
    if show_num_nearest_neighbour_surface_boundary_demonstration > 0:
        return nearest_neighbour_boundary_distances_df, nearest_distance_to_structure_boundary, nearest_point_index_to_structure_boundary, nn_cache, nominal_and_dilated_structures_with_end_caps_list_of_2d_arr
    else:
        return nearest_neighbour_boundary_distances_df, nearest_distance_to_structure_boundary, nearest_point_index_to_structure_boundary, nn_cache, None
    





def vectorized_nn_search(candidates, queries):
    """
    Vectorized nearest neighbor search using CuPy.
    
    Parameters:
      candidates (cp.ndarray): Array of candidate points, shape (N, d).
      queries (cp.ndarray): Array of query points, shape (M, d).
    
    Returns:
      nearest_distances (cp.ndarray): (M,) array of Euclidean distances to the nearest candidate.
      nearest_indices (cp.ndarray): (M,) array of indices of the nearest candidate.
    """
    # Compute squared norms.
    cand_norm = cp.sum(candidates ** 2, axis=1)  # (N,)
    query_norm = cp.sum(queries ** 2, axis=1)      # (M,)
    
    # Compute pairwise squared distances: shape (M, N)
    dists_sq = query_norm[:, None] + cand_norm[None, :] - 2 * cp.dot(queries, candidates.T)
    
    # Take square root (with numerical safeguard)
    dists = cp.sqrt(cp.maximum(dists_sq, 0))
    
    # Get nearest candidate for each query.
    nn_indices = cp.argmin(dists, axis=1)
    nn_dists = dists[cp.arange(queries.shape[0]), nn_indices]
    
    return nn_dists, nn_indices

def gpu_cupy_dilation_structure_boundary_biopsy_nearest_neighbor_search(
    nominal_and_dilated_structures_list_of_lists_of_const_zslice_arr,
    test_struct_to_relative_struct_1d_mapping_array,
    combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff,
    interp_dist_caps,
    num_sample_pts_in_bx,
    num_MC_containment_simulations
):
    """
    For each trial, using its mapped relative structure (which is end-cap padded),
    performs a GPU-accelerated vectorized nearest neighbor search.
    
    Parameters:
      nominal_and_dilated_structures_list_of_lists_of_const_zslice_arr: List (per structure) of lists of constant z-slice arrays.
      test_struct_to_relative_struct_1d_mapping_array: 1D array mapping each trial to a structure index.
      combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff:
          Array of shape (num_trials, num_sample_pts_in_bx, 3) with query points.
      interp_dist_caps: Parameter for creating end caps.
      num_sample_pts_in_bx: Number of query points per trial.
      num_MC_containment_simulations: Total number of trials minus one.
    
    Returns:
      A Pandas DataFrame containing, for each trial and each point, the nearest neighbor distance
      and the candidate point index (within its candidate structure).
    """
    num_trials = combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff.shape[0]
    total_num_test_points = num_trials * num_sample_pts_in_bx
    
    # Preallocate results (on CPU)
    nearest_distance = np.empty(total_num_test_points, dtype=np.float32)
    nearest_index = np.empty(total_num_test_points, dtype=np.int32)

    
    # First, compute (and cache) the candidate structure arrays for each unique structure.
    unique_structures = np.unique(test_struct_to_relative_struct_1d_mapping_array)
    candidate_cache = {}
    for sp in unique_structures:
        structure_trial_zslices_list = nominal_and_dilated_structures_list_of_lists_of_const_zslice_arr[sp]
        structure_trial_zslices_with_end_caps_list = polygon_dilation_helpers_numpy.create_end_caps_for_zslices(
            structure_trial_zslices_list, interp_dist_caps)
        structure_trial_zslices_with_end_caps_2darr, _ = polygon_dilation_helpers_numpy.convert_to_2d_array_and_indices_numpy(
            structure_trial_zslices_with_end_caps_list)
        candidate_cache[sp] = structure_trial_zslices_with_end_caps_2darr  # (Nc, 3)
    
    # Now process trials by grouping them by their candidate structure.
    # For each candidate structure sp, find the trial indices where test_struct_to_relative_struct_1d_mapping_array == sp.
    for sp in unique_structures:
        # Get the candidate array and convert it to a CuPy array.
        candidates = cp.asarray(candidate_cache[sp])  # shape (Nc, 3)
        
        # Find trial indices (from the mapping) that use this candidate structure.
        trial_idxs = np.where(test_struct_to_relative_struct_1d_mapping_array == sp)[0]
        if trial_idxs.size == 0:
            continue
        
        # Extract all query points for these trials.
        # queries_group: shape (n_trials, num_sample_pts_in_bx, 3)
        queries_group = combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff[trial_idxs, :, :]
        # Reshape to (n_trials*num_sample_pts_in_bx, 3)
        queries_group = queries_group.reshape(-1, 3)
        queries_group_cp = cp.asarray(queries_group)
        
        # Do the vectorized NN search on the group.
        dists_cp, inds_cp = vectorized_nn_search(candidates, queries_group_cp)
        
        # Bring the results back to host.
        dists = cp.asnumpy(dists_cp)
        inds = cp.asnumpy(inds_cp)
        
        # Now, assign these results into the global arrays.
        # The order of queries_group is the same as the order of trials in trial_idxs, each contributing num_sample_pts_in_bx points.
        for i, trial_idx in enumerate(trial_idxs):
            start = trial_idx * num_sample_pts_in_bx
            end = (trial_idx + 1) * num_sample_pts_in_bx
            offset = i * num_sample_pts_in_bx
            nearest_distance[start:end] = dists[offset: offset + num_sample_pts_in_bx]
            nearest_index[start:end] = inds[offset: offset + num_sample_pts_in_bx]
    
    # Build the final DataFrame.
    result_df = pd.DataFrame({
        "Trial num": np.repeat(np.arange(num_MC_containment_simulations + 1), num_sample_pts_in_bx),
        "Original pt index": np.tile(np.arange(num_sample_pts_in_bx), num_MC_containment_simulations + 1),
        "Struct. boundary NN dist.": nearest_distance,
        "Relative struct input index": np.repeat(test_struct_to_relative_struct_1d_mapping_array, num_sample_pts_in_bx)
    })
    
    return result_df, nearest_distance, nearest_index, candidate_cache




