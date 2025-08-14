import numpy as np
from shapely.geometry import Polygon, Point

import plotting_funcs
import point_containment_tools
import time
import pathlib
import cProfile
import pstats
import io
import multiprocess
import os 
import cupy as cp
import custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p

def convert_to_2d_array_and_indices_numpy(list_of_arrays, num_columns = 3, main_arr_dtype = np.float32, indices_arr_dtype = np.int32):
    # Calculate the total number of points
    total_points = sum(len(polygon) for polygon in list_of_arrays)
    
    # Preallocate the points array using NumPy
    points_array = np.empty((total_points, num_columns), dtype=main_arr_dtype)
    
    # Preallocate the indices array using NumPy
    indices_array = np.empty((len(list_of_arrays), 2), dtype=indices_arr_dtype)
    
    # Fill the points array and indices array
    current_index = 0
    for i, polygon in enumerate(list_of_arrays):
        num_points = len(polygon)
        points_array[current_index:current_index + num_points] = np.array(polygon)
        indices_array[i, 0] = current_index
        indices_array[i, 1] = current_index + num_points
        current_index += num_points
    
    return points_array, indices_array


def reconstruct_list_from_2d_array(points_array, indices_array):
    """
    Reconstructs a list of arrays from a 2D points array and an indices array.
    
    Parameters:
        points_array (np.ndarray): The stacked points array of shape (total_points, num_columns).
        indices_array (np.ndarray): The indices array of shape (num_slices, 2), where each row contains
                                    the start and end indices of each slice.

    Returns:
        list_of_arrays (list of np.ndarray): A list of arrays, each containing points from a constant Z slice.
    """
    list_of_arrays = [points_array[start:end] for start, end in indices_array]
    return list_of_arrays


def convert_to_2d_array_and_indices_from_3d_arr_numpy(pts_3d_arr):
    # Calculate the total number of points
    total_points = pts_3d_arr.shape[1]
    num_slices = pts_3d_arr.shape[0]

    stacked_2d_arr = pts_3d_arr.reshape(-1, 3)
    indices_flat_arr = np.arange(num_slices + 1) * total_points  # Generate the start and end points

    indices_2d_arr = np.stack([indices_flat_arr[:-1], indices_flat_arr[1:]], axis=1)

    return stacked_2d_arr, indices_2d_arr

def dilate_polygons_z_direction(points_array, indices_array, dilation_distance_z, show_z_dilation_bool=False):
    """
    Dilate the Z slices by adjusting the Z coordinates using provided distance.
    
    Parameters:
    - points_array: 2D array of points with their Z coordinates.
    - indices_array: Array of indices that separate slices.
    - dilation_distance_z: Distance by which to dilate the Z coordinates.
    
    Returns:
    - dilated_points_array: 2D array of dilated points with adjusted Z coordinates.
    - new_indices_array: Array of indices that separate slices after filtering.
    """
    # Extract Z coordinates
    z_coords = points_array[:, 2]
    central_z = np.mean(z_coords)
    
    # Calculate new Z coordinates
    z_signs = np.sign(z_coords - central_z)
    new_z_coords = z_coords + z_signs * dilation_distance_z
    
    # Create a mask to filter out slices that move past or equal to the central slice
    mask = (z_signs * (new_z_coords - central_z) > 0) | (z_coords == central_z)
    
    # Apply the mask to filter points
    filtered_points_array = points_array[mask]
    new_z_coords_filtered = new_z_coords[mask]
    
    # Update Z coordinates in the filtered points
    filtered_points_array[:, 2] = new_z_coords_filtered
    
    if show_z_dilation_bool:
        original_pcd = point_containment_tools.create_point_cloud(points_array, color=np.array([0, 0, 1]))  # paint tested structure in blue
        dilated_pcd = point_containment_tools.create_point_cloud(filtered_points_array, color=np.array([1, 0, 0]))  # paint tested structure in red
        plotting_funcs.plot_geometries(original_pcd, dilated_pcd)
    
    # Initialize empty array for new indices
    new_indices_array = np.empty((0, 2), dtype=np.int32)
    current_index = 0
    
    for start_index, end_index in indices_array:
        num_points = np.sum(mask[start_index:end_index])
        if num_points > 0:
            new_indices_array = np.vstack((new_indices_array, np.array([[current_index, current_index + num_points]], dtype=np.int32)))
            current_index += num_points
    
    return filtered_points_array, new_indices_array

# Note that I set the quad_segs_inp to 3, which will hopefully quicken the dilation process a bit
# Reminder that quad_segs determines the number of segments to approximate a quarter circle around the buffered vertex of the original polygon
def dilate_polygons_xy_plane(points_array_org, points_array, indices_array, dilation_distance_xy, show_xy_dilation_bool, min_area=1e-6, quad_segs_inp = 3):
    """
    Dilate the polygons in the XY plane using provided distance.
    
    Parameters:
    - points_array: 2D array of points with their Z coordinates.
    - indices_array: 2D array of indices that separate slices, where each row contains [start_index, end_index].
    - dilation_distance_xy: Distance by which to dilate the XY coordinates.
    - min_area: Minimum area to ensure the polygon is not deleted.
    
    Returns:
    - dilated_points_array: 2D array of dilated points in the XY plane.
    - new_indices_array: Array of indices that separate slices after dilation.
    """
    current_index = 0

    # Initialize empty arrays for dilated points and new indices
    dilated_points_array = np.empty((0, 3), dtype=np.float32)
    new_indices_array = np.empty((0, 2), dtype=np.int32)

    for start_index, end_index in indices_array:
        polygon = points_array[start_index:end_index]
        shapely_polygon = Polygon(polygon[:, :2])

        # Calculate the maximum inward dilation distance
        min_inward_dilation = -shapely_polygon.minimum_clearance
        
        # Apply the dilation
        dilated_shapely_polygon = shapely_polygon.buffer(
            float(dilation_distance_xy), 
            quad_segs=quad_segs_inp, 
            cap_style='square', 
            join_style='mitre'
        )
        
        # Check if the dilated polygon is empty or too small
        if dilated_shapely_polygon.is_empty or dilated_shapely_polygon.area < min_area:
            # If empty or too small, iteratively decrease the inward dilation distance towards zero
            inward_dilation_distance = min_inward_dilation
            while dilated_shapely_polygon.is_empty or dilated_shapely_polygon.area < min_area:
                inward_dilation_distance += 0.1 * abs(min_inward_dilation)  # Move closer to zero
                if inward_dilation_distance >= 0:
                    break  # Prevent moving past zero
                dilated_shapely_polygon = shapely_polygon.buffer(
                    inward_dilation_distance, 
                    quad_segs=quad_segs_inp, 
                    cap_style='square', 
                    join_style='mitre'
                )
        
        # Handle the case where the result is a MultiPolygon
        if dilated_shapely_polygon.geom_type == 'MultiPolygon':
            largest_polygon = max(dilated_shapely_polygon.geoms, key=lambda p: p.area)
            dilated_shapely_polygon = largest_polygon

        # Convert the dilated polygon back to a NumPy array
        dilated_polygon_coords = np.array(dilated_shapely_polygon.exterior.coords) # IMPORTANT! This line returns a polygon such that the first and last points are the same!
        dilated_polygon = np.hstack((dilated_polygon_coords, np.full((dilated_polygon_coords.shape[0], 1), polygon[0, 2])))
        
        num_dilated_points = dilated_polygon.shape[0]
        
        # Stack the new dilated points to the dilated_points_array
        dilated_points_array = np.vstack((dilated_points_array, dilated_polygon))
        
        # Stack the new indices to the new_indices_array
        new_indices_array = np.vstack((new_indices_array, np.array([[current_index, current_index + num_dilated_points]], dtype=np.int32)))
        
        current_index += num_dilated_points
    
    if show_xy_dilation_bool:
        original_pcd = point_containment_tools.create_point_cloud(points_array_org, color=np.array([0, 0, 1]))  # paint tested structure in blue
        dilated_z_pcd = point_containment_tools.create_point_cloud(points_array, color=np.array([1, 0, 0]))  # paint tested structure in red
        dilated_total_pcd = point_containment_tools.create_point_cloud(dilated_points_array, color=np.array([0, 1, 0]))  # paint tested structure in green
        plotting_funcs.plot_geometries(original_pcd, dilated_z_pcd, dilated_total_pcd)

    return dilated_points_array, new_indices_array

def dilate_polygons(points_array, indices_array, dilation_distance_z, dilation_distance_xy, show_z_dilation_bool, show_xy_dilation_bool):
    """
    Dilate the polygons in both Z direction and XY plane using provided distances.
    
    Parameters:
    - points_array: 2D array of points with their Z coordinates.
    - indices_array: Array of indices that separate slices.
    - dilation_distance_z: Distance by which to dilate the Z coordinates.
    - dilation_distance_xy: Distance by which to dilate the XY coordinates.
    
    Returns:
    - dilated_points_array: 2D array of dilated points.
    - indices_array: Array of indices that separate slices (unchanged).
    """
    # Dilate Z slices
    dilated_points_z, indices_array = dilate_polygons_z_direction(points_array, indices_array, dilation_distance_z, show_z_dilation_bool)
    
    # Dilate XY plane
    dilated_points_xy, indices_array = dilate_polygons_xy_plane(points_array, dilated_points_z, indices_array, dilation_distance_xy, show_xy_dilation_bool)
    
    return dilated_points_xy, indices_array

def generate_dilated_structures(points_array, indices_array, dilation_distances, show_z_dilation_bool, show_xy_dilation_bool):
    """
    Generate a list of dilated structures for each trial.
    
    Parameters:
    - points_array: 2D array of original points.
    - indices_array: Array of indices that separate slices.
    - dilation_distances: 2D array of distances by which to dilate the XY and Z coordinates for each trial.
    
    Returns:
    - dilated_structures_list: List of dilated structures.
    """
    num_trials = dilation_distances.shape[0]
    dilated_structures_list = []
    dilated_structures_slices_indices_list = []
    
    for trial in range(num_trials):
        dilation_distance_z = dilation_distances[trial, 1]
        dilation_distance_xy = dilation_distances[trial, 0]
        dilated_points, indices_array = dilate_polygons(points_array, indices_array, dilation_distance_z, dilation_distance_xy, show_z_dilation_bool, show_xy_dilation_bool)
        dilated_structures_list.append(dilated_points)
        dilated_structures_slices_indices_list.append(indices_array)
    
    return dilated_structures_list, dilated_structures_slices_indices_list


def generate_dilated_structures_parallelized(points_array, indices_array, dilation_distances, show_z_dilation_bool, show_xy_dilation_bool, parallel_pool):
    """
    Generate a list of dilated structures for each trial.
    
    Parameters:
    - points_array: 2D array of original points.
    - indices_array: Array of indices that separate slices.
    - dilation_distances: 2D array of distances by which to dilate the XY and Z coordinates for each trial.
    
    Returns:
    - dilated_structures_list: List of dilated structures.
    - dilated_structures_slices_indices_list: List of indices that separate slices for each dilated structure.
    """
    num_trials = dilation_distances.shape[0]

    # Prepare arguments for parallel processing
    args = [(points_array, indices_array, dilation_distances[trial, 1], dilation_distances[trial, 0], show_z_dilation_bool, show_xy_dilation_bool) for trial in range(num_trials)]

    # Use parallel processing to generate dilated structures
    results = parallel_pool.starmap(dilate_polygons, args)

    # Separate the results into dilated structures and their corresponding indices
    dilated_structures_list = [result[0] for result in results]
    dilated_structures_slices_indices_list = [result[1] for result in results]

    return dilated_structures_list, dilated_structures_slices_indices_list

def extract_constant_z_values(dilated_structures_list, dilated_structures_slices_indices_list):
    z_values_list = []
    for trial, indices_array in zip(dilated_structures_list, dilated_structures_slices_indices_list):
        trial_z_values = []
        for start_index, end_index in indices_array:
            z_value = trial[start_index][2]  # Assuming the z value is constant for all points in the structure
            trial_z_values.append(z_value)
        z_values_list.append(trial_z_values)
    return z_values_list

def extract_constant_z_values_arr_version(all_structures_list_of_2d_arr, all_structures_slices_indices_list):
    num_structures = len(all_structures_list_of_2d_arr)    
    z_values_list_of_arrays = [None]*num_structures
    for index in range(num_structures):
        structure_2d_arr = all_structures_list_of_2d_arr[index]
        indices_array = all_structures_slices_indices_list[index]
        start_indices = indices_array[:, 0]
        z_vals_arr = structure_2d_arr[start_indices, 2]
        z_values_list_of_arrays[index] = z_vals_arr
    return z_values_list_of_arrays


def extract_constant_z_values_single_configuration(list_of_constant_z_slices_arrays):
    z_values_list = []
    for z_slice_arr in list_of_constant_z_slices_arrays:
        z_val = z_slice_arr[0][2]
        z_values_list.append(z_val)
    return z_values_list


# This function although vectorized nicely, assumes that each dilated structure has the same number of slices, which is not necessarily true
def nearest_zslice_vals_and_indices_all_trials_vectorized(non_bx_struct_nominal_and_all_dilations_zvals_list, combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff):
    """
    Find the nearest z values and associated indices for all biopsy points across all trials using NumPy.
    
    Parameters:
    - non_bx_struct_nominal_and_all_dilations_zvals_list: List of z values for each trial of the dilated relative structure.
    - combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff: 3D array of biopsy points for all trials.
    
    Returns:
    - grand_all_dilations_sp_trial_nearest_interpolated_zslice_index_array: Array of indices of the nearest z values for each biopsy point.
    - grand_all_dilation_sp_trial_nearest_interpolated_zslice_vals_array: Array of the nearest z values for each biopsy point.
    """
    num_trials = len(non_bx_struct_nominal_and_all_dilations_zvals_list)
    num_points_per_trial = combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff.shape[1]
    
    # Convert the list of z values to a 2D NumPy array for broadcasting
    zvals_array = np.array(non_bx_struct_nominal_and_all_dilations_zvals_list)
    
    # Extract the z coordinates of the biopsy points
    biopsy_z_coords = np.array(combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff[:, :, 2])
    
    # Use broadcasting to compute the absolute differences between z values
    z_diffs = np.abs(zvals_array[:, :, np.newaxis] - biopsy_z_coords[np.newaxis, :, :])
    
    # Find the indices of the minimum differences
    nearest_zslice_indices = np.argmin(z_diffs, axis=1)
    
    # Get the nearest z values using the indices
    nearest_zslice_vals = np.take_along_axis(zvals_array, nearest_zslice_indices, axis=1)
    
    # Check if the biopsy z values are outside the range of the relative structure's z values
    min_zvals = np.min(zvals_array, axis=1)[:, np.newaxis]
    max_zvals = np.max(zvals_array, axis=1)[:, np.newaxis]
    
    outside_min_mask = biopsy_z_coords < min_zvals
    outside_max_mask = biopsy_z_coords > max_zvals
    
    # Set the indices and values to NaN where the biopsy z values are outside the range
    nearest_zslice_indices = np.where(outside_min_mask | outside_max_mask, np.nan, nearest_zslice_indices)
    nearest_zslice_vals = np.where(outside_min_mask | outside_max_mask, np.nan, nearest_zslice_vals)
    
    # Flatten the results to match the expected output format
    grand_all_dilations_sp_trial_nearest_interpolated_zslice_index_array = nearest_zslice_indices.flatten()
    grand_all_dilation_sp_trial_nearest_interpolated_zslice_vals_array = nearest_zslice_vals.flatten()
    
    return grand_all_dilations_sp_trial_nearest_interpolated_zslice_index_array, grand_all_dilation_sp_trial_nearest_interpolated_zslice_vals_array


def nearest_zslice_vals_and_indices_all_structures_3d_point_arr(relative_structures_list_of_zvals_1d_arrays_or_lists, points_to_test_3d_arr, test_struct_to_relative_struct_1d_mapping_array):
    """
    Find the nearest z values and associated indices for all biopsy points across all trials using NumPy. This function is for test points that have the same number of points for each relative structure (likely the same object that has been transformed for each trial, or is just the same object for each trial).
    
    Parameters:
    - non_bx_struct_nominal_and_all_dilations_zvals_list: List of z values for each trial of the dilated relative structure.
    - combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff: 3D array of biopsy points for all trials.
    - test_struct_to_relative_struct_1d_mapping_array: 1D array that maps each test structure to its associated relative structure.

    Returns:
    - grand_all_dilations_sp_trial_nearest_interpolated_zslice_array: Array of the relative structure index, nearest z index slice and nearest z values on the associated relative structure for each test structure point (row).
    """
    #num_trials = len(relative_structures_list_of_zvals_1d_arrays_or_lists)
    num_test_structures = points_to_test_3d_arr.shape[0]
    num_points_per_trial = points_to_test_3d_arr.shape[1]
    
    # Initialize array to store the results
    nearest_zslice_index_and_values_3d_arr = np.empty((num_test_structures, num_points_per_trial, 4), dtype=np.float32)
    
    for test_structure_index, relative_structure_index in enumerate(test_struct_to_relative_struct_1d_mapping_array):
        zvals_array = np.array(relative_structures_list_of_zvals_1d_arrays_or_lists[relative_structure_index])
        pts_to_test_z_coords = points_to_test_3d_arr[test_structure_index, :, 2]
        
        # Use broadcasting to compute the absolute differences between z values
        z_diffs = np.abs(zvals_array[:, np.newaxis] - pts_to_test_z_coords[np.newaxis, :])
        
        # Find the indices of the minimum differences
        nearest_zslice_indices = np.argmin(z_diffs, axis=0)
        
        # Get the nearest z values using the indices
        nearest_zslice_vals = zvals_array[nearest_zslice_indices]
        
        # Check if the biopsy z values are outside the range of the relative structure's z values
        min_zval = np.min(zvals_array)
        max_zval = np.max(zvals_array)
        
        outside_min_mask = pts_to_test_z_coords < min_zval
        outside_max_mask = pts_to_test_z_coords > max_zval
        
        # Set the indices and values to NaN where the biopsy z values are outside the range
        #nearest_zslice_indices = np.where(outside_min_mask | outside_max_mask, np.nan, nearest_zslice_indices)
        #nearest_zslice_vals = np.where(outside_min_mask | outside_max_mask, np.nan, nearest_zslice_vals)

        # Create an out-of-bounds flag (1 if out of bounds, 0 otherwise)
        out_of_bounds_flag = np.where(outside_min_mask | outside_max_mask, 1, 0)
        
        # Store the results
        nearest_zslice_index_and_values_3d_arr[test_structure_index, :, 0] = relative_structure_index
        nearest_zslice_index_and_values_3d_arr[test_structure_index, :, 1] = nearest_zslice_indices
        nearest_zslice_index_and_values_3d_arr[test_structure_index, :, 2] = nearest_zslice_vals
        nearest_zslice_index_and_values_3d_arr[test_structure_index, :, 3] = out_of_bounds_flag

    return nearest_zslice_index_and_values_3d_arr


# much better to use cupy for the z_diffs and argmin lines!!
def nearest_zslice_vals_and_indices_all_structures_3d_point_arr_ver4(relative_structures_list_of_zvals_1d_arrays_or_lists, points_to_test_3d_arr, test_struct_to_relative_struct_1d_mapping_array):
    """
    Find the nearest z values and associated indices for all biopsy points across all trials using NumPy. This function is for test points that have the same number of points for each relative structure (likely the same object that has been transformed for each trial, or is just the same object for each trial).
    
    Parameters:
    - non_bx_struct_nominal_and_all_dilations_zvals_list: List of z values for each trial of the dilated relative structure.
    - combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff: 3D array of biopsy points for all trials.
    - test_struct_to_relative_struct_1d_mapping_array: 1D array that maps each test structure to its associated relative structure.

    Returns:
    - grand_all_dilations_sp_trial_nearest_interpolated_zslice_array: Array of the relative structure index, nearest z index slice and nearest z values on the associated relative structure for each test structure point (row).
    """
    #num_trials = len(relative_structures_list_of_zvals_1d_arrays_or_lists)
    num_test_structures = points_to_test_3d_arr.shape[0]
    num_points_per_trial = points_to_test_3d_arr.shape[1]
    
    # Initialize array to store the results
    nearest_zslice_index_and_values_3d_arr = np.empty((num_test_structures, num_points_per_trial, 4), dtype=np.float32)
    
    for test_structure_index, relative_structure_index in enumerate(test_struct_to_relative_struct_1d_mapping_array):
        zvals_array = np.array(relative_structures_list_of_zvals_1d_arrays_or_lists[relative_structure_index])
        pts_to_test_z_coords = points_to_test_3d_arr[test_structure_index, :, 2]
        

        # Converting to cupy for zdiffs and argmin then converting back to numpy much faster for large arrays!!
        """
        # Use broadcasting to compute the absolute differences between z values
        z_diffs = np.abs(zvals_array[:, np.newaxis] - pts_to_test_z_coords[np.newaxis, :])
        
        # Find the indices of the minimum differences
        nearest_zslice_indices = np.argmin(z_diffs, axis=0)
        """

        # Convert them to CuPy arrays
        zvals_cp = cp.asarray(zvals_array)  # shape: (num_slices,)
        pts_cp   = cp.asarray(pts_to_test_z_coords)  # shape: (num_points,)

        # Use broadcasting on GPU:
        z_diffs_cp = cp.abs(zvals_cp[:, cp.newaxis] - pts_cp[cp.newaxis, :])
        nearest_zslice_indices_cp = cp.argmin(z_diffs_cp, axis=0)  # shape: (num_points,)

        # Convert back to NumPy array
        nearest_zslice_indices = cp.asnumpy(nearest_zslice_indices_cp)
        
        # Get the nearest z values using the indices
        nearest_zslice_vals = zvals_array[nearest_zslice_indices]
        
        # Check if the biopsy z values are outside the range of the relative structure's z values
        min_zval = np.min(zvals_array)
        max_zval = np.max(zvals_array)
        
        outside_min_mask = pts_to_test_z_coords < min_zval
        outside_max_mask = pts_to_test_z_coords > max_zval
        
        # Set the indices and values to NaN where the biopsy z values are outside the range
        #nearest_zslice_indices = np.where(outside_min_mask | outside_max_mask, np.nan, nearest_zslice_indices)
        #nearest_zslice_vals = np.where(outside_min_mask | outside_max_mask, np.nan, nearest_zslice_vals)

        # Create an out-of-bounds flag (1 if out of bounds, 0 otherwise)
        out_of_bounds_flag = np.where(outside_min_mask | outside_max_mask, 1, 0)
        
        # Store the results
        nearest_zslice_index_and_values_3d_arr[test_structure_index, :, 0] = relative_structure_index
        nearest_zslice_index_and_values_3d_arr[test_structure_index, :, 1] = nearest_zslice_indices
        nearest_zslice_index_and_values_3d_arr[test_structure_index, :, 2] = nearest_zslice_vals
        nearest_zslice_index_and_values_3d_arr[test_structure_index, :, 3] = out_of_bounds_flag

    return nearest_zslice_index_and_values_3d_arr




def nearest_zslice_vals_and_indices_all_structures_3d_point_arr_ver5(
    relative_structures_list_of_zvals_1d_arrays_or_lists,
    points_to_test_3d_arr,
    test_struct_to_relative_struct_1d_mapping_array):
    """
    Find the nearest z values and associated indices for all biopsy points across all trials.
    Returns a float32 array shaped (num_test_structures, num_points_per_trial, 4) with:
      [:, :, 0] = relative_structure_index
      [:, :, 1] = nearest z-slice index (float32-encoded)
      [:, :, 2] = nearest z value
      [:, :, 3] = out-of-bounds flag (1 if point z < min(zvals) or > max(zvals), else 0)
    """

    num_test_structures = points_to_test_3d_arr.shape[0]
    num_points_per_trial = points_to_test_3d_arr.shape[1]

    # results: [rel_struct_idx, nearest_idx, nearest_val, oob_flag]
    nearest_zslice_index_and_values_3d_arr = np.empty(
        (num_test_structures, num_points_per_trial, 4), dtype=np.float32
    )

    for test_structure_index, relative_structure_index in enumerate(test_struct_to_relative_struct_1d_mapping_array):
        # Host arrays (ensure float32 to reduce device memory traffic)
        zvals_array = np.asarray(
            relative_structures_list_of_zvals_1d_arrays_or_lists[relative_structure_index],
            dtype=np.float32
        )
        pts_to_test_z_coords = np.asarray(
            points_to_test_3d_arr[test_structure_index, :, 2],
            dtype=np.float32
        )

        # Device copies
        zvals_cp = cp.asarray(zvals_array)           # (S,)
        pts_cp   = cp.asarray(pts_to_test_z_coords)  # (P,)

        # If z-values may be unsorted, sort once and map back
        # (If you know they are sorted ascending, you can skip this block and use the fast path below.)
        is_sorted = (
            zvals_array.size < 2 or
            bool(np.all(zvals_array[1:] >= zvals_array[:-1]))
        )

        if not is_sorted:
            order = cp.argsort(zvals_cp)               # (S,)
            z_sorted = zvals_cp[order]                 # (S,)

            idx_insert = cp.searchsorted(z_sorted, pts_cp, side='left')        # (P,)
            idx0 = cp.clip(idx_insert - 1, 0, z_sorted.size - 1)
            idx1 = cp.clip(idx_insert,       0, z_sorted.size - 1)

            # choose the closer neighbor
            choose_right = cp.abs(z_sorted[idx1] - pts_cp) < cp.abs(pts_cp - z_sorted[idx0])
            nearest_sorted_idx = cp.where(choose_right, idx1, idx0)            # (P,)
            nearest_idx_cp = order[nearest_sorted_idx]                          # map back to original indexing
        else:
            # Fast path when zvals are already sorted ascending
            idx_insert = cp.searchsorted(zvals_cp, pts_cp, side='left')        # (P,)
            idx0 = cp.clip(idx_insert - 1, 0, zvals_cp.size - 1)
            idx1 = cp.clip(idx_insert,       0, zvals_cp.size - 1)
            choose_right = cp.abs(zvals_cp[idx1] - pts_cp) < cp.abs(pts_cp - zvals_cp[idx0])
            nearest_idx_cp = cp.where(choose_right, idx1, idx0)                 # (P,)

        nearest_vals_cp = zvals_cp[nearest_idx_cp]                               # (P,)

        # Bring back to host once
        nearest_zslice_indices = cp.asnumpy(nearest_idx_cp)                      # int64 on host
        nearest_zslice_vals    = cp.asnumpy(nearest_vals_cp).astype(np.float32)  # float32

        # Out-of-bounds flag on host (same logic as original)
        min_zval = float(np.min(zvals_array))
        max_zval = float(np.max(zvals_array))
        outside_min_mask = pts_to_test_z_coords < min_zval
        outside_max_mask = pts_to_test_z_coords > max_zval
        out_of_bounds_flag = np.where(outside_min_mask | outside_max_mask, 1, 0).astype(np.float32)

        # Store results (indices are stored as float32 in your original output array)
        nearest_zslice_index_and_values_3d_arr[test_structure_index, :, 0] = np.float32(relative_structure_index)
        nearest_zslice_index_and_values_3d_arr[test_structure_index, :, 1] = nearest_zslice_indices.astype(np.float32)
        nearest_zslice_index_and_values_3d_arr[test_structure_index, :, 2] = nearest_zslice_vals
        nearest_zslice_index_and_values_3d_arr[test_structure_index, :, 3] = out_of_bounds_flag

    return nearest_zslice_index_and_values_3d_arr





def nearest_zslice_vals_and_indices_all_structures_3d_point_arr_ver6(
    relative_structures_list_of_zvals_1d_arrays_or_lists,
    points_to_test_3d_arr,
    test_struct_to_relative_struct_1d_mapping_array,
    prefer_searchsorted_over=0.75,   # if estimated temp > this fraction of free VRAM -> use searchsorted
    vram_safety=0.6,                 # when chunking, use only this fraction of free VRAM
    dtype=np.float32
):
    """
    Hybrid method: chooses between broadcast+argmin (chunked) and searchsorted.
    Returns the same (N_test_struct, N_points, 4) float32 array as ver4/ver5:
      [:,:,0] = relative_structure_index
      [:,:,1] = nearest z-slice INDEX (stored as float32)
      [:,:,2] = nearest z VALUE
      [:,:,3] = out-of-bounds flag (1/0)
    """


    num_test_structures = points_to_test_3d_arr.shape[0]
    num_points_per_trial = points_to_test_3d_arr.shape[1]
    out = np.empty((num_test_structures, num_points_per_trial, 4), dtype=np.float32)

    # helper: chunked broadcast+argmin for a single (S,P)
    def _broadcast_chunked(zvals_cp, pts_cp):
        S = zvals_cp.size
        P = pts_cp.size
        free_mem, total_mem = cp.cuda.runtime.memGetInfo()
        free_mem = int(free_mem * vram_safety)

        # temp per chunk ≈ S*chunk*sizeof(float32) for z_diffs + a little overhead
        bytes_per_elem = 4  # float32
        max_chunk = max(1, int(free_mem // max(bytes_per_elem * max(S, 1), 1)))
        max_chunk = min(P, max_chunk)

        parts_idx = []
        for start in range(0, P, max_chunk):
            end = min(P, start + max_chunk)
            z_diffs = cp.abs(zvals_cp[:, None] - pts_cp[None, start:end])  # (S, chunk)
            parts_idx.append(cp.argmin(z_diffs, axis=0))
            del z_diffs
        idx_cp = cp.concatenate(parts_idx, axis=0)
        vals_cp = zvals_cp[idx_cp]
        return idx_cp, vals_cp

    # helper: searchsorted method
    def _searchsorted(zvals_cp, pts_cp, zvals_sorted_known=True):
        if zvals_sorted_known:
            idx_insert = cp.searchsorted(zvals_cp, pts_cp, side='left')
            i0 = cp.clip(idx_insert - 1, 0, zvals_cp.size - 1)
            i1 = cp.clip(idx_insert,       0, zvals_cp.size - 1)
            choose_right = cp.abs(zvals_cp[i1] - pts_cp) < cp.abs(pts_cp - zvals_cp[i0])
            idx_cp = cp.where(choose_right, i1, i0)
            vals_cp = zvals_cp[idx_cp]
            return idx_cp, vals_cp
        else:
            order = cp.argsort(zvals_cp)
            zs = zvals_cp[order]
            idx_insert = cp.searchsorted(zs, pts_cp, side='left')
            i0 = cp.clip(idx_insert - 1, 0, zs.size - 1)
            i1 = cp.clip(idx_insert,       0, zs.size - 1)
            choose_right = cp.abs(zs[i1] - pts_cp) < cp.abs(pts_cp - zs[i0])
            nearest_sorted = cp.where(choose_right, i1, i0)
            idx_cp = order[nearest_sorted]
            vals_cp = zvals_cp[idx_cp]
            return idx_cp, vals_cp

    for test_structure_index, relative_structure_index in enumerate(test_struct_to_relative_struct_1d_mapping_array):
        zvals = np.asarray(
            relative_structures_list_of_zvals_1d_arrays_or_lists[relative_structure_index],
            dtype=dtype
        )
        pts_z = np.asarray(points_to_test_3d_arr[test_structure_index, :, 2], dtype=dtype)
        S = zvals.size
        P = pts_z.size

        # device
        zvals_cp = cp.asarray(zvals)
        pts_cp   = cp.asarray(pts_z)

        # decide path
        # Estimate temp for full broadcast: S*P*4 bytes (float32), then be conservative ×1.2
        full_temp = int(S) * int(P) * 4
        free_mem, total_mem = cp.cuda.runtime.memGetInfo()
        use_searchsorted = (full_temp > prefer_searchsorted_over * free_mem)

        # If zvals are sorted ascending, we can use the faster path in searchsorted
        z_sorted_known = bool(np.all(zvals[1:] >= zvals[:-1])) if S > 1 else True

        if use_searchsorted:
            idx_cp, vals_cp = _searchsorted(zvals_cp, pts_cp, z_sorted_known)
        else:
            # still use chunking so we never OOM
            idx_cp, vals_cp = _broadcast_chunked(zvals_cp, pts_cp)

        # back to host
        nearest_idx = cp.asnumpy(idx_cp).astype(np.float32)
        nearest_val = cp.asnumpy(vals_cp).astype(np.float32)

        # OOB flags on host (same as your logic)
        mn = float(np.min(zvals)) if S else np.inf
        mx = float(np.max(zvals)) if S else -np.inf
        oob = ((pts_z < mn) | (pts_z > mx)).astype(np.float32)

        out[test_structure_index, :, 0] = np.float32(relative_structure_index)
        out[test_structure_index, :, 1] = nearest_idx
        out[test_structure_index, :, 2] = nearest_val
        out[test_structure_index, :, 3] = oob

    return out



def nearest_zslice_vals_and_indices_all_structures_3d_point_arr_ver7(
    relative_structures_list_of_zvals_1d_arrays_or_lists,
    points_to_test_3d_arr,
    test_struct_to_relative_struct_1d_mapping_array,
    prefer_searchsorted_over=0.75,   # if estimated temp > this fraction of free VRAM -> use searchsorted
    vram_safety=0.6,                 # when chunking, use only this fraction of free VRAM
    dtype=np.float32,
    use_method="auto"                # "auto" | "broadcast" | "searchsorted"
):
    """
    Hybrid method with override:
      - use_method="auto"        -> pick per-structure using VRAM heuristic
      - use_method="broadcast"   -> force broadcast+argmin (chunked safely)
      - use_method="searchsorted"-> force searchsorted+neighbors

    Returns:
      out: float32 array (N_test_structures, N_points, 4)
           [:,:,0] = relative_structure_index
           [:,:,1] = nearest z-slice INDEX (stored as float32)
           [:,:,2] = nearest z VALUE
           [:,:,3] = out-of-bounds flag (1/0)
      methods_used: list[str] of length N_test_structures with "broadcast" or "searchsorted"
    """
    import numpy as np
    import cupy as cp

    num_test_structures = points_to_test_3d_arr.shape[0]
    num_points_per_trial = points_to_test_3d_arr.shape[1]
    out = np.empty((num_test_structures, num_points_per_trial, 4), dtype=np.float32)
    methods_used = []

    # helper: chunked broadcast+argmin for a single (S,P)
    def _broadcast_chunked(zvals_cp, pts_cp):
        S = zvals_cp.size
        P = pts_cp.size
        free_mem, _ = cp.cuda.runtime.memGetInfo()
        free_mem = int(free_mem * vram_safety)

        # temp per chunk ≈ S*chunk*sizeof(float32)
        bytes_per_elem = 4  # float32
        denom = max(bytes_per_elem * max(S, 1), 1)
        max_chunk = max(1, int(free_mem // denom))
        max_chunk = min(P, max_chunk)

        parts_idx = []
        for start in range(0, P, max_chunk):
            end = min(P, start + max_chunk)
            z_diffs = cp.abs(zvals_cp[:, None] - pts_cp[None, start:end])  # (S, chunk)
            parts_idx.append(cp.argmin(z_diffs, axis=0))
            del z_diffs
        idx_cp = cp.concatenate(parts_idx, axis=0)
        vals_cp = zvals_cp[idx_cp]
        return idx_cp, vals_cp

    # helper: searchsorted method
    def _searchsorted(zvals_cp, pts_cp, zvals_sorted_known=True):
        if zvals_sorted_known:
            idx_insert = cp.searchsorted(zvals_cp, pts_cp, side='left')
            i0 = cp.clip(idx_insert - 1, 0, zvals_cp.size - 1)
            i1 = cp.clip(idx_insert,       0, zvals_cp.size - 1)
            choose_right = cp.abs(zvals_cp[i1] - pts_cp) < cp.abs(pts_cp - zvals_cp[i0])
            idx_cp = cp.where(choose_right, i1, i0)
            vals_cp = zvals_cp[idx_cp]
            return idx_cp, vals_cp
        else:
            order = cp.argsort(zvals_cp)
            zs = zvals_cp[order]
            idx_insert = cp.searchsorted(zs, pts_cp, side='left')
            i0 = cp.clip(idx_insert - 1, 0, zs.size - 1)
            i1 = cp.clip(idx_insert,       0, zs.size - 1)
            choose_right = cp.abs(zs[i1] - pts_cp) < cp.abs(pts_cp - zs[i0])
            nearest_sorted = cp.where(choose_right, i1, i0)
            idx_cp = order[nearest_sorted]
            vals_cp = zvals_cp[idx_cp]
            return idx_cp, vals_cp

    for test_structure_index, relative_structure_index in enumerate(test_struct_to_relative_struct_1d_mapping_array):
        zvals = np.asarray(
            relative_structures_list_of_zvals_1d_arrays_or_lists[relative_structure_index],
            dtype=dtype
        )
        pts_z = np.asarray(points_to_test_3d_arr[test_structure_index, :, 2], dtype=dtype)
        S = zvals.size
        P = pts_z.size

        # device
        zvals_cp = cp.asarray(zvals)
        pts_cp   = cp.asarray(pts_z)

        # determine method
        if use_method == "broadcast":
            method = "broadcast"
        elif use_method == "searchsorted":
            method = "searchsorted"
        else:  # auto
            # Estimate temp for full broadcast: S*P*4 bytes (float32), conservative ×1.2 in threshold
            full_temp = int(S) * int(P) * 4
            free_mem, _ = cp.cuda.runtime.memGetInfo()
            method = "searchsorted" if (full_temp > prefer_searchsorted_over * free_mem) else "broadcast"

        # choose path
        if method == "broadcast":
            idx_cp, vals_cp = _broadcast_chunked(zvals_cp, pts_cp)
        else:
            z_sorted_known = bool(np.all(zvals[1:] >= zvals[:-1])) if S > 1 else True
            idx_cp, vals_cp = _searchsorted(zvals_cp, pts_cp, z_sorted_known)

        # back to host
        nearest_idx = cp.asnumpy(idx_cp).astype(np.float32)
        nearest_val = cp.asnumpy(vals_cp).astype(np.float32)

        # OOB flags on host (same as before)
        mn = float(np.min(zvals)) if S else np.inf
        mx = float(np.max(zvals)) if S else -np.inf
        oob = ((pts_z < mn) | (pts_z > mx)).astype(np.float32)

        out[test_structure_index, :, 0] = np.float32(relative_structure_index)
        out[test_structure_index, :, 1] = nearest_idx
        out[test_structure_index, :, 2] = nearest_val
        out[test_structure_index, :, 3] = oob

        methods_used.append(method)

    return out, methods_used




## Also very slow compared to nearest_zslice_vals_and_indices_all_structures_3d_point_arr for large arrays points_to_test_3d_arr.shape = (1750,10000,3)
def nearest_zslice_vals_and_indices_all_structures_3d_point_arr_ver3(
    relative_structures_list_of_zvals_1d_arrays, 
    points_to_test_3d_arr, 
    test_struct_to_relative_struct_1d_mapping_array
):
    """
    Vectorized NumPy version to compute, for each test point,
    the nearest z-slice index, the corresponding z value, and an out-of-bounds flag.
    
    Parameters:
      - relative_structures_list_of_zvals_1d_arrays: list of 1D arrays (or lists) of z-values for each relative structure.
      - points_to_test_3d_arr: NumPy array of shape (num_test_structures, num_points_per_trial, 3) containing test points.
      - test_struct_to_relative_struct_1d_mapping_array: 1D array mapping each test structure to a relative structure index.
    
    Returns:
      - A NumPy array of shape (num_test_structures, num_points_per_trial, 4) with:
            Column 0: the relative structure index (repeated)
            Column 1: the nearest z-slice index (within the padded z-values array)
            Column 2: the nearest z value
            Column 3: out-of-bounds flag (1 if the test point's z is outside the min/max of the relative structure's z-values, 0 otherwise)
    """
    num_test_structures, num_points_per_trial, _ = points_to_test_3d_arr.shape
    num_structures = len(relative_structures_list_of_zvals_1d_arrays)
    
    # Determine maximum number of slices among all relative structures
    max_slices = max(len(zvals) for zvals in relative_structures_list_of_zvals_1d_arrays)
    
    # Create a padded array of z-values with shape (num_structures, max_slices), filled with np.nan
    padded_zvals = np.full((num_structures, max_slices), np.nan, dtype=np.float32)
    for i, zvals in enumerate(relative_structures_list_of_zvals_1d_arrays):
        zvals = np.array(zvals, dtype=np.float32)
        padded_zvals[i, :len(zvals)] = zvals

    # Get test points' z coordinates; shape: (num_test_structures, num_points_per_trial)
    test_z = points_to_test_3d_arr[..., 2]
    
    # For each test structure, select its corresponding relative structure's z-values
    # test_struct_to_relative_struct_1d_mapping_array should be an array of integers of length num_test_structures.
    mapping = np.asarray(test_struct_to_relative_struct_1d_mapping_array, dtype=np.int32)
    struct_zvals = padded_zvals[mapping]  # shape: (num_test_structures, max_slices)
    
    # --- The two critical (and heavy) lines ---
    # Compute the absolute differences between every candidate z value and each test point's z.
    # This yields an array of shape: (num_test_structures, max_slices, num_points_per_trial)
    z_diffs = np.abs(struct_zvals[:, :, np.newaxis] - test_z[:, np.newaxis, :])
    
    # For each test point, find the index of the z value with the minimum difference.
    nearest_z_indices = np.argmin(z_diffs, axis=1)  # shape: (num_test_structures, num_points_per_trial)
    # ---------------------------------------------------------------
    
    # Retrieve the nearest z values using advanced indexing.
    row_idx = np.arange(num_test_structures)[:, np.newaxis]
    nearest_zvals = struct_zvals[row_idx, nearest_z_indices]  # shape: (num_test_structures, num_points_per_trial)
    
    # Compute the minimum and maximum z values for each structure (ignoring nan)
    min_zvals = np.nanmin(struct_zvals, axis=1, keepdims=True)  # shape: (num_test_structures, 1)
    max_zvals = np.nanmax(struct_zvals, axis=1, keepdims=True)  # shape: (num_test_structures, 1)
    
    # Determine out-of-bounds flag: 1 if test point's z is less than min or greater than max, 0 otherwise.
    out_of_bounds = ((test_z < min_zvals) | (test_z > max_zvals)).astype(np.float32)
    
    # Build the final output array with shape (num_test_structures, num_points_per_trial, 4)
    # Column 0: relative structure index (repeated for each test point)
    out0 = np.broadcast_to(mapping[:, np.newaxis].astype(np.float32), (num_test_structures, num_points_per_trial))
    out1 = nearest_z_indices.astype(np.float32)
    out2 = nearest_zvals
    out3 = out_of_bounds
    
    result = np.stack([out0, out1, out2, out3], axis=-1)
    return result




### Note this one seems actually very slow compared to nearest_zslice_vals_and_indices_all_structures_3d_point_arr for large arrays points_to_test_3d_arr.shape = (1750,10000,3)
def nearest_zslice_vals_and_indices_all_structures_3d_point_arr_ver2(
    relative_structures_list_of_zvals_1d_arrays_or_lists, 
    points_to_test_3d_arr, 
    test_struct_to_relative_struct_1d_mapping_array):
    """
    Optimized version using vectorized NumPy operations to find nearest z-slice indices and values.
    """
    num_test_structures, num_points_per_trial = points_to_test_3d_arr.shape[:2]
    
    # Convert list of z-values arrays into a padded 2D NumPy array (fast indexing)
    max_slices = max(len(z) for z in relative_structures_list_of_zvals_1d_arrays_or_lists)
    zvals_padded = np.full((len(relative_structures_list_of_zvals_1d_arrays_or_lists), max_slices), np.nan, dtype=np.float32)
    for i, zvals in enumerate(relative_structures_list_of_zvals_1d_arrays_or_lists):
        zvals_padded[i, :len(zvals)] = zvals  # Fill valid entries
    
    # Get the Z-coordinates of test points
    test_z_coords = points_to_test_3d_arr[..., 2]  # Shape: (num_test_structures, num_points_per_trial)

    # Retrieve corresponding structure z-values for each test structure
    struct_zvals = zvals_padded[test_struct_to_relative_struct_1d_mapping_array]  # Shape: (num_test_structures, max_slices)

    # Compute absolute differences in a vectorized manner
    z_diffs = np.abs(struct_zvals[:, :, np.newaxis] - test_z_coords[:, np.newaxis, :])  # Shape: (num_test_structures, max_slices, num_points_per_trial)

    # Find the indices of the minimum differences (nearest z-slice)
    nearest_zslice_indices = np.nanargmin(z_diffs, axis=1)  # Shape: (num_test_structures, num_points_per_trial)

    # Get the nearest z values using the indices
    nearest_zslice_vals = np.take_along_axis(struct_zvals, nearest_zslice_indices[:, np.newaxis], axis=1)[:, 0]

    # Find out-of-bounds flags
    min_zvals = np.nanmin(struct_zvals, axis=1)[:, np.newaxis]  # Shape: (num_test_structures, 1)
    max_zvals = np.nanmax(struct_zvals, axis=1)[:, np.newaxis]  # Shape: (num_test_structures, 1)

    out_of_bounds_flag = ((test_z_coords < min_zvals) | (test_z_coords > max_zvals)).astype(np.float32)

    # Construct final result array (fast with NumPy stacking)
    nearest_zslice_index_and_values_3d_arr = np.stack([
        test_struct_to_relative_struct_1d_mapping_array[:, np.newaxis] * np.ones((1, num_points_per_trial), dtype=np.float32),
        nearest_zslice_indices.astype(np.float32),
        nearest_zslice_vals,
        out_of_bounds_flag
    ], axis=-1)

    return nearest_zslice_index_and_values_3d_arr



def nearest_zslice_vals_and_indices_all_structures_2d_point_arr(relative_structures_list_of_zvals_1d_arrays, points_to_test_2d_arr, points_to_test_indices_arr, test_struct_to_relative_struct_1d_mapping_array):
    """
    Find the nearest z values and associated indices for all test points across all assoicated structures using NumPy.

    Parameters:
    - relative_structures_list_of_zvals_1d_arrays: List of z value 1d arrays for each structure.
    - points_to_test_2d_arr: 2D array of points to test.
    - points_to_test_indices_arr: 2D array of indices that separate the points to test for each structure.
    - test_struct_to_relative_struct_1d_mapping_array: 1D array that maps each test structure to its associated relative structure.

    Returns:
    - nearest_zslice_index_and_values_2d_arr: 2D array of the relative structure index, their indices for each test point and nearest z values.

    Note: the points_to_test_indices_arr is a 2D array where each row contains the start and end indices of the points to test for each structure, are used to separate the nearest_zslice_index_and_values_2d_arr sections for each point for each structure.
    """
    #num_structures = len(relative_structures_list_of_zvals_1d_arrays)
    total_num_points = points_to_test_2d_arr.shape[0]
    # Initialize array to store the results
    nearest_zslice_index_and_values_2d_arr = np.empty((total_num_points, 4), dtype=np.float32) 
    
    for test_structure_index, relative_structure_index in enumerate(test_struct_to_relative_struct_1d_mapping_array):
        zvals_array = relative_structures_list_of_zvals_1d_arrays[relative_structure_index]
        start_index = points_to_test_indices_arr[test_structure_index, 0]
        end_index = points_to_test_indices_arr[test_structure_index, 1]
        pts_to_test_z_coords = points_to_test_2d_arr[start_index:end_index, 2]
        
        # Use broadcasting to compute the absolute differences between z values
        z_diffs = np.abs(zvals_array[:, np.newaxis] - pts_to_test_z_coords[np.newaxis, :])
        
        # Find the indices of the minimum differences
        nearest_zslice_indices = np.argmin(z_diffs, axis=0)
        
        # Get the nearest z values using the indices
        nearest_zslice_vals = zvals_array[nearest_zslice_indices]
        
        # Check if the points to test z values are outside the range of the relative structure's z values
        min_zval = np.min(zvals_array)
        max_zval = np.max(zvals_array)
        
        outside_min_mask = pts_to_test_z_coords < min_zval
        outside_max_mask = pts_to_test_z_coords > max_zval
        
        # Set the indices and values to NaN where the biopsy z values are outside the range
        #nearest_zslice_indices = np.where(outside_min_mask | outside_max_mask, np.nan, nearest_zslice_indices)
        #nearest_zslice_vals = np.where(outside_min_mask | outside_max_mask, np.nan, nearest_zslice_vals)

        # Create an out-of-bounds flag (1 if out of bounds, 0 otherwise)
        out_of_bounds_flag = np.where(outside_min_mask | outside_max_mask, 1, 0)
        
        # Store the results
        nearest_zslice_index_and_values_2d_arr[start_index:end_index, 0] = relative_structure_index
        nearest_zslice_index_and_values_2d_arr[start_index:end_index, 1] = nearest_zslice_indices
        nearest_zslice_index_and_values_2d_arr[start_index:end_index, 2] = nearest_zslice_vals
        nearest_zslice_index_and_values_2d_arr[start_index:end_index, 3] = out_of_bounds_flag
    
    return nearest_zslice_index_and_values_2d_arr





def remove_consecutive_duplicate_points_numpy(slice_arr):
    """
    Remove consecutive duplicate points from a 2D NumPy array.
    
    Parameters:
    - slice_arr: 2D NumPy array of 3D points. Shape is (N, 3).
    
    Returns:
    - unique_slice_arr: 2D NumPy array of unique points.
    """
    # Compute the differences between consecutive points
    diffs = np.diff(slice_arr, axis=0)
    
    # Find the indices of non-zero differences
    non_zero_indices = np.any(diffs != 0, axis=1)
    
    # Include the last point
    non_zero_indices = np.hstack((non_zero_indices, [True]))
    
    # Extract the unique points
    unique_slice_arr = slice_arr[non_zero_indices]
    
    return unique_slice_arr




# extremely slow
def create_end_caps_for_zslices(zslices_list, maximum_point_distance):
    """
    Creates end caps for the first and last Z slices in the given zslices_list.

    Parameters:
        zslices_list (list of np.ndarray): List of N_j x 3 arrays where each element is a constant Z slice.
        maximum_point_distance (float): The maximum allowed distance between points for the grid spacing.

    Returns:
        new_zslices_list (list of np.ndarray): A new zslices list where the first and last slices have end caps.
    """
    if not zslices_list:
        raise ValueError("The input zslices_list is empty.")
    
    grid_spacing = maximum_point_distance / np.sqrt(2)

    def create_fill_points(threeDdata_zslice):
        """
        Generates fill points within the polygon formed by the given 3D Z slice.

        Parameters:
            threeDdata_zslice (np.ndarray): A single slice of N x 3 points where Z is constant.

        Returns:
            np.ndarray: The fill points as an array (M, 3).
        """
        z_val = threeDdata_zslice[0, 2]
        min_x, min_y = np.amin(threeDdata_zslice[:, 0:2], axis=0)
        max_x, max_y = np.amax(threeDdata_zslice[:, 0:2], axis=0)

        # Create a grid of candidate points
        fill_points_xy_grid_arr = np.mgrid[
            min_x - grid_spacing : max_x + grid_spacing : grid_spacing,
            min_y - grid_spacing : max_y + grid_spacing : grid_spacing
        ].reshape(2, -1).T

        # Convert to 3D by adding the constant Z value
        fill_points_xyz_grid_arr = np.empty((len(fill_points_xy_grid_arr), 3), dtype=float)
        fill_points_xyz_grid_arr[:, 0:2] = fill_points_xy_grid_arr
        fill_points_xyz_grid_arr[:, 2] = z_val

        # Create a polygon from the original slice
        zslice_polygon_shapely = Polygon(threeDdata_zslice[:, 0:2])

        # Filter points that are inside the polygon
        valid_fill_points = np.array(
            [pt for pt in fill_points_xyz_grid_arr if Point(pt[:2]).within(zslice_polygon_shapely)]
        )

        return valid_fill_points

    # Generate the end caps
    first_end_cap = create_fill_points(zslices_list[0])
    last_end_cap = create_fill_points(zslices_list[-1])

    # Construct the new zslices list
    new_zslices_list = [first_end_cap] + zslices_list + [last_end_cap]

    return new_zslices_list





def create_end_caps_for_zslices_ver2(all_structures_zslices_list, 
                                     grid_factor = 0.1, 
                                     kernel_type = "one_to_one_pip_kernel_advanced_reparameterized_version_gpu_memory_performance_optimized"):
    """
    Fast version: builds a grid and uses the GPU PIP kernel to generate end caps.
    """
    if not all_structures_zslices_list:
        raise ValueError("The input zslices_list is empty.")

    # Build candidate 3d arr
    # Note: each slice is a 2D array of points, with the first candidate end cap stacked on top of the last candidate end cap
    all_structures_first_and_last_zslices_only = [[zslices_list[0], zslices_list[-1]] for zslices_list in all_structures_zslices_list]  # just the first and last slice
    end_caps_candidates_3d_arr, num_max_end_cap_candidate_pts = get_fill_candidates_max_3d_arr(all_structures_first_and_last_zslices_only, grid_factor)
    
    num_structures = len(all_structures_zslices_list)

    # Use your fast CUDA-based PIP function
    mapping_array = np.arange(0, num_structures, dtype=np.int32)

    containment_results_cp_arr, _ = custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p.custom_point_containment_mother_function(
        all_structures_first_and_last_zslices_only,
        end_caps_candidates_3d_arr,
        mapping_array,
        constant_z_slice_polygons_handler_option='auto-close-if-open',
        remove_consecutive_duplicate_points_in_polygons=True,
        log_sub_dirs_list=[],
        log_file_name=None,
        include_edges_in_log=False,
        kernel_type=kernel_type
    )

    all_structures_zslices_list_filled_end_caps = []
    for structure_idx, sp_structure_containment_result_row in enumerate(containment_results_cp_arr):
        first_end_cap_candidates = end_caps_candidates_3d_arr[structure_idx][0:num_max_end_cap_candidate_pts]
        first_end_cap_valid_pts = first_end_cap_candidates[sp_structure_containment_result_row[0:num_max_end_cap_candidate_pts].get()]

        last_end_cap_candidates = end_caps_candidates_3d_arr[structure_idx][num_max_end_cap_candidate_pts:]
        last_end_cap_valid_pts = last_end_cap_candidates[sp_structure_containment_result_row[num_max_end_cap_candidate_pts:].get()]

        sp_structure_zslices_list_filled_end_caps = [first_end_cap_valid_pts] + all_structures_zslices_list[structure_idx] + [last_end_cap_valid_pts]
        all_structures_zslices_list_filled_end_caps.append(sp_structure_zslices_list_filled_end_caps)

    return all_structures_zslices_list_filled_end_caps

def get_max_fill_candidates(all_structures_zslices_list):
    """
    Get the maximum bounding box for the end cap fill candidates.
    """
    min_x, min_y, max_x, max_y = np.inf, np.inf, -np.inf, -np.inf
    for structure_zslices_list in all_structures_zslices_list:
        first_slice = structure_zslices_list[0]
        last_slice = structure_zslices_list[-1]

        min_x_first, min_y_first = np.amin(first_slice[:, 0:2], axis=0)
        max_x_first, max_y_first = np.amax(first_slice[:, 0:2], axis=0)
        
        min_x_last, min_y_last = np.amin(last_slice[:, 0:2], axis=0)
        max_x_last, max_y_last = np.amax(last_slice[:, 0:2], axis=0)

        min_x = min(min_x, min_x_first, min_x_last)
        min_y = min(min_y, min_y_first, min_y_last)
        max_x = max(max_x, max_x_first, max_x_last)
        max_y = max(max_y, max_y_first, max_y_last)

    return min_x, min_y, max_x, max_y


def get_fill_candidates_max_3d_arr(all_structures_zslices_list, grid_factor = 0.1):
    min_x, min_y, max_x, max_y = get_max_fill_candidates(all_structures_zslices_list)

    diameter = np.sqrt((max_x - min_x)**2 + (max_y - min_y)**2)

    maximum_point_distance = grid_factor * diameter
    grid_spacing = maximum_point_distance / np.sqrt(2)


    # build grid
    xx, yy = np.meshgrid(
        np.arange(min_x - grid_spacing, max_x + grid_spacing, grid_spacing),
        np.arange(min_y - grid_spacing, max_y + grid_spacing, grid_spacing)
    )

    num_structures = len(all_structures_zslices_list)

    num_max_end_cap_candidate_pts = xx.size

    end_caps_candidates_3d_arr = np.empty((num_structures, 2 * num_max_end_cap_candidate_pts, 3), dtype=float)
    for i, structure_zslices_list in enumerate(all_structures_zslices_list):
        first_slice = structure_zslices_list[0]
        last_slice = structure_zslices_list[-1]

        first_slice_zval = first_slice[0, 2]
        last_slice_zval = last_slice[0, 2]

        
        candidates_xy = np.column_stack((xx.ravel(), yy.ravel()))
        candidates_xyz_first = np.column_stack((candidates_xy, np.full(len(candidates_xy), first_slice_zval)))
        candidates_xyz_last = np.column_stack((candidates_xy, np.full(len(candidates_xy), last_slice_zval)))

        candidates_xyz = np.vstack((candidates_xyz_first, candidates_xyz_last))

        end_caps_candidates_3d_arr[i, :, :] = candidates_xyz


    return end_caps_candidates_3d_arr, num_max_end_cap_candidate_pts


"""
def get_fill_candidates(zslice_3d_arr, grid_spacing):
    z_val = zslice_3d_arr[0, 2]
    min_x, min_y = np.amin(zslice_3d_arr[:, 0:2], axis=0)
    max_x, max_y = np.amax(zslice_3d_arr[:, 0:2], axis=0)

    # Build grid
    xx, yy = np.meshgrid(
        np.arange(min_x - grid_spacing, max_x + grid_spacing, grid_spacing),
        np.arange(min_y - grid_spacing, max_y + grid_spacing, grid_spacing)
    )
    candidates_xy = np.column_stack((xx.ravel(), yy.ravel()))
    candidates_xyz = np.column_stack((candidates_xy, np.full(len(candidates_xy), z_val)))
    return candidates_xyz
"""














def example():
    load_example_data_bool = True
    use_which_prostate_test = 'non_interp'  # 'interp' or 'non_interp'

    if load_example_data_bool:
        file_path_parent_folder = pathlib.Path(__file__).parents[0]
        if use_which_prostate_test == 'interp':
            test_prostate_file = file_path_parent_folder.joinpath('test_prostate_interslice_interp.npy')
        elif use_which_prostate_test == 'non_interp':
            test_prostate_file = file_path_parent_folder.joinpath('test_non_interp_prostate.npy')
        
        non_bx_struct_zslices_list = list(np.load(test_prostate_file, allow_pickle=True))

        test_dilations_file = file_path_parent_folder.joinpath('test_dilations.npy')
        dilation_distances = np.load(test_dilations_file, allow_pickle=True)
    else:
        non_bx_struct_zslices_list = [
            np.array([(0, 0, 0), (1, 0, 0), (1.5, 0.5, 0), (1, 1, 0), (0.5, 1.5, 0), (0, 1, 0), (-0.5, 0.5, 0), (0, 0, 0)]),
            np.array([(0, 0, 1), (1, 0, 1), (1.5, 0.5, 1), (1, 1, 1), (0.5, 1.5, 1), (0, 1, 1), (-0.5, 0.5, 1), (0, 0, 1)]),
            np.array([(0, 0, 2), (1, 0, 2), (1.5, 0.5, 2), (1, 1, 2), (0.5, 1.5, 2), (0, 1, 2), (-0.5, 0.5, 2), (0, 0, 2)]),
            np.array([(0, 0, 3), (1, 0, 3), (1.5, 0.5, 3), (1, 1, 3), (0.5, 1.5, 3), (0, 1, 3), (-0.5, 0.5, 3), (0, 0, 3)]),
            np.array([(0, 0, 4), (1, 0, 4), (1.5, 0.5, 4), (1, 1, 4), (0.5, 1.5, 4), (0, 1, 4), (-0.5, 0.5, 4), (0, 0, 4)]),
            np.array([(0, 0, 5), (1, 0, 5), (1.5, 0.5, 5), (1, 1, 5), (0.5, 1.5, 5), (0, 1, 5), (-0.5, 0.5, 5), (0, 0, 5)])
        ]

        # Example dilation distances for 3 trials
        dilation_distances = np.array([
            [-2, -0.5],
            [0.2, 0.5],
            [0.2, -0.6],
            [-0.2, -0.6],
            [-2, -0.5],
            [0.3, 1]
        ])

    show_z_dilation_bool = False
    show_xy_dilation_bool = False

    # Convert to 2D array and indices
    points_array, indices_array = convert_to_2d_array_and_indices_numpy(non_bx_struct_zslices_list)

    # Generate dilated structures # 

    dilation_distances = np.tile(dilation_distances, (100, 1))

    # Parallelized
    cpu_count = os.cpu_count()
    with multiprocess.Pool(cpu_count) as parallel_pool:
        st = time.time()
        dilated_structures_list, dilated_structures_slices_indices_list = generate_dilated_structures_parallelized(points_array, indices_array, dilation_distances, show_z_dilation_bool, show_xy_dilation_bool, parallel_pool)
        et = time.time()
        print("\n🔹 Parallelized time:", et-st)

    # Non-parallelized
    #st = time.time()
    #dilated_structures_list, dilated_structures_slices_indices_list = generate_dilated_structures(points_array, indices_array, dilation_distances[0:10], show_z_dilation_bool, show_xy_dilation_bool)
    #et = time.time()
    #print("\n🔹 Non-parallelized time:", et-st)




    # Extract constant Z values
    z_values_list = extract_constant_z_values(dilated_structures_list, dilated_structures_slices_indices_list)
    print("test")

def profile_example():
    pr = cProfile.Profile()
    pr.enable()
    example()
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
    print('test')

if __name__ == "__main__":
    #example()
    profile_example()

import matplotlib.pyplot as plt
from shapely.geometry import MultiPolygon

def plot_multipolygon(multipolygon):
    # Plot the MultiPolygon
    fig, ax = plt.subplots(figsize=(6, 6))

    for polygon in multipolygon.geoms:  # Iterate over individual polygons
        x, y = polygon.exterior.xy
        ax.plot(x, y, 'b-')  # Plot the boundary
        ax.fill(x, y, 'b', alpha=0.3)  # Fill with color

    # Adjust plot settings
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Shapely MultiPolygon")
    ax.set_aspect('equal')  # Keep aspect ratio square
    plt.grid(True)
    plt.show()






