import cupy as cp
import numpy as np
from shapely.geometry import Polygon
import plotting_funcs
import point_containment_tools
from multiprocess import Pool, cpu_count
import time
import pathlib
import cProfile
import pstats
import io


def convert_to_2d_array_and_indices_cupy(polygons_list):
    # Calculate the total number of points
    total_points = sum(len(polygon) for polygon in polygons_list)
    
    # Preallocate the points array using CuPy
    points_array = cp.empty((total_points, 3), dtype=cp.float32)
    
    # Preallocate the indices array using CuPy
    indices_array = cp.empty((len(polygons_list), 2), dtype=cp.int32)
    
    # Fill the points array and indices array
    current_index = 0
    for i, polygon in enumerate(polygons_list):
        num_points = len(polygon)
        points_array[current_index:current_index + num_points] = cp.array(polygon)
        indices_array[i, 0] = current_index
        indices_array[i, 1] = current_index + num_points
        current_index += num_points
    
    return points_array, indices_array



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
    central_z = cp.mean(z_coords)
    
    # Calculate new Z coordinates
    z_signs = cp.sign(z_coords - central_z)
    new_z_coords = z_coords + z_signs * dilation_distance_z
    
    # Create a mask to filter out slices that move past or equal to the central slice
    mask = (z_signs * (new_z_coords - central_z) > 0) | (z_coords == central_z)
    
    # Apply the mask to filter points
    filtered_points_array = points_array[mask]
    new_z_coords_filtered = new_z_coords[mask]
    
    # Update Z coordinates in the filtered points
    filtered_points_array[:, 2] = new_z_coords_filtered
    
    if show_z_dilation_bool:
        original_pcd = point_containment_tools.create_point_cloud(cp.asnumpy(points_array), color=np.array([0, 0, 1]))  # paint tested structure in blue
        dilated_pcd = point_containment_tools.create_point_cloud(cp.asnumpy(filtered_points_array), color=np.array([1, 0, 0]))  # paint tested structure in red
        plotting_funcs.plot_geometries(original_pcd, dilated_pcd)
    
    # Initialize empty array for new indices
    new_indices_array = cp.empty((0, 2), dtype=cp.int32)
    current_index = 0
    
    for start_index, end_index in indices_array:
        num_points = cp.sum(mask[start_index:end_index]).item()
        if num_points > 0:
            new_indices_array = cp.vstack((new_indices_array, cp.array([[current_index, current_index + num_points]], dtype=cp.int32)))
            current_index += num_points
    
    return filtered_points_array, new_indices_array


def dilate_polygons_xy_plane(points_array_org, points_array, indices_array, dilation_distance_xy, show_xy_dilation_bool, min_area=1e-6):
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
    dilated_points_array = cp.empty((0, 3), dtype=cp.float32)
    new_indices_array = cp.empty((0, 2), dtype=cp.int32)

    for start_index, end_index in indices_array:
        polygon = points_array[start_index:end_index]
        shapely_polygon = Polygon(polygon[:, :2].get())

        # Calculate the maximum inward dilation distance
        min_inward_dilation = -shapely_polygon.minimum_clearance
        
        # Apply the dilation
        dilated_shapely_polygon = shapely_polygon.buffer(
            float(dilation_distance_xy), 
            quad_segs=16, 
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
                    quad_segs=16, 
                    cap_style='square', 
                    join_style='mitre'
                )
        
        # Handle the case where the result is a MultiPolygon
        if dilated_shapely_polygon.geom_type == 'MultiPolygon':
            largest_polygon = max(dilated_shapely_polygon.geoms, key=lambda p: p.area)
            dilated_shapely_polygon = largest_polygon

        # Convert the dilated polygon back to a CuPy array
        dilated_polygon_coords = cp.array(dilated_shapely_polygon.exterior.coords)
        dilated_polygon = cp.hstack((dilated_polygon_coords, cp.full((dilated_polygon_coords.shape[0], 1), polygon[0, 2])))
        
        num_dilated_points = dilated_polygon.shape[0]
        
        # Stack the new dilated points to the dilated_points_array
        dilated_points_array = cp.vstack((dilated_points_array, dilated_polygon))
        
        # Stack the new indices to the new_indices_array
        new_indices_array = cp.vstack((new_indices_array, cp.array([[current_index, current_index + num_dilated_points]], dtype=cp.int32)))
        
        current_index += num_dilated_points
    
    if show_xy_dilation_bool:
        original_pcd = point_containment_tools.create_point_cloud(cp.asnumpy(points_array_org), color=np.array([0, 0, 1]))  # paint tested structure in blue
        dilated_z_pcd = point_containment_tools.create_point_cloud(cp.asnumpy(points_array), color=np.array([1, 0, 0]))  # paint tested structure in red
        dilated_total_pcd = point_containment_tools.create_point_cloud(cp.asnumpy(dilated_points_array), color=np.array([0, 1, 0]))  # paint tested structure in green
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


def extract_constant_z_values(dilated_structures_list, indices_array):
    z_values_list = []
    for trial in dilated_structures_list:
        trial_z_values = []
        for start_index, end_index in indices_array:
            z_value = trial[start_index][2]  # Assuming the z value is constant for all points in the structure
            trial_z_values.append(z_value)
        z_values_list.append(trial_z_values)
    return z_values_list


# This function although vectorized nicely, assumes that each dilated structure has the same number of slices, which is not necessarily true
def nearest_zslice_vals_and_indices_all_trials_cupy_vectorized(non_bx_struct_nominal_and_all_dilations_zvals_list, combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff):
    """
    Find the nearest z values and associated indices for all biopsy points across all trials using CuPy.
    
    Parameters:
    - non_bx_struct_nominal_and_all_dilations_zvals_list: List of z values for each trial of the dilated relative structure.
    - combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff: 3D array of biopsy points for all trials.
    
    Returns:
    - grand_all_dilations_sp_trial_nearest_interpolated_zslice_index_array: Array of indices of the nearest z values for each biopsy point.
    - grand_all_dilation_sp_trial_nearest_interpolated_zslice_vals_array: Array of the nearest z values for each biopsy point.
    """
    num_trials = len(non_bx_struct_nominal_and_all_dilations_zvals_list)
    num_points_per_trial = combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff.shape[1]
    
    # Convert the list of z values to a 2D CuPy array for broadcasting
    zvals_array = cp.array(non_bx_struct_nominal_and_all_dilations_zvals_list)
    
    # Extract the z coordinates of the biopsy points
    biopsy_z_coords = cp.array(combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff[:, :, 2])
    
    # Use broadcasting to compute the absolute differences between z values
    z_diffs = cp.abs(zvals_array[:, :, cp.newaxis] - biopsy_z_coords[cp.newaxis, :, :])
    
    # Find the indices of the minimum differences
    nearest_zslice_indices = cp.argmin(z_diffs, axis=1)
    
    # Get the nearest z values using the indices
    nearest_zslice_vals = cp.take_along_axis(zvals_array, nearest_zslice_indices, axis=1)
    
    # Check if the biopsy z values are outside the range of the relative structure's z values
    min_zvals = cp.min(zvals_array, axis=1)[:, cp.newaxis]
    max_zvals = cp.max(zvals_array, axis=1)[:, cp.newaxis]
    
    outside_min_mask = biopsy_z_coords < min_zvals
    outside_max_mask = biopsy_z_coords > max_zvals
    
    # Set the indices and values to NaN where the biopsy z values are outside the range
    nearest_zslice_indices = cp.where(outside_min_mask | outside_max_mask, cp.nan, nearest_zslice_indices)
    nearest_zslice_vals = cp.where(outside_min_mask | outside_max_mask, cp.nan, nearest_zslice_vals)
    
    # Flatten the results to match the expected output format
    grand_all_dilations_sp_trial_nearest_interpolated_zslice_index_array = nearest_zslice_indices.flatten()
    grand_all_dilation_sp_trial_nearest_interpolated_zslice_vals_array = nearest_zslice_vals.flatten()
    
    return grand_all_dilations_sp_trial_nearest_interpolated_zslice_index_array, grand_all_dilation_sp_trial_nearest_interpolated_zslice_vals_array





def nearest_zslice_vals_and_indices_all_trials_cupy(non_bx_struct_nominal_and_all_dilations_zvals_list, combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff):
    """
    Find the nearest z values and associated indices for all biopsy points across all trials using CuPy.
    
    Parameters:
    - non_bx_struct_nominal_and_all_dilations_zvals_list: List of z values for each trial of the dilated relative structure.
    - combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff: 3D array of biopsy points for all trials.
    
    Returns:
    - grand_all_dilations_sp_trial_nearest_interpolated_zslice_array: Array of the nearest z values and their indices for each biopsy point.
    """
    num_trials = len(non_bx_struct_nominal_and_all_dilations_zvals_list)
    num_points_per_trial = combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff.shape[1]
    
    # Initialize array to store the results
    grand_all_dilations_sp_trial_nearest_interpolated_zslice_array = cp.empty((num_trials, num_points_per_trial, 3), dtype=cp.float32)
    
    for trial_index in range(num_trials):
        zvals_array = cp.array(non_bx_struct_nominal_and_all_dilations_zvals_list[trial_index])
        biopsy_z_coords = cp.array(combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff[trial_index, :, 2])
        
        # Use broadcasting to compute the absolute differences between z values
        z_diffs = cp.abs(zvals_array[:, cp.newaxis] - biopsy_z_coords[cp.newaxis, :])
        
        # Find the indices of the minimum differences
        nearest_zslice_indices = cp.argmin(z_diffs, axis=0)
        
        # Get the nearest z values using the indices
        nearest_zslice_vals = zvals_array[nearest_zslice_indices]
        
        # Check if the biopsy z values are outside the range of the relative structure's z values
        min_zval = cp.min(zvals_array)
        max_zval = cp.max(zvals_array)
        
        outside_min_mask = biopsy_z_coords < min_zval
        outside_max_mask = biopsy_z_coords > max_zval
        
        # Set the indices and values to NaN where the biopsy z values are outside the range
        nearest_zslice_indices = cp.where(outside_min_mask | outside_max_mask, cp.nan, nearest_zslice_indices)
        nearest_zslice_vals = cp.where(outside_min_mask | outside_max_mask, cp.nan, nearest_zslice_vals)
        
        # Store the results
        grand_all_dilations_sp_trial_nearest_interpolated_zslice_array[trial_index, :, 0] = trial_index
        grand_all_dilations_sp_trial_nearest_interpolated_zslice_array[trial_index, :, 1] = nearest_zslice_indices
        grand_all_dilations_sp_trial_nearest_interpolated_zslice_array[trial_index, :, 2] = nearest_zslice_vals
    
    return grand_all_dilations_sp_trial_nearest_interpolated_zslice_array



def example():
    load_example_data_bool = True

    if load_example_data_bool:
        file_path_parent_folder = pathlib.Path(__file__).parents[0]
        test_prostate_file = file_path_parent_folder.joinpath('test_prostate_interslice_interp.npy')
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
    points_array, indices_array = convert_to_2d_array_and_indices_cupy(non_bx_struct_zslices_list)

    # Generate dilated structures
    st = time.time()
    dilated_structures_list, dilated_structures_slices_indices_list = generate_dilated_structures(points_array, indices_array, dilation_distances[0:5], show_z_dilation_bool, show_xy_dilation_bool)
    et = time.time()
    print("\n🔹 Non-parallelized time:", et-st)

    # Extract constant Z values
    z_values_list = extract_constant_z_values(dilated_structures_list, dilated_structures_slices_indices_list)
    print("\n🔹 Z values list:", z_values_list)




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


