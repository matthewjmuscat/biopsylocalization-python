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

def dilate_polygons_z_direction(polygons_list, dilation_distance_z, show_z_dilation_bool = False):
    """
    Dilate the Z slices by adjusting the Z coordinates using provided distance.
    
    Parameters:
    - polygons_list: List of polygons with their Z coordinates.
    - dilation_distance_z: Distance by which to dilate the Z coordinates.
    
    Returns:
    - dilated_polygons_list: List of dilated polygons with adjusted Z coordinates.
    """
    # Extract Z coordinates
    z_coords = cp.array([polygon[0][2] for polygon in polygons_list])
    central_z = cp.mean(z_coords)
    
    # Adjust Z coordinates
    dilated_polygons_list = []
    for polygon in polygons_list:
        original_z = polygon[0][2]
        new_z = float(original_z + cp.sign(original_z - central_z).item() * dilation_distance_z)
        
        # Skip slices that move past or equal to the central slice, except those originally on the central value
        if (cp.sign(original_z - central_z) * (new_z - central_z)) <= 0 and original_z != central_z:
            continue
        
        dilated_polygon = [[point[0], point[1], new_z] for point in polygon]
        dilated_polygons_list.append(dilated_polygon)
    
    if show_z_dilation_bool:
        original_pcd = point_containment_tools.create_point_cloud(cp.asnumpy(cp.array([point for polygon in polygons_list for point in polygon])), color=np.array([0, 0, 1]))  # paint tested structure in blue
        dilated_pcd = point_containment_tools.create_point_cloud(cp.asnumpy(cp.array([point for polygon in dilated_polygons_list for point in polygon])), color=np.array([1, 0, 0]))  # paint tested structure in red

        plotting_funcs.plot_geometries(original_pcd, dilated_pcd)

    return dilated_polygons_list

def dilate_polygons_z_direction_faster(polygons_list, dilation_distance_z, show_z_dilation_bool = False):
    """
    Dilate the Z slices by adjusting the Z coordinates using provided distance.
    
    Parameters:
    - polygons_list: List of polygons with their Z coordinates.
    - dilation_distance_z: Distance by which to dilate the Z coordinates.
    
    Returns:
    - dilated_polygons_list: List of dilated polygons with adjusted Z coordinates.
    """
    # Convert polygons_list to a CuPy array
    polygons_cp = cp.array(polygons_list)
    
    # Extract Z coordinates
    z_coords = polygons_cp[:, 0, 2]
    central_z = cp.mean(z_coords)
    
    # Calculate new Z coordinates
    z_signs = cp.sign(z_coords - central_z)
    new_z_coords = z_coords + z_signs * dilation_distance_z
    
    # Create a mask to filter out slices that move past or equal to the central slice
    mask = (z_signs * (new_z_coords - central_z) > 0) | (z_coords == central_z)
    
    # Apply the mask to filter polygons
    filtered_polygons_cp = polygons_cp[mask]
    new_z_coords_filtered = new_z_coords[mask]
    
    # Update Z coordinates in the filtered polygons
    filtered_polygons_cp[:, :, 2] = new_z_coords_filtered[:, cp.newaxis]
    
    # Convert back to a list of arrays
    # dilated_polygons_list = [cp.asnumpy(polygon) for polygon in filtered_polygons_cp] # dont think we need to do this
    dilated_polygons_array = cp.asnumpy(filtered_polygons_cp)
    
    if show_z_dilation_bool:
        original_pcd = point_containment_tools.create_point_cloud(cp.asnumpy(polygons_cp.reshape(-1, 3)), color=np.array([0, 0, 1]))  # paint tested structure in blue
        dilated_pcd = point_containment_tools.create_point_cloud(cp.asnumpy(filtered_polygons_cp.reshape(-1, 3)), color=np.array([1, 0, 0]))  # paint tested structure in red

        plotting_funcs.plot_geometries(original_pcd, dilated_pcd)

    return dilated_polygons_array

def dilate_one_polygon(polygon, dilation_distance_xy, min_area):
    shapely_polygon = Polygon([(point[0], point[1]) for point in polygon])
    
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
    
    dilated_polygon = [[point[0], point[1], polygon[0][2]] for point in list(dilated_shapely_polygon.exterior.coords)]
    return dilated_polygon

def dilate_polygons_xy_plane_parallelized(polygons_list, dilation_distance_xy, original_polygons_list, show_xy_dilation_bool, parallel_pool, min_area=1e-6):
    """
    Dilate the polygons in the XY plane using provided distance.
    
    Parameters:
    - polygons_list: List of polygons with their Z coordinates.
    - dilation_distance_xy: Distance by which to dilate the XY coordinates.
    - min_area: Minimum area to ensure the polygon is not deleted.
    
    Returns:
    - dilated_polygons_list: List of dilated polygons in the XY plane.
    """
    
    dilated_polygons_list = parallel_pool.starmap(dilate_one_polygon, [(polygon, dilation_distance_xy, min_area) for polygon in polygons_list])
    
    if show_xy_dilation_bool:
        original_pcd = point_containment_tools.create_point_cloud(np.array([point for polygon in original_polygons_list for point in polygon]), color=np.array([0, 0, 1]))  # paint tested structure in blue
        z_dilation_pcd = point_containment_tools.create_point_cloud(np.array([point for polygon in polygons_list for point in polygon]), color=np.array([1, 0, 0]))  # paint tested structure in red
        xyz_dilated_pcd = point_containment_tools.create_point_cloud(np.array([point for polygon in dilated_polygons_list for point in polygon]), color=np.array([0, 1, 0]))  # paint tested structure in green

        plotting_funcs.plot_geometries(original_pcd, z_dilation_pcd, xyz_dilated_pcd)

    return dilated_polygons_list

def dilate_polygons_xy_plane(polygons_list, dilation_distance_xy, original_polygons_list, show_xy_dilation_bool, min_area=1e-6):
    """
    Dilate the polygons in the XY plane using provided distance.
    
    Parameters:
    - polygons_list: List of polygons with their Z coordinates.
    - dilation_distance_xy: Distance by which to dilate the XY coordinates.
    - min_area: Minimum area to ensure the polygon is not deleted.
    
    Returns:
    - dilated_polygons_list: List of dilated polygons in the XY plane.
    """
    dilated_polygons_list = []
    for polygon in polygons_list:
        shapely_polygon = Polygon([(point[0], point[1]) for point in polygon])
        
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
        
        dilated_polygon = [[point[0], point[1], polygon[0][2]] for point in list(dilated_shapely_polygon.exterior.coords)]
        dilated_polygons_list.append(dilated_polygon)
    
    if show_xy_dilation_bool:
        original_pcd = point_containment_tools.create_point_cloud(cp.asnumpy(cp.array([point for polygon in original_polygons_list for point in polygon])), color=np.array([0, 0, 1]))  # paint tested structure in blue
        z_dilation_pcd = point_containment_tools.create_point_cloud(cp.asnumpy(cp.array([point for polygon in polygons_list for point in polygon])), color=np.array([1, 0, 0]))  # paint tested structure in red
        xyz_dilated_pcd = point_containment_tools.create_point_cloud(cp.asnumpy(cp.array([point for polygon in dilated_polygons_list for point in polygon])), color=np.array([0, 1, 0]))  # paint tested structure in green

        plotting_funcs.plot_geometries(original_pcd, z_dilation_pcd, xyz_dilated_pcd)

    return dilated_polygons_list

def dilate_polygons(polygons_list, dilation_distance_z, dilation_distance_xy, show_z_dilation_bool, show_xy_dilation_bool, parallel_pool = None):
    """
    Dilate the polygons in both Z direction and XY plane using provided distances.
    
    Parameters:
    - polygons_list: List of polygons with their Z coordinates.
    - dilation_distance_z: Distance by which to dilate the Z coordinates.
    - dilation_distance_xy: Distance by which to dilate the XY coordinates.
    
    Returns:
    - dilated_polygons_list: List of dilated polygons.
    """
    # Dilate Z slices
    dilated_polygons_z = dilate_polygons_z_direction_faster(polygons_list, dilation_distance_z, show_z_dilation_bool)
    
    #dilated_polygons_z_slower = dilate_polygons_z_direction(polygons_list, dilation_distance_z, show_z_dilation_bool)
    #are_equal = compare_nested_lists(dilated_polygons_z, dilated_polygons_z_slower)
    #print("\n🔹 Are the z-direction functions results equal?", are_equal)

    
    # Dilate XY plane
    if parallel_pool == None:
        dilated_polygons_xy = dilate_polygons_xy_plane(dilated_polygons_z, dilation_distance_xy, polygons_list, show_xy_dilation_bool)
    else:
        dilated_polygons_xy = dilate_polygons_xy_plane_parallelized(dilated_polygons_z, dilation_distance_xy, polygons_list, show_xy_dilation_bool, parallel_pool)
    
    return dilated_polygons_xy

def generate_dilated_structures(non_bx_struct_zslices_list, dilation_distances, show_z_dilation_bool, show_xy_dilation_bool, parallel_pool = None):
    """
    Generate a list of dilated structures for each trial.
    
    Parameters:
    - non_bx_struct_zslices_list: List of original Z slices.
    - dilation_distances: 2D array of distances by which to dilate the XY and Z coordinates for each trial.
    
    Returns:
    - dilated_structures_list: List of dilated structures.
    """
    num_trials = dilation_distances.shape[0]
    dilated_structures_list = []
    
    for trial in range(num_trials):
        dilation_distance_z = dilation_distances[trial, 1]
        dilation_distance_xy = dilation_distances[trial, 0]
        if parallel_pool == None:
            dilated_polygons = dilate_polygons(non_bx_struct_zslices_list, dilation_distance_z, dilation_distance_xy, show_z_dilation_bool, show_xy_dilation_bool)
        else:
            dilated_polygons = dilate_polygons(non_bx_struct_zslices_list, dilation_distance_z, dilation_distance_xy, show_z_dilation_bool, show_xy_dilation_bool, parallel_pool)
        dilated_polygons = [np.array(polygon_slice) for polygon_slice in dilated_polygons]

        dilated_structures_list.append(dilated_polygons)
    
    return dilated_structures_list


def extract_constant_z_values(dilated_structures_4dlist):
    z_values_list = []
    for trial in dilated_structures_4dlist:
        trial_z_values = []
        for structure in trial:
            z_value = structure[0][2]  # Assuming the z value is constant for all points in the structure
            trial_z_values.append(z_value)
        z_values_list.append(trial_z_values)
    return z_values_list


def compare_nested_lists(list1, list2):
    if isinstance(list1, list) and isinstance(list2, list):
        if len(list1) != len(list2):
            return False
        return all(compare_nested_lists(sublist1, sublist2) for sublist1, sublist2 in zip(list1, list2))
    else:
        return np.array_equal(list1, list2)




def convert_to_2d_array_and_indices(polygons_list):
    # Calculate the total number of points
    total_points = sum(len(polygon) for polygon in polygons_list)
    
    # Preallocate the points array
    points_array = np.empty((total_points, 3), dtype=float)
    
    # Preallocate the indices array
    indices_array = np.empty((len(polygons_list), 2), dtype=int)
    
    # Fill the points array and indices array
    current_index = 0
    for i, polygon in enumerate(polygons_list):
        num_points = len(polygon)
        points_array[current_index:current_index + num_points] = polygon
        indices_array[i] = [current_index, current_index + num_points]
        current_index += num_points
    
    return points_array, indices_array




def convert_to_2d_array_and_indices_cupy(polygons_list):
    # Calculate the total number of points
    total_points = sum(len(polygon) for polygon in polygons_list)
    
    # Preallocate the points array using CuPy
    points_array = cp.empty((total_points, 3), dtype=cp.float32)
    
    # Preallocate the indices array using CuPy
    indices_array = cp.empty(len(polygons_list), dtype=cp.int32)
    
    # Fill the points array and indices array
    current_index = 0
    for i, polygon in enumerate(polygons_list):
        num_points = len(polygon)
        points_array[current_index:current_index + num_points] = cp.array(polygon)
        current_index += num_points
        indices_array[i] = current_index
    
    return points_array, indices_array















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


    polygons_2d_arr, polygons_slices_indices_arr = convert_to_2d_array_and_indices_cupy(polygons_list)

    
    do_parallel_version_bool = False
    if do_parallel_version_bool == True:
        with Pool(cpu_count()) as parallel_pool:  # Ensure parallel_pool is properly initialized and closed
            st = time.time()
            dilated_structures_list_parallelized = generate_dilated_structures(non_bx_struct_zslices_list, dilation_distances[0:5], show_z_dilation_bool, show_xy_dilation_bool, parallel_pool)
            et = time.time()
            print("\n🔹 Parallelized time:", et-st)

    st = time.time()
    dilated_structures_list = generate_dilated_structures(non_bx_struct_zslices_list, dilation_distances[0:50], show_z_dilation_bool, show_xy_dilation_bool)
    et = time.time()
    print("\n🔹 Non-parallelized time:", et-st)

    # Compare the lists to ensure they are exactly equal
    if do_parallel_version_bool == True:

        are_equal = compare_nested_lists(dilated_structures_list_parallelized, dilated_structures_list)
        print("\n🔹 Are the parallelized and non-parallelized results equal?", are_equal)

    #input()

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