import cupy as cp
import cuspatial
import cudf
import geopandas as gpd
from shapely.geometry import Polygon, Point
import numpy as np
import time
import pandas as pd
import cProfile
import pstats
import io
import matplotlib.pyplot as plt
import struct
import pathlib


# -------------------------------
# 🔹 Fixed CUDA Kernel
# -------------------------------
one_to_one_pip_kernel = cp.RawKernel(r'''
extern "C" __global__
void one_to_one_pip(const double* px, const double* py,
                    const double* poly_x, const double* poly_y,
                    const long long int* poly_part_offsets, // ✅ FIX: Use long long int*
                    int* results, int num_points) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points) return; // Prevent out-of-bounds execution

    double x = px[i];
    double y = py[i];

    // ✅ Validate offsets before using
    long long ring_start = poly_part_offsets[i];  // ✅ Correct type
    long long ring_end = poly_part_offsets[i + 1];

    printf("[Thread %d] ✅ ring_start=%lld, ring_end=%lld for Point (%.2f, %.2f)\n", i, ring_start, ring_end, x, y);

    if (ring_end <= ring_start || ring_start < 0) {
        printf("[Thread %d] ❌ ERROR: Invalid ring indices: ring_start=%lld, ring_end=%lld\n", i, ring_start, ring_end);
        results[i] = 0;
        return;
    }

    // 🔹 Point-in-Polygon Test
    bool inside = false;
    for (long long j = ring_start, k = ring_end - 1; j < ring_end; k = j++) {
        double xj = poly_x[j], yj = poly_y[j];
        double xk = poly_x[k], yk = poly_y[k];

        printf("[Thread %d] Edge from (%.2f, %.2f) to (%.2f, %.2f)\n", i, xj, yj, xk, yk);

        bool intersect = ((yj > y) != (yk > y)) &&
                         (x < (xk - xj) * (y - yj) / (yk - yj) + xj);

        if (intersect) {
            inside = !inside;
            printf("[Thread %d] 🔥 Intersection detected! Inside flipped to %d\n", i, inside);
        }
    }

    results[i] = inside ? 1 : 0;
    printf("[Thread %d] ✅ Final result: %d\n", i, results[i]);
}

''', 'one_to_one_pip')





one_to_one_pip_kernel_advanced = cp.RawKernel(r'''
extern "C" __global__
void one_to_one_pip(const double* px, const double* py,
                    const double* poly_x, const double* poly_y,
                    const long long int* poly_part_offsets,
                    int* results, int num_points, int poly_offsets_size,
                    long long* log_buffer) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points) return;

    // ✅ Define log entry size INSIDE the kernel
    const int LOG_ENTRY_SIZE = 7; // Each log entry has 7 values (int64)

    // ✅ Ensure valid memory access
    if (i + 1 >= poly_offsets_size) {  
        log_buffer[i * LOG_ENTRY_SIZE + 0] = i;
        log_buffer[i * LOG_ENTRY_SIZE + 1] = -1;  // Error flag for out-of-bounds
        return;
    }

    // ✅ Fetch values correctly
    double x = px[i];
    double y = py[i];

    long long ring_start = poly_part_offsets[i];
    long long ring_end = poly_part_offsets[i + 1];

    // ✅ Validate range
    if (ring_end <= ring_start || ring_start < 0) {
        log_buffer[i * LOG_ENTRY_SIZE + 0] = i;
        log_buffer[i * LOG_ENTRY_SIZE + 1] = -2;  // Invalid ring range
        return;
    }

    // ✅ Store debug values
    log_buffer[i * LOG_ENTRY_SIZE + 0] = i;              // Thread index
    log_buffer[i * LOG_ENTRY_SIZE + 1] = ring_start;
    log_buffer[i * LOG_ENTRY_SIZE + 2] = ring_end;
    log_buffer[i * LOG_ENTRY_SIZE + 3] = __double_as_longlong(x);
    log_buffer[i * LOG_ENTRY_SIZE + 4] = __double_as_longlong(y);
    log_buffer[i * LOG_ENTRY_SIZE + 5] = 0; // Inside flag before computing
    log_buffer[i * LOG_ENTRY_SIZE + 6] = 0; // Placeholder for intersection logs

    bool inside = false;
    long long j, k;
    #define EPSILON 1e-9 

    for (j = ring_start, k = ring_end - 1; j < ring_end; k = j++) {
        if (j < 0 || j >= ring_end || k < 0 || k >= ring_end) {
            results[i] = 0;
            return;
        }

        double xj = poly_x[j], yj = poly_y[j];
        double xk = poly_x[k], yk = poly_y[k];

        // ✅ Handle Ray Intersection with Edge
        if ((yj > y) != (yk > y)) {
            if (fabs(yk - yj) > EPSILON) {  
                double intersect_x = (xk - xj) * (y - yj) / (yk - yj) + xj;
                if (x < intersect_x) {
                    inside = !inside;
                    log_buffer[i * LOG_ENTRY_SIZE + 5] = inside;  // Update inside flag
                    log_buffer[i * LOG_ENTRY_SIZE + 6] = 1;       // Intersection detected
                }
            }
        } 
        // ✅ Handle Ray Exactly on Edge
        else if (fabs(y - yj) < EPSILON && fabs(y - yk) < EPSILON) {  
            if (x >= fmin(xj, xk) && x <= fmax(xj, xk)) {
                inside = true;
                log_buffer[i * LOG_ENTRY_SIZE + 5] = inside;  // Inside set to true
                log_buffer[i * LOG_ENTRY_SIZE + 6] = 2;       // Ray on edge
                break;
            }
        } 
        // ✅ Handle Ray Intersecting a Vertex
        else if (fabs(y - yj) < EPSILON || fabs(y - yk) < EPSILON) {  
            if (fabs(y - yj) < EPSILON && y < yk) {
                double intersect_x = xj;
                if (x < intersect_x) {
                    inside = !inside;
                    log_buffer[i * LOG_ENTRY_SIZE + 5] = inside;  // Flip inside
                    log_buffer[i * LOG_ENTRY_SIZE + 6] = 3;       // Vertex intersection
                }
            } else if (fabs(y - yk) < EPSILON && y < yj) {
                double intersect_x = xk;
                if (x < intersect_x) {
                    inside = !inside;
                    log_buffer[i * LOG_ENTRY_SIZE + 5] = inside;  // Flip inside
                    log_buffer[i * LOG_ENTRY_SIZE + 6] = 4;       // Vertex intersection
                }
            }
        }
    }

    results[i] = inside ? 1 : 0;
}
''', 'one_to_one_pip')








def one_to_one_point_in_polygon_geoseries_version(points_gs, polygons_gs):
    num_points = len(points_gs)

    # ✅ Extract CuPy arrays
    points_x = points_gs.points.x.to_cupy()
    points_y = points_gs.points.y.to_cupy()

    # ✅ Extract polygons properly
    polygons_gpd = polygons_gs.to_geopandas()

    poly_x = cp.concatenate([cp.array(p.exterior.xy[0]) for p in polygons_gpd], axis=0)
    poly_y = cp.concatenate([cp.array(p.exterior.xy[1]) for p in polygons_gpd], axis=0)

    # ✅ Compute proper offsets
    vertex_counts = [len(p.exterior.xy[0]) for p in polygons_gpd]
    poly_part_offsets = cp.array([0] + vertex_counts, dtype=cp.int64).cumsum()  # ✅ Fix type
    poly_part_offsets = poly_part_offsets.astype(cp.int64)  # Ensure it's explicitly CuPy int64


    # ✅ Debugging print statements
    print("\n🔹 Debugging Python Extraction")
    print("Points X:", points_x)
    print("Points Y:", points_y)
    print("Polygon X:", poly_x)
    print("Polygon Y:", poly_y)
    print("Polygon Part Offsets:", poly_part_offsets)

    # ✅ Allocate GPU memory for results
    results = cp.zeros(num_points, dtype=cp.int32)

    # ✅ Launch Kernel
    block_size = 256
    grid_size = (num_points + block_size - 1) // block_size
    one_to_one_pip_kernel_advanced((grid_size,), (block_size,), (
        points_x, points_y, poly_x, poly_y, poly_part_offsets, results, num_points
    ))

    # ✅ Retrieve results
    results_host = results.get()
    print("\n🔹 CUDA Results Host:", results_host)

    return results





def one_to_one_point_in_polygon_cupy_arr_version(points, poly_points, poly_indices, block_size=256, log_file_name="cuda_log.txt"):
    num_points = cp.int32(points.shape[0])

    points_x = points[:, 0]
    points_y = points[:, 1]
    poly_x = poly_points[:, 0]
    poly_y = poly_points[:, 1]

    poly_part_offsets = cp.zeros(poly_indices.shape[0] + 1, dtype=cp.int64)
    poly_part_offsets[1:] = cp.cumsum(poly_indices[:, 1] - poly_indices[:, 0]).astype(cp.int64)

    poly_offsets_size = cp.int32(poly_part_offsets.shape[0])

    results = cp.zeros(num_points, dtype=cp.int32)

    LOG_ENTRY_SIZE = 7  # Matches kernel definition
    log_buffer = cp.zeros(num_points * LOG_ENTRY_SIZE, dtype=cp.int64)  

    grid_size = (num_points + block_size - 1) // block_size

    one_to_one_pip_kernel_advanced(
        (grid_size,), (block_size,),
        (points_x, points_y, poly_x, poly_y, poly_part_offsets, results, num_points, poly_offsets_size, log_buffer)
    )

    # ✅ Retrieve log data
    logs_host = log_buffer.get().reshape(-1, LOG_ENTRY_SIZE)

    # ✅ Process logs in Python
    log_dir = pathlib.Path(__file__).parents[0].joinpath("cuda_containment_logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir.joinpath(log_file_name)
    with open(log_file, "w") as f:
        for log in logs_host:
            thread_id = log[0]
            ring_start = log[1]
            ring_end = log[2]
            x_coord = struct.unpack('d', struct.pack('q', log[3]))[0]
            y_coord = struct.unpack('d', struct.pack('q', log[4]))[0]
            inside_flag = log[5]
            special_case = log[6]

            if ring_start == -1:
                f.write(f"[Thread {thread_id}] ❌ ERROR: Out-of-bounds poly_part_offsets\n")
            elif ring_start == -2:
                f.write(f"[Thread {thread_id}] ❌ ERROR: Invalid ring indices\n")
            elif ring_start == -3:
                f.write(f"[Thread {thread_id}] ❌ ERROR: Out-of-bounds access\n")
            else:
                f.write(f"[Thread {thread_id}] ✅ Checking Point ({x_coord:.6f}, {y_coord:.6f}) "
                        f"-> ring_start={ring_start}, ring_end={ring_end}, Inside={inside_flag}, Case={special_case}\n")

    return results.get()






def test_points_against_polygons_geoseries_version(grand_all_dilations_sp_trial_nearest_interpolated_zslice_index_and_zval_array, 
                                 combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff, 
                                 nominal_and_dilated_structures_list_of_2d_arr, 
                                 nominal_and_dilated_structures_slices_indices_list):
    """
    Test points against polygons using CuSpatial.
    
    Parameters:
    - grand_all_dilations_sp_trial_nearest_interpolated_zslice_index_and_zval_array: Array of the nearest z values and their indices for each biopsy point.
    - combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff: 3D array of biopsy points for all trials.
    - nominal_and_dilated_structures_list_of_2d_arr: List of (N, 3) arrays for each trial, representing the dilated structures.
    - nominal_and_dilated_structures_slices_indices_list: List of indices indicating the start and end indices of each z slice for every dilated structure (trial).
    
    Returns:
    - result_cp_arr: Array indicating whether each point is inside the corresponding polygon.
    """
    num_trials = grand_all_dilations_sp_trial_nearest_interpolated_zslice_index_and_zval_array.shape[0]
    num_points_per_trial = grand_all_dilations_sp_trial_nearest_interpolated_zslice_index_and_zval_array.shape[1]
    
    # Initialize an array to store the results
    result_cp_arr = cp.zeros((num_trials, num_points_per_trial), dtype=cp.bool_)
    
    # Flatten the input arrays for easier processing
    flat_indices = grand_all_dilations_sp_trial_nearest_interpolated_zslice_index_and_zval_array[:, :, 1].flatten()
    flat_points = combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff[:, :, :2].reshape(-1, 2)
    
    # Filter out NaN indices
    valid_mask = ~np.isnan(flat_indices)
    valid_indices = flat_indices[valid_mask].astype(int)
    valid_points = flat_points[valid_mask]
    
    # Create a list of polygons for the valid indices
    valid_polygons_list = []
    for trial_index in range(num_trials):
        for point_index in range(num_points_per_trial):
            if not valid_mask[trial_index * num_points_per_trial + point_index]:
                continue
            slice_index = int(grand_all_dilations_sp_trial_nearest_interpolated_zslice_index_and_zval_array[trial_index, point_index, 1])
            start_idx, end_idx = nominal_and_dilated_structures_slices_indices_list[trial_index][slice_index]
            polygon_points = nominal_and_dilated_structures_list_of_2d_arr[trial_index][start_idx:end_idx, :2]
            valid_polygons_list.append(Polygon(polygon_points))
    
    # Create a CuSpatial GeoSeries of polygons
    polygons_geoseries = cuspatial.GeoSeries(valid_polygons_list)
    
    # Create a CuSpatial GeoSeries of points
    points_geoseries = cuspatial.GeoSeries.from_points_xy(valid_points.flatten())
    
    # Test each point against the corresponding polygon
    valid_results = one_to_one_point_in_polygon_geoseries_version(points_geoseries, polygons_geoseries)
    
    # Map the valid results back to the original result array
    result_cp_arr_flat = result_cp_arr.flatten()
    result_cp_arr_flat[valid_mask] = valid_results
    
    # Reshape the result array back to the original shape
    result_cp_arr = result_cp_arr_flat.reshape(num_trials, num_points_per_trial)
    
    return result_cp_arr




def test_points_against_polygons_cupy_arr_version(grand_all_dilations_sp_trial_nearest_interpolated_zslice_index_and_zval_array, 
                                 combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff, 
                                 nominal_and_dilated_structures_list_of_2d_arr, 
                                 nominal_and_dilated_structures_slices_indices_list):
    """
    Test points against polygons using CuPy arrays directly.
    
    Parameters:
    - grand_all_dilations_sp_trial_nearest_interpolated_zslice_index_and_zval_array: Array of the nearest z values and their indices for each biopsy point.
    - combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff: 3D array of biopsy points for all trials.
    - nominal_and_dilated_structures_list_of_2d_arr: List of (N, 3) arrays for each trial, representing the dilated structures.
    - nominal_and_dilated_structures_slices_indices_list: List of indices indicating the start and end indices of each z slice for every dilated structure (trial).
    
    Returns:
    - result_cp_arr: Array indicating whether each point is inside the corresponding polygon.
    """
    num_trials = grand_all_dilations_sp_trial_nearest_interpolated_zslice_index_and_zval_array.shape[0]
    num_points_per_trial = grand_all_dilations_sp_trial_nearest_interpolated_zslice_index_and_zval_array.shape[1]
    
    # Initialize an array to store the results
    result_cp_arr = cp.zeros((num_trials, num_points_per_trial), dtype=cp.bool_)
    
    # Flatten the input arrays for easier processing
    flat_indices = grand_all_dilations_sp_trial_nearest_interpolated_zslice_index_and_zval_array[:, :, 1].flatten()
    flat_points = combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff[:, :, :2].reshape(-1, 2)
    
    # Filter out NaN indices
    valid_mask = ~np.isnan(flat_indices)
    valid_indices = flat_indices[valid_mask].astype(int)
    valid_points = flat_points[valid_mask]
    
    # Create CuPy arrays for the points
    points = cp.array(valid_points)
    
    # Create CuPy arrays for the polygons and indices
    poly_points = []
    poly_indices = []
    current_index = 0
    
    for trial_index in range(num_trials):
        for point_index in range(num_points_per_trial):
            if not valid_mask[trial_index * num_points_per_trial + point_index]:
                continue
            slice_index = int(grand_all_dilations_sp_trial_nearest_interpolated_zslice_index_and_zval_array[trial_index, point_index, 1])
            start_idx, end_idx = nominal_and_dilated_structures_slices_indices_list[trial_index][slice_index]
            polygon_points = nominal_and_dilated_structures_list_of_2d_arr[trial_index][start_idx:end_idx, :2]
            poly_points.append(polygon_points)
            poly_indices.append([current_index, current_index + len(polygon_points)])
            current_index += len(polygon_points)
    
    poly_points = cp.array(np.vstack(poly_points))
    poly_indices = cp.array(poly_indices)
    
    # Test each point against the corresponding polygon
    valid_results = one_to_one_point_in_polygon_cupy_arr_version(points, poly_points, poly_indices)
    
    # Map the valid results back to the original result array
    result_cp_arr_flat = result_cp_arr.flatten()
    result_cp_arr_flat[valid_mask] = valid_results
    
    # Reshape the result array back to the original shape
    result_cp_arr = result_cp_arr_flat.reshape(num_trials, num_points_per_trial)
    
    return result_cp_arr



def create_containment_results_dataframe(patientUID, biopsy_structure_info, structure_info, 
                                         grand_all_dilations_sp_trial_nearest_interpolated_zslice_index_and_zval_array, 
                                         test_points_array, result_cp_arr):
    """
    Create a DataFrame to keep track of the containment results.
    
    Parameters:
    - patientUID: Patient ID.
    - biopsy_structure_info: Dictionary containing biopsy structure information.
    - structure_info: List containing relative structure information.
    - grand_all_dilations_sp_trial_nearest_interpolated_zslice_index_and_zval_array: Array of the nearest z values and their indices for each biopsy point.
    - test_points_array: Array of test points.
    - result_cp_arr: Array indicating whether each point is inside the corresponding polygon.
    
    Returns:
    - containment_results_df: DataFrame containing the containment results.
    """
    num_trials = grand_all_dilations_sp_trial_nearest_interpolated_zslice_index_and_zval_array.shape[0]
    num_points_per_trial = grand_all_dilations_sp_trial_nearest_interpolated_zslice_index_and_zval_array.shape[1]
    
    # Flatten the input arrays for easier processing
    flat_nearest_zslices_vals_arr = grand_all_dilations_sp_trial_nearest_interpolated_zslice_index_and_zval_array[:, :, 2].flatten()
    flat_nearest_zslices_indices_arr = grand_all_dilations_sp_trial_nearest_interpolated_zslice_index_and_zval_array[:, :, 1].flatten()
    flat_trial_number_arr = grand_all_dilations_sp_trial_nearest_interpolated_zslice_index_and_zval_array[:, :, 0].flatten().astype(int)
    flat_test_points_array = test_points_array.reshape(-1, 3)
    flat_result_cp_arr = result_cp_arr.get().flatten()
    
    # Create RGB color arrays based on the containment results
    pt_clr_r = np.where(flat_result_cp_arr, 0, 1)  # Red for false
    pt_clr_g = np.where(flat_result_cp_arr, 1, 0)  # Green for true
    pt_clr_b = np.zeros_like(pt_clr_r)  # Blue is always 0
    
    # Create a dictionary to store the results
    results_dictionary = {
        "Patient ID": [patientUID] * len(flat_result_cp_arr),
        "Bx ID": [biopsy_structure_info["Structure ID"]] * len(flat_result_cp_arr),
        "Biopsy refnum": [biopsy_structure_info["Dicom ref num"]] * len(flat_result_cp_arr),
        "Bx index": [biopsy_structure_info["Index number"]] * len(flat_result_cp_arr),
        "Relative structure ROI": [structure_info[0]] * len(flat_result_cp_arr),
        "Relative structure type": [structure_info[1]] * len(flat_result_cp_arr),
        "Relative structure index": [structure_info[3]] * len(flat_result_cp_arr),
        "Original pt index": np.tile(np.arange(num_points_per_trial), num_trials),
        "Pt contained bool": flat_result_cp_arr,
        "Nearest zslice zval": flat_nearest_zslices_vals_arr,
        "Nearest zslice index": flat_nearest_zslices_indices_arr,
        "Pt clr R": pt_clr_r,
        "Pt clr G": pt_clr_g,
        "Pt clr B": pt_clr_b,
        "Test pt X": flat_test_points_array[:, 0],
        "Test pt Y": flat_test_points_array[:, 1],
        "Test pt Z": flat_test_points_array[:, 2],
        "Trial num": flat_trial_number_arr
    }
    
    containment_results_df = pd.DataFrame(results_dictionary)
    return containment_results_df






















"""
# -------------------------------
# 🔹 Example Usage
# -------------------------------
points_list = [Point(0.5, 0.5), Point(1.5, 1.5), Point(3.5, 3.5)]
polygons_list = [
    Polygon([(0,0), (1,0), (1,1), (0,1), (0,0)]),  # Point (0.5, 0.5) is inside
    Polygon([(1,1), (2,1), (2,2), (1,2), (1,1)]),  # Point (1.5, 1.5) is inside
    Polygon([(2,2), (3,2), (3,3), (2,3), (2,2)])   # Point (3.5, 3.5) is outside
]

# ✅ Convert to `cuspatial.GeoSeries`
polygons_gs = cuspatial.GeoSeries(gpd.GeoSeries(polygons_list))
points_gs = cuspatial.GeoSeries(gpd.GeoSeries(points_list))

# ✅ Run Fixed Kernel
one_to_one_results = one_to_one_point_in_polygon(points_gs, polygons_gs)
print("\n🔹 Optimized One-to-One Results:")
print(one_to_one_results)
"""


def example():
    # -------------------------------
    # 🔹 Generate 10,000 Random Points & Polygons
    # -------------------------------
    num_points = 750
    print('WHAT THE FUCK')
    # Generate random points inside (0,3)x(0,3)
    random_points = [Point(x, y) for x, y in zip(np.random.uniform(0, 3, num_points), 
                                                np.random.uniform(0, 3, num_points))]

    # Generate corresponding polygons around slightly different random locations
    random_polygons = [
        Polygon([(x-0.1, y-0.1), (x+0.1, y-0.1), (x+0.1, y+0.1), (x-0.1, y+0.1), (x-0.1, y-0.1)]) 
        for x, y in zip(np.random.uniform(0, 3, num_points), np.random.uniform(0, 3, num_points))
    ]

    # ✅ Convert to `cuspatial.GeoSeries`
    polygons_gs = cuspatial.GeoSeries(gpd.GeoSeries(random_polygons))
    points_gs = cuspatial.GeoSeries(gpd.GeoSeries(random_points))

    # -------------------------------
    # 🔹 Run Custom One-to-One Test
    # -------------------------------
    st = time.time()
    one_to_one_results = one_to_one_point_in_polygon_geoseries_version(points_gs, polygons_gs)
    et= time.time()
    print("\n🔹 Custom One-to-One Time:", et-st)

    # -------------------------------
    # 🔹 Run Default `cuspatial.point_in_polygon` in Chunks (Max 31 Polygons per Batch)
    # -------------------------------
    def chunked_point_in_polygon(points_gs, polygons_gs, chunk_size=31):
        num_polygons = len(polygons_gs)
        num_points = len(points_gs)
        
        # ✅ Allocate a results matrix
        results_matrix = cp.zeros((num_points, num_polygons), dtype=cp.bool_)

        for start in range(0, num_polygons, chunk_size):
            end = min(start + chunk_size, num_polygons)
            print(f"🔹 Processing polygons {start}-{end}")

            # ✅ Extract the chunk of polygons
            polygons_chunk = polygons_gs[start:end]

            # ✅ Run cuSpatial's `point_in_polygon` on this chunk
            chunk_results = cuspatial.point_in_polygon(points_gs, polygons_chunk).to_cupy()

            # ✅ Store the results in the full matrix
            results_matrix[:, start:end] = chunk_results

        return results_matrix

    # 🔹 Run Chunked Default Test
    st = time.time()
    default_results = chunked_point_in_polygon(points_gs, polygons_gs)
    et = time.time()
    print("\n🔹 Default CuSpatial Time (Chunked):", et - st)

    # Extract diagonal results from default `point_in_polygon`
    diagonal_results = cp.diag(default_results)

    # -------------------------------
    # 🔹 Compare Results
    # -------------------------------
    print("\n🔹 Optimized One-to-One Results:")
    print(one_to_one_results[0:20])  # Print first 20 results

    print("\n🔹 Diagonal Results from Default CuSpatial:")
    print(diagonal_results[0:20])  # Print first 20 results

    # Check if results match
    print("\n✅ Do Results Match?", cp.all(one_to_one_results == diagonal_results))


import cupy as cp
import numpy as np
import time
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point

def generate_polygons_and_points(num_points, num_vertices):
    """
    Generate random points and corresponding polygons with a specified number of vertices.
    
    Parameters:
    - num_points: Number of points and polygons to generate.
    - num_vertices: Number of vertices for each polygon.
    
    Returns:
    - points: CuPy array of shape (num_points, 2) containing the x and y coordinates of the points.
    - poly_points: CuPy array of shape (num_polygons * num_vertices, 2) containing the x and y coordinates of all polygon vertices.
    - poly_indices: CuPy array of shape (num_polygons, 2) containing the start and end indices of each polygon in poly_points.
    """
    points = []
    poly_points = []
    poly_indices = []
    current_index = 0
    
    for _ in range(num_points):
        # Generate a random point
        point = Point(np.random.uniform(0, 3), np.random.uniform(0, 3))
        points.append([point.x, point.y])
        
        # Generate a corresponding polygon around the point
        angle_step = 2 * np.pi / num_vertices
        polygon = Polygon([
            (point.x + 0.1 * np.cos(i * angle_step), point.y + 0.1 * np.sin(i * angle_step))
            for i in range(num_vertices)
        ])
        
        # Add the polygon vertices to the poly_points list
        for x, y in polygon.exterior.coords[:-1]:  # Exclude the last point because it's a duplicate of the first
            poly_points.append([x, y])
        
        # Add the start and end indices to the poly_indices list
        poly_indices.append([current_index, current_index + num_vertices])
        current_index += num_vertices
    
    points = cp.array(points)
    poly_points = cp.array(poly_points)
    poly_indices = cp.array(poly_indices)
    
    return points, poly_points, poly_indices

def test_block_sizes(points, poly_points, poly_indices, num_vertices):
    block_sizes = [32, 64, 128, 256, 512, 1024]
    times = []

    for block_size in block_sizes:
        start_time = time.time()
        one_to_one_point_in_polygon_cupy_arr_version(points, poly_points, poly_indices, block_size, log_file_name=f"cuda_log_block_size_{str(block_size)}-num_vrtcs{str(num_vertices)}.txt")
        end_time = time.time()
        times.append(end_time - start_time)
        print(f"Block size: {block_size}, Time: {end_time - start_time:.6f} seconds")

    return block_sizes, times

def test_block_sizes_main():
    num_points = 10000
    num_vertices_list = [4, 8, 16, 32]
    all_times = []

    for num_vertices in num_vertices_list:
        print(f"\nTesting with {num_vertices}-vertex polygons:")
        points, poly_points, poly_indices = generate_polygons_and_points(num_points, num_vertices)
        
        block_sizes, times = test_block_sizes(points, poly_points, poly_indices, num_vertices)
        all_times.append(times)
        
        # Plot the results
        plt.plot(block_sizes, times, marker='o', label=f'{num_vertices} vertices')
    
    plt.xlabel('Block Size')
    plt.ylabel('Time (seconds)')
    plt.title('Performance vs. Block Size')
    plt.legend()
    plt.show()

    for num_vertices, times in zip(num_vertices_list, all_times):
        print(f"\nResults for {num_vertices}-vertex polygons:")
        for ind, time in enumerate(times):
            print(f"Block size: {block_sizes[ind]}, Time: {time:.6f} seconds")

    input()

if __name__ == "__main__":
    #example()
    test_block_sizes_main()