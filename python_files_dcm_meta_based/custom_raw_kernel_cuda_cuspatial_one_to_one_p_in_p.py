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



### IMPORTANT NOTE: NOTE THAT WHEN CALLING THIS KERNEL, ALL DATA STORED ON GPU MUST BE CONTIGUOUSLY STORED IN MEMORY
### IMPORTANT NOTE: THE POLYGONS THAT ARE PASSED TO THIS KERNEL IS ASSUMED TO BE BUILT SUCH THAT THE FIRST AND LAST POINTS ARE THE SAME!
one_to_one_pip_kernel_advanced = cp.RawKernel(r'''
extern "C" __global__
void one_to_one_pip(const double* px, const double* py,
                    const double* poly_x, const double* poly_y,
                    const long long int* poly_part_offsets,
                    const long long int* edge_offsets,  // ✅ Now explicitly long long int
                    int* results, int num_points, int poly_offsets_size,
                    long long int* log_buffer) {  // ✅ Now explicitly long long int
    
    const int LOG_ENTRY_SIZE = 14;  // ✅ Increased to store debug info

    long long int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= num_points) return;

    long long int edge_log_offset = edge_offsets[i];  // ✅ Now correctly stores large offsets
    long long int log_position = i * LOG_ENTRY_SIZE + edge_log_offset * 7;
    //printf("Thread %lld: | Log Position: %lld | edge_log_offset: %lld\n", i, log_position, edge_log_offset);
                                                                             
    
                                              

    if (i + 1 >= poly_offsets_size) {  
        log_buffer[log_position + 0] = i;
        log_buffer[log_position + 1] = -1;
        results[i] = 0;
        return;
    }

    double x = px[i];
    double y = py[i];

    long long ring_start = poly_part_offsets[i];
    long long ring_end = poly_part_offsets[i + 1];
    int num_edges = (ring_end - ring_start) - 1;

    /*                                            
    printf("Thread: %lld | Num Edges: %d\n", i, num_edges);
    printf("Thread: %lld | ring_start: %lld\n", i, ring_start);                                                                                    
    printf("Thread: %lld | ring_end: %lld\n", i, ring_end);
    */                                        

    if (num_edges <= 0 || ring_start < 0) {
        log_buffer[log_position + 0] = i;
        log_buffer[log_position + 1] = -2;
        results[i] = 0;
        return;
    }

    double min_x = poly_x[ring_start], max_x = poly_x[ring_start];
    double min_y = poly_y[ring_start], max_y = poly_y[ring_start];

    // Find the bounding box of the polygon, used for determining the ray length
    // Note that we dont need to check the last point because the data that is passed in here is assumed that the first and last point of the polygon are the exact same!
    for (long long j = ring_start + 1; j < ring_end; j++) {
        if (poly_x[j] < min_x) min_x = poly_x[j];
        if (poly_x[j] > max_x) max_x = poly_x[j];
        if (poly_y[j] < min_y) min_y = poly_y[j];
        if (poly_y[j] > max_y) max_y = poly_y[j];
    }

    /*                                        
    printf("Thread: %lld | Min X: %f, Max X: %f, Min Y: %f, Max Y: %f\n", i, min_x, max_x, min_y, max_y);
    // Now print all the points of the polygon as well to compare
    for (long long j = ring_start; j <= ring_end; j++) {
        printf("Thread: %lld | Polygon Point %lld: (%f, %f)\n", i, j, poly_x[j], poly_y[j]);
    } 

    for (long long j = ring_start; j <= ring_end; j++) {
        printf("X: %f \n",poly_x[j]);
        printf("Y: %f \n",poly_y[j]);
    }          
    */                                                                  
                                              
    double ray_length = fmax(max_x - min_x, max_y - min_y) * 2.5;
                                              
    //printf("Thread: %lld | Ray Length: %f\n", i, ray_length);                                          

    bool inside = false;
    int intersection_count = 0;

                                              
    // Defines the tolerances for safe division and checking if a point is on the boundary and if a ray is too close to a vertex
    #define EPSILON 1e-7 
    #define M_PI 3.14159265358979323846
    #define EPSILON_VERTEX 1e-8 
    #define EPSILON_BOUNDARY 1e-7 
    //
    

    double angle = 0.0;
    double dx = cos(angle);
    double dy = sin(angle);

    int max_attempts = 10;
    int attempt = 0;

    

    while (attempt < max_attempts) {
        inside = false;
        intersection_count = 0;
        bool valid_ray = true;
        long long int log_position_temp = log_position;  // Initialize the temporary log position

        
        // Check if the point being tested lies on the boundary of the polygon, only need to check first attempt since the point never moves, additional attempts only relevant to rays
        if (attempt == 0) {
            bool point_on_boundary = false;

            for (long long j = ring_start; j < ring_end - 1; j++) {
                int intersection_type = 0;

                long long k = j + 1;

                double xj = poly_x[j], yj = poly_y[j];
                double xk = poly_x[k], yk = poly_y[k];         

                double cross = (x - xj) * (yk - yj) - (y - yj) * (xk - xj);
                                                
                if (fabs(cross) < EPSILON_BOUNDARY) {
                    printf("❓ Point potentially on polygon boundary | Checking... (Thread: %lld)\n", i);
                    
                    double dot1 = (x - xj) * (xk - xj) + (y - yj) * (yk - yj);
                    double dot2 = (x - xk) * (xj - xk) + (y - yk) * (yj - yk);

                    // Check if the point is on the line segment
                    if (dot1 >= 0 && dot2 >= 0) {
                        intersection_type = 2;
                        point_on_boundary = true;
                        inside = true;
                        printf("🔥 Point on polygon boundary | Setting to inside (Thread: %lld)\n", i);
                    }
                    else {
                        printf("🖤 Point not on polygon boundary | Continuing... (Thread: %lld)\n", i);
                    }
                }
                                                
                log_buffer[log_position_temp + LOG_ENTRY_SIZE + 0] = j;
                log_buffer[log_position_temp + LOG_ENTRY_SIZE + 1] = k;
                log_buffer[log_position_temp + LOG_ENTRY_SIZE + 2] = intersection_type;
                log_buffer[log_position_temp + LOG_ENTRY_SIZE + 3] = __double_as_longlong(xj);
                log_buffer[log_position_temp + LOG_ENTRY_SIZE + 4] = __double_as_longlong(yj);
                log_buffer[log_position_temp + LOG_ENTRY_SIZE + 5] = __double_as_longlong(xk);
                log_buffer[log_position_temp + LOG_ENTRY_SIZE + 6] = __double_as_longlong(yk);
                log_position_temp += 7;  // ✅ Move to the next available slot for edges
            }
            // If on boundary, no need to check further                                      
            if (point_on_boundary) break;
        }
                                              


                                              
        // Check if the ray is too close to any vertex
        bool too_close_to_vertex = false;
        double denom_norm = sqrt(dx * dx + dy * dy);  // Normalize denominator for distance calc

        for (long long j = ring_start; j < ring_end; j++) {
            double x_v = poly_x[j], y_v = poly_y[j];

            double d = fabs(dx * (y_v - y) - dy * (x_v - x)) / denom_norm;

            if (d < EPSILON_VERTEX) {
                too_close_to_vertex = true;
                printf("Ray too close to a vertex, setting bool (Thread: %lld)\n", i);
                break;
            }
        }

        if (too_close_to_vertex) {
            printf("Ray too close to a vertex, regenerating (Thread: %lld)\n", i);
            attempt++;
            angle = fmod(angle + (2*M_PI / max_attempts), M_PI);
            dx = cos(angle);
            dy = sin(angle);
            continue;  // Retry with a new ray
        }

                                              


        // Check if the point is inside the polygon
        log_position_temp = log_position;  // Reset the temporary log position
                                              
        for (long long j = ring_start; j < ring_end - 1; j++) {
                                  
            long long k = j + 1;

            double xj = poly_x[j], yj = poly_y[j];
            double xk = poly_x[k], yk = poly_y[k];
                                              
            // Defines the scale of the edge for tolerance of denom
            double edge_scale = fmax(fabs(xj - xk), fabs(yj - yk));
            double tol_edge = EPSILON * edge_scale;
            //

            double denom = (xj - xk) * (dy) - (yj - yk) * (dx);
                                              
            //printf("Thread: %lld | j: %lld | k: %lld | (x_t, y_t): (%.2f, %.2f) | (xj, yj): (%.2f, %.2f) | (xk, yk): (%.2f, %.2f)\n", i, j, k, x ,y , xj, yj, xk, yk);
            //printf("Thread: %lld | j: %lld | k: %lld | dx: %f | dy: %f | denom: %f | ray_length: %f\n", i, j, k, dx, dy, denom, ray_length);

                                              
            int intersection_type = 0;

            if (fabs(denom) > tol_edge) {  
                double t_edge = ((x - xj) * (-dy) + (y - yj) * dx) / denom;
                double s_ray = ((xk - xj) * (y - yj) - (yk - yj) * (x - xj)) / denom;

                if (t_edge >= 0 && t_edge <= 1 && s_ray > 0 && s_ray <= ray_length) {
                    //printf("🔥1 Thread: %lld | j: %lld | k: %lld | (x_t, y_t): (%.2f, %.2f) | (xj, yj): (%.2f, %.2f) | (xk, yk): (%.2f, %.2f)\n", i, j, k, x ,y , xj, yj, xk, yk);
                    //printf("🔥2 Thread: %lld | j: %lld | k: %lld | dx: %f | dy: %f | denom: %f | ray_length: %f\n", i, j, k, dx, dy, denom, ray_length); 
                    //printf("🔥3 Thread: %lld | j: %lld | k: %lld | attempt: %d | t_edge: %f | s_ray: %f | Intersection found \n", i, j, k, attempt, t_edge, s_ray);
                    inside = !inside;
                    intersection_count++;
                    intersection_type = 1;
                }
                else {
                    //printf("🖤1 Thread: %lld | j: %lld | k: %lld | (x_t, y_t): (%.2f, %.2f) | (xj, yj): (%.2f, %.2f) | (xk, yk): (%.2f, %.2f)\n", i, j, k, x ,y , xj, yj, xk, yk);
                    //printf("🖤2 Thread: %lld | j: %lld | k: %lld | dx: %f | dy: %f | denom: %f | ray_length: %f\n", i, j, k, dx, dy, denom, ray_length); 
                    //printf("🖤3 Thread: %lld | j: %lld | k: %lld | attempt: %d | t_edge: %f | s_ray: %f | No intersection found \n", i, j, k, attempt, t_edge, s_ray);
                }
            }
            else {
                printf("Denom is zero on attempt: %d, (Thread: %lld)\n", attempt, i);
                valid_ray = false;
                break;                                
            }                                  
            
            // Commenting out most of these except the intersection type, because added boundary check which writes all of these already
            //log_buffer[log_position_temp + LOG_ENTRY_SIZE + 0] = j;
            //log_buffer[log_position_temp + LOG_ENTRY_SIZE + 1] = k;
            log_buffer[log_position_temp + LOG_ENTRY_SIZE + 2] = intersection_type;
            //log_buffer[log_position_temp + LOG_ENTRY_SIZE + 3] = __double_as_longlong(xj);
            //log_buffer[log_position_temp + LOG_ENTRY_SIZE + 4] = __double_as_longlong(yj);
            //log_buffer[log_position_temp + LOG_ENTRY_SIZE + 5] = __double_as_longlong(xk);
            //log_buffer[log_position_temp + LOG_ENTRY_SIZE + 6] = __double_as_longlong(yk);
            log_position_temp += 7;  // ✅ Move to the next available slot for edges

        }

        if (valid_ray) break;

        attempt++;
        angle = fmod(angle + (2*M_PI / max_attempts), M_PI);
        dx = cos(angle);
        dy = sin(angle);
    }

    log_buffer[log_position + 0] = i;
    log_buffer[log_position + 1] = ring_start;
    log_buffer[log_position + 2] = ring_end;
    log_buffer[log_position + 3] = __double_as_longlong(x);
    log_buffer[log_position + 4] = __double_as_longlong(y);
    log_buffer[log_position + 5] = inside;
    log_buffer[log_position + 6] = intersection_count;
    log_buffer[log_position + 7] = attempt;
    log_buffer[log_position + 8] = __double_as_longlong(angle);
    log_buffer[log_position + 9] = __double_as_longlong(dx);
    log_buffer[log_position + 10] = __double_as_longlong(dy);
    log_buffer[log_position + 11] = num_edges;
    log_buffer[log_position + 12] = edge_log_offset;  // ✅ Debugging: Log the offset
    log_buffer[log_position + 13] = log_position;  
                                              
    results[i] = inside ? 1 : 0;
}


''', 'one_to_one_pip')







def one_to_one_point_in_polygon_cupy_arr_version(points, poly_points, poly_indices, block_size=256, log_sub_dirs_list = [], log_file_name="cuda_log.txt"):
    """
    Test each point against the corresponding polygon using CuPy arrays directly, note that this mapping is one-to-one.
    
    Parameters:
    - points: CuPy array of shape (num_points, 2) containing the x and y coordinates of the points.
    - poly_points: CuPy array of shape (num_vertices, 2) containing the x and y coordinates of the polygon vertices. Important! The first and last points must be the same!
    - poly_indices: CuPy array of shape (num_points, 2) containing the start and end indices of the polygon in poly_points. Important! The indices must be such that the end index is exclusive!
    - block_size: Block size for the CUDA kernel.
    - log_file_name: Name of the log file to write the debug information. If None, no log file is written to file. Important, the log file writing is quite slow, so turning off logging should be considered for performance.
    """

    num_points = cp.int32(points.shape[0])

    points_x = points[:, 0]
    points_y = points[:, 1]
    points_x = cp.ascontiguousarray(points_x, dtype=cp.float64)
    points_y = cp.ascontiguousarray(points_y, dtype=cp.float64)

    poly_x = poly_points[:, 0]
    poly_y = poly_points[:, 1]
    poly_x = cp.ascontiguousarray(poly_x, dtype=cp.float64)
    poly_y = cp.ascontiguousarray(poly_y, dtype=cp.float64)

    poly_part_offsets = cp.zeros(poly_indices.shape[0] + 1, dtype=cp.int64)
    poly_part_offsets[1:] = cp.cumsum(poly_indices[:, 1] - poly_indices[:, 0]).astype(cp.int64)
    #poly_part_offsets = cp.ascontiguousarray(poly_part_offsets, dtype=cp.int64)

    poly_offsets_size = cp.int32(poly_part_offsets.shape[0])

    results = cp.zeros(num_points, dtype=cp.int32)
    results = cp.ascontiguousarray(results, dtype=cp.int32)

    log_entry_size = 14  # ✅ Matches CUDA kernel

    num_edges_per_polygon = (poly_indices[:, 1] - poly_indices[:, 0]) - 1  
    edge_offsets = cp.zeros(num_points.item() + 1, dtype=cp.int64)  
    edge_offsets[1:] = cp.cumsum(num_edges_per_polygon).astype(cp.int64)
    edge_offsets = cp.ascontiguousarray(edge_offsets, dtype=cp.int64)  

    total_edge_entries = edge_offsets[-1].item() * 7
    log_buffer = cp.zeros(num_points.item() * log_entry_size + total_edge_entries, dtype=cp.int64)
    log_buffer = cp.ascontiguousarray(log_buffer, dtype=cp.int64)

    grid_size = (num_points.item() + block_size - 1) // block_size

    ### IMPORTANT NOTE: WHEN CALLING THIS KERNEL, ALL DATA STORED ON GPU MUST BE CONTIGUOUSLY STORED IN MEMORY
    ### IMPORTANT NOTE: THE POLYGONS THAT ARE PASSED TO THIS KERNEL IS ASSUMED TO BE BUILT SUCH THAT THE FIRST AND LAST POINTS ARE THE SAME!
    one_to_one_pip_kernel_advanced(
        (grid_size,), (block_size,),
        (points_x, points_y, poly_x, poly_y, poly_part_offsets, edge_offsets, results, num_points, poly_offsets_size, log_buffer)
    )

    if log_file_name is not None:
        logs_host = log_buffer.get()

        log_dir = pathlib.Path(__file__).parents[0].joinpath("cuda_containment_logs")
        for log_sub_dir in log_sub_dirs_list:
            log_dir = log_dir.joinpath(log_sub_dir)
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir.joinpath(log_file_name)

        with open(log_file, "w") as f:
            for i in range(num_points.item()):
                # ✅ Compute `log_position` exactly as in the CUDA kernel
                edge_log_offset = edge_offsets[i].item() * 7
                log_position = i * log_entry_size + edge_log_offset  # ✅ Now perfectly aligned with kernel

                meta_start = log_position
                static_meta_end = meta_start + log_entry_size
                log = logs_host[meta_start:static_meta_end]

                thread_id = log[0]
                ring_start = log[1]
                ring_end = log[2]
                x_coord = struct.unpack('d', struct.pack('q', log[3]))[0]
                y_coord = struct.unpack('d', struct.pack('q', log[4]))[0]
                inside_flag = log[5]
                intersection_count = log[6]
                retries = log[7]
                angle = struct.unpack('d', struct.pack('q', log[8]))[0]
                dx = struct.unpack('d', struct.pack('q', log[9]))[0]
                dy = struct.unpack('d', struct.pack('q', log[10]))[0]
                num_edges = log[11]
                log_offset_debug = log[12]
                log_position_debug = log[13]

                # ✅ Extract checked edges using `edge_offsets[i]`
                edge_log_start = static_meta_end
                edge_log_end = edge_log_start + num_edges_per_polygon[i].item() * 7
                checked_edges = logs_host[edge_log_start:edge_log_end].reshape(-1, 7)
                
                """
                checked_edges[:, 3:7] = checked_edges[:, 3:7].view(np.float64)
                # Optionally round:
                checked_edges[:, 3:7] = np.around(checked_edges[:, 3:7], decimals=2)

                checked_edges_str = np.array2string(checked_edges, separator=', ')

                # Replace newlines (which separate rows) with " | "
                checked_edges_str = checked_edges_str.replace('\n', ' | ')
                """
                
                """
                checked_edges_list = checked_edges.tolist()
                checked_edges_list_converted_long_to_double = [
                        [round(struct.unpack('d', struct.pack('q', element))[0],2) if i in (3, 4, 5, 6) else element 
                        for i, element in enumerate(inner_list)]
                        for inner_list in checked_edges_list
                    ]
                """

                checked_edges_list_converted_long_to_double = format_edges_for_point(checked_edges)

                # ✅ Write correct logs to the file
                f.write(f"[Thread {thread_id}] ✅ Checked Point ({x_coord:.4f}, {y_coord:.4f}) -> "
                        f"ring_start={ring_start}, ring_end={ring_end}, num_edges={num_edges}, Inside={inside_flag}, "
                        f"Intersections={intersection_count}, Retries={retries}, dx={dx:.4f}, dy={dy:.4f}, "
                        f"angle={angle:.4f}, edge_log_offset={log_offset_debug}, log_position={log_position_debug}, "
                        f"Checked Edges={checked_edges_list_converted_long_to_double}\n")

    return results.get()






def format_edges_for_point(checked_edges):
    # Vectorized conversion:
    conv = checked_edges.copy()
    conv[:, 3:7] = conv[:, 3:7].view(np.float64)
    conv[:, 3:7] = np.around(conv[:, 3:7], decimals=2)
    # Now convert each row into a string.
    # Because the number of edges per point is typically small, a list comprehension is efficient.
    #row_strings = [", ".join(map(str, row)) for row in conv]
    
    # Join all rows with the separator " | "
    return conv.tolist()















def test_points_against_polygons_cupy_arr_version(grand_all_dilations_sp_trial_nearest_interpolated_zslice_index_and_zval_array, 
                                 combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff, 
                                 nominal_and_dilated_structures_list_of_2d_arr, 
                                 nominal_and_dilated_structures_slices_indices_list,
                                 log_sub_dirs_list = [],
                                 log_file_name="cuda_log.txt"):
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
    valid_points_cp_arr = cp.array(valid_points)
    
    # Create CuPy arrays for the polygons and indices
    poly_points = []
    poly_indices = []
    current_index = 0
    
    for trial_index in range(num_trials):
        for point_index in range(num_points_per_trial):
            if not valid_mask[trial_index * num_points_per_trial + point_index]:
                continue
            nearest_zslice_index = int(grand_all_dilations_sp_trial_nearest_interpolated_zslice_index_and_zval_array[trial_index, point_index, 1])
            start_idx, end_idx = nominal_and_dilated_structures_slices_indices_list[trial_index][nearest_zslice_index]
            polygon_points = nominal_and_dilated_structures_list_of_2d_arr[trial_index][start_idx:end_idx, :2]
            poly_points.append(polygon_points)
            poly_indices.append([current_index, current_index + len(polygon_points)])
            current_index += len(polygon_points)
    
    poly_points = cp.array(np.vstack(poly_points))
    poly_indices = cp.array(poly_indices)
    
    # Test each point against the corresponding polygon
    valid_results = one_to_one_point_in_polygon_cupy_arr_version(valid_points_cp_arr, poly_points, poly_indices, log_sub_dirs_list = [], log_file_name=log_file_name)
    
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





















def example():


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

    num_points = 10000
    num_vertices_list = [4, 16, 32]

    for num_vertices in num_vertices_list:
        print(f"\nTesting with {num_vertices}-vertex polygons:")
        points, poly_points, poly_indices = generate_polygons_and_points(num_points, num_vertices, radius=2)


        # Convert to `cuspatial.GeoSeries`
        polygons_gs = cuspatial.GeoSeries(gpd.GeoSeries([Polygon(poly_points[start.item():end.item()].get()) for start, end in poly_indices]))
        points_gs = cuspatial.GeoSeries(gpd.GeoSeries([Point(x, y) for x, y in points.get()]))

        plot_points_and_polys_bool = False
        if plot_points_and_polys_bool:
            for index in np.arange(num_points):
                plot_one_point_and_polygon(points[index], poly_points, poly_indices[index])
                #plot_one_point_and_polygon_geoseries(points_gs[index], polygons_gs[index])

        # -------------------------------
        # 🔹 Run Custom One-to-One Test
        # -------------------------------
        st = time.time()
        log_file_name = f"cuda_log_{num_vertices}_vertices.txt"
        one_to_one_results = one_to_one_point_in_polygon_cupy_arr_version(points, poly_points, poly_indices, log_file_name=log_file_name)
        et = time.time()
        print(f"\n🔹 Custom One-to-One Time for {num_vertices}-vertex polygons:", et - st)

        # -------------------------------
        # 🔹 Run Default `cuspatial.point_in_polygon` in Chunks (Max 31 Polygons per Batch)
        # -------------------------------
        st = time.time()
        default_results = chunked_point_in_polygon(points_gs, polygons_gs)
        et = time.time()
        print(f"\n🔹 Default CuSpatial Time (Chunked) for {num_vertices}-vertex polygons:", et - st)

        # Extract diagonal results from default `point_in_polygon`
        diagonal_results = cp.diag(default_results)

        # -------------------------------
        # 🔹 Compare Results
        # -------------------------------
        print(f"\n🔹 Optimized One-to-One Results for {num_vertices}-vertex polygons:")
        print(one_to_one_results[0:50])  # Print first 20 results

        print(f"\n🔹 Diagonal Results from Default CuSpatial for {num_vertices}-vertex polygons:")
        print(diagonal_results[0:50])  # Print first 20 results

        # Check if results match
        print(f"\n✅ Do Results Match for {num_vertices}-vertex polygons?", np.all(one_to_one_results == diagonal_results.get()))
        print('test')
    input()


import matplotlib.pyplot as plt


def plot_one_point_and_polygon(points, poly_points, poly_indices):
    """
    Plot one point and one polygon using CuPy arrays.
    
    Parameters:
    - points: CuPy array of shape (1, 2) containing the x and y coordinates of the point.
    - poly_points: CuPy array of shape (num_vertices, 2) containing the x and y coordinates of the polygon vertices.
    - poly_indices: CuPy array of shape (1, 2) containing the start and end indices of the polygon in poly_points.
    """
    points = points.get()
    poly_points = poly_points.get()
    poly_indices = poly_indices.get()

    plt.figure(figsize=(10, 10))
    
    # Plot point
    plt.scatter(points[0], points[1], color='blue', label='Point')
    
    # Plot polygon
    start, end = poly_indices
    polygon = poly_points[start:end]
    plt.plot(polygon[:, 0], polygon[:, 1], color='red')
    plt.fill(polygon[:, 0], polygon[:, 1], color='red', alpha=0.3)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Point and Polygon')
    plt.legend()
    plt.show()


def plot_one_point_and_polygon_geoseries(point_gs, polygon_gs):
    """
    Plot one point and one polygon using GeoSeries.
    
    Parameters:
    - point_gs: GeoSeries containing the point.
    - polygon_gs: GeoSeries containing the polygon.
    """
    plt.figure(figsize=(10, 10))
    
    # Plot point
    point_x = point_gs.x
    point_y = point_gs.y
    plt.scatter([point_x], [point_y], color='blue', label='Point')
    
    # Plot polygon
    x, y = polygon_gs.exterior.xy
    plt.plot(x, y, color='red')
    plt.fill(x, y, color='red', alpha=0.3)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Point and Polygon')
    plt.legend()
    plt.show()



def generate_polygons_and_points(num_points, num_vertices, radius=1):
    """
    Generate random points and corresponding polygons with a specified number of vertices.
    
    Parameters:
    - num_points: Number of points and polygons to generate.
    - num_vertices: Number of vertices for each polygon.
    - radius: Radius of the polygons.
    
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
        # Generate a random center point for the polygon
        center_x = np.random.uniform(0, 3)
        center_y = np.random.uniform(0, 3)
        
        # Generate a corresponding polygon around the point
        angle_step = 2 * np.pi / num_vertices
        polygon = Polygon([
            (center_x + radius * np.cos(i * angle_step), center_y + radius * np.sin(i * angle_step))
            for i in range(num_vertices)
        ])
        # Generate a random point
        point = Point(np.random.uniform(-3, 6), np.random.uniform(-3, 6))
        points.append([point.x, point.y])

        # Add the polygon vertices to the poly_points list
        for x, y in polygon.exterior.coords:  # Exclude the last point because it's a duplicate of the first
            poly_points.append([x, y])
        
        # Add the start and end indices to the poly_indices list
        poly_indices.append([current_index, current_index + num_vertices + 1])
        current_index += num_vertices + 1
    
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
    # Run this to see if our custom functions match the output of cuspatial.point_in_polygon default
    example()

    # Run this to test block sizes for timings and also just general functionality of pipeline
    #test_block_sizes_main()