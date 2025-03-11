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
from line_profiler import LineProfiler
import copy
import polygon_dilation_helpers_numpy
import dataframe_builders



### IMPORTANT NOTE: NOTE THAT WHEN CALLING THIS KERNEL, ALL DATA STORED ON GPU MUST BE CONTIGUOUSLY STORED IN MEMORY
### IMPORTANT NOTE: THE POLYGONS THAT ARE PASSED TO THIS KERNEL IS ASSUMED TO BE BUILT SUCH THAT THE FIRST AND LAST POINTS ARE THE SAME!
### IMPORTANT NOTE: POLYGONS WITH WITH EDGES THAT ARE TOO SHORT WILL BE SKIPPED AND THEREFORE MAY GIVE INCORRECT RESULTS! ENSURE THAT POLYGON EDGES ARE NOT TOO SHORT!
one_to_one_pip_kernel_advanced = cp.RawKernel(r'''
extern "C" __global__
void one_to_one_pip(const double* px, const double* py, const double* pz,
                    const double* poly_x, const double* poly_y, const double* poly_z,
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
    double z = pz[i];

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
    

    double angle_perturbation = M_PI / 30; // this is to avoid perfectly vertical and perfectly horiztontal rays
    double angle = 0 + angle_perturbation;
    double dx = cos(angle);
    double dy = sin(angle);

    int attempt = 0;
    int max_attempts = 10;

    

    while (attempt < max_attempts) {
        inside = false;
        intersection_count = 0;
        bool valid_ray = true;
        long long int log_position_temp = log_position;  // Initialize the temporary log position
        int intersection_type = 0;
        
        // Check if the point being tested lies on the boundary of the polygon, only need to check first attempt since the point never moves, additional attempts only relevant to rays
        if (attempt == 0) {
            bool point_on_boundary = false;

            for (long long j = ring_start; j < ring_end - 1; j++) {
                intersection_type = 0;

                long long k = j + 1;

                double xj = poly_x[j], yj = poly_y[j], zj = poly_z[j];
                double xk = poly_x[k], yk = poly_y[k], zk = poly_z[k];

                // Compute the edge vector and its length
                double edge_dx = xk - xj;
                double edge_dy = yk - yj;
                double edge_length = sqrt(edge_dx * edge_dx + edge_dy * edge_dy);

                // If the edge is too short (degenerate), skip it (assume no intersection from this edge)
                if (edge_length < EPSILON) {
                    intersection_type = 3; // Set intersection type to degenerate (3), meaning we skipped this edge
                    //log_buffer[log_position_temp + LOG_ENTRY_SIZE + 2] = intersection_type;
                    //log_position_temp += 7;  // ✅ Move to the next available slot for edges
                    //continue;
                }
                else {       

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
                }
                                                                             
                log_buffer[log_position_temp + LOG_ENTRY_SIZE + 0] = j;
                log_buffer[log_position_temp + LOG_ENTRY_SIZE + 1] = k;
                log_buffer[log_position_temp + LOG_ENTRY_SIZE + 2] = intersection_type;
                log_buffer[log_position_temp + LOG_ENTRY_SIZE + 3] = __double_as_longlong(xj);
                log_buffer[log_position_temp + LOG_ENTRY_SIZE + 4] = __double_as_longlong(yj);
                log_buffer[log_position_temp + LOG_ENTRY_SIZE + 5] = __double_as_longlong(zj);                                                      
                log_buffer[log_position_temp + LOG_ENTRY_SIZE + 6] = __double_as_longlong(xk);
                log_buffer[log_position_temp + LOG_ENTRY_SIZE + 7] = __double_as_longlong(yk);
                log_buffer[log_position_temp + LOG_ENTRY_SIZE + 8] = __double_as_longlong(zk);
                log_position_temp += num_information_per_edge;  // ✅ Move to the next available slot for edges
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
            angle = fmod(angle + (2*M_PI / max_attempts), 2*M_PI);
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

            intersection_type = 0;
                                              
            // Compute the edge vector and its length
            double edge_dx = xk - xj;
            double edge_dy = yk - yj;
            double edge_length = sqrt(edge_dx * edge_dx + edge_dy * edge_dy);

            // If the edge is too short (degenerate), skip it (assume no intersection from this edge)
            if (edge_length < EPSILON) {
                intersection_type = 3; // Set intersection type to degenerate (3), meaning we skipped this edge
                //log_buffer[log_position_temp + LOG_ENTRY_SIZE + 2] = intersection_type;
                //log_position_temp += 7;  // ✅ Move to the next available slot for edges
                //continue;
            }

            else {                            
                                           
                // Defines the scale of the edge for tolerance of denom
                double edge_scale = fmax(fabs(xj - xk), fabs(yj - yk));
                double tol_edge = EPSILON * edge_scale;
                //

                double denom = (xj - xk) * (dy) - (yj - yk) * (dx);
                                                
                //printf("Thread: %lld | j: %lld | k: %lld | (x_t, y_t): (%.2f, %.2f) | (xj, yj): (%.2f, %.2f) | (xk, yk): (%.2f, %.2f)\n", i, j, k, x ,y , xj, yj, xk, yk);
                //printf("Thread: %lld | j: %lld | k: %lld | dx: %f | dy: %f | denom: %f | ray_length: %f\n", i, j, k, dx, dy, denom, ray_length);

                                                
                

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
            }
                                              
            // Commenting out most of these except the intersection type, because added boundary check which writes all of these already
            //log_buffer[log_position_temp + LOG_ENTRY_SIZE + 0] = j;
            //log_buffer[log_position_temp + LOG_ENTRY_SIZE + 1] = k;
            log_buffer[log_position_temp + LOG_ENTRY_SIZE + 2] = intersection_type;
            //log_buffer[log_position_temp + LOG_ENTRY_SIZE + 3] = __double_as_longlong(xj);
            //log_buffer[log_position_temp + LOG_ENTRY_SIZE + 4] = __double_as_longlong(yj);
            //log_buffer[log_position_temp + LOG_ENTRY_SIZE + 5] = __double_as_longlong(zj);                                                      
            //log_buffer[log_position_temp + LOG_ENTRY_SIZE + 6] = __double_as_longlong(xk);
            //log_buffer[log_position_temp + LOG_ENTRY_SIZE + 7] = __double_as_longlong(yk);
            //log_buffer[log_position_temp + LOG_ENTRY_SIZE + 8] = __double_as_longlong(zk);
            log_position_temp += num_information_per_edge;  // ✅ Move to the next available slot for edges

        }

        if (valid_ray) break;

        attempt++;
        angle = fmod(angle + (2*M_PI / max_attempts), 2*M_PI);
        dx = cos(angle);
        dy = sin(angle);
    }

    log_buffer[log_position + 0] = i;
    log_buffer[log_position + 1] = ring_start;
    log_buffer[log_position + 2] = ring_end;
    log_buffer[log_position + 3] = __double_as_longlong(x);
    log_buffer[log_position + 4] = __double_as_longlong(y);
    log_buffer[log_position + 5] = __double_as_longlong(z);               
    log_buffer[log_position + 6] = inside;
    log_buffer[log_position + 7] = intersection_count;
    log_buffer[log_position + 8] = attempt;
    log_buffer[log_position + 9] = __double_as_longlong(angle);
    log_buffer[log_position + 10] = __double_as_longlong(dx);
    log_buffer[log_position + 11] = __double_as_longlong(dy);
    log_buffer[log_position + 12] = num_edges;
    log_buffer[log_position + 13] = edge_log_offset;  // ✅ Debugging: Log the offset
    log_buffer[log_position + 14] = log_position;  
                                              
    results[i] = inside ? 1 : 0;
}


''', 'one_to_one_pip')




### IMPORTANT NOTE: NOTE THAT WHEN CALLING THIS KERNEL, ALL DATA STORED ON GPU MUST BE CONTIGUOUSLY STORED IN MEMORY
### IMPORTANT NOTE: THE POLYGONS THAT ARE PASSED TO THIS KERNEL IS ASSUMED TO BE BUILT SUCH THAT THE FIRST AND LAST POINTS ARE THE SAME!
### IMPORTANT NOTE: POLYGONS WITH WITH EDGES THAT ARE TOO SHORT WILL BE SKIPPED AND THEREFORE MAY GIVE INCORRECT RESULTS! ENSURE THAT POLYGON EDGES ARE NOT TOO SHORT!
### IMPORTANT NOTE: "one_to_one_pip_kernel_advanced_reparameterized_version" is a version of the kernel that ALSO uses the reparameterized version of the mathematics which should in theory be more robust to regenerating rays
one_to_one_pip_kernel_advanced_reparameterized_version = cp.RawKernel(r'''
extern "C" __global__
void one_to_one_pip(const double* px, const double* py, const double* pz,
                    const double* poly_x, const double* poly_y, const double* poly_z,
                    const long long int* poly_part_offsets,
                    const long long int* edge_offsets,  // ✅ Now explicitly long long int
                    int* results, int num_points, int poly_offsets_size,
                    long long int* log_buffer) {  // ✅ Now explicitly long long int
    
    const int LOG_ENTRY_SIZE = 15;  // ✅ Increased to store debug info

    long long int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= num_points) return;

    int num_information_per_edge = 9;  // ✅ Number of information stored per edge                                                                  

    long long int edge_log_offset = edge_offsets[i];  // ✅ Now correctly stores large offsets
    long long int log_position = i * LOG_ENTRY_SIZE + edge_log_offset * num_information_per_edge;
    //printf("Thread %lld: | Log Position: %lld | edge_log_offset: %lld\n", i, log_position, edge_log_offset);
                                                                             
    
                                              

    if (i + 1 >= poly_offsets_size) {  
        log_buffer[log_position + 0] = i;
        log_buffer[log_position + 1] = -1;
        results[i] = 0;
        return;
    }

    double x = px[i];
    double y = py[i];
    double z = pz[i];

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
    #define EPSILON_REPARAM 1e-7    // Relative tolerance for deciding if dx and dy are "close"
    //
    

    double angle_perturbation = M_PI / 30; // this is to avoid perfectly vertical and perfectly horiztontal rays
    double angle = 0 + angle_perturbation;
    double dx = cos(angle);
    double dy = sin(angle);

    int max_attempts = 10;
    int attempt = 0;

    

    while (attempt < max_attempts) {
        inside = false;
        intersection_count = 0;
        bool valid_ray = true;
        long long int log_position_temp = log_position;  // Initialize the temporary log position
        int intersection_type = 0;
        
        // Check if the point being tested lies on the boundary of the polygon, only need to check first attempt since the point never moves, additional attempts only relevant to rays
        if (attempt == 0) {
            bool point_on_boundary = false;

            for (long long j = ring_start; j < ring_end - 1; j++) {
                intersection_type = 0;

                long long k = j + 1;

                double xj = poly_x[j], yj = poly_y[j], zj = poly_z[j];
                double xk = poly_x[k], yk = poly_y[k], zk = poly_z[k];

                // Compute the edge vector and its length
                double edge_dx = xk - xj;
                double edge_dy = yk - yj;
                double edge_length = sqrt(edge_dx * edge_dx + edge_dy * edge_dy);

                // If the edge is too short (degenerate), skip it (assume no intersection from this edge)
                if (edge_length < EPSILON) {
                    intersection_type = 3; // Set intersection type to degenerate (3), meaning we skipped this edge
                    //log_buffer[log_position_temp + LOG_ENTRY_SIZE + 2] = intersection_type;
                    //log_position_temp += num_information_per_edge;  // ✅ Move to the next available slot for edges
                    //continue;
                }
                else {                                                               

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
                }
                                                                                                      
                log_buffer[log_position_temp + LOG_ENTRY_SIZE + 0] = j;
                log_buffer[log_position_temp + LOG_ENTRY_SIZE + 1] = k;
                log_buffer[log_position_temp + LOG_ENTRY_SIZE + 2] = intersection_type;
                log_buffer[log_position_temp + LOG_ENTRY_SIZE + 3] = __double_as_longlong(xj);
                log_buffer[log_position_temp + LOG_ENTRY_SIZE + 4] = __double_as_longlong(yj);
                log_buffer[log_position_temp + LOG_ENTRY_SIZE + 5] = __double_as_longlong(zj);                                                      
                log_buffer[log_position_temp + LOG_ENTRY_SIZE + 6] = __double_as_longlong(xk);
                log_buffer[log_position_temp + LOG_ENTRY_SIZE + 7] = __double_as_longlong(yk);
                log_buffer[log_position_temp + LOG_ENTRY_SIZE + 8] = __double_as_longlong(zk);
                log_position_temp += num_information_per_edge;  // ✅ Move to the next available slot for edges
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
            angle = fmod(angle + (2*M_PI / max_attempts), 2*M_PI);
            dx = cos(angle);
            dy = sin(angle);
            continue;  // Retry with a new ray
        }

                                              


        // Check if the point is inside the polygon
        log_position_temp = log_position;  // Reset the temporary log position
                                              
        for (long long j = ring_start; j < ring_end - 1; j++) {
                                  
            long long k = j + 1;

            double xj = poly_x[j], yj = poly_y[j], zj = poly_z[j];
            double xk = poly_x[k], yk = poly_y[k], zk = poly_z[k];
                                                                      
            intersection_type = 0;


                                              
            // Compute the edge vector and its length
            double edge_dx = xk - xj;
            double edge_dy = yk - yj;
            double edge_length = sqrt(edge_dx * edge_dx + edge_dy * edge_dy);

            // If the edge is too short (degenerate), skip it (assume no intersection from this edge)
            if (edge_length < EPSILON) {
                intersection_type = 3; // Set intersection type to degenerate (3), meaning we skipped this edge
                //log_buffer[log_position_temp + LOG_ENTRY_SIZE + 2] = intersection_type;
                //log_position_temp += num_information_per_edge;  // ✅ Move to the next available slot for edges
                //continue;
            }
            else {
                         
                // Defines the scale of the edge for tolerance of denom
                double edge_scale = fmax(fabs(xj - xk), fabs(yj - yk));
                double tol_edge = EPSILON * edge_scale;
                
                                                                        
                                                                
                double t_edge, s_ray;
                double denom = (xj - xk) * dy - (yj - yk) * dx;                                                          
                
                                                                        
                // Decide whether to use cross product method or reparameterization.
                // If denom is larger than tol_edge, use the cross product method.
                // If dx and dy differ by more than EPSILON_REPARAM (relative to the larger of the two),
                // then use the dominant coordinate; otherwise, regenerate the ray.

                if (fabs(denom) > tol_edge) {
                    
                    //printf("Thread: %lld | j: %lld | k: %lld | (x_t, y_t): (%.2f, %.2f) | (xj, yj): (%.2f, %.2f) | (xk, yk): (%.2f, %.2f)\n", i, j, k, x ,y , xj, yj, xk, yk);
                    //printf("Thread: %lld | j: %lld | k: %lld | dx: %f | dy: %f | denom: %f | ray_length: %f\n", i, j, k, dx, dy, denom, ray_length);  
                    t_edge = ((x - xj) * (-dy) + (y - yj) * dx) / denom;
                    s_ray  = ((xk - xj) * (y - yj) - (yk - yj) * (x - xj)) / denom;                                   
                } else if (fabs(dx - dy) / fmax(fabs(dx), fabs(dy)) > EPSILON_REPARAM) {
                    if (fabs(dx) > fabs(dy)) {
                        // Use x-coordinate reparameterization
                        if (fabs(edge_dx) > EPSILON && fabs(dy) > EPSILON) {
                            t_edge = (x - xj) / edge_dx;
                            // Compute the intersection's y-coordinate using t_edge, then derive s_ray from y
                            s_ray = ( (yj + t_edge * edge_dy) - y ) / dy;
                        }
                    } else {
                        // Use y-coordinate reparameterization
                        if (fabs(edge_dy) > EPSILON && fabs(dy) > EPSILON) {
                            t_edge = (y - yj) / edge_dy;
                            // Compute the intersection's x-coordinate using t_edge, then derive s_ray from x
                            s_ray = ( (xj + t_edge * edge_dx) - x ) / dx;
                        }
                    }
                } else {
                    printf("Denom is zero, and reparameterization methods invalid on attempt: %d, (Thread: %lld)\n", attempt, i);
                    valid_ray = false;
                    break;
                }

                /*                                                       
                // If reparameterization wasn’t performed (either because dx and dy are too close
                // or because the chosen edge component was too small), fall back to the original method.
                if (!computed) {
                    double denom = (xj - xk) * dy - (yj - yk) * dx;
                    //printf("Thread: %lld | j: %lld | k: %lld | (x_t, y_t): (%.2f, %.2f) | (xj, yj): (%.2f, %.2f) | (xk, yk): (%.2f, %.2f)\n", i, j, k, x ,y , xj, yj, xk, yk);
                    //printf("Thread: %lld | j: %lld | k: %lld | dx: %f | dy: %f | denom: %f | ray_length: %f\n", i, j, k, dx, dy, denom, ray_length);
                    if (fabs(denom) > tol_edge) {
                        t_edge = ((x - xj) * (-dy) + (y - yj) * dx) / denom;
                        s_ray  = ((xk - xj) * (y - yj) - (yk - yj) * (x - xj)) / denom;
                    } else {
                        printf("Denom is zero on attempt: %d, (Thread: %lld)\n", attempt, i);
                        valid_ray = false;
                        break;
                    }
                }
                */

                // Now, if t_edge and s_ray fall within acceptable ranges, count this as an intersection.
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
                                                                      
            // Log the intersection type (and other data if needed)
                                                                      
            // Commenting out most of these except the intersection type, because added boundary check which writes all of these already
            //log_buffer[log_position_temp + LOG_ENTRY_SIZE + 0] = j;
            //log_buffer[log_position_temp + LOG_ENTRY_SIZE + 1] = k;
            log_buffer[log_position_temp + LOG_ENTRY_SIZE + 2] = intersection_type;
            //log_buffer[log_position_temp + LOG_ENTRY_SIZE + 3] = __double_as_longlong(xj);
            //log_buffer[log_position_temp + LOG_ENTRY_SIZE + 4] = __double_as_longlong(yj);
            //log_buffer[log_position_temp + LOG_ENTRY_SIZE + 5] = __double_as_longlong(zj);                                                      
            //log_buffer[log_position_temp + LOG_ENTRY_SIZE + 6] = __double_as_longlong(xk);
            //log_buffer[log_position_temp + LOG_ENTRY_SIZE + 7] = __double_as_longlong(yk);
            //log_buffer[log_position_temp + LOG_ENTRY_SIZE + 8] = __double_as_longlong(zk);
            log_position_temp += num_information_per_edge;  // ✅ Move to the next available slot for edges


                                              
            

            /*
                                                                                                        
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
            log_position_temp += num_information_per_edge;  // ✅ Move to the next available slot for edges
            
            */

                                                                      
        }

        if (valid_ray) break;

        attempt++;
        angle = fmod(angle + (2*M_PI / max_attempts), 2*M_PI);
        dx = cos(angle);
        dy = sin(angle);
    }

    log_buffer[log_position + 0] = i;
    log_buffer[log_position + 1] = ring_start;
    log_buffer[log_position + 2] = ring_end;
    log_buffer[log_position + 3] = __double_as_longlong(x);
    log_buffer[log_position + 4] = __double_as_longlong(y);
    log_buffer[log_position + 5] = __double_as_longlong(z);
    log_buffer[log_position + 6] = inside;
    log_buffer[log_position + 7] = intersection_count;
    log_buffer[log_position + 8] = attempt;
    log_buffer[log_position + 9] = __double_as_longlong(angle);
    log_buffer[log_position + 10] = __double_as_longlong(dx);
    log_buffer[log_position + 11] = __double_as_longlong(dy);
    log_buffer[log_position + 12] = num_edges;
    log_buffer[log_position + 13] = edge_log_offset;  // ✅ Debugging: Log the offset
    log_buffer[log_position + 14] = log_position;  
                                              
    results[i] = inside ? 1 : 0;
}


''', 'one_to_one_pip')





def one_to_one_point_in_polygon_cupy_arr_version(points, poly_points, poly_indices, block_size=256, log_sub_dirs_list = [], log_file_name="cuda_log.txt", include_edges_in_log = False, kernel_type = "one_to_one_pip_kernel_advanced", return_array_as = "cupy"):
    """
    Test each point against the corresponding polygon using CuPy arrays directly, note that this mapping is one-to-one.
    
    Parameters:
    - points: CuPy array of shape (num_points, 2) containing the x and y coordinates of the points.
    - poly_points: CuPy array of shape (num_vertices, 2) containing the x and y coordinates of the polygon vertices. Important! The first and last points must be the same!
    - poly_indices: CuPy array of shape (num_points, 2) containing the start and end indices of the polygon in poly_points. Important! The indices must be such that the end index is exclusive!
    - block_size: Block size for the CUDA kernel.
    - log_file_name: Name of the log file to write the debug information. If None, no log file is written to file. Important, the log file writing is quite slow, so turning off logging should be considered for performance.
    - include_edges_in_log: If True, the log file will include the edges checked for each point. This can be useful for debugging, but it can also significantly increase the log file size and more importantly vastly increase the computation time.
    - kernel_type: The type of kernel to use. The default is "one_to_one_pip_kernel_advanced" which is the most advanced version of the kernel. The other option is "one_to_one_pip_kernel_advanced_reparameterized_version" which is a version of that kernel that ALSO uses the reparameterized version of the mathematics which should in theory be more robust to regenerating rays.
    - return_array_as: The type of array to return the results as. The default is "cupy" which returns the results as a CuPy array. The other option is anything (but should type "numpy" for clarity) which returns the results as a NumPy array. Cupy is slightly faster because there is no conversion step before return, but then the array remains on gpu memory.
    """

    num_points = cp.int32(points.shape[0])

    points_x = points[:, 0]
    points_y = points[:, 1]
    points_z = points[:, 2]
    points_x = cp.ascontiguousarray(points_x, dtype=cp.float64)
    points_y = cp.ascontiguousarray(points_y, dtype=cp.float64)
    points_z = cp.ascontiguousarray(points_z, dtype=cp.float64)

    poly_x = poly_points[:, 0]
    poly_y = poly_points[:, 1]
    poly_z = poly_points[:, 2]
    poly_x = cp.ascontiguousarray(poly_x, dtype=cp.float64)
    poly_y = cp.ascontiguousarray(poly_y, dtype=cp.float64)
    poly_z = cp.ascontiguousarray(poly_z, dtype=cp.float64)


    # Faster to compute on CPU using NumPy then convert back to CuPy
    """
    poly_part_offsets = cp.zeros(poly_indices.shape[0] + 1, dtype=cp.int64)
    poly_part_offsets[1:] = cp.cumsum(poly_indices[:, 1] - poly_indices[:, 0]).astype(cp.int64)
    #poly_part_offsets = cp.ascontiguousarray(poly_part_offsets, dtype=cp.int64)
    """

    # Compute cumsum on CPU using NumPy
    poly_part_offsets_np = np.cumsum(np.hstack(([0], (poly_indices.get()[:, 1] - poly_indices.get()[:, 0]))), dtype=np.int64)

    # Convert back to CuPy
    poly_part_offsets = cp.asarray(poly_part_offsets_np)
    poly_part_offsets = cp.ascontiguousarray(poly_part_offsets, dtype=cp.int64)

    """
    print(cp.array_equal(poly_part_offsets, poly_part_offsets_1))  # True

    if cp.array_equal(poly_part_offsets, poly_part_offsets_1) == False:
        input("Enter to continue")
    """

    poly_offsets_size = cp.int32(poly_part_offsets.shape[0])

    results = cp.zeros(num_points, dtype=cp.int32)
    results = cp.ascontiguousarray(results, dtype=cp.int32)

    log_entry_size = 15  # ✅ Number of static log information stored per thread prior to edge info. IMPORTANT: MUST MATCH KERNEL DEFINITION
    num_information_per_edge = 9 # ✅ Number of information stored per edge. IMPORTANT: MUST MATCH KERNEL DEFINITION

    num_edges_per_polygon = (poly_indices[:, 1] - poly_indices[:, 0]) - 1  
    #edge_offsets = cp.zeros(num_points.item() + 1, dtype=cp.int64) ## which is faster?
    edge_offsets = cp.zeros(num_points + 1, dtype=cp.int64)  ## which is faster?  

    edge_offsets[1:] = cp.cumsum(num_edges_per_polygon).astype(cp.int64)
    edge_offsets = cp.ascontiguousarray(edge_offsets, dtype=cp.int64)  

    #total_edge_entries = edge_offsets[-1].item() * num_information_per_edge ## which is faster?
    total_edge_entries = int(edge_offsets[-1].get()) * num_information_per_edge ## which is faster?

    log_buffer = cp.zeros(num_points.item() * log_entry_size + total_edge_entries, dtype=cp.int64)
    log_buffer = cp.ascontiguousarray(log_buffer, dtype=cp.int64)

    grid_size = (num_points.item() + block_size - 1) // block_size

    ### IMPORTANT NOTE: WHEN CALLING THIS KERNEL, ALL DATA STORED ON GPU MUST BE CONTIGUOUSLY STORED IN MEMORY
    ### IMPORTANT NOTE: THE POLYGONS THAT ARE PASSED TO THIS KERNEL IS ASSUMED TO BE BUILT SUCH THAT THE FIRST AND LAST POINTS ARE THE SAME!
    ### IMPORTANT NOTE: POLYGONS WITH WITH EDGES THAT ARE TOO SHORT WILL BE SKIPPED AND THEREFORE MAY GIVE INCORRECT RESULTS! ENSURE THAT POLYGON EDGES ARE NOT TOO SHORT!
    if kernel_type == "one_to_one_pip_kernel_advanced":
        one_to_one_pip_kernel_advanced(
            (grid_size,), (block_size,),
            (points_x, points_y, points_z, poly_x, poly_y, poly_z, poly_part_offsets, edge_offsets, results, num_points, poly_offsets_size, log_buffer)
        )
    elif kernel_type == "one_to_one_pip_kernel_advanced_reparameterized_version":
        one_to_one_pip_kernel_advanced_reparameterized_version(
            (grid_size,), (block_size,),
            (points_x, points_y, points_z, poly_x, poly_y, poly_z, poly_part_offsets, edge_offsets, results, num_points, poly_offsets_size, log_buffer)
        )

    if log_file_name is not None:
        logs_host = log_buffer.get()

        log_dir = pathlib.Path(__file__).parents[0].joinpath("cuda_containment_logs")
        for log_sub_dir in log_sub_dirs_list:
            log_dir = log_dir.joinpath(log_sub_dir)
        log_dir.mkdir(exist_ok=True, parents=True)
        log_file = log_dir.joinpath(log_file_name)

        with open(log_file, "w") as f:
            # Move the entire edge_offsets array to CPU **once** before the loop
            edge_offsets_host = edge_offsets.get()

            for i in range(num_points.item()):
                # ✅ Compute `log_position` exactly as in the CUDA kernel
                #edge_log_offset = edge_offsets[i].item() * num_information_per_edge ## which is faster?
                edge_log_offset = edge_offsets_host[i] * num_information_per_edge ## which is faster? ... This one is slightly faster

                log_position = i * log_entry_size + edge_log_offset  # ✅ Now perfectly aligned with kernel

                meta_start = log_position
                static_meta_end = meta_start + log_entry_size
                log = logs_host[meta_start:static_meta_end]

                thread_id = log[0]
                ring_start = log[1]
                ring_end = log[2]
                x_coord = struct.unpack('d', struct.pack('q', log[3]))[0]
                y_coord = struct.unpack('d', struct.pack('q', log[4]))[0]
                z_coord = struct.unpack('d', struct.pack('q', log[5]))[0]
                inside_flag = log[6]
                intersection_count = log[7]
                retries = log[8]
                angle = struct.unpack('d', struct.pack('q', log[9]))[0]
                dx = struct.unpack('d', struct.pack('q', log[10]))[0]
                dy = struct.unpack('d', struct.pack('q', log[11]))[0]
                num_edges = log[12]
                log_offset_debug = log[13]
                log_position_debug = log[14]

                # ✅ Extract checked edges using `edge_offsets[i]`
                if include_edges_in_log == True:
                    edge_log_start = static_meta_end
                    edge_log_end = edge_log_start + num_edges_per_polygon[i].item() * num_information_per_edge
                    checked_edges = logs_host[edge_log_start:edge_log_end].reshape(-1, num_information_per_edge)
                    
                    """
                    checked_edges[:, 3:num_information_per_edge] = checked_edges[:, 3:num_information_per_edge].view(np.float64)
                    # Optionally round:
                    checked_edges[:, 3:num_information_per_edge] = np.around(checked_edges[:, 3:num_information_per_edge], decimals=2)

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

                    double_info_start_index = 3 
                    checked_edges_list_converted_long_to_double = format_edges_for_point_ver2(checked_edges, double_info_start_index, num_information_per_edge)


                    # ✅ Write correct logs to the file
                    f.write(f"[Thread {thread_id}] ✅ Checked Point ({x_coord:.2f}, {y_coord:.2f}, {z_coord:.2f}) -> "
                            f"ring_start={ring_start}, ring_end={ring_end}, num_edges={num_edges}, Inside={inside_flag}, "
                            f"Intersections={intersection_count}, Retries={retries}, dx={dx:.3f}, dy={dy:.3f}, "
                            f"angle={angle:.3f}, edge_log_offset={log_offset_debug}, log_position={log_position_debug}, "
                            f"Checked Edges (j,k,intersection type, edge_xj,yj,zj, edge_xk,yk,zk)={checked_edges_list_converted_long_to_double}\n")
                else:

                    f.write(f"[Thread {thread_id}] ✅ Checked Point ({x_coord:.2f}, {y_coord:.2f}, {z_coord:.2f}) -> "
                            f"ring_start={ring_start}, ring_end={ring_end}, num_edges={num_edges}, Inside={inside_flag}, "
                            f"Intersections={intersection_count}, Retries={retries}, dx={dx:.3f}, dy={dy:.3f}, "
                            f"angle={angle:.3f}, edge_log_offset={log_offset_debug}, log_position={log_position_debug}\n")

    if return_array_as == "cupy":
        return results
    else:
        return results.get()




def format_edges_for_point_upcast_ver(checked_edges, double_info_start_index, num_information_per_edge, decimals=3):
    # Create a copy to avoid modifying the original array
    conv = checked_edges.copy()
    
    # Preserve the first columns as integers
    int_cols = conv[:, :double_info_start_index].astype(np.int64)  # Integer columns
    
    # Convert only the floating-point columns
    int_view = conv[:, double_info_start_index:num_information_per_edge].astype(np.int64)
    float_view = np.frombuffer(int_view.tobytes(), dtype=np.float64).reshape(int_view.shape)
    
    # Apply rounding to floating-point values
    float_view = np.around(float_view, decimals=decimals)
    
    # Combine integer and float parts back together
    combined = np.hstack([int_cols, float_view])  # Keep int columns as int, float columns as float

    # Convert each row into a formatted string
    row_strings = [
        ", ".join(map(str, row)) for row in combined  # Vectorized conversion to string
    ]

    # Join all rows with the separator " | "
    return " | ".join(row_strings)


# This function is the quickest
def format_edges_for_point_ver2(checked_edges, double_info_start_index, num_information_per_edge, decimals=3):
    # Convert the first columns to integers
    int_cols = checked_edges[:, :double_info_start_index].astype(np.int64)
    
    # Convert the remaining columns from int64 to float64
    int_view = checked_edges[:, double_info_start_index:num_information_per_edge].astype(np.int64)
    float_view = np.frombuffer(int_view.tobytes(), dtype=np.float64).reshape(int_view.shape)
    
    # Apply rounding
    float_view = np.around(float_view, decimals=decimals)
    
    # Efficiently format each row while keeping integer and float types
    row_strings = []
    for int_part, float_part in zip(int_cols, float_view):
        int_str = ", ".join(map(str, int_part))  # Convert integer columns to string
        float_str = ", ".join(f"{val:.{decimals}f}" for val in float_part)  # Convert float columns with formatting
        row_strings.append(f"{int_str}, {float_str}")  # Combine both parts

    # Join all rows with separator
    return " | ".join(row_strings)




def format_edges_for_point_optimized(checked_edges, double_info_start_index, num_information_per_edge, decimals=3):
    """
    Efficiently formats the first columns as integers and the last columns as floats.

    Parameters:
    - checked_edges: NumPy array where first `double_info_start_index` columns are integers
                     and remaining columns are float values (stored as int64 and reinterpreted).
    - double_info_start_index: Number of columns that should be formatted as integers.
    - decimals: Number of decimal places for float formatting.

    Returns:
    - A formatted string where each row is separated by " | " and columns are comma-separated.
    """

    # Step 1: Extract integer and float columns
    int_cols = checked_edges[:, :double_info_start_index].astype(np.int64)
    float_int_view = checked_edges[:, double_info_start_index:num_information_per_edge].astype(np.int64)

    # Step 2: Convert integer part to strings efficiently
    int_str_arr = np.char.mod('%d', int_cols)  # Vectorized integer conversion

    # Step 3: Convert float values from int64 bit representation
    float_view = np.frombuffer(float_int_view.tobytes(), dtype=np.float64).reshape(float_int_view.shape)

    # Step 4: Round floats and convert them efficiently
    float_view = np.around(float_view, decimals=decimals)
    float_str_arr = np.char.mod(f"%.{decimals}f", float_view)  # Vectorized float formatting

    # Step 5: Stack the integer and float string arrays column-wise
    combined_str_arr = np.hstack((int_str_arr, float_str_arr))  # ✅ No broadcasting issues

    # Step 6: Join columns within each row
    row_strings = np.apply_along_axis(lambda row: ", ".join(row), axis=1, arr=combined_str_arr)

    # Step 7: Join all rows with separator
    return " | ".join(row_strings)




def format_edges_for_point_optimized_ver2(checked_edges, double_info_start_index, decimals=3):
    """
    Efficiently formats the first columns as integers and the last columns as properly rounded floats.

    Parameters:
    - checked_edges: NumPy array where first `double_info_start_index` columns are integers
                     and remaining columns are float values (stored as int64 and reinterpreted).
    - double_info_start_index: Number of columns that should be formatted as integers.
    - decimals: Number of decimal places for float formatting.

    Returns:
    - A formatted string where each row is separated by " | " and columns are comma-separated.
    """

    # Step 1: Extract integer and float columns
    int_cols = checked_edges[:, :double_info_start_index].astype(np.int64)
    float_int_view = checked_edges[:, double_info_start_index:].astype(np.int64)

    # Step 2: Convert integer part to strings efficiently
    int_str_arr = np.char.mod('%d', int_cols)  # Vectorized integer conversion

    # Step 3: Convert float values from int64 representation
    float_view = np.frombuffer(float_int_view.tobytes(), dtype=np.float64).reshape(float_int_view.shape)

    # Step 4: Round the floats (ensuring proper decimal precision)
    float_view = np.round(float_view, decimals=decimals)

    # Step 5: Convert rounded floats **explicitly to strings with forced formatting**
    float_format = f"%.{decimals}f"
    float_str_arr = np.core.defchararray.mod(float_format, float_view)  # ✅ Forces correct rounding in output

    # Step 6: Stack the integer and float string arrays column-wise
    combined_str_arr = np.hstack((int_str_arr, float_str_arr))  # ✅ No broadcasting issues

    # Step 7: Join columns within each row
    row_strings = np.apply_along_axis(lambda row: ", ".join(row), axis=1, arr=combined_str_arr)

    # Step 8: Join all rows with separator
    return " | ".join(row_strings)




def custom_point_containment_mother_function(list_of_relative_structures_containting_list_of_constant_zslices_arrays,
                                            points_to_test_3d_arr_or_list_of_2d_arrays,
                                            test_struct_to_relative_struct_1d_mapping_array,
                                            constant_z_slice_polygons_handler_option = 'auto-close-if-open',
                                            remove_consecutive_duplicate_points_in_polygons = False,
                                            log_sub_dirs_list = [],
                                            log_file_name = "cuda_log.txt",
                                            include_edges_in_log = False,
                                            kernel_type = "one_to_one_pip_kernel_advanced_reparameterized_version"):

    prepper_output_tuple = test_points_against_polygons_cupy_arr_version_prepper(list_of_relative_structures_containting_list_of_constant_zslices_arrays,
                                                          points_to_test_3d_arr_or_list_of_2d_arrays,
                                                          test_struct_to_relative_struct_1d_mapping_array,
                                                          constant_z_slice_polygons_handler_option = constant_z_slice_polygons_handler_option,
                                                          remove_consecutive_duplicate_points_in_polygons = remove_consecutive_duplicate_points_in_polygons)

    if prepper_output_tuple[0].ndim == 2:
        result_cp_arr = test_points_against_polygons_cupy_2d_arr_version(prepper_output_tuple[0], 
                                 prepper_output_tuple[3],
                                 prepper_output_tuple[4], 
                                 prepper_output_tuple[1], 
                                 prepper_output_tuple[2],
                                 log_sub_dirs_list = log_sub_dirs_list,
                                 log_file_name=log_file_name,
                                 include_edges_in_log = include_edges_in_log,
                                 kernel_type=kernel_type)
        
        return result_cp_arr, prepper_output_tuple
    
    elif prepper_output_tuple[0].ndim == 3:
        result_cp_arr = test_points_against_polygons_cupy_3d_arr_version(prepper_output_tuple[0], 
                                 points_to_test_3d_arr_or_list_of_2d_arrays, 
                                 prepper_output_tuple[1], 
                                 prepper_output_tuple[2],
                                 log_sub_dirs_list = log_sub_dirs_list,
                                 log_file_name=log_file_name,
                                 include_edges_in_log = include_edges_in_log,
                                 kernel_type=kernel_type)
        
        return result_cp_arr, prepper_output_tuple
    else:
        raise ValueError("The nearest zslice index and values array must be either a 2d or 3d array!")



def test_points_against_polygons_cupy_arr_version_prepper(list_of_relative_structures_containting_list_of_constant_zslices_arrays,
                                                          points_to_test_3d_arr_or_list_of_2d_arrays,
                                                          test_struct_to_relative_struct_1d_mapping_array,
                                                          constant_z_slice_polygons_handler_option = 'auto-close-if-open',
                                                          remove_consecutive_duplicate_points_in_polygons = False):
    """
    This function prepares the data for the custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p.test_points_against_polygons_cupy_3d_arr_version or test_points_against_polygons_cupy_2d_arr_version function.
    It takes the list of relative structures containing the constant z slices arrays and the points to test, and then it closes the polygons if they are not closed, and then converts the structures to 2d arrays with accompanying indices arrays.
    It then extracts the z values of all slices of every relative structure and finds the nearest z slices for every point.

    Parameters:
    - list_of_relative_structures_containting_list_of_constant_zslices_arrays: A list of relative structures where each relative structure is a list of constant z slices arrays.
    - points_to_test_3d_arr_or_list_of_2d_arrays: Either a list of 2d arrays or a 3d array where the first dimension (or element of the list) are constant 2 dimensional arrays assoicated with each structure to test against in the relative structures list.
    - test_struct_to_relative_struct_2d_mapping_array: A 1d array of length the number of test objects, so either same as the number of slices of the 3d test array or same number of elements of the list of 2d arrays. Each element indicates which test structure (indicated by the index of the array itself) should be tested against which relative structure (indicated by the value stored at that index). This is used to map the test structures to test against the correct relative structure. Therefore the number of relative structures need not be the same as the number of test structures.
    - constant_z_slice_polygons_handler_option: Note that for the test_points_against_polygons_cupy_3d_arr_version or test_points_against_polygons_cupy_2d_arr_version function that this function feeds, all polygons must be closed. The option for handling the constant z slice polygons. The default is 'auto-close-if-open' which closes the polygons if they are not closed. The other options are 'close-all' which closes every single polygon, and None which makes no changes to the polygons.
    
    Returns:
    - nearest_zslice_index_and_values_3d_arr or nearest_zslice_index_and_values_2d_arr: Either a 3d array Shape: (num_test_structures, num_test_points_per, 4) or a 2d array Shape: (total num test points, 4) indicating for each test point the assoicated relative structure index, the nearest z index on that structure, the nearest z val on that structure and a flag whether the point lies outside the z extent of the relative structure.
    - all_structures_list_of_2d_arr: A list of 2d arrays where each 2d array is a structure with accompanying indices array where each row is a zslice with two indices indicating the start_index and end_index+1 of that slice (note that the +1 is for easy python slicing, therefore end index +1 index itself (point) is NOT in the slice!)
    - all_structures_slices_indices_list: A list of indices arrays where each array is a structure with accompanying indices array where each row is a zslice with two indices indicating the start_index and end_index+1 of that slice (note that the +1 is for easy python slicing, therefore end index +1 index itself (point) is NOT in the slice!)
    - points_to_test_2d_arr: only output if the input test points were a list of 2d arrays
    - points_to_test_indices_arr: only output if the input test points were a list of 2d arrays, accompnaies points_to_test_2d_arr
    """

    
    ### Step 0: Convert the points to test to a 2d array with accompanying 2d indices array, each row is a point with two indices indicating the start_index and end_index+1 of that point (note that the +1 is for easy python slicing, therefore end index +1 index itself (point) is NOT in the slice!)
    if isinstance(points_to_test_3d_arr_or_list_of_2d_arrays, list):
        points_to_test_2d_arr, points_to_test_indices_arr = polygon_dilation_helpers_numpy.convert_to_2d_array_and_indices_numpy(points_to_test_3d_arr_or_list_of_2d_arrays)



    num_relative_structures = len(list_of_relative_structures_containting_list_of_constant_zslices_arrays)


    ### Step 1: Remove consecutive duplicate points
    if remove_consecutive_duplicate_points_in_polygons == True:
        list_of_relative_structures_containting_list_of_constant_zslices_arrays_no_consecutive_duplicates = [None]*num_relative_structures
        for relative_structure_index, relative_structure_zslices_list in enumerate(list_of_relative_structures_containting_list_of_constant_zslices_arrays):
            relative_structure_zslices_list_no_consecutive_duplicates = [None]*len(relative_structure_zslices_list)
            for sp_rel_struct_zslice_index, zslice_arr in enumerate(relative_structure_zslices_list):
                # Remove consecutive duplicate points
                relative_structure_zslices_list_no_consecutive_duplicates[sp_rel_struct_zslice_index] = polygon_dilation_helpers_numpy.remove_consecutive_duplicate_points_numpy(zslice_arr)

            list_of_relative_structures_containting_list_of_constant_zslices_arrays_no_consecutive_duplicates[relative_structure_index] = relative_structure_zslices_list_no_consecutive_duplicates
    else:
        # Make no changes
        list_of_relative_structures_containting_list_of_constant_zslices_arrays_no_consecutive_duplicates = copy.deepcopy(list_of_relative_structures_containting_list_of_constant_zslices_arrays)

    

    ### Step 2: Close all constant slice polygons in every structure
    if constant_z_slice_polygons_handler_option == 'auto-close-if-open':
        # Check whether each relative structure has closed polygons, and close them if not
        list_of_relative_structures_containting_list_of_constant_zslices_arrays_closed_polygons = [None]*num_relative_structures
        for index, relative_structure_zslices_list in enumerate(list_of_relative_structures_containting_list_of_constant_zslices_arrays):
            relative_structure_zslices_list_closed_polygons = [None]*len(relative_structure_zslices_list)
            for i, zslice_arr in enumerate(relative_structure_zslices_list):
                # Check if the first and last points are the same
                if not np.all(zslice_arr[0] == zslice_arr[-1]):
                    # Append the first point to the end of the array
                    relative_structure_zslices_list_closed_polygons[i] = np.append(zslice_arr, zslice_arr[0][np.newaxis, :], axis=0)
                else:
                    # Make no changes if detected to be closed
                    relative_structure_zslices_list_closed_polygons[i] = zslice_arr

            list_of_relative_structures_containting_list_of_constant_zslices_arrays_closed_polygons[index] = relative_structure_zslices_list_closed_polygons

    elif constant_z_slice_polygons_handler_option == 'close-all':
        # Close every polygon
        list_of_relative_structures_containting_list_of_constant_zslices_arrays_closed_polygons = [None]*num_relative_structures
        for index, relative_structure_zslices_list in enumerate(list_of_relative_structures_containting_list_of_constant_zslices_arrays):
            relative_structure_zslices_list_closed_polygons = [None]*len(relative_structure_zslices_list)
            for i, zslice_arr in enumerate(relative_structure_zslices_list):
                # append the first point to the end of the array
                relative_structure_zslices_list_closed_polygons[i] = np.append(zslice_arr, zslice_arr[0][np.newaxis, :], axis=0)

            list_of_relative_structures_containting_list_of_constant_zslices_arrays_closed_polygons[index] = relative_structure_zslices_list_closed_polygons
    elif constant_z_slice_polygons_handler_option == None:
        # Make no changes
        list_of_relative_structures_containting_list_of_constant_zslices_arrays_closed_polygons = list_of_relative_structures_containting_list_of_constant_zslices_arrays

    

    # Step 3: Convert every structure to a 2d array with accompanying indices array
    all_structures_list_of_2d_arr = [None]*num_relative_structures
    all_structures_slices_indices_list = [None]*num_relative_structures
    for index, relative_structure_zslices_list_closed_polygons in enumerate(list_of_relative_structures_containting_list_of_constant_zslices_arrays_closed_polygons):
        # This converts the structure from a list of constant z slice arrays to a 2d array with a partner indices array where each row is a zslice with two indices indicating the start_index and end_index+1 of that slice (note that the +1 is for easy python slicing, therefore end index +1 index itself (point) is NOT in the slice!)
        relative_structure_closed_polygons_2d_arr, relative_structure_closed_polygons_indices_arr = polygon_dilation_helpers_numpy.convert_to_2d_array_and_indices_numpy(relative_structure_zslices_list_closed_polygons)
        all_structures_list_of_2d_arr[index] = relative_structure_closed_polygons_2d_arr
        all_structures_slices_indices_list[index] = relative_structure_closed_polygons_indices_arr
    

    # Step 4: Get the z values of all slices of every relative structure
    relative_structures_list_of_zvals_1d_arrays = polygon_dilation_helpers_numpy.extract_constant_z_values_arr_version(all_structures_list_of_2d_arr, all_structures_slices_indices_list)


    if isinstance(points_to_test_3d_arr_or_list_of_2d_arrays, list):
        # Step 5: Find the nearest z slices for every point (list of 2d arrays input)    
        nearest_zslice_index_and_values_2d_arr = polygon_dilation_helpers_numpy.nearest_zslice_vals_and_indices_all_structures_2d_point_arr(relative_structures_list_of_zvals_1d_arrays, points_to_test_2d_arr, points_to_test_indices_arr, test_struct_to_relative_struct_1d_mapping_array)
        
        return nearest_zslice_index_and_values_2d_arr, all_structures_list_of_2d_arr, all_structures_slices_indices_list, points_to_test_2d_arr, points_to_test_indices_arr
    else:
        # Step 5: Find the nearest z slices for every point (3d array input)   
        nearest_zslice_index_and_values_3d_arr = polygon_dilation_helpers_numpy.nearest_zslice_vals_and_indices_all_structures_3d_point_arr(relative_structures_list_of_zvals_1d_arrays, points_to_test_3d_arr_or_list_of_2d_arrays, test_struct_to_relative_struct_1d_mapping_array)
        
        return nearest_zslice_index_and_values_3d_arr, all_structures_list_of_2d_arr, all_structures_slices_indices_list
    









def test_points_against_polygons_cupy_3d_arr_version(nearest_zslice_index_and_values_3d_arr, 
                                 points_to_test_3d_arr, 
                                 all_structures_list_of_2d_arr, 
                                 all_structures_slices_indices_list,
                                 log_sub_dirs_list = [],
                                 log_file_name="cuda_log.txt",
                                 include_edges_in_log = False,
                                 kernel_type="one_to_one_pip_kernel_advanced"):
    """
    Important! This verion is to handle test structures with EQUAL number of points per test structure. 

    Test points against polygons using CuPy arrays directly. 
    
    Parameters:
    - nearest_zslice_index_and_values_3d_arr: Array of the relative structure index for that test structure, nearest z indices and z values for each test point (row), as well as a flag indicating if point is outside z extent. Shape is (num_test_structures, num_points_per_trial, 4). 
    - points_to_test_3d_arr: 3D array of test structures. Shape is (num_test_structures, num_points_per_structure (all equal), 3).
    - all_structures_list_of_2d_arr: List (num_trials is the number of elements in the list), each element is an (N_i, 3) array for each trial (i), where N_i is the number of points (3-element row vectors) representing the structure associated with trial (i).
    - all_structures_slices_indices_list: List (num_trials is the number of elements in the list), each element is an (Z_i, 2) array for each trial (i), where Z_i is the number of constant plane slices of the structure, and each row of that array is the start_index and end_index+1 (done so for easy python slicing) describing how to access a constant plane slice of the assoicated (N_i, 3) of the above (nominal_and_dilated_structures_list_of_2d_arr) list elements.
    - log_sub_dirs_list: List of subdirectories for the log file.
    - log_file_name: Name of the log file to write the debug information. If None, no log file is written to file. Important, the log file writing is quite slow, so turning off logging should be considered for performance.
    - include_edges_in_log: If True, the log file will include the edges checked for each point. (slower)
    - kernel_type: The type of kernel to use. The default is "one_to_one_pip_kernel_advanced" which is the most advanced version of the kernel. The other option is "one_to_one_pip_kernel_advanced_reparameterized_version" which is a version of that kernel that ALSO uses the reparameterized version of the mathematics which should in theory be more robust to regenerating rays.
    Returns:
    - result_cp_arr: Array indicating whether each point is inside the corresponding polygon. Shape of the output array is (num_trials, num_points_per_trial) full of True/False values.
    """
    num_test_structures = nearest_zslice_index_and_values_3d_arr.shape[0]
    num_points_in_every_test_structure = nearest_zslice_index_and_values_3d_arr.shape[1]
    
    # Initialize an array to store the results
    result_cp_arr = cp.zeros((num_test_structures, num_points_in_every_test_structure), dtype=cp.bool_)
    
    # Flatten the input arrays for easier processing
    flat_points = points_to_test_3d_arr[:, :, :3].reshape(-1, 3)
    
    # Filter out invalid points using the flag in column 3 (1 = out of bounds, 0 = valid)
    valid_mask = nearest_zslice_index_and_values_3d_arr[:, :, 3].flatten() == 0
    
    # Get the valid points
    valid_points = flat_points[valid_mask]
    
    # Create CuPy arrays for the points
    valid_points_cp_arr = cp.array(valid_points)
    
    # This method is slightly slower
    """
    st = time.time()
    # Create CuPy arrays for the polygons and indices
    poly_points = []
    poly_indices = []
    current_index = 0

    # Precompute the necessary values outside the loop
    valid_indices = np.where(valid_mask)[0]
    valid_points_nearest_zslice_index_and_values_3d_arr = nearest_zslice_index_and_values_3d_arr.reshape(-1, 3)[valid_indices]

    for nearest_zslice_index_and_values in valid_points_nearest_zslice_index_and_values_3d_arr:
        relative_structure_index = nearest_zslice_index_and_values[0].astype(int)
        nearest_zslice_index = nearest_zslice_index_and_values[1].astype(int)
        start_idx, end_idx = all_structures_slices_indices_list[relative_structure_index][nearest_zslice_index]
        polygon_points = all_structures_list_of_2d_arr[relative_structure_index][start_idx:end_idx, :2]
        poly_points.append(polygon_points)
        poly_indices.append([current_index, current_index + len(polygon_points)])
        current_index += len(polygon_points)
    
    et = time.time()
    print("Time taken to prepare the polygons (1): ", et-st)
    """
    
    # Create CuPy arrays for the polygons and indices
    poly_points = []
    poly_indices = []
    current_index = 0
    
    for trial_index in range(num_test_structures):
        for point_index in range(num_points_in_every_test_structure):
            if not valid_mask[trial_index * num_points_in_every_test_structure + point_index]:
                continue
            relative_structure_index = int(nearest_zslice_index_and_values_3d_arr[trial_index, point_index, 0])
            nearest_zslice_index = int(nearest_zslice_index_and_values_3d_arr[trial_index, point_index, 1])
            start_idx, end_idx = all_structures_slices_indices_list[relative_structure_index][nearest_zslice_index]
            polygon_points = all_structures_list_of_2d_arr[relative_structure_index][start_idx:end_idx, :3]
            poly_points.append(polygon_points)
            poly_indices.append([current_index, current_index + len(polygon_points)])
            current_index += len(polygon_points)
    
    poly_points = cp.array(np.vstack(poly_points))
    poly_indices = cp.array(poly_indices)

    #poly_points_old = cp.array(np.vstack(poly_points_old))
    #poly_indices_old = cp.array(poly_indices_old)

    # compare and ensure they are equivalent
    #assert cp.all(poly_points == poly_points_old)
    #assert cp.all(poly_indices == poly_indices_old)

    
    # Test each point against the corresponding polygon
    valid_results = one_to_one_point_in_polygon_cupy_arr_version(valid_points_cp_arr, 
                                                                 poly_points, 
                                                                 poly_indices, 
                                                                 log_sub_dirs_list = log_sub_dirs_list, 
                                                                 log_file_name=log_file_name, 
                                                                 include_edges_in_log = include_edges_in_log, 
                                                                 kernel_type=kernel_type, 
                                                                 return_array_as="cupy")
    
    # Map the valid results back to the original result array
    result_cp_arr_flat = result_cp_arr.flatten()
    result_cp_arr_flat[valid_mask] = valid_results
    
    # Reshape the result array back to the original shape
    result_cp_arr = result_cp_arr_flat.reshape(num_test_structures, num_points_in_every_test_structure)
    
    return result_cp_arr # result is 2d array with shape (num_test_structures, num_points_in_every_test_structure)




def test_points_against_polygons_cupy_2d_arr_version(nearest_zslice_index_and_values_2d_arr, 
                                 points_to_test_2d_arr,
                                 points_to_test_indices_arr, 
                                 all_structures_list_of_2d_arr, 
                                 all_structures_slices_indices_list,
                                 log_sub_dirs_list = [],
                                 log_file_name="cuda_log.txt",
                                 include_edges_in_log = False,
                                 kernel_type="one_to_one_pip_kernel_advanced"):
    """
    Important! This verion is to handle test structures with UNEQUAL number of points per test structure. 

    Test points against polygons using CuPy arrays directly.
    
    Parameters:
    - nearest_zslice_index_and_values_3d_arr: Array of the relative structure index for that test structure, nearest z indices and z values for each test point (row). Shape is (num_test_structures, num_points_per_trial, 3). 
    - points_to_test_2d_arr: 2D array of test structures. Shape is (total num test points, 3).
    - points_to_test_indices_arr: 2D array of indices indicating the start and end index+1 of each test structure in the points_to_test_2d_arr. Shape is (num_test_structures, 2).
    - all_structures_list_of_2d_arr: List (num_trials is the number of elements in the list), each element is an (N_i, 3) array for each trial (i), where N_i is the number of points (3-element row vectors) representing the structure associated with trial (i).
    - all_structures_slices_indices_list: List (num_trials is the number of elements in the list), each element is an (Z_i, 2) array for each trial (i), where Z_i is the number of constant plane slices of the structure, and each row of that array is the start_index and end_index+1 (done so for easy python slicing) describing how to access a constant plane slice of the assoicated (N_i, 3) of the above (nominal_and_dilated_structures_list_of_2d_arr) list elements.
    - log_sub_dirs_list: List of subdirectories for the log file.
    - log_file_name: Name of the log file to write the debug information. If None, no log file is written to file. Important, the log file writing is quite slow, so turning off logging should be considered for performance.
    - include_edges_in_log: If True, the log file will include the edges checked for each point. (slower)
    - kernel_type: The type of kernel to use. The default is "one_to_one_pip_kernel_advanced" which is the most advanced version of the kernel. The other option is "one_to_one_pip_kernel_advanced_reparameterized_version" which is a version of that kernel that ALSO uses the reparameterized version of the mathematics which should in theory be more robust to regenerating rays.
    Returns:
    - result_cp_arr: Array indicating whether each point is inside the corresponding polygon. Shape of the output array is (total_num_points) full of True/False values.
    """
    total_num_points_to_test = points_to_test_2d_arr.shape[0]
    #num_test_structures = points_to_test_indices_arr.shape[0]
    #num_points_in_every_test_structure = nearest_zslice_index_and_values_3d_arr.shape[1]
    
    # Initialize an array to store the results
    result_cp_arr = cp.zeros((total_num_points_to_test), dtype=cp.bool_)
    
    # Flatten the input arrays for easier processing
    flat_points = points_to_test_2d_arr[:, :3].reshape(-1, 3)
    
    # Filter out invalid points using the flag in column 3 (1 = out of bounds, 0 = valid)
    valid_mask = nearest_zslice_index_and_values_2d_arr[:, 3].flatten() == 0

    valid_points = flat_points[valid_mask]
    
    # Create CuPy arrays for the points
    valid_points_cp_arr = cp.array(valid_points)
    
    # Create CuPy arrays for the polygons and indices
    poly_points = []
    poly_indices = []
    current_index = 0

    # Precompute the necessary values outside the loop
    valid_indices = np.where(valid_mask)[0]
    valid_nearest_zslice_index_and_values_2d_arr = nearest_zslice_index_and_values_2d_arr[valid_indices]

    for nearest_zslice_index_and_values in valid_nearest_zslice_index_and_values_2d_arr:
        relative_structure_index = nearest_zslice_index_and_values[0].astype(int)
        nearest_zslice_index = nearest_zslice_index_and_values[1].astype(int)
        start_idx, end_idx = all_structures_slices_indices_list[relative_structure_index][nearest_zslice_index]
        polygon_points = all_structures_list_of_2d_arr[relative_structure_index][start_idx:end_idx, :3]
        poly_points.append(polygon_points)
        poly_indices.append([current_index, current_index + len(polygon_points)])
        current_index += len(polygon_points)
    
    poly_points = cp.array(np.vstack(poly_points))
    poly_indices = cp.array(poly_indices)
    
    # Test each point against the corresponding polygon
    valid_results = one_to_one_point_in_polygon_cupy_arr_version(valid_points_cp_arr, poly_points, poly_indices, log_sub_dirs_list = log_sub_dirs_list, log_file_name=log_file_name, include_edges_in_log = include_edges_in_log, kernel_type=kernel_type, return_array_as="cupy")
    
    # Map the valid results back to the original result array
    result_cp_arr_flat = result_cp_arr.flatten()
    result_cp_arr_flat[valid_mask] = valid_results
    
    
    return result_cp_arr # result is 1d array of length total_num_points_to_test, refer to points_to_test_indices_arr to get the mapping of the results to the original test structures



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





def create_containment_results_dataframe_type_2I(structure_info, 
                                         nearest_zslice_index_and_values_3d_arr, 
                                         test_points_array, 
                                         result_cp_arr,
                                         do_not_convert_column_names_to_categorical = [],
                                         float_dtype = np.float32,
                                         int_dtype = np.int32):
    """
    Create a DataFrame to keep track of the containment results. This one is meant to handle two 3d array inputs.
    
    Parameters:
    - structure_info: List containing relative structure information.
    - nearest_zslice_index_and_values_3d_arr: 3d array of the nearest z values and their indices for each biopsy point.
    - test_points_array: 3d array of test points.
    - result_cp_arr: Array indicating whether each point is inside the corresponding polygon.
    - float_dtype: The float data type to use for the DataFrame.
    - int_dtype: The integer data type to use for the DataFrame.
    
    Returns:
    - containment_results_df: DataFrame containing the containment results.
    """
    
    # Flatten the input arrays for easier processing
    flat_nearest_zslices_vals_arr = nearest_zslice_index_and_values_3d_arr[:, :, 2].flatten()
    flat_nearest_zslices_indices_arr = nearest_zslice_index_and_values_3d_arr[:, :, 1].flatten()
    flat_test_points_array = test_points_array.reshape(-1, 3)
    flat_result_cp_arr = result_cp_arr.get().flatten()
    
    # Create RGB color arrays based on the containment results
    pt_clr_r = np.where(flat_result_cp_arr, 0., 1.)  # Red for false
    pt_clr_g = np.where(flat_result_cp_arr, 1., 0.)  # Green for true
    pt_clr_b = np.zeros_like(pt_clr_r)  # Blue is always 0
    
    # Create a dictionary to store the results
    results_dictionary = {
        "Relative structure ROI": [structure_info['Structure ID']] * len(flat_result_cp_arr),
        "Relative structure type": [structure_info['Struct ref type']] * len(flat_result_cp_arr),
        "Relative structure index": np.full(len(flat_result_cp_arr), [structure_info['Index number']]).astype(int_dtype),
        "Pt contained bool": flat_result_cp_arr,
        "Nearest zslice zval": flat_nearest_zslices_vals_arr.astype(float_dtype),
        "Nearest zslice index": flat_nearest_zslices_indices_arr.astype(int_dtype),
        "Pt clr R": pt_clr_r.astype(float_dtype),
        "Pt clr G": pt_clr_g.astype(float_dtype),
        "Pt clr B": pt_clr_b.astype(float_dtype),
        "Test pt X": flat_test_points_array[:, 0].astype(float_dtype),
        "Test pt Y": flat_test_points_array[:, 1].astype(float_dtype),
        "Test pt Z": flat_test_points_array[:, 2].astype(float_dtype),
    }
    
    containment_results_df = pd.DataFrame(results_dictionary)

    containment_results_df = dataframe_builders.convert_columns_to_categorical_and_downcast(containment_results_df, 
                                                                                            threshold=0.25, 
                                                                                            do_not_convert_column_names_to_categorical = do_not_convert_column_names_to_categorical)

    return containment_results_df



def create_containment_results_dataframe_type_2II(structure_info, 
                                         nearest_zslice_index_and_values_2d_arr,
                                         points_to_test_2d_arr, 
                                         result_cp_arr):
    """
    Create a DataFrame to keep track of the containment results. This one is meant to handle two stacked 2d array inputs
    
    Parameters:
    - structure_info: List containing relative structure information.
    - nearest_zslice_index_and_values_2d_arr: 2d array of the nearest z values and their indices for each biopsy point.
    - test_points_array: Array of test points.
    - result_cp_arr: Array indicating whether each point is inside the corresponding polygon.
    
    Returns:
    - containment_results_df: DataFrame containing the containment results.
    """
    
    # Flatten the input arrays for easier processing
    flat_nearest_zslices_vals_arr = nearest_zslice_index_and_values_2d_arr[:, 2]
    flat_nearest_zslices_indices_arr = nearest_zslice_index_and_values_2d_arr[:, 1]
    flat_test_points_array = points_to_test_2d_arr
    flat_result_cp_arr = result_cp_arr.get().flatten()
    
    # Create RGB color arrays based on the containment results
    pt_clr_r = np.where(flat_result_cp_arr, 0, 1)  # Red for false
    pt_clr_g = np.where(flat_result_cp_arr, 1, 0)  # Green for true
    pt_clr_b = np.zeros_like(pt_clr_r)  # Blue is always 0
    
    # Create a dictionary to store the results
    results_dictionary = {
        "Relative structure ROI": [structure_info[0]] * len(flat_result_cp_arr),
        "Relative structure type": [structure_info[1]] * len(flat_result_cp_arr),
        "Relative structure index": [structure_info[3]] * len(flat_result_cp_arr),
        "Pt contained bool": flat_result_cp_arr,
        "Nearest zslice zval": flat_nearest_zslices_vals_arr,
        "Nearest zslice index": flat_nearest_zslices_indices_arr,
        "Pt clr R": pt_clr_r,
        "Pt clr G": pt_clr_g,
        "Pt clr B": pt_clr_b,
        "Test pt X": flat_test_points_array[:, 0],
        "Test pt Y": flat_test_points_array[:, 1],
        "Test pt Z": flat_test_points_array[:, 2],
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

    num_points = 1000
    num_vertices_list = [4, 16, 32]
    include_cuspatial = False

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
        log_file_name = f"cuda_log_{num_vertices}_vertices_adv_ker.txt"
        one_to_one_results_adv_ker = one_to_one_point_in_polygon_cupy_arr_version(points, poly_points, poly_indices, log_file_name=log_file_name, kernel_type="one_to_one_pip_kernel_advanced", return_array_as="numpy")
        et = time.time()
        print(f"\n🔹 Custom One-to-One Time for {num_vertices}-vertex polygons (advanced kernel):", et - st)

        # -------------------------------
        # 🔹 Run Custom One-to-One Test more robust kernel (with reparameterization)
        # -------------------------------
        st = time.time()
        log_file_name = f"cuda_log_{num_vertices}_vertices_adv_ker_reparam.txt"
        one_to_one_results_adv_ker_reparam = one_to_one_point_in_polygon_cupy_arr_version(points, poly_points, poly_indices, log_file_name=log_file_name, kernel_type="one_to_one_pip_kernel_advanced_reparameterized_version", return_array_as="numpy")
        et = time.time()
        print(f"\n🔹 Custom One-to-One Time for {num_vertices}-vertex polygons (advanced kernel with reparameterization):", et - st)

        # -------------------------------
        # 🔹 Run Default `cuspatial.point_in_polygon` in Chunks (Max 31 Polygons per Batch)
        # -------------------------------
        if include_cuspatial:
            st = time.time()
            default_results = chunked_point_in_polygon(points_gs, polygons_gs)
            et = time.time()
            print(f"\n🔹 Default CuSpatial Time (Chunked) for {num_vertices}-vertex polygons:", et - st)

            # Extract diagonal results from default `point_in_polygon`
            diagonal_results = cp.diag(default_results)

        # -------------------------------
        # 🔹 Compare Results
        # -------------------------------
        print(f"\n🔹 Optimized One-to-One Results for {num_vertices}-vertex polygons (advanced):")
        print(one_to_one_results_adv_ker[0:50])  # Print first 20 results

        print(f"\n🔹 Optimized One-to-One Results for {num_vertices}-vertex polygons:")
        print(one_to_one_results_adv_ker_reparam[0:50])  # Print first 20 results
        
        if include_cuspatial:
            print(f"\n🔹 Diagonal Results from Default CuSpatial for {num_vertices}-vertex polygons (advanced-reparameterized):")
            print(diagonal_results[0:50])  # Print first 20 results

        # Check if results match
        if include_cuspatial:
            print(f"\n✅ Do Results Match for {num_vertices}-vertex polygons?", np.all(one_to_one_results_adv_ker == diagonal_results.get()))
            print(f"\n✅ Do Results Match for {num_vertices}-vertex polygons?", np.all(one_to_one_results_adv_ker_reparam == diagonal_results.get()))
        print(f"\n✅ Do Results Match for {num_vertices}-vertex polygons?", np.all(one_to_one_results_adv_ker == one_to_one_results_adv_ker_reparam))
    

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