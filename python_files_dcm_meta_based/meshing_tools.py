import open3d as o3d
import numpy as np
import point_containment_tools
import os
import numpy as np
from multiprocess import Pool
import threading
import time
import queue
import multiprocessing as mp


def trimesh_reconstruction_ball_pivot(threeD_data_arr, ball_radii,radius_for_normals_estimation,max_nn_for_normals_estimation):
    num_points = threeD_data_arr.shape[0]
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(threeD_data_arr)
    pcd_color = np.random.uniform(0, 0.7, size=3)
    point_cloud.paint_uniform_color(pcd_color) 
    #point_cloud.estimate_normals()
    #point_cloud.orient_normals_consistent_tangent_plane(num_points)
    #point_cloud.orient_normals_to_align_with_direction(np.array([0.0,0.0,0.0],dtype=float))
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_for_normals_estimation, max_nn=max_nn_for_normals_estimation))
    num_points_for_orientation = min([num_points/2,100])
    point_cloud.orient_normals_consistent_tangent_plane(int(num_points_for_orientation))
    point_cloud.normalize_normals()
    struct_tri_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                            point_cloud,  o3d.utility.DoubleVector(ball_radii))
    return struct_tri_mesh

def trimesh_reconstruction_poisson(threeD_data_arr):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(threeD_data_arr)
    pcd_color = np.random.uniform(0, 0.7, size=3)
    point_cloud.paint_uniform_color(pcd_color) 
    point_cloud.estimate_normals()
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        struct_tri_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=9)
    return struct_tri_mesh

def trimesh_reconstruction_alphashape(threeD_data_arr, 
                                      radius_for_normals_estimation, 
                                      max_nn_for_normals_estimation, 
                                      alpha_param):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(threeD_data_arr)
    pcd_color = np.random.uniform(0, 0.7, size=3)
    point_cloud.paint_uniform_color(pcd_color) 
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_for_normals_estimation, max_nn=max_nn_for_normals_estimation))
    alpha = alpha_param
    tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(point_cloud)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(point_cloud, alpha, tetra_mesh, pt_map)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)


def trimesh_reconstruction_alphashape_simple(threeD_data_arr,
                                            alpha_param):
    pcd = point_containment_tools.create_point_cloud(threeD_data_arr)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha_param)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)






#### NON BLOCKING (ie this allows for the livedisplay feature to not be frozen since open3d methods are blocking methods such as estimate_normals)
    
"""
def trimesh_reconstruction_worker(threeD_data_arr, ball_radii, radius_for_normals_estimation, max_nn_for_normals_estimation, result_queue, live_display):
    # Perform Open3D operations here
    num_points = threeD_data_arr.shape[0]
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(threeD_data_arr)
    pcd_color = np.random.uniform(0, 0.7, size=3)
    point_cloud.paint_uniform_color(pcd_color)
    
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_for_normals_estimation, max_nn=max_nn_for_normals_estimation))
    num_points_for_orientation = min([num_points / 2, 100])
    point_cloud.orient_normals_consistent_tangent_plane(int(num_points_for_orientation))
    point_cloud.normalize_normals()

    struct_tri_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        point_cloud, o3d.utility.DoubleVector(ball_radii))

    # Place the result in the queue so the main thread can retrieve it
    result_queue.put(struct_tri_mesh)

def trimesh_reconstruction_ball_pivot_non_blocking(threeD_data_arr, ball_radii, radius_for_normals_estimation, max_nn_for_normals_estimation, live_display=None):
    if live_display is not None:
        live_display.start(refresh=True)

    result_queue = queue.Queue()  # Create a queue to hold the result

    # Use threading to run the Open3D operations
    worker_thread = threading.Thread(target=trimesh_reconstruction_worker,
                                     args=(threeD_data_arr, ball_radii, radius_for_normals_estimation, max_nn_for_normals_estimation, result_queue, live_display))
    worker_thread.start()

    while worker_thread.is_alive():
        if live_display is not None:
            live_display.refresh()
        time.sleep(0.1)  # Allow other operations to continue

    worker_thread.join()  # Ensure the worker thread has finished

    # Get the result from the queue
    struct_tri_mesh = result_queue.get()

    return struct_tri_mesh, live_display

"""







# Worker function for Open3D operations (runs in the main thread)
def trimesh_reconstruction_worker(threeD_data_arr, ball_radii, radius_for_normals_estimation, max_nn_for_normals_estimation):
    num_points = threeD_data_arr.shape[0]
    
    # Create the point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(threeD_data_arr)
    
    # Assign random color to the point cloud
    pcd_color = np.random.uniform(0, 0.7, size=3)
    point_cloud.paint_uniform_color(pcd_color) 
    
    # Estimate normals
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_for_normals_estimation, max_nn=max_nn_for_normals_estimation))
    
    # Orient normals and normalize
    num_points_for_orientation = min([num_points/2, 100])
    point_cloud.orient_normals_consistent_tangent_plane(int(num_points_for_orientation))
    point_cloud.normalize_normals()
    
    # Create triangle mesh using ball pivoting
    struct_tri_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                            point_cloud, o3d.utility.DoubleVector(ball_radii))
    
    return struct_tri_mesh

# Live display worker (runs in its own thread)
def run_live_display(live_display, stop_event):
    while not stop_event.is_set():
        if live_display is not None:
            live_display.refresh()  # Refresh the live display periodically
        time.sleep(0.1)

# Main function (runs the Open3D operation in the main thread, live display in a separate thread)
def trimesh_reconstruction_ball_pivot_non_blocking(threeD_data_arr, ball_radii, radius_for_normals_estimation, max_nn_for_normals_estimation, live_display=None):
    # Start live display in a separate thread
    stop_event = threading.Event()
    if live_display is not None:
        live_thread = threading.Thread(target=run_live_display, args=(live_display, stop_event))
        live_thread.start()

    # Perform Open3D operations in the main thread
    struct_tri_mesh = trimesh_reconstruction_worker(threeD_data_arr, ball_radii, radius_for_normals_estimation, max_nn_for_normals_estimation)

    # Stop live display once Open3D task is done
    stop_event.set()
    if live_display is not None:
        live_thread.join()  # Wait for live display thread to finish

    return struct_tri_mesh, live_display









########## PARALLELIZED VERSION





# Function to estimate normals and create mesh for a subset of points
def process_point_cloud(threeD_data_arr_chunk, ball_radii, radius_for_normals_estimation, max_nn_for_normals_estimation):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(threeD_data_arr_chunk)
    
    # Estimate normals for this subset of the point cloud
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius_for_normals_estimation, max_nn=max_nn_for_normals_estimation))

    num_points_for_orientation = min([threeD_data_arr_chunk.shape[0] / 2, 100])
    point_cloud.orient_normals_consistent_tangent_plane(int(num_points_for_orientation))
    point_cloud.normalize_normals()
    
    # Create mesh using ball pivoting
    struct_tri_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        point_cloud, o3d.utility.DoubleVector(ball_radii))
    
    return struct_tri_mesh

# Main function to parallelize
def trimesh_reconstruction_ball_pivot_parallel(threeD_data_arr, ball_radii, radius_for_normals_estimation, max_nn_for_normals_estimation, parallel_pool, live_display=None):
    if live_display is not None:
        live_display.start(refresh=True)
    
    # Split the data into chunks (each chunk will be processed by a separate process)
    cpu_count = min(os.cpu_count(), 4)  # Try 4 cores instead of the full 16
    chunks = np.array_split(threeD_data_arr, cpu_count)
    
    # Use multiprocess to parallelize the normal estimation and mesh creation
    results = parallel_pool.starmap(
        process_point_cloud, 
        [(chunk, ball_radii, radius_for_normals_estimation, max_nn_for_normals_estimation) for chunk in chunks]
    )

    # Merge results from each process (if needed, you can combine the meshes)
    combined_mesh = o3d.geometry.TriangleMesh()
    for mesh in results:
        combined_mesh += mesh

    if live_display is not None:
        return combined_mesh, live_display
    else:
        return combined_mesh
