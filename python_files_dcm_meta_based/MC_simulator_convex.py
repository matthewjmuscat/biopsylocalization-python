import time
import biopsy_creator
import loading_tools # imported for more sophisticated loading bar
import numpy as np
import open3d as o3d
import point_containment_tools
import plotting_funcs

def simulator(master_structure_reference_dict, structs_referenced_list, num_simulations):

    ref_list = ["Bx ref","OAR ref","DIL ref"] # note that Bx ref has to be the first entry for other parts of the code to work!
    uncertainty_sources_list = []
    uncertainty_sources_raw_dict = {}

    num_biopsies = sum([len(patient_dict[1][ref_list[0]]) for patient_dict in master_structure_reference_dict.items()])

    print('Number of biopsy tracks to simulate are: ', num_biopsies,'.')
    print('Number of simulations per biopsy track has been set to: ', num_simulations,'.')
    

    st = time.time()
    with loading_tools.Loader(num_biopsies,"Reconstructing and sampling biopsies...") as loader:
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            Bx_structs = structs_referenced_list[0]
            for specific_BX_structure_index, specific_BX_structure in enumerate(pydicom_item[Bx_structs]):
                centroid_line = specific_BX_structure["Best fit line of centroid pts"]
                origin_to_first_centroid_vector = specific_BX_structure["Centroid line sample pts"][0]
                list_origin_to_first_centroid_vector = np.squeeze(origin_to_first_centroid_vector).tolist()
                biopsy_samples = biopsy_creator.biopsy_points_reconstruction_and_uniform_sampler(list_origin_to_first_centroid_vector,centroid_line)
                biopsy_samples_point_cloud = o3d.geometry.PointCloud()
                biopsy_samples_point_cloud.points = o3d.utility.Vector3dVector(biopsy_samples[:,0:3])
                pcd_color = np.random.uniform(0, 0.7, size=3)
                biopsy_samples_point_cloud.paint_uniform_color(pcd_color)
                master_structure_reference_dict[patientUID][Bx_structs][specific_BX_structure_index]["Random uniformly sampled volume pts"] = biopsy_samples
                biopsy_raw_point_cloud = master_structure_reference_dict[patientUID][Bx_structs][specific_BX_structure_index]["Point cloud raw"]
                pcd_color = np.random.uniform(0, 0.7, size=3)
                biopsy_raw_point_cloud.paint_uniform_color(pcd_color)

                # plot point clouds?
                o3d.visualization.draw_geometries([biopsy_raw_point_cloud,biopsy_samples_point_cloud])



    with loading_tools.Loader(num_biopsies*num_simulations,"Simulating biopsy uncertainties...") as loader:
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            Bx_structs = structs_referenced_list[0]
            for specific_BX_structure_index, specific_BX_structure in enumerate(pydicom_item[Bx_structs]):
                structure_uncertainty_array = np.empty([num_simulations, specific_BX_structure["Uncertainty params"].size])
                for mu,sigma in specific_BX_structure["Uncertainty params"]: 
                    specific_uncertainty_array = np.random.normal(mu, sigma, num_simulations)

                    for j in range(0, num_simulations):
                        print(1)
                        
    return master_structure_reference_dict



def simulator_parallel(master_structure_reference_dict, structs_referenced_list, num_simulations):
    
    # build args list for parallel computing
    args_list_forstructure_shift = []
    for patientUID,pydicom_item in master_structure_reference_dict.items():
        for structure_type in structs_referenced_list:        
            for specific_structure_index, specific_structure in enumerate(pydicom_item[structure_type]):
                uncertainty_data_obj = specific_structure["Uncertainty data"]
                if structure_type == structs_referenced_list[0]:
                    randomly_sampled_bx_pts = specific_structure["Random uniformly sampled volume pts"]
                    structure_to_shift = randomly_sampled_bx_pts
                else:
                    structure_pts = specific_structure["Random uniformly sampled volume pts"]
                    structure_to_shift = randomly_sampled_bx_pts
            
                


def MC_simulator_shift_structure(specific_structure_dict, structs_referenced_list, num_simulations):

    normal_simulations = np.random.normal(loc=0.0, scale=1.0, size=None)
    pass


def box_simulator_delaunay_zslice_wise_parallel(parallel_pool, num_simulations, deulaunay_objs_zslice_wise_list, point_cloud):
    # test points to test for inclusion
    num_pts = num_simulations
    max_bnd = point_cloud.get_max_bound()
    min_bnd = point_cloud.get_min_bound()
    center = point_cloud.get_center()
    if np.linalg.norm(max_bnd-center) >= np.linalg.norm(min_bnd-center): 
        largest_bnd = max_bnd
    else:
        largest_bnd = min_bnd
    bounding_box_size = np.linalg.norm(largest_bnd-center)
    test_pts = [np.random.uniform(-bounding_box_size,bounding_box_size, size = 3) for i in range(num_pts)]
    test_pts_arr = np.array(test_pts) + center
    test_pts_list = test_pts_arr.tolist()
    test_pts_point_cloud = o3d.geometry.PointCloud()
    test_pts_point_cloud.points = o3d.utility.Vector3dVector(test_pts_arr)
    test_pt_colors = np.empty([num_pts,3], dtype=float)
    
    
    test_points_results = point_containment_tools.test_zslice_wise_containment_delaunay_parallel(parallel_pool, deulaunay_objs_zslice_wise_list, test_pts_list)
    for index,result in enumerate(test_points_results):
        test_pt_colors[index] = result[2]
    test_pts_point_cloud.colors = o3d.utility.Vector3dVector(test_pt_colors)

    return test_points_results, test_pts_point_cloud


def box_simulator_delaunay_global_convex_structure_parallel(parallel_pool, num_simulations, deulaunay_obj, point_cloud):
    # test points to test for inclusion
    num_pts = num_simulations
    max_bnd = point_cloud.get_max_bound()
    min_bnd = point_cloud.get_min_bound()
    center = point_cloud.get_center()
    if np.linalg.norm(max_bnd-center) >= np.linalg.norm(min_bnd-center): 
        largest_bnd = max_bnd
    else:
        largest_bnd = min_bnd
    bounding_box_size = np.linalg.norm(largest_bnd-center)
    test_pts = [np.random.uniform(-bounding_box_size,bounding_box_size, size = 3) for i in range(num_pts)]
    test_pts_arr = np.array(test_pts) + center
    test_pts_list = test_pts_arr.tolist()
    test_pts_point_cloud = o3d.geometry.PointCloud()
    test_pts_point_cloud.points = o3d.utility.Vector3dVector(test_pts_arr)
    test_pt_colors = np.empty([num_pts,3], dtype=float)
    
    
    test_points_results = point_containment_tools.test_global_convex_structure_containment_delaunay_parallel(parallel_pool, deulaunay_obj, test_pts_list)
    for index,result in enumerate(test_points_results):
        test_pt_colors[index] = result[2]
    test_pts_point_cloud.colors = o3d.utility.Vector3dVector(test_pt_colors)

    return test_points_results, test_pts_point_cloud





def point_sampler_from_global_delaunay_convex_structure_for_sequential(num_samples, delaunay_global_convex_structure_tri, reconstructed_bx_point_cloud):
    insert_index = 0
    reconstructed_bx_point_cloud_color = np.array([0,0,1])
    reconstructed_bx_point_cloud.paint_uniform_color(reconstructed_bx_point_cloud_color)

    bx_samples_arr = np.empty((num_samples,3),dtype=float)
    axis_aligned_bounding_box = reconstructed_bx_point_cloud.get_axis_aligned_bounding_box()
    axis_aligned_bounding_box_points_arr = np.asarray(axis_aligned_bounding_box.get_box_points())
    bounding_box_color = np.array([0,0,0], dtype=float)
    axis_aligned_bounding_box.color = bounding_box_color
    max_bounds = np.amax(axis_aligned_bounding_box_points_arr, axis=0)
    min_bounds = np.amin(axis_aligned_bounding_box_points_arr, axis=0)

    
    while insert_index < num_samples:
        x_val = np.random.uniform(min_bounds[0], max_bounds[0])
        y_val = np.random.uniform(min_bounds[1], max_bounds[1])
        z_val = np.random.uniform(min_bounds[2], max_bounds[2])
        random_point_within_bounding_box = np.array([x_val,y_val,z_val],dtype=float)
        
        containment_result_bool = point_containment_tools.convex_bx_structure_global_test_point_containment(delaunay_global_convex_structure_tri,random_point_within_bounding_box)
        
        
        random_point_pcd = o3d.geometry.PointCloud()
        random_point_pcd.points = o3d.utility.Vector3dVector(np.array([random_point_within_bounding_box]))
        random_point_pcd_color = np.array([0,1,0])
        random_point_pcd.paint_uniform_color(random_point_pcd_color)
        #plotting_funcs.plot_geometries(reconstructed_bx_point_cloud,random_point_pcd)
        #print(containment_result_bool)
        if containment_result_bool == True:
            bx_samples_arr[insert_index] = random_point_within_bounding_box
            insert_index = insert_index + 1
        else:
            pass
    
    bx_samples_arr_point_cloud_color = np.random.uniform(0, 0.7, size=3)
    bx_samples_arr_point_cloud = point_containment_tools.create_point_cloud(bx_samples_arr, bx_samples_arr_point_cloud_color)
    
    return bx_samples_arr, bx_samples_arr_point_cloud, axis_aligned_bounding_box


def point_sampler_from_global_delaunay_convex_structure_parallel(num_samples, delaunay_global_convex_structure_tri, reconstructed_bx_arr):
    insert_index = 0
    reconstructed_bx_point_cloud = point_containment_tools.create_point_cloud(reconstructed_bx_arr)
    reconstructed_bx_point_cloud_color = np.array([0,0,1])
    reconstructed_bx_point_cloud.paint_uniform_color(reconstructed_bx_point_cloud_color)

    bx_samples_arr = np.empty((num_samples,3),dtype=float)
    axis_aligned_bounding_box = reconstructed_bx_point_cloud.get_axis_aligned_bounding_box()
    axis_aligned_bounding_box_points_arr = np.asarray(axis_aligned_bounding_box.get_box_points())
    bounding_box_color = np.array([0,0,0], dtype=float)
    axis_aligned_bounding_box.color = bounding_box_color
    max_bounds = np.amax(axis_aligned_bounding_box_points_arr, axis=0)
    min_bounds = np.amin(axis_aligned_bounding_box_points_arr, axis=0)

    
    while insert_index < num_samples:
        x_val = np.random.uniform(min_bounds[0], max_bounds[0])
        y_val = np.random.uniform(min_bounds[1], max_bounds[1])
        z_val = np.random.uniform(min_bounds[2], max_bounds[2])
        random_point_within_bounding_box = np.array([x_val,y_val,z_val],dtype=float)
        
        containment_result_bool = point_containment_tools.convex_bx_structure_global_test_point_containment(delaunay_global_convex_structure_tri,random_point_within_bounding_box)
        
        
        random_point_pcd = o3d.geometry.PointCloud()
        random_point_pcd.points = o3d.utility.Vector3dVector(np.array([random_point_within_bounding_box]))
        random_point_pcd_color = np.array([0,1,0])
        random_point_pcd.paint_uniform_color(random_point_pcd_color)
        #plotting_funcs.plot_geometries(reconstructed_bx_point_cloud,random_point_pcd)
        #print(containment_result_bool)
        if containment_result_bool == True:
            bx_samples_arr[insert_index] = random_point_within_bounding_box
            insert_index = insert_index + 1
        else:
            pass
    
    bx_samples_arr_point_cloud_color = np.random.uniform(0, 0.7, size=3)
    bx_samples_arr_point_cloud = point_containment_tools.create_point_cloud(bx_samples_arr, bx_samples_arr_point_cloud_color)
    
    return bx_samples_arr, axis_aligned_bounding_box_points_arr
    