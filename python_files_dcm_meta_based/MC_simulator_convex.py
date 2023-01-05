import time
import biopsy_creator
import loading_tools # imported for more sophisticated loading bar
import numpy as np
import open3d as o3d
import point_containment_tools
import plotting_funcs
import rich
from rich.progress import Progress, track
import time 

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
                master_structure_reference_dict[patientUID][Bx_structs][specific_BX_structure_index]["Random uniformly sampled volume pts arr"] = biopsy_samples
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



def simulator_parallel(parallel_pool, master_structure_reference_dict, structs_referenced_list, num_simulations, master_structure_info_dict, spinner_type):
    
    num_patients = master_structure_info_dict["Global"]["Num patients"]
    num_structures = master_structure_info_dict["Global"]["Num structures"]
    with Progress(rich.progress.SpinnerColumn(spinner_type),
                *Progress.get_default_columns(),
                rich.progress.TimeElapsedColumn()) as progress:
        generating_MCsamples_task = progress.add_task("[red]Generating " + str(num_simulations) + " samples for " + str(num_structures) + " structures (parallel)...", total=num_patients)
        # simulate all structure shifts in parallel and update the master reference dict
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            patient_dict_updated_with_all_structs_generated_norm_dist_translation_samples = MC_simulator_shift_all_structures_generator_parallel(parallel_pool, pydicom_item, structs_referenced_list, num_simulations)
            master_structure_reference_dict[patientUID] = patient_dict_updated_with_all_structs_generated_norm_dist_translation_samples
            progress.update(generating_MCsamples_task, advance=1)
    
    num_biopsies = master_structure_info_dict["Global"]["Num biopsies"]
    num_OARs = master_structure_info_dict["Global"]["Num OARs"]
    num_DILs = master_structure_info_dict["Global"]["Num DILs"]
    
    print("Simulation data: # MC samples =",str(num_simulations),"| # biopsies =",str(num_biopsies),"| # anatomical structures =",str(num_structures-num_biopsies),"| # patients =",str(num_patients),".")
    with Progress(rich.progress.SpinnerColumn(spinner_type),
                *Progress.get_default_columns(),
                rich.progress.TimeElapsedColumn()) as progress:
        translating_structures_task = progress.add_task("[red]MC simulating biopsy and anatomy translation (parallel)...", total=num_biopsies)
        # simulate every biopsy sequentially
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            # create a dictionary of all non bx structures
            structure_shifted_bx_data_dict = {}
            for non_bx_struct_type in structs_referenced_list[1:]:
                for specific_non_bx_structure_index, specific_non_bx_structure in enumerate(pydicom_item[non_bx_struct_type]):
                    specific_non_bx_struct_roi = specific_non_bx_structure["ROI"]
                    specific_non_bx_struct_refnum = specific_non_bx_structure["Ref #"]
                    structure_shifted_bx_data_dict[specific_non_bx_struct_roi,non_bx_struct_type,specific_non_bx_struct_refnum,specific_non_bx_structure_index] = None
            MC_translation_results_for_fixed_bx_dict = structure_shifted_bx_data_dict.copy()
            # set structure type to BX 
            structure_type = structs_referenced_list[0]
                    
            for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[structure_type]):
                # Do all trials in parallel
                bx_only_shifted_randomly_sampled_bx_pts_3Darr = MC_simulator_translate_sampled_bx_points_arr_bx_only_shift_parallel(parallel_pool, specific_bx_structure)
                # THIS SHOULD BE SAVED AT THE END. Save the 3d array of the bx only shifted data containing all MC trials as slices to the master reference dictionary
                master_structure_reference_dict[patientUID][structure_type][specific_bx_structure_index]["MC data: bx only shifted 3darr"] = bx_only_shifted_randomly_sampled_bx_pts_3Darr
                
                
                structure_shifted_bx_data_dict = MC_simulator_translate_sampled_bx_points_3darr_structure_only_shift_parallel(parallel_pool, pydicom_item, structs_referenced_list, bx_only_shifted_randomly_sampled_bx_pts_3Darr, structure_shifted_bx_data_dict)
                master_structure_reference_dict[patientUID][structure_type][specific_bx_structure_index]["MC data: bx and structure shifted dict"] = structure_shifted_bx_data_dict
                progress.update(translating_structures_task, advance=1)


    with Progress(rich.progress.SpinnerColumn(spinner_type),
                *Progress.get_default_columns(),
                rich.progress.TimeElapsedColumn()) as progress:
        testing_biopsy_containment_task = progress.add_task("[red]Testing biopsy containment in all anatomical structures (overall progress)...", total=num_biopsies)
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            bx_structure_type = structs_referenced_list[0]           
            for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structure_type]):
                sp_patient_total_num_structs = master_structure_info_dict["By patient"][patientUID]["All ref"]["Total num structs"]
                sp_patient_total_num_BXs = master_structure_info_dict["By patient"][patientUID][bx_structure_type]["Num structs"]
                sp_patient_total_num_non_BXs = sp_patient_total_num_structs - sp_patient_total_num_BXs
                testing_each_non_bx_structure_containment_task = progress.add_task("[green]Testing each structure for containment...", total=sp_patient_total_num_non_BXs)
                structure_shifted_bx_data_dict = master_structure_reference_dict[patientUID][bx_structure_type][specific_bx_structure_index]["MC data: bx and structure shifted dict"] 
                for structure_info,shifted_bx_data_3darr in structure_shifted_bx_data_dict.items():
                    structure_roi = structure_info[0]
                    non_bx_structure_type = structure_info[1]
                    structure_refnum = structure_info[2]
                    structure_index = structure_info[3]
                    non_bx_struct_deulaunay_objs_zslice_wise_list = master_structure_reference_dict[patientUID][non_bx_structure_type][structure_index]["Delaunay triangulation zslice-wise list"] 
                    non_bx_struct_deulaunay_obj_global_convex = master_structure_reference_dict[patientUID][non_bx_structure_type][structure_index]["Delaunay triangulation global structure"] 
                    non_bx_struct_interslice_interpolation_information = master_structure_reference_dict[patientUID][non_bx_structure_type][structure_index]["Inter-slice interpolation information"] 
                    testing_each_trial_task = progress.add_task("[blue]Testing each MC trial (points within trial in parallel)...", total=num_simulations)
                    all_trials_POP_test_results_and_point_clouds_tuple = []
                    for single_trial_shifted_bx_data_arr in shifted_bx_data_3darr:
                        single_trial_shifted_bx_data_results_fully_concave_and_point_cloud_tuple = point_containment_test_delaunay_zslice_wise_parallel(parallel_pool, num_simulations, non_bx_struct_deulaunay_obj_global_convex, non_bx_struct_interslice_interpolation_information, single_trial_shifted_bx_data_arr)
                        all_trials_POP_test_results_and_point_clouds_tuple.append(single_trial_shifted_bx_data_results_fully_concave_and_point_cloud_tuple)
                        progress.update(testing_each_trial_task, advance=1)
                    
                    progress.remove_task(testing_each_trial_task)
                    MC_translation_results_for_fixed_bx_dict[structure_info] = all_trials_POP_test_results_and_point_clouds_tuple

                    progress.update(testing_each_non_bx_structure_containment_task, advance=1)
                
                progress.remove_task(testing_each_non_bx_structure_containment_task)
                # Update the master dictionary
                master_structure_reference_dict[patientUID][bx_structure_type][specific_bx_structure_index]["MC data: MC sim translation results dict"] = MC_translation_results_for_fixed_bx_dict


                progress.update(testing_biopsy_containment_task, advance=1)
    return master_structure_reference_dict


def MC_simulator_shift_all_structures_generator_parallel(parallel_pool, patient_dict, structs_referenced_list, num_simulations):

    # build args list for parallel computing
    args_list = []
    patient_dict_updated_with_generated_samples = patient_dict.copy()
    for structure_type in structs_referenced_list:
        for specific_structure_index, specific_structure in enumerate(patient_dict[structure_type]):
            #spec_structure_zslice_wise_delaunay_obj_list = specific_structure["Delaunay triangulation zslice-wise list"]
            uncertainty_data_obj = specific_structure["Uncertainty data"]
            sp_struct_uncertainty_data_mean_arr = uncertainty_data_obj.uncertainty_data_mean_arr
            sp_struct_uncertainty_data_sigma_arr = uncertainty_data_obj.uncertainty_data_sigma_arr
            specific_structure_args = (structure_type,specific_structure_index,sp_struct_uncertainty_data_mean_arr,sp_struct_uncertainty_data_sigma_arr,num_simulations)
            args_list.append(specific_structure_args)

    
    # conduct random samples of all structure shifts in parallel
    sp_structure_normal_dist_shift_samples_and_structure_reference_list = parallel_pool.starmap(MC_simulator_shift_structure_generator,args_list)
    
    # update the patient dictionary
    for generated_shifts_info_list in sp_structure_normal_dist_shift_samples_and_structure_reference_list:
        structure_type = generated_shifts_info_list[0]
        specific_structure_index = generated_shifts_info_list[1]
        specific_structure_structure_normal_dist_shift_samples_arr = generated_shifts_info_list[2]
        patient_dict_updated_with_generated_samples[structure_type][specific_structure_index]["MC data: Generated normal dist random samples arr"] = specific_structure_structure_normal_dist_shift_samples_arr

    return patient_dict_updated_with_generated_samples


def MC_simulator_shift_structure_generator(structure_type, specific_structure_index, sp_struct_uncertainty_data_mean_arr, sp_struct_uncertainty_data_sigma_arr, num_simulations):
    structure_normal_dist_shift_samples_arr = np.array([ 
            np.random.normal(loc=sp_struct_uncertainty_data_mean_arr[0], scale=sp_struct_uncertainty_data_sigma_arr[0], size=num_simulations),  
            np.random.normal(loc=sp_struct_uncertainty_data_mean_arr[1], scale=sp_struct_uncertainty_data_sigma_arr[1], size=num_simulations),  
            np.random.normal(loc=sp_struct_uncertainty_data_mean_arr[2], scale=sp_struct_uncertainty_data_sigma_arr[2], size=num_simulations)],   
            dtype = float).T
    generated_shifts_info_list = [structure_type, specific_structure_index, structure_normal_dist_shift_samples_arr]
    return generated_shifts_info_list


def MC_simulator_translate_sampled_bx_points_arr_bx_only_shift_parallel(parallel_pool, specific_bx_structure):
    randomly_sampled_bx_pts_arr = specific_bx_structure["Random uniformly sampled volume pts arr"]
    randomly_sampled_bx_shifts_arr = specific_bx_structure["MC data: Generated normal dist random samples arr"]

    args_list = []
    for bx_shift_ind in range(randomly_sampled_bx_shifts_arr.shape[0]):
        arg = (randomly_sampled_bx_pts_arr, randomly_sampled_bx_shifts_arr[bx_shift_ind])
        args_list.append(arg)

    randomly_sampled_bx_pts_arr_bx_only_shift_arr_each_sampled_shift_list = parallel_pool.starmap(MC_simulator_translate_sampled_bx_points_arr_bx_only_shift,args_list)

    num_bx_sampled_points = randomly_sampled_bx_pts_arr.shape[0]
    num_bx_sampled_shifts = randomly_sampled_bx_shifts_arr.shape[0]

    # create a 3d array that stores all the shifted bx data where each 3d slice is the shifted bx data for a fixed sampled bx shift, ie each slice is a sampled bx shift trial
    randomly_sampled_bx_pts_arr_bx_only_shift_3Darr = np.empty([num_bx_sampled_shifts,num_bx_sampled_points,3],dtype=float)

    # build the above described 3d array from the parallel results list
    for index,randomly_sampled_bx_pts_arr_bx_only_shift_arr in enumerate(randomly_sampled_bx_pts_arr_bx_only_shift_arr_each_sampled_shift_list):
        randomly_sampled_bx_pts_arr_bx_only_shift_3Darr[index] = randomly_sampled_bx_pts_arr_bx_only_shift_arr

    return randomly_sampled_bx_pts_arr_bx_only_shift_3Darr




def MC_simulator_translate_sampled_bx_points_arr_bx_only_shift(randomly_sampled_bx_pts_arr, bx_shift_vector):
    # For a single trial, all BX points are shifted by the same vector!
    randomly_sampled_bx_pts_arr_bx_only_shift_arr = randomly_sampled_bx_pts_arr + bx_shift_vector
    return randomly_sampled_bx_pts_arr_bx_only_shift_arr


def MC_simulator_translate_sampled_bx_points_3darr_structure_only_shift_parallel(parallel_pool, pydicom_item, structs_referenced_list, bx_only_shifted_randomly_sampled_bx_pts_3Darr, structure_shifted_bx_data_dict):
    # do each non bx structure sequentially
    for non_bx_struct_type in structs_referenced_list[1:]:
        for specific_non_bx_struct_index,specific_non_bx_struct in enumerate(pydicom_item[non_bx_struct_type]):
            # build args list for parallel computing
            specific_non_bx_struct_roi = specific_non_bx_struct["ROI"]
            specific_non_bx_struct_refnum = specific_non_bx_struct["Ref #"]

            args_list_non_bx_struct_shift = []
            num_2dslices_in_bx_3darr = bx_only_shifted_randomly_sampled_bx_pts_3Darr.shape[0]
            for slice_index in range(num_2dslices_in_bx_3darr): 
                bx_data_only_bx_shifted_2dslice_from_3d_arr = bx_only_shifted_randomly_sampled_bx_pts_3Darr[slice_index]
                non_bx_structure_shift_vector = specific_non_bx_struct["MC data: Generated normal dist random samples arr"][slice_index]
                bx_shift_vector = -non_bx_structure_shift_vector
                arg = (bx_data_only_bx_shifted_2dslice_from_3d_arr,bx_shift_vector)
                args_list_non_bx_struct_shift.append(arg)

            parallel_results_bx_shift_of_non_bx_structure_translation = parallel_pool.starmap(MC_simulator_translate_sampled_bx_points_arr_structure_only_shift, args_list_non_bx_struct_shift)
            
            bx_data_both_non_bx_structure_shifted_and_bx_structure_shifted_3darr = np.asarray(parallel_results_bx_shift_of_non_bx_structure_translation)

            structure_shifted_bx_data_dict[specific_non_bx_struct_roi,non_bx_struct_type,specific_non_bx_struct_refnum,specific_non_bx_struct_index] = bx_data_both_non_bx_structure_shifted_and_bx_structure_shifted_3darr

    return structure_shifted_bx_data_dict


def MC_simulator_translate_sampled_bx_points_arr_structure_only_shift(randomly_sampled_bx_pts_arr, bx_shift_vector):
    # For a single trial, all BX points are shifted by the same vector!
    randomly_sampled_bx_pts_arr_struct_only_shift_arr = randomly_sampled_bx_pts_arr + bx_shift_vector
    return randomly_sampled_bx_pts_arr_struct_only_shift_arr



def point_containment_test_delaunay_zslice_wise_parallel(parallel_pool, num_simulations, deulaunay_objs_zslice_wise_list, interslice_interpolation_information_of_containment_structure, test_pts_arr):
    num_pts = test_pts_arr.shape[0]
    test_pts_list = test_pts_arr.tolist()
    test_pts_point_cloud_zslice_delaunay = o3d.geometry.PointCloud()
    test_pts_point_cloud_zslice_delaunay.points = o3d.utility.Vector3dVector(test_pts_arr)
    test_pt_colors = np.empty([num_pts,3], dtype=float)
    
    #st = time.time() # THIS IS THE SLOW SECTION!! 2 ORDERS OF MAGNTIUDE SLOWER THAN THE LOWER SECTION

    test_points_results_zslice_delaunay = point_containment_tools.test_global_convex_structure_containment_delaunay_parallel(parallel_pool, deulaunay_objs_zslice_wise_list, test_pts_list)
    for index,result in enumerate(test_points_results_zslice_delaunay):
        test_pt_colors[index] = result[2]
    test_pts_point_cloud_zslice_delaunay.colors = o3d.utility.Vector3dVector(test_pt_colors)

    #et = time.time()
    #elapsed_time = et - st
    #print('___')
    #print('\n Execution time (delaunay parallel):', elapsed_time, 'seconds')
    #print('___')
    #st = time.time()

    test_points_results_fully_concave, test_pts_point_cloud_concave_zslice_updated = point_containment_tools.plane_point_in_polygon_concave(test_points_results_zslice_delaunay,interslice_interpolation_information_of_containment_structure, test_pts_point_cloud_zslice_delaunay)

    #et = time.time()
    #elapsed_time = et - st
    #print('___')
    #print('\n Execution time (concave test):', elapsed_time, 'seconds')
    #print('___')


    return test_points_results_fully_concave, test_pts_point_cloud_concave_zslice_updated






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


def point_sampler_from_global_delaunay_convex_structure_parallel(num_samples, delaunay_global_convex_structure_tri, reconstructed_bx_arr, patientUID, structure_type, specific_structure_index):
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
        
        return bx_samples_arr, axis_aligned_bounding_box_points_arr, {"Patient UID": patientUID, "Structure type": structure_type, "Specific structure index": specific_structure_index}
    