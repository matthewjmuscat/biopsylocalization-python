import MC_simulator_convex
import cupy as cp
import cupy_functions
import numpy as np
import copy 
import point_containment_tools
import plotting_funcs 
import biopsy_creator
import pca
    
def generate_transformations(master_structure_reference_dict,
                            simulate_uniform_bx_shifts_due_to_bx_needle_compartment,
                            bx_ref,
                            biopsy_needle_compartment_length,
                            max_simulations,
                            structs_referenced_list):

    # simulate all structure shifts in parallel and update the master reference dict
    for patientUID,pydicom_item in master_structure_reference_dict.items():
        sp_bx_structure_uniform_dist_shift_samples_and_structure_reference_list = cupy_functions.MC_simulator_non_bx_structs_dilations_generator_cupy(pydicom_item, structs_referenced_list, max_simulations)
        # update the patient dictionary
        for generated_shifts_info_list in sp_bx_structure_uniform_dist_shift_samples_and_structure_reference_list:
            structure_type = generated_shifts_info_list[0]
            specific_structure_index = generated_shifts_info_list[1]
            specific_structure_normal_dist_dilations_samples_arr = generated_shifts_info_list[2]
            pydicom_item[structure_type][specific_structure_index]["MC data: Generated normal dist random samples dilations arr"] = specific_structure_normal_dist_dilations_samples_arr

        sp_bx_structure_uniform_dist_shift_samples_and_structure_reference_list = cupy_functions.MC_simulator_non_bx_structs_rotations_generator_cupy(pydicom_item, structs_referenced_list, max_simulations)
        # update the patient dictionary
        for generated_shifts_info_list in sp_bx_structure_uniform_dist_shift_samples_and_structure_reference_list:
            structure_type = generated_shifts_info_list[0]
            specific_structure_index = generated_shifts_info_list[1]
            specific_structure_normal_dist_rotations_samples_arr = generated_shifts_info_list[2]
            pydicom_item[structure_type][specific_structure_index]["MC data: Generated normal dist random samples rotations arr"] = specific_structure_normal_dist_rotations_samples_arr

        if simulate_uniform_bx_shifts_due_to_bx_needle_compartment == True:
            #MC_simulator_shift_biopsy_structures_uniform_generator_parallel(parallel_pool, pydicom_item, structs_referenced_list, bx_ref, biopsy_needle_compartment_length, max_simulations)
            sp_bx_structure_uniform_dist_shift_samples_and_structure_reference_list = cupy_functions.MC_simulator_shift_biopsy_structures_uniform_generator_cupy(pydicom_item, bx_ref, biopsy_needle_compartment_length, max_simulations)
            # update the patient dictionary
            for generated_shifts_info_list in sp_bx_structure_uniform_dist_shift_samples_and_structure_reference_list:
                structure_type = generated_shifts_info_list[0]
                specific_structure_index = generated_shifts_info_list[1]
                specific_structure_structure_uniform_dist_shift_samples_arr = generated_shifts_info_list[2]
                pydicom_item[structure_type][specific_structure_index]["MC data: Generated uniform dist (biopsy needle compartment) random distance (z_needle) samples arr"] = specific_structure_structure_uniform_dist_shift_samples_arr
        
        sp_structure_normal_dist_shift_samples_and_structure_reference_list = cupy_functions.MC_simulator_shift_all_structures_generator_cupy(pydicom_item, structs_referenced_list, max_simulations)
        # update the patient dictionary
        for generated_shifts_info_list in sp_structure_normal_dist_shift_samples_and_structure_reference_list:
            structure_type = generated_shifts_info_list[0]
            specific_structure_index = generated_shifts_info_list[1]
            specific_structure_structure_normal_dist_shift_samples_arr = generated_shifts_info_list[2]
            pydicom_item[structure_type][specific_structure_index]["MC data: Generated normal dist random samples arr"] = cp.asnumpy(specific_structure_structure_normal_dist_shift_samples_arr)
        
        #MC_simulator_shift_all_structures_generator_parallel(parallel_pool, pydicom_item, structs_referenced_list, max_simulations)
        #master_structure_reference_dict[patientUID] = patient_dict_updated_with_all_structs_generated_norm_dist_translation_samples



def biopsy_only_transformer(master_structure_reference_dict,
                            bx_ref,
                            max_simulations,
                            simulate_uniform_bx_shifts_due_to_bx_needle_compartment):
    for patientUID,pydicom_item in master_structure_reference_dict.items():
        for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
            

            ### Self biopsy dilate
            randomly_sampled_bx_pts_arr = specific_bx_structure["Random uniformly sampled volume pts arr"]
            randomly_sampled_bx_pts_cp_arr = cp.array(randomly_sampled_bx_pts_arr) # convert numpy array to cp array

            bx_normal_dist_dilations_samples_arr = specific_bx_structure["MC data: Generated normal dist random samples dilations arr"]
            bx_normal_dist_dilations_samples_cp_arr = cp.array(bx_normal_dist_dilations_samples_arr) # convert numpy array to cp array

            bx_global_centroid = specific_bx_structure["Structure global centroid"].reshape((1,3))
            bx_global_centroid_cp_arr = cp.array(bx_global_centroid) # convert numpy array to cp array
        
            randomly_sampled_bx_pts_cp_arr_dilated_max_simulations = biopsy_dilator_step_1(randomly_sampled_bx_pts_cp_arr, 
                                                                                           bx_normal_dist_dilations_samples_cp_arr, 
                                                                                           bx_global_centroid_cp_arr, 
                                                                                           max_simulations)
            
            for trial_index in np.arange(max_simulations):
                nominal_bx_pcd_color = np.array([1,0,1])
                nominal_bx_pcd = point_containment_tools.create_point_cloud(randomly_sampled_bx_pts_arr, nominal_bx_pcd_color)
                
                pcd_color_bx_self_dilation = np.array([0,1,1]) 
                self_bx_dilation_step_point_cloud = point_containment_tools.create_point_cloud(cp.asnumpy(randomly_sampled_bx_pts_cp_arr_dilated_max_simulations[trial_index]), pcd_color_bx_self_dilation)
                
                bx_global_centroid_pcd = point_containment_tools.create_point_cloud(bx_global_centroid, np.array([0,0,0]))
                plotting_funcs.plot_geometries(nominal_bx_pcd,self_bx_dilation_step_point_cloud, bx_global_centroid_pcd)
            ### Self biopsy dilate


            ### Self biopsy rotate
            specific_structure_normal_dist_dilations_samples_arr = specific_bx_structure["MC data: Generated normal dist random samples rotations arr"]
            specific_structure_normal_dist_dilations_samples_cp_arr = cp.array(specific_structure_normal_dist_dilations_samples_arr) # convert numpy array to cp array

            randomly_sampled_bx_pts_cp_arr_dilated_and_rotated_max_simulations = biopsy_rotator_step_2(randomly_sampled_bx_pts_cp_arr_dilated_max_simulations, specific_structure_normal_dist_dilations_samples_cp_arr, bx_global_centroid_cp_arr, max_simulations)
            ### Self biopsy rotate


            ### Self biopsy translate
            if simulate_uniform_bx_shifts_due_to_bx_needle_compartment == True:
                random_uniformly_sampled_bx_shifts_cp_arr = specific_bx_structure["MC data: Generated uniform dist (biopsy needle compartment) random distance (z_needle) samples arr"]
                
                # Need to calculate the unit vector along the major axis for every trial after other two transformations
                all_trials_bx_unit_vecs_tip_to_handle = cp.empty((randomly_sampled_bx_pts_cp_arr_dilated_and_rotated_max_simulations.shape[0],3))
                for trial_index,trial_slice in enumerate(randomly_sampled_bx_pts_cp_arr_dilated_and_rotated_max_simulations):
                    trial_centroid_line = pca.linear_fitter(trial_slice.T)
                    point_1 = trial_centroid_line[0]
                    point_2 = trial_centroid_line[1]
                    if point_1[2] > point_2[2]:
                        point_sup = point_1
                        point_inf = point_2
                    else:
                        point_sup = point_2
                        point_inf = point_1
                    biopsy_vec_handle_to_tip_unit = (point_sup - point_inf)/np.linalg.norm(point_sup - point_inf)
                    biopsy_vec_tip_to_handle_unit = -biopsy_vec_handle_to_tip_unit
                    biopsy_vec_tip_to_handle_unit_cp_arr = cp.array(biopsy_vec_tip_to_handle_unit)
                    all_trials_bx_unit_vecs_tip_to_handle[trial_index] = biopsy_vec_tip_to_handle_unit_cp_arr
                
                # Multiply unit vectors by shift distances 
                bx_needle_uniform_compartment_shift_vectors_cp_array = cp.multiply(all_trials_bx_unit_vecs_tip_to_handle,random_uniformly_sampled_bx_shifts_cp_arr[...,None])
                specific_bx_structure["MC data: Generated uniform (biopsy needle compartment) random vectors (z_needle) samples arr"] = cp.asnumpy(bx_needle_uniform_compartment_shift_vectors_cp_array)
                
                # calculate total vectors
                bx_normal_translation_arr = specific_bx_structure["MC data: Generated normal dist random samples arr"]
                bx_total_only_translation_arr = bx_needle_uniform_compartment_shift_vectors_cp_array + bx_normal_translation_arr
            
            else:
                bx_total_only_translation_arr = specific_bx_structure["MC data: Generated normal dist random samples arr"]


            bx_only_shifted_randomly_sampled_bx_pts_3Darr = biopsy_translator_step_3(randomly_sampled_bx_pts_cp_arr_dilated_and_rotated_max_simulations, bx_total_only_translation_arr)
            ### Self biopsy translate

            specific_bx_structure["MC data: bx only shifted 3darr"] = bx_only_shifted_randomly_sampled_bx_pts_3Darr

    
def biopsy_dilator_step_1(randomly_sampled_bx_pts_cp_arr, dilation_factors, structure_global_centroid_cp_arr, num_simulations):
    """
    Applies UNIFORM dilation transformations to biopsy points for multiple simulations.

    Parameters:
    - randomly_sampled_bx_pts_cp_arr (numpy.ndarray): Coordinates of biopsy points (N, 3).
    - dilation_factors (numpy.ndarray): Dilation factors per simulation and dimension (num_simulations, 3).
    - structure_global_centroid_cp_arr (numpy.ndarray): Global centroid for dilation (3,).
    - num_simulations (int): Number of simulations to generate.

    Returns:
    - numpy.ndarray: New coordinates of the dilated biopsy points for each simulation (num_simulations, N, 3).
    """
    # Calculate the separation vectors from the centroid
    separation_vectors = randomly_sampled_bx_pts_cp_arr - structure_global_centroid_cp_arr  # (N, 3)

    # Calc norms for each separation vector
    norms = cp.linalg.norm(separation_vectors, axis=1, keepdims=True)

    # Normalize the vectors
    separation_unit_vectors = separation_vectors / norms

    # Tile separation vectors for each simulation
    tiled_separation_unit_vectors = cp.tile(separation_unit_vectors, (num_simulations, 1, 1))  # (num_simulations, N, 3)

    # Tile dilation factors
    tiled_dilation_factors = cp.tile(dilation_factors.reshape(num_simulations,1,3), (1,randomly_sampled_bx_pts_cp_arr.shape[0],1))


    # Calculate dilation vectors
    tiled_separation_dilation_vectors = cp.multiply(tiled_separation_unit_vectors,tiled_dilation_factors)

    # Apply the dilation factors
    #dilated_separation_vectors = cp.tile(randomly_sampled_bx_pts_cp_arr, (num_simulations, 1, 1)) + tiled_separation_dilation_vectors  # (num_simulations, N, 3)

    # Calculate new positions for each simulation
    new_positions = structure_global_centroid_cp_arr + separation_vectors + tiled_separation_dilation_vectors  # (num_simulations, N, 3)

    return new_positions




def biopsy_rotator_step_2(dilated_biopsy_pts, rotation_angles, structure_global_centroid_cp_arr, num_simulations):
    """
    Applies variable rotation transformations to dilated biopsy points for multiple simulations.
    """
    new_positions = cp.zeros_like(dilated_biopsy_pts)

    # Apply rotations
    for i in range(num_simulations):
        # Calculate the separation vectors from the centroid
        shifted_points = dilated_biopsy_pts[i] - structure_global_centroid_cp_arr

        # Get rotation matrix from Euler angles
        rx, ry, rz = rotation_angles[i]
        Rx = cp.array([[1, 0, 0],
                       [0, cp.cos(rx), -cp.sin(rx)],
                       [0, cp.sin(rx), cp.cos(rx)]])
        Ry = cp.array([[cp.cos(ry), 0, cp.sin(ry)],
                       [0, 1, 0],
                       [-cp.sin(ry), 0, cp.cos(ry)]])
        Rz = cp.array([[cp.cos(rz), -cp.sin(rz), 0],
                       [cp.sin(rz), cp.cos(rz), 0],
                       [0, 0, 1]])

        # Combined rotation matrix
        R = Rz @ Ry @ Rx

        # Apply rotation
        rotated_points = shifted_points @ R.T

        # Shift points back
        new_positions[i] = rotated_points + structure_global_centroid_cp_arr

    return new_positions




def biopsy_translator_step_3(rotated_biopsy_pts, translation_vectors):
    """
    Applies translations to rotated biopsy points for multiple simulations using broadcasting.

    Parameters:
    - rotated_biopsy_pts (cupy.ndarray): Coordinates of rotated biopsy points for each simulation (num_simulations, N, 3).
    - translation_vectors (cupy.ndarray): Translation vectors per simulation (num_simulations, 3).
    - num_simulations (int): Number of simulations.

    Returns:
    - cupy.ndarray: New coordinates of the translated biopsy points for each simulation (num_simulations, N, 3).
    """
    # Apply translation using broadcasting
    translated_positions = rotated_biopsy_pts + translation_vectors[:, None, :]

    return translated_positions






































#### OLD TRANSLATION ONLY METHODOLOGY

def biopsy_and_structure_shifter(master_structure_reference_dict,
                                 bx_ref,
                                 structs_referenced_list,
                                 simulate_uniform_bx_shifts_due_to_bx_needle_compartment,
                                 max_simulations,
                                 plot_uniform_shifts_to_check_plotly,
                                 plot_shifted_biopsies = False,
                                 plot_translation_vectors_pointclouds = False
                                 ):
    
    # simulate every biopsy sequentially
    for patientUID,pydicom_item in master_structure_reference_dict.items():
        
        # create a dictionary of all non bx structures
        structure_organized_for_bx_data_blank_dict = MC_simulator_convex.create_patient_specific_structure_dict_for_data(pydicom_item,structs_referenced_list)
                
        for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
            specific_bx_structure_roi = specific_bx_structure["ROI"]
            num_sampled_sp_bx_pts = specific_bx_structure["Num sampled bx pts"]
            
            bx_only_shifted_randomly_sampled_bx_pts_3Darr = cupy_functions.MC_simulator_translate_sampled_bx_points_arr_bx_only_shift_cupy(specific_bx_structure, simulate_uniform_bx_shifts_due_to_bx_needle_compartment, plot_uniform_shifts_to_check_plotly, num_sampled_sp_bx_pts, max_simulations)
            
            structure_shifted_bx_data_dict = cupy_functions.MC_simulator_translate_sampled_bx_points_3darr_structure_only_shift_cupy(pydicom_item, structs_referenced_list, bx_only_shifted_randomly_sampled_bx_pts_3Darr, structure_organized_for_bx_data_blank_dict, max_simulations, num_sampled_sp_bx_pts)

            for relative_structure_key, cupy_array in structure_shifted_bx_data_dict.items():
                structure_shifted_bx_data_dict[relative_structure_key] = cp.asnumpy(cupy_array)
            
            master_structure_reference_dict[patientUID][bx_ref][specific_bx_structure_index]["MC data: bx only shifted 3darr"] = bx_only_shifted_randomly_sampled_bx_pts_3Darr
            master_structure_reference_dict[patientUID][bx_ref][specific_bx_structure_index]["MC data: bx and structure shifted dict"] = structure_shifted_bx_data_dict

            if plot_shifted_biopsies == True:
                for relative_structure_key, bx_and_structure_shifted_bx_pts_3darray in structure_shifted_bx_data_dict.items():
                    bx_only_shifted_randomly_sampled_bx_pts_2darr = np.reshape(cp.asnumpy(bx_only_shifted_randomly_sampled_bx_pts_3Darr),(-1,3))
                    bx_only_shifted_bx_pts_pcd = point_containment_tools.create_point_cloud(bx_only_shifted_randomly_sampled_bx_pts_2darr, color = np.array([0,1,0]), random_color = False)
                    bx_and_structure_shifted_bx_pts_2darr = np.reshape(bx_and_structure_shifted_bx_pts_3darray,(-1,3))
                    bx_and_structure_shifted_bx_pts_pcd = point_containment_tools.create_point_cloud(bx_and_structure_shifted_bx_pts_2darr, color = np.array([1,0,0]), random_color = False)
                    unshifted_bx_sampled_pts_pcd_copy = copy.copy(specific_bx_structure['Random uniformly sampled volume pts pcd'])
                    unshifted_bx_sampled_pts_pcd_copy.paint_uniform_color(np.array([0,1,1]))
                    non_bx_structure_type = relative_structure_key[1]
                    structure_index = relative_structure_key[3]
                    non_bx_struct_interslice_interpolation_information = master_structure_reference_dict[patientUID][non_bx_structure_type][structure_index]["Inter-slice interpolation information"]
                    non_bx_struct_interpolated_pts_np_arr = non_bx_struct_interslice_interpolation_information.interpolated_pts_np_arr
                    non_bx_struct_interpolated_pts_pcd = point_containment_tools.create_point_cloud(non_bx_struct_interpolated_pts_np_arr, color = np.array([0,0,1]))
                    plotting_funcs.plot_geometries(bx_and_structure_shifted_bx_pts_pcd,unshifted_bx_sampled_pts_pcd_copy,non_bx_struct_interpolated_pts_pcd, bx_only_shifted_bx_pts_pcd)
                    



    
    #live_display.stop()
    if plot_translation_vectors_pointclouds == True:
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            for structs in structs_referenced_list:
                for specific_structure_index, specific_structure in enumerate(pydicom_item[structs]):
                    if structs == bx_ref:
                        total_rigid_shift_vectors_arr = cp.asnumpy(specific_structure["MC data: Total rigid shift vectors arr"])
                        #randomly_sampled_bx_pts_arr = specific_bx_structure["Random uniformly sampled volume pts arr"]
                        plotting_funcs.plot_point_clouds(total_rigid_shift_vectors_arr)
                        plotting_funcs.plotly_3dscatter_arbitrary_number_of_arrays([total_rigid_shift_vectors_arr], title_text = str(patientUID) + str(specific_structure["ROI"]))
                    else: 
                        total_rigid_shift_vectors_arr = cp.asnumpy(specific_structure["MC data: Generated normal dist random samples arr"])
                        plotting_funcs.plot_point_clouds(total_rigid_shift_vectors_arr)
                        plotting_funcs.plotly_3dscatter_arbitrary_number_of_arrays([total_rigid_shift_vectors_arr], title_text = str(patientUID) + str(specific_structure["ROI"]))

