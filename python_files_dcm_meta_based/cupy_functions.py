import cupy as cp
import plotting_funcs


def MC_simulator_all_structs_dilations_generator_cupy(pydicom_item, structs_referenced_list, max_simulations):
    
    sp_structure_normal_dist_shift_samples_and_structure_reference_list = []
    for structure_type in structs_referenced_list:
        for specific_structure_index, specific_structure in enumerate(pydicom_item[structure_type]):
            uncertainty_data_obj = specific_structure["Uncertainty data"]
            sp_struct_uncertainty_data_dilations_mean_arr = uncertainty_data_obj.uncertainty_data_dilations_mean_arr
            sp_struct_uncertainty_data_dilations_sigma_arr = uncertainty_data_obj.uncertainty_data_dilations_sigma_arr

            structure_normal_dist_dilations_distances_samples_arr = cp.array([ 
            cp.random.normal(loc=sp_struct_uncertainty_data_dilations_mean_arr[0], scale=sp_struct_uncertainty_data_dilations_sigma_arr[0], size=max_simulations),    
            cp.random.normal(loc=sp_struct_uncertainty_data_dilations_mean_arr[1], scale=sp_struct_uncertainty_data_dilations_sigma_arr[1], size=max_simulations)],   
            dtype = float).T
            
            generated_shifts_info_list = [structure_type, specific_structure_index, structure_normal_dist_dilations_distances_samples_arr]
            
            sp_structure_normal_dist_shift_samples_and_structure_reference_list.append(generated_shifts_info_list)
    

    return sp_structure_normal_dist_shift_samples_and_structure_reference_list

def MC_simulator_all_structs_rotations_generator_cupy(pydicom_item, structs_referenced_list, max_simulations):
    
    sp_structure_normal_dist_shift_samples_and_structure_reference_list = []
    for structure_type in structs_referenced_list:
        for specific_structure_index, specific_structure in enumerate(pydicom_item[structure_type]):
            uncertainty_data_obj = specific_structure["Uncertainty data"]
            sp_struct_uncertainty_data_rotations_mean_arr = uncertainty_data_obj.uncertainty_data_rotations_mean_arr
            sp_struct_uncertainty_data_rotations_sigma_arr = uncertainty_data_obj.uncertainty_data_rotations_sigma_arr

            structure_normal_dist_rotations_factors_samples_arr = cp.array([ 
            cp.random.normal(loc=sp_struct_uncertainty_data_rotations_mean_arr[0], scale=sp_struct_uncertainty_data_rotations_sigma_arr[0], size=max_simulations),  
            cp.random.normal(loc=sp_struct_uncertainty_data_rotations_mean_arr[1], scale=sp_struct_uncertainty_data_rotations_sigma_arr[1], size=max_simulations),  
            cp.random.normal(loc=sp_struct_uncertainty_data_rotations_mean_arr[2], scale=sp_struct_uncertainty_data_rotations_sigma_arr[2], size=max_simulations)],   
            dtype = float).T
            
            generated_shifts_info_list = [structure_type, specific_structure_index, structure_normal_dist_rotations_factors_samples_arr]
            
            sp_structure_normal_dist_shift_samples_and_structure_reference_list.append(generated_shifts_info_list)
    

    return sp_structure_normal_dist_shift_samples_and_structure_reference_list



def MC_simulator_shift_biopsy_structures_uniform_generator_cupy(patient_dict, bx_ref, biopsy_needle_compartment_length, num_simulations):
    # build args list for parallel computing
    sp_bx_structure_uniform_dist_shift_samples_and_structure_reference_list = []
    for specific_bx_structure_index, specific_bx_structure in enumerate(patient_dict[bx_ref]):
        bx_core_length = specific_bx_structure['Reconstructed biopsy cylinder length (from contour data)']
        core_length_compartment_length_difference = biopsy_needle_compartment_length - bx_core_length
        if core_length_compartment_length_difference <= 0:
            core_length_compartment_length_difference = 0.
        
        structure_uniform_dist_shift_samples_arr = cp.random.uniform(low=0, high=core_length_compartment_length_difference, size=num_simulations)
        generated_shifts_info_list = [bx_ref, specific_bx_structure_index, structure_uniform_dist_shift_samples_arr]
        sp_bx_structure_uniform_dist_shift_samples_and_structure_reference_list.append(generated_shifts_info_list)

    return sp_bx_structure_uniform_dist_shift_samples_and_structure_reference_list



def MC_simulator_shift_all_structures_generator_cupy(patient_dict, structs_referenced_list, num_simulations):

    # build args list for parallel computing
    sp_structure_normal_dist_shift_samples_and_structure_reference_list = []
    #patient_dict_updated_with_generated_samples = patient_dict.copy()
    for structure_type in structs_referenced_list:
        for specific_structure_index, specific_structure in enumerate(patient_dict[structure_type]):
            #spec_structure_zslice_wise_delaunay_obj_list = specific_structure["Delaunay triangulation zslice-wise list"]
            uncertainty_data_obj = specific_structure["Uncertainty data"]
            sp_struct_uncertainty_data_mean_arr = uncertainty_data_obj.uncertainty_data_mean_arr
            sp_struct_uncertainty_data_sigma_arr = uncertainty_data_obj.uncertainty_data_sigma_arr

            structure_normal_dist_shift_samples_arr = cp.array([ 
            cp.random.normal(loc=sp_struct_uncertainty_data_mean_arr[0], scale=sp_struct_uncertainty_data_sigma_arr[0], size=num_simulations),  
            cp.random.normal(loc=sp_struct_uncertainty_data_mean_arr[1], scale=sp_struct_uncertainty_data_sigma_arr[1], size=num_simulations),  
            cp.random.normal(loc=sp_struct_uncertainty_data_mean_arr[2], scale=sp_struct_uncertainty_data_sigma_arr[2], size=num_simulations)],   
            dtype = float).T
            
            generated_shifts_info_list = [structure_type, specific_structure_index, structure_normal_dist_shift_samples_arr]
            
            sp_structure_normal_dist_shift_samples_and_structure_reference_list.append(generated_shifts_info_list)
    

    return sp_structure_normal_dist_shift_samples_and_structure_reference_list




def MC_simulator_translate_sampled_bx_points_arr_bx_only_shift_cupy(specific_bx_structure, simulate_uniform_bx_shifts_due_to_bx_needle_compartment, plot_uniform_shifts_to_check_plotly, num_sampled_bx_pts, max_simulations):
    """
    create a 3d array that stores all the shifted bx data where each 3d slice is the shifted bx data for 
    a fixed sampled bx shift, ie each slice is a sampled bx shift trial. Note though that if the uniform 
    compartment shifts are done, these are still slices of constant shift vectors, but they are now 
    composed of two shifts, ie each slice is the result of a unique uniform shift vector plus a unique norm shift vector 
    applied to each bx point (row).
    """
    
    randomly_sampled_bx_pts_arr = specific_bx_structure["Random uniformly sampled volume pts arr"]
    randomly_sampled_bx_pts_cp_arr = cp.array(randomly_sampled_bx_pts_arr) # convert numpy array to cp array
    # prep the sampled biopsy points to be added via vectorization
    randomly_sampled_bx_pts_cp_arr_resized_3darr = cp.resize(randomly_sampled_bx_pts_cp_arr,(1,num_sampled_bx_pts,3))
    randomly_sampled_bx_pts_cp_arr_3darr = cp.repeat(randomly_sampled_bx_pts_cp_arr_resized_3darr,max_simulations,0)

    randomly_sampled_normal_bx_shifts_cp_arr = cp.array(specific_bx_structure["MC data: Generated normal dist random samples arr"])
    #num_shift_vecs = randomly_sampled_normal_bx_shifts_cp_arr.shape[0]
    #randomly_sampled_normal_bx_shifts_cp_arr_reshaped_for_vectorization = cp.reshape(randomly_sampled_normal_bx_shifts_cp_arr,(max_simulations,1,3))
    #randomly_sampled_normal_bx_shifts_cp_arr_3d_arr_for_vectorization = cp.tile(randomly_sampled_normal_bx_shifts_cp_arr_reshaped_for_vectorization,(1,num_sampled_bx_pts,1))
    
    if simulate_uniform_bx_shifts_due_to_bx_needle_compartment == True:
        random_uniformly_sampled_bx_shifts_cp_arr = specific_bx_structure["MC data: Generated uniform dist (biopsy needle compartment) random distance (z_needle) samples arr"]
        #num_uniform_shifts = random_uniformly_sampled_bx_shifts_cp_arr.shape[0]
        # notice the minus sign below!!
        bx_needle_centroid_vec_tip_to_handle_unit_vec = -specific_bx_structure["Centroid line unit vec (bx needle base to bx needle tip)"]
        bx_needle_centroid_vec_tip_to_handle_unit_cp_vec = cp.array(bx_needle_centroid_vec_tip_to_handle_unit_vec)
        bx_needle_centroid_vec_tip_to_handle_unit_cp_arr = cp.tile(bx_needle_centroid_vec_tip_to_handle_unit_cp_vec,(max_simulations,1))
        bx_needle_uniform_compartment_shift_vectors_cp_array = cp.multiply(bx_needle_centroid_vec_tip_to_handle_unit_cp_arr,random_uniformly_sampled_bx_shifts_cp_arr[...,None]) # The [...,None] converts the row vector to a column vector for proper element-wise multiplication
        specific_bx_structure["MC data: Generated uniform (biopsy needle compartment) random vectors (z_needle) samples arr"] = cp.asnumpy(bx_needle_uniform_compartment_shift_vectors_cp_array)
        total_rigid_shift_vectors_cp_arr = bx_needle_uniform_compartment_shift_vectors_cp_array + randomly_sampled_normal_bx_shifts_cp_arr
        specific_bx_structure["MC data: Total rigid shift vectors arr"] = cp.asnumpy(total_rigid_shift_vectors_cp_arr)

        if plot_uniform_shifts_to_check_plotly == True:
            bx_needle_uniform_compartment_shift_vectors_cp_arr_reshaped_for_vectorization = cp.reshape(bx_needle_uniform_compartment_shift_vectors_cp_array,(max_simulations,1,3))
            bx_needle_uniform_compartment_shift_vectors_cp_arr_3d_arr_for_vectorization = cp.tile(bx_needle_uniform_compartment_shift_vectors_cp_arr_reshaped_for_vectorization,(1,num_sampled_bx_pts,1))
            
            randomly_sampled_bx_pts_arr_bx_uniform_compartment_only_shift_cp_3d_arr = randomly_sampled_bx_pts_cp_arr_3darr + bx_needle_uniform_compartment_shift_vectors_cp_arr_3d_arr_for_vectorization
            for uniform_only_shifted_slice in randomly_sampled_bx_pts_arr_bx_uniform_compartment_only_shift_cp_3d_arr: 
                uniform_only_shifted_slice_np_arr = cp.asnumpy(uniform_only_shifted_slice)
                plotting_funcs.plotly_3dscatter_arbitrary_number_of_arrays([uniform_only_shifted_slice_np_arr, randomly_sampled_bx_pts_arr], colors_for_arrays_list = ['blue','red'], aspect_mode_input = 'data')
        
    else:
        total_rigid_shift_vectors_cp_arr = randomly_sampled_normal_bx_shifts_cp_arr

    total_rigid_shift_vectors_cp_arr_reshaped_for_vectorization = cp.reshape(total_rigid_shift_vectors_cp_arr,(max_simulations,1,3))
    total_rigid_shift_vectors_cp_arr_3d_arr_for_vectorization = cp.tile(total_rigid_shift_vectors_cp_arr_reshaped_for_vectorization,(1,num_sampled_bx_pts,1))

    randomly_sampled_bx_pts_arr_bx_only_shift_cp_3Darr = randomly_sampled_bx_pts_cp_arr_3darr + total_rigid_shift_vectors_cp_arr_3d_arr_for_vectorization

    return randomly_sampled_bx_pts_arr_bx_only_shift_cp_3Darr



def MC_simulator_translate_sampled_bx_points_3darr_structure_only_shift_cupy(pydicom_item, structs_referenced_list, bx_only_shifted_randomly_sampled_bx_pts_3Darr, blank_structure_shifted_bx_data_dict, max_simulations, num_sampled_sp_bx_pts):
    # do each non bx structure sequentially
    structure_shifted_bx_data_dict = blank_structure_shifted_bx_data_dict.copy()
    for non_bx_struct_type in structs_referenced_list[1:]:
        for specific_non_bx_struct_index,specific_non_bx_struct in enumerate(pydicom_item[non_bx_struct_type]):
            specific_non_bx_struct_roi = specific_non_bx_struct["ROI"]
            specific_non_bx_struct_refnum = specific_non_bx_struct["Ref #"]

            non_bx_struct_cp_arr = specific_non_bx_struct["MC data: Generated normal dist random samples arr"]
            non_bx_struct_cp_arr_for_bx_shift = -non_bx_struct_cp_arr
            non_bx_struct_cp_arr_for_bx_shift_reshaped_for_vectorization = cp.reshape(non_bx_struct_cp_arr_for_bx_shift,(max_simulations,1,3))
            non_bx_struct_cp_arr_for_bx_shift_3d_arr_for_vectorization = cp.tile(non_bx_struct_cp_arr_for_bx_shift_reshaped_for_vectorization,(1,num_sampled_sp_bx_pts,1))

            bx_data_both_non_bx_structure_shifted_and_bx_structure_shifted_3darr = bx_only_shifted_randomly_sampled_bx_pts_3Darr + non_bx_struct_cp_arr_for_bx_shift_3d_arr_for_vectorization

            structure_shifted_bx_data_dict[specific_non_bx_struct_roi,non_bx_struct_type,specific_non_bx_struct_refnum,specific_non_bx_struct_index] = bx_data_both_non_bx_structure_shifted_and_bx_structure_shifted_3darr
        
    return structure_shifted_bx_data_dict







def fanova_translate_sampled_bx_points_arr_bx_only_shift_cupy(specific_bx_structure, 
                                                              simulate_uniform_bx_shifts_due_to_bx_needle_compartment, 
                                                              plot_uniform_shifts_to_check_plotly, 
                                                              num_sampled_bx_pts, 
                                                              max_simulations, 
                                                              fanova_bx_shift_samples_2d_arr,
                                                              matrix_key):
    """
    create a 3d array that stores all the shifted bx data where each 3d slice is the shifted bx data for 
    a fixed sampled bx shift, ie each slice is a sampled bx shift trial. Note though that if the uniform 
    compartment shifts are done, these are still slices of constant shift vectors, but they are now 
    composed of two shifts, ie each slice is the result of a unique uniform shift vector plus a unique norm shift vector 
    applied to each bx point (row).
    """
    
    randomly_sampled_bx_pts_arr = specific_bx_structure["Random uniformly sampled volume pts arr"]
    randomly_sampled_bx_pts_cp_arr = cp.array(randomly_sampled_bx_pts_arr) # convert numpy array to cp array
    # prep the sampled biopsy points to be added via vectorization
    randomly_sampled_bx_pts_cp_arr_resized_3darr = cp.resize(randomly_sampled_bx_pts_cp_arr,(1,num_sampled_bx_pts,3))
    randomly_sampled_bx_pts_cp_arr_3darr = cp.repeat(randomly_sampled_bx_pts_cp_arr_resized_3darr,max_simulations,0)

    randomly_sampled_normal_bx_shifts_cp_arr = fanova_bx_shift_samples_2d_arr[:,0:3]
    
    if simulate_uniform_bx_shifts_due_to_bx_needle_compartment == True:
        random_uniformly_sampled_bx_shifts_cp_arr = fanova_bx_shift_samples_2d_arr[:,3]
        #num_uniform_shifts = random_uniformly_sampled_bx_shifts_cp_arr.shape[0]
        # notice the minus sign below!!
        bx_needle_centroid_vec_tip_to_handle_unit_vec = -specific_bx_structure["Centroid line unit vec (bx needle base to bx needle tip)"]
        bx_needle_centroid_vec_tip_to_handle_unit_cp_vec = cp.array(bx_needle_centroid_vec_tip_to_handle_unit_vec)
        bx_needle_centroid_vec_tip_to_handle_unit_cp_arr = cp.tile(bx_needle_centroid_vec_tip_to_handle_unit_cp_vec,(max_simulations,1))
        bx_needle_uniform_compartment_shift_vectors_cp_array = cp.multiply(bx_needle_centroid_vec_tip_to_handle_unit_cp_arr,random_uniformly_sampled_bx_shifts_cp_arr[...,None]) # The [...,None] converts the row vector to a column vector for proper element-wise multiplication
        specific_bx_structure["FANOVA: Generated uniform (biopsy needle compartment) random vectors samples arr dict"][matrix_key] = bx_needle_uniform_compartment_shift_vectors_cp_array
        total_rigid_shift_vectors_cp_arr = bx_needle_uniform_compartment_shift_vectors_cp_array + randomly_sampled_normal_bx_shifts_cp_arr
        specific_bx_structure["FANOVA: Total rigid shift vectors arr dict"][matrix_key] = total_rigid_shift_vectors_cp_arr

        if plot_uniform_shifts_to_check_plotly == True:
            bx_needle_uniform_compartment_shift_vectors_cp_arr_reshaped_for_vectorization = cp.reshape(bx_needle_uniform_compartment_shift_vectors_cp_array,(max_simulations,1,3))
            bx_needle_uniform_compartment_shift_vectors_cp_arr_3d_arr_for_vectorization = cp.tile(bx_needle_uniform_compartment_shift_vectors_cp_arr_reshaped_for_vectorization,(1,num_sampled_bx_pts,1))
            
            randomly_sampled_bx_pts_arr_bx_uniform_compartment_only_shift_cp_3d_arr = randomly_sampled_bx_pts_cp_arr_3darr + bx_needle_uniform_compartment_shift_vectors_cp_arr_3d_arr_for_vectorization
            for uniform_only_shifted_slice in randomly_sampled_bx_pts_arr_bx_uniform_compartment_only_shift_cp_3d_arr: 
                uniform_only_shifted_slice_np_arr = cp.asnumpy(uniform_only_shifted_slice)
                plotting_funcs.plotly_3dscatter_arbitrary_number_of_arrays([uniform_only_shifted_slice_np_arr, randomly_sampled_bx_pts_arr], colors_for_arrays_list = ['blue','red'], aspect_mode_input = 'data')
        
    else:
        total_rigid_shift_vectors_cp_arr = randomly_sampled_normal_bx_shifts_cp_arr

    total_rigid_shift_vectors_cp_arr_reshaped_for_vectorization = cp.reshape(total_rigid_shift_vectors_cp_arr,(max_simulations,1,3))
    total_rigid_shift_vectors_cp_arr_3d_arr_for_vectorization = cp.tile(total_rigid_shift_vectors_cp_arr_reshaped_for_vectorization,(1,num_sampled_bx_pts,1))

    randomly_sampled_bx_pts_arr_bx_only_shift_cp_3Darr = randomly_sampled_bx_pts_cp_arr_3darr + total_rigid_shift_vectors_cp_arr_3d_arr_for_vectorization

    return randomly_sampled_bx_pts_arr_bx_only_shift_cp_3Darr


