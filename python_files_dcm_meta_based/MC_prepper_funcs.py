import MC_simulator_convex
import cupy as cp
import cupy_functions
import numpy as np
import copy 
import point_containment_tools
import plotting_funcs 

    
def generate_shifts(master_structure_reference_dict,
                    simulate_uniform_bx_shifts_due_to_bx_needle_compartment,
                    bx_ref,
                    biopsy_needle_compartment_length,
                    max_simulations,
                    structs_referenced_list):

    # simulate all structure shifts in parallel and update the master reference dict
    for patientUID,pydicom_item in master_structure_reference_dict.items():
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

