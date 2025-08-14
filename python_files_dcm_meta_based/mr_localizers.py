import numpy as np
import pandas
import custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p
import lattice_reconstruction_tools
import misc_tools

def mr_localization_dataframe_version(bx_only_shifted_stacked_2darr,
                                              patientUID, 
                                              bx_structure_info_dict, 
                                              mr_lattice_phys_space_data_KDtree, 
                                              mr_vals_arr, 
                                              num_mr_calc_NN,
                                              num_MC_mr_sims,
                                              num_bx_points,
                                              idw_power):
    
    """
    Note that this function uses an inverse distance weighting to interpolate between dose points on the mr_adc lattice!
    The inverse distance weighting is specified by the idw_power, and the number of NN to find is determined by num_mr_calc_NN!
    """


    structureID = bx_structure_info_dict["Structure ID"]
    struct_type = bx_structure_info_dict["Struct ref type"]
    structure_reference_number = bx_structure_info_dict["Dicom ref num"]
    structure_index_number = bx_structure_info_dict["Index number"]

    total_num_trials_and_nominal_times_bx_points = (num_MC_mr_sims+1)*num_bx_points

    # perform NN search in one line!
    NN_search_output = mr_lattice_phys_space_data_KDtree.query(bx_only_shifted_stacked_2darr, k=num_mr_calc_NN)

    NN_distances_arr = NN_search_output[0] 
    NN_indices_arr = NN_search_output[1] 
    # Reshaping is performed to handle the case of num_mr_calc_NN = 1, if this is the output arrays are the wrong dimensions, the below two lines corrects this 
    NN_distances_2d_arr = np.reshape(NN_distances_arr, (total_num_trials_and_nominal_times_bx_points,num_mr_calc_NN)) # first index of the 2d arr is the point tested, second index is the kth nearest neighbour
    NN_indices_2d_arr = np.reshape(NN_indices_arr, (total_num_trials_and_nominal_times_bx_points,num_mr_calc_NN)) # first index of the 2d arr is the point tested, second index is the kth nearest neighbour

    comparison_structure_NN_distances_reciprocal = np.reciprocal(NN_distances_2d_arr)
    comparison_structure_NN_distances_reciprocal_power = comparison_structure_NN_distances_reciprocal**idw_power
    comparison_structure_NN_distances_reciprocal_fixed = np.nan_to_num(comparison_structure_NN_distances_reciprocal_power, copy=True, nan=0.0, posinf=None, neginf=None) # replaces Inf with very large number!!!!!! This fixes the problem for distance = 0, because you have a divide by zero!
    
    nearest_points_on_comparison_struct_list = mr_lattice_phys_space_data_KDtree.data[NN_indices_2d_arr].tolist()
    nearest_mr_vals_arr = mr_vals_arr[NN_indices_2d_arr]
    nearest_mr_vals_list = nearest_mr_vals_arr.tolist()
    nearest_mr_vals_weighted_mean_arr = np.average(nearest_mr_vals_arr, axis=1, weights = comparison_structure_NN_distances_reciprocal_fixed)

    NN_distances_2d_list = NN_distances_2d_arr.tolist()
    comparison_structure_NN_distances_reciprocal_fixed_2d_list = comparison_structure_NN_distances_reciprocal_fixed.tolist()


    dose_nearest_neighbour_results_dict_for_dataframe = {"Patient ID": [patientUID]*total_num_trials_and_nominal_times_bx_points,
                                                        "Struct ID": [structureID]*total_num_trials_and_nominal_times_bx_points,
                                                        "Struct type": [struct_type]*total_num_trials_and_nominal_times_bx_points,
                                                        "Struct dicom ref num": [structure_reference_number]*total_num_trials_and_nominal_times_bx_points,
                                                        "Struct index": [structure_index_number]*total_num_trials_and_nominal_times_bx_points,
                                                        "Original pt index": np.tile(np.arange(0,num_bx_points), num_MC_mr_sims+1),
                                                        "Struct test pt vec": bx_only_shifted_stacked_2darr.tolist(),
                                                        "Trial num": np.repeat(np.arange(0,num_MC_mr_sims+1),num_bx_points),
                                                        "MR val (interpolated)": nearest_mr_vals_weighted_mean_arr,
                                                        "Nearest phys space points": nearest_points_on_comparison_struct_list,
                                                        "Nearest distances": NN_distances_2d_list,
                                                        "Averaging weights": comparison_structure_NN_distances_reciprocal_fixed_2d_list,
                                                        "Nearest MR vals": nearest_mr_vals_list
                                                        }
    
    dose_nearest_neighbour_results_dataframe = pandas.DataFrame(dose_nearest_neighbour_results_dict_for_dataframe)

    return dose_nearest_neighbour_results_dataframe





def grab_mr_adc_2d_arr(pydicom_item,
                       mr_adc_ref,
                       filter_out_negatives = True):


    mr_adc_subdict = pydicom_item[mr_adc_ref]

    adc_mr_phys_space_arr = lattice_reconstruction_tools.reconstruct_mr_lattice_with_coordinates_from_dict_v2(mr_adc_subdict, filter_out_negatives = filter_out_negatives)
    # This will return an (N, 4) array where each row is (x, y, z, ADC value)

    return adc_mr_phys_space_arr

"""
def determine_mr_points_within_prostate_minus_urethra_minus_dils(pydicom_item,
                                                                 adc_mr_phys_space_arr):




    sp_patient_selected_structure_info_dataframe = pydicom_item[all_ref_key]["Multi-structure pre-processing output dataframes dict"]["Selected structures"]                 

    specific_prostate_info_df = sp_patient_selected_structure_info_dataframe[sp_patient_selected_structure_info_dataframe["Struct ref type"] == oar_ref]
    selected_prostate_info = specific_prostate_info_df.to_dict('records')[0]

    prostate_ID = selected_prostate_info["Structure ID"]
    prostate_ref_type = selected_prostate_info["Struct ref type"]
    prostate_ref_num = selected_prostate_info["Dicom ref num"]
    prostate_structure_index = selected_prostate_info["Index number"]
    prostate_found_bool = selected_prostate_info["Struct found bool"]


    if prostate_found_bool == True:
        prostate_centroid = pydicom_item[prostate_ref_type][prostate_structure_index]["Structure global centroid"].reshape(3)
    else: 
        important_info.add_text_line('Prostate not found! Defaulting prostate centroid to Zero-vector')
        prostate_centroid = np.array([0,0,0])



    structureID_default = "Initializing"
    num_dil_structs_patient_specific = master_structure_info_dict["By patient"][patientUID][dil_ref]["Num structs"]
    processing_structures_task_main_description = "[cyan]Processing structures [{},{}]...".format(patientUID,structureID_default)
    processing_structures_task = structures_progress.add_task(processing_structures_task_main_description, total=num_dil_structs_patient_specific)
    for specific_dil_structure_index, specific_dil_structure in enumerate(pydicom_item[dil_ref]):
        structureID_dil = specific_dil_structure["ROI"]
        structure_reference_number_dil = specific_dil_structure["Ref #"]
        processing_structures_task_main_description = "[cyan]Processing structures [{},{}]...".format(patientUID,structureID_dil)
        structures_progress.update(processing_structures_task, description = processing_structures_task_main_description)

        ### FIND OPTIMAL POSITION FOR BIOPSY SAMPLING (DIL ONLY)
        structure_info = misc_tools.specific_structure_info_dict_creator('given', specific_structure = specific_dil_structure)

        interslice_interpolation_information = specific_dil_structure["Inter-slice interpolation information"]
        interpolated_pts_np_arr = interslice_interpolation_information.interpolated_pts_np_arr
        interpolated_zvals_list = interslice_interpolation_information.zslice_vals_after_interpolation_list
        zslices_list = interslice_interpolation_information.interpolated_pts_list
        # Extract the dil centroid
        dil_global_centroid = specific_dil_structure["Structure global centroid"]


        ### OLD METHODOLOGY FOR REMOVING POINTS OUTSIDE OF DIL

    ### NEW METHODOLOGY FOR REMOVING POINTS OUTSIDE OF DIL
    #pr = cProfile.Profile()
    #pr.enable()

    # maps the first test structure to the first relative structure (since there is only 1 test structure and 1 relative structure)              
    test_struct_to_relative_struct_1d_mapping_array = np.array([0])          
    log_sub_dirs_list = [patientUID, structureID_dil]
    if generate_cuda_log_files_biopsy_optimizer == True:
        custom_cuda_log_file_name = "cuda_dil_bioposy_optimization_lattice.txt"
    else:
        custom_cuda_log_file_name = None 


    containment_result_for_all_lattice_points_cp_arr, prepper_output_tuple = custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p.custom_point_containment_mother_function([zslices_list],
                                            all_geometries_centered_cubic_lattice_arr[np.newaxis,:,:],
                                            test_struct_to_relative_struct_1d_mapping_array,
                                            constant_z_slice_polygons_handler_option = constant_z_slice_polygons_handler_option,
                                            remove_consecutive_duplicate_points_in_polygons = remove_consecutive_duplicate_points_in_polygons,
                                            log_sub_dirs_list = log_sub_dirs_list,
                                            log_file_name = custom_cuda_log_file_name,
                                            include_edges_in_log = include_edges_in_log_files,
                                            kernel_type = custom_cuda_kernel_type)
"""  


def test_points_of_given_2d_lattice_from_within_given_structure_and_return_dataframe_type_2III(given_2d_lattice_arr,
                                                                  zslices_list,
                                                                  structure_info,
                                                                  constant_z_slice_polygons_handler_option, 
                                                                  remove_consecutive_duplicate_points_in_polygons,
                                                                  custom_cuda_kernel_type,
                                                                  associated_value_str = "MR ADC value",
                                                                  minimalize_dataframe = True):
    



    

    # maps the first test structure to the first relative structure (since there is only 1 test structure and 1 relative structure)              
    test_struct_to_relative_struct_1d_mapping_array = np.array([0])

    containment_result_for_all_lattice_points_cp_arr, prepper_output_tuple = custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p.custom_point_containment_mother_function([zslices_list],
                                            given_2d_lattice_arr[np.newaxis,:,0:3],
                                            test_struct_to_relative_struct_1d_mapping_array,
                                            constant_z_slice_polygons_handler_option = constant_z_slice_polygons_handler_option,
                                            remove_consecutive_duplicate_points_in_polygons = remove_consecutive_duplicate_points_in_polygons,
                                            log_sub_dirs_list = [],
                                            log_file_name = None,
                                            include_edges_in_log = False,
                                            kernel_type = custom_cuda_kernel_type)


    containment_info_for_all_lattice_points_grand_pandas_dataframe = custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p.create_containment_results_dataframe_type_2III(structure_info, 
                                                                                                                                prepper_output_tuple[0], 
                                                                                                                                given_2d_lattice_arr[np.newaxis,:,:], 
                                                                                                                                containment_result_for_all_lattice_points_cp_arr,
                                                                                                                                convert_to_categorical_and_downcast = True,
                                                                                                                                do_not_convert_column_names_to_categorical = ["Pt contained bool"],
                                                                                                                                float_dtype = np.float32,
                                                                                                                                int_dtype = np.int32,
                                                                                                                                associated_value_str=associated_value_str,
                                                                                                                                minimalize_dataframe = minimalize_dataframe)

    return containment_info_for_all_lattice_points_grand_pandas_dataframe



