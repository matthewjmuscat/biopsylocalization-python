import numpy as np
import pandas

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


