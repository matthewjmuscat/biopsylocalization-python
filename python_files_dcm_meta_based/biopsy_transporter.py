import centroid_finder
import numpy as np

def biopsy_transporter_centroid(pydicom_item,
                                specific_structure,
                                threeDdata_zslice_list
                                ):
    
    # first extract the appropriate relative structure to transform biopsies to
    relative_structure_ref_num_from_bx_info = specific_structure["Relative structure ref #"]
    relative_structure_struct_type_from_bx_info = specific_structure["Relative structure type"]
    #for relative_struct_type in simulate_biopsies_relative_to_struct_type_list:
    for relative_specific_structure_index, relative_specific_structure in enumerate(pydicom_item[relative_structure_struct_type_from_bx_info]):
        if relative_structure_ref_num_from_bx_info == relative_specific_structure["Ref #"]:
            simulated_bx_relative_to_specific_structure_index = relative_specific_structure_index
            #simulate_biopsies_relative_to_specific_structure_struct_type = relative_struct_type
            break
        else:
            pass

    relative_structure_for_sim_bx_global_centroid = pydicom_item[relative_structure_struct_type_from_bx_info][simulated_bx_relative_to_specific_structure_index]["Structure global centroid"].copy()
    threeDdata_arr_temp = np.concatenate(threeDdata_zslice_list, axis=0)
    simulated_bx_global_centroid_before_translation = centroid_finder.centeroidfinder_numpy_3D(threeDdata_arr_temp)
    translation_vector_to_relative_structure_centroid = relative_structure_for_sim_bx_global_centroid - simulated_bx_global_centroid_before_translation
    threeDdata_zslice_list_temp = threeDdata_zslice_list.copy()
    for bx_zslice_arr_index, bx_zslice_arr in enumerate(threeDdata_zslice_list_temp):
        temp_bx_zslice_arr = bx_zslice_arr.copy()
        translated_bx_zslice_arr = temp_bx_zslice_arr + translation_vector_to_relative_structure_centroid
        threeDdata_zslice_list_temp[bx_zslice_arr_index] = translated_bx_zslice_arr
    threeDdata_zslice_list = threeDdata_zslice_list_temp

    return threeDdata_zslice_list



def biopsy_transporter_optimal(pydicom_item,
                                specific_structure,
                                threeDdata_zslice_list
                                ):
    
    # first extract the appropriate relative structure to transform biopsies to
    relative_structure_ref_num_from_bx_info = specific_structure["Relative structure ref #"]
    relative_structure_struct_type_from_bx_info = specific_structure["Relative structure type"]
    for relative_specific_structure_index, relative_specific_structure in enumerate(pydicom_item[relative_structure_struct_type_from_bx_info]):
        if relative_structure_ref_num_from_bx_info == relative_specific_structure["Ref #"]:
            simulated_bx_relative_to_specific_structure_index = relative_specific_structure_index
            break
        else:
            pass
            
    optimal_locations_dataframe = pydicom_item[relative_structure_struct_type_from_bx_info][simulated_bx_relative_to_specific_structure_index]["Biopsy optimization: Optimal biopsy location dataframe"]
    # optimal_locations_dataframe should have only one row, however we do it this way just in case it doesnt't! ie if it has more
    # than one value, we take the position that is closest to the dil centroid!
    relative_structure_for_sim_optimal_position_vector = np.array(optimal_locations_dataframe[optimal_locations_dataframe['Dist to DIL centroid'] == optimal_locations_dataframe['Dist to DIL centroid'].min()].at[0,'Test location vector'])

    
    threeDdata_arr_temp = np.concatenate(threeDdata_zslice_list, axis=0)
    simulated_bx_global_centroid_before_translation = centroid_finder.centeroidfinder_numpy_3D(threeDdata_arr_temp)
    translation_vector_to_relative_structure_centroid = relative_structure_for_sim_optimal_position_vector - simulated_bx_global_centroid_before_translation
    threeDdata_zslice_list_temp = threeDdata_zslice_list.copy()
    for bx_zslice_arr_index, bx_zslice_arr in enumerate(threeDdata_zslice_list_temp):
        temp_bx_zslice_arr = bx_zslice_arr.copy()
        translated_bx_zslice_arr = temp_bx_zslice_arr + translation_vector_to_relative_structure_centroid
        threeDdata_zslice_list_temp[bx_zslice_arr_index] = translated_bx_zslice_arr
    threeDdata_zslice_list = threeDdata_zslice_list_temp

    return threeDdata_zslice_list