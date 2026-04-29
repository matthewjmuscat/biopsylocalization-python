import centroid_finder
import numpy as np


def _find_relative_structure(pydicom_item,
                             specific_structure
                             ):

    relative_structure_ref_num_from_bx_info = specific_structure["Relative structure ref #"]
    relative_structure_struct_type_from_bx_info = specific_structure["Relative structure type"]

    for relative_specific_structure_index, relative_specific_structure in enumerate(pydicom_item[relative_structure_struct_type_from_bx_info]):
        if relative_structure_ref_num_from_bx_info == relative_specific_structure["Ref #"]:
            return relative_specific_structure_index, relative_specific_structure

    raise ValueError(
        "Could not find relative structure for simulated biopsy {} ({}, {}).".format(
            specific_structure.get("ROI"),
            relative_structure_struct_type_from_bx_info,
            relative_structure_ref_num_from_bx_info,
        )
    )


def _translate_biopsy_zslice_list_to_target_vector(threeDdata_zslice_list,
                                                   target_vector
                                                   ):

    threeDdata_arr_temp = np.concatenate(threeDdata_zslice_list, axis=0)
    simulated_bx_global_centroid_before_translation = centroid_finder.centeroidfinder_numpy_3D(threeDdata_arr_temp)
    translation_vector_to_target = target_vector - simulated_bx_global_centroid_before_translation
    threeDdata_zslice_list_temp = threeDdata_zslice_list.copy()
    for bx_zslice_arr_index, bx_zslice_arr in enumerate(threeDdata_zslice_list_temp):
        temp_bx_zslice_arr = bx_zslice_arr.copy()
        translated_bx_zslice_arr = temp_bx_zslice_arr + translation_vector_to_target
        threeDdata_zslice_list_temp[bx_zslice_arr_index] = translated_bx_zslice_arr

    return threeDdata_zslice_list_temp


def _resolve_optimal_target_vector(relative_specific_structure):
    optimal_locations_dataframe = relative_specific_structure["Biopsy optimization: Optimal biopsy location dataframe"]

    optimal_row = optimal_locations_dataframe[
        optimal_locations_dataframe['Dist to DIL centroid'] == optimal_locations_dataframe['Dist to DIL centroid'].min()
    ].iloc[0]

    return np.array([
        optimal_row['Test location (X)'],
        optimal_row['Test location (Y)'],
        optimal_row['Test location (Z)']
    ])


def transport_planned_biopsy(pydicom_item,
                             specific_structure,
                             threeDdata_zslice_list,
                             transport_family
                             ):

    if transport_family == "identity":
        return threeDdata_zslice_list

    _, relative_specific_structure = _find_relative_structure(pydicom_item,
                                                              specific_structure)

    if transport_family == "centroid":
        target_vector = relative_specific_structure["Structure global centroid"].copy()
        return _translate_biopsy_zslice_list_to_target_vector(threeDdata_zslice_list,
                                                              target_vector)

    if transport_family == "optimal":
        target_vector = _resolve_optimal_target_vector(relative_specific_structure)
        return _translate_biopsy_zslice_list_to_target_vector(threeDdata_zslice_list,
                                                              target_vector)

    raise ValueError(
        "Unsupported simulated biopsy transport family: {}".format(
            transport_family,
        )
    )

def biopsy_transporter_centroid(pydicom_item,
                                specific_structure,
                                threeDdata_zslice_list
                                ):
    return transport_planned_biopsy(pydicom_item,
                                    specific_structure,
                                    threeDdata_zslice_list,
                                    transport_family="centroid")



def biopsy_transporter_optimal(pydicom_item,
                                specific_structure,
                                threeDdata_zslice_list
                                ):
    return transport_planned_biopsy(pydicom_item,
                                    specific_structure,
                                    threeDdata_zslice_list,
                                    transport_family="optimal")