import centroid_finder
import numpy as np
import dataframe_dtype_policy


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


def _build_relative_structure_transport_info(relative_specific_structure_index,
                                             relative_specific_structure
                                             ):
    return {
        "Relative structure ID": relative_specific_structure["ROI"],
        "Relative structure type": relative_specific_structure.get("Struct type", relative_specific_structure.get("Relative structure type")),
        "Relative structure ref #": relative_specific_structure["Ref #"],
        "Relative structure index": relative_specific_structure_index,
    }


def _normalize_target_vector(target_vector):
    return np.asarray(target_vector, dtype=float).reshape(3)


def _resolve_optimizer_dataframe_numeric_columns(selection_dataframe):
    numeric_column_names = [
        column_name
        for column_name in dataframe_dtype_policy.OPTIMIZER_V1_LOCATION_NUMERIC_COLUMNS
        if column_name in selection_dataframe.columns
    ]

    if len(numeric_column_names) == 0:
        return selection_dataframe.copy()

    resolved_selection_dataframe = selection_dataframe.copy()
    resolved_selection_dataframe[numeric_column_names] = dataframe_dtype_policy.resolve_numeric_columns(
        selection_dataframe,
        numeric_column_names,
    )

    return resolved_selection_dataframe


def _resolve_optimal_target_selection(relative_specific_structure):
    optimal_locations_dataframe = _resolve_optimizer_dataframe_numeric_columns(
        relative_specific_structure["Biopsy optimization: Optimal biopsy location dataframe"]
    )
    all_tested_locations_dataframe = relative_specific_structure.get("Biopsy optimization: Optimal biopsy location (all tested lattice points) dataframe")
    if all_tested_locations_dataframe is not None:
        all_tested_locations_dataframe = _resolve_optimizer_dataframe_numeric_columns(
            all_tested_locations_dataframe
        )

    retained_candidates_by_min_distance = optimal_locations_dataframe[
        optimal_locations_dataframe['Dist to DIL centroid'] == optimal_locations_dataframe['Dist to DIL centroid'].min()
    ]
    optimal_row = retained_candidates_by_min_distance.iloc[0]

    max_contained_candidate_count = None
    min_distance_candidate_count = int(len(retained_candidates_by_min_distance))
    all_tested_candidate_count = None
    if all_tested_locations_dataframe is not None:
        all_tested_candidate_count = int(len(all_tested_locations_dataframe))
        if all_tested_candidate_count > 0:
            max_contained_value = all_tested_locations_dataframe['Number of normal dist points contained'].max()
            max_contained_mask = all_tested_locations_dataframe['Number of normal dist points contained'] == max_contained_value
            max_contained_candidate_count = int(max_contained_mask.sum())
            min_distance_value = all_tested_locations_dataframe.loc[max_contained_mask, 'Dist to DIL centroid'].min()
            min_distance_candidate_count = int((max_contained_mask & (all_tested_locations_dataframe['Dist to DIL centroid'] == min_distance_value)).sum())

    target_vector = np.array([
        optimal_row['Test location (X)'],
        optimal_row['Test location (Y)'],
        optimal_row['Test location (Z)']
    ])

    selection_metadata = {
        "Optimizer output candidate count": int(len(optimal_locations_dataframe)),
        "All tested candidate count": all_tested_candidate_count,
        "Max containment candidate count": max_contained_candidate_count,
        "Min distance candidate count within max containment set": min_distance_candidate_count,
        "Retained candidate rank": 1,
        "Retained candidate Dist to DIL centroid": float(optimal_row['Dist to DIL centroid']),
        "Retained candidate Number of normal dist points contained": int(optimal_row['Number of normal dist points contained']),
        "Retained candidate Proportion of normal dist points contained": float(optimal_row['Proportion of normal dist points contained']),
        "Retained candidate X": float(target_vector[0]),
        "Retained candidate Y": float(target_vector[1]),
        "Retained candidate Z": float(target_vector[2]),
        "Selection tie-break rule": "max contained -> min Dist to DIL centroid -> first remaining optimizer row",
    }

    return target_vector, selection_metadata


def _resolve_optimal_target_vector(relative_specific_structure):
    target_vector, _ = _resolve_optimal_target_selection(relative_specific_structure)

    return target_vector


def transport_planned_biopsy_with_metadata(pydicom_item,
                                           specific_structure,
                                           threeDdata_zslice_list,
                                           transport_family
                                           ):

    transport_metadata = {
        "Transport family": transport_family,
        "Transport source": None,
        "Relative structure ID": specific_structure.get("Relative structure name"),
        "Relative structure type": specific_structure.get("Relative structure type"),
        "Relative structure ref #": specific_structure.get("Relative structure ref #"),
        "Relative structure index": None,
        "Target X": None,
        "Target Y": None,
        "Target Z": None,
    }

    if transport_family == "identity":
        transport_metadata["Transport source"] = "identity"
        return {
            "Transported raw contour pts zslice list": threeDdata_zslice_list,
            "Simulated biopsy transport dict": transport_metadata,
        }

    relative_specific_structure_index, relative_specific_structure = _find_relative_structure(pydicom_item,
                                                                                               specific_structure)
    transport_metadata.update(
        _build_relative_structure_transport_info(
            relative_specific_structure_index,
            relative_specific_structure,
        )
    )

    if transport_family == "centroid":
        target_vector = _normalize_target_vector(relative_specific_structure["Structure global centroid"].copy())
        transport_metadata["Transport source"] = "relative structure centroid"
        transport_metadata["Target X"] = float(target_vector[0])
        transport_metadata["Target Y"] = float(target_vector[1])
        transport_metadata["Target Z"] = float(target_vector[2])
        return {
            "Transported raw contour pts zslice list": _translate_biopsy_zslice_list_to_target_vector(
                threeDdata_zslice_list,
                target_vector,
            ),
            "Simulated biopsy transport dict": transport_metadata,
        }

    if transport_family == "optimal":
        target_vector, selection_metadata = _resolve_optimal_target_selection(relative_specific_structure)
        transport_metadata.update(selection_metadata)
        transport_metadata["Transport source"] = "optimal biopsy location dataframe"
        transport_metadata["Target X"] = float(target_vector[0])
        transport_metadata["Target Y"] = float(target_vector[1])
        transport_metadata["Target Z"] = float(target_vector[2])
        return {
            "Transported raw contour pts zslice list": _translate_biopsy_zslice_list_to_target_vector(
                threeDdata_zslice_list,
                target_vector,
            ),
            "Simulated biopsy transport dict": transport_metadata,
        }

    raise ValueError(
        "Unsupported simulated biopsy transport family: {}".format(
            transport_family,
        )
    )

def transport_planned_biopsy(pydicom_item,
                             specific_structure,
                             threeDdata_zslice_list,
                             transport_family
                             ):
    return transport_planned_biopsy_with_metadata(
        pydicom_item,
        specific_structure,
        threeDdata_zslice_list,
        transport_family,
    )["Transported raw contour pts zslice list"]

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