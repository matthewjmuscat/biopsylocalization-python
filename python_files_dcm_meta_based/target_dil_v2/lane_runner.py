import pandas

from target_dil_v2.state import MC_PREP_BIOPSY_CYLINDER_LENGTH_KEY
from target_dil_v2.state import TARGET_DIL_V2_DICT_KEY
from target_dil_v2.state import TARGET_DIL_V2_INFO_DICT_KEY
from target_dil_v2.state import TARGET_DIL_V2_SUMMARY_DATAFRAME_KEY


TARGET_DIL_V2_SUMMARY_COLUMNS = [
    "Patient ID",
    "Bx index",
    "Bx ROI",
    "Simulated type",
    "Relative structure type",
    "Relative structure ref #",
    "MC prep biopsy cylinder length",
    "Legacy optimal location ready",
    "Legacy optimal location (X)",
    "Legacy optimal location (Y)",
    "Legacy optimal location (Z)",
]


def _find_relative_structure(pydicom_item,
                             relative_structure_type,
                             relative_structure_refnum
                             ):
    if relative_structure_type not in pydicom_item:
        return None

    for specific_structure in pydicom_item[relative_structure_type]:
        if specific_structure["Ref #"] == relative_structure_refnum:
            return specific_structure

    return None


def _extract_legacy_optimal_location(relative_structure):
    if relative_structure is None:
        return False, (None, None, None)

    optimal_locations_dataframe = relative_structure["Biopsy optimization: Optimal biopsy location dataframe"]
    if optimal_locations_dataframe is None or len(optimal_locations_dataframe.index) == 0:
        return False, (None, None, None)

    if "Dist to DIL centroid" in optimal_locations_dataframe.columns:
        selected_row = optimal_locations_dataframe.loc[
            optimal_locations_dataframe["Dist to DIL centroid"].idxmin()
        ]
    else:
        selected_row = optimal_locations_dataframe.iloc[0]

    selected_location = (
        float(selected_row["Test location (X)"]),
        float(selected_row["Test location (Y)"]),
        float(selected_row["Test location (Z)"]),
    )
    return True, selected_location


def run_first_pass_target_dil_v2_lane(master_structure_reference_dict,
                                      bx_ref,
                                      dil_ref,
                                      all_ref_key,
                                      enable_target_dil_v2
                                      ):
    for patientUID, pydicom_item in master_structure_reference_dict.items():
        patient_rows = []
        num_biopsies_linked_to_dil = 0
        num_legacy_optimal_locations_available = 0

        for specific_structure in pydicom_item.get(bx_ref, []):
            relative_structure_type = specific_structure.get("Relative structure type")
            relative_structure_refnum = specific_structure.get("Relative structure ref #")

            if specific_structure["Simulated bool"] == False or relative_structure_type != dil_ref or relative_structure_refnum is None:
                continue

            num_biopsies_linked_to_dil = num_biopsies_linked_to_dil + 1

            relative_structure = _find_relative_structure(pydicom_item,
                                                          relative_structure_type,
                                                          relative_structure_refnum)
            legacy_optimal_location_ready, legacy_optimal_location = _extract_legacy_optimal_location(relative_structure)
            if legacy_optimal_location_ready == True:
                num_legacy_optimal_locations_available = num_legacy_optimal_locations_available + 1

            target_dil_v2_dict = specific_structure[TARGET_DIL_V2_DICT_KEY]
            target_dil_v2_dict["Enabled"] = enable_target_dil_v2
            target_dil_v2_dict["Target source"] = "Relative structure"
            target_dil_v2_dict["Target DIL type"] = relative_structure_type
            target_dil_v2_dict["Target DIL ref #"] = relative_structure_refnum
            target_dil_v2_dict["Legacy optimal location ready"] = legacy_optimal_location_ready
            target_dil_v2_dict["Legacy optimal location"] = legacy_optimal_location
            target_dil_v2_dict["Lane run"] = enable_target_dil_v2
            target_dil_v2_dict["Notes"] = "First pass sidecar only; no target-specific optimizer math has been run yet."

            patient_rows.append({
                "Patient ID": patientUID,
                "Bx index": specific_structure["Index number"],
                "Bx ROI": specific_structure["ROI"],
                "Simulated type": specific_structure["Simulated type"],
                "Relative structure type": relative_structure_type,
                "Relative structure ref #": relative_structure_refnum,
                "MC prep biopsy cylinder length": specific_structure[MC_PREP_BIOPSY_CYLINDER_LENGTH_KEY],
                "Legacy optimal location ready": legacy_optimal_location_ready,
                "Legacy optimal location (X)": legacy_optimal_location[0],
                "Legacy optimal location (Y)": legacy_optimal_location[1],
                "Legacy optimal location (Z)": legacy_optimal_location[2],
            })

        patient_summary_dataframe = pandas.DataFrame(patient_rows,
                                                     columns=TARGET_DIL_V2_SUMMARY_COLUMNS)
        pydicom_item[all_ref_key]["Multi-structure pre-processing output dataframes dict"][TARGET_DIL_V2_SUMMARY_DATAFRAME_KEY] = patient_summary_dataframe

        patient_v2_info_dict = pydicom_item[all_ref_key]["Multi-structure information dict (not for csv output)"][TARGET_DIL_V2_INFO_DICT_KEY]
        patient_v2_info_dict["Enabled"] = enable_target_dil_v2
        patient_v2_info_dict["Legacy-adjacent lane run"] = enable_target_dil_v2
        patient_v2_info_dict["Num biopsies linked to DIL"] = num_biopsies_linked_to_dil
        patient_v2_info_dict["Num legacy optimal locations available"] = num_legacy_optimal_locations_available