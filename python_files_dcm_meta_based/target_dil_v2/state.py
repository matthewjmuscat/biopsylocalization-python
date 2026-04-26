MC_PREP_BIOPSY_CYLINDER_LENGTH_KEY = "MC prep biopsy cylinder length"
MC_PREP_BIOPSY_CYLINDER_LENGTH_METHOD_KEY = "MC prep biopsy cylinder length method"
TARGET_DIL_V2_DICT_KEY = "Target DIL v2 dict"
TARGET_DIL_V2_INFO_DICT_KEY = "Target DIL v2 info dict"
TARGET_DIL_V2_SUMMARY_DATAFRAME_KEY = "Target DIL v2 summary dataframe"


def create_default_target_dil_v2_dict():
    return {
        "Enabled": False,
        "Length sidecar source": None,
        "Nominal length mm": None,
        "Target source": None,
        "Target DIL type": None,
        "Target DIL ref #": None,
        "Legacy optimal location ready": False,
        "Legacy optimal location": None,
        "Lane run": False,
        "Notes": None,
    }


def create_default_target_dil_v2_info_dict():
    return {
        "Enabled": False,
        "Length sidecar complete": False,
        "Length sidecar method": None,
        "Legacy-adjacent lane run": False,
        "Num biopsies linked to DIL": 0,
        "Num legacy optimal locations available": 0,
        "Real biopsy length mean mm": None,
        "Real biopsy length std mm": None,
    }


def ensure_target_dil_v2_state(master_structure_reference_dict,
                               bx_ref,
                               all_ref_key
                               ):
    for _, pydicom_item in master_structure_reference_dict.items():
        all_ref_dict = pydicom_item[all_ref_key]

        info_dict = all_ref_dict["Multi-structure information dict (not for csv output)"]
        if info_dict.get(TARGET_DIL_V2_INFO_DICT_KEY) is None:
            info_dict[TARGET_DIL_V2_INFO_DICT_KEY] = create_default_target_dil_v2_info_dict()

        preproc_dict = all_ref_dict["Multi-structure pre-processing output dataframes dict"]
        preproc_dict.setdefault(TARGET_DIL_V2_SUMMARY_DATAFRAME_KEY, None)

        for specific_structure in pydicom_item.get(bx_ref, []):
            specific_structure.setdefault(MC_PREP_BIOPSY_CYLINDER_LENGTH_KEY, None)
            specific_structure.setdefault(MC_PREP_BIOPSY_CYLINDER_LENGTH_METHOD_KEY, None)
            if specific_structure.get(TARGET_DIL_V2_DICT_KEY) is None:
                specific_structure[TARGET_DIL_V2_DICT_KEY] = create_default_target_dil_v2_dict()