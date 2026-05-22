import pandas

import misc_tools


SELECTED_STRUCTURES_DF_KEY = "Selected structures"
MULTI_STRUCTURE_PREPROCESSING_OUTPUT_DF_DICT_KEY = "Multi-structure pre-processing output dataframes dict"


def build_patient_selected_structures_dataframe(
        *,
        patient_uid,
        pydicom_item,
        structs_referenced_dict,
        structs_referenced_list_generalized_unique_structs,
        important_info,
        live_display):
    selected_structures_dataframe = pandas.DataFrame()

    for structure_type in structs_referenced_list_generalized_unique_structs:
        structure_type_contour_names_list = structs_referenced_dict[structure_type]["Contour names"]
        selected_structure_info_dataframe, message_string = misc_tools.specific_structure_selector_dataframe_version(
            pydicom_item,
            structure_type,
            structure_type_contour_names_list,
        )
        important_info.add_text_line(message_string, live_display)
        selected_structures_dataframe = pandas.concat(
            [selected_structures_dataframe, selected_structure_info_dataframe],
            ignore_index=True,
        )

    selected_structures_dataframe.insert(loc=0, column="Patient ID", value=patient_uid)
    return selected_structures_dataframe


def store_patient_selected_structures_dataframe(
        *,
        pydicom_item,
        all_ref_key,
        selected_structures_dataframe):
    pydicom_item[all_ref_key][MULTI_STRUCTURE_PREPROCESSING_OUTPUT_DF_DICT_KEY][SELECTED_STRUCTURES_DF_KEY] = (
        selected_structures_dataframe
    )


def prune_patient_unselected_duplicate_structures(
        *,
        patient_uid,
        pydicom_item,
        master_structure_info_dict,
        all_ref_key,
        selected_structures_dataframe):
    more_than_one_struct_found_subset_dataframe = selected_structures_dataframe[
        selected_structures_dataframe["Total num structs found"] > 1
    ]
    num_structs_difference = 0
    for _row_index, row in more_than_one_struct_found_subset_dataframe.iterrows():
        struct_selected_type = row["Struct ref type"]
        struct_selected_index = row["Index number"]

        updated_structure_list = (
            [pydicom_item[struct_selected_type][struct_selected_index]]
            if 0 <= struct_selected_index < len(pydicom_item[struct_selected_type])
            else []
        )
        pydicom_item[struct_selected_type] = updated_structure_list

        current_num_structs = master_structure_info_dict["By patient"][patient_uid][struct_selected_type]["Num structs"]
        updated_num_structs = len(updated_structure_list)
        num_structs_difference += current_num_structs - updated_num_structs
        master_structure_info_dict["By patient"][patient_uid][struct_selected_type]["Num structs"] = updated_num_structs

    current_total_num_structs = master_structure_info_dict["By patient"][patient_uid][all_ref_key]["Total num structs"]
    master_structure_info_dict["By patient"][patient_uid][all_ref_key]["Total num structs"] = (
        current_total_num_structs - num_structs_difference
    )


def update_global_structure_count(
        *,
        master_structure_reference_dict,
        master_structure_info_dict,
        structs_referenced_list_generalized):
    total_num_structs_updated = 0
    for _patient_uid, pydicom_item in master_structure_reference_dict.items():
        for structure_type in structs_referenced_list_generalized:
            total_num_structs_updated += len(pydicom_item[structure_type])

    master_structure_info_dict["Global"]["Num structures"] = total_num_structs_updated


def select_patient_unique_structures(
        *,
        patient_uid,
        pydicom_item,
        master_structure_reference_dict,
        master_structure_info_dict,
        structs_referenced_dict,
        structs_referenced_list_generalized,
        structs_referenced_list_generalized_unique_structs,
        all_ref_key,
        important_info,
        live_display):
    selected_structures_dataframe = build_patient_selected_structures_dataframe(
        patient_uid=patient_uid,
        pydicom_item=pydicom_item,
        structs_referenced_dict=structs_referenced_dict,
        structs_referenced_list_generalized_unique_structs=structs_referenced_list_generalized_unique_structs,
        important_info=important_info,
        live_display=live_display,
    )
    store_patient_selected_structures_dataframe(
        pydicom_item=pydicom_item,
        all_ref_key=all_ref_key,
        selected_structures_dataframe=selected_structures_dataframe,
    )
    prune_patient_unselected_duplicate_structures(
        patient_uid=patient_uid,
        pydicom_item=pydicom_item,
        master_structure_info_dict=master_structure_info_dict,
        all_ref_key=all_ref_key,
        selected_structures_dataframe=selected_structures_dataframe,
    )
    update_global_structure_count(
        master_structure_reference_dict=master_structure_reference_dict,
        master_structure_info_dict=master_structure_info_dict,
        structs_referenced_list_generalized=structs_referenced_list_generalized,
    )
    return selected_structures_dataframe


def select_unique_structures_for_cohort(
        *,
        master_structure_reference_dict,
        master_structure_info_dict,
        structs_referenced_dict,
        structs_referenced_list_generalized,
        structs_referenced_list_generalized_unique_structs,
        all_ref_key,
        patients_progress,
        completed_progress,
        important_info,
        live_display):
    patient_uid_default = "Initializing"
    processing_patients_task_main_description = "[red]Selecting unique structures [{}]...".format(
        patient_uid_default,
    )
    processing_patients_task_completed_main_description = "[green]Selecting unique structures"
    processing_patients_task = patients_progress.add_task(
        processing_patients_task_main_description,
        total=master_structure_info_dict["Global"]["Num cases"],
    )
    processing_patients_task_completed = completed_progress.add_task(
        processing_patients_task_completed_main_description,
        total=master_structure_info_dict["Global"]["Num cases"],
        visible=False,
    )

    for patient_uid, pydicom_item in master_structure_reference_dict.items():
        processing_patients_task_main_description = "[red]Selecting unique structures [{}]...".format(patient_uid)
        patients_progress.update(processing_patients_task, description=processing_patients_task_main_description)

        select_patient_unique_structures(
            patient_uid=patient_uid,
            pydicom_item=pydicom_item,
            master_structure_reference_dict=master_structure_reference_dict,
            master_structure_info_dict=master_structure_info_dict,
            structs_referenced_dict=structs_referenced_dict,
            structs_referenced_list_generalized=structs_referenced_list_generalized,
            structs_referenced_list_generalized_unique_structs=structs_referenced_list_generalized_unique_structs,
            all_ref_key=all_ref_key,
            important_info=important_info,
            live_display=live_display,
        )

        patients_progress.update(processing_patients_task, advance=1)
        completed_progress.update(processing_patients_task_completed, advance=1)

    patients_progress.update(processing_patients_task, visible=False)
    completed_progress.update(processing_patients_task_completed, visible=True)