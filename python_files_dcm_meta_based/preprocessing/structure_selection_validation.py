import copy

import pandas
from pandas.testing import assert_frame_equal

import misc_tools


SELECTED_STRUCTURES_DF_KEY = "Selected structures"
SELECTED_STRUCTURES_VALIDATION_DF_KEY = "Selected structures validation"
MULTI_STRUCTURE_PREPROCESSING_OUTPUT_DF_DICT_KEY = "Multi-structure pre-processing output dataframes dict"


def capture_selected_structures_live_state(
        *,
        master_structure_reference_dict,
        master_structure_info_dict,
        structs_referenced_list_generalized,
        all_ref_key):
    patient_states = {}
    for patient_uid, pydicom_item in master_structure_reference_dict.items():
        patient_states[patient_uid] = {
            "structures": {
                structure_type: copy.deepcopy(pydicom_item[structure_type])
                for structure_type in structs_referenced_list_generalized
            },
            "patient_output_dataframes_dict": copy.deepcopy(
                pydicom_item[all_ref_key][MULTI_STRUCTURE_PREPROCESSING_OUTPUT_DF_DICT_KEY]
            ),
        }
    return {
        "patient_states": patient_states,
        "by_patient_info": copy.deepcopy(master_structure_info_dict["By patient"]),
        "global_num_structures": copy.deepcopy(master_structure_info_dict["Global"]["Num structures"]),
    }


def restore_selected_structures_live_state(
        *,
        master_structure_reference_dict,
        master_structure_info_dict,
        all_ref_key,
        saved_state):
    if saved_state is None:
        return

    for patient_uid, patient_state in saved_state["patient_states"].items():
        for structure_type, structure_state in patient_state["structures"].items():
            master_structure_reference_dict[patient_uid][structure_type] = structure_state
        master_structure_reference_dict[patient_uid][all_ref_key][MULTI_STRUCTURE_PREPROCESSING_OUTPUT_DF_DICT_KEY] = (
            patient_state["patient_output_dataframes_dict"]
        )
    master_structure_info_dict["By patient"] = saved_state["by_patient_info"]
    master_structure_info_dict["Global"]["Num structures"] = saved_state["global_num_structures"]


def run_selected_structures_legacy_stage(
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
    patientUID_default = "Initializing"
    processing_patients_task_main_description = "[red]Selecting unique structures [{}]...".format(patientUID_default)
    processing_patients_task_completed_main_description = "[green]Selecting unique structures"
    processing_patients_task = patients_progress.add_task(processing_patients_task_main_description, total=master_structure_info_dict["Global"]["Num cases"])
    processing_patients_task_completed = completed_progress.add_task(processing_patients_task_completed_main_description, total=master_structure_info_dict["Global"]["Num cases"], visible=False)

    for patientUID, pydicom_item in master_structure_reference_dict.items():
        processing_patients_task_main_description = "[red]Selecting unique structures [{}]...".format(patientUID)
        patients_progress.update(processing_patients_task, description=processing_patients_task_main_description)

        sp_patient_selected_structure_info_dataframe = pandas.DataFrame()

        for structure_type in structs_referenced_list_generalized_unique_structs:
            structure_type_contour_names_list = structs_referenced_dict[structure_type]["Contour names"]

            selected_structure_info_dataframe, message_string = misc_tools.specific_structure_selector_dataframe_version(
                pydicom_item,
                structure_type,
                structure_type_contour_names_list,
            )

            important_info.add_text_line(message_string, live_display)

            sp_patient_selected_structure_info_dataframe = pandas.concat(
                [sp_patient_selected_structure_info_dataframe, selected_structure_info_dataframe],
                ignore_index=True,
            )

        sp_patient_selected_structure_info_dataframe.insert(loc=0, column="Patient ID", value=patientUID)

        pydicom_item[all_ref_key][MULTI_STRUCTURE_PREPROCESSING_OUTPUT_DF_DICT_KEY][SELECTED_STRUCTURES_DF_KEY] = (
            sp_patient_selected_structure_info_dataframe
        )

        sp_patient_selected_structure_info_dataframe_more_than_one_struct_found_subset_dataframe = (
            sp_patient_selected_structure_info_dataframe[
                sp_patient_selected_structure_info_dataframe["Total num structs found"] > 1
            ]
        )
        num_structs_difference = 0
        for _row_index, row in sp_patient_selected_structure_info_dataframe_more_than_one_struct_found_subset_dataframe.iterrows():
            struct_selected_type = row["Struct ref type"]
            struct_selected_index = row["Index number"]

            updated_sp_structure_list = (
                [pydicom_item[struct_selected_type][struct_selected_index]]
                if 0 <= struct_selected_index < len(pydicom_item[struct_selected_type])
                else []
            )

            pydicom_item[struct_selected_type] = updated_sp_structure_list

            current_num_structs = master_structure_info_dict["By patient"][patientUID][struct_selected_type]["Num structs"]
            updated_num_structs = len(updated_sp_structure_list)
            difference = current_num_structs - updated_num_structs
            num_structs_difference += difference

            master_structure_info_dict["By patient"][patientUID][struct_selected_type]["Num structs"] = updated_num_structs

        current_total_num_structs = master_structure_info_dict["By patient"][patientUID][all_ref_key]["Total num structs"]
        master_structure_info_dict["By patient"][patientUID][all_ref_key]["Total num structs"] = (
            current_total_num_structs - num_structs_difference
        )

        total_num_structs_updated = 0
        for _patientUID, pydicom_item_for_count in master_structure_reference_dict.items():
            for structure_type in structs_referenced_list_generalized:
                num_structs = len(pydicom_item_for_count[structure_type])
                total_num_structs_updated += num_structs

        master_structure_info_dict["Global"]["Num structures"] = total_num_structs_updated

        patients_progress.update(processing_patients_task, advance=1)
        completed_progress.update(processing_patients_task_completed, advance=1)
    patients_progress.update(processing_patients_task, visible=False)
    completed_progress.update(processing_patients_task_completed, visible=True)


def capture_selected_structures_validation_snapshot(
        *,
        master_structure_reference_dict,
        master_structure_info_dict,
        structs_referenced_list_generalized,
        all_ref_key):
    patient_snapshots = {}
    for patient_uid, pydicom_item in master_structure_reference_dict.items():
        patient_output_dataframes_dict = pydicom_item[all_ref_key][MULTI_STRUCTURE_PREPROCESSING_OUTPUT_DF_DICT_KEY]
        patient_snapshots[patient_uid] = {
            "selected_structures_dataframe": copy.deepcopy(
                patient_output_dataframes_dict.get(SELECTED_STRUCTURES_DF_KEY)
            ),
            "structure_lengths": {
                structure_type: len(pydicom_item[structure_type])
                for structure_type in structs_referenced_list_generalized
            },
            "selected_structure_identity": {
                structure_type: [
                    (
                        specific_structure.get("ROI"),
                        specific_structure.get("Ref #"),
                    )
                    for specific_structure in pydicom_item[structure_type]
                ]
                for structure_type in structs_referenced_list_generalized
            },
            "patient_structure_info": copy.deepcopy(master_structure_info_dict["By patient"][patient_uid]),
        }
    return {
        "patients": patient_snapshots,
        "global_num_structures": copy.deepcopy(master_structure_info_dict["Global"]["Num structures"]),
    }


def _compare_dataframes(path, modular_dataframe, legacy_dataframe, mismatches):
    if modular_dataframe is None or legacy_dataframe is None:
        if modular_dataframe is not legacy_dataframe:
            mismatches.append(f"{path}: dataframe presence mismatch")
        return
    try:
        assert_frame_equal(
            modular_dataframe,
            legacy_dataframe,
            check_dtype=True,
            check_like=False,
        )
    except AssertionError as exc:
        mismatches.append(f"{path}: {str(exc).splitlines()[0]}")


def compare_selected_structures_validation_snapshots(modular_snapshot, legacy_snapshot):
    results = []
    if modular_snapshot["global_num_structures"] != legacy_snapshot["global_num_structures"]:
        global_mismatch = (
            "global_num_structures: modular={} legacy={}".format(
                modular_snapshot["global_num_structures"],
                legacy_snapshot["global_num_structures"],
            )
        )
    else:
        global_mismatch = ""

    patient_uids = sorted(
        set(modular_snapshot["patients"].keys()).union(legacy_snapshot["patients"].keys())
    )
    for patient_uid in patient_uids:
        mismatches = []
        if global_mismatch:
            mismatches.append(global_mismatch)
        modular_patient_snapshot = modular_snapshot["patients"].get(patient_uid)
        legacy_patient_snapshot = legacy_snapshot["patients"].get(patient_uid)
        if modular_patient_snapshot is None or legacy_patient_snapshot is None:
            mismatches.append("patient presence mismatch")
        else:
            _compare_dataframes(
                "selected_structures_dataframe",
                modular_patient_snapshot["selected_structures_dataframe"],
                legacy_patient_snapshot["selected_structures_dataframe"],
                mismatches,
            )
            for snapshot_field in (
                    "structure_lengths",
                    "selected_structure_identity",
                    "patient_structure_info"):
                if modular_patient_snapshot[snapshot_field] != legacy_patient_snapshot[snapshot_field]:
                    mismatches.append(f"{snapshot_field}: modular and legacy values differ")
        results.append(
            {
                "patient_uid": patient_uid,
                "overall_match_bool": len(mismatches) == 0,
                "mismatch_count": len(mismatches),
                "mismatch_fields": [mismatch.split(":", 1)[0] for mismatch in mismatches],
                "mismatch_summary": " | ".join(mismatches[:10]),
            }
        )
    return results


def append_selected_structures_validation_result(
        *,
        master_structure_reference_dict,
        patient_uid,
        all_ref_key,
        validation_result):
    patient_output_dataframes_dict = master_structure_reference_dict[patient_uid][all_ref_key][
        MULTI_STRUCTURE_PREPROCESSING_OUTPUT_DF_DICT_KEY
    ]
    validation_result_row = pandas.DataFrame(
        {
            "Patient ID": [validation_result["patient_uid"]],
            "Overall match bool": [validation_result["overall_match_bool"]],
            "Mismatch count": [validation_result["mismatch_count"]],
            "Mismatch fields": ["; ".join(validation_result["mismatch_fields"])],
            "Mismatch summary": [validation_result["mismatch_summary"]],
        }
    )
    existing_validation_dataframe = patient_output_dataframes_dict.get(SELECTED_STRUCTURES_VALIDATION_DF_KEY)
    if existing_validation_dataframe is None:
        patient_output_dataframes_dict[SELECTED_STRUCTURES_VALIDATION_DF_KEY] = validation_result_row
        return

    patient_output_dataframes_dict[SELECTED_STRUCTURES_VALIDATION_DF_KEY] = pandas.concat(
        [existing_validation_dataframe, validation_result_row],
        ignore_index=True,
    )


def begin_selected_structures_legacy_validation(
        *,
        master_structure_reference_dict,
        master_structure_info_dict,
        structs_referenced_list_generalized,
        all_ref_key):
    return {
        "pre_modular_live_state": capture_selected_structures_live_state(
            master_structure_reference_dict=master_structure_reference_dict,
            master_structure_info_dict=master_structure_info_dict,
            structs_referenced_list_generalized=structs_referenced_list_generalized,
            all_ref_key=all_ref_key,
        )
    }


def finalize_selected_structures_legacy_validation(
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
        live_display,
        validation_context,
        runtime_logger=None):
    if validation_context is None:
        return live_display

    modular_validation_snapshot = capture_selected_structures_validation_snapshot(
        master_structure_reference_dict=master_structure_reference_dict,
        master_structure_info_dict=master_structure_info_dict,
        structs_referenced_list_generalized=structs_referenced_list_generalized,
        all_ref_key=all_ref_key,
    )
    modular_live_state = capture_selected_structures_live_state(
        master_structure_reference_dict=master_structure_reference_dict,
        master_structure_info_dict=master_structure_info_dict,
        structs_referenced_list_generalized=structs_referenced_list_generalized,
        all_ref_key=all_ref_key,
    )
    restore_selected_structures_live_state(
        master_structure_reference_dict=master_structure_reference_dict,
        master_structure_info_dict=master_structure_info_dict,
        all_ref_key=all_ref_key,
        saved_state=validation_context["pre_modular_live_state"],
    )
    if runtime_logger is not None:
        runtime_logger.checkpoint(
            "preprocessing.structure_selection.validation.prepare",
            "Captured modular selected-structures output and restored pre-stage state for legacy validation.",
        )

    run_selected_structures_legacy_stage(
        master_structure_reference_dict=master_structure_reference_dict,
        master_structure_info_dict=master_structure_info_dict,
        structs_referenced_dict=structs_referenced_dict,
        structs_referenced_list_generalized=structs_referenced_list_generalized,
        structs_referenced_list_generalized_unique_structs=structs_referenced_list_generalized_unique_structs,
        all_ref_key=all_ref_key,
        patients_progress=patients_progress,
        completed_progress=completed_progress,
        important_info=important_info,
        live_display=live_display,
    )
    legacy_validation_snapshot = capture_selected_structures_validation_snapshot(
        master_structure_reference_dict=master_structure_reference_dict,
        master_structure_info_dict=master_structure_info_dict,
        structs_referenced_list_generalized=structs_referenced_list_generalized,
        all_ref_key=all_ref_key,
    )
    validation_results = compare_selected_structures_validation_snapshots(
        modular_validation_snapshot,
        legacy_validation_snapshot,
    )

    restore_selected_structures_live_state(
        master_structure_reference_dict=master_structure_reference_dict,
        master_structure_info_dict=master_structure_info_dict,
        all_ref_key=all_ref_key,
        saved_state=modular_live_state,
    )

    for validation_result in validation_results:
        append_selected_structures_validation_result(
            master_structure_reference_dict=master_structure_reference_dict,
            patient_uid=validation_result["patient_uid"],
            all_ref_key=all_ref_key,
            validation_result=validation_result,
        )
        if runtime_logger is not None:
            runtime_logger.checkpoint(
                "preprocessing.structure_selection.validation",
                "Validated modular selected-structures stage against legacy inline path.",
                details={
                    "patient_uid": validation_result["patient_uid"],
                    "overall_match_bool": validation_result["overall_match_bool"],
                    "mismatch_count": validation_result["mismatch_count"],
                    "mismatch_fields": validation_result["mismatch_fields"][:10],
                },
            )
        if validation_result["overall_match_bool"] == False:
            mismatch_summary = validation_result["mismatch_summary"]
            if len(mismatch_summary) > 400:
                mismatch_summary = mismatch_summary[:400] + "..."
            important_info.add_text_line(
                f"WARNING! Modular selected-structures mismatch for patient {validation_result['patient_uid']}. {mismatch_summary}",
                live_display,
            )
    return live_display