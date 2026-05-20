import copy

from preprocessing.structure_processing.non_biopsy_structure_processing import preprocess_non_biopsy_structure
from preprocessing.structure_processing.validation import append_non_biopsy_structure_validation_result
from preprocessing.structure_processing.validation import capture_non_biopsy_structure_processing_snapshot
from preprocessing.structure_processing.validation import compare_non_biopsy_structure_processing_snapshots


def capture_non_biopsy_modular_live_state(master_structure_reference_dict,
                                          patient_uid,
                                          struct_ref_type,
                                          specific_structure_index,
                                          all_ref_key):
    return {
        "structure": copy.deepcopy(
            master_structure_reference_dict[patient_uid][struct_ref_type][specific_structure_index]
        ),
        "patient_output_dataframes_dict": copy.deepcopy(
            master_structure_reference_dict[patient_uid][all_ref_key][
                "Multi-structure pre-processing output dataframes dict"
            ]
        ),
    }


def restore_non_biopsy_modular_live_state(master_structure_reference_dict,
                                          patient_uid,
                                          struct_ref_type,
                                          specific_structure_index,
                                          all_ref_key,
                                          saved_state):
    if saved_state is None:
        return

    master_structure_reference_dict[patient_uid][struct_ref_type][specific_structure_index] = (
        saved_state["structure"]
    )
    master_structure_reference_dict[patient_uid][all_ref_key][
        "Multi-structure pre-processing output dataframes dict"
    ] = saved_state["patient_output_dataframes_dict"]


def run_non_biopsy_structure_modular_primary_or_prepare_legacy_validation(
        *,
        validate_non_biopsy_structure_preprocessing_equivalence_bool,
        patient_uid,
        pydicom_item,
        master_structure_reference_dict,
        struct_ref_type,
        specific_structure_index,
        structs_referenced_dict,
        config,
        parallel_pool,
        layout_groups,
        structures_progress,
        processing_structures_task,
        indeterminate_progress_sub,
        important_info,
        live_display,
        runtime_logger,
        sp_patient_selected_structure_info_dataframe=None):
    live_display = preprocess_non_biopsy_structure(
        patient_uid=patient_uid,
        pydicom_item=pydicom_item,
        master_structure_reference_dict=master_structure_reference_dict,
        struct_ref_type=struct_ref_type,
        specific_structure_index=specific_structure_index,
        structs_referenced_dict=structs_referenced_dict,
        config=config,
        parallel_pool=parallel_pool,
        layout_groups=layout_groups,
        structures_progress=structures_progress,
        indeterminate_progress_sub=indeterminate_progress_sub,
        important_info=important_info,
        live_display=live_display,
        runtime_logger=runtime_logger,
        sp_patient_selected_structure_info_dataframe=sp_patient_selected_structure_info_dataframe,
    )

    if validate_non_biopsy_structure_preprocessing_equivalence_bool != True:
        structures_progress.update(processing_structures_task, advance=1)
        return live_display, None, None, True

    modular_validation_snapshot = capture_non_biopsy_structure_processing_snapshot(
        master_structure_reference_dict=master_structure_reference_dict,
        patient_uid=patient_uid,
        struct_ref_type=struct_ref_type,
        specific_structure_index=specific_structure_index,
        all_ref_key=config.all_ref_key,
    )
    modular_live_state = capture_non_biopsy_modular_live_state(
        master_structure_reference_dict=master_structure_reference_dict,
        patient_uid=patient_uid,
        struct_ref_type=struct_ref_type,
        specific_structure_index=specific_structure_index,
        all_ref_key=config.all_ref_key,
    )
    return live_display, modular_validation_snapshot, modular_live_state, False


def finalize_non_biopsy_structure_legacy_validation(
        *,
        master_structure_reference_dict,
        patient_uid,
        struct_ref_type,
        specific_structure_index,
        all_ref_key,
        structure_id,
        modular_validation_snapshot,
        modular_live_state,
        important_info,
        live_display,
        runtime_logger):
    if modular_validation_snapshot is None:
        return live_display

    legacy_validation_snapshot = capture_non_biopsy_structure_processing_snapshot(
        master_structure_reference_dict=master_structure_reference_dict,
        patient_uid=patient_uid,
        struct_ref_type=struct_ref_type,
        specific_structure_index=specific_structure_index,
        all_ref_key=all_ref_key,
    )
    validation_result = compare_non_biopsy_structure_processing_snapshots(
        modular_validation_snapshot,
        legacy_validation_snapshot,
    )
    restore_non_biopsy_modular_live_state(
        master_structure_reference_dict=master_structure_reference_dict,
        patient_uid=patient_uid,
        struct_ref_type=struct_ref_type,
        specific_structure_index=specific_structure_index,
        all_ref_key=all_ref_key,
        saved_state=modular_live_state,
    )
    append_non_biopsy_structure_validation_result(
        master_structure_reference_dict=master_structure_reference_dict,
        patient_uid=patient_uid,
        all_ref_key=all_ref_key,
        validation_result=validation_result,
    )
    if runtime_logger is not None:
        runtime_logger.checkpoint(
            "preprocessing.structure.validation",
            f"Validated modular non-biopsy preprocessing against legacy inline path for {structure_id}.",
            details={
                "patient_uid": patient_uid,
                "structure_id": structure_id,
                "structure_type": struct_ref_type,
                "structure_index": specific_structure_index,
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
            f"WARNING! Modular non-biopsy preprocessing mismatch for patient {patient_uid}, structure {structure_id} ({struct_ref_type}). {mismatch_summary}",
            live_display,
        )
    return live_display