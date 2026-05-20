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


def run_non_biopsy_structure_primary(
        *,
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
    return preprocess_non_biopsy_structure(
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


def prepare_non_biopsy_structure_legacy_validation(
        *,
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
    live_display = run_non_biopsy_structure_primary(
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
        processing_structures_task=processing_structures_task,
        indeterminate_progress_sub=indeterminate_progress_sub,
        important_info=important_info,
        live_display=live_display,
        runtime_logger=runtime_logger,
        sp_patient_selected_structure_info_dataframe=sp_patient_selected_structure_info_dataframe,
    )

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
    return live_display, modular_validation_snapshot, modular_live_state


def process_non_biopsy_structure_family(
        *,
        master_structure_reference_dict,
        master_structure_info_dict,
        struct_ref_type,
        patient_task_label,
        structs_referenced_dict,
        config,
        parallel_pool,
        layout_groups,
        patients_progress,
        structures_progress,
        completed_progress,
        indeterminate_progress_sub,
        important_info,
        live_display,
        runtime_logger,
        structure_task_template="[cyan]Processing structures [{},{}]...",
        use_master_info_structure_count=False,
        pass_selected_structure_dataframe=False):
    patient_uid_default = "Initializing"
    processing_patients_task_main_description = "[red]{} [{}]...".format(
        patient_task_label,
        patient_uid_default,
    )
    processing_patients_task_completed_main_description = "[green]{}".format(patient_task_label)
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
        processing_patients_task_main_description = "[red]{} [{}]...".format(
            patient_task_label,
            patient_uid,
        )
        patients_progress.update(
            processing_patients_task,
            description=processing_patients_task_main_description,
        )

        structure_id_default = "Initializing"
        if use_master_info_structure_count:
            structure_count = master_structure_info_dict["By patient"][patient_uid][struct_ref_type]["Num structs"]
        else:
            structure_count = len(pydicom_item[struct_ref_type])

        processing_structures_task_main_description = structure_task_template.format(
            patient_uid,
            structure_id_default,
        )
        processing_structures_task = structures_progress.add_task(
            processing_structures_task_main_description,
            total=structure_count,
        )

        sp_patient_selected_structure_info_dataframe = None
        if pass_selected_structure_dataframe:
            sp_patient_selected_structure_info_dataframe = pydicom_item[config.all_ref_key][
                "Multi-structure pre-processing output dataframes dict"
            ]["Selected structures"]

        for specific_structure_index, specific_structure in enumerate(pydicom_item[struct_ref_type]):
            structure_id = specific_structure["ROI"]
            processing_structures_task_main_description = structure_task_template.format(
                patient_uid,
                structure_id,
            )
            structures_progress.update(
                processing_structures_task,
                description=processing_structures_task_main_description,
            )
            live_display = run_non_biopsy_structure_primary(
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
                processing_structures_task=processing_structures_task,
                indeterminate_progress_sub=indeterminate_progress_sub,
                important_info=important_info,
                live_display=live_display,
                runtime_logger=runtime_logger,
                sp_patient_selected_structure_info_dataframe=(
                    sp_patient_selected_structure_info_dataframe
                ),
            )
            structures_progress.update(processing_structures_task, advance=1)

        structures_progress.remove_task(processing_structures_task)
        patients_progress.update(processing_patients_task, advance=1)
        completed_progress.update(processing_patients_task_completed, advance=1)
    patients_progress.update(processing_patients_task, visible=False)
    completed_progress.update(processing_patients_task_completed, visible=True)

    return live_display


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