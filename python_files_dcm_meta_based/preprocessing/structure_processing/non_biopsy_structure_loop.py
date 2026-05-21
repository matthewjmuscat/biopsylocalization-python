from preprocessing.structure_processing.non_biopsy_structure_processing import preprocess_non_biopsy_structure


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


def process_non_biopsy_structure_families(
        *,
        family_configs,
        master_structure_reference_dict,
        master_structure_info_dict,
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
        runtime_logger):
    for family_config in family_configs:
        live_display = process_non_biopsy_structure_family(
            master_structure_reference_dict=master_structure_reference_dict,
            master_structure_info_dict=master_structure_info_dict,
            struct_ref_type=family_config["struct_ref_type"],
            patient_task_label=family_config["patient_task_label"],
            structs_referenced_dict=structs_referenced_dict,
            config=config,
            parallel_pool=parallel_pool,
            layout_groups=layout_groups,
            patients_progress=patients_progress,
            structures_progress=structures_progress,
            completed_progress=completed_progress,
            indeterminate_progress_sub=indeterminate_progress_sub,
            important_info=important_info,
            live_display=live_display,
            runtime_logger=runtime_logger,
            structure_task_template=family_config.get(
                "structure_task_template",
                "[cyan]Processing structures [{},{}]...",
            ),
            use_master_info_structure_count=family_config.get(
                "use_master_info_structure_count",
                False,
            ),
            pass_selected_structure_dataframe=family_config.get(
                "pass_selected_structure_dataframe",
                False,
            ),
        )
    return live_display


def build_standard_non_biopsy_structure_family_configs(
        *,
        oar_ref,
        rectum_ref_key,
        urethra_ref_key,
        dil_ref):
    return [
        {
            "struct_ref_type": oar_ref,
            "patient_task_label": "Processing patient prostates",
            "structure_task_template": "[cyan]Processing structures [{},{}]...",
        },
        {
            "struct_ref_type": rectum_ref_key,
            "patient_task_label": "Processing patient rectums",
            "structure_task_template": "[cyan]Processing [{},{}]...",
            "use_master_info_structure_count": True,
        },
        {
            "struct_ref_type": urethra_ref_key,
            "patient_task_label": "Processing patient urethras",
            "structure_task_template": "[cyan]Processing [{},{}]...",
            "use_master_info_structure_count": True,
        },
        {
            "struct_ref_type": dil_ref,
            "patient_task_label": "Processing patient DILs",
            "structure_task_template": "[cyan]Processing structures [{},{}]...",
            "pass_selected_structure_dataframe": True,
        },
    ]


def process_standard_non_biopsy_structure_families(
        *,
        oar_ref,
        rectum_ref_key,
        urethra_ref_key,
        dil_ref,
        master_structure_reference_dict,
        master_structure_info_dict,
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
        runtime_logger):
    family_configs = build_standard_non_biopsy_structure_family_configs(
        oar_ref=oar_ref,
        rectum_ref_key=rectum_ref_key,
        urethra_ref_key=urethra_ref_key,
        dil_ref=dil_ref,
    )
    return process_non_biopsy_structure_families(
        family_configs=family_configs,
        master_structure_reference_dict=master_structure_reference_dict,
        master_structure_info_dict=master_structure_info_dict,
        structs_referenced_dict=structs_referenced_dict,
        config=config,
        parallel_pool=parallel_pool,
        layout_groups=layout_groups,
        patients_progress=patients_progress,
        structures_progress=structures_progress,
        completed_progress=completed_progress,
        indeterminate_progress_sub=indeterminate_progress_sub,
        important_info=important_info,
        live_display=live_display,
        runtime_logger=runtime_logger,
    )


def process_standard_non_biopsy_structure_preprocessing_stage(
        *,
        oar_ref,
        rectum_ref_key,
        urethra_ref_key,
        dil_ref,
        master_structure_reference_dict,
        master_structure_info_dict,
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
        runtime_logger):
    """Main-facing stage wrapper for the validated modular non-biopsy path."""
    return process_standard_non_biopsy_structure_families(
        oar_ref=oar_ref,
        rectum_ref_key=rectum_ref_key,
        urethra_ref_key=urethra_ref_key,
        dil_ref=dil_ref,
        master_structure_reference_dict=master_structure_reference_dict,
        master_structure_info_dict=master_structure_info_dict,
        structs_referenced_dict=structs_referenced_dict,
        config=config,
        parallel_pool=parallel_pool,
        layout_groups=layout_groups,
        patients_progress=patients_progress,
        structures_progress=structures_progress,
        completed_progress=completed_progress,
        indeterminate_progress_sub=indeterminate_progress_sub,
        important_info=important_info,
        live_display=live_display,
        runtime_logger=runtime_logger,
    )