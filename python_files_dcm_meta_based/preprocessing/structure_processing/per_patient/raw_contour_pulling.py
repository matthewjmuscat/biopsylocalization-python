from preprocessing.structure_processing.raw_contour_pulling import pull_raw_structure_contour_for_structure


def pull_raw_structure_contours_for_patient(
    *,
    patient_uid,
    pydicom_item,
    rtstruct_dicom_path,
    structs_referenced_list_generalized,
    bx_ref,
    structures_progress=None,
    pulling_structures_task=None,
):
    """Pull raw contour points for every eligible structure in one patient."""
    for structs in structs_referenced_list_generalized:
        for specific_structure_index, specific_structure in enumerate(pydicom_item[structs]):
            structureID = specific_structure["ROI"]
            structure_reference_number = specific_structure["Ref #"]
            if structs == bx_ref:
                simulated_bool = specific_structure["Simulated bool"]
            else:
                simulated_bool = None

            pulling_structures_task_main_description = "[cyan]Pulling structures [{},{}]...".format(
                patient_uid,
                structureID,
            )
            if structures_progress is not None and pulling_structures_task is not None:
                structures_progress.update(pulling_structures_task, description=pulling_structures_task_main_description)

            if simulated_bool == True:
                if structures_progress is not None and pulling_structures_task is not None:
                    structures_progress.update(pulling_structures_task, advance=1)
                continue

            pull_raw_structure_contour_for_structure(
                pydicom_item,
                rtstruct_dicom_path,
                structs,
                specific_structure_index,
                bx_ref,
            )

            if structures_progress is not None and pulling_structures_task is not None:
                structures_progress.update(pulling_structures_task, advance=1)