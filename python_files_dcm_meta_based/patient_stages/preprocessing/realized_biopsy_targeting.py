from preprocessing.biopsy_processing.realized_biopsy_targeting import _get_selected_prostate_info
from preprocessing.biopsy_processing.realized_biopsy_targeting import apply_legacy_realized_biopsy_targeting_fields


def determine_patient_realized_biopsy_targeting(*,
                                                patient_uid,
                                                pydicom_item,
                                                all_ref_key,
                                                bx_ref,
                                                oar_ref,
                                                dil_ref,
                                                structures_progress,
                                                processing_structures_task):
    """Determine realized biopsy targeting fields for one patient."""
    selected_prostate_info, prostate_found_bool = _get_selected_prostate_info(
        pydicom_item,
        all_ref_key,
        oar_ref,
    )

    for specific_bx_structure in pydicom_item[bx_ref]:
        structure_id = specific_bx_structure["ROI"]
        processing_structures_task_main_description = "[cyan]Determining biopsy targets [{},{}]...".format(
            patient_uid,
            structure_id,
        )
        structures_progress.update(processing_structures_task, description=processing_structures_task_main_description)

        apply_legacy_realized_biopsy_targeting_fields(
            pydicom_item,
            specific_bx_structure,
            selected_prostate_info,
            prostate_found_bool,
            oar_ref,
            dil_ref,
        )

        structures_progress.update(processing_structures_task, advance=1)
