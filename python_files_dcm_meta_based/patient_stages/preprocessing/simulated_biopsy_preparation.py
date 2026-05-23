import numpy as np

from preprocessing.biopsy_processing.simulated_biopsy_preparation import _find_nearest_dil_refnum
from preprocessing.biopsy_processing.simulated_biopsy_preparation import _find_structure_info_from_refnum
from preprocessing.biopsy_processing.simulated_biopsy_preparation import _set_target_information


def assign_patient_simulated_biopsy_targets(pydicom_item,
                                            bx_ref,
                                            dil_ref):
    dil_centroids_by_ref = {}
    if dil_ref in pydicom_item:
        for specific_dil_structure in pydicom_item[dil_ref]:
            dil_refnum = specific_dil_structure["Ref #"]
            dil_centroid = np.array(specific_dil_structure["Structure global centroid"]).reshape(3)
            dil_centroids_by_ref[dil_refnum] = dil_centroid

    for specific_structure in pydicom_item[bx_ref]:
        if specific_structure["Simulated bool"] == False:
            target_determined = False
            target_source = None
            target_structure_type = None
            target_structure_refnum = None
            target_structure_index = None
            target_structure_id = None

            if dil_centroids_by_ref and specific_structure.get("Structure global centroid") is not None:
                bx_centroid = np.array(specific_structure["Structure global centroid"]).reshape(3)
                target_structure_refnum = _find_nearest_dil_refnum(
                    bx_centroid,
                    dil_centroids_by_ref,
                )
                if target_structure_refnum is not None:
                    target_structure_type = dil_ref
                    target_structure_index, target_structure = _find_structure_info_from_refnum(
                        pydicom_item,
                        dil_ref,
                        target_structure_refnum,
                    )
                    if target_structure is not None:
                        target_determined = True
                        target_source = "Nearest DIL by centroid"
                        target_structure_id = target_structure["ROI"]

            _set_target_information(
                specific_structure,
                target_determined,
                target_source,
                target_structure_type,
                target_structure_refnum,
                target_structure_index,
                target_structure_id,
            )
            continue

        relative_structure_type = specific_structure.get("Relative structure type")
        relative_structure_refnum = specific_structure.get("Relative structure ref #")
        target_determined = False
        target_source = None
        target_structure_index = None
        target_structure_id = None

        if relative_structure_type == dil_ref and relative_structure_refnum is not None:
            target_structure_index, target_structure = _find_structure_info_from_refnum(
                pydicom_item,
                relative_structure_type,
                relative_structure_refnum,
            )
            if target_structure is not None:
                target_determined = True
                target_source = "Relative structure"
                target_structure_id = target_structure["ROI"]

        _set_target_information(
            specific_structure,
            target_determined,
            target_source,
            relative_structure_type,
            relative_structure_refnum,
            target_structure_index,
            target_structure_id,
        )
