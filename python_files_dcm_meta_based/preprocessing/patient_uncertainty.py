"""Headless patient bridge to the unchanged legacy uncertainty calculation.

Runs after real reconstruction and simulated planning. The CSV round trip is
intentional migration compatibility: pandas' default reader can change the last
bits of floats. Direct attachment would be a separate numerical change. This
module owns no file, GUI, cohort, or RNG state.
"""

from io import StringIO
import pandas as pd
from config.uncertainty import UncertaintyPreparationConfig


def prepare_patient_uncertainty_data(*, patient_uid, pydicom_item,
                                     master_structure_info_dict,
                                     structs_referenced_list, structs_referenced_dict,
                                     policy: UncertaintyPreparationConfig):
    """Attach generated parameters and return this patient's resolved table.

    Mutates the structures' legacy ``Uncertainty data`` fields; retains existing
    translation/dilation mm and rotation radians in biopsy/lab frames. Edited
    external spreadsheets need a future explicit, sealed input contract.
    """
    from uncertainty_file_writer import uncertainty_file_preper_by_struct_type_dataframe_NEW
    from preprocessing.uncertainty_attachment import (
        attach_patient_uncertainty_data_from_dataframe, uncertainty_data,
    )

    generated = uncertainty_file_preper_by_struct_type_dataframe_NEW(
        {patient_uid: pydicom_item}, structs_referenced_list, structs_referenced_dict,
        policy.biopsy_variation_uncertainty_setting,
        policy.non_biopsy_variation_uncertainty_setting,
        policy.use_added_in_quad_errors_as, master_structure_info_dict,
    )
    if generated.empty:
        raise ValueError("uncertainty preparation requires patient structures")
    resolved = pd.read_csv(StringIO(generated.to_csv()))
    if not (resolved["Patient UID"] == patient_uid).all():
        raise ValueError("legacy uncertainty CSV round trip changed patient identity")
    count = attach_patient_uncertainty_data_from_dataframe(
        patient_uid=patient_uid, pydicom_item=pydicom_item,
        read_uncertainties_dataframe=resolved, uncertainty_data_cls=uncertainty_data,
    )
    expected = sum(len(pydicom_item[family]) for family in structs_referenced_list)
    if count != expected:
        raise ValueError("uncertainty preparation did not cover every patient structure")
    return resolved
