import pandas

from preprocessing.biopsy_processing.biopsy_double_sextant import PER_SAMPLE_POINT_DOUBLE_SEXTANT_DF_KEY
from preprocessing.biopsy_processing.biopsy_double_sextant import PER_VOXEL_DOUBLE_SEXTANT_DF_KEY
from preprocessing.biopsy_processing.biopsy_double_sextant import _build_patient_per_sample_point_double_sextant_dataframe
from preprocessing.biopsy_processing.biopsy_double_sextant import _build_per_voxel_double_sextant_dataframe


def build_patient_biopsy_double_sextant_sample_point_fragment(*,
                                                              patient_uid,
                                                              pydicom_item,
                                                              all_ref_key,
                                                              bx_ref,
                                                              oar_ref,
                                                              biopsy_z_voxel_length):
    """Build and store one patient's per-sample-point double-sextant fragment."""
    patient_sample_point_dataframe = _build_patient_per_sample_point_double_sextant_dataframe(
        patient_uid,
        pydicom_item,
        all_ref_key,
        bx_ref,
        oar_ref,
        biopsy_z_voxel_length,
    )
    pydicom_item[all_ref_key]["Multi-structure pre-processing output dataframes dict"][
        PER_SAMPLE_POINT_DOUBLE_SEXTANT_DF_KEY
    ] = patient_sample_point_dataframe
    return patient_sample_point_dataframe


def assemble_biopsy_double_sextant_classification_fragments(patient_sample_point_dataframes):
    """Assemble patient double-sextant fragments using the legacy voxel aggregation path."""
    if patient_sample_point_dataframes:
        per_sample_point_dataframe = pandas.concat(patient_sample_point_dataframes, ignore_index=True)
    else:
        per_sample_point_dataframe = pandas.DataFrame()

    per_voxel_dataframe = _build_per_voxel_double_sextant_dataframe(per_sample_point_dataframe)
    return per_sample_point_dataframe, per_voxel_dataframe


def store_patient_biopsy_double_sextant_voxel_fragment(*,
                                                       patient_uid,
                                                       pydicom_item,
                                                       all_ref_key,
                                                       per_voxel_dataframe):
    """Store one patient's per-voxel double-sextant fragment after run-level aggregation."""
    if "Patient ID" in per_voxel_dataframe.columns:
        patient_voxel_dataframe = per_voxel_dataframe[
            per_voxel_dataframe["Patient ID"].eq(patient_uid)
        ].reset_index(drop=True)
    else:
        patient_voxel_dataframe = pandas.DataFrame()

    pydicom_item[all_ref_key]["Multi-structure pre-processing output dataframes dict"][
        PER_VOXEL_DOUBLE_SEXTANT_DF_KEY
    ] = patient_voxel_dataframe
    return patient_voxel_dataframe