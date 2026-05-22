from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PatientMRADCInputCheckResult:
    patient_uid: str
    has_mr_adc: bool
    selected_series_uid: Any = None
    selected_units: Any = None
    num_mr_adc_series: int = 0
    multiple_series_found: bool = False
    units_match_previous: bool = True


@dataclass(frozen=True)
class MRADCInputCheckResult:
    mr_adc_units: Any
    no_cohort_mr_adc_flag: bool
    patient_results: tuple[PatientMRADCInputCheckResult, ...]


def normalize_patient_mr_adc_input(
        *,
        patient_uid,
        pydicom_item,
        mr_adc_ref,
        previous_mr_adc_units,
        important_info,
        live_display):
    if mr_adc_ref not in pydicom_item:
        important_info.add_text_line("Notice! no ADC MR for: " + str(patient_uid), live_display)
        return PatientMRADCInputCheckResult(
            patient_uid=patient_uid,
            has_mr_adc=False,
        )

    num_mr_adc_series = len(pydicom_item[mr_adc_ref])
    multiple_series_found = num_mr_adc_series > 1
    if multiple_series_found == True:
        important_info.add_text_line(
            "Notice! There are " + str(num_mr_adc_series) + "ADC MRs for: " + str(patient_uid),
            live_display,
        )
        important_info.add_text_line("Removing all MR ADCs except the first for: " + str(patient_uid), live_display)

    series_uid, mr_adc_subdict = next(iter(pydicom_item[mr_adc_ref].items()))
    pydicom_item[mr_adc_ref] = mr_adc_subdict

    selected_units = mr_adc_subdict["Units"]
    units_match_previous = True
    if previous_mr_adc_units is not None and previous_mr_adc_units != selected_units:
        units_match_previous = False
        important_info.add_text_line(
            "The units of your MRs are not the same between patients! Detected on patient: " + str(patient_uid),
            live_display,
        )

    return PatientMRADCInputCheckResult(
        patient_uid=patient_uid,
        has_mr_adc=True,
        selected_series_uid=series_uid,
        selected_units=selected_units,
        num_mr_adc_series=num_mr_adc_series,
        multiple_series_found=multiple_series_found,
        units_match_previous=units_match_previous,
    )


def validate_and_normalize_mr_adc_inputs_for_cohort(
        *,
        master_structure_reference_dict,
        mr_adc_ref,
        important_info,
        live_display):
    mr_adc_units = None
    patient_results = []
    for patient_uid, pydicom_item in master_structure_reference_dict.items():
        patient_result = normalize_patient_mr_adc_input(
            patient_uid=patient_uid,
            pydicom_item=pydicom_item,
            mr_adc_ref=mr_adc_ref,
            previous_mr_adc_units=mr_adc_units,
            important_info=important_info,
            live_display=live_display,
        )
        patient_results.append(patient_result)
        if patient_result.has_mr_adc == True and mr_adc_units is None:
            mr_adc_units = patient_result.selected_units

    no_cohort_mr_adc_flag = all(patient_result.has_mr_adc == False for patient_result in patient_results)
    return MRADCInputCheckResult(
        mr_adc_units=mr_adc_units,
        no_cohort_mr_adc_flag=no_cohort_mr_adc_flag,
        patient_results=tuple(patient_results),
    )