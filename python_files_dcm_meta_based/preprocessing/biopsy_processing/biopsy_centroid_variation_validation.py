import numpy as np
import pandas

from preprocessing.biopsy_processing.biopsy_uncertainty_summary import get_biopsy_maximum_projected_distance_value
from preprocessing.biopsy_processing.biopsy_uncertainty_summary import get_biopsy_mean_centroid_variation_value


SIMULATED_BIOPSY_PLANNED_VS_REALIZED_CENTROID_VALIDATION_DF_KEY = "Simulated biopsy planned vs realized centroid variation validation"


def _absolute_delta_or_none(first_value,
                            second_value
                            ):
    if first_value is None or second_value is None:
        return None

    return abs(float(second_value) - float(first_value))


def _mean_or_none(values):
    if not values:
        return None

    return float(np.mean(np.array(values, dtype=float)))


def validate_simulated_biopsy_planned_vs_realized_centroid_variation(master_structure_reference_dict,
                                                                     bx_ref,
                                                                     all_ref_key=None
                                                                     ):
    """Validate simulated biopsy planning values and store patient fragments."""

    validation_rows = []
    mean_variation_absolute_deltas = []
    maximum_projected_distance_absolute_deltas = []
    missing_planned_mean_count = 0
    missing_realized_mean_count = 0
    missing_planned_max_count = 0
    missing_realized_max_count = 0

    for patient_uid, pydicom_item in master_structure_reference_dict.items():
        patient_validation_rows = []
        for specific_bx_structure in pydicom_item[bx_ref]:
            if specific_bx_structure["Simulated bool"] is False:
                continue

            planned_mean_centroid_variation = get_biopsy_mean_centroid_variation_value(
                specific_bx_structure,
                simulated_preference="planned",
            )
            realized_mean_centroid_variation = get_biopsy_mean_centroid_variation_value(
                specific_bx_structure,
                simulated_preference="realized",
            )
            planned_maximum_projected_distance = get_biopsy_maximum_projected_distance_value(
                specific_bx_structure,
                simulated_preference="planned",
            )
            realized_maximum_projected_distance = get_biopsy_maximum_projected_distance_value(
                specific_bx_structure,
                simulated_preference="realized",
            )

            planned_mean_missing = specific_bx_structure.get("Simulated biopsy planning dict", {}).get("Planned mean centroid variation") is None
            realized_mean_missing = specific_bx_structure.get("Mean centroid variation") is None
            planned_max_missing = specific_bx_structure.get("Simulated biopsy planning dict", {}).get("Planned maximum projected distance between original centroids") is None
            realized_max_missing = specific_bx_structure.get("Maximum projected distance between original centroids") is None

            if planned_mean_missing:
                missing_planned_mean_count = missing_planned_mean_count + 1
            if realized_mean_missing:
                missing_realized_mean_count = missing_realized_mean_count + 1
            if planned_max_missing:
                missing_planned_max_count = missing_planned_max_count + 1
            if realized_max_missing:
                missing_realized_max_count = missing_realized_max_count + 1

            mean_variation_absolute_delta = _absolute_delta_or_none(
                planned_mean_centroid_variation,
                realized_mean_centroid_variation,
            )
            maximum_projected_distance_absolute_delta = _absolute_delta_or_none(
                planned_maximum_projected_distance,
                realized_maximum_projected_distance,
            )

            if mean_variation_absolute_delta is not None:
                mean_variation_absolute_deltas.append(mean_variation_absolute_delta)
            if maximum_projected_distance_absolute_delta is not None:
                maximum_projected_distance_absolute_deltas.append(maximum_projected_distance_absolute_delta)

            validation_row = {
                "Patient ID": patient_uid,
                "Bx index": specific_bx_structure["Index number"],
                "Bx ROI": specific_bx_structure["ROI"],
                "Bx ref #": specific_bx_structure["Ref #"],
                "Simulated type": specific_bx_structure["Simulated type"],
                "Planned mean centroid variation": planned_mean_centroid_variation,
                "Realized mean centroid variation": realized_mean_centroid_variation,
                "Mean centroid variation absolute delta": mean_variation_absolute_delta,
                "Planned maximum projected distance between original centroids": planned_maximum_projected_distance,
                "Realized maximum projected distance between original centroids": realized_maximum_projected_distance,
                "Maximum projected distance absolute delta": maximum_projected_distance_absolute_delta,
                "Planned mean centroid variation missing": planned_mean_missing,
                "Realized mean centroid variation missing": realized_mean_missing,
                "Planned max projected distance missing": planned_max_missing,
                "Realized max projected distance missing": realized_max_missing,
            }
            validation_rows.append(validation_row)
            patient_validation_rows.append(validation_row)

        if all_ref_key is not None:
            pydicom_item[all_ref_key]["Multi-structure pre-processing output dataframes dict"][SIMULATED_BIOPSY_PLANNED_VS_REALIZED_CENTROID_VALIDATION_DF_KEY] = pandas.DataFrame(patient_validation_rows)

    validation_dataframe = pandas.DataFrame(validation_rows)
    summary_dict = {
        "Num simulated biopsies": len(validation_rows),
        "Num mean centroid variation comparisons": len(mean_variation_absolute_deltas),
        "Num max projected distance comparisons": len(maximum_projected_distance_absolute_deltas),
        "Mean mean-centroid-variation absolute delta": _mean_or_none(mean_variation_absolute_deltas),
        "Max mean-centroid-variation absolute delta": max(mean_variation_absolute_deltas) if mean_variation_absolute_deltas else None,
        "Mean max-projected-distance absolute delta": _mean_or_none(maximum_projected_distance_absolute_deltas),
        "Max max-projected-distance absolute delta": max(maximum_projected_distance_absolute_deltas) if maximum_projected_distance_absolute_deltas else None,
        "Num missing planned mean centroid variation": missing_planned_mean_count,
        "Num missing realized mean centroid variation": missing_realized_mean_count,
        "Num missing planned maximum projected distance": missing_planned_max_count,
        "Num missing realized maximum projected distance": missing_realized_max_count,
    }

    return validation_dataframe, summary_dict