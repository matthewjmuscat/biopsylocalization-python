import numpy as np
import pandas

from preprocessing.biopsy_processing.biopsy_centroid_variation_validation import SIMULATED_BIOPSY_PLANNED_VS_REALIZED_CENTROID_VALIDATION_DF_KEY
from preprocessing.biopsy_processing.biopsy_uncertainty_summary import get_biopsy_maximum_projected_distance_value
from preprocessing.biopsy_processing.biopsy_uncertainty_summary import get_biopsy_mean_centroid_variation_value


def _absolute_delta_or_none(first_value,
                            second_value):
    if first_value is None or second_value is None:
        return None

    return abs(float(second_value) - float(first_value))


def _mean_or_none(values):
    if not values:
        return None

    return float(np.mean(np.array(values, dtype=float)))


def build_patient_simulated_biopsy_centroid_variation_validation_fragment(*,
                                                                         patient_uid,
                                                                         pydicom_item,
                                                                         bx_ref,
                                                                         all_ref_key=None):
    """Build one patient's planned-vs-realized simulated-biopsy validation fragment."""
    validation_rows = []

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

        mean_variation_absolute_delta = _absolute_delta_or_none(
            planned_mean_centroid_variation,
            realized_mean_centroid_variation,
        )
        maximum_projected_distance_absolute_delta = _absolute_delta_or_none(
            planned_maximum_projected_distance,
            realized_maximum_projected_distance,
        )

        validation_rows.append({
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
        })

    patient_validation_dataframe = pandas.DataFrame(validation_rows)
    if all_ref_key is not None:
        pydicom_item[all_ref_key]["Multi-structure pre-processing output dataframes dict"][
            SIMULATED_BIOPSY_PLANNED_VS_REALIZED_CENTROID_VALIDATION_DF_KEY
        ] = patient_validation_dataframe

    return patient_validation_dataframe


def summarize_simulated_biopsy_centroid_variation_validation_dataframe(validation_dataframe):
    """Summarize a planned-vs-realized validation dataframe assembled from patient fragments."""
    if validation_dataframe.empty:
        return {
            "Num simulated biopsies": 0,
            "Num mean centroid variation comparisons": 0,
            "Num max projected distance comparisons": 0,
            "Mean mean-centroid-variation absolute delta": None,
            "Max mean-centroid-variation absolute delta": None,
            "Mean max-projected-distance absolute delta": None,
            "Max max-projected-distance absolute delta": None,
            "Num missing planned mean centroid variation": 0,
            "Num missing realized mean centroid variation": 0,
            "Num missing planned maximum projected distance": 0,
            "Num missing realized maximum projected distance": 0,
        }

    mean_variation_absolute_deltas = validation_dataframe["Mean centroid variation absolute delta"].dropna().tolist()
    maximum_projected_distance_absolute_deltas = validation_dataframe["Maximum projected distance absolute delta"].dropna().tolist()

    return {
        "Num simulated biopsies": len(validation_dataframe),
        "Num mean centroid variation comparisons": len(mean_variation_absolute_deltas),
        "Num max projected distance comparisons": len(maximum_projected_distance_absolute_deltas),
        "Mean mean-centroid-variation absolute delta": _mean_or_none(mean_variation_absolute_deltas),
        "Max mean-centroid-variation absolute delta": max(mean_variation_absolute_deltas) if mean_variation_absolute_deltas else None,
        "Mean max-projected-distance absolute delta": _mean_or_none(maximum_projected_distance_absolute_deltas),
        "Max max-projected-distance absolute delta": max(maximum_projected_distance_absolute_deltas) if maximum_projected_distance_absolute_deltas else None,
        "Num missing planned mean centroid variation": int(validation_dataframe["Planned mean centroid variation missing"].sum()),
        "Num missing realized mean centroid variation": int(validation_dataframe["Realized mean centroid variation missing"].sum()),
        "Num missing planned maximum projected distance": int(validation_dataframe["Planned max projected distance missing"].sum()),
        "Num missing realized maximum projected distance": int(validation_dataframe["Realized max projected distance missing"].sum()),
    }


def assemble_simulated_biopsy_centroid_variation_validation_fragments(patient_validation_dataframes):
    """Assemble patient validation fragments and return the cohort-style summary."""
    validation_rows = []
    for patient_validation_dataframe in patient_validation_dataframes:
        if not patient_validation_dataframe.empty:
            validation_rows.extend(patient_validation_dataframe.to_dict("records"))

    validation_dataframe = pandas.DataFrame(validation_rows)

    return validation_dataframe, summarize_simulated_biopsy_centroid_variation_validation_dataframe(validation_dataframe)