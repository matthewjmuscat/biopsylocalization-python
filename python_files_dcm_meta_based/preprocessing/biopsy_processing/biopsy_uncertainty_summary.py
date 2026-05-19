import numpy as np


def _mean_or_none(values):
    if not values:
        return None

    return float(np.mean(np.array(values, dtype=float)))


def _get_simulated_biopsy_metric_value(specific_bx_structure,
                                       planned_key,
                                       realized_key,
                                       simulated_preference="realized"
                                       ):
    simulated_biopsy_planning_dict = specific_bx_structure.get("Simulated biopsy planning dict") or {}
    planned_value = simulated_biopsy_planning_dict.get(planned_key)
    realized_value = specific_bx_structure.get(realized_key)

    if simulated_preference == "planned":
        if planned_value is not None:
            return float(planned_value)
        if realized_value is not None:
            return float(realized_value)
        return None

    if realized_value is not None:
        return float(realized_value)
    if planned_value is not None:
        return float(planned_value)

    return None


def get_biopsy_mean_centroid_variation_value(specific_bx_structure,
                                             simulated_preference="realized"
                                             ):
    if specific_bx_structure["Simulated bool"] is True:
        return _get_simulated_biopsy_metric_value(
            specific_bx_structure,
            planned_key="Planned mean centroid variation",
            realized_key="Mean centroid variation",
            simulated_preference=simulated_preference,
        )

    mean_centroid_variation = specific_bx_structure.get("Mean centroid variation")
    if mean_centroid_variation is None:
        return None

    return float(mean_centroid_variation)


def get_biopsy_maximum_projected_distance_value(specific_bx_structure,
                                                simulated_preference="realized"
                                                ):
    if specific_bx_structure["Simulated bool"] is True:
        return _get_simulated_biopsy_metric_value(
            specific_bx_structure,
            planned_key="Planned maximum projected distance between original centroids",
            realized_key="Maximum projected distance between original centroids",
            simulated_preference=simulated_preference,
        )

    maximum_projected_distance = specific_bx_structure.get("Maximum projected distance between original centroids")
    if maximum_projected_distance is None:
        return None

    return float(maximum_projected_distance)


def calculate_biopsy_centroid_variation_summary(master_structure_reference_dict,
                                                bx_ref,
                                                simulated_preference="realized"
                                                ):
    real_mean_variations = []
    simulated_mean_variations = []

    for _patient_uid, pydicom_item in master_structure_reference_dict.items():
        for specific_bx_structure in pydicom_item[bx_ref]:
            mean_centroid_variation = get_biopsy_mean_centroid_variation_value(
                specific_bx_structure,
                simulated_preference=simulated_preference,
            )
            if mean_centroid_variation is None:
                continue

            if specific_bx_structure["Simulated bool"] is True:
                simulated_mean_variations.append(mean_centroid_variation)
            else:
                real_mean_variations.append(mean_centroid_variation)

    all_mean_variations = real_mean_variations + simulated_mean_variations

    return {
        "Mean real biopsy centroid variation": _mean_or_none(real_mean_variations),
        "Mean simulated biopsy centroid variation": _mean_or_none(simulated_mean_variations),
        "Mean all biopsy centroid variation": _mean_or_none(all_mean_variations),
        "Num real biopsies with centroid variation": len(real_mean_variations),
        "Num simulated biopsies with centroid variation": len(simulated_mean_variations),
        "Num biopsies with centroid variation": len(all_mean_variations),
    }


def apply_biopsy_centroid_variation_summary(master_structure_info_dict,
                                            summary_dict,
                                            legacy_mean_source="real"
                                            ):
    master_structure_info_global_dict = master_structure_info_dict["Global"]

    legacy_key_lookup = {
        "real": "Mean real biopsy centroid variation",
        "simulated": "Mean simulated biopsy centroid variation",
        "all": "Mean all biopsy centroid variation",
    }

    legacy_source_key = legacy_key_lookup.get(legacy_mean_source, "Mean real biopsy centroid variation")
    legacy_mean_centroid_variation = summary_dict.get(legacy_source_key)
    if legacy_mean_centroid_variation is None:
        legacy_mean_centroid_variation = summary_dict.get("Mean all biopsy centroid variation")
    if legacy_mean_centroid_variation is None:
        legacy_mean_centroid_variation = 0.0

    master_structure_info_global_dict["Mean biopsy centroid variation"] = legacy_mean_centroid_variation
    master_structure_info_global_dict["Mean real biopsy centroid variation"] = summary_dict["Mean real biopsy centroid variation"]
    master_structure_info_global_dict["Mean simulated biopsy centroid variation"] = summary_dict["Mean simulated biopsy centroid variation"]
    master_structure_info_global_dict["Mean all biopsy centroid variation"] = summary_dict["Mean all biopsy centroid variation"]
    master_structure_info_global_dict["Num real biopsies with centroid variation"] = summary_dict["Num real biopsies with centroid variation"]
    master_structure_info_global_dict["Num simulated biopsies with centroid variation"] = summary_dict["Num simulated biopsies with centroid variation"]
    master_structure_info_global_dict["Num biopsies with centroid variation"] = summary_dict["Num biopsies with centroid variation"]

    return summary_dict


def biopsy_centroid_variation_summary_processer(master_structure_reference_dict,
                                                master_structure_info_dict,
                                                bx_ref,
                                                patients_progress,
                                                completed_progress,
                                                live_display,
                                                legacy_mean_source="real",
                                                simulated_preference="realized"
                                                ):
    patient_uid_default = "Initializing"
    processing_patient_description = "Determining biopsy uncertainty [{}]...".format(patient_uid_default)
    processing_patients_task = patients_progress.add_task(
        "[red]" + processing_patient_description,
        total=master_structure_info_dict["Global"]["Num cases"],
    )
    processing_patient_description_completed = "Determining biopsy uncertainty"
    processing_patients_completed_task = completed_progress.add_task(
        "[green]" + processing_patient_description_completed,
        total=master_structure_info_dict["Global"]["Num cases"],
        visible=False,
    )

    for patient_uid in master_structure_reference_dict.keys():
        processing_patient_description = "Determining biopsy uncertainty [{}]...".format(patient_uid)
        patients_progress.update(processing_patients_task, description="[red]" + processing_patient_description)
        patients_progress.update(processing_patients_task, advance=1)
        completed_progress.update(processing_patients_completed_task, advance=1)

    patients_progress.update(processing_patients_task, visible=False)
    completed_progress.update(processing_patients_completed_task, visible=True)

    summary_dict = calculate_biopsy_centroid_variation_summary(
        master_structure_reference_dict,
        bx_ref,
        simulated_preference=simulated_preference,
    )
    apply_biopsy_centroid_variation_summary(
        master_structure_info_dict,
        summary_dict,
        legacy_mean_source=legacy_mean_source,
    )

    return summary_dict, live_display