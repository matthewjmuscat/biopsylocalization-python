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