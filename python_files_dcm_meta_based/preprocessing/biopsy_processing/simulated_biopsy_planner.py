import biopsy_creator

from preprocessing.biopsy_processing.simulated_biopsy_preparation import get_prepared_simulated_biopsy_length_mm


def _create_default_simulated_biopsy_planning_dict():
    return {
        "Planning complete": False,
        "Planning frame": None,
        "Nominal length mm": None,
        "Planned biopsy radius mm": None,
        "Planned centroid count": None,
        "Planned centroid separation mm": None,
        "Planned raw contour pts zslice list": None,
        "Planning source": None,
    }


def _get_simulated_biopsy_planning_dict(specific_structure):
    if specific_structure.get("Simulated biopsy planning dict") is None:
        specific_structure["Simulated biopsy planning dict"] = _create_default_simulated_biopsy_planning_dict()

    return specific_structure["Simulated biopsy planning dict"]


def build_simulated_biopsy_planning_state(specific_structure,
                                          centroid_line_vec_sim_list,
                                          centroid_first_pos_sim_list,
                                          num_centroids_for_sim_bxs,
                                          simulated_bx_rad,
                                          plot_simulated_cores_immediately
                                          ):
    nominal_length_mm = get_prepared_simulated_biopsy_length_mm(specific_structure)
    planned_centroid_separation_mm = nominal_length_mm / (num_centroids_for_sim_bxs - 1)

    planned_raw_contour_pts_zslice_list = biopsy_creator.biopsy_points_creater_by_transport_for_sim_bxs(
        centroid_line_vec_sim_list,
        centroid_first_pos_sim_list,
        num_centroids_for_sim_bxs,
        planned_centroid_separation_mm,
        simulated_bx_rad,
        plot_simulated_cores_immediately,
    )

    simulated_biopsy_planning_dict = _get_simulated_biopsy_planning_dict(specific_structure)
    simulated_biopsy_planning_dict["Planning complete"] = True
    simulated_biopsy_planning_dict["Planning frame"] = "Canonical local biopsy frame"
    simulated_biopsy_planning_dict["Nominal length mm"] = float(nominal_length_mm)
    simulated_biopsy_planning_dict["Planned biopsy radius mm"] = float(simulated_bx_rad)
    simulated_biopsy_planning_dict["Planned centroid count"] = int(num_centroids_for_sim_bxs)
    simulated_biopsy_planning_dict["Planned centroid separation mm"] = float(planned_centroid_separation_mm)
    simulated_biopsy_planning_dict["Planned raw contour pts zslice list"] = [
        bx_zslice_arr.copy() for bx_zslice_arr in planned_raw_contour_pts_zslice_list
    ]
    simulated_biopsy_planning_dict["Planning source"] = "Prepared simulated biopsy nominal length"

    return simulated_biopsy_planning_dict


def get_planned_simulated_biopsy_zslice_list(specific_structure):
    simulated_biopsy_planning_dict = _get_simulated_biopsy_planning_dict(specific_structure)
    planned_raw_contour_pts_zslice_list = simulated_biopsy_planning_dict.get("Planned raw contour pts zslice list")

    if planned_raw_contour_pts_zslice_list is None:
        raise ValueError(
            "Simulated biopsy planning geometry is missing for {}".format(
                specific_structure.get("ROI")
            )
        )

    return [bx_zslice_arr.copy() for bx_zslice_arr in planned_raw_contour_pts_zslice_list]