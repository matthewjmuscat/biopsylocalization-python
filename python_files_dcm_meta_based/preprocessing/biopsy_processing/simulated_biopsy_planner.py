import biopsy_creator

from preprocessing.biopsy_processing.biopsy_geometry_helper import build_reconstructed_biopsy_model_for_sampling_from_zslice_list
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
        "Planned reconstructed biopsy model dict": None,
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
    planned_reconstructed_biopsy_model_dict = build_reconstructed_biopsy_model_for_sampling_from_zslice_list(
        planned_raw_contour_pts_zslice_list,
        simulated_bx_rad,
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
    simulated_biopsy_planning_dict["Planned reconstructed biopsy model dict"] = planned_reconstructed_biopsy_model_dict
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


def get_planned_simulated_biopsy_model_dict(specific_structure):
    simulated_biopsy_planning_dict = _get_simulated_biopsy_planning_dict(specific_structure)
    planned_reconstructed_biopsy_model_dict = simulated_biopsy_planning_dict.get("Planned reconstructed biopsy model dict")

    if planned_reconstructed_biopsy_model_dict is None:
        raise ValueError(
            "Simulated biopsy planning sampling model is missing for {}".format(
                specific_structure.get("ROI")
            )
        )

    return planned_reconstructed_biopsy_model_dict


def simulated_biopsy_planner_processer(master_structure_reference_dict,
                                       master_structure_info_dict,
                                       bx_ref,
                                       patients_progress,
                                       structures_progress,
                                       completed_progress,
                                       live_display,
                                       centroid_line_vec_sim_list,
                                       centroid_first_pos_sim_list,
                                       num_centroids_for_sim_bxs,
                                       simulated_bx_rad,
                                       plot_simulated_cores_immediately
                                       ):
    patientUID_default = "Initializing"
    processing_patients_task_main_description = "[red]Planning patient sim-bx data [{}]...".format(patientUID_default)
    processing_patients_task_completed_main_description = "[green]Planning patient sim-bx data"
    processing_patients_task = patients_progress.add_task(processing_patients_task_main_description, total=master_structure_info_dict["Global"]["Num cases"])
    processing_patients_task_completed = completed_progress.add_task(processing_patients_task_completed_main_description, total=master_structure_info_dict["Global"]["Num cases"], visible=False)

    for patientUID, pydicom_item in master_structure_reference_dict.items():
        processing_patients_task_main_description = "[red]Planning patient sim-bx data [{}]...".format(patientUID)
        patients_progress.update(processing_patients_task, description=processing_patients_task_main_description)

        structureID_default = "Initializing"
        num_sim_bx_structs_patient_specific = master_structure_info_dict["By patient"][patientUID][bx_ref]["Num sim structs"]
        processing_structures_task_main_description = "[cyan]Planning structures [{},{}]...".format(patientUID, structureID_default)
        processing_structures_task = structures_progress.add_task(processing_structures_task_main_description, total=num_sim_bx_structs_patient_specific)

        for specific_structure in pydicom_item[bx_ref]:
            structureID = specific_structure["ROI"]
            simulated_bool = specific_structure["Simulated bool"]

            if simulated_bool == False:
                continue

            processing_structures_task_main_description = "[cyan]Planning structures [{},{}]...".format(patientUID, structureID)
            structures_progress.update(processing_structures_task, description=processing_structures_task_main_description)

            build_simulated_biopsy_planning_state(
                specific_structure,
                centroid_line_vec_sim_list,
                centroid_first_pos_sim_list,
                num_centroids_for_sim_bxs,
                simulated_bx_rad,
                plot_simulated_cores_immediately,
            )

            structures_progress.update(processing_structures_task, advance=1)

        structures_progress.remove_task(processing_structures_task)
        patients_progress.update(processing_patients_task, advance=1)
        completed_progress.update(processing_patients_task_completed, advance=1)

    patients_progress.update(processing_patients_task, visible=False)
    completed_progress.update(processing_patients_task_completed, visible=True)

    return live_display