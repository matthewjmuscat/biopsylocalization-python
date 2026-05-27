from sampling import biopsy_point_sampler

from preprocessing.biopsy_processing.simulated_biopsy_planner import _apply_simulated_biopsy_planning_sample_state
from preprocessing.biopsy_processing.simulated_biopsy_planner import build_simulated_biopsy_planning_state
from preprocessing.biopsy_processing.simulated_biopsy_planner import get_planned_simulated_biopsy_model_dict

from ._presentation import resolve_patient_biopsy_presentation_boundary


def plan_patient_simulated_biopsies(*,
                                    patient_uid,
                                    pydicom_item,
                                    bx_ref,
                                    bx_sample_pts_lattice_spacing,
                                    parallel_pool,
                                    centroid_line_vec_sim_list,
                                    centroid_first_pos_sim_list,
                                    num_centroids_for_sim_bxs,
                                    simulated_bx_rad,
                                    plot_simulated_cores_immediately,
                                    structures_progress=None,
                                    processing_structures_task=None):
    """Build planned simulated-biopsy geometry and samples for one patient."""
    boundary = resolve_patient_biopsy_presentation_boundary(
        structures_progress=structures_progress,
        processing_structures_task=processing_structures_task,
        task_description="Planning simulated biopsy structures [{}]".format(patient_uid),
        task_total=sum(
            1 for specific_structure in pydicom_item.get(bx_ref, ())
            if bool(specific_structure.get("Simulated bool"))
        ),
    )
    structures_progress = boundary.structures_progress
    processing_structures_task = boundary.processing_structures_task
    planning_sample_targets = []
    planning_sample_args_list = []

    for specific_structure in pydicom_item[bx_ref]:
        structureID = specific_structure["ROI"]
        simulated_bool = specific_structure["Simulated bool"]

        if simulated_bool == False:
            continue

        processing_structures_task_main_description = "[cyan]Planning structures [{},{}]...".format(
            patient_uid,
            structureID,
        )
        structures_progress.update(processing_structures_task, description=processing_structures_task_main_description)

        build_simulated_biopsy_planning_state(
            specific_structure,
            centroid_line_vec_sim_list,
            centroid_first_pos_sim_list,
            num_centroids_for_sim_bxs,
            simulated_bx_rad,
            plot_simulated_cores_immediately,
        )

        planned_reconstructed_biopsy_model_dict = get_planned_simulated_biopsy_model_dict(specific_structure)
        planning_sample_targets.append(specific_structure)
        planning_sample_args_list.append((
            bx_sample_pts_lattice_spacing,
            planned_reconstructed_biopsy_model_dict["Reconstructed structure delaunay global"].delaunay_triangulation,
            planned_reconstructed_biopsy_model_dict["Reconstructed structure pts arr"],
            patient_uid,
            bx_ref,
            specific_structure["Index number"],
            planned_reconstructed_biopsy_model_dict["Centroid line to z axis rotation matrix"].T,
        ))

    if planning_sample_args_list:
        parallel_results_sampled_bx_points = parallel_pool.starmap(
            biopsy_point_sampler.sample_biopsy_points_from_reconstructed_global_delaunay_convex_structure,
            planning_sample_args_list,
        )

        for specific_structure, parallel_result in zip(planning_sample_targets, parallel_results_sampled_bx_points):
            sampled_bx_pts_arr, bounding_box_pts_arr, num_sampled_bx_pts, sampling_metadata = parallel_result
            _apply_simulated_biopsy_planning_sample_state(
                specific_structure,
                bx_sample_pts_lattice_spacing,
                sampled_bx_pts_arr,
                bounding_box_pts_arr,
                num_sampled_bx_pts,
                sampling_metadata,
            )
            structures_progress.update(processing_structures_task, advance=1)