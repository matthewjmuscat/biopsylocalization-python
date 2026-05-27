from preprocessing.biopsy_processing.simulated_biopsy_planner import get_planned_simulated_biopsy_zslice_list
from preprocessing.biopsy_processing.simulated_biopsy_processor import _finalize_simulated_biopsy_geometry
from preprocessing.biopsy_processing.simulated_biopsy_processor import _transport_planned_simulated_biopsy

from ._presentation import resolve_patient_biopsy_presentation_boundary


def process_patient_simulated_biopsies(*,
                                       patient_uid,
                                       pydicom_item,
                                       master_structure_reference_dict,
                                       structs_referenced_dict,
                                       bx_ref,
                                       parallel_pool,
                                       interp_inter_slice_dist,
                                       interp_intra_slice_dist,
                                       interp_dist_caps,
                                       biopsy_radius,
                                       voxel_size_for_structure_volume_calc_non_bx,
                                       factor_for_voxel_size,
                                       cupy_array_upper_limit_NxN_size_input,
                                       nearest_zslice_vals_and_indices_cupy_generic_max_size,
                                       generate_cuda_log_files_volume_calculation,
                                       constant_z_slice_polygons_handler_option,
                                       remove_consecutive_duplicate_points_in_polygons,
                                       include_edges_in_log_files,
                                       custom_cuda_kernel_type,
                                       demonstrate_volume_calculation_correctness_bool_1,
                                       plot_volume_calculation_containment_result_bool_1_old,
                                       plot_binary_mask_bool,
                                       layout_groups=None,
                                       structures_progress=None,
                                       processing_structures_task=None,
                                       indeterminate_progress_sub=None,
                                       live_display=None):
    """Finalize all simulated biopsy geometry for one patient."""
    boundary = resolve_patient_biopsy_presentation_boundary(
        layout_groups=layout_groups,
        structures_progress=structures_progress,
        processing_structures_task=processing_structures_task,
        indeterminate_progress_sub=indeterminate_progress_sub,
        live_display=live_display,
        task_description="Processing simulated biopsy structures [{}]".format(patient_uid),
        task_total=sum(
            1 for specific_structure in pydicom_item.get(bx_ref, ())
            if bool(specific_structure.get("Simulated bool"))
        ),
    )
    layout_groups = boundary.layout_groups
    structures_progress = boundary.structures_progress
    processing_structures_task = boundary.processing_structures_task
    indeterminate_progress_sub = boundary.indeterminate_progress_sub
    live_display = boundary.live_display

    for specific_structure_index, specific_structure in enumerate(pydicom_item[bx_ref]):
        structureID = specific_structure["ROI"]
        simulated_bool = specific_structure["Simulated bool"]

        if simulated_bool == False:
            continue

        processing_structures_task_main_description = "[cyan]Processing structures [{},{}]...".format(
            patient_uid,
            structureID,
        )
        structures_progress.update(processing_structures_task, description=processing_structures_task_main_description)

        planned_threeDdata_zslice_list = get_planned_simulated_biopsy_zslice_list(specific_structure)
        transport_result_dict = _transport_planned_simulated_biopsy(
            pydicom_item,
            specific_structure,
            planned_threeDdata_zslice_list,
        )
        threeDdata_zslice_list = transport_result_dict["Transported raw contour pts zslice list"]
        specific_structure["Simulated biopsy transport dict"] = transport_result_dict["Simulated biopsy transport dict"]

        live_display = _finalize_simulated_biopsy_geometry(
            master_structure_reference_dict,
            patient_uid,
            bx_ref,
            specific_structure_index,
            specific_structure,
            threeDdata_zslice_list,
            structs_referenced_dict,
            parallel_pool,
            interp_inter_slice_dist,
            interp_intra_slice_dist,
            interp_dist_caps,
            biopsy_radius,
            voxel_size_for_structure_volume_calc_non_bx,
            factor_for_voxel_size,
            cupy_array_upper_limit_NxN_size_input,
            layout_groups,
            nearest_zslice_vals_and_indices_cupy_generic_max_size,
            generate_cuda_log_files_volume_calculation,
            constant_z_slice_polygons_handler_option,
            remove_consecutive_duplicate_points_in_polygons,
            include_edges_in_log_files,
            custom_cuda_kernel_type,
            demonstrate_volume_calculation_correctness_bool_1,
            plot_volume_calculation_containment_result_bool_1_old,
            plot_binary_mask_bool,
            structures_progress,
            indeterminate_progress_sub,
            live_display,
        )

        structures_progress.update(processing_structures_task, advance=1)

    return live_display