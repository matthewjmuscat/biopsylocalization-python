from preprocessing.biopsy_processing.biopsy_geometry_helper import finalize_biopsy_geometry_from_zslice_list


def process_patient_real_biopsies(*,
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
                                  display_pca_fit_variation_for_biopsies_bool,
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
                                  processing_structures_task,
                                  indeterminate_progress_sub,
                                  live_display):
    """Finalize all real biopsy geometry for one patient."""
    for specific_structure_index, specific_structure in enumerate(pydicom_item[bx_ref]):
        structureID = specific_structure["ROI"]
        simulated_bool = specific_structure["Simulated bool"]

        if simulated_bool == True:
            continue

        processing_structures_task_main_description = "[cyan]Processing structures [{},{}]...".format(
            patient_uid,
            structureID,
        )
        structures_progress.update(processing_structures_task, description=processing_structures_task_main_description)

        threeDdata_zslice_list = specific_structure["Raw contour pts zslice list"].copy()
        live_display = finalize_biopsy_geometry_from_zslice_list(
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
            display_pca_fit_variation_for_biopsies_bool=display_pca_fit_variation_for_biopsies_bool,
            store_raw_contour_pts_zslice_list_bool=False,
        )

        structures_progress.update(processing_structures_task, advance=1)

    return live_display
