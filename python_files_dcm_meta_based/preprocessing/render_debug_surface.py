import plotting_funcs


def _build_processed_dataset_pcd_list(pydicom_item,
                                      structs_referenced_list,
                                      bx_ref):
    pcd_list = []

    for structs in structs_referenced_list:
        for specific_structure in pydicom_item[structs]:
            if structs == bx_ref:
                structure_pcd = specific_structure["Reconstructed structure point cloud"]
            else:
                structure_pcd = specific_structure["Interpolated structure point cloud dict"]["Full"]
            pcd_list.append(structure_pcd)

    return pcd_list


def _build_processed_dataset_plotly_arrays(pydicom_item,
                                          structs_referenced_list,
                                          structs_referenced_dict,
                                          bx_ref):
    arr_list = []
    arr_const_zslice_arrs_list = []
    arr_names = []
    arr_colors = []

    for structs in structs_referenced_list:
        for specific_structure in pydicom_item[structs]:
            if structs == bx_ref:
                color = structs_referenced_dict[structs]["PCD color dict"][specific_structure["Simulated type"]]
            else:
                color = structs_referenced_dict[structs]["PCD color"]

            rgb_color = plotting_funcs.rgb_array_to_string(color)

            arr_names.append(specific_structure["ROI"])
            arr_colors.append(rgb_color)
            if structs == bx_ref:
                structure_arr = specific_structure["Reconstructed structure pts arr"]
                structure_list_of_arr = [structure_arr]
            else:
                structure_arr = specific_structure["Intra-slice interpolation information"].interpolated_pts_np_arr
                structure_list_of_arr = specific_structure["Equal num zslice contour pts"]

            arr_list.append(structure_arr)
            arr_const_zslice_arrs_list.append(structure_list_of_arr)

    return arr_list, arr_const_zslice_arrs_list, arr_names, arr_colors


def _resolve_dose_threshold(lower_bound_dose_value,
                           pydicom_item,
                           plan_ref):
    if lower_bound_dose_value is not None:
        return lower_bound_dose_value

    try:
        return pydicom_item[plan_ref]["Prescription doses dict"]["TARGET"]
    except Exception:
        return 0


def _render_processed_datasets_open3d(master_structure_reference_dict,
                                      structs_referenced_list,
                                      bx_ref,
                                      dose_ref,
                                      mr_adc_ref,
                                      live_display):
    live_display.stop()

    for patient_uid, pydicom_item in master_structure_reference_dict.items():
        print(patient_uid)
        pcd_list = _build_processed_dataset_pcd_list(
            pydicom_item,
            structs_referenced_list,
            bx_ref,
        )

        plotting_funcs.plot_geometries(*pcd_list)

        if dose_ref in pydicom_item:
            dose_ref_dict = pydicom_item[dose_ref]
            dose_grid_pcd = dose_ref_dict["Dose grid point cloud thresholded"]
            dose_grid_gradient_pcd = dose_ref_dict["Dose grid gradient point cloud thresholded"]
            plotting_funcs.plot_geometries(*(pcd_list + [dose_grid_pcd, dose_grid_gradient_pcd]))

        if mr_adc_ref in pydicom_item:
            mr_adc_subdict = pydicom_item[mr_adc_ref]
            thresholded_mr_adc_point_cloud = mr_adc_subdict["MR ADC grid point cloud thresholded"]
            plotting_funcs.plot_geometries(*(pcd_list + [thresholded_mr_adc_point_cloud]))

        if (mr_adc_ref in pydicom_item) and (dose_ref in pydicom_item):
            dose_ref_dict = pydicom_item[dose_ref]
            mr_adc_subdict = pydicom_item[mr_adc_ref]
            dose_grid_pcd = dose_ref_dict["Dose grid point cloud thresholded"]
            dose_grid_gradient_pcd = dose_ref_dict["Dose grid gradient point cloud thresholded"]
            thresholded_mr_adc_point_cloud = mr_adc_subdict["MR ADC grid point cloud thresholded"]
            plotting_funcs.plot_geometries(*(pcd_list + [thresholded_mr_adc_point_cloud, dose_grid_pcd, dose_grid_gradient_pcd]))

    live_display.start()
    return live_display


def _render_processed_datasets_plotly(master_structure_reference_dict,
                                      structs_referenced_list,
                                      structs_referenced_dict,
                                      bx_ref,
                                      dose_ref,
                                      plan_ref,
                                      lower_bound_dose_value,
                                      show_processed_3d_datasets_renderings_plotly_dict):
    resolved_lower_bound_dose_value = lower_bound_dose_value

    for patient_uid, pydicom_item in master_structure_reference_dict.items():
        arr_list, arr_const_zslice_arrs_list, arr_names, arr_colors = _build_processed_dataset_plotly_arrays(
            pydicom_item,
            structs_referenced_list,
            structs_referenced_dict,
            bx_ref,
        )

        if show_processed_3d_datasets_renderings_plotly_dict["SS Scatter"] is True:
            plotting_funcs.plotly_3dscatter_arbitrary_number_of_arrays_generalized_with_optional_dosimetry(
                arrays_to_plot_list=arr_list,
                colors_for_arrays_list=arr_colors,
                legend_labels=arr_names,
                title_text=f"Processed 3D structure set for {patient_uid}",
                xaxis_title="Left(+)-Right(-), X Axis (mm)",
                yaxis_title="Posterior(+)-Anterior(-), Y Axis (mm)",
                zaxis_title="Superior(+)-Inferior(-), Z Axis (mm)",
                marker_size=0.7,
                bg_color="rgb(245,245,245)",
            )

        if show_processed_3d_datasets_renderings_plotly_dict["SS Contour"] is True:
            plotting_funcs.plotly_3dscatter_arbitrary_number_of_arrays_generalized_with_optional_dosimetry(
                arrays_to_plot_list=arr_const_zslice_arrs_list,
                colors_for_arrays_list=arr_colors,
                legend_labels=arr_names,
                title_text=f"Processed 3D structure set with dosimetry for {patient_uid}",
                xaxis_title="Left(+)-Right(-), X Axis (mm)",
                yaxis_title="Posterior(+)-Anterior(-), Y Axis (mm)",
                zaxis_title="Superior(+)-Inferior(-), Z Axis (mm)",
                marker_size=0.7,
                bg_color="rgb(245,245,245)",
                plot_contours=True,
            )

        if dose_ref not in pydicom_item:
            continue

        dose_ref_dict = pydicom_item[dose_ref]
        phys_space_dose_map_and_gradient_map_3d_arr = dose_ref_dict["Dose and gradient phys space and pixel 3d arr"]
        resolved_lower_bound_dose_value = _resolve_dose_threshold(
            resolved_lower_bound_dose_value,
            pydicom_item,
            plan_ref,
        )

        if show_processed_3d_datasets_renderings_plotly_dict["SS Scatter"] is True:
            plotting_funcs.plotly_3dscatter_arbitrary_number_of_arrays_generalized_with_optional_dosimetry(
                arrays_to_plot_list=arr_list,
                colors_for_arrays_list=arr_colors,
                legend_labels=arr_names,
                title_text=f"Processed 3D structure set with dosimetry for {patient_uid}",
                xaxis_title="Left(+)-Right(-), X Axis (mm)",
                yaxis_title="Posterior(+)-Anterior(-), Y Axis (mm)",
                zaxis_title="Superior(+)-Inferior(-), Z Axis (mm)",
                marker_size=0.7,
                bg_color="rgb(245,245,245)",
                plot_contours=False,
                phys_space_dose_map_and_gradient_map_3d_arr=phys_space_dose_map_and_gradient_map_3d_arr,
                dose_threshold=resolved_lower_bound_dose_value,
                log_scale_colors=show_processed_3d_datasets_renderings_plotly_dict["Dosimetric dose log scale"],
                dose_marker_size=1.2,
                colorbar_title="Dose (Gy)",
                dosimetric_render_mode=show_processed_3d_datasets_renderings_plotly_dict["Dosimetric render mode"],
                dosimetric_opacity=0.05,
                volume_surface_count=20,
                colorbar_x_offset=0.9,
                colorbar_color="Picnic",
                reversescale=False,
            )

        if show_processed_3d_datasets_renderings_plotly_dict["SS Contour"] is True:
            plotting_funcs.plotly_3dscatter_arbitrary_number_of_arrays_generalized_with_optional_dosimetry(
                arrays_to_plot_list=arr_const_zslice_arrs_list,
                colors_for_arrays_list=arr_colors,
                legend_labels=arr_names,
                title_text=f"Processed 3D structure set with dosimetry for {patient_uid}",
                xaxis_title="Left(+)-Right(-), X Axis (mm)",
                yaxis_title="Posterior(+)-Anterior(-), Y Axis (mm)",
                zaxis_title="Superior(+)-Inferior(-), Z Axis (mm)",
                marker_size=0.7,
                bg_color="rgb(245,245,245)",
                plot_contours=True,
                phys_space_dose_map_and_gradient_map_3d_arr=phys_space_dose_map_and_gradient_map_3d_arr,
                dose_threshold=resolved_lower_bound_dose_value,
                log_scale_colors=show_processed_3d_datasets_renderings_plotly_dict["Dosimetric dose log scale"],
                dose_marker_size=1.2,
                colorbar_title="Dose (Gy)",
                dosimetric_render_mode=show_processed_3d_datasets_renderings_plotly_dict["Dosimetric render mode"],
                dosimetric_opacity=0.05,
                volume_surface_count=20,
                colorbar_x_offset=0.9,
                colorbar_color="Picnic",
                reversescale=False,
            )

    return resolved_lower_bound_dose_value


def render_processed_dataset_debug_processer(master_structure_reference_dict,
                                             structs_referenced_list,
                                             structs_referenced_dict,
                                             bx_ref,
                                             dose_ref,
                                             mr_adc_ref,
                                             plan_ref,
                                             lower_bound_dose_value,
                                             show_processed_3d_datasets_renderings,
                                             show_processed_3d_datasets_renderings_plotly_dict,
                                             live_display):
    if show_processed_3d_datasets_renderings is True:
        live_display = _render_processed_datasets_open3d(
            master_structure_reference_dict,
            structs_referenced_list,
            bx_ref,
            dose_ref,
            mr_adc_ref,
            live_display,
        )

    resolved_lower_bound_dose_value = lower_bound_dose_value
    if show_processed_3d_datasets_renderings_plotly_dict["Plot"] is True:
        resolved_lower_bound_dose_value = _render_processed_datasets_plotly(
            master_structure_reference_dict,
            structs_referenced_list,
            structs_referenced_dict,
            bx_ref,
            dose_ref,
            plan_ref,
            lower_bound_dose_value,
            show_processed_3d_datasets_renderings_plotly_dict,
        )

    return resolved_lower_bound_dose_value, live_display