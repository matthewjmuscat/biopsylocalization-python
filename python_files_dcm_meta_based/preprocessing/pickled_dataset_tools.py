import copy
import pickle

import numpy as np

import dataframe_builders
import lattice_reconstruction_tools
import misc_tools
import plotting_funcs
import point_containment_tools


PREPROCESSED_EXPORT_MODE = "preprocessed"
RESULTS_EXPORT_MODE = "results"


def _pop_keys(mapping_obj, key_names):
    for key_name in key_names:
        mapping_obj.pop(key_name, None)


def _clear_delaunay_lineset_if_present(delaunay_obj):
    if delaunay_obj is not None and hasattr(delaunay_obj, "delaunay_line_set"):
        delaunay_obj.delaunay_line_set = None


def _sanitize_planned_simulated_biopsy_runtime_objects(specific_bx_structure):
    simulated_biopsy_planning_dict = specific_bx_structure.get("Simulated biopsy planning dict")
    if simulated_biopsy_planning_dict is None:
        return

    planned_reconstructed_biopsy_model_dict = simulated_biopsy_planning_dict.get("Planned reconstructed biopsy model dict")
    if planned_reconstructed_biopsy_model_dict is None:
        return

    planned_reconstructed_biopsy_model_dict["Reconstructed structure point cloud"] = None
    _clear_delaunay_lineset_if_present(planned_reconstructed_biopsy_model_dict.get("Reconstructed structure delaunay global"))


def _sanitize_general_structure_for_pickle(specific_structure,
                                           remove_uncertainty_data=False):
    _pop_keys(
        specific_structure,
        [
            "Point cloud raw",
            "Interpolated structure point cloud dict",
            "Structure OPEN3D triangle mesh object",
        ],
    )

    if remove_uncertainty_data:
        specific_structure.pop("Uncertainty data", None)


def _sanitize_bx_structure_for_pickle(specific_bx_structure,
                                      export_mode):
    _sanitize_planned_simulated_biopsy_runtime_objects(specific_bx_structure)
    _sanitize_general_structure_for_pickle(
        specific_bx_structure,
        remove_uncertainty_data=(export_mode == RESULTS_EXPORT_MODE),
    )

    specific_bx_structure["Reconstructed structure point cloud"] = None
    _clear_delaunay_lineset_if_present(specific_bx_structure.get("Reconstructed structure delaunay global"))

    if export_mode == RESULTS_EXPORT_MODE:
        _pop_keys(
            specific_bx_structure,
            [
                "Random uniformly sampled volume pts pcd",
                "Random uniformly sampled volume pts bx coord sys pcd",
                "Bounding box for random uniformly sampled volume pts",
                "MC data: bx and structure shifted dict",
                "MC data: bx to dose NN search objects list",
                "FANOVA: sobol indices (containment)",
                "FANOVA: sobol indices (dose)",
                "FANOVA: sobol indices (DIL tissue)",
            ],
        )


def _sanitize_dose_dict_for_pickle(dose_ref_dict):
    _pop_keys(
        dose_ref_dict,
        [
            "Dose grid point cloud",
            "Dose grid point cloud thresholded",
            "Dose grid gradient point cloud",
            "Dose grid gradient point cloud thresholded",
            "KDtree",
            "KDtree gradient",
        ],
    )


def _sanitize_mr_dict_for_pickle(mr_adc_subdict):
    _pop_keys(
        mr_adc_subdict,
        [
            "MR ADC grid point cloud",
            "MR ADC grid point cloud thresholded",
            "KDtree",
        ],
    )


def build_pickle_safe_master_structure_reference_dict(master_structure_reference_dict,
                                                      export_mode,
                                                      bx_ref,
                                                      oar_ref,
                                                      dil_ref,
                                                      rectum_ref_key,
                                                      urethra_ref_key,
                                                      dose_ref,
                                                      mr_adc_ref):
    master_structure_reference_dict_safe = copy.deepcopy(master_structure_reference_dict)

    for _patient_uid, pydicom_item in master_structure_reference_dict_safe.items():
        for specific_bx_structure in pydicom_item.get(bx_ref, []):
            _sanitize_bx_structure_for_pickle(specific_bx_structure, export_mode)

        remove_uncertainty_data = export_mode == RESULTS_EXPORT_MODE
        for specific_oar_structure in pydicom_item.get(oar_ref, []):
            _sanitize_general_structure_for_pickle(specific_oar_structure, remove_uncertainty_data=remove_uncertainty_data)
        for specific_dil_structure in pydicom_item.get(dil_ref, []):
            _sanitize_general_structure_for_pickle(specific_dil_structure, remove_uncertainty_data=remove_uncertainty_data)
        for specific_rectum_structure in pydicom_item.get(rectum_ref_key, []):
            _sanitize_general_structure_for_pickle(specific_rectum_structure, remove_uncertainty_data=remove_uncertainty_data)
        for specific_urethra_structure in pydicom_item.get(urethra_ref_key, []):
            _sanitize_general_structure_for_pickle(specific_urethra_structure, remove_uncertainty_data=remove_uncertainty_data)

        if dose_ref in pydicom_item:
            _sanitize_dose_dict_for_pickle(pydicom_item[dose_ref])
        if mr_adc_ref in pydicom_item:
            _sanitize_mr_dict_for_pickle(pydicom_item[mr_adc_ref])

    return master_structure_reference_dict_safe


def load_pickle_bundle(master_structure_reference_dict_path,
                       master_structure_info_dict_path):
    with open(master_structure_reference_dict_path, "rb") as master_structure_reference_dict_file:
        master_structure_reference_dict = pickle.load(master_structure_reference_dict_file)

    with open(master_structure_info_dict_path, "rb") as master_structure_info_dict_file:
        master_structure_info_dict = pickle.load(master_structure_info_dict_file)

    return master_structure_reference_dict, master_structure_info_dict


def export_preprocessed_pickle_bundle(master_structure_reference_dict,
                                      master_structure_info_dict,
                                      export_dir,
                                      reference_dict_filename,
                                      info_dict_filename,
                                      summary_filename,
                                      structs_referenced_list,
                                      bx_ref,
                                      oar_ref,
                                      dil_ref,
                                      rectum_ref_key,
                                      urethra_ref_key,
                                      dose_ref,
                                      mr_adc_ref):
    master_structure_reference_dict_safe = build_pickle_safe_master_structure_reference_dict(
        master_structure_reference_dict,
        PREPROCESSED_EXPORT_MODE,
        bx_ref,
        oar_ref,
        dil_ref,
        rectum_ref_key,
        urethra_ref_key,
        dose_ref,
        mr_adc_ref,
    )

    reference_dict_path = export_dir.joinpath(reference_dict_filename)
    with open(reference_dict_path, "wb") as master_structure_reference_dict_file:
        pickle.dump(master_structure_reference_dict_safe, master_structure_reference_dict_file)

    info_dict_path = export_dir.joinpath(info_dict_filename)
    with open(info_dict_path, "wb") as master_structure_info_dict_file:
        pickle.dump(master_structure_info_dict, master_structure_info_dict_file)

    summary_path = export_dir.joinpath(summary_filename)
    preprocessed_info_dataframe = dataframe_builders.preprocessed_dataset_summary_dataframe_builder(
        master_structure_reference_dict,
        master_structure_info_dict,
        structs_referenced_list,
    )
    preprocessed_info_dataframe.to_csv(summary_path, index=False)

    return {
        "reference_dict_path": reference_dict_path,
        "info_dict_path": info_dict_path,
        "summary_path": summary_path,
    }


def export_results_pickle_bundle(master_structure_reference_dict,
                                 master_structure_info_dict,
                                 export_dir,
                                 reference_dict_filename,
                                 info_dict_filename,
                                 bx_ref,
                                 oar_ref,
                                 dil_ref,
                                 rectum_ref_key,
                                 urethra_ref_key,
                                 dose_ref,
                                 mr_adc_ref):
    master_structure_reference_dict_safe = build_pickle_safe_master_structure_reference_dict(
        master_structure_reference_dict,
        RESULTS_EXPORT_MODE,
        bx_ref,
        oar_ref,
        dil_ref,
        rectum_ref_key,
        urethra_ref_key,
        dose_ref,
        mr_adc_ref,
    )

    reference_dict_path = export_dir.joinpath(reference_dict_filename)
    with open(reference_dict_path, "wb") as master_structure_reference_dict_file:
        pickle.dump(master_structure_reference_dict_safe, master_structure_reference_dict_file)

    info_dict_path = export_dir.joinpath(info_dict_filename)
    with open(info_dict_path, "wb") as master_structure_info_dict_file:
        pickle.dump(master_structure_info_dict, master_structure_info_dict_file)

    return {
        "reference_dict_path": reference_dict_path,
        "info_dict_path": info_dict_path,
    }


def _structure_pcd_color(structs_referenced_dict,
                         structure_type,
                         specific_structure,
                         bx_ref):
    if structure_type == bx_ref:
        return structs_referenced_dict[structure_type]["PCD color dict"][specific_structure["Simulated type"]]

    return structs_referenced_dict[structure_type]["PCD color"]


def _rebuild_planned_simulated_biopsy_runtime_objects(specific_bx_structure,
                                                      pcd_struct_color):
    simulated_biopsy_planning_dict = specific_bx_structure.get("Simulated biopsy planning dict")
    if simulated_biopsy_planning_dict is None:
        return

    planned_reconstructed_biopsy_model_dict = simulated_biopsy_planning_dict.get("Planned reconstructed biopsy model dict")
    if planned_reconstructed_biopsy_model_dict is None:
        return

    drawn_biopsy_array = planned_reconstructed_biopsy_model_dict.get("Reconstructed structure pts arr")
    if drawn_biopsy_array is not None:
        planned_reconstructed_biopsy_model_dict["Reconstructed structure point cloud"] = point_containment_tools.create_point_cloud(
            drawn_biopsy_array,
            pcd_struct_color,
        )

    planned_reconstructed_bx_delaunay_global_convex_structure_obj = planned_reconstructed_biopsy_model_dict.get("Reconstructed structure delaunay global")
    if planned_reconstructed_bx_delaunay_global_convex_structure_obj is not None:
        planned_reconstructed_bx_delaunay_global_convex_structure_obj.generate_lineset()
        planned_reconstructed_biopsy_model_dict["Reconstructed structure delaunay global"] = planned_reconstructed_bx_delaunay_global_convex_structure_obj


def rebuild_loaded_preprocessed_runtime_objects(master_structure_reference_dict,
                                                master_structure_info_dict,
                                                structs_referenced_list_generalized,
                                                structs_referenced_dict,
                                                bx_ref,
                                                dose_ref,
                                                mr_adc_ref,
                                                interp_inter_slice_dist,
                                                interp_intra_slice_dist,
                                                radius_for_normals_estimation,
                                                max_nn_for_normals_estimation,
                                                lower_bound_dose_value,
                                                lower_bound_dose_gradient_value,
                                                lower_bound_mr_adc_value,
                                                upper_bound_mr_adc_value,
                                                color_flattening_deg_MR,
                                                patients_progress,
                                                completed_progress,
                                                indeterminate_progress_sub,
                                                live_display):
    patient_uid_default = "Initializing"
    pickling_dose_patients_task_main_description = "[red]Rebuilding non-picklable dose data [{}]...".format(patient_uid_default)
    pickling_dose_patients_task_completed_main_description = "[green]Rebuilding non-picklable dose data"
    pickling_dose_patients_task = patients_progress.add_task(pickling_dose_patients_task_main_description, total=master_structure_info_dict["Global"]["Num cases"])
    pickling_dose_patients_task_completed = completed_progress.add_task(pickling_dose_patients_task_completed_main_description, total=master_structure_info_dict["Global"]["Num cases"], visible=False)

    for patient_uid, pydicom_item in master_structure_reference_dict.items():
        pickling_dose_patients_task_main_description = "[red]Rebuilding non-picklable dose data [{}]...".format(patient_uid)
        patients_progress.update(pickling_dose_patients_task, description=pickling_dose_patients_task_main_description)

        if dose_ref not in pydicom_item:
            patients_progress.update(pickling_dose_patients_task, advance=1)
            completed_progress.update(pickling_dose_patients_task_completed, advance=1)
            continue

        dose_ref_dict = pydicom_item[dose_ref]
        phys_space_dose_map_and_gradient_map_3d_arr = dose_ref_dict["Dose and gradient phys space and pixel 3d arr"]

        dose_point_cloud, dose_gradient_arrows_point_cloud = plotting_funcs.create_dose_point_cloud_with_gradients(
            phys_space_dose_map_and_gradient_map_3d_arr,
            paint_dose_color=True,
            arrow_scale=1.0,
            truncate_below_dose=None,
            truncate_below_gradient_norm=None,
        )
        thresholded_dose_point_cloud, thresholded_dose_gradient_arrows_point_cloud = plotting_funcs.create_dose_point_cloud_with_gradients(
            phys_space_dose_map_and_gradient_map_3d_arr,
            paint_dose_color=True,
            arrow_scale=1.0,
            truncate_below_dose=lower_bound_dose_value,
            truncate_below_gradient_norm=lower_bound_dose_gradient_value,
        )

        dose_ref_dict["Dose grid point cloud"] = dose_point_cloud
        dose_ref_dict["Dose grid point cloud thresholded"] = thresholded_dose_point_cloud
        dose_ref_dict["Dose grid gradient point cloud"] = dose_gradient_arrows_point_cloud
        dose_ref_dict["Dose grid gradient point cloud thresholded"] = thresholded_dose_gradient_arrows_point_cloud
        master_structure_reference_dict[patient_uid][dose_ref] = dose_ref_dict

        patients_progress.update(pickling_dose_patients_task, advance=1)
        completed_progress.update(pickling_dose_patients_task_completed, advance=1)

    patients_progress.update(pickling_dose_patients_task, visible=False)
    completed_progress.update(pickling_dose_patients_task_completed, visible=True)

    patient_uid_default = "Initializing"
    pickling_mr_patients_task_main_description = "[red]Rebuilding non-picklable MR data [{}]...".format(patient_uid_default)
    pickling_mr_patients_task_completed_main_description = "[green]Rebuilding non-picklable MR data"
    pickling_mr_patients_task = patients_progress.add_task(pickling_mr_patients_task_main_description, total=master_structure_info_dict["Global"]["Num cases"])
    pickling_mr_patients_task_completed = completed_progress.add_task(pickling_mr_patients_task_completed_main_description, total=master_structure_info_dict["Global"]["Num cases"], visible=False)

    for patient_uid, pydicom_item in master_structure_reference_dict.items():
        pickling_mr_patients_task_main_description = "[red]Rebuilding non-picklable MR data [{}]...".format(patient_uid)
        patients_progress.update(pickling_mr_patients_task, description=pickling_mr_patients_task_main_description)

        if mr_adc_ref not in pydicom_item:
            patients_progress.update(pickling_mr_patients_task, advance=1)
            completed_progress.update(pickling_mr_patients_task_completed, advance=1)
            continue

        mr_adc_subdict = pydicom_item[mr_adc_ref]
        filtered_non_negative_adc_mr_phys_space_arr = lattice_reconstruction_tools.reconstruct_mr_lattice_with_coordinates_from_dict_v2(
            mr_adc_subdict,
            filter_out_negatives=True,
        )

        mr_adc_point_cloud = plotting_funcs.create_MR_point_cloud(
            filtered_non_negative_adc_mr_phys_space_arr,
            color_flattening_deg_MR,
            paint_mr_color=True,
        )
        thresholded_mr_adc_point_cloud = plotting_funcs.create_thresholded_MR_ADC_point_cloud(
            filtered_non_negative_adc_mr_phys_space_arr,
            color_flattening_deg_MR,
            paint_mr_color=True,
            lower_bound=lower_bound_mr_adc_value,
            upper_bound=upper_bound_mr_adc_value,
        )

        mr_adc_subdict["MR ADC grid point cloud"] = mr_adc_point_cloud
        mr_adc_subdict["MR ADC grid point cloud thresholded"] = thresholded_mr_adc_point_cloud

        patients_progress.update(pickling_mr_patients_task, advance=1)
        completed_progress.update(pickling_mr_patients_task_completed, advance=1)

    patients_progress.update(pickling_mr_patients_task, visible=False)
    completed_progress.update(pickling_mr_patients_task_completed, visible=True)

    patient_uid_default = "Initializing"
    pickling_structure_patients_task_main_description = "[red]Rebuilding non-picklable structure data [{}]...".format(patient_uid_default)
    pickling_structure_patients_task_completed_main_description = "[green]Rebuilding non-picklable structure data"
    pickling_structure_patients_task = patients_progress.add_task(pickling_structure_patients_task_main_description, total=master_structure_info_dict["Global"]["Num cases"])
    pickling_structure_patients_task_completed = completed_progress.add_task(pickling_structure_patients_task_completed_main_description, total=master_structure_info_dict["Global"]["Num cases"], visible=False)

    for patient_uid, pydicom_item in master_structure_reference_dict.items():
        pickling_structure_patients_task_main_description = "[red]Rebuilding non-picklable structure data [{}]...".format(patient_uid)
        patients_progress.update(pickling_structure_patients_task, description=pickling_structure_patients_task_main_description)

        for structure_type in structs_referenced_list_generalized:
            for specific_structure_index, specific_structure in enumerate(pydicom_item[structure_type]):
                specific_structure_roi = specific_structure["ROI"]
                pcd_struct_color = _structure_pcd_color(structs_referenced_dict, structure_type, specific_structure, bx_ref)

                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Creating pcds of interp structures [{}]".format(specific_structure_roi), total=None)
                interslice_interpolation_information = pydicom_item[structure_type][specific_structure_index]["Inter-slice interpolation information"]
                interpolation_information = pydicom_item[structure_type][specific_structure_index]["Intra-slice interpolation information"]
                three_d_data_array_fully_interpolated = interpolation_information.interpolated_pts_np_arr
                three_d_data_array_fully_interpolated_with_end_caps = interpolation_information.interpolated_pts_with_end_caps_np_arr
                three_d_data_array_interslice_interpolation = np.vstack(interslice_interpolation_information.interpolated_pts_list)

                interslice_interp_pcd = point_containment_tools.create_point_cloud(three_d_data_array_interslice_interpolation, pcd_struct_color)
                inter_and_intra_interp_pcd = point_containment_tools.create_point_cloud(three_d_data_array_fully_interpolated, pcd_struct_color)
                inter_and_intra_and_end_caps_interp_pcd = point_containment_tools.create_point_cloud(three_d_data_array_fully_interpolated_with_end_caps, pcd_struct_color)
                interpolated_pcd_dict = {"Interslice": interslice_interp_pcd, "Full": inter_and_intra_interp_pcd, "Full with end caps": inter_and_intra_and_end_caps_interp_pcd}
                master_structure_reference_dict[patient_uid][structure_type][specific_structure_index]["Interpolated structure point cloud dict"] = interpolated_pcd_dict
                indeterminate_progress_sub.update(indeterminate_task, visible=False)

                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Creating pcds of raw structures [{}]".format(specific_structure_roi), total=None)
                three_d_data_array = pydicom_item[structure_type][specific_structure_index]["Raw contour pts"]
                three_d_data_point_cloud = point_containment_tools.create_point_cloud(three_d_data_array, pcd_struct_color)
                master_structure_reference_dict[patient_uid][structure_type][specific_structure_index]["Point cloud raw"] = three_d_data_point_cloud
                indeterminate_progress_sub.update(indeterminate_task, visible=False)

                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Creating trimesh [{}]".format(specific_structure_roi), total=None)
                live_display.refresh()
                fully_interp_with_end_caps_structure_triangle_mesh, _ = misc_tools.compute_structure_triangle_mesh(
                    interp_inter_slice_dist,
                    interp_intra_slice_dist,
                    three_d_data_array_fully_interpolated_with_end_caps,
                    radius_for_normals_estimation,
                    max_nn_for_normals_estimation,
                )
                master_structure_reference_dict[patient_uid][structure_type][specific_structure_index]["Structure OPEN3D triangle mesh object"] = fully_interp_with_end_caps_structure_triangle_mesh
                indeterminate_progress_sub.update(indeterminate_task, visible=False)

                if structure_type == bx_ref:
                    _rebuild_planned_simulated_biopsy_runtime_objects(specific_structure, pcd_struct_color)

                    indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Creating pcd of rcn bpsy [{}]".format(specific_structure_roi), total=None)
                    drawn_biopsy_array = pydicom_item[structure_type][specific_structure_index]["Reconstructed structure pts arr"]
                    reconstructed_biopsy_point_cloud = point_containment_tools.create_point_cloud(drawn_biopsy_array, pcd_struct_color)
                    master_structure_reference_dict[patient_uid][structure_type][specific_structure_index]["Reconstructed structure point cloud"] = reconstructed_biopsy_point_cloud
                    indeterminate_progress_sub.update(indeterminate_task, visible=False)

                    indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Creating delaunay lineset of rcn bpsy [{}]".format(specific_structure_roi), total=None)
                    reconstructed_bx_delaunay_global_convex_structure_obj = pydicom_item[structure_type][specific_structure_index]["Reconstructed structure delaunay global"]
                    if reconstructed_bx_delaunay_global_convex_structure_obj is not None:
                        reconstructed_bx_delaunay_global_convex_structure_obj.generate_lineset()
                        master_structure_reference_dict[patient_uid][structure_type][specific_structure_index]["Reconstructed structure delaunay global"] = reconstructed_bx_delaunay_global_convex_structure_obj
                    indeterminate_progress_sub.update(indeterminate_task, visible=False)

        patients_progress.update(pickling_structure_patients_task, advance=1)
        completed_progress.update(pickling_structure_patients_task_completed, advance=1)

    patients_progress.update(pickling_structure_patients_task, visible=False)
    completed_progress.update(pickling_structure_patients_task_completed, visible=True)

    return live_display