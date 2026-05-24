import copy

import numpy as np
import open3d as o3d

import math_funcs as mf
import plotting_funcs
import point_containment_tools

from sampling import biopsy_point_sampler


def build_patient_sampled_biopsy_sampling_args(*,
                                               patient_uid,
                                               pydicom_item,
                                               bx_ref,
                                               bx_sample_pts_lattice_spacing,
                                               biopsies_progress=None,
                                               processing_biopsies_task=None):
    args_list = []

    for specific_structure_index, specific_structure in enumerate(pydicom_item[bx_ref]):
        specific_bx_structure_roi = specific_structure["ROI"]
        if biopsies_progress is not None and processing_biopsies_task is not None:
            processing_biopsies_main_description = "[cyan]Preparing biopsy data for parallel processing [{},{}]...".format(
                patient_uid,
                specific_bx_structure_roi,
            )
            biopsies_progress.update(processing_biopsies_task, description=processing_biopsies_main_description)

        reconstructed_biopsy_arr = specific_structure["Reconstructed structure pts arr"]
        reconstructed_delaunay_global_convex_structure_obj = specific_structure["Reconstructed structure delaunay global"]

        z_axis_np_vec = np.array([0, 0, 1], dtype=float)
        apex_to_base_bx_best_fit_vec = specific_structure["Centroid line vec (bx needle base to bx needle tip)"]
        z_axis_to_centroid_vec_rotation_matrix = mf.rotation_matrix_from_vectors(
            z_axis_np_vec,
            apex_to_base_bx_best_fit_vec,
        )

        args_list.append(
            (
                bx_sample_pts_lattice_spacing,
                reconstructed_delaunay_global_convex_structure_obj.delaunay_triangulation,
                reconstructed_biopsy_arr,
                patient_uid,
                bx_ref,
                specific_structure_index,
                z_axis_to_centroid_vec_rotation_matrix,
            )
        )

        if biopsies_progress is not None and processing_biopsies_task is not None:
            biopsies_progress.update(processing_biopsies_task, advance=1)

    return args_list


def store_patient_sampled_biopsy_results(*,
                                         patient_uid,
                                         pydicom_item,
                                         bx_ref,
                                         parallel_results_sampled_bx_points_from_global_delaunay_arr_and_bounding_box_arr,
                                         biopsies_progress=None,
                                         parsing_sampled_biopsy_data_task=None,
                                         completed_progress=None,
                                         parsing_sampled_biopsy_data_task_completed=None,
                                         stopwatch=None,
                                         live_display=None):
    stored_result_count = 0

    for sampled_bx_pts_arr, axis_aligned_bounding_box_arr, num_sample_pts_in_specific_bx, structure_info_dict in parallel_results_sampled_bx_points_from_global_delaunay_arr_and_bounding_box_arr:
        temp_patient_uid = structure_info_dict["Patient UID"]
        temp_structure_type = structure_info_dict["Structure type"]
        temp_specific_structure_index = structure_info_dict["Specific structure index"]

        if temp_patient_uid != patient_uid or temp_structure_type != bx_ref:
            raise ValueError(
                "store_patient_sampled_biopsy_results received a non-patient-local result fragment"
            )

        temp_structure_id = pydicom_item[temp_structure_type][temp_specific_structure_index]["ROI"]

        if biopsies_progress is not None and parsing_sampled_biopsy_data_task is not None:
            parsing_sampled_biopsy_data_task_main_description = "Parsing sampled biopsy information [{},{}]".format(
                temp_patient_uid,
                temp_structure_id,
            )
            biopsies_progress.update(
                parsing_sampled_biopsy_data_task,
                description="[red]" + parsing_sampled_biopsy_data_task_main_description,
                refresh=True,
            )
        if live_display is not None:
            live_display.refresh()

        sampled_bx_points_from_global_delaunay_point_cloud_color = np.random.uniform(0, 0.7, size=3)
        sampled_bx_points_from_global_delaunay_point_cloud = point_containment_tools.create_point_cloud(
            sampled_bx_pts_arr,
            sampled_bx_points_from_global_delaunay_point_cloud_color,
        )
        axis_aligned_bounding_box = o3d.geometry.AxisAlignedBoundingBox()
        axis_aligned_bounding_box_o3d3dvector_points = o3d.utility.Vector3dVector(axis_aligned_bounding_box_arr)
        axis_aligned_bounding_box = axis_aligned_bounding_box.create_from_points(axis_aligned_bounding_box_o3d3dvector_points)
        axis_aligned_bounding_box.color = np.array([0, 0, 0], dtype=float)

        specific_structure = pydicom_item[temp_structure_type][temp_specific_structure_index]
        specific_structure["Random uniformly sampled volume pts arr"] = sampled_bx_pts_arr
        specific_structure["Random uniformly sampled volume pts pcd"] = sampled_bx_points_from_global_delaunay_point_cloud
        specific_structure["Bounding box for random uniformly sampled volume pts"] = axis_aligned_bounding_box
        specific_structure["Num sampled bx pts"] = num_sample_pts_in_specific_bx
        reconstructed_bx_pcd = specific_structure["Reconstructed structure point cloud"]

        if biopsies_progress is not None and parsing_sampled_biopsy_data_task is not None:
            biopsies_progress.stop_task(parsing_sampled_biopsy_data_task)
        if completed_progress is not None and parsing_sampled_biopsy_data_task_completed is not None:
            completed_progress.stop_task(parsing_sampled_biopsy_data_task_completed)
        if stopwatch is not None:
            stopwatch.stop()
        #plotting_funcs.plot_geometries(sampled_bx_points_from_global_delaunay_point_cloud, reconstructed_bx_pcd, axis_aligned_bounding_box)
        #plotting_funcs.plot_geometries(sampled_bx_points_from_global_delaunay_point_cloud, reconstructed_bx_pcd)
        del reconstructed_bx_pcd
        if stopwatch is not None:
            stopwatch.start()
        if biopsies_progress is not None and parsing_sampled_biopsy_data_task is not None:
            biopsies_progress.start_task(parsing_sampled_biopsy_data_task)
        if completed_progress is not None and parsing_sampled_biopsy_data_task_completed is not None:
            completed_progress.start_task(parsing_sampled_biopsy_data_task_completed)

        if biopsies_progress is not None and parsing_sampled_biopsy_data_task is not None:
            biopsies_progress.update(parsing_sampled_biopsy_data_task, advance=1, refresh=True)
        if completed_progress is not None and parsing_sampled_biopsy_data_task_completed is not None:
            completed_progress.update(parsing_sampled_biopsy_data_task_completed, advance=1, refresh=True)
        if live_display is not None:
            live_display.refresh()

        stored_result_count += 1

    return stored_result_count


def create_patient_biopsy_oriented_coordinate_system(*,
                                                     patient_uid,
                                                     pydicom_item,
                                                     bx_ref,
                                                     stopwatch=None,
                                                     show_reconstructed_biopsy_in_biopsy_coord_sys_tr_and_rot=False,
                                                     patients_progress=None,
                                                     processing_patients_task=None,
                                                     completed_progress=None,
                                                     processing_patients_completed_task=None):
    processed_biopsy_count = 0

    if patients_progress is not None and processing_patients_task is not None:
        processing_patient_rotating_bx_main_description = "Creating biopsy oriented coordinate system [{}]...".format(
            patient_uid,
        )
        patients_progress.update(processing_patients_task, description="[red]" + processing_patient_rotating_bx_main_description)

    for specific_structure in pydicom_item[bx_ref]:
        bx_best_fit_line_of_reconstructed_centroids = specific_structure["Best fit line of centroid pts"]
        vec_with_largest_z_val_index = bx_best_fit_line_of_reconstructed_centroids[:, 2].argmax()
        vec_with_largest_z_val = bx_best_fit_line_of_reconstructed_centroids[vec_with_largest_z_val_index, :]
        base_sup_vec_bx_centroid_arr = vec_with_largest_z_val

        vec_with_smallest_z_val_index = bx_best_fit_line_of_reconstructed_centroids[:, 2].argmin()
        vec_with_smallest_z_val = bx_best_fit_line_of_reconstructed_centroids[vec_with_smallest_z_val_index, :]
        apex_inf_vec_bx_centroid_arr = vec_with_smallest_z_val

        translation_vec_bx_coord_sys_origin = -apex_inf_vec_bx_centroid_arr
        apex_to_base_bx_best_fit_vec = specific_structure["Centroid line vec (bx needle base to bx needle tip)"]

        reconstructed_biopsy_point_cloud = specific_structure["Reconstructed structure point cloud"]
        reconstructed_biopsy_arr = specific_structure["Reconstructed structure pts arr"]
        sampled_bx_points_pcd = specific_structure["Random uniformly sampled volume pts pcd"]
        sampled_bx_points_arr = specific_structure["Random uniformly sampled volume pts arr"]
        axis_aligned_bounding_box = specific_structure["Bounding box for random uniformly sampled volume pts"]

        reconstructed_biopsy_bx_coord_sys_tr_arr = reconstructed_biopsy_arr + translation_vec_bx_coord_sys_origin
        sampled_bx_points_bx_coord_sys_tr_arr = sampled_bx_points_arr + translation_vec_bx_coord_sys_origin
        reconstructed_biopsy_bx_coord_sys_tr_point_cloud = copy.copy(reconstructed_biopsy_point_cloud)
        reconstructed_biopsy_bx_coord_sys_tr_from_arr_point_cloud = point_containment_tools.create_point_cloud(reconstructed_biopsy_bx_coord_sys_tr_arr)
        sampled_bx_points_bx_coord_sys_tr_pcd = copy.copy(sampled_bx_points_pcd)

        reconstructed_biopsy_bx_coord_sys_tr_point_cloud.translate(translation_vec_bx_coord_sys_origin)
        sampled_bx_points_bx_coord_sys_tr_pcd.translate(translation_vec_bx_coord_sys_origin)

        if patients_progress is not None and processing_patients_task is not None:
            patients_progress.stop_task(processing_patients_task)
        if completed_progress is not None and processing_patients_completed_task is not None:
            completed_progress.stop_task(processing_patients_completed_task)
        if stopwatch is not None:
            stopwatch.stop()
        #plotting_funcs.plot_geometries(sampled_bx_points_pcd, reconstructed_biopsy_point_cloud, axis_aligned_bounding_box, reconstructed_biopsy_bx_coord_sys_tr_point_cloud, sampled_bx_points_bx_coord_sys_tr_pcd)
        if stopwatch is not None:
            stopwatch.start()
        if patients_progress is not None and processing_patients_task is not None:
            patients_progress.start_task(processing_patients_task)
        if completed_progress is not None and processing_patients_completed_task is not None:
            completed_progress.start_task(processing_patients_completed_task)

        z_axis_np_vec = np.array([0, 0, 1], dtype=float)
        centroid_line_to_z_axis_rotation_matrix_other = mf.rotation_matrix_from_vectors(apex_to_base_bx_best_fit_vec, z_axis_np_vec)

        reconstructed_biopsy_bx_coord_sys_tr_and_rot_point_cloud = copy.copy(reconstructed_biopsy_bx_coord_sys_tr_point_cloud)
        sampled_bx_points_bx_coord_sys_tr_and_rot_pcd = copy.copy(sampled_bx_points_bx_coord_sys_tr_pcd)

        reconstructed_biopsy_bx_coord_sys_tr_and_rot_point_cloud.rotate(centroid_line_to_z_axis_rotation_matrix_other, center=(0, 0, 0))
        sampled_bx_points_bx_coord_sys_tr_and_rot_pcd.rotate(centroid_line_to_z_axis_rotation_matrix_other, center=(0, 0, 0))

        reconstructed_biopsy_bx_coord_sys_tr_and_rot_arr = (centroid_line_to_z_axis_rotation_matrix_other @ reconstructed_biopsy_bx_coord_sys_tr_arr.T).T
        sampled_bx_points_bx_coord_sys_tr_and_rot_arr = (centroid_line_to_z_axis_rotation_matrix_other @ sampled_bx_points_bx_coord_sys_tr_arr.T).T
        reconstructed_biopsy_bx_coord_sys_tr_and_rot_arr_point_cloud = point_containment_tools.create_point_cloud(reconstructed_biopsy_bx_coord_sys_tr_and_rot_arr)
        sampled_bx_points_bx_coord_sys_tr_and_rot_arr_point_cloud = point_containment_tools.create_point_cloud(sampled_bx_points_bx_coord_sys_tr_and_rot_arr)

        if show_reconstructed_biopsy_in_biopsy_coord_sys_tr_and_rot is True:
            reconstructed_biopsy_bx_coord_sys_tr_and_rot_axis_aligned_bounding_box = reconstructed_biopsy_bx_coord_sys_tr_and_rot_point_cloud.get_axis_aligned_bounding_box()
            reconstructed_biopsy_bx_coord_sys_tr_and_rot_axis_aligned_bounding_box.color = np.array([0, 0, 0], dtype=float)
            if patients_progress is not None and processing_patients_task is not None:
                patients_progress.stop_task(processing_patients_task)
            if completed_progress is not None and processing_patients_completed_task is not None:
                completed_progress.stop_task(processing_patients_completed_task)
            if stopwatch is not None:
                stopwatch.stop()
            plotting_funcs.plot_geometries(reconstructed_biopsy_bx_coord_sys_tr_and_rot_point_cloud, sampled_bx_points_bx_coord_sys_tr_and_rot_pcd, reconstructed_biopsy_bx_coord_sys_tr_and_rot_axis_aligned_bounding_box)
            plotting_funcs.plot_geometries(sampled_bx_points_bx_coord_sys_tr_and_rot_pcd, reconstructed_biopsy_bx_coord_sys_tr_and_rot_axis_aligned_bounding_box)
            plotting_funcs.plotly_3dscatter_arbitrary_number_of_arrays(
                arrays_to_plot_list=[reconstructed_biopsy_bx_coord_sys_tr_and_rot_arr, sampled_bx_points_bx_coord_sys_tr_and_rot_arr],
                colors_for_arrays_list=['red', 'black'],
            )
            if stopwatch is not None:
                stopwatch.start()
            if patients_progress is not None and processing_patients_task is not None:
                patients_progress.start_task(processing_patients_task)
            if completed_progress is not None and processing_patients_completed_task is not None:
                completed_progress.start_task(processing_patients_completed_task)

        sampled_bx_points_bx_coord_sys_tr_and_rot_arr_from_pcd_transform = np.asarray(sampled_bx_points_bx_coord_sys_tr_and_rot_pcd.points)
        del reconstructed_biopsy_bx_coord_sys_tr_from_arr_point_cloud
        del reconstructed_biopsy_bx_coord_sys_tr_and_rot_arr_point_cloud
        del sampled_bx_points_bx_coord_sys_tr_and_rot_arr_from_pcd_transform
        del axis_aligned_bounding_box

        specific_structure["Random uniformly sampled volume pts bx coord sys arr"] = sampled_bx_points_bx_coord_sys_tr_and_rot_arr
        specific_structure["Random uniformly sampled volume pts bx coord sys pcd"] = sampled_bx_points_bx_coord_sys_tr_and_rot_arr_point_cloud
        processed_biopsy_count += 1

    return processed_biopsy_count


def process_patient_sampled_biopsies(*,
                                     patient_uid,
                                     pydicom_item,
                                     bx_ref,
                                     bx_sample_pts_lattice_spacing,
                                     parallel_pool,
                                     stopwatch=None,
                                     show_reconstructed_biopsy_in_biopsy_coord_sys_tr_and_rot=False,
                                     live_display=None,
                                     biopsies_progress=None,
                                     processing_biopsies_task=None,
                                     completed_progress=None,
                                     parsing_sampled_biopsy_data_task=None,
                                     parsing_sampled_biopsy_data_task_completed=None,
                                     patients_progress=None,
                                     processing_patients_task=None,
                                     processing_patients_completed_task=None):
    args_list = build_patient_sampled_biopsy_sampling_args(
        patient_uid=patient_uid,
        pydicom_item=pydicom_item,
        bx_ref=bx_ref,
        bx_sample_pts_lattice_spacing=bx_sample_pts_lattice_spacing,
        biopsies_progress=biopsies_progress,
        processing_biopsies_task=processing_biopsies_task,
    )

    if args_list:
        parallel_results_sampled_bx_points_from_global_delaunay_arr_and_bounding_box_arr = parallel_pool.starmap(
            biopsy_point_sampler.sample_biopsy_points_from_reconstructed_global_delaunay_convex_structure,
            args_list,
        )
    else:
        parallel_results_sampled_bx_points_from_global_delaunay_arr_and_bounding_box_arr = []

    stored_result_count = store_patient_sampled_biopsy_results(
        patient_uid=patient_uid,
        pydicom_item=pydicom_item,
        bx_ref=bx_ref,
        parallel_results_sampled_bx_points_from_global_delaunay_arr_and_bounding_box_arr=parallel_results_sampled_bx_points_from_global_delaunay_arr_and_bounding_box_arr,
        biopsies_progress=biopsies_progress,
        parsing_sampled_biopsy_data_task=parsing_sampled_biopsy_data_task,
        completed_progress=completed_progress,
        parsing_sampled_biopsy_data_task_completed=parsing_sampled_biopsy_data_task_completed,
        stopwatch=stopwatch,
        live_display=live_display,
    )
    processed_biopsy_count = create_patient_biopsy_oriented_coordinate_system(
        patient_uid=patient_uid,
        pydicom_item=pydicom_item,
        bx_ref=bx_ref,
        stopwatch=stopwatch,
        show_reconstructed_biopsy_in_biopsy_coord_sys_tr_and_rot=show_reconstructed_biopsy_in_biopsy_coord_sys_tr_and_rot,
        patients_progress=patients_progress,
        processing_patients_task=processing_patients_task,
        completed_progress=completed_progress,
        processing_patients_completed_task=processing_patients_completed_task,
    )

    return {
        "num_sampling_args": len(args_list),
        "num_sampled_result_fragments": stored_result_count,
        "num_biopsy_coordinate_systems": processed_biopsy_count,
    }


def _build_sampling_args(master_structure_reference_dict,
                         master_structure_info_dict,
                         bx_ref,
                         bx_sample_pts_lattice_spacing,
                         patients_progress,
                         biopsies_progress,
                         completed_progress):
    args_list = []
    patient_uid_default = "Initializing"
    processing_patient_parallel_computing_main_description = "Preparing patient for parallel processing [{}]...".format(patient_uid_default)
    processing_patients_task = patients_progress.add_task("[red]" + processing_patient_parallel_computing_main_description, total=master_structure_info_dict["Global"]["Num cases"])
    processing_patient_parallel_computing_main_description_completed = "Preparing patient for parallel processing"
    processing_patients_completed_task = completed_progress.add_task("[green]" + processing_patient_parallel_computing_main_description_completed, total=master_structure_info_dict["Global"]["Num cases"], visible=False)

    for patient_uid, pydicom_item in master_structure_reference_dict.items():
        processing_patient_parallel_computing_main_description = "Preparing patient for parallel processing [{}]...".format(patient_uid)
        patients_progress.update(processing_patients_task, description="[red]" + processing_patient_parallel_computing_main_description)

        num_biopsies_per_patient = master_structure_info_dict["By patient"][patient_uid][bx_ref]["Num structs"]
        biopsy_id_default = "Initializing"
        processing_biopsies_main_description = "[cyan]Preparing biopsy data for parallel processing [{},{}]...".format(patient_uid, biopsy_id_default)
        processing_biopsies_task = biopsies_progress.add_task(processing_biopsies_main_description, total=num_biopsies_per_patient)

        for specific_structure_index, specific_structure in enumerate(pydicom_item[bx_ref]):
            specific_bx_structure_roi = specific_structure["ROI"]
            processing_biopsies_main_description = "[cyan]Preparing biopsy data for parallel processing [{},{}]...".format(patient_uid, specific_bx_structure_roi)
            biopsies_progress.update(processing_biopsies_task, description=processing_biopsies_main_description)

            reconstructed_biopsy_arr = specific_structure["Reconstructed structure pts arr"]
            reconstructed_delaunay_global_convex_structure_obj = specific_structure["Reconstructed structure delaunay global"]

            z_axis_np_vec = np.array([0, 0, 1], dtype=float)
            apex_to_base_bx_best_fit_vec = specific_structure["Centroid line vec (bx needle base to bx needle tip)"]
            z_axis_to_centroid_vec_rotation_matrix = mf.rotation_matrix_from_vectors(z_axis_np_vec, apex_to_base_bx_best_fit_vec)

            args_list.append(
                (
                    bx_sample_pts_lattice_spacing,
                    reconstructed_delaunay_global_convex_structure_obj.delaunay_triangulation,
                    reconstructed_biopsy_arr,
                    patient_uid,
                    bx_ref,
                    specific_structure_index,
                    z_axis_to_centroid_vec_rotation_matrix,
                )
            )
            biopsies_progress.update(processing_biopsies_task, advance=1)

        biopsies_progress.update(processing_biopsies_task, visible=False)
        patients_progress.update(processing_patients_task, advance=1)
        completed_progress.update(processing_patients_completed_task, advance=1)

    patients_progress.update(processing_patients_task, visible=False)
    completed_progress.update(processing_patients_completed_task, visible=True)

    return args_list


def _store_sampled_biopsy_results(master_structure_reference_dict,
                                  total_num_biopsies,
                                  parallel_results_sampled_bx_points_from_global_delaunay_arr_and_bounding_box_arr,
                                  biopsies_progress,
                                  completed_progress,
                                  stopwatch,
                                  live_display):
    patient_uid_default = "Initializing"
    bx_id_default = "Initializing"
    parsing_sampled_biopsy_data_task_main_description = "Parsing sampled biopsy information [{},{}]".format(patient_uid_default, bx_id_default)
    parsing_sampled_biopsy_data_task_main_description_completed = "Parsing sampled biopsy information"
    parsing_sampled_biopsy_data_task = biopsies_progress.add_task("[red]" + parsing_sampled_biopsy_data_task_main_description, total=total_num_biopsies)
    parsing_sampled_biopsy_data_task_completed = completed_progress.add_task("[green]" + parsing_sampled_biopsy_data_task_main_description_completed, total=total_num_biopsies, visible=False)

    for sampled_bx_pts_arr, axis_aligned_bounding_box_arr, num_sample_pts_in_specific_bx, structure_info_dict in parallel_results_sampled_bx_points_from_global_delaunay_arr_and_bounding_box_arr:
        temp_patient_uid = structure_info_dict["Patient UID"]
        temp_structure_type = structure_info_dict["Structure type"]
        temp_specific_structure_index = structure_info_dict["Specific structure index"]
        temp_structure_id = master_structure_reference_dict[temp_patient_uid][temp_structure_type][temp_specific_structure_index]["ROI"]

        parsing_sampled_biopsy_data_task_main_description = "Parsing sampled biopsy information [{},{}]".format(temp_patient_uid, temp_structure_id)
        biopsies_progress.update(parsing_sampled_biopsy_data_task, description="[red]" + parsing_sampled_biopsy_data_task_main_description, refresh=True)
        live_display.refresh()

        sampled_bx_points_from_global_delaunay_point_cloud_color = np.random.uniform(0, 0.7, size=3)
        sampled_bx_points_from_global_delaunay_point_cloud = point_containment_tools.create_point_cloud(sampled_bx_pts_arr, sampled_bx_points_from_global_delaunay_point_cloud_color)
        axis_aligned_bounding_box = o3d.geometry.AxisAlignedBoundingBox()
        axis_aligned_bounding_box_o3d3dvector_points = o3d.utility.Vector3dVector(axis_aligned_bounding_box_arr)
        axis_aligned_bounding_box = axis_aligned_bounding_box.create_from_points(axis_aligned_bounding_box_o3d3dvector_points)
        axis_aligned_bounding_box.color = np.array([0, 0, 0], dtype=float)

        specific_structure = master_structure_reference_dict[temp_patient_uid][temp_structure_type][temp_specific_structure_index]
        specific_structure["Random uniformly sampled volume pts arr"] = sampled_bx_pts_arr
        specific_structure["Random uniformly sampled volume pts pcd"] = sampled_bx_points_from_global_delaunay_point_cloud
        specific_structure["Bounding box for random uniformly sampled volume pts"] = axis_aligned_bounding_box
        specific_structure["Num sampled bx pts"] = num_sample_pts_in_specific_bx
        reconstructed_bx_pcd = specific_structure["Reconstructed structure point cloud"]

        biopsies_progress.stop_task(parsing_sampled_biopsy_data_task)
        completed_progress.stop_task(parsing_sampled_biopsy_data_task_completed)
        stopwatch.stop()
        #plotting_funcs.plot_geometries(sampled_bx_points_from_global_delaunay_point_cloud, reconstructed_bx_pcd, axis_aligned_bounding_box)
        #plotting_funcs.plot_geometries(sampled_bx_points_from_global_delaunay_point_cloud, reconstructed_bx_pcd)
        del reconstructed_bx_pcd
        stopwatch.start()
        biopsies_progress.start_task(parsing_sampled_biopsy_data_task)
        completed_progress.start_task(parsing_sampled_biopsy_data_task_completed)

        biopsies_progress.update(parsing_sampled_biopsy_data_task, advance=1, refresh=True)
        completed_progress.update(parsing_sampled_biopsy_data_task_completed, advance=1, refresh=True)
        live_display.refresh()

    biopsies_progress.update(parsing_sampled_biopsy_data_task, visible=False, refresh=True)
    completed_progress.update(parsing_sampled_biopsy_data_task_completed, visible=True, refresh=True)
    live_display.refresh()


def _create_biopsy_oriented_coordinate_system(master_structure_reference_dict,
                                              master_structure_info_dict,
                                              bx_ref,
                                              patients_progress,
                                              completed_progress,
                                              stopwatch,
                                              show_reconstructed_biopsy_in_biopsy_coord_sys_tr_and_rot):
    patient_uid_default = "Initializing"
    processing_patient_rotating_bx_main_description = "Creating biopsy oriented coordinate system [{}]...".format(patient_uid_default)
    processing_patients_task = patients_progress.add_task("[red]" + processing_patient_rotating_bx_main_description, total=master_structure_info_dict["Global"]["Num cases"])
    processing_patient_rotating_bx_main_description_completed = "Creating biopsy oriented coordinate system"
    processing_patients_completed_task = completed_progress.add_task("[green]" + processing_patient_rotating_bx_main_description_completed, total=master_structure_info_dict["Global"]["Num cases"], visible=False)

    for patient_uid, pydicom_item in master_structure_reference_dict.items():
        processing_patient_rotating_bx_main_description = "Creating biopsy oriented coordinate system [{}]...".format(patient_uid)
        patients_progress.update(processing_patients_task, description="[red]" + processing_patient_rotating_bx_main_description)

        for specific_structure in pydicom_item[bx_ref]:
            bx_best_fit_line_of_reconstructed_centroids = specific_structure["Best fit line of centroid pts"]
            vec_with_largest_z_val_index = bx_best_fit_line_of_reconstructed_centroids[:, 2].argmax()
            vec_with_largest_z_val = bx_best_fit_line_of_reconstructed_centroids[vec_with_largest_z_val_index, :]
            base_sup_vec_bx_centroid_arr = vec_with_largest_z_val

            vec_with_smallest_z_val_index = bx_best_fit_line_of_reconstructed_centroids[:, 2].argmin()
            vec_with_smallest_z_val = bx_best_fit_line_of_reconstructed_centroids[vec_with_smallest_z_val_index, :]
            apex_inf_vec_bx_centroid_arr = vec_with_smallest_z_val

            translation_vec_bx_coord_sys_origin = -apex_inf_vec_bx_centroid_arr
            apex_to_base_bx_best_fit_vec = specific_structure["Centroid line vec (bx needle base to bx needle tip)"]

            reconstructed_biopsy_point_cloud = specific_structure["Reconstructed structure point cloud"]
            reconstructed_biopsy_arr = specific_structure["Reconstructed structure pts arr"]
            sampled_bx_points_pcd = specific_structure["Random uniformly sampled volume pts pcd"]
            sampled_bx_points_arr = specific_structure["Random uniformly sampled volume pts arr"]
            axis_aligned_bounding_box = specific_structure["Bounding box for random uniformly sampled volume pts"]

            reconstructed_biopsy_bx_coord_sys_tr_arr = reconstructed_biopsy_arr + translation_vec_bx_coord_sys_origin
            sampled_bx_points_bx_coord_sys_tr_arr = sampled_bx_points_arr + translation_vec_bx_coord_sys_origin
            reconstructed_biopsy_bx_coord_sys_tr_point_cloud = copy.copy(reconstructed_biopsy_point_cloud)
            reconstructed_biopsy_bx_coord_sys_tr_from_arr_point_cloud = point_containment_tools.create_point_cloud(reconstructed_biopsy_bx_coord_sys_tr_arr)
            sampled_bx_points_bx_coord_sys_tr_pcd = copy.copy(sampled_bx_points_pcd)

            reconstructed_biopsy_bx_coord_sys_tr_point_cloud.translate(translation_vec_bx_coord_sys_origin)
            sampled_bx_points_bx_coord_sys_tr_pcd.translate(translation_vec_bx_coord_sys_origin)

            patients_progress.stop_task(processing_patients_task)
            completed_progress.stop_task(processing_patients_completed_task)
            stopwatch.stop()
            #plotting_funcs.plot_geometries(sampled_bx_points_pcd, reconstructed_biopsy_point_cloud, axis_aligned_bounding_box, reconstructed_biopsy_bx_coord_sys_tr_point_cloud, sampled_bx_points_bx_coord_sys_tr_pcd)
            stopwatch.start()
            patients_progress.start_task(processing_patients_task)
            completed_progress.start_task(processing_patients_completed_task)

            z_axis_np_vec = np.array([0, 0, 1], dtype=float)
            centroid_line_to_z_axis_rotation_matrix_other = mf.rotation_matrix_from_vectors(apex_to_base_bx_best_fit_vec, z_axis_np_vec)

            reconstructed_biopsy_bx_coord_sys_tr_and_rot_point_cloud = copy.copy(reconstructed_biopsy_bx_coord_sys_tr_point_cloud)
            sampled_bx_points_bx_coord_sys_tr_and_rot_pcd = copy.copy(sampled_bx_points_bx_coord_sys_tr_pcd)

            reconstructed_biopsy_bx_coord_sys_tr_and_rot_point_cloud.rotate(centroid_line_to_z_axis_rotation_matrix_other, center=(0, 0, 0))
            sampled_bx_points_bx_coord_sys_tr_and_rot_pcd.rotate(centroid_line_to_z_axis_rotation_matrix_other, center=(0, 0, 0))

            reconstructed_biopsy_bx_coord_sys_tr_and_rot_arr = (centroid_line_to_z_axis_rotation_matrix_other @ reconstructed_biopsy_bx_coord_sys_tr_arr.T).T
            sampled_bx_points_bx_coord_sys_tr_and_rot_arr = (centroid_line_to_z_axis_rotation_matrix_other @ sampled_bx_points_bx_coord_sys_tr_arr.T).T
            reconstructed_biopsy_bx_coord_sys_tr_and_rot_arr_point_cloud = point_containment_tools.create_point_cloud(reconstructed_biopsy_bx_coord_sys_tr_and_rot_arr)
            sampled_bx_points_bx_coord_sys_tr_and_rot_arr_point_cloud = point_containment_tools.create_point_cloud(sampled_bx_points_bx_coord_sys_tr_and_rot_arr)

            if show_reconstructed_biopsy_in_biopsy_coord_sys_tr_and_rot is True:
                reconstructed_biopsy_bx_coord_sys_tr_and_rot_axis_aligned_bounding_box = reconstructed_biopsy_bx_coord_sys_tr_and_rot_point_cloud.get_axis_aligned_bounding_box()
                reconstructed_biopsy_bx_coord_sys_tr_and_rot_axis_aligned_bounding_box.color = np.array([0, 0, 0], dtype=float)
                patients_progress.stop_task(processing_patients_task)
                completed_progress.stop_task(processing_patients_completed_task)
                stopwatch.stop()
                plotting_funcs.plot_geometries(reconstructed_biopsy_bx_coord_sys_tr_and_rot_point_cloud, sampled_bx_points_bx_coord_sys_tr_and_rot_pcd, reconstructed_biopsy_bx_coord_sys_tr_and_rot_axis_aligned_bounding_box)
                plotting_funcs.plot_geometries(sampled_bx_points_bx_coord_sys_tr_and_rot_pcd, reconstructed_biopsy_bx_coord_sys_tr_and_rot_axis_aligned_bounding_box)
                plotting_funcs.plotly_3dscatter_arbitrary_number_of_arrays(
                    arrays_to_plot_list=[reconstructed_biopsy_bx_coord_sys_tr_and_rot_arr, sampled_bx_points_bx_coord_sys_tr_and_rot_arr],
                    colors_for_arrays_list=['red', 'black'],
                )
                stopwatch.start()
                patients_progress.start_task(processing_patients_task)
                completed_progress.start_task(processing_patients_completed_task)

            sampled_bx_points_bx_coord_sys_tr_and_rot_arr_from_pcd_transform = np.asarray(sampled_bx_points_bx_coord_sys_tr_and_rot_pcd.points)
            del reconstructed_biopsy_bx_coord_sys_tr_from_arr_point_cloud
            del reconstructed_biopsy_bx_coord_sys_tr_and_rot_arr_point_cloud
            del sampled_bx_points_bx_coord_sys_tr_and_rot_arr_from_pcd_transform

            specific_structure["Random uniformly sampled volume pts bx coord sys arr"] = sampled_bx_points_bx_coord_sys_tr_and_rot_arr
            specific_structure["Random uniformly sampled volume pts bx coord sys pcd"] = sampled_bx_points_bx_coord_sys_tr_and_rot_arr_point_cloud

        patients_progress.update(processing_patients_task, advance=1)
        completed_progress.update(processing_patients_completed_task, advance=1)

    patients_progress.update(processing_patients_task, visible=False)
    completed_progress.update(processing_patients_completed_task, visible=True)


def sampled_biopsy_processing_processer(master_structure_reference_dict,
                                        master_structure_info_dict,
                                        bx_ref,
                                        bx_sample_pts_lattice_spacing,
                                        parallel_pool,
                                        indeterminate_progress_main,
                                        patients_progress,
                                        biopsies_progress,
                                        completed_progress,
                                        live_display,
                                        stopwatch,
                                        show_reconstructed_biopsy_in_biopsy_coord_sys_tr_and_rot):
    live_display.stop()
    master_structure_info_dict["Global"]["MC info"]["BX sample pt lattice spacing (mm)"] = bx_sample_pts_lattice_spacing
    master_structure_info_dict["Global"]["MC info"]["BX sample pt volume element (mm^3)"] = bx_sample_pts_lattice_spacing ** 3

    args_list = _build_sampling_args(
        master_structure_reference_dict,
        master_structure_info_dict,
        bx_ref,
        bx_sample_pts_lattice_spacing,
        patients_progress,
        biopsies_progress,
        completed_progress,
    )

    sampling_points_task_indeterminate = indeterminate_progress_main.add_task("[red]Sampling points from all patient biopsies (parallel)...", total=None)
    sampling_points_task_indeterminate_completed = completed_progress.add_task("[green]Sampling points from all patient biopsies (parallel)", visible=False, total=master_structure_info_dict["Global"]["Num cases"])

    parallel_results_sampled_bx_points_from_global_delaunay_arr_and_bounding_box_arr = parallel_pool.starmap(
        biopsy_point_sampler.sample_biopsy_points_from_reconstructed_global_delaunay_convex_structure,
        args_list,
    )

    indeterminate_progress_main.update(sampling_points_task_indeterminate, visible=False, refresh=True)
    completed_progress.update(
        sampling_points_task_indeterminate_completed,
        advance=master_structure_info_dict["Global"]["Num cases"],
        visible=True,
        refresh=True,
    )
    live_display.refresh()

    _store_sampled_biopsy_results(
        master_structure_reference_dict,
        master_structure_info_dict["Global"]["Num biopsies"],
        parallel_results_sampled_bx_points_from_global_delaunay_arr_and_bounding_box_arr,
        biopsies_progress,
        completed_progress,
        stopwatch,
        live_display,
    )
    _create_biopsy_oriented_coordinate_system(
        master_structure_reference_dict,
        master_structure_info_dict,
        bx_ref,
        patients_progress,
        completed_progress,
        stopwatch,
        show_reconstructed_biopsy_in_biopsy_coord_sys_tr_and_rot,
    )

    return live_display