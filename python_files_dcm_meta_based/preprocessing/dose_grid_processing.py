from __future__ import annotations

"""Dose-grid preprocessing adapters.

The dose-grid builder preserves the established mapped array layout: slice, row,
column, physical X/Y/Z, dose, gradient X/Y/Z, gradient norm, and normalized
feature X/Y/Z. The wrappers here only move main-facing orchestration; they do
not change the dose calculation.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

import dose_lattice_helper_funcs
import plotting_funcs


@dataclass(frozen=True)
class DoseGridProcessingConfig:
    dose_ref: str
    plan_ref: str
    lower_bound_dose_value: Any
    lower_bound_dose_gradient_value: Any
    show_3d_dose_renderings: bool
    show_3d_dose_renderings_thresholded: bool


@dataclass(frozen=True)
class DoseGridProcessingResult:
    lower_bound_dose_value: Any


def build_dose_grid_runtime_objects_for_patient(
    pydicom_item,
    config,
    patients_progress,
    completed_progress,
    processing_patients_dose_task,
    processing_patients_dose_task_completed,
    stopwatch,
):
    """Build and store dose-grid runtime objects for one patient.

    This is a behavior-preserving adapter around the former main-body dose-grid
    block. It writes the same dictionary keys and keeps the same dose,
    gradient, threshold, and optional debug-rendering calls.
    """
    dose_ref_dict = pydicom_item[config.dose_ref]
    conversion_matrix = np.array([
        [dose_ref_dict["Image orientation patient"][0] * dose_ref_dict["Pixel spacing"][1],
        dose_ref_dict["Image orientation patient"][3] * dose_ref_dict["Pixel spacing"][0],
        0, dose_ref_dict["Image position patient"][0]],
        [dose_ref_dict["Image orientation patient"][1] * dose_ref_dict["Pixel spacing"][1],
        dose_ref_dict["Image orientation patient"][4] * dose_ref_dict["Pixel spacing"][0],
        0, dose_ref_dict["Image position patient"][1]],
        [dose_ref_dict["Image orientation patient"][2] * dose_ref_dict["Pixel spacing"][1],
        dose_ref_dict["Image orientation patient"][5] * dose_ref_dict["Pixel spacing"][0],
        0, dose_ref_dict["Image position patient"][2]],
        [0, 0, 0, 1]
    ])

    phys_space_dose_map_3d_arr = dose_lattice_helper_funcs.build_dose_grid(
        dose_pixel_slices=dose_ref_dict["Dose pixel arr"],
        scaling_factor=dose_ref_dict["Dose grid scaling"],
        conversion_matrix=conversion_matrix,
        grid_frame_offset_vec_list=dose_ref_dict["Grid frame offset vector"]
    )

    ### DOSE GRADIENT

    # Scale the dose values before computing gradients
    scaled_dose_data = dose_ref_dict["Dose pixel arr"] * dose_ref_dict["Dose grid scaling"]

    gradient_vector_lattice, gradient_norm_lattice, normalized_gradient_vector_lattice = dose_lattice_helper_funcs.calculate_gradient_lattices(scaled_dose_data, dose_ref_dict["Pixel spacing"], dose_ref_dict["Grid frame offset vector"])

    phys_space_dose_map_and_gradient_map_3d_arr = dose_lattice_helper_funcs.map_gradient_to_physical_space(
        phys_space_dose_map_3d_arr=phys_space_dose_map_3d_arr,
        gradient_vector_lattice=gradient_vector_lattice,
        gradient_norm_lattice=gradient_norm_lattice,
        normalized_gradient_vector_lattice = normalized_gradient_vector_lattice
    )

    dose_point_cloud, dose_gradient_arrows_point_cloud = plotting_funcs.create_dose_point_cloud_with_gradients(phys_space_dose_map_and_gradient_map_3d_arr,
                                                                                                        paint_dose_color=True,
                                                                                                        arrow_scale=1.0,
                                                                                                        truncate_below_dose=None,
                                                                                                        truncate_below_gradient_norm=None
                                                                                                    )
    if config.show_3d_dose_renderings == True:
        patients_progress.stop_task(processing_patients_dose_task)
        completed_progress.stop_task(processing_patients_dose_task_completed)
        stopwatch.stop()
        plotting_funcs.plot_geometries(dose_point_cloud, dose_gradient_arrows_point_cloud)
        stopwatch.start()
        patients_progress.start_task(processing_patients_dose_task)
        completed_progress.start_task(processing_patients_dose_task_completed)

    lower_bound_dose_value = config.lower_bound_dose_value
    if lower_bound_dose_value == None:
        try:
            lower_bound_dose_value = pydicom_item[config.plan_ref]["Prescription doses dict"]["TARGET"]
        except Exception as e:
            lower_bound_dose_value = 0

    thresholded_dose_point_cloud, thresholded_dose_gradient_arrows_point_cloud = plotting_funcs.create_dose_point_cloud_with_gradients(phys_space_dose_map_and_gradient_map_3d_arr,
                                                                                                        paint_dose_color=True,
                                                                                                        arrow_scale=1.0,
                                                                                                        truncate_below_dose=lower_bound_dose_value,
                                                                                                        truncate_below_gradient_norm=config.lower_bound_dose_gradient_value
                                                                                                    )

    # plot dose point cloud thresholded cubic lattice (color only)
    if config.show_3d_dose_renderings_thresholded == True:
        patients_progress.stop_task(processing_patients_dose_task)
        completed_progress.stop_task(processing_patients_dose_task_completed)
        stopwatch.stop()
        plotting_funcs.plot_geometries(thresholded_dose_point_cloud, thresholded_dose_gradient_arrows_point_cloud)
        stopwatch.start()
        patients_progress.start_task(processing_patients_dose_task)
        completed_progress.start_task(processing_patients_dose_task_completed)

    dose_ref_dict["Dose and gradient phys space and pixel 3d arr"] = phys_space_dose_map_and_gradient_map_3d_arr
    #dose_ref_dict["Dose phys space and pixel 3d arr"] = phys_space_dose_map_3d_arr
    dose_ref_dict["Dose grid point cloud"] = dose_point_cloud
    dose_ref_dict["Dose grid point cloud thresholded"] = thresholded_dose_point_cloud
    dose_ref_dict["Dose grid gradient point cloud"] = dose_gradient_arrows_point_cloud
    dose_ref_dict["Dose grid gradient point cloud thresholded"] = thresholded_dose_gradient_arrows_point_cloud

    return lower_bound_dose_value


def build_dose_grids_for_cohort(
    master_structure_reference_dict,
    master_structure_info_dict,
    config,
    patients_progress,
    completed_progress,
    stopwatch,
):
    """Run the main-facing dose-grid preprocessing block."""
    lower_bound_dose_value = config.lower_bound_dose_value

    patientUID_default = "Initializing"
    processing_patients_dose_task_main_description = "[red]Building dose grids [{}]...".format(patientUID_default)
    processing_patients_dose_task_completed_main_description = "[green]Building dose grids"

    processing_patients_dose_task = patients_progress.add_task(processing_patients_dose_task_main_description, total=master_structure_info_dict["Global"]["Num cases"])
    processing_patients_dose_task_completed = completed_progress.add_task(processing_patients_dose_task_completed_main_description, total=master_structure_info_dict["Global"]["Num cases"], visible=False)

    # Main loop for processing patients
    for patientUID, pydicom_item in master_structure_reference_dict.items():
        processing_patients_dose_task_main_description = "[red]Building dose grids [{}]...".format(patientUID)
        patients_progress.update(processing_patients_dose_task, description=processing_patients_dose_task_main_description)

        if config.dose_ref not in pydicom_item:
            patients_progress.update(processing_patients_dose_task, advance=1)
            completed_progress.update(processing_patients_dose_task_completed, advance=1)
            continue

        config_for_patient = DoseGridProcessingConfig(
            dose_ref=config.dose_ref,
            plan_ref=config.plan_ref,
            lower_bound_dose_value=lower_bound_dose_value,
            lower_bound_dose_gradient_value=config.lower_bound_dose_gradient_value,
            show_3d_dose_renderings=config.show_3d_dose_renderings,
            show_3d_dose_renderings_thresholded=config.show_3d_dose_renderings_thresholded,
        )
        lower_bound_dose_value = build_dose_grid_runtime_objects_for_patient(
            pydicom_item,
            config_for_patient,
            patients_progress,
            completed_progress,
            processing_patients_dose_task,
            processing_patients_dose_task_completed,
            stopwatch,
        )

        # Update progress
        patients_progress.update(processing_patients_dose_task, advance=1)
        completed_progress.update(processing_patients_dose_task_completed, advance=1)

    # Finalize progress display
    patients_progress.update(processing_patients_dose_task, visible=False)
    completed_progress.update(processing_patients_dose_task_completed, visible=True)

    return DoseGridProcessingResult(
        lower_bound_dose_value=lower_bound_dose_value,
    )
