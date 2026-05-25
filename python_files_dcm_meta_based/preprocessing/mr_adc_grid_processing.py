from __future__ import annotations

"""MR ADC grid preprocessing adapters.

These wrappers preserve the current ADC lattice reconstruction and point-cloud
side effects while making MR mapping independently callable from dose mapping.
They do not change ADC scaling, filtering, zero handling, series selection, or
threshold policy.
"""

from dataclasses import dataclass
from typing import Any

import lattice_reconstruction_tools
import plotting_funcs


@dataclass(frozen=True)
class MRADCGridProcessingConfig:
    mr_adc_ref: str
    color_flattening_deg_mr: Any
    lower_bound_mr_adc_value: Any
    upper_bound_mr_adc_value: Any
    show_3d_mr_adc_renderings: bool
    show_3d_mr_adc_renderings_thresholded: bool


@dataclass(frozen=True)
class MRADCGridProcessingResult:
    no_cohort_mr_adc_flag: bool


def _cohort_has_mr_adc(master_structure_reference_dict, mr_adc_ref):
    for patientUID, pydicom_item in master_structure_reference_dict.items():
        if mr_adc_ref in pydicom_item:
            return True
    return False


def build_mr_adc_grid_runtime_objects_for_patient(
    patientUID,
    pydicom_item,
    config,
    patients_progress,
    completed_progress,
    processing_patients_adc_mr_task,
    processing_patients_adc_mr_task_completed,
    stopwatch,
):
    """Build and store MR ADC grid point-cloud runtime objects.

    This wrapper intentionally preserves the current ADC lattice reconstruction
    call, including `filter_out_negatives=True`.
    """
    mr_adc_subdict = pydicom_item[config.mr_adc_ref]

    filtered_non_negative_adc_mr_phys_space_arr = lattice_reconstruction_tools.reconstruct_mr_lattice_with_coordinates_from_dict_v2(mr_adc_subdict, filter_out_negatives = True)
    # Don't store this, it is too large, just call the above function if you want to retrieve the MR information lattice
    #mr_adc_subdict["MR ADC phys space Nx4 arr (filtered, non-negative)"] = filtered_non_negative_adc_mr_phys_space_arr

    mr_adc_point_cloud = plotting_funcs.create_MR_point_cloud(filtered_non_negative_adc_mr_phys_space_arr,
                                                                    config.color_flattening_deg_mr,
                                                                    paint_mr_color = True)

    thresholded_mr_adc_point_cloud = plotting_funcs.create_thresholded_MR_ADC_point_cloud(filtered_non_negative_adc_mr_phys_space_arr,
                                                                                                config.color_flattening_deg_mr,
                                                                                                paint_mr_color = True,
                                                                                                lower_bound = config.lower_bound_mr_adc_value,
                                                                                                upper_bound = config.upper_bound_mr_adc_value,
                                                                                                z_val_range_list = None)

    del filtered_non_negative_adc_mr_phys_space_arr

    if config.show_3d_mr_adc_renderings == True:
        patients_progress.stop_task(processing_patients_adc_mr_task)
        completed_progress.stop_task(processing_patients_adc_mr_task_completed)
        stopwatch.stop()
        print(f"MR ADC render: {patientUID}")
        plotting_funcs.plot_geometries(mr_adc_point_cloud)
        stopwatch.start()
        patients_progress.start_task(processing_patients_adc_mr_task)
        completed_progress.start_task(processing_patients_adc_mr_task_completed)

    # plot dose point cloud thresholded cubic lattice (color only)
    if config.show_3d_mr_adc_renderings_thresholded == True:
        patients_progress.stop_task(processing_patients_adc_mr_task)
        completed_progress.stop_task(processing_patients_adc_mr_task_completed)
        stopwatch.stop()
        print(f"MR ADC render (tresholded): {patientUID}")
        plotting_funcs.plot_geometries(thresholded_mr_adc_point_cloud)
        stopwatch.start()
        patients_progress.start_task(processing_patients_adc_mr_task)
        completed_progress.start_task(processing_patients_adc_mr_task_completed)

    # Store computed objects
    mr_adc_subdict["MR ADC grid point cloud"] = mr_adc_point_cloud
    mr_adc_subdict["MR ADC grid point cloud thresholded"] = thresholded_mr_adc_point_cloud


def build_mr_adc_grids_for_cohort(
    master_structure_reference_dict,
    master_structure_info_dict,
    config,
    patients_progress,
    completed_progress,
    stopwatch,
):
    """Run the main-facing ADC MR grid preprocessing block."""
    no_cohort_mr_adc_flag = not _cohort_has_mr_adc(
        master_structure_reference_dict,
        config.mr_adc_ref,
    )

    patientUID_default = "Initializing"
    processing_patients_adc_mr_task_main_description = "[red]Building ADC MR grids [{}]...".format(patientUID_default)
    processing_patients_adc_mr_task_completed_main_description = "[green]Building ADC MR grids"

    processing_patients_adc_mr_task = patients_progress.add_task(processing_patients_adc_mr_task_main_description, total=master_structure_info_dict["Global"]["Num cases"])
    processing_patients_adc_mr_task_completed = completed_progress.add_task(processing_patients_adc_mr_task_completed_main_description, total=master_structure_info_dict["Global"]["Num cases"], visible=False)

    for patientUID,pydicom_item in master_structure_reference_dict.items():
        processing_patients_adc_mr_task_main_description = "[red]Building ADC MR grids [{}]...".format(patientUID)
        patients_progress.update(processing_patients_adc_mr_task, description=processing_patients_adc_mr_task_main_description)

        if config.mr_adc_ref not in pydicom_item:
            patients_progress.update(processing_patients_adc_mr_task, advance=1)
            completed_progress.update(processing_patients_adc_mr_task_completed, advance=1)
            continue

        build_mr_adc_grid_runtime_objects_for_patient(
            patientUID,
            pydicom_item,
            config,
            patients_progress,
            completed_progress,
            processing_patients_adc_mr_task,
            processing_patients_adc_mr_task_completed,
            stopwatch,
        )

        patients_progress.update(processing_patients_adc_mr_task, advance=1)
        completed_progress.update(processing_patients_adc_mr_task_completed, advance=1)

    patients_progress.update(processing_patients_adc_mr_task, visible=False)
    completed_progress.update(processing_patients_adc_mr_task_completed, visible=True)

    return MRADCGridProcessingResult(
        no_cohort_mr_adc_flag=no_cohort_mr_adc_flag,
    )
