"""Patient-local optimizer-v1 scientific stage."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import math
from typing import Any, Mapping

import numpy as np
import pandas

import MC_simulator_convex
import custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p
import dataframe_builders
import dataframe_dtype_policy
import misc_tools
import point_containment_tools
from biopsy_optimizer.v1 import biopsy_optimizer_module_v1_helpers
from presentation import LegacyNullProgress
from presentation import LegacyPresentationContext
from random_seed_policy import build_optimizer_v1_patient_rng

from .legacy_adapter import OptimizerV1LegacyConfig
from .legacy_adapter import build_patient_info_from_reference
from .legacy_adapter import build_single_patient_master_structure_info
from .legacy_adapter import collect_optimizer_v1_patient_outputs


@dataclass(slots=True)
class OptimizerV1PatientStageResult:
    """Output bundle from the patient-local optimizer-v1 stage."""

    patient_uid: str
    patient_reference_dict: dict[str, Any]
    master_structure_info_dict: dict[str, Any]
    optimizer_outputs: dict[str, Any]
    dil_count: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.patient_uid = str(self.patient_uid)
        self.optimizer_outputs = dict(self.optimizer_outputs or {})
        self.dil_count = int(self.dil_count)
        self.metadata = dict(self.metadata or {})


def _build_optimizer_v1_null_presentation_context() -> LegacyPresentationContext:
    context = LegacyPresentationContext.null()
    if context.layout_groups is None:
        mc_trial_progress = LegacyNullProgress()
        progress_group_info_list = [
            context.completed_progress,
            context.completed_sections_progress,
            context.patients_progress,
            context.structures_progress,
            context.biopsies_progress,
            mc_trial_progress,
            context.indeterminate_progress_main,
            context.indeterminate_progress_sub,
            None,
        ]
        context.layout_groups = (None, progress_group_info_list, context.important_info, None)
    return context


def _reject_optimizer_v1_stage_side_effect_options(config: OptimizerV1LegacyConfig) -> None:
    side_effect_options = {
        "plot_each_normal_dist_containment_result_bool": config.plot_each_normal_dist_containment_result_bool,
        "plot_optimization_point_lattice_bool": config.plot_optimization_point_lattice_bool,
        "show_optimization_point_bool": config.show_optimization_point_bool,
        "demonstrate_dil_optimization_points_inside_correctness_bool_1": (
            config.demonstrate_dil_optimization_points_inside_correctness_bool_1
        ),
        "demonstrate_dil_optimization_points_inside_correctness_bool_2": (
            config.demonstrate_dil_optimization_points_inside_correctness_bool_2
        ),
        "generate_cuda_log_files_biopsy_optimizer": config.generate_cuda_log_files_biopsy_optimizer,
        "display_optimization_contour_plots_bool": config.display_optimization_contour_plots_bool,
    }
    requested_options = [name for name, requested in side_effect_options.items() if bool(requested)]
    if requested_options:
        raise ValueError(
            "Optimizer-v1 patient scientific stage does not run plotting/debug/log side effects; "
            "use the legacy adapter for oracle/debug validation. Requested options: "
            + ", ".join(requested_options)
        )


def _resolve_selected_prostate(patient_uid: str,
                               patient_reference_dict: Mapping[str, Any],
                               config: OptimizerV1LegacyConfig,
                               context: LegacyPresentationContext) -> tuple[dict[str, Any], np.ndarray]:
    selected_structures_dataframe = patient_reference_dict[config.all_ref_key][
        "Multi-structure pre-processing output dataframes dict"
    ]["Selected structures"]
    specific_prostate_info_dataframe = selected_structures_dataframe[
        selected_structures_dataframe["Struct ref type"] == config.oar_ref
    ]
    selected_prostate_info = specific_prostate_info_dataframe.to_dict("records")[0]
    if selected_prostate_info["Struct found bool"] == True:
        prostate_centroid = patient_reference_dict[selected_prostate_info["Struct ref type"]][
            selected_prostate_info["Index number"]
        ]["Structure global centroid"].reshape(3)
    else:
        context.important_info.add_text_line(
            f"Patient {patient_uid}: prostate not found; defaulting prostate centroid to zero-vector.",
            context.live_display,
        )
        prostate_centroid = np.array([0, 0, 0])
    return selected_prostate_info, prostate_centroid


def _build_patient_optimizer_lattice(patient_reference_dict: Mapping[str, Any],
                                     config: OptimizerV1LegacyConfig) -> np.ndarray:
    list_of_all_dils_interpolated_pts = []
    for specific_dil_structure in patient_reference_dict[config.dil_ref]:
        interslice_interpolation_information = specific_dil_structure["Inter-slice interpolation information"]
        list_of_all_dils_interpolated_pts.append(interslice_interpolation_information.interpolated_pts_np_arr)

    list_of_all_oar_interpolated_pts = []
    for specific_oar_structure in patient_reference_dict[config.oar_ref]:
        interslice_interpolation_information = specific_oar_structure["Inter-slice interpolation information"]
        list_of_all_oar_interpolated_pts.append(interslice_interpolation_information.interpolated_pts_np_arr)

    all_geometries_interpolated_pts = np.vstack(
        list_of_all_dils_interpolated_pts + list_of_all_oar_interpolated_pts
    )
    all_geometries_point_cloud = point_containment_tools.create_point_cloud(all_geometries_interpolated_pts)
    all_geometries_point_cloud.paint_uniform_color(np.array([0, 0, 1]))
    all_geometries_axis_aligned_bounding_box = all_geometries_point_cloud.get_axis_aligned_bounding_box()
    all_geometries_bounding_box_points = np.asarray(all_geometries_axis_aligned_bounding_box.get_box_points())
    all_geometries_axis_aligned_bounding_box.color = np.array([0, 0, 0], dtype=float)
    all_geometries_max_bounds = np.amax(all_geometries_bounding_box_points, axis=0)
    all_geometries_min_bounds = np.amin(all_geometries_bounding_box_points, axis=0)

    lattice_sizex = int(
        math.ceil(abs(all_geometries_max_bounds[0] - all_geometries_min_bounds[0]) / config.voxel_size_for_dil_optimizer_grid)
        + 1
    )
    lattice_sizey = int(
        math.ceil(abs(all_geometries_max_bounds[1] - all_geometries_min_bounds[1]) / config.voxel_size_for_dil_optimizer_grid)
        + 1
    )
    lattice_sizez = int(
        math.ceil(abs(all_geometries_max_bounds[2] - all_geometries_min_bounds[2]) / config.voxel_size_for_dil_optimizer_grid)
        + 1
    )
    return MC_simulator_convex.generate_cubic_lattice(
        config.voxel_size_for_dil_optimizer_grid,
        lattice_sizex,
        lattice_sizey,
        lattice_sizez,
        all_geometries_min_bounds,
    )


def _assign_optimizer_plane_indices(dataframe: pandas.DataFrame,
                                    config: OptimizerV1LegacyConfig) -> pandas.DataFrame:
    return misc_tools.assign_plane_indices(
        dataframe,
        config.voxel_size_for_dil_optimizer_grid,
        "Test location (Prostate centroid origin) (X)",
        "Test location (Prostate centroid origin) (Y)",
        "Test location (Prostate centroid origin) (Z)",
    )


def _run_patient_dil_optimizer(*,
                               patient_uid: str,
                               specific_dil_structure: dict[str, Any],
                               all_geometries_centered_cubic_lattice_arr: np.ndarray,
                               selected_prostate_info: Mapping[str, Any],
                               prostate_centroid: np.ndarray,
                               config: OptimizerV1LegacyConfig,
                               context: LegacyPresentationContext,
                               rng: Any = None) -> None:
    structure_id_dil = specific_dil_structure["ROI"]
    structure_info = misc_tools.specific_structure_info_dict_creator(
        "given",
        specific_structure=specific_dil_structure,
    )
    interslice_interpolation_information = specific_dil_structure["Inter-slice interpolation information"]
    interpolated_pts_np_arr = interslice_interpolation_information.interpolated_pts_np_arr
    interpolated_zvals_list = interslice_interpolation_information.zslice_vals_after_interpolation_list
    zslices_list = interslice_interpolation_information.interpolated_pts_list
    dil_global_centroid = specific_dil_structure["Structure global centroid"]

    containment_result_for_all_lattice_points_cp_arr, prepper_output_tuple = (
        custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p.custom_point_containment_mother_function(
            [zslices_list],
            all_geometries_centered_cubic_lattice_arr[np.newaxis, :, :],
            np.array([0]),
            constant_z_slice_polygons_handler_option=config.constant_z_slice_polygons_handler_option,
            remove_consecutive_duplicate_points_in_polygons=config.remove_consecutive_duplicate_points_in_polygons,
            log_sub_dirs_list=[patient_uid, structure_id_dil],
            log_file_name=None,
            include_edges_in_log=config.include_edges_in_log_files,
            kernel_type=config.custom_cuda_kernel_type,
        )
    )
    containment_info_for_all_lattice_points_dataframe = (
        custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p.create_containment_results_dataframe_type_2I(
            structure_info,
            prepper_output_tuple[0],
            all_geometries_centered_cubic_lattice_arr[np.newaxis, :, :],
            containment_result_for_all_lattice_points_cp_arr,
            do_not_convert_column_names_to_categorical=["Pt contained bool"],
            float_dtype=np.float32,
            int_dtype=np.int32,
        )
    )

    contained_lattice_points_dataframe = containment_info_for_all_lattice_points_dataframe.drop(
        containment_info_for_all_lattice_points_dataframe[
            containment_info_for_all_lattice_points_dataframe["Pt contained bool"] == False
        ].index
    ).reset_index()
    not_contained_lattice_points_dataframe = containment_info_for_all_lattice_points_dataframe.drop(
        containment_info_for_all_lattice_points_dataframe[
            containment_info_for_all_lattice_points_dataframe["Pt contained bool"] == True
        ].index
    ).reset_index()
    del containment_info_for_all_lattice_points_dataframe

    contained_lattice_points_arr = all_geometries_centered_cubic_lattice_arr[
        contained_lattice_points_dataframe["index"].to_numpy()
    ]
    not_contained_lattice_points_arr = all_geometries_centered_cubic_lattice_arr[
        not_contained_lattice_points_dataframe["index"].to_numpy()
    ]
    del contained_lattice_points_dataframe

    optimal_locations_dataframe, potential_optimal_locations_dataframe, zero_locations_dataframe, live_display = (
        biopsy_optimizer_module_v1_helpers.find_dil_optimal_sampling_position(
            specific_dil_structure,
            config.optimal_normal_dist_option,
            config.bias_LR_multiplier,
            config.bias_AP_multiplier,
            config.bias_SI_multiplier,
            patient_uid,
            config.structs_referenced_dict,
            config.bx_ref,
            config.dil_ref,
            interpolated_pts_np_arr,
            interpolated_zvals_list,
            zslices_list,
            structure_info,
            dil_global_centroid,
            config.voxel_size_for_dil_optimizer_grid,
            config.num_normal_dist_points_for_biopsy_optimizer,
            config.normal_dist_sigma_factor_biopsy_optimizer,
            prostate_centroid,
            selected_prostate_info,
            config.plot_each_normal_dist_containment_result_bool,
            config.plot_optimization_point_lattice_bool,
            config.show_optimization_point_bool,
            context.layout_groups,
            context.live_display,
            config.cupy_array_upper_limit_NxN_size_input,
            config.numpy_array_upper_limit_NxN_size_input,
            config.nearest_zslice_vals_and_indices_cupy_generic_max_size,
            config.nearest_zslice_vals_and_indices_numpy_generic_max_size,
            context.structures_progress,
            config.constant_z_slice_polygons_handler_option,
            config.remove_consecutive_duplicate_points_in_polygons,
            config.include_edges_in_log_files,
            config.custom_cuda_kernel_type,
            config.demonstrate_dil_optimization_points_inside_correctness_bool_2,
            config.demonstrate_dil_optimization_points_inside_correctness_num_3,
            config.generate_cuda_log_files_biopsy_optimizer,
            test_lattice_arr=contained_lattice_points_arr,
            all_points_to_set_to_zero_arr=not_contained_lattice_points_arr,
            rng=rng,
        )
    )
    context.live_display = live_display

    potential_optimal_locations_dataframe = _assign_optimizer_plane_indices(
        potential_optimal_locations_dataframe,
        config,
    )
    zero_locations_dataframe = _assign_optimizer_plane_indices(zero_locations_dataframe, config)
    optimal_locations_dataframe = _assign_optimizer_plane_indices(optimal_locations_dataframe, config)
    context.live_display.refresh()

    dil_centroids_optimization_locations_dataframe = pandas.DataFrame(
        potential_optimal_locations_dataframe.loc[[0], :]
    )
    dil_centroids_optimization_locations_dataframe = _assign_optimizer_plane_indices(
        dil_centroids_optimization_locations_dataframe,
        config,
    )

    dil_centroids_optimization_locations_dataframe = _downcast_optimizer_v1_location_dataframe(
        dil_centroids_optimization_locations_dataframe
    )
    optimal_locations_dataframe = _downcast_optimizer_v1_location_dataframe(optimal_locations_dataframe)
    potential_optimal_locations_dataframe = _downcast_optimizer_v1_location_dataframe(potential_optimal_locations_dataframe)
    zero_locations_dataframe = _downcast_optimizer_v1_location_dataframe(zero_locations_dataframe)

    specific_dil_structure["Biopsy optimization: DIL centroid optimal biopsy location dataframe"] = (
        dil_centroids_optimization_locations_dataframe
    )
    specific_dil_structure["Biopsy optimization: Optimal biopsy location dataframe"] = optimal_locations_dataframe
    specific_dil_structure["Biopsy optimization: Optimal biopsy location (all tested lattice points) dataframe"] = (
        potential_optimal_locations_dataframe
    )
    specific_dil_structure["Biopsy optimization: Optimal biopsy location (zero lattice) dataframe"] = zero_locations_dataframe
    specific_dil_structure["Biopsy optimization: cubic lattice of optimization points only in dil"] = contained_lattice_points_arr


def _downcast_optimizer_v1_location_dataframe(dataframe: pandas.DataFrame) -> pandas.DataFrame:
    return dataframe_builders.convert_columns_to_categorical_and_downcast(
        dataframe,
        threshold=0.25,
        ignore_types=(np.floating,),
        do_not_convert_column_names_to_categorical=(
            dataframe_dtype_policy.OPTIMIZER_V1_LOCATION_NEVER_CATEGORICAL_COLUMNS
        ),
    )


def _store_patient_guidance_and_lattice_outputs(*,
                                                patient_uid: str,
                                                patient_reference_dict: dict[str, Any],
                                                config: OptimizerV1LegacyConfig,
                                                context: LegacyPresentationContext) -> None:
    for specific_dil_structure in patient_reference_dict[config.dil_ref]:
        structure_id_dil = specific_dil_structure["ROI"]
        potential_optimal_locations_dataframe = specific_dil_structure[
            "Biopsy optimization: Optimal biopsy location (all tested lattice points) dataframe"
        ]
        zero_locations_dataframe = specific_dil_structure[
            "Biopsy optimization: Optimal biopsy location (zero lattice) dataframe"
        ]
        potential_optimal_locations_dataframe_centroid_dropped = potential_optimal_locations_dataframe.drop([0])
        sp_dil_optimal_locations_dataframe = specific_dil_structure[
            "Biopsy optimization: Optimal biopsy location dataframe"
        ]
        guidance_map_max_planes_dataframe = biopsy_optimizer_module_v1_helpers.guidance_map_max_planes_dataframe(
            potential_optimal_locations_dataframe_centroid_dropped,
            sp_dil_optimal_locations_dataframe,
            config.voxel_size_for_dil_optimizer_grid,
            zero_locations_dataframe,
            structure_id_dil,
            patient_uid,
            context.important_info,
            context.live_display,
        )
        guidance_map_max_planes_dataframe = dataframe_builders.convert_columns_to_categorical_and_downcast(
            guidance_map_max_planes_dataframe,
            threshold=0.25,
            ignore_types=(np.floating,),
            do_not_convert_column_names_to_categorical=(
                dataframe_dtype_policy.OPTIMIZER_V1_GUIDANCE_MAP_MAX_PLANES_NEVER_CATEGORICAL_COLUMNS
            ),
        )
        specific_dil_structure["Biopsy optimization: guidance map max-planes dataframe"] = guidance_map_max_planes_dataframe

    all_zero_locations_dataframe_list = []
    all_potential_centroid_dropped_dataframe_list = []
    for specific_dil_structure in patient_reference_dict[config.dil_ref]:
        all_zero_locations_dataframe_list.append(
            specific_dil_structure["Biopsy optimization: Optimal biopsy location (zero lattice) dataframe"]
        )
        potential_optimal_locations_dataframe = specific_dil_structure[
            "Biopsy optimization: Optimal biopsy location (all tested lattice points) dataframe"
        ]
        all_potential_centroid_dropped_dataframe_list.append(potential_optimal_locations_dataframe.drop([0]))

    all_potential_centroid_dropped_dataframe = pandas.concat(
        all_potential_centroid_dropped_dataframe_list,
        ignore_index=True,
    )
    all_zero_locations_dataframe = misc_tools.intersect_dataframes(all_zero_locations_dataframe_list)
    entire_overlapped_lattice_dataframe = pandas.concat(
        [all_potential_centroid_dropped_dataframe, all_zero_locations_dataframe],
        ignore_index=True,
    )
    cumulative_projection_dataframe = (
        biopsy_optimizer_module_v1_helpers.guidance_map_cumulative_projection_dataframe_creator(
            entire_overlapped_lattice_dataframe
        )
    )

    multi_structure_information_dict = patient_reference_dict[config.all_ref_key][
        "Multi-structure information dict (not for csv output)"
    ]
    preprocessing_output_dataframes_dict = patient_reference_dict[config.all_ref_key][
        "Multi-structure pre-processing output dataframes dict"
    ]
    multi_structure_information_dict["Biopsy optimization: All points outside of DILs (zero points) dataframe"] = (
        _downcast_optimizer_v1_location_dataframe(all_zero_locations_dataframe)
    )
    multi_structure_information_dict["Biopsy optimization: All points within DILs (tested points) dataframe"] = (
        _downcast_optimizer_v1_location_dataframe(all_potential_centroid_dropped_dataframe)
    )
    multi_structure_information_dict["Biopsy optimization: Optimal biopsy location (entire cubic lattice) dataframe"] = (
        _downcast_optimizer_v1_location_dataframe(entire_overlapped_lattice_dataframe)
    )
    preprocessing_output_dataframes_dict[
        "Biopsy optimization - Cumulative projection (all points within prostate) dataframe"
    ] = dataframe_builders.convert_columns_to_categorical_and_downcast(
        cumulative_projection_dataframe,
        threshold=0.25,
        ignore_types=(np.floating,),
        do_not_convert_column_names_to_categorical=(
            dataframe_dtype_policy.OPTIMIZER_V1_CUMULATIVE_PROJECTION_NEVER_CATEGORICAL_COLUMNS
        ),
    )


def run_patient_optimizer_v1_stage(
    *,
    patient_uid: str,
    patient_reference_dict: dict[str, Any],
    patient_info_dict: Mapping[str, Any] | None,
    config: OptimizerV1LegacyConfig,
    mutate_input: bool = True,
) -> OptimizerV1PatientStageResult:
    """Run optimizer-v1 for one patient without routing through the cohort oracle."""
    _reject_optimizer_v1_stage_side_effect_options(config)
    resolved_patient_uid = str(patient_uid)
    working_patient_reference_dict = patient_reference_dict if mutate_input else copy.deepcopy(patient_reference_dict)
    if patient_info_dict is None:
        working_patient_info_dict = build_patient_info_from_reference(
            resolved_patient_uid,
            working_patient_reference_dict,
            bx_ref=config.bx_ref,
            dil_ref=config.dil_ref,
            oar_ref=config.oar_ref,
            all_ref_key=config.all_ref_key,
        )
    else:
        working_patient_info_dict = copy.deepcopy(dict(patient_info_dict))
    master_structure_info_dict = build_single_patient_master_structure_info(
        resolved_patient_uid,
        working_patient_info_dict,
        bx_ref=config.bx_ref,
        dil_ref=config.dil_ref,
        all_ref_key=config.all_ref_key,
    )
    context = _build_optimizer_v1_null_presentation_context()
    selected_prostate_info, prostate_centroid = _resolve_selected_prostate(
        resolved_patient_uid,
        working_patient_reference_dict,
        config,
        context,
    )
    all_geometries_centered_cubic_lattice_arr = _build_patient_optimizer_lattice(
        working_patient_reference_dict,
        config,
    )
    optimizer_v1_rng, optimizer_v1_seed_metadata = build_optimizer_v1_patient_rng(
        master_structure_info_dict,
        resolved_patient_uid,
        optimizer_v1_random_seed=config.optimizer_v1_random_seed,
    )

    structure_task = context.structures_progress.add_task(
        f"[cyan]Processing optimizer-v1 DIL structures [{resolved_patient_uid}]...",
        total=len(working_patient_reference_dict[config.dil_ref]),
    )
    for specific_dil_structure in working_patient_reference_dict[config.dil_ref]:
        context.structures_progress.update(
            structure_task,
            description="[cyan]Processing optimizer-v1 DIL structures [{},{}]...".format(
                resolved_patient_uid,
                specific_dil_structure["ROI"],
            ),
        )
        _run_patient_dil_optimizer(
            patient_uid=resolved_patient_uid,
            specific_dil_structure=specific_dil_structure,
            all_geometries_centered_cubic_lattice_arr=all_geometries_centered_cubic_lattice_arr,
            selected_prostate_info=selected_prostate_info,
            prostate_centroid=prostate_centroid,
            config=config,
            context=context,
            rng=optimizer_v1_rng,
        )
        context.structures_progress.update(structure_task, advance=1)
    context.structures_progress.update(structure_task, visible=False)

    _store_patient_guidance_and_lattice_outputs(
        patient_uid=resolved_patient_uid,
        patient_reference_dict=working_patient_reference_dict,
        config=config,
        context=context,
    )
    return OptimizerV1PatientStageResult(
        patient_uid=resolved_patient_uid,
        patient_reference_dict=working_patient_reference_dict,
        master_structure_info_dict=master_structure_info_dict,
        optimizer_outputs=collect_optimizer_v1_patient_outputs(
            working_patient_reference_dict,
            dil_ref=config.dil_ref,
            all_ref_key=config.all_ref_key,
        ),
        dil_count=len(working_patient_reference_dict[config.dil_ref]),
        metadata={"side_effect_options_rejected": True, **optimizer_v1_seed_metadata},
    )