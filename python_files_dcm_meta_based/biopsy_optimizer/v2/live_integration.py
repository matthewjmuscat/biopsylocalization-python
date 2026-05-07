"""Live pipeline bridge for the optimizer-v2 target-DIL family."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas

import custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p_grandparents
import polygon_dilation_helpers_numpy
from biopsy_optimizer.v2.contracts import OptimizerV2ChunkLayout
from biopsy_optimizer.v2.candidate_pool import build_target_candidate_pool
from biopsy_optimizer.v2.output import (
    annotate_target_dil_optimizer_dataframe_with_biopsy_sampling_audit,
    annotate_target_dil_optimizer_dataframe_with_downstream_mc,
    build_target_dil_optimization_summary_dataframe,
    build_target_dil_ranked_candidate_output_dataframe,
    build_target_dil_tested_candidate_output_dataframe,
)
from biopsy_optimizer.v2.render import (
    OptimizerV2PlotlyExportConfig,
    OptimizerV2StageBoundaryRenderJob,
    build_success_failure_render_layers_from_chunk_score_result,
    build_contour_line_render_layer,
    build_point_cloud_render_layer,
    build_stage_boundary_render_jobs,
    render_scene_render_jobs,
)
from biopsy_optimizer.v2.runner import run_target_staged_candidate_search
from biopsy_optimizer.v2.scoring import score_target_candidate_chunk
from preprocessing.biopsy_processing.simulated_biopsy_planner import (
    get_planned_simulated_biopsy_model_dict,
    get_planned_simulated_biopsy_sampled_points_arr,
)
from preprocessing.transform_bank import (
    get_biopsy_transform_bank_prefix,
    get_structure_transform_bank_prefix,
)


TARGET_DIL_OPTIMIZER_V2_LANE_NAME = "target_dil_optimizer_v2"
TARGET_DIL_OPTIMIZER_V2_SUMMARY_DF_KEY = (
    "Biopsy optimization - Target DIL optimizer v2 summary dataframe"
)
TARGET_DIL_OPTIMIZER_V2_RANKED_DF_KEY = (
    "Biopsy optimization - Target DIL optimizer v2 ranked candidates dataframe"
)
TARGET_DIL_OPTIMIZER_V2_TESTED_DF_KEY = (
    "Biopsy optimization - Target DIL optimizer v2 tested candidates dataframe"
)
TARGET_DIL_OPTIMIZER_V2_STAGE_BOUNDARY_RENDER_JOBS_KEY = (
    "Biopsy optimization - Target DIL optimizer v2 stage boundary render jobs"
)
TARGET_DIL_OPTIMIZER_V2_DOWNSTREAM_MC_SOURCE_DF_KEY = (
    "Tissue class - Global tissue by structure statistics"
)

DEFAULT_MAX_TEST_STRUCTURES_PER_CALL_CALIBRATION_SAFETY_FACTOR = 0.7
DEFAULT_MAX_TEST_STRUCTURES_PER_CALL_CALIBRATION_EXPANSION_FACTOR = 1.25
DEFAULT_MAX_TEST_STRUCTURES_PER_CALL_CALIBRATION_MAX_EXPANSION_ROUNDS = 2
DEFAULT_MAX_TEST_STRUCTURES_PER_CALL_CALIBRATION_MAX_BINARY_SEARCH_ROUNDS = 6


@dataclass(frozen=True)
class OptimizerV2CallCapacityCalibrationInputs:
    prototype_test_structure_points_2d_arr: np.ndarray
    target_relative_structures_nominal_plus_trials: Sequence[Sequence[np.ndarray]]


def run_target_dil_optimizer_v2_for_live_simulated_family(
    master_structure_reference_dict,
    master_structure_info_dict,
    structs_referenced_dict,
    bx_ref,
    dil_ref,
    all_ref_key,
    optimizer_simulated_type,
    search_config,
    parallel_pool,
    constant_z_slice_polygons_handler_option,
    remove_consecutive_duplicate_points_in_polygons,
    include_edges_in_log,
    kernel_type,
    patients_progress,
    structures_progress,
    completed_progress,
    live_display,
    max_candidates_per_chunk=8,
    max_test_structures_per_call=None,
    auto_calibrate_max_test_structures_per_call=True,
    downstream_comparable_trial_count=None,
    render_stage_boundary_candidate_clouds_bool=False,
    render_stage_names_to_render=None,
    render_backend="open3d",
    render_layer_style_by_name=None,
    render_plotly_export_bool=False,
    render_plotly_export_formats=("svg", "pdf"),
    render_plotly_export_width=1920,
    render_plotly_export_height=1080,
    render_plotly_export_scale=1.0,
    render_plotly_export_camera_eye=(1.6, -1.8, 1.15),
    render_plotly_export_camera_center=(0.0, 0.0, 0.0),
    render_plotly_export_camera_up=(0.0, 0.0, 1.0),
    render_winner_containment_debug_bool=False,
    render_winner_containment_backend=None,
    render_patient_whitelist=None,
    render_roi_whitelist=None,
    render_include_planned_sampled_points_bool=True,
    render_include_planned_core_structure_bool=True,
    render_include_planned_centroid_line_bool=True,
    render_include_target_surface_bool=True,
    render_include_selected_anatomy_bool=True,
    oar_ref=None,
    rectum_ref=None,
    urethra_ref=None,
):
    patientUID_default = "Initializing"
    processing_patients_task_main_description = "[red]Running optimizer-v2 sim-bx targeting [{}]...".format(
        patientUID_default
    )
    processing_patients_task_completed_main_description = "[green]Running optimizer-v2 sim-bx targeting"
    processing_patients_task = patients_progress.add_task(
        processing_patients_task_main_description,
        total=master_structure_info_dict["Global"]["Num cases"],
    )
    processing_patients_task_completed = completed_progress.add_task(
        processing_patients_task_completed_main_description,
        total=master_structure_info_dict["Global"]["Num cases"],
        visible=False,
    )

    resolved_max_test_structures_per_call = _resolve_effective_max_test_structures_per_call(
        master_structure_reference_dict=master_structure_reference_dict,
        bx_ref=bx_ref,
        dil_ref=dil_ref,
        optimizer_simulated_type=optimizer_simulated_type,
        search_config=search_config,
        parallel_pool=parallel_pool,
        constant_z_slice_polygons_handler_option=constant_z_slice_polygons_handler_option,
        remove_consecutive_duplicate_points_in_polygons=remove_consecutive_duplicate_points_in_polygons,
        include_edges_in_log=include_edges_in_log,
        kernel_type=kernel_type,
        max_test_structures_per_call=max_test_structures_per_call,
        auto_calibrate_max_test_structures_per_call=auto_calibrate_max_test_structures_per_call,
        downstream_comparable_trial_count=downstream_comparable_trial_count,
        structures_progress=structures_progress,
        completed_progress=completed_progress,
    )

    for patientUID, pydicom_item in master_structure_reference_dict.items():
        processing_patients_task_main_description = "[red]Running optimizer-v2 sim-bx targeting [{}]...".format(
            patientUID
        )
        patients_progress.update(
            processing_patients_task,
            description=processing_patients_task_main_description,
        )

        optimizer_target_structures = [
            specific_structure
            for specific_structure in pydicom_item[bx_ref]
            if specific_structure["Simulated bool"] == True
            and specific_structure["Simulated type"] == optimizer_simulated_type
        ]
        num_optimizer_target_structures = len(optimizer_target_structures)
        if num_optimizer_target_structures == 0:
            pydicom_item[all_ref_key]["Multi-structure pre-processing output dataframes dict"][
                TARGET_DIL_OPTIMIZER_V2_SUMMARY_DF_KEY
            ] = None
            pydicom_item[all_ref_key]["Multi-structure pre-processing output dataframes dict"][
                TARGET_DIL_OPTIMIZER_V2_RANKED_DF_KEY
            ] = None
            pydicom_item[all_ref_key]["Multi-structure pre-processing output dataframes dict"][
                TARGET_DIL_OPTIMIZER_V2_TESTED_DF_KEY
            ] = None
            patients_progress.update(processing_patients_task, advance=1)
            completed_progress.update(processing_patients_task_completed, advance=1)
            continue

        structureID_default = "Initializing"
        processing_structures_task_main_description = "[cyan]Optimizer-v2 structures [{},{}]...".format(
            patientUID,
            structureID_default,
        )
        processing_structures_task = structures_progress.add_task(
            processing_structures_task_main_description,
            total=num_optimizer_target_structures,
        )

        candidate_pool_cache = {}
        target_structure_pack_cache = {}
        patient_summary_dataframes = []
        patient_ranked_dataframes = []
        patient_tested_dataframes = []

        for specific_structure in optimizer_target_structures:
            structureID = specific_structure["ROI"]
            structures_progress.update(
                processing_structures_task,
                description="[cyan]Optimizer-v2 structures [{},{}]...".format(patientUID, structureID),
            )

            target_structure = _resolve_target_dil_structure(
                pydicom_item,
                specific_structure,
                dil_ref,
            )
            target_structure_cache_key = int(target_structure["Ref #"])

            candidate_pool = candidate_pool_cache.get(target_structure_cache_key)
            if candidate_pool is None:
                candidate_pool = build_target_candidate_pool(
                    target_points_array=np.asarray(
                        target_structure["Inter-slice interpolation information"].interpolated_pts_np_arr,
                        dtype=float,
                    ),
                    target_zslices_list=_copy_zslice_list(
                        target_structure["Inter-slice interpolation information"].interpolated_pts_list
                    ),
                    search_config=search_config,
                    constant_z_slice_polygons_handler_option=constant_z_slice_polygons_handler_option,
                    remove_consecutive_duplicate_points_in_polygons=remove_consecutive_duplicate_points_in_polygons,
                    kernel_type=kernel_type,
                    include_edges_in_log=include_edges_in_log,
                )
                candidate_pool_cache[target_structure_cache_key] = candidate_pool

            planned_biopsy_model_dict = get_planned_simulated_biopsy_model_dict(specific_structure)
            nominal_biopsy_points = np.asarray(
                get_planned_simulated_biopsy_sampled_points_arr(specific_structure),
                dtype=float,
            )
            nominal_biopsy_centroid = np.asarray(
                planned_biopsy_model_dict["Structure global centroid"],
                dtype=float,
            ).reshape(3)
            nominal_biopsy_centroid_line = np.asarray(
                planned_biopsy_model_dict["Best fit line of centroid pts"],
                dtype=float,
            )
            target_structure_centroid = np.asarray(
                target_structure["Structure global centroid"],
                dtype=float,
            ).reshape(3)

            def biopsy_transform_bank_prefix_provider(num_trials):
                return get_biopsy_transform_bank_prefix(specific_structure, num_trials)

            def target_transform_bank_prefix_provider(num_trials):
                return get_structure_transform_bank_prefix(target_structure, num_trials)

            def target_relative_structures_nominal_plus_trials_provider(num_trials):
                cache_key = (target_structure_cache_key, int(num_trials))
                cached_target_structure_pack = target_structure_pack_cache.get(cache_key)
                if cached_target_structure_pack is not None:
                    return cached_target_structure_pack

                resolved_target_structure_pack = _build_target_structure_nominal_plus_trials(
                    target_structure,
                    num_trials,
                    parallel_pool,
                )
                target_structure_pack_cache[cache_key] = resolved_target_structure_pack
                return resolved_target_structure_pack

            search_result = run_target_staged_candidate_search(
                candidate_pool=candidate_pool,
                search_config=search_config,
                nominal_biopsy_points=nominal_biopsy_points,
                nominal_biopsy_centroid=nominal_biopsy_centroid,
                nominal_biopsy_centroid_line=nominal_biopsy_centroid_line,
                biopsy_transform_bank_prefix_provider=biopsy_transform_bank_prefix_provider,
                target_relative_structures_nominal_plus_trials_provider=target_relative_structures_nominal_plus_trials_provider,
                target_structure_centroid=target_structure_centroid,
                target_transform_bank_prefix_provider=target_transform_bank_prefix_provider,
                max_candidates_per_chunk=max_candidates_per_chunk,
                max_test_structures_per_call=resolved_max_test_structures_per_call,
                include_edges_in_log=include_edges_in_log,
                kernel_type=kernel_type,
                downstream_comparable_trial_count=downstream_comparable_trial_count,
                return_array_as="numpy",
            )

            winner_candidate_point = _resolve_operational_winner_candidate_point(
                search_result,
                candidate_pool,
            )
            additional_render_layers = _build_additional_stage_boundary_render_layers(
                structs_referenced_dict=structs_referenced_dict,
                pydicom_item=pydicom_item,
                specific_structure=specific_structure,
                nominal_biopsy_centroid=nominal_biopsy_centroid,
                winner_candidate_point=winner_candidate_point,
                bx_ref=bx_ref,
                target_structure=target_structure,
                render_include_planned_sampled_points_bool=render_include_planned_sampled_points_bool,
                render_include_planned_core_structure_bool=render_include_planned_core_structure_bool,
                render_include_planned_centroid_line_bool=render_include_planned_centroid_line_bool,
                render_include_target_surface_bool=render_include_target_surface_bool,
                render_include_selected_anatomy_bool=render_include_selected_anatomy_bool,
                render_layer_style_by_name=render_layer_style_by_name,
                oar_ref=oar_ref,
                rectum_ref=rectum_ref,
                urethra_ref=urethra_ref,
            )

            stage_boundary_render_jobs = build_stage_boundary_render_jobs(
                search_result=search_result,
                candidate_pool=candidate_pool,
                target_points_array=np.asarray(
                    target_structure["Inter-slice interpolation information"].interpolated_pts_np_arr,
                    dtype=float,
                ),
                nominal_biopsy_centroid=nominal_biopsy_centroid,
                stage_names_to_render=render_stage_names_to_render,
                additional_render_layers=additional_render_layers,
                scene_name_prefix="{}__{}".format(patientUID, structureID),
                render_layer_style_by_name=render_layer_style_by_name,
            )
            specific_structure[
                TARGET_DIL_OPTIMIZER_V2_STAGE_BOUNDARY_RENDER_JOBS_KEY
            ] = stage_boundary_render_jobs
            should_render_structure = _should_render_structure_stage_boundary_candidate_clouds(
                patientUID,
                structureID,
                render_patient_whitelist,
                render_roi_whitelist,
            )
            plotly_export_config = None
            if render_plotly_export_bool and should_render_structure and stage_boundary_render_jobs:
                plotly_export_config = _build_plotly_export_config_for_scene_group(
                    master_structure_info_dict,
                    patientUID,
                    structureID,
                    "stage_boundary",
                    render_plotly_export_formats,
                    render_plotly_export_width,
                    render_plotly_export_height,
                    render_plotly_export_scale,
                    render_plotly_export_camera_eye,
                    render_plotly_export_camera_center,
                    render_plotly_export_camera_up,
                )

            should_render_interactively = (
                render_stage_boundary_candidate_clouds_bool
                and should_render_structure
                and stage_boundary_render_jobs
            )
            should_export_structure = (
                plotly_export_config is not None and bool(stage_boundary_render_jobs)
            )
            if should_render_interactively or should_export_structure:
                effective_render_backend = render_backend if should_render_interactively else "none"
                should_pause_live_display = (
                    should_render_interactively or should_export_structure
                )
                if should_pause_live_display:
                    live_display.stop()
                try:
                    render_scene_render_jobs(
                        stage_boundary_render_jobs,
                        render_backend=effective_render_backend,
                        plotly_export_config=plotly_export_config,
                    )
                finally:
                    if should_pause_live_display:
                        live_display.start(refresh=True)
                        live_display.refresh()

            resolved_winner_containment_backend = render_backend
            if render_winner_containment_backend is not None:
                resolved_winner_containment_backend = render_winner_containment_backend
            if render_winner_containment_debug_bool and should_render_structure:
                winner_containment_render_job, winner_containment_chunk_score_result = (
                    _build_winner_containment_debug_render_job(
                        patientUID=patientUID,
                        structureID=structureID,
                        search_result=search_result,
                        candidate_pool=candidate_pool,
                        nominal_biopsy_points=nominal_biopsy_points,
                        nominal_biopsy_centroid=nominal_biopsy_centroid,
                        nominal_biopsy_centroid_line=nominal_biopsy_centroid_line,
                        target_structure=target_structure,
                        target_structure_centroid=target_structure_centroid,
                        biopsy_transform_bank_prefix_provider=biopsy_transform_bank_prefix_provider,
                        target_relative_structures_nominal_plus_trials_provider=target_relative_structures_nominal_plus_trials_provider,
                        target_transform_bank_prefix_provider=target_transform_bank_prefix_provider,
                        downstream_comparable_trial_count=downstream_comparable_trial_count,
                        additional_render_layers=additional_render_layers,
                        render_layer_style_by_name=render_layer_style_by_name,
                        max_test_structures_per_call=resolved_max_test_structures_per_call,
                        include_edges_in_log=include_edges_in_log,
                        kernel_type=kernel_type,
                    )
                )
                if winner_containment_render_job is not None:
                    _print_winner_containment_debug_summary(
                        patientUID,
                        structureID,
                        winner_containment_chunk_score_result,
                    )
                    winner_containment_export_config = None
                    if render_plotly_export_bool:
                        winner_containment_export_config = _build_plotly_export_config_for_scene_group(
                            master_structure_info_dict,
                            patientUID,
                            structureID,
                            "winner_containment",
                            render_plotly_export_formats,
                            render_plotly_export_width,
                            render_plotly_export_height,
                            render_plotly_export_scale,
                            render_plotly_export_camera_eye,
                            render_plotly_export_camera_center,
                            render_plotly_export_camera_up,
                        )

                    should_render_winner_interactively = (
                        _normalize_requested_render_backend(resolved_winner_containment_backend) != "none"
                    )
                    should_export_winner = winner_containment_export_config is not None
                    if should_render_winner_interactively or should_export_winner:
                        effective_winner_backend = (
                            resolved_winner_containment_backend
                            if should_render_winner_interactively
                            else "none"
                        )
                        should_pause_live_display = (
                            should_render_winner_interactively or should_export_winner
                        )
                        if should_pause_live_display:
                            live_display.stop()
                        try:
                            render_scene_render_jobs(
                                (winner_containment_render_job,),
                                render_backend=effective_winner_backend,
                                plotly_export_config=winner_containment_export_config,
                            )
                        finally:
                            if should_pause_live_display:
                                live_display.start(refresh=True)
                                live_display.refresh()

            metadata = _build_search_metadata(
                patientUID,
                specific_structure,
                target_structure,
            )
            summary_dataframe = build_target_dil_optimization_summary_dataframe(
                search_result,
                metadata=metadata,
            )
            ranked_candidate_dataframe = build_target_dil_ranked_candidate_output_dataframe(
                search_result,
                metadata=metadata,
            )
            tested_candidate_dataframe = build_target_dil_tested_candidate_output_dataframe(
                search_result,
                metadata=metadata,
            )

            if summary_dataframe.empty:
                summary_dataframe = _build_target_centroid_fallback_summary_dataframe(
                    target_structure_centroid,
                    metadata,
                )
                ranked_candidate_dataframe = pandas.DataFrame()
                tested_candidate_dataframe = pandas.DataFrame()
                target_vector = target_structure_centroid
                transport_source = "{}:target_centroid_fallback".format(TARGET_DIL_OPTIMIZER_V2_LANE_NAME)
            else:
                summary_row = summary_dataframe.iloc[0]
                target_vector = np.array(
                    [
                        summary_row["Target optimizer selected X"],
                        summary_row["Target optimizer selected Y"],
                        summary_row["Target optimizer selected Z"],
                    ],
                    dtype=float,
                )
                transport_source = TARGET_DIL_OPTIMIZER_V2_LANE_NAME

            specific_structure["Simulated biopsy transport request dict"] = {
                "Transport family": "identity",
                "Target vector": np.asarray(target_vector, dtype=float),
                "Transport source": transport_source,
                "Selection metadata": _build_transport_selection_metadata(summary_dataframe),
            }

            patient_summary_dataframes.append(summary_dataframe)
            if not ranked_candidate_dataframe.empty:
                patient_ranked_dataframes.append(ranked_candidate_dataframe)
            if not tested_candidate_dataframe.empty:
                patient_tested_dataframes.append(tested_candidate_dataframe)

            structures_progress.update(processing_structures_task, advance=1)

        structures_progress.remove_task(processing_structures_task)

        pydicom_item[all_ref_key]["Multi-structure pre-processing output dataframes dict"][
            TARGET_DIL_OPTIMIZER_V2_SUMMARY_DF_KEY
        ] = _concat_dataframes_or_none(patient_summary_dataframes)
        pydicom_item[all_ref_key]["Multi-structure pre-processing output dataframes dict"][
            TARGET_DIL_OPTIMIZER_V2_RANKED_DF_KEY
        ] = _concat_dataframes_or_none(patient_ranked_dataframes)
        pydicom_item[all_ref_key]["Multi-structure pre-processing output dataframes dict"][
            TARGET_DIL_OPTIMIZER_V2_TESTED_DF_KEY
        ] = _concat_dataframes_or_none(patient_tested_dataframes)

        patients_progress.update(processing_patients_task, advance=1)
        completed_progress.update(processing_patients_task_completed, advance=1)

    patients_progress.update(processing_patients_task, visible=False)
    completed_progress.update(processing_patients_task_completed, visible=True)
    return live_display


def annotate_target_dil_optimizer_v2_outputs_with_downstream_mc_scores(
    master_structure_reference_dict,
    all_ref_key,
    downstream_trial_count,
):
    for _patient_uid, pydicom_item in master_structure_reference_dict.items():
        downstream_dataframe_dict = (
            pydicom_item[all_ref_key].get("Multi-structure MC simulation output dataframes dict") or {}
        )
        downstream_structure_score_dataframe = downstream_dataframe_dict.get(
            TARGET_DIL_OPTIMIZER_V2_DOWNSTREAM_MC_SOURCE_DF_KEY
        )
        pre_processing_dataframe_dict = pydicom_item[all_ref_key][
            "Multi-structure pre-processing output dataframes dict"
        ]
        pre_processing_dataframe_dict[TARGET_DIL_OPTIMIZER_V2_SUMMARY_DF_KEY] = (
            annotate_target_dil_optimizer_dataframe_with_downstream_mc(
                pre_processing_dataframe_dict.get(TARGET_DIL_OPTIMIZER_V2_SUMMARY_DF_KEY),
                downstream_structure_score_dataframe,
                downstream_trial_count,
            )
        )
        pre_processing_dataframe_dict[TARGET_DIL_OPTIMIZER_V2_RANKED_DF_KEY] = (
            annotate_target_dil_optimizer_dataframe_with_downstream_mc(
                pre_processing_dataframe_dict.get(TARGET_DIL_OPTIMIZER_V2_RANKED_DF_KEY),
                downstream_structure_score_dataframe,
                downstream_trial_count,
            )
        )
        pre_processing_dataframe_dict[TARGET_DIL_OPTIMIZER_V2_TESTED_DF_KEY] = (
            annotate_target_dil_optimizer_dataframe_with_downstream_mc(
                pre_processing_dataframe_dict.get(TARGET_DIL_OPTIMIZER_V2_TESTED_DF_KEY),
                downstream_structure_score_dataframe,
                downstream_trial_count,
            )
        )


def annotate_target_dil_optimizer_v2_outputs_with_biopsy_sampling_audit(
    master_structure_reference_dict,
    bx_ref,
    all_ref_key,
):
    for patient_uid, pydicom_item in master_structure_reference_dict.items():
        biopsy_sampling_audit_dataframe = _build_biopsy_sampling_audit_source_dataframe(
            patient_uid,
            pydicom_item,
            bx_ref,
        )
        pre_processing_dataframe_dict = pydicom_item[all_ref_key][
            "Multi-structure pre-processing output dataframes dict"
        ]
        pre_processing_dataframe_dict[TARGET_DIL_OPTIMIZER_V2_SUMMARY_DF_KEY] = (
            annotate_target_dil_optimizer_dataframe_with_biopsy_sampling_audit(
                pre_processing_dataframe_dict.get(TARGET_DIL_OPTIMIZER_V2_SUMMARY_DF_KEY),
                biopsy_sampling_audit_dataframe,
            )
        )
        pre_processing_dataframe_dict[TARGET_DIL_OPTIMIZER_V2_RANKED_DF_KEY] = (
            annotate_target_dil_optimizer_dataframe_with_biopsy_sampling_audit(
                pre_processing_dataframe_dict.get(TARGET_DIL_OPTIMIZER_V2_RANKED_DF_KEY),
                biopsy_sampling_audit_dataframe,
            )
        )
        pre_processing_dataframe_dict[TARGET_DIL_OPTIMIZER_V2_TESTED_DF_KEY] = (
            annotate_target_dil_optimizer_dataframe_with_biopsy_sampling_audit(
                pre_processing_dataframe_dict.get(TARGET_DIL_OPTIMIZER_V2_TESTED_DF_KEY),
                biopsy_sampling_audit_dataframe,
            )
        )


def _resolve_target_dil_structure(
    pydicom_item,
    specific_structure,
    dil_ref,
):
    simulated_biopsy_preparation_dict = specific_structure.get("Simulated biopsy preparation dict") or {}
    target_structure_type = simulated_biopsy_preparation_dict.get("Target structure type")
    target_structure_index = simulated_biopsy_preparation_dict.get("Target structure index")
    target_structure_refnum = simulated_biopsy_preparation_dict.get("Target structure ref #")

    if target_structure_type == dil_ref and target_structure_index is not None:
        return pydicom_item[dil_ref][int(target_structure_index)]

    relative_structure_refnum = specific_structure.get("Relative structure ref #")
    if relative_structure_refnum is not None:
        target_structure_refnum = relative_structure_refnum

    for candidate_target_structure in pydicom_item[dil_ref]:
        if candidate_target_structure["Ref #"] == target_structure_refnum:
            return candidate_target_structure

    raise ValueError(
        "Could not resolve target DIL structure for optimizer-v2 simulated biopsy {}.".format(
            specific_structure.get("ROI")
        )
    )


def _build_target_structure_nominal_plus_trials(
    target_structure,
    num_trials,
    parallel_pool,
):
    nominal_target_zslice_list = _copy_zslice_list(
        target_structure["Inter-slice interpolation information"].interpolated_pts_list
    )
    if num_trials <= 0:
        return [nominal_target_zslice_list]

    target_transform_bank_prefix = get_structure_transform_bank_prefix(target_structure, num_trials)
    dilation_samples = _coerce_samples_to_numpy_array(target_transform_bank_prefix.dilation_samples)

    nominal_plus_trials = [nominal_target_zslice_list]
    if not np.any(dilation_samples):
        for _ in range(num_trials):
            nominal_plus_trials.append(_copy_zslice_list(nominal_target_zslice_list))
        return nominal_plus_trials

    nominal_target_points_array, nominal_target_indices_array = (
        polygon_dilation_helpers_numpy.convert_to_2d_array_and_indices_numpy(nominal_target_zslice_list)
    )
    dilated_structures_list, dilated_structures_indices_list = (
        polygon_dilation_helpers_numpy.generate_dilated_structures_parallelized(
            nominal_target_points_array,
            nominal_target_indices_array,
            dilation_samples,
            False,
            False,
            parallel_pool,
        )
    )
    for dilated_structure_points_array, dilated_structure_indices_array in zip(
        dilated_structures_list,
        dilated_structures_indices_list,
    ):
        nominal_plus_trials.append(
            _copy_zslice_list(
                polygon_dilation_helpers_numpy.reconstruct_list_from_2d_array(
                    dilated_structure_points_array,
                    dilated_structure_indices_array,
                )
            )
        )

    if len(nominal_plus_trials) != num_trials + 1:
        raise ValueError(
            "target structure trial-pack size mismatch: expected {}, found {}".format(
                num_trials + 1,
                len(nominal_plus_trials),
            )
        )

    return nominal_plus_trials


def _resolve_effective_max_test_structures_per_call(
    master_structure_reference_dict,
    bx_ref,
    dil_ref,
    optimizer_simulated_type,
    search_config,
    parallel_pool,
    constant_z_slice_polygons_handler_option,
    remove_consecutive_duplicate_points_in_polygons,
    include_edges_in_log,
    kernel_type,
    max_test_structures_per_call,
    auto_calibrate_max_test_structures_per_call,
    downstream_comparable_trial_count,
    structures_progress=None,
    completed_progress=None,
):
    if max_test_structures_per_call is not None:
        return int(max_test_structures_per_call)
    if not auto_calibrate_max_test_structures_per_call:
        return None

    calibration_elapsed_start_time = time.perf_counter()
    calibration_progress_task = None
    if structures_progress is not None:
        calibration_progress_task = structures_progress.add_task(
            "[yellow]Optimizer-v2 calibration: building representative workload",
            total=3,
        )

    try:
        calibration_inputs = _build_optimizer_v2_call_capacity_calibration_inputs(
            master_structure_reference_dict=master_structure_reference_dict,
            bx_ref=bx_ref,
            dil_ref=dil_ref,
            optimizer_simulated_type=optimizer_simulated_type,
            search_config=search_config,
            downstream_comparable_trial_count=downstream_comparable_trial_count,
            parallel_pool=parallel_pool,
        )
        if calibration_inputs is None:
            if calibration_progress_task is not None:
                structures_progress.update(
                    calibration_progress_task,
                    advance=3,
                    description="[yellow]Optimizer-v2 calibration: skipped (no eligible optimizer-v2 targets)",
                )
                structures_progress.remove_task(calibration_progress_task)
            return None

        if calibration_progress_task is not None:
            structures_progress.update(
                calibration_progress_task,
                advance=1,
                description="[yellow]Optimizer-v2 calibration: probing safe containment-call budget",
            )

        calibration_result = (
            custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p_grandparents.calibrate_max_test_structures_per_call(
                list_of_relative_structures_containting_list_of_constant_zslices_arrays=(
                    calibration_inputs.target_relative_structures_nominal_plus_trials
                ),
                prototype_test_structure_points_2d_arr=(
                    calibration_inputs.prototype_test_structure_points_2d_arr
                ),
                constant_z_slice_polygons_handler_option=constant_z_slice_polygons_handler_option,
                remove_consecutive_duplicate_points_in_polygons=(
                    remove_consecutive_duplicate_points_in_polygons
                ),
                log_file_name=None,
                include_edges_in_log=include_edges_in_log,
                kernel_type=kernel_type,
                safety_factor=DEFAULT_MAX_TEST_STRUCTURES_PER_CALL_CALIBRATION_SAFETY_FACTOR,
                verification_expansion_factor=(
                    DEFAULT_MAX_TEST_STRUCTURES_PER_CALL_CALIBRATION_EXPANSION_FACTOR
                ),
                max_verification_expansion_rounds=(
                    DEFAULT_MAX_TEST_STRUCTURES_PER_CALL_CALIBRATION_MAX_EXPANSION_ROUNDS
                ),
                max_binary_search_rounds=(
                    DEFAULT_MAX_TEST_STRUCTURES_PER_CALL_CALIBRATION_MAX_BINARY_SEARCH_ROUNDS
                ),
            )
        )
        resolved_max_test_structures_per_call = int(
            calibration_result.safe_max_test_structures_per_call
        )
        calibration_elapsed_seconds = time.perf_counter() - calibration_elapsed_start_time
        calibration_summary_description = _build_optimizer_v2_calibration_summary_description(
            calibration_result,
            resolved_max_test_structures_per_call,
            calibration_elapsed_seconds,
        )

        if calibration_progress_task is not None:
            structures_progress.update(
                calibration_progress_task,
                advance=1,
                description="[yellow]Optimizer-v2 calibration: finalizing optimizer budget",
            )
            structures_progress.update(
                calibration_progress_task,
                advance=1,
                description=calibration_summary_description,
            )
            structures_progress.remove_task(calibration_progress_task)
        _record_optimizer_v2_calibration_completion(
            completed_progress,
            calibration_summary_description,
        )
        return resolved_max_test_structures_per_call
    except Exception:
        if calibration_progress_task is not None:
            structures_progress.update(
                calibration_progress_task,
                description="[red]Optimizer-v2 calibration failed",
            )
        raise


def _build_optimizer_v2_calibration_summary_description(
    calibration_result,
    resolved_max_test_structures_per_call,
    calibration_elapsed_seconds,
):
    cache_status = "cache hit" if calibration_result.from_cache else "cache miss"
    return (
        "[green]Optimizer-v2 calibration: safe max_test_structures_per_call={} "
        "({}; attempts={}; {:.1f}s)"
    ).format(
        int(resolved_max_test_structures_per_call),
        cache_status,
        int(calibration_result.verification_attempt_count),
        float(calibration_elapsed_seconds),
    )


def _record_optimizer_v2_calibration_completion(
    completed_progress,
    calibration_summary_description,
):
    if completed_progress is None:
        return

    completed_calibration_task = completed_progress.add_task(
        calibration_summary_description,
        total=1,
    )
    completed_progress.update(completed_calibration_task, advance=1)


def _build_optimizer_v2_call_capacity_calibration_inputs(
    master_structure_reference_dict,
    bx_ref,
    dil_ref,
    optimizer_simulated_type,
    search_config,
    downstream_comparable_trial_count,
    parallel_pool,
):
    representative_biopsy_points = None
    representative_target_structure = None
    representative_biopsy_point_count = -1
    representative_target_point_count = -1

    for patientUID, pydicom_item in master_structure_reference_dict.items():
        del patientUID
        optimizer_target_structures = [
            specific_structure
            for specific_structure in pydicom_item[bx_ref]
            if specific_structure["Simulated bool"] == True
            and specific_structure["Simulated type"] == optimizer_simulated_type
        ]
        for specific_structure in optimizer_target_structures:
            nominal_biopsy_points = np.asarray(
                get_planned_simulated_biopsy_sampled_points_arr(specific_structure),
                dtype=float,
            )
            if nominal_biopsy_points.shape[0] > representative_biopsy_point_count:
                representative_biopsy_points = nominal_biopsy_points
                representative_biopsy_point_count = int(nominal_biopsy_points.shape[0])

            target_structure = _resolve_target_dil_structure(
                pydicom_item,
                specific_structure,
                dil_ref,
            )
            target_point_count = _count_constant_zslice_points(
                target_structure["Inter-slice interpolation information"].interpolated_pts_list
            )
            if target_point_count > representative_target_point_count:
                representative_target_structure = target_structure
                representative_target_point_count = int(target_point_count)

    if representative_biopsy_points is None or representative_target_structure is None:
        return None

    calibration_trial_count = search_config.resolve_required_transform_bank_size(
        downstream_trial_count=downstream_comparable_trial_count,
    )
    return OptimizerV2CallCapacityCalibrationInputs(
        prototype_test_structure_points_2d_arr=np.asarray(
            representative_biopsy_points,
            dtype=float,
        ),
        target_relative_structures_nominal_plus_trials=_build_target_structure_nominal_plus_trials(
            representative_target_structure,
            calibration_trial_count,
            parallel_pool,
        ),
    )


def _count_constant_zslice_points(list_of_constant_zslice_arrays):
    return int(
        sum(
            np.asarray(constant_zslice_array).shape[0]
            for constant_zslice_array in list_of_constant_zslice_arrays
        )
    )


def _build_search_metadata(
    patientUID,
    specific_structure,
    target_structure,
):
    simulated_biopsy_preparation_dict = specific_structure.get("Simulated biopsy preparation dict") or {}
    return {
        "Patient ID": patientUID,
        "Biopsy ROI": specific_structure.get("ROI"),
        "Biopsy ref #": specific_structure.get("Ref #"),
        "Biopsy index": _normalize_scalar(specific_structure.get("Index number")),
        "Simulated biopsy type": specific_structure.get("Simulated type"),
        "Biopsy multiplicity": _normalize_scalar(simulated_biopsy_preparation_dict.get("Multiplicity")),
        "Biopsy multiplicity index": _normalize_scalar(
            simulated_biopsy_preparation_dict.get("Multiplicity index")
        ),
        "Target structure type": simulated_biopsy_preparation_dict.get("Target structure type")
        or specific_structure.get("Relative structure type"),
        "Target structure ref #": simulated_biopsy_preparation_dict.get("Target structure ref #")
        or target_structure.get("Ref #"),
        "Target structure index": _normalize_scalar(
            simulated_biopsy_preparation_dict.get("Target structure index")
        ),
        "Target structure ID": simulated_biopsy_preparation_dict.get("Target structure ID")
        or target_structure.get("ROI"),
        "Target optimizer planned biopsy sampled point count": _resolve_planned_sampled_point_count(
            specific_structure
        ),
    }


def _build_biopsy_sampling_audit_source_dataframe(
    patient_uid,
    pydicom_item,
    bx_ref,
):
    source_rows = []
    for specific_structure in pydicom_item.get(bx_ref, []):
        planned_sampled_point_count = _resolve_planned_sampled_point_count(specific_structure)
        finalized_sampled_point_count = _resolve_finalized_sampled_point_count(specific_structure)
        if planned_sampled_point_count is None and finalized_sampled_point_count is None:
            continue

        source_rows.append(
            {
                "Patient ID": patient_uid,
                "Biopsy ROI": specific_structure.get("ROI"),
                "Biopsy ref #": specific_structure.get("Ref #"),
                "Biopsy index": _normalize_scalar(specific_structure.get("Index number")),
                "Target optimizer planned biopsy sampled point count": planned_sampled_point_count,
                "Target optimizer finalized biopsy sampled point count": finalized_sampled_point_count,
            }
        )

    if len(source_rows) == 0:
        return pandas.DataFrame(
            columns=[
                "Patient ID",
                "Biopsy ROI",
                "Biopsy ref #",
                "Biopsy index",
                "Target optimizer planned biopsy sampled point count",
                "Target optimizer finalized biopsy sampled point count",
            ]
        )

    return pandas.DataFrame(source_rows)


def _resolve_planned_sampled_point_count(specific_structure):
    simulated_biopsy_planning_dict = specific_structure.get("Simulated biopsy planning dict") or {}
    planned_sampled_point_count = simulated_biopsy_planning_dict.get("Planned sampled point count")
    if planned_sampled_point_count is not None:
        return int(planned_sampled_point_count)

    planned_sampled_points = simulated_biopsy_planning_dict.get("Planned sampled volume pts arr")
    if planned_sampled_points is None:
        return None

    return int(np.asarray(planned_sampled_points).shape[0])


def _resolve_finalized_sampled_point_count(specific_structure):
    finalized_sampled_point_count = specific_structure.get("Num sampled bx pts")
    if finalized_sampled_point_count is not None:
        return int(finalized_sampled_point_count)

    finalized_sampled_points = specific_structure.get("Random uniformly sampled volume pts arr")
    if finalized_sampled_points is None:
        return None

    return int(np.asarray(finalized_sampled_points).shape[0])


def _build_target_centroid_fallback_summary_dataframe(
    target_structure_centroid,
    metadata,
):
    fallback_summary_row = {
        "Target optimizer lane": TARGET_DIL_OPTIMIZER_V2_LANE_NAME,
        "Target optimizer final stage name": "",
        "Target optimizer num stages": 0,
        "Target optimizer num tested candidate rows": 0,
        "Target optimizer num final ranked candidates": 0,
        "Target optimizer operational winner candidate index": np.nan,
        "Target optimizer selected X": float(target_structure_centroid[0]),
        "Target optimizer selected Y": float(target_structure_centroid[1]),
        "Target optimizer selected Z": float(target_structure_centroid[2]),
        "Target optimizer retained score": np.nan,
        "Target optimizer retained nominal score": np.nan,
        "Target optimizer objective reducer name": np.nan,
        "Target optimizer distance to target centroid mm": 0.0,
        "Target optimizer winner determination method": "target_centroid_fallback_no_ranked_candidates",
        "Target optimizer tie-break warning flag": False,
        "Target optimizer centroid fallback flag": True,
        "Target optimizer final resolution trial count": np.nan,
        "Target optimizer selection score trial count": np.nan,
        "Target optimizer additional rescore attempts used": np.nan,
        "Target optimizer final tie candidate count": np.nan,
        "Target optimizer resolved objective value": np.nan,
        "Target optimizer resolved nominal objective value": np.nan,
        "Target optimizer selected winner optimizer-side target score": np.nan,
        "Target optimizer selected winner optimizer-side trial count": np.nan,
        "Target optimizer downstream comparable target score": np.nan,
        "Target optimizer downstream comparable trial count": np.nan,
        "Target optimizer selected winner downstream-comparable target score": np.nan,
        "Target optimizer selected winner downstream-comparable trial count": np.nan,
        "Target optimizer selected winner downstream MC target score": np.nan,
        "Target optimizer selected winner downstream MC trial count": np.nan,
        "Target optimizer selected winner downstream MC agreement delta": np.nan,
        "Target optimizer downstream comparable score trial count": np.nan,
        "Target optimizer selected winner score-surface delta": np.nan,
        "Target optimizer agreement delta": np.nan,
        "Target optimizer fallback reason": "no_ranked_candidates",
    }
    summary_dataframe = pandas.DataFrame([fallback_summary_row])
    for key, value in metadata.items():
        summary_dataframe[key] = value
    return summary_dataframe


def _should_render_structure_stage_boundary_candidate_clouds(
    patient_uid,
    roi_name,
    render_patient_whitelist,
    render_roi_whitelist,
):
    normalized_patient_uid = str(patient_uid).strip().lower()
    normalized_roi_name = str(roi_name).strip().lower()

    patient_allowed = _matches_render_whitelist(
        normalized_patient_uid,
        render_patient_whitelist,
        require_exact_match=True,
    )
    roi_allowed = _matches_render_whitelist(
        normalized_roi_name,
        render_roi_whitelist,
        require_exact_match=False,
    )

    return patient_allowed and roi_allowed


def _normalize_render_whitelist(raw_whitelist):
    return tuple(
        str(raw_item).strip().lower()
        for raw_item in raw_whitelist
        if str(raw_item).strip() != ""
    )


def _matches_render_whitelist(
    normalized_candidate_name,
    raw_whitelist,
    require_exact_match,
):
    if raw_whitelist is None:
        return True

    normalized_whitelist = _normalize_render_whitelist(raw_whitelist)
    if len(normalized_whitelist) == 0:
        return False

    if require_exact_match:
        return normalized_candidate_name in normalized_whitelist

    return any(
        whitelist_entry == normalized_candidate_name or whitelist_entry in normalized_candidate_name
        for whitelist_entry in normalized_whitelist
    )


def _build_plotly_export_config_for_scene_group(
    master_structure_info_dict,
    patient_uid,
    roi_name,
    scene_group_name,
    file_formats,
    width,
    height,
    scale,
    camera_eye,
    camera_center,
    camera_up,
):
    global_info = master_structure_info_dict.get("Global") or {}
    specific_output_dir = global_info.get("Specific output dir")
    if specific_output_dir is None:
        return None

    export_dir = Path(specific_output_dir).joinpath(
        "scientific_communication",
        "optimizer_v2",
        _sanitize_output_path_fragment(scene_group_name),
        _sanitize_output_path_fragment(patient_uid),
        _sanitize_output_path_fragment(roi_name),
        "plotly_vector",
    )
    export_dir.mkdir(parents=True, exist_ok=True)
    return OptimizerV2PlotlyExportConfig(
        output_dir=export_dir,
        file_formats=tuple(file_formats),
        width=int(width),
        height=int(height),
        scale=float(scale),
        camera_eye=tuple(float(value) for value in camera_eye),
        camera_center=tuple(float(value) for value in camera_center),
        camera_up=tuple(float(value) for value in camera_up),
    )


def _sanitize_output_path_fragment(raw_fragment):
    sanitized_characters = []
    for character in str(raw_fragment):
        if character.isalnum() or character in ("-", "_", "."):
            sanitized_characters.append(character)
        else:
            sanitized_characters.append("_")

    sanitized_fragment = "".join(sanitized_characters).strip("_")
    if sanitized_fragment == "":
        return "optimizer_v2"
    return sanitized_fragment


def _build_winner_containment_debug_render_job(
    patientUID,
    structureID,
    search_result,
    candidate_pool,
    nominal_biopsy_points,
    nominal_biopsy_centroid,
    nominal_biopsy_centroid_line,
    target_structure,
    target_structure_centroid,
    biopsy_transform_bank_prefix_provider,
    target_relative_structures_nominal_plus_trials_provider,
    target_transform_bank_prefix_provider,
    downstream_comparable_trial_count,
    additional_render_layers,
    render_layer_style_by_name,
    max_test_structures_per_call,
    include_edges_in_log,
    kernel_type,
):
    winner_candidate_index_global = search_result.operational_winner_candidate_index_global
    if winner_candidate_index_global is None:
        return None, None

    resolved_trial_count = _resolve_winner_containment_trial_count(
        search_result,
        downstream_comparable_trial_count,
    )
    winner_chunk_layout = OptimizerV2ChunkLayout(
        candidate_indices_global=(int(winner_candidate_index_global),),
        num_trials=resolved_trial_count,
        include_nominal=True,
        nominal_relative_structure_index=0,
        trial_relative_structure_start_index=1,
    )
    winner_chunk_score_result = score_target_candidate_chunk(
        candidate_pool=candidate_pool,
        chunk_layout=winner_chunk_layout,
        nominal_biopsy_points=nominal_biopsy_points,
        nominal_biopsy_centroid=nominal_biopsy_centroid,
        nominal_biopsy_centroid_line=nominal_biopsy_centroid_line,
        biopsy_transform_bank_prefix=biopsy_transform_bank_prefix_provider(resolved_trial_count),
        target_relative_structures_nominal_plus_trials=target_relative_structures_nominal_plus_trials_provider(
            resolved_trial_count
        ),
        target_structure_centroid=target_structure_centroid,
        target_transform_bank_prefix=target_transform_bank_prefix_provider(resolved_trial_count),
        objective_reducer_name="mean_pd",
        max_test_structures_per_call=max_test_structures_per_call,
        create_tested_candidate_dataframe=True,
        include_relative_structure_localized_points_for_debug=True,
        include_edges_in_log=include_edges_in_log,
        kernel_type=kernel_type,
        return_array_as="numpy",
    )
    containment_render_layers = build_success_failure_render_layers_from_chunk_score_result(
        winner_chunk_score_result,
        candidate_local_chunk_index=0,
        include_nominal_slice=False,
    )

    winner_candidate_point = np.asarray(
        candidate_pool.candidate_points[int(winner_candidate_index_global)],
        dtype=float,
    ).reshape(1, 3)
    render_layers = [
        build_point_cloud_render_layer(
            layer_name="target_points",
            points=np.asarray(
                target_structure["Inter-slice interpolation information"].interpolated_pts_np_arr,
                dtype=float,
            ),
            color=_resolve_layer_style_color(
                render_layer_style_by_name,
                "target_points",
                np.array([0.0, 0.0, 1.0]),
            ),
            marker_size=_resolve_layer_style_float(
                render_layer_style_by_name,
                "target_points",
                "marker_size",
            ),
            opacity=_resolve_layer_style_float(
                render_layer_style_by_name,
                "target_points",
                "opacity",
            ),
        ),
        build_point_cloud_render_layer(
            layer_name="operational_winner",
            points=winner_candidate_point,
            color=_resolve_layer_style_color(
                render_layer_style_by_name,
                "operational_winner",
                np.array([1.0, 0.0, 1.0]),
            ),
            marker_size=_resolve_layer_style_float(
                render_layer_style_by_name,
                "operational_winner",
                "marker_size",
            ),
            opacity=_resolve_layer_style_float(
                render_layer_style_by_name,
                "operational_winner",
                "opacity",
            ),
        ),
    ]
    render_layers.extend(additional_render_layers)
    render_layers.extend(containment_render_layers)

    render_job = OptimizerV2StageBoundaryRenderJob(
        scene_name="{}__{}__optimizer_v2_winner_containment".format(patientUID, structureID),
        stage_name="winner_containment",
        input_candidate_points=winner_candidate_point.copy(),
        survivor_candidate_points=winner_candidate_point.copy(),
        target_points=np.asarray(
            target_structure["Inter-slice interpolation information"].interpolated_pts_np_arr,
            dtype=float,
        ),
        render_layers=tuple(render_layers),
    )
    return render_job, winner_chunk_score_result


def _resolve_winner_containment_trial_count(
    search_result,
    downstream_comparable_trial_count,
):
    if downstream_comparable_trial_count is not None and int(downstream_comparable_trial_count) > 0:
        return int(downstream_comparable_trial_count)

    winner_resolution_result = getattr(search_result, "winner_resolution_result", None)
    if winner_resolution_result is not None:
        resolved_trial_count = int(winner_resolution_result.final_resolution_trial_count)
        if resolved_trial_count > 0:
            return resolved_trial_count

    if len(search_result.stage_results) == 0:
        raise ValueError("winner containment debug requires at least one optimizer-v2 stage result")
    return int(search_result.stage_results[-1].num_trials)


def _print_winner_containment_debug_summary(
    patient_uid,
    roi_name,
    winner_chunk_score_result,
):
    tested_candidate_dataframe = winner_chunk_score_result.tested_candidate_dataframe
    if tested_candidate_dataframe is None or tested_candidate_dataframe.empty:
        return

    winner_row = tested_candidate_dataframe.iloc[0]
    num_trials_used = int(winner_row["Num trials used"])
    num_biopsy_sample_points = int(winner_row["Num biopsy sample points"])
    total_successes = int(winner_row["Total successes all points"])
    total_possible_successes = max(1, num_trials_used * num_biopsy_sample_points)
    print(
        "[optimizer-v2 containment] patient={} roi={} candidate={} trials={} score={:.6f} nominal={:.6f} distance_mm={:.3f} bx_pts={} total_successes={} success_rate={:.6f}".format(
            patient_uid,
            roi_name,
            int(winner_row["Candidate global index"]),
            num_trials_used,
            float(winner_row["Objective value"]),
            float(winner_row["Nominal objective value"]),
            float(winner_row["Distance to target centroid mm"]),
            num_biopsy_sample_points,
            total_successes,
            float(total_successes) / float(total_possible_successes),
        )
    )


def _normalize_requested_render_backend(render_backend):
    normalized_render_backend = str(render_backend).strip().lower()
    if normalized_render_backend == "":
        return "none"
    return normalized_render_backend


def _resolve_operational_winner_candidate_point(
    search_result,
    candidate_pool,
):
    winner_candidate_index_global = search_result.operational_winner_candidate_index_global
    if winner_candidate_index_global is None:
        return None

    candidate_points = np.asarray(candidate_pool.candidate_points, dtype=float)
    if winner_candidate_index_global < 0 or winner_candidate_index_global >= candidate_points.shape[0]:
        return None

    return candidate_points[int(winner_candidate_index_global)].reshape(3)


def _build_additional_stage_boundary_render_layers(
    structs_referenced_dict,
    pydicom_item,
    specific_structure,
    nominal_biopsy_centroid,
    winner_candidate_point,
    bx_ref,
    target_structure,
    render_include_planned_sampled_points_bool,
    render_include_planned_core_structure_bool,
    render_include_planned_centroid_line_bool,
    render_include_target_surface_bool,
    render_include_selected_anatomy_bool,
    render_layer_style_by_name,
    oar_ref,
    rectum_ref,
    urethra_ref,
):
    additional_render_layers = []
    planned_translation_vec = _resolve_planned_to_winner_translation_vector(
        nominal_biopsy_centroid,
        winner_candidate_point,
    )
    biopsy_render_color = _resolve_biopsy_render_color(
        structs_referenced_dict,
        bx_ref,
        specific_structure,
        fallback_color=np.array([0.7, 0.7, 0.7]),
    )
    target_structure_render_color = _resolve_structure_render_color(
        structs_referenced_dict,
        specific_structure.get("Relative structure type"),
        fallback_color=np.array([0.1, 0.1, 0.6]),
    )

    if render_include_planned_sampled_points_bool:
        planned_sampled_points = _coerce_optional_points_array(
            get_planned_simulated_biopsy_sampled_points_arr(specific_structure)
        )
        if planned_sampled_points is not None:
            additional_render_layers.append(
                build_point_cloud_render_layer(
                    layer_name="planned_sampled_points",
                    points=planned_sampled_points + planned_translation_vec,
                    color=_resolve_layer_style_color(
                        render_layer_style_by_name,
                        "planned_sampled_points",
                        _lighten_color(biopsy_render_color, factor=0.25),
                    ),
                    marker_size=_resolve_layer_style_float(
                        render_layer_style_by_name,
                        "planned_sampled_points",
                        "marker_size",
                    ),
                    opacity=_resolve_layer_style_float(
                        render_layer_style_by_name,
                        "planned_sampled_points",
                        "opacity",
                    ),
                )
            )

    simulated_biopsy_planning_dict = specific_structure.get("Simulated biopsy planning dict") or {}
    planned_biopsy_model_dict = get_planned_simulated_biopsy_model_dict(specific_structure)
    if render_include_planned_core_structure_bool:
        planned_core_structure_contours = _coerce_optional_point_groups(
            simulated_biopsy_planning_dict.get("Planned raw contour pts zslice list"),
            translation_vec=planned_translation_vec,
        )
        if planned_core_structure_contours is not None:
            additional_render_layers.append(
                build_contour_line_render_layer(
                    layer_name="planned_core_structure",
                    point_groups=planned_core_structure_contours,
                    color=_resolve_layer_style_color(
                        render_layer_style_by_name,
                        "planned_core_structure",
                        biopsy_render_color,
                    ),
                    line_width=_resolve_layer_style_float(
                        render_layer_style_by_name,
                        "planned_core_structure",
                        "line_width",
                    ),
                    opacity=_resolve_layer_style_float(
                        render_layer_style_by_name,
                        "planned_core_structure",
                        "opacity",
                    ),
                )
            )
        else:
            planned_core_structure_points = _coerce_optional_points_array(
                planned_biopsy_model_dict.get("Reconstructed structure pts arr")
            )
            if planned_core_structure_points is not None:
                additional_render_layers.append(
                    build_point_cloud_render_layer(
                        layer_name="planned_core_structure",
                        points=planned_core_structure_points + planned_translation_vec,
                        color=_resolve_layer_style_color(
                            render_layer_style_by_name,
                            "planned_core_structure",
                            biopsy_render_color,
                        ),
                        marker_size=_resolve_layer_style_float(
                            render_layer_style_by_name,
                            "planned_core_structure",
                            "marker_size",
                        ),
                        opacity=_resolve_layer_style_float(
                            render_layer_style_by_name,
                            "planned_core_structure",
                            "opacity",
                        ),
                    )
                )

    if render_include_planned_centroid_line_bool:
        planned_centroid_line = _coerce_optional_points_array(
            planned_biopsy_model_dict.get("Best fit line of centroid pts")
        )
        if planned_centroid_line is not None:
            additional_render_layers.append(
                build_contour_line_render_layer(
                    layer_name="planned_centroid_line",
                    point_groups=(planned_centroid_line + planned_translation_vec,),
                    color=_resolve_layer_style_color(
                        render_layer_style_by_name,
                        "planned_centroid_line",
                        _darken_color(biopsy_render_color, factor=0.35),
                    ),
                    line_width=_resolve_layer_style_float(
                        render_layer_style_by_name,
                        "planned_centroid_line",
                        "line_width",
                    ),
                    opacity=_resolve_layer_style_float(
                        render_layer_style_by_name,
                        "planned_centroid_line",
                        "opacity",
                    ),
                )
            )

    if render_include_target_surface_bool:
        target_structure_contours = _coerce_optional_point_groups(
            target_structure.get("Equal num zslice contour pts")
        )
        if target_structure_contours is not None:
            additional_render_layers.append(
                build_contour_line_render_layer(
                    layer_name="target_structure_surface",
                    point_groups=target_structure_contours,
                    color=_resolve_layer_style_color(
                        render_layer_style_by_name,
                        "target_structure_surface",
                        target_structure_render_color,
                    ),
                    line_width=_resolve_layer_style_float(
                        render_layer_style_by_name,
                        "target_structure_surface",
                        "line_width",
                    ),
                    opacity=_resolve_layer_style_float(
                        render_layer_style_by_name,
                        "target_structure_surface",
                        "opacity",
                    ),
                )
            )
        else:
            target_structure_surface_points = _coerce_optional_points_array(
                target_structure["Inter-slice interpolation information"].interpolated_pts_np_arr
            )
            if target_structure_surface_points is not None:
                additional_render_layers.append(
                    build_point_cloud_render_layer(
                        layer_name="target_structure_surface",
                        points=target_structure_surface_points,
                        color=_resolve_layer_style_color(
                            render_layer_style_by_name,
                            "target_structure_surface",
                            target_structure_render_color,
                        ),
                        marker_size=_resolve_layer_style_float(
                            render_layer_style_by_name,
                            "target_structure_surface",
                            "marker_size",
                        ),
                        opacity=_resolve_layer_style_float(
                            render_layer_style_by_name,
                            "target_structure_surface",
                            "opacity",
                        ),
                    )
                )

    raw_target_structure_centroid = target_structure.get("Structure global centroid")
    if raw_target_structure_centroid is not None:
        additional_render_layers.append(
            build_point_cloud_render_layer(
                layer_name="target_structure_centroid",
                points=np.asarray(raw_target_structure_centroid, dtype=float).reshape(1, 3),
                color=_resolve_layer_style_color(
                    render_layer_style_by_name,
                    "target_structure_centroid",
                    _darken_color(target_structure_render_color, factor=0.35),
                ),
                marker_size=_resolve_layer_style_float(
                    render_layer_style_by_name,
                    "target_structure_centroid",
                    "marker_size",
                ),
                opacity=_resolve_layer_style_float(
                    render_layer_style_by_name,
                    "target_structure_centroid",
                    "opacity",
                ),
            )
        )

    if render_include_selected_anatomy_bool:
        additional_render_layers.extend(
            _build_selected_anatomy_render_layers(
                structs_referenced_dict,
                pydicom_item,
                render_layer_style_by_name,
                oar_ref=oar_ref,
                rectum_ref=rectum_ref,
                urethra_ref=urethra_ref,
            )
        )

    return tuple(additional_render_layers)


def _resolve_planned_to_winner_translation_vector(
    nominal_biopsy_centroid,
    winner_candidate_point,
):
    if winner_candidate_point is None:
        return np.zeros(3, dtype=float)

    return np.asarray(winner_candidate_point, dtype=float).reshape(3) - np.asarray(
        nominal_biopsy_centroid,
        dtype=float,
    ).reshape(3)


def _build_selected_anatomy_render_layers(
    structs_referenced_dict,
    pydicom_item,
    render_layer_style_by_name,
    oar_ref,
    rectum_ref,
    urethra_ref,
):
    anatomy_specs = (
        ("prostate_structure", oar_ref, ("prostate",)),
        ("urethra_structure", urethra_ref, ("urethra", "ureth")),
        ("rectum_structure", rectum_ref, ("rectum", "rect")),
    )
    resolved_render_layers = []

    for layer_name, structure_ref_key, roi_fragments in anatomy_specs:
        if structure_ref_key is None or structure_ref_key not in pydicom_item:
            continue

        layer_color = _resolve_structure_render_color(
            structs_referenced_dict,
            structure_ref_key,
            fallback_color=np.array([0.55, 0.55, 0.55]),
        )

        resolved_structure = _resolve_structure_by_roi_fragments(
            pydicom_item[structure_ref_key],
            roi_fragments,
        )
        if resolved_structure is None:
            continue

        interpolation_information = resolved_structure.get("Inter-slice interpolation information")
        if interpolation_information is None:
            continue

        anatomy_contours = _coerce_optional_point_groups(
            resolved_structure.get("Equal num zslice contour pts")
        )
        if anatomy_contours is not None:
            resolved_render_layers.append(
                build_contour_line_render_layer(
                    layer_name=layer_name,
                    point_groups=anatomy_contours,
                    color=_resolve_layer_style_color(
                        render_layer_style_by_name,
                        layer_name,
                        layer_color,
                    ),
                    line_width=_resolve_layer_style_float(
                        render_layer_style_by_name,
                        layer_name,
                        "line_width",
                    ),
                    opacity=_resolve_layer_style_float(
                        render_layer_style_by_name,
                        layer_name,
                        "opacity",
                    ),
                )
            )
            continue

        anatomy_points = _coerce_optional_points_array(interpolation_information.interpolated_pts_np_arr)
        if anatomy_points is None:
            continue

        resolved_render_layers.append(
            build_point_cloud_render_layer(
                layer_name=layer_name,
                points=anatomy_points,
                color=_resolve_layer_style_color(
                    render_layer_style_by_name,
                    layer_name,
                    layer_color,
                ),
                marker_size=_resolve_layer_style_float(
                    render_layer_style_by_name,
                    layer_name,
                    "marker_size",
                ),
                opacity=_resolve_layer_style_float(
                    render_layer_style_by_name,
                    layer_name,
                    "opacity",
                ),
            )
        )

    return tuple(resolved_render_layers)


def _resolve_structure_by_roi_fragments(structure_list, roi_fragments):
    normalized_roi_fragments = tuple(str(fragment).strip().lower() for fragment in roi_fragments)
    for candidate_structure in structure_list:
        candidate_roi_name = str(candidate_structure.get("ROI", "")).strip().lower()
        if any(roi_fragment in candidate_roi_name for roi_fragment in normalized_roi_fragments):
            return candidate_structure

    return None


def _coerce_optional_points_array(points_like):
    if points_like is None:
        return None

    normalized_points = np.asarray(points_like, dtype=float)
    if normalized_points.size == 0:
        return None

    return normalized_points


def _coerce_optional_point_groups(
    point_groups_like,
    translation_vec=None,
):
    if point_groups_like is None:
        return None

    if isinstance(point_groups_like, np.ndarray):
        point_group_iterable = (point_groups_like,)
    else:
        point_group_iterable = point_groups_like

    resolved_translation_vec = np.zeros(3, dtype=float)
    if translation_vec is not None:
        resolved_translation_vec = np.asarray(translation_vec, dtype=float).reshape(3)

    normalized_point_groups = []
    for point_group in point_group_iterable:
        if point_group is None:
            continue

        normalized_group = np.asarray(point_group, dtype=float)
        if normalized_group.size == 0:
            continue
        if normalized_group.ndim != 2 or normalized_group.shape[1] != 3:
            continue
        normalized_point_groups.append(normalized_group + resolved_translation_vec)

    if len(normalized_point_groups) == 0:
        return None

    return tuple(normalized_point_groups)


def _resolve_biopsy_render_color(
    structs_referenced_dict,
    bx_ref,
    specific_structure,
    fallback_color,
):
    try:
        sim_type = specific_structure.get("Simulated type")
        if sim_type is None:
            sim_type = "Real"
        return np.asarray(
            structs_referenced_dict[bx_ref]["PCD color dict"][sim_type],
            dtype=float,
        ).reshape(3)
    except Exception:
        return np.asarray(fallback_color, dtype=float).reshape(3)


def _resolve_structure_render_color(
    structs_referenced_dict,
    structure_ref_key,
    fallback_color,
):
    try:
        return np.asarray(
            structs_referenced_dict[structure_ref_key]["PCD color"],
            dtype=float,
        ).reshape(3)
    except Exception:
        return np.asarray(fallback_color, dtype=float).reshape(3)


def _lighten_color(color, factor):
    normalized_color = np.asarray(color, dtype=float).reshape(3)
    return np.clip(normalized_color + (1.0 - normalized_color) * float(factor), 0.0, 1.0)


def _darken_color(color, factor):
    normalized_color = np.asarray(color, dtype=float).reshape(3)
    return np.clip(normalized_color * (1.0 - float(factor)), 0.0, 1.0)


def _resolve_layer_style_color(render_layer_style_by_name, layer_name, default_color):
    if render_layer_style_by_name is None:
        return np.asarray(default_color, dtype=float).reshape(3)

    layer_style = render_layer_style_by_name.get(layer_name)
    if layer_style is None or layer_style.get("color") is None:
        return np.asarray(default_color, dtype=float).reshape(3)

    return np.asarray(layer_style["color"], dtype=float).reshape(3)


def _resolve_layer_style_float(render_layer_style_by_name, layer_name, style_key):
    if render_layer_style_by_name is None:
        return None

    layer_style = render_layer_style_by_name.get(layer_name)
    if layer_style is None:
        return None

    style_value = layer_style.get(style_key)
    if style_value is None:
        return None

    return float(style_value)


def _build_transport_selection_metadata(summary_dataframe):
    if summary_dataframe.empty:
        return {}

    summary_row = summary_dataframe.iloc[0].to_dict()
    return {
        key: _normalize_scalar(value)
        for key, value in summary_row.items()
        if key.startswith("Target optimizer")
    }


def _concat_dataframes_or_none(dataframe_list):
    if len(dataframe_list) == 0:
        return None
    return pandas.concat(dataframe_list, ignore_index=True)


def _coerce_samples_to_numpy_array(samples):
    if hasattr(samples, "get"):
        samples = samples.get()
    return np.asarray(samples, dtype=float)


def _copy_zslice_list(zslice_list: Sequence[np.ndarray]) -> list[np.ndarray]:
    return [np.asarray(zslice_arr, dtype=float).copy() for zslice_arr in zslice_list]


def _normalize_scalar(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


__all__ = [
    "TARGET_DIL_OPTIMIZER_V2_DOWNSTREAM_MC_SOURCE_DF_KEY",
    "TARGET_DIL_OPTIMIZER_V2_LANE_NAME",
    "TARGET_DIL_OPTIMIZER_V2_RANKED_DF_KEY",
    "TARGET_DIL_OPTIMIZER_V2_SUMMARY_DF_KEY",
    "TARGET_DIL_OPTIMIZER_V2_TESTED_DF_KEY",
    "annotate_target_dil_optimizer_v2_outputs_with_biopsy_sampling_audit",
    "annotate_target_dil_optimizer_v2_outputs_with_downstream_mc_scores",
    "run_target_dil_optimizer_v2_for_live_simulated_family",
]