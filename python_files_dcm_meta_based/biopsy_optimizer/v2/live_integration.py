"""Live pipeline bridge for the optimizer-v2 target-DIL family."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas

import polygon_dilation_helpers_numpy
from biopsy_optimizer.v2.candidate_pool import build_target_candidate_pool
from biopsy_optimizer.v2.output import (
    annotate_target_dil_optimizer_dataframe_with_biopsy_sampling_audit,
    annotate_target_dil_optimizer_dataframe_with_downstream_mc,
    build_target_dil_optimization_summary_dataframe,
    build_target_dil_ranked_candidate_output_dataframe,
)
from biopsy_optimizer.v2.render import (
    build_contour_line_render_layer,
    build_point_cloud_render_layer,
    build_stage_boundary_render_jobs,
    render_scene_render_jobs,
)
from biopsy_optimizer.v2.runner import run_target_staged_candidate_search
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
TARGET_DIL_OPTIMIZER_V2_STAGE_BOUNDARY_RENDER_JOBS_KEY = (
    "Biopsy optimization - Target DIL optimizer v2 stage boundary render jobs"
)
TARGET_DIL_OPTIMIZER_V2_DOWNSTREAM_MC_SOURCE_DF_KEY = (
    "Tissue class - Global tissue by structure statistics"
)


def run_target_dil_optimizer_v2_for_live_simulated_family(
    master_structure_reference_dict,
    master_structure_info_dict,
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
    downstream_comparable_trial_count=None,
    render_stage_boundary_candidate_clouds_bool=False,
    render_stage_names_to_render=None,
    render_backend="open3d",
    render_patient_whitelist=None,
    render_roi_whitelist=None,
    render_include_planned_sampled_points_bool=True,
    render_include_planned_core_structure_bool=True,
    render_include_planned_centroid_line_bool=True,
    render_include_target_surface_bool=False,
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
                pydicom_item=pydicom_item,
                specific_structure=specific_structure,
                nominal_biopsy_centroid=nominal_biopsy_centroid,
                winner_candidate_point=winner_candidate_point,
                target_structure=target_structure,
                render_include_planned_sampled_points_bool=render_include_planned_sampled_points_bool,
                render_include_planned_core_structure_bool=render_include_planned_core_structure_bool,
                render_include_planned_centroid_line_bool=render_include_planned_centroid_line_bool,
                render_include_target_surface_bool=render_include_target_surface_bool,
                render_include_selected_anatomy_bool=render_include_selected_anatomy_bool,
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
            if (
                render_stage_boundary_candidate_clouds_bool
                and should_render_structure
                and stage_boundary_render_jobs
            ):
                live_display.stop()
                try:
                    render_scene_render_jobs(
                        stage_boundary_render_jobs,
                        render_backend=render_backend,
                    )
                finally:
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

            if summary_dataframe.empty:
                summary_dataframe = _build_target_centroid_fallback_summary_dataframe(
                    target_structure_centroid,
                    metadata,
                )
                ranked_candidate_dataframe = pandas.DataFrame()
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

            structures_progress.update(processing_structures_task, advance=1)

        structures_progress.remove_task(processing_structures_task)

        pydicom_item[all_ref_key]["Multi-structure pre-processing output dataframes dict"][
            TARGET_DIL_OPTIMIZER_V2_SUMMARY_DF_KEY
        ] = _concat_dataframes_or_none(patient_summary_dataframes)
        pydicom_item[all_ref_key]["Multi-structure pre-processing output dataframes dict"][
            TARGET_DIL_OPTIMIZER_V2_RANKED_DF_KEY
        ] = _concat_dataframes_or_none(patient_ranked_dataframes)

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
    pydicom_item,
    specific_structure,
    nominal_biopsy_centroid,
    winner_candidate_point,
    target_structure,
    render_include_planned_sampled_points_bool,
    render_include_planned_core_structure_bool,
    render_include_planned_centroid_line_bool,
    render_include_target_surface_bool,
    render_include_selected_anatomy_bool,
    oar_ref,
    rectum_ref,
    urethra_ref,
):
    additional_render_layers = []
    planned_translation_vec = _resolve_planned_to_winner_translation_vector(
        nominal_biopsy_centroid,
        winner_candidate_point,
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
                    color=np.array([0.0, 0.8, 0.8]),
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
                    color=np.array([0.7, 0.7, 0.7]),
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
                        color=np.array([0.7, 0.7, 0.7]),
                    )
                )

    if render_include_planned_centroid_line_bool:
        planned_centroid_line = _coerce_optional_points_array(
            planned_biopsy_model_dict.get("Best fit line of centroid pts")
        )
        if planned_centroid_line is not None:
            additional_render_layers.append(
                build_point_cloud_render_layer(
                    layer_name="planned_centroid_line",
                    points=planned_centroid_line + planned_translation_vec,
                    color=np.array([0.4, 0.0, 0.8]),
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
                    color=np.array([0.1, 0.1, 0.6]),
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
                        color=np.array([0.1, 0.1, 0.6]),
                    )
                )

    if render_include_selected_anatomy_bool:
        additional_render_layers.extend(
            _build_selected_anatomy_render_layers(
                pydicom_item,
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
    pydicom_item,
    oar_ref,
    rectum_ref,
    urethra_ref,
):
    anatomy_specs = (
        ("prostate_structure", oar_ref, ("prostate",), np.array([0.55, 0.55, 0.55])),
        ("urethra_structure", urethra_ref, ("urethra", "ureth"), np.array([1.0, 0.85, 0.0])),
        ("rectum_structure", rectum_ref, ("rectum", "rect"), np.array([0.8, 0.35, 0.0])),
    )
    resolved_render_layers = []

    for layer_name, structure_ref_key, roi_fragments, layer_color in anatomy_specs:
        if structure_ref_key is None or structure_ref_key not in pydicom_item:
            continue

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
                    color=layer_color,
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
                color=layer_color,
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
    "annotate_target_dil_optimizer_v2_outputs_with_biopsy_sampling_audit",
    "annotate_target_dil_optimizer_v2_outputs_with_downstream_mc_scores",
    "run_target_dil_optimizer_v2_for_live_simulated_family",
]