"""Live pipeline bridge for the optimizer-v2 target-DIL family."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas

import polygon_dilation_helpers_numpy
from biopsy_optimizer.v2.candidate_pool import build_target_candidate_pool
from biopsy_optimizer.v2.output import (
    build_target_dil_optimization_summary_dataframe,
    build_target_dil_ranked_candidate_output_dataframe,
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
                downstream_comparable_trial_count=None,
                return_array_as="numpy",
            )

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
    }


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
        "Target optimizer additional rescore attempts used": np.nan,
        "Target optimizer final tie candidate count": np.nan,
        "Target optimizer resolved objective value": np.nan,
        "Target optimizer resolved nominal objective value": np.nan,
        "Target optimizer downstream comparable target score": np.nan,
        "Target optimizer downstream comparable trial count": np.nan,
        "Target optimizer agreement delta": np.nan,
        "Target optimizer fallback reason": "no_ranked_candidates",
    }
    summary_dataframe = pandas.DataFrame([fallback_summary_row])
    for key, value in metadata.items():
        summary_dataframe[key] = value
    return summary_dataframe


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
    "TARGET_DIL_OPTIMIZER_V2_LANE_NAME",
    "TARGET_DIL_OPTIMIZER_V2_RANKED_DF_KEY",
    "TARGET_DIL_OPTIMIZER_V2_SUMMARY_DF_KEY",
    "run_target_dil_optimizer_v2_for_live_simulated_family",
]