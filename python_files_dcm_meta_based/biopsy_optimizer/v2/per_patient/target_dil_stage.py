"""Patient-local optimizer-v2 target-DIL stage."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd

from biopsy_optimizer.v2.candidate_pool import build_target_candidate_pool
from biopsy_optimizer.v2.live_integration import (
    TARGET_DIL_OPTIMIZER_V2_LANE_NAME,
    TARGET_DIL_OPTIMIZER_V2_RANKED_DF_KEY,
    TARGET_DIL_OPTIMIZER_V2_SUMMARY_DF_KEY,
    TARGET_DIL_OPTIMIZER_V2_TESTED_DF_KEY,
    _build_bound_biopsy_transform_bank_prefix_provider,
    _build_bound_prepared_target_relative_structures_pack_provider,
    _build_bound_target_relative_structures_nominal_plus_trials_provider,
    _build_bound_target_transform_bank_prefix_provider,
    _build_optimizer_v2_chunk_timing_details,
    _build_search_metadata,
    _build_target_centroid_fallback_summary_dataframe,
    _build_transport_selection_metadata,
    _copy_zslice_list,
    _release_optimizer_v2_target_structure_cache_entries,
    _resolve_effective_max_test_structures_per_call,
    _resolve_optimizer_v2_max_candidates_per_chunk,
    _resolve_target_dil_structure,
    _run_optimizer_v2_isolated_winner_validation_benchmark,
    _runtime_checkpoint,
    _runtime_memory_snapshot,
    _concat_dataframes_or_none,
)
from biopsy_optimizer.v2.output import (
    build_target_dil_optimization_summary_dataframe,
    build_target_dil_ranked_candidate_output_dataframe,
    build_target_dil_tested_candidate_output_dataframe,
)
from biopsy_optimizer.v2.runner import run_target_staged_candidate_search
from legacy_data_keys import legacy_data_keys
from preprocessing.biopsy_processing.simulated_biopsy_planner import (
    get_planned_simulated_biopsy_model_dict,
    get_planned_simulated_biopsy_sampled_points_arr,
)

from .live_adapter import OptimizerV2LiveConfig
from .live_adapter import build_single_patient_optimizer_v2_master_info
from .live_adapter import collect_optimizer_v2_patient_outputs


LEGACY_STRUCTURE_RECORD_KEYS = legacy_data_keys.structure_record
LEGACY_PATIENT_ALL_REFERENCE_KEYS = legacy_data_keys.patient_all_reference
LEGACY_BIOPSY_RUNTIME_KEYS = legacy_data_keys.biopsy_runtime


@dataclass(frozen=True, slots=True)
class OptimizerV2PatientStageResult:
    """Output bundle from the patient-local optimizer-v2 stage."""

    patient_uid: str
    patient_reference_dict: dict[str, Any]
    master_structure_info_dict: dict[str, Any]
    optimizer_outputs: Mapping[str, Any]
    target_structure_count: int
    resolved_max_test_structures_per_call: int | None
    resolved_max_candidates_per_chunk: int | None
    resolved_max_candidates_per_chunk_mode: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "patient_uid", str(self.patient_uid))
        object.__setattr__(self, "optimizer_outputs", dict(self.optimizer_outputs or {}))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


def _render_requested(config: OptimizerV2LiveConfig) -> bool:
    return bool(
        config.render_stage_boundary_candidate_clouds_bool
        or config.render_plotly_export_bool
        or config.render_winner_containment_debug_bool
    )


def _patient_optimizer_target_structures(patient_reference_dict: Mapping[str, Any],
                                         config: OptimizerV2LiveConfig) -> list[dict[str, Any]]:
    return [
        specific_structure
        for specific_structure in patient_reference_dict.get(config.bx_ref, ())
        if bool(specific_structure[LEGACY_STRUCTURE_RECORD_KEYS.simulated_bool_key])
        and specific_structure[LEGACY_STRUCTURE_RECORD_KEYS.simulated_type_key] == config.optimizer_simulated_type
    ]


def _store_no_optimizer_targets(patient_reference_dict: dict[str, Any],
                                all_ref_key: str) -> None:
    pre_processing_dataframe_dict = patient_reference_dict[all_ref_key][
        LEGACY_PATIENT_ALL_REFERENCE_KEYS.preprocessing_output_dataframes_key
    ]
    pre_processing_dataframe_dict[TARGET_DIL_OPTIMIZER_V2_SUMMARY_DF_KEY] = None
    pre_processing_dataframe_dict[TARGET_DIL_OPTIMIZER_V2_RANKED_DF_KEY] = None
    pre_processing_dataframe_dict[TARGET_DIL_OPTIMIZER_V2_TESTED_DF_KEY] = None


def _store_patient_optimizer_dataframes(patient_reference_dict: dict[str, Any],
                                        all_ref_key: str,
                                        summary_dataframes: list[pd.DataFrame],
                                        ranked_dataframes: list[pd.DataFrame],
                                        tested_dataframes: list[pd.DataFrame]) -> None:
    pre_processing_dataframe_dict = patient_reference_dict[all_ref_key][
        LEGACY_PATIENT_ALL_REFERENCE_KEYS.preprocessing_output_dataframes_key
    ]
    pre_processing_dataframe_dict[TARGET_DIL_OPTIMIZER_V2_SUMMARY_DF_KEY] = _concat_dataframes_or_none(
        summary_dataframes
    )
    pre_processing_dataframe_dict[TARGET_DIL_OPTIMIZER_V2_RANKED_DF_KEY] = _concat_dataframes_or_none(
        ranked_dataframes
    )
    pre_processing_dataframe_dict[TARGET_DIL_OPTIMIZER_V2_TESTED_DF_KEY] = _concat_dataframes_or_none(
        tested_dataframes
    )


def _build_target_transport_request(summary_dataframe: pd.DataFrame,
                                    target_structure_centroid: np.ndarray,
                                    *,
                                    fallback_used: bool = False) -> dict[str, Any]:
    if fallback_used:
        return {
            "Transport family": "identity",
            "Target vector": np.asarray(target_structure_centroid, dtype=float),
            "Transport source": f"{TARGET_DIL_OPTIMIZER_V2_LANE_NAME}:target_centroid_fallback",
            "Selection metadata": _build_transport_selection_metadata(summary_dataframe),
        }

    summary_row = summary_dataframe.iloc[0]
    target_vector = np.array(
        [
            summary_row["Target optimizer selected X"],
            summary_row["Target optimizer selected Y"],
            summary_row["Target optimizer selected Z"],
        ],
        dtype=float,
    )
    return {
        "Transport family": "identity",
        "Target vector": target_vector,
        "Transport source": TARGET_DIL_OPTIMIZER_V2_LANE_NAME,
        "Selection metadata": _build_transport_selection_metadata(summary_dataframe),
    }


def run_patient_target_dil_optimizer_v2_stage(
    *,
    patient_uid: str,
    patient_reference_dict: dict[str, Any],
    patient_info_dict: Mapping[str, Any] | None,
    config: OptimizerV2LiveConfig,
    parallel_pool: Any,
    global_info: Mapping[str, Any] | None = None,
    resolved_max_test_structures_per_call: int | None = None,
    resolved_max_candidates_per_chunk: int | None = None,
) -> OptimizerV2PatientStageResult:
    """Run optimizer-v2 scientific targeting for one patient without cohort UI routing."""
    if _render_requested(config):
        raise ValueError(
            "Optimizer-v2 patient scientific stage does not render; use the live adapter for render validation."
        )

    patient_uid = str(patient_uid)
    master_structure_reference_dict = {patient_uid: patient_reference_dict}
    master_structure_info_dict = build_single_patient_optimizer_v2_master_info(
        patient_uid,
        patient_info_dict,
        global_info=global_info,
    )
    optimizer_target_structures = _patient_optimizer_target_structures(patient_reference_dict, config)
    if not optimizer_target_structures:
        _store_no_optimizer_targets(patient_reference_dict, config.all_ref_key)
        return OptimizerV2PatientStageResult(
            patient_uid=patient_uid,
            patient_reference_dict=patient_reference_dict,
            master_structure_info_dict=master_structure_info_dict,
            optimizer_outputs=collect_optimizer_v2_patient_outputs(
                patient_reference_dict,
                bx_ref=config.bx_ref,
                all_ref_key=config.all_ref_key,
            ),
            target_structure_count=0,
            resolved_max_test_structures_per_call=resolved_max_test_structures_per_call,
            resolved_max_candidates_per_chunk=resolved_max_candidates_per_chunk,
            resolved_max_candidates_per_chunk_mode=None,
            metadata={"optimizer_simulated_type": config.optimizer_simulated_type},
        )

    if resolved_max_test_structures_per_call is None:
        _runtime_checkpoint(
            "optimizer_v2.patient_stage.calibration.start",
            "Resolving patient optimizer-v2 containment call budget.",
            patient_uid=patient_uid,
            details={
                "requested_max_test_structures_per_call": config.max_test_structures_per_call,
                "fallback_max_test_structures_per_call": config.fallback_max_test_structures_per_call,
                "auto_calibrate_max_test_structures_per_call": bool(
                    config.auto_calibrate_max_test_structures_per_call
                ),
                "verify_calibrated_max_test_structures_per_call": bool(
                    config.verify_calibrated_max_test_structures_per_call
                ),
                "downstream_comparable_trial_count": config.downstream_comparable_trial_count,
            },
        )
        resolved_max_test_structures_per_call = _resolve_effective_max_test_structures_per_call(
            master_structure_reference_dict=master_structure_reference_dict,
            bx_ref=config.bx_ref,
            dil_ref=config.dil_ref,
            optimizer_simulated_type=config.optimizer_simulated_type,
            search_config=config.search_config,
            parallel_pool=parallel_pool,
            constant_z_slice_polygons_handler_option=config.constant_z_slice_polygons_handler_option,
            remove_consecutive_duplicate_points_in_polygons=config.remove_consecutive_duplicate_points_in_polygons,
            include_edges_in_log=config.include_edges_in_log,
            kernel_type=config.kernel_type,
            max_test_structures_per_call=config.max_test_structures_per_call,
            fallback_max_test_structures_per_call=config.fallback_max_test_structures_per_call,
            auto_calibrate_max_test_structures_per_call=config.auto_calibrate_max_test_structures_per_call,
            verify_calibrated_max_test_structures_per_call=(
                config.verify_calibrated_max_test_structures_per_call
            ),
            downstream_comparable_trial_count=config.downstream_comparable_trial_count,
        )

    if resolved_max_candidates_per_chunk is None:
        resolved_max_candidates_per_chunk, resolved_max_candidates_per_chunk_mode = (
            _resolve_optimizer_v2_max_candidates_per_chunk(
                requested_max_candidates_per_chunk=config.max_candidates_per_chunk,
                resolved_max_test_structures_per_call=resolved_max_test_structures_per_call,
                search_config=config.search_config,
                downstream_comparable_trial_count=config.downstream_comparable_trial_count,
                include_nominal=True,
            )
        )
    else:
        resolved_max_candidates_per_chunk_mode = "provided"

    candidate_pool_cache: dict[int, Any] = {}
    target_structure_pack_cache: dict[Any, Any] = {}
    target_structure_prepared_pack_cache: dict[Any, Any] = {}
    patient_summary_dataframes: list[pd.DataFrame] = []
    patient_ranked_dataframes: list[pd.DataFrame] = []
    patient_tested_dataframes: list[pd.DataFrame] = []

    for specific_structure in optimizer_target_structures:
        structure_id = specific_structure[LEGACY_STRUCTURE_RECORD_KEYS.roi_key]
        target_structure = _resolve_target_dil_structure(
            patient_reference_dict,
            specific_structure,
            config.dil_ref,
        )
        target_structure_cache_key = int(target_structure[LEGACY_STRUCTURE_RECORD_KEYS.ref_number_key])
        _runtime_checkpoint(
            "optimizer_v2.patient_stage.structure.prepare",
            "Preparing patient optimizer-v2 target structure.",
            patient_uid=patient_uid,
            structure_id=structure_id,
            details={"target_structure_ref": target_structure_cache_key},
        )

        candidate_pool = candidate_pool_cache.get(target_structure_cache_key)
        if candidate_pool is None:
            candidate_pool_build_start_time = time.perf_counter()
            candidate_pool = build_target_candidate_pool(
                target_points_array=np.asarray(
                    target_structure["Inter-slice interpolation information"].interpolated_pts_np_arr,
                    dtype=float,
                ),
                target_zslices_list=_copy_zslice_list(
                    target_structure["Inter-slice interpolation information"].interpolated_pts_list
                ),
                search_config=config.search_config,
                constant_z_slice_polygons_handler_option=config.constant_z_slice_polygons_handler_option,
                remove_consecutive_duplicate_points_in_polygons=(
                    config.remove_consecutive_duplicate_points_in_polygons
                ),
                kernel_type=config.kernel_type,
                include_edges_in_log=config.include_edges_in_log,
            )
            candidate_pool_cache[target_structure_cache_key] = candidate_pool
            _runtime_checkpoint(
                "optimizer_v2.patient_stage.structure.candidate_pool.end",
                "Built patient optimizer-v2 candidate pool.",
                patient_uid=patient_uid,
                structure_id=structure_id,
                details={
                    "candidate_count": int(np.asarray(candidate_pool.candidate_points).shape[0]),
                    "elapsed_seconds": round(time.perf_counter() - candidate_pool_build_start_time, 3),
                },
            )

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

        biopsy_transform_bank_prefix_provider = _build_bound_biopsy_transform_bank_prefix_provider(
            specific_structure
        )
        target_transform_bank_prefix_provider = _build_bound_target_transform_bank_prefix_provider(
            target_structure
        )
        target_relative_structures_nominal_plus_trials_provider = (
            _build_bound_target_relative_structures_nominal_plus_trials_provider(
                target_structure=target_structure,
                target_structure_cache_key=target_structure_cache_key,
                target_structure_pack_cache=target_structure_pack_cache,
                parallel_pool=parallel_pool,
                patient_uid=patient_uid,
                structure_id=structure_id,
            )
        )
        prepared_target_relative_structures_pack_provider = (
            _build_bound_prepared_target_relative_structures_pack_provider(
                target_structure_cache_key=target_structure_cache_key,
                target_structure_prepared_pack_cache=target_structure_prepared_pack_cache,
                target_relative_structures_nominal_plus_trials_provider=(
                    target_relative_structures_nominal_plus_trials_provider
                ),
                patient_uid=patient_uid,
                structure_id=structure_id,
            )
        )

        _runtime_checkpoint(
            "optimizer_v2.patient_stage.structure.search.start",
            "Starting patient optimizer-v2 staged candidate search.",
            patient_uid=patient_uid,
            structure_id=structure_id,
            details={
                "candidate_count": int(np.asarray(candidate_pool.candidate_points).shape[0]),
                "resolved_max_test_structures_per_call": resolved_max_test_structures_per_call,
                "resolved_max_candidates_per_chunk": resolved_max_candidates_per_chunk,
                "resolved_max_candidates_per_chunk_mode": resolved_max_candidates_per_chunk_mode,
                "downstream_comparable_trial_count": config.downstream_comparable_trial_count,
            },
        )
        search_start_time = time.perf_counter()
        search_result = run_target_staged_candidate_search(
            candidate_pool=candidate_pool,
            search_config=config.search_config,
            nominal_biopsy_points=nominal_biopsy_points,
            nominal_biopsy_centroid=nominal_biopsy_centroid,
            nominal_biopsy_centroid_line=nominal_biopsy_centroid_line,
            biopsy_transform_bank_prefix_provider=biopsy_transform_bank_prefix_provider,
            target_relative_structures_nominal_plus_trials_provider=(
                target_relative_structures_nominal_plus_trials_provider
            ),
            prepared_target_relative_structures_pack_provider=(
                prepared_target_relative_structures_pack_provider
            ),
            target_structure_centroid=target_structure_centroid,
            target_transform_bank_prefix_provider=target_transform_bank_prefix_provider,
            max_candidates_per_chunk=resolved_max_candidates_per_chunk,
            max_test_structures_per_call=resolved_max_test_structures_per_call,
            validate_nearest_z_helper_against_ver5=config.validate_nearest_z_helper_against_ver5,
            include_edges_in_log=config.include_edges_in_log,
            kernel_type=config.kernel_type,
            downstream_comparable_trial_count=config.downstream_comparable_trial_count,
            return_array_as="numpy",
        )
        search_elapsed_seconds = time.perf_counter() - search_start_time
        stage_total_elapsed_seconds = sum(
            float(stage_result.total_elapsed_seconds)
            for stage_result in search_result.stage_results
        )
        winner_resolution_elapsed_seconds = float(search_result.winner_resolution_elapsed_seconds)
        winner_validation_elapsed_seconds = float(search_result.winner_validation_elapsed_seconds)
        winner_resolution_chunk_score_result = (
            search_result.winner_resolution_result.chunk_score_result
            if search_result.winner_resolution_result is not None
            else None
        )
        winner_validation_chunk_score_result = (
            search_result.winner_validation_result.chunk_score_result
            if search_result.winner_validation_result is not None
            else None
        )
        _runtime_checkpoint(
            "optimizer_v2.patient_stage.structure.search.end",
            "Completed patient optimizer-v2 staged candidate search.",
            patient_uid=patient_uid,
            structure_id=structure_id,
            details={
                "stage_count": len(search_result.stage_results),
                "winner_candidate_index_global": search_result.operational_winner_candidate_index_global,
                "elapsed_seconds": round(search_elapsed_seconds, 3),
                "stage_total_elapsed_seconds": round(stage_total_elapsed_seconds, 3),
                "winner_resolution_elapsed_seconds": round(winner_resolution_elapsed_seconds, 3),
                "winner_validation_elapsed_seconds": round(winner_validation_elapsed_seconds, 3),
                "winner_resolution_chunk_timing": _build_optimizer_v2_chunk_timing_details(
                    winner_resolution_chunk_score_result
                ),
                "winner_validation_chunk_timing": _build_optimizer_v2_chunk_timing_details(
                    winner_validation_chunk_score_result
                ),
            },
        )

        if config.benchmark_isolated_winner_validation_bool:
            _run_optimizer_v2_isolated_winner_validation_benchmark(
                patient_uid=patient_uid,
                structure_id=structure_id,
                search_result=search_result,
                candidate_pool=candidate_pool,
                nominal_biopsy_points=nominal_biopsy_points,
                nominal_biopsy_centroid=nominal_biopsy_centroid,
                nominal_biopsy_centroid_line=nominal_biopsy_centroid_line,
                biopsy_transform_bank_prefix_provider=biopsy_transform_bank_prefix_provider,
                target_relative_structures_nominal_plus_trials_provider=(
                    target_relative_structures_nominal_plus_trials_provider
                ),
                prepared_target_relative_structures_pack_provider=(
                    prepared_target_relative_structures_pack_provider
                ),
                target_structure_centroid=target_structure_centroid,
                target_transform_bank_prefix_provider=target_transform_bank_prefix_provider,
                downstream_comparable_trial_count=config.downstream_comparable_trial_count,
                resolved_max_test_structures_per_call=resolved_max_test_structures_per_call,
                validate_nearest_z_helper_against_ver5=config.validate_nearest_z_helper_against_ver5,
                include_edges_in_log=config.include_edges_in_log,
                kernel_type=config.kernel_type,
            )

        dataframe_build_start_time = time.perf_counter()
        metadata = _build_search_metadata(
            patient_uid,
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
        fallback_used = summary_dataframe.empty
        if fallback_used:
            summary_dataframe = _build_target_centroid_fallback_summary_dataframe(
                target_structure_centroid,
                metadata,
            )
            ranked_candidate_dataframe = pd.DataFrame()
            tested_candidate_dataframe = pd.DataFrame()

        specific_structure[LEGACY_BIOPSY_RUNTIME_KEYS.simulated_biopsy_transport_request_key] = (
            _build_target_transport_request(
                summary_dataframe,
                target_structure_centroid,
                fallback_used=fallback_used,
            )
        )
        patient_summary_dataframes.append(summary_dataframe)
        if not ranked_candidate_dataframe.empty:
            patient_ranked_dataframes.append(ranked_candidate_dataframe)
        if not tested_candidate_dataframe.empty:
            patient_tested_dataframes.append(tested_candidate_dataframe)

        _runtime_checkpoint(
            "optimizer_v2.patient_stage.structure.outputs.end",
            "Built patient optimizer-v2 structure outputs.",
            patient_uid=patient_uid,
            structure_id=structure_id,
            details={
                "summary_rows": len(summary_dataframe),
                "ranked_rows": len(ranked_candidate_dataframe),
                "tested_rows": len(tested_candidate_dataframe),
                "elapsed_seconds": round(time.perf_counter() - dataframe_build_start_time, 3),
            },
        )

        _runtime_memory_snapshot(
            "optimizer_v2.patient_stage.structure.cache_release.memory.before",
            "Captured memory snapshot before releasing patient optimizer-v2 per-target caches.",
            patient_uid=patient_uid,
            structure_id=structure_id,
            details={
                "candidate_pool_cache_entries": int(len(candidate_pool_cache)),
                "target_structure_pack_cache_entries": int(len(target_structure_pack_cache)),
                "target_structure_prepared_pack_cache_entries": int(len(target_structure_prepared_pack_cache)),
            },
        )
        released_cache_details = _release_optimizer_v2_target_structure_cache_entries(
            target_structure_cache_key=target_structure_cache_key,
            candidate_pool_cache=candidate_pool_cache,
            target_structure_pack_cache=target_structure_pack_cache,
            target_structure_prepared_pack_cache=target_structure_prepared_pack_cache,
        )
        _runtime_memory_snapshot(
            "optimizer_v2.patient_stage.structure.cache_release.memory.after",
            "Captured memory snapshot after releasing patient optimizer-v2 per-target caches.",
            patient_uid=patient_uid,
            structure_id=structure_id,
            details=released_cache_details,
        )

    _store_patient_optimizer_dataframes(
        patient_reference_dict,
        config.all_ref_key,
        patient_summary_dataframes,
        patient_ranked_dataframes,
        patient_tested_dataframes,
    )
    return OptimizerV2PatientStageResult(
        patient_uid=patient_uid,
        patient_reference_dict=patient_reference_dict,
        master_structure_info_dict=master_structure_info_dict,
        optimizer_outputs=collect_optimizer_v2_patient_outputs(
            patient_reference_dict,
            bx_ref=config.bx_ref,
            all_ref_key=config.all_ref_key,
        ),
        target_structure_count=len(optimizer_target_structures),
        resolved_max_test_structures_per_call=resolved_max_test_structures_per_call,
        resolved_max_candidates_per_chunk=resolved_max_candidates_per_chunk,
        resolved_max_candidates_per_chunk_mode=resolved_max_candidates_per_chunk_mode,
        metadata={"optimizer_simulated_type": config.optimizer_simulated_type},
    )