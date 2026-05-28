from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np
import pandas

from guidance_maps.config import GuidanceMapPlanningConfig
from legacy_data_keys import legacy_data_keys


LEGACY_PATIENT_ALL_REFERENCE_KEYS = legacy_data_keys.patient_all_reference
PATIENT_GUIDANCE_MAP_FIRING_DEPTH_DF_KEY = (
    "Biopsy optimization - Guidance-map firing depth recommendations dataframe"
)
COHORT_GUIDANCE_MAP_FIRING_DEPTH_DF_KEY = (
    "Cohort: Guidance-map firing depth recommendations dataframe"
)


@dataclass(frozen=True)
class GuidanceMapPatientPrecomputeResult:
    patient_uid: str
    patient_dataframe_key: str
    patient_row_count: int
    dataframe: pandas.DataFrame
    elapsed_seconds: float


@dataclass(frozen=True)
class GuidanceMapPlanningResult:
    patient_dataframe_key: str
    cohort_dataframe_key: str
    patient_dataframe_count: int
    patient_row_count: int
    cohort_row_count: int
    elapsed_seconds: float


def _empty_dataframe() -> pandas.DataFrame:
    return pandas.DataFrame()


def _downcast_guidance_dataframe(dataframe: pandas.DataFrame, threshold: float) -> pandas.DataFrame:
    import dataframe_builders

    if not isinstance(dataframe, pandas.DataFrame) or dataframe.empty:
        return _empty_dataframe()
    return dataframe_builders.convert_columns_to_categorical_and_downcast(
        dataframe,
        threshold=threshold,
        ignore_types=(np.floating,),
    )


def _log_patient_result(runtime_logger: Any,
                        patient_uid: str,
                        row_count: int,
                        elapsed_seconds: float) -> None:
    if runtime_logger is None:
        return
    runtime_logger.checkpoint(
        "guidance_maps.precompute.patient",
        "Precomputed guidance-map firing-depth recommendations for patient.",
        patient_uid=patient_uid,
        details={
            "row_count": int(row_count),
            "elapsed_seconds": round(float(elapsed_seconds), 3),
        },
    )


def precompute_guidance_map_firing_depth_recommendations_for_patient(
    *,
    patient_uid: str,
    pydicom_item: dict[str, Any],
    dil_ref: str,
    all_ref_key: str,
    oar_ref: str,
    rectum_ref: str,
    biopsy_fire_travel_distances,
    biopsy_needle_compartment_length: float,
    interp_inter_slice_dist: float,
    interp_intra_slice_dist: float,
    radius_for_normals_estimation: float,
    max_nn_for_normals_estimation: int,
    biopsy_needle_tip_length: float,
    planning_config: GuidanceMapPlanningConfig | None = None,
    runtime_logger: Any = None,
) -> GuidanceMapPatientPrecomputeResult:
    """Precompute the guidance-map firing-depth table for one patient."""
    if planning_config is None:
        planning_config = GuidanceMapPlanningConfig()

    import advanced_guidance_map_creator

    patient_start_time = time.perf_counter()
    patient_dataframe = advanced_guidance_map_creator.precompute_guidance_map_firing_depths_for_patient(
        patientUID=patient_uid,
        pydicom_item=pydicom_item,
        dil_ref=dil_ref,
        all_ref_key=all_ref_key,
        oar_ref=oar_ref,
        rectum_ref=rectum_ref,
        biopsy_fire_travel_distances=biopsy_fire_travel_distances,
        biopsy_needle_compartment_length=biopsy_needle_compartment_length,
        interp_inter_slice_dist=interp_inter_slice_dist,
        interp_intra_slice_dist=interp_intra_slice_dist,
        radius_for_normals_estimation=radius_for_normals_estimation,
        max_nn_for_normals_estimation=max_nn_for_normals_estimation,
        biopsy_needle_tip_length=biopsy_needle_tip_length,
        candidate_holes_k=planning_config.candidate_holes_k,
        candidate_axis_line_length_mm=planning_config.candidate_axis_line_length_mm,
    )
    patient_dataframe = _downcast_guidance_dataframe(
        patient_dataframe,
        threshold=planning_config.downcast_threshold,
    )
    pydicom_item[all_ref_key][LEGACY_PATIENT_ALL_REFERENCE_KEYS.preprocessing_output_dataframes_key][
        PATIENT_GUIDANCE_MAP_FIRING_DEPTH_DF_KEY
    ] = patient_dataframe

    elapsed_seconds = time.perf_counter() - patient_start_time
    row_count = int(len(patient_dataframe))
    _log_patient_result(runtime_logger, patient_uid, row_count, elapsed_seconds)
    return GuidanceMapPatientPrecomputeResult(
        patient_uid=str(patient_uid),
        patient_dataframe_key=PATIENT_GUIDANCE_MAP_FIRING_DEPTH_DF_KEY,
        patient_row_count=row_count,
        dataframe=patient_dataframe,
        elapsed_seconds=elapsed_seconds,
    )


def precompute_guidance_map_firing_depth_recommendations_for_run(
    *,
    master_structure_reference_dict: dict[str, Any],
    master_cohort_patient_data_and_dataframes: dict[str, Any],
    dil_ref: str,
    all_ref_key: str,
    oar_ref: str,
    rectum_ref: str,
    biopsy_fire_travel_distances,
    biopsy_needle_compartment_length: float,
    interp_inter_slice_dist: float,
    interp_intra_slice_dist: float,
    radius_for_normals_estimation: float,
    max_nn_for_normals_estimation: int,
    biopsy_needle_tip_length: float,
    planning_config: GuidanceMapPlanningConfig | None = None,
    runtime_logger: Any = None,
) -> GuidanceMapPlanningResult:
    """Precompute patient and cohort guidance-map firing-depth recommendation tables.

    The current monolithic pipeline calls this after simulated cores are finalized and
    before guidance-map rendering. Future entrypoints can call the same function after
    loading or constructing equivalent patient geometry dictionaries.
    """
    if planning_config is None:
        planning_config = GuidanceMapPlanningConfig()

    import dataframe_builders

    start_time = time.perf_counter()
    patient_dataframe_count = 0
    patient_row_count = 0
    num_cases = len(master_structure_reference_dict)

    if runtime_logger is not None:
        runtime_logger.phase_start(
            "guidance_maps.precompute",
            "Precomputing guidance-map firing-depth recommendations.",
            details={
                "num_cases": num_cases,
                "candidate_holes_k": int(planning_config.candidate_holes_k),
                "candidate_axis_line_length_mm": float(planning_config.candidate_axis_line_length_mm),
            },
        )

    for patient_uid, pydicom_item in master_structure_reference_dict.items():
        patient_result = precompute_guidance_map_firing_depth_recommendations_for_patient(
            patient_uid=patient_uid,
            pydicom_item=pydicom_item,
            dil_ref=dil_ref,
            all_ref_key=all_ref_key,
            oar_ref=oar_ref,
            rectum_ref=rectum_ref,
            biopsy_fire_travel_distances=biopsy_fire_travel_distances,
            biopsy_needle_compartment_length=biopsy_needle_compartment_length,
            interp_inter_slice_dist=interp_inter_slice_dist,
            interp_intra_slice_dist=interp_intra_slice_dist,
            radius_for_normals_estimation=radius_for_normals_estimation,
            max_nn_for_normals_estimation=max_nn_for_normals_estimation,
            biopsy_needle_tip_length=biopsy_needle_tip_length,
            planning_config=planning_config,
            runtime_logger=runtime_logger,
        )
        row_count = patient_result.patient_row_count
        if row_count > 0:
            patient_dataframe_count += 1
            patient_row_count += row_count

    cohort_dataframe = dataframe_builders.cohort_guidance_map_firing_depth_recommendations_dataframe_builder(
        master_structure_reference_dict,
        all_ref_key,
        dil_ref,
        downcast_threshold=planning_config.downcast_threshold,
    )
    master_cohort_patient_data_and_dataframes["Dataframes"][
        COHORT_GUIDANCE_MAP_FIRING_DEPTH_DF_KEY
    ] = cohort_dataframe

    elapsed_seconds = time.perf_counter() - start_time
    result = GuidanceMapPlanningResult(
        patient_dataframe_key=PATIENT_GUIDANCE_MAP_FIRING_DEPTH_DF_KEY,
        cohort_dataframe_key=COHORT_GUIDANCE_MAP_FIRING_DEPTH_DF_KEY,
        patient_dataframe_count=patient_dataframe_count,
        patient_row_count=patient_row_count,
        cohort_row_count=int(len(cohort_dataframe)),
        elapsed_seconds=elapsed_seconds,
    )

    if runtime_logger is not None:
        runtime_logger.phase_end(
            "guidance_maps.precompute",
            "Completed guidance-map firing-depth precompute.",
            details={
                "num_cases": num_cases,
                "patient_dataframe_count": result.patient_dataframe_count,
                "patient_row_count": result.patient_row_count,
                "cohort_row_count": result.cohort_row_count,
                "elapsed_seconds": round(result.elapsed_seconds, 3),
            },
            clear_phase=True,
        )

    return result