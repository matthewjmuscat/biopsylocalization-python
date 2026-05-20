"""Batch orchestration for patient-local runs carved from legacy state."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter
from typing import Any, Mapping, MutableMapping, Sequence

from .contracts import LegacyCohortRuntimeState
from .contracts import PatientBatchRunConfig
from .contracts import PatientBatchExecutionBackend
from .contracts import PatientBatchRunResult
from .contracts import PatientCase
from .contracts import PatientRunResult
from .contracts import PatientStageName
from .contracts import PatientStageResult
from .contracts import resolve_legacy_patient_uids
from .legacy_bridge import carve_patient_runtime_state_by_uid
from .runner import PatientStage
from .runner import run_patient_case


def resolve_patient_uids(master_structure_reference_dict: Mapping[str, Any],
                         patient_uids: Sequence[str] = ()) -> tuple[str, ...]:
    """Resolve the exact ordered patient UID list for a batch run.

    An empty requested list means all patient keys currently present in the
    legacy reference dictionary. Explicit requests are validated up front so a
    batch does not silently skip misspelled or stale patient IDs. Patient IDs are
    preserved exactly because they are lookup keys in the legacy dictionaries.
    """
    return resolve_legacy_patient_uids(master_structure_reference_dict, patient_uids)


def run_patient_batch(legacy_cohort_state: LegacyCohortRuntimeState,
                      batch_config: PatientBatchRunConfig,
                      stages: Sequence[PatientStage] | None = None) -> PatientBatchRunResult:
    """Run patient-local cases from an explicit legacy cohort boundary."""
    if not isinstance(legacy_cohort_state, LegacyCohortRuntimeState):
        raise TypeError("legacy_cohort_state must be a LegacyCohortRuntimeState instance")
    if legacy_cohort_state.legacy_keys != batch_config.legacy_keys:
        raise ValueError("legacy_cohort_state.legacy_keys must match batch_config.legacy_keys")

    patient_uids = legacy_cohort_state.resolve_patient_uids(batch_config.patient_uids)
    start_time = perf_counter()
    if batch_config.execution_backend == PatientBatchExecutionBackend.SEQUENTIAL or len(patient_uids) <= 1:
        patient_results = tuple(
            _run_one_patient_from_legacy(
                patient_uid,
                legacy_cohort_state,
                batch_config,
                stages,
            )
            for patient_uid in patient_uids
        )
    elif batch_config.execution_backend == PatientBatchExecutionBackend.THREAD:
        patient_results = _run_patient_batch_threaded(
            patient_uids,
            legacy_cohort_state,
            batch_config,
            stages,
        )
    else:
        raise ValueError(f"Unsupported patient batch backend: {batch_config.execution_backend}")

    return PatientBatchRunResult.from_patient_results(
        batch_config.output_root,
        patient_results,
        elapsed_seconds=perf_counter() - start_time,
        metadata=_batch_result_metadata(batch_config, len(patient_uids)),
    )


def run_patient_batch_from_legacy(master_structure_reference_dict: MutableMapping[str, Any],
                                  master_structure_info_dict: MutableMapping[str, Any],
                                  batch_config: PatientBatchRunConfig,
                                  stages: Sequence[PatientStage] | None = None) -> PatientBatchRunResult:
    """Run a batch of patient-local cases from raw legacy dictionaries.

    This compatibility entrypoint immediately wraps the raw dictionaries in a
    `LegacyCohortRuntimeState` so the rest of the runner works against a named
    transitional boundary.
    """
    legacy_cohort_state = LegacyCohortRuntimeState(
        master_structure_reference_dict=master_structure_reference_dict,
        master_structure_info_dict=master_structure_info_dict,
        legacy_keys=batch_config.legacy_keys,
        metadata=batch_config.metadata,
    )
    return run_patient_batch(legacy_cohort_state, batch_config, stages=stages)


def _batch_result_metadata(batch_config: PatientBatchRunConfig, patient_count: int) -> dict[str, Any]:
    metadata = dict(batch_config.metadata)
    metadata.update(
        {
            "run_id": batch_config.run_id,
            "patient_count": patient_count,
            "max_workers": batch_config.max_workers,
            "execution_backend": batch_config.execution_backend.value,
        }
    )
    if batch_config.source_run_id:
        metadata["source_run_id"] = batch_config.source_run_id
    if batch_config.input_manifest_id:
        metadata["input_manifest_id"] = batch_config.input_manifest_id
    return metadata


def _run_patient_batch_threaded(patient_uids: Sequence[str],
                                legacy_cohort_state: LegacyCohortRuntimeState,
                                batch_config: PatientBatchRunConfig,
                                stages: Sequence[PatientStage] | None) -> tuple[PatientRunResult, ...]:
    worker_count = min(batch_config.max_workers, len(patient_uids))
    ordered_results: list[PatientRunResult | None] = [None] * len(patient_uids)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_index = {
            executor.submit(
                _run_one_patient_from_legacy,
                patient_uid,
                legacy_cohort_state,
                batch_config,
                stages,
            ): index
            for index, patient_uid in enumerate(patient_uids)
        }
        for future in as_completed(future_to_index):
            ordered_results[future_to_index[future]] = future.result()

    resolved_results: list[PatientRunResult] = []
    for patient_result in ordered_results:
        if patient_result is None:
            raise RuntimeError("patient batch ended without a result for one patient")
        resolved_results.append(patient_result)
    return tuple(resolved_results)


def _run_one_patient_from_legacy(patient_uid: str,
                                 legacy_cohort_state: LegacyCohortRuntimeState,
                                 batch_config: PatientBatchRunConfig,
                                 stages: Sequence[PatientStage] | None) -> PatientRunResult:
    patient_start_time = perf_counter()
    patient_label = batch_config.patient_labels.get(patient_uid, "")
    try:
        runtime_state = carve_patient_runtime_state_by_uid(
            patient_uid,
            legacy_cohort_state.master_structure_reference_dict,
            legacy_cohort_state.master_structure_info_dict,
            legacy_keys=legacy_cohort_state.legacy_keys,
            patient_label=patient_label,
            source_run_id=batch_config.source_run_id,
            input_manifest_id=batch_config.input_manifest_id,
            metadata=batch_config.metadata,
        )
        return run_patient_case(runtime_state, batch_config.patient_config, stages=stages)
    except Exception as exc:
        if batch_config.patient_config.raise_on_stage_error:
            raise
        return _patient_setup_failure_result(
            patient_uid,
            patient_label,
            batch_config,
            exc,
            perf_counter() - patient_start_time,
        )


def _patient_setup_failure_result(patient_uid: str,
                                  patient_label: str,
                                  batch_config: PatientBatchRunConfig,
                                  exception: BaseException,
                                  elapsed_seconds: float) -> PatientRunResult:
    patient_case = PatientCase(
        patient_uid=patient_uid,
        patient_label=patient_label or patient_uid,
        source_run_id=batch_config.source_run_id,
        input_manifest_id=batch_config.input_manifest_id,
        metadata=batch_config.metadata,
    )
    stage_result = PatientStageResult.failure(
        PatientStageName.LEGACY_BRIDGE,
        elapsed_seconds=elapsed_seconds,
        exception=exception,
        metadata={"patient_uid": patient_uid},
    )
    return PatientRunResult.from_stage_results(
        patient_case,
        batch_config.patient_config.patient_output_dir(patient_case),
        (stage_result,),
        elapsed_seconds=elapsed_seconds,
        metadata={"run_id": batch_config.run_id},
    )
