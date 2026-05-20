"""Minimal patient-runner orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Protocol, Sequence

from .contracts import LegacyPatientRuntimeState
from .contracts import PatientRunConfig
from .contracts import PatientRunResult
from .contracts import PatientStageName
from .contracts import PatientStageResult
from .manifests import write_patient_run_manifest
from .stages import write_patient_artifacts_stage


class PatientStageRunner(Protocol):
    """Callable interface for one patient-local stage."""

    def __call__(self,
                 runtime_state: LegacyPatientRuntimeState,
                 config: PatientRunConfig) -> PatientStageResult:
        ...


@dataclass(frozen=True, slots=True)
class PatientStage:
    """Named patient-local stage function."""

    stage_name: PatientStageName | str
    runner: Callable[[LegacyPatientRuntimeState, PatientRunConfig], PatientStageResult]


def default_patient_stages() -> tuple[PatientStage, ...]:
    """Return the current Phase C.0 default stage list."""
    return (
        PatientStage(PatientStageName.PATIENT_ARTIFACT_WRITING, write_patient_artifacts_stage),
    )


def run_patient_case(runtime_state: LegacyPatientRuntimeState,
                     config: PatientRunConfig,
                     stages: Sequence[PatientStage] | None = None) -> PatientRunResult:
    """Run one patient-local state through the supplied stage sequence."""
    start_time = perf_counter()
    stage_results = run_patient_stages(
        runtime_state,
        config,
        stages=default_patient_stages() if stages is None else stages,
    )
    patient_result = PatientRunResult.from_stage_results(
        runtime_state.patient_case,
        config.patient_output_dir(runtime_state.patient_case),
        stage_results,
        elapsed_seconds=perf_counter() - start_time,
        metadata={"run_id": config.run_id},
    )
    if config.write_patient_run_manifest:
        write_patient_run_manifest(patient_result)
    return patient_result


def run_patient_stages(runtime_state: LegacyPatientRuntimeState,
                       config: PatientRunConfig,
                       stages: Sequence[PatientStage]) -> tuple[PatientStageResult, ...]:
    """Run the supplied patient stages with timing and error capture."""
    stage_results: list[PatientStageResult] = []
    for stage in stages:
        stage_start_time = perf_counter()
        try:
            stage_result = stage.runner(runtime_state, config)
            stage_result = stage_result.with_elapsed_seconds(perf_counter() - stage_start_time)
        except Exception as exc:
            elapsed_seconds = perf_counter() - stage_start_time
            if config.raise_on_stage_error:
                raise
            stage_result = PatientStageResult.failure(
                stage.stage_name,
                elapsed_seconds=elapsed_seconds,
                exception=exc,
                metadata={"patient_uid": runtime_state.patient_uid},
            )
        stage_results.append(stage_result)
        if not stage_result.succeeded and config.stop_on_stage_error:
            break
    return tuple(stage_results)