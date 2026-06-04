"""Public orchestration facade for patient-local scientific runs."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from .batch import run_patient_batch_from_legacy
from .contracts import LegacyRuntimeKeys
from .contracts import PatientBatchExecutionBackend
from .contracts import PatientBatchRunConfig
from .contracts import PatientBatchRunResult
from .contracts import PatientRunConfig
from .contracts import PatientStageName
from .contracts import resolve_legacy_patient_uids
from .contracts import validate_patient_uids
from .scientific_config import PatientRunnerScientificConfig
from .scientific_config_builder import PatientRunnerScientificConfigBuildContext
from .scientific_config_builder import build_patient_runner_scientific_config
from .scientific_dependencies import PatientScientificPathwayName
from .scientific_dependencies import executable_patient_scientific_pathway_stage_names
from .scientific_dependencies import resolve_patient_scientific_pathway_name
from .scientific_dependencies import resolve_patient_scientific_stage_names
from .scientific_stages import build_patient_scientific_stages_for_pathway


@dataclass(frozen=True, slots=True)
class PatientScientificRunConfig:
    """Typed config for the live per-patient scientific runner.

    The batch layer owns patient selection, output location, and scheduling. This
    layer owns only the scientific pathway and stage graph choices.
    """

    batch_config: PatientBatchRunConfig
    scientific_config: PatientRunnerScientificConfig
    pathway_name: PatientScientificPathwayName | str = PatientScientificPathwayName.CURRENT_DOSIMETRY_SHADOW
    satisfied_stage_names: Sequence[PatientStageName | str] = ()
    include_artifact_writing: bool = True
    validate_dependencies: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.batch_config, PatientBatchRunConfig):
            raise TypeError("batch_config must be a PatientBatchRunConfig instance")
        if not isinstance(self.scientific_config, PatientRunnerScientificConfig):
            raise TypeError("scientific_config must be a PatientRunnerScientificConfig instance")
        object.__setattr__(self, "pathway_name", resolve_patient_scientific_pathway_name(self.pathway_name))
        object.__setattr__(
            self,
            "satisfied_stage_names",
            resolve_patient_scientific_stage_names(self.satisfied_stage_names),
        )
        object.__setattr__(self, "include_artifact_writing", bool(self.include_artifact_writing))
        object.__setattr__(self, "validate_dependencies", bool(self.validate_dependencies))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def patient_uids(self) -> tuple[str, ...]:
        return self.batch_config.patient_uids

    @property
    def planned_stage_names(self) -> tuple[PatientStageName, ...]:
        return executable_patient_scientific_pathway_stage_names(
            self.pathway_name,
            satisfied_stage_names=self.satisfied_stage_names,
        )


def build_patient_scientific_run_config_from_pipeline(
    pipeline_config: Any,
    context: PatientRunnerScientificConfigBuildContext | None,
    *,
    output_root: Path,
    legacy_keys: LegacyRuntimeKeys | None = None,
    pathway_name: PatientScientificPathwayName | str = PatientScientificPathwayName.CURRENT_DOSIMETRY_SHADOW,
    patient_uids: Sequence[str] = (),
    run_id: str = "",
    max_workers: int = 1,
    execution_backend: PatientBatchExecutionBackend | str = PatientBatchExecutionBackend.SEQUENTIAL,
    include_artifact_writing: bool = True,
    write_patient_run_manifests: bool = True,
    write_batch_run_manifest: bool = True,
    stop_on_stage_error: bool = True,
    raise_on_stage_error: bool = False,
    satisfied_stage_names: Sequence[PatientStageName | str] = (),
    validate_dependencies: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> PatientScientificRunConfig:
    """Build a live scientific-runner config from the canonical PipelineConfig."""
    refs = pipeline_config.legacy_refs
    resolved_legacy_keys = legacy_keys or LegacyRuntimeKeys(
        all_ref_key=refs.all_ref_key,
        bx_ref=refs.bx_ref,
        by_patient_key=refs.by_patient_key,
        global_key=refs.global_key,
        global_num_cases_key=refs.global_num_cases_key,
    )
    resolved_metadata = dict(metadata or {})
    patient_config = PatientRunConfig(
        output_root=Path(output_root),
        legacy_keys=resolved_legacy_keys,
        run_id=run_id,
        write_patient_run_manifest=write_patient_run_manifests,
        stop_on_stage_error=stop_on_stage_error,
        raise_on_stage_error=raise_on_stage_error,
    )
    batch_config = PatientBatchRunConfig(
        patient_config=patient_config,
        patient_uids=patient_uids,
        max_workers=max_workers,
        execution_backend=execution_backend,
        write_batch_run_manifest=write_batch_run_manifest,
        metadata=resolved_metadata,
    )
    return PatientScientificRunConfig(
        batch_config=batch_config,
        scientific_config=build_patient_runner_scientific_config(pipeline_config, context),
        pathway_name=pathway_name,
        satisfied_stage_names=satisfied_stage_names,
        include_artifact_writing=include_artifact_writing,
        validate_dependencies=validate_dependencies,
        metadata={"source": "PipelineConfig", **resolved_metadata},
    )


def build_patient_scientific_runner_stages(
    run_config: PatientScientificRunConfig,
) -> tuple[Any, ...]:
    """Build the concrete stage sequence for a scientific run config."""
    return build_patient_scientific_stages_for_pathway(
        run_config.scientific_config,
        run_config.pathway_name,
        include_artifact_writing=run_config.include_artifact_writing,
        satisfied_stage_names=run_config.satisfied_stage_names,
        validate_dependencies=run_config.validate_dependencies,
    )


def run_patient_scientific_runner_from_legacy(
    master_structure_reference_dict: MutableMapping[str, Any],
    master_structure_info_dict: MutableMapping[str, Any],
    run_config: PatientScientificRunConfig,
) -> PatientBatchRunResult:
    """Run one scientific pathway through the patient batch runner."""
    if not isinstance(run_config, PatientScientificRunConfig):
        raise TypeError("run_config must be a PatientScientificRunConfig instance")
    resolve_legacy_patient_uids(master_structure_reference_dict, run_config.patient_uids)
    stages = build_patient_scientific_runner_stages(run_config)
    batch_config = replace(
        run_config.batch_config,
        metadata={
            **run_config.batch_config.metadata,
            **run_config.metadata,
            "pathway_name": run_config.pathway_name.value,
            "planned_stage_names": tuple(stage_name.value for stage_name in run_config.planned_stage_names),
        },
    )
    return run_patient_batch_from_legacy(
        master_structure_reference_dict,
        master_structure_info_dict,
        batch_config,
        stages=stages,
    )


def summarize_patient_scientific_run_config(run_config: PatientScientificRunConfig) -> dict[str, Any]:
    """Return a compact JSON-ready description of a planned scientific run."""
    if not isinstance(run_config, PatientScientificRunConfig):
        raise TypeError("run_config must be a PatientScientificRunConfig instance")
    return {
        "pathway_name": run_config.pathway_name.value,
        "patient_uids": tuple(validate_patient_uids(run_config.patient_uids, "patient_uids")),
        "planned_stage_names": tuple(stage_name.value for stage_name in run_config.planned_stage_names),
        "include_artifact_writing": run_config.include_artifact_writing,
        "output_root": run_config.batch_config.output_root.as_posix(),
        "execution_backend": run_config.batch_config.execution_backend.value,
        "max_workers": run_config.batch_config.max_workers,
        "metadata": dict(run_config.metadata),
    }