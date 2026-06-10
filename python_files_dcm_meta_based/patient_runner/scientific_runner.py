"""Public orchestration facade for patient-local scientific runs."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
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


PATIENT_SCIENTIFIC_RUNNER_PLAN_SCHEMA_VERSION = "patient_scientific_runner_plan_v1"
DEFAULT_PATIENT_SCIENTIFIC_RUNNER_DIR_NAME = "patient_scientific_runner"


@dataclass(frozen=True, slots=True)
class PatientScientificRunnerCheckpoint:
    """One outward validation checkpoint for the live scientific runner."""

    checkpoint_name: str
    pathway_name: PatientScientificPathwayName | str
    summary: str
    validation_after_run: Sequence[str] = ()
    next_checkpoint_name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint_name", _non_empty_string(self.checkpoint_name, "checkpoint_name"))
        object.__setattr__(self, "pathway_name", resolve_patient_scientific_pathway_name(self.pathway_name))
        object.__setattr__(self, "summary", _non_empty_string(self.summary, "summary"))
        object.__setattr__(
            self,
            "validation_after_run",
            tuple(_non_empty_string(value, "validation_after_run item") for value in self.validation_after_run),
        )
        object.__setattr__(self, "next_checkpoint_name", str(self.next_checkpoint_name).strip())


def _non_empty_string(value: Any, field_name: str) -> str:
    resolved_value = str(value).strip()
    if resolved_value == "":
        raise ValueError(f"{field_name} cannot be empty")
    return resolved_value


DEFAULT_PATIENT_SCIENTIFIC_RUNNER_CHECKPOINTS = (
    PatientScientificRunnerCheckpoint(
        checkpoint_name="anatomical_qa",
        pathway_name=PatientScientificPathwayName.ANATOMICAL_QA,
        summary="First live runner checkpoint: grid preprocessing plus anatomical preprocessing.",
        validation_after_run=(
            "Confirm patient_scientific_runner_plan.json lists grid_preprocessing and anatomical_preprocessing only.",
            "Inspect patient batch/run manifests for succeeded stage status per patient.",
            "Confirm patient_runner_main_validation_summary.json if SHADOW_OUTPUT validation was enabled.",
        ),
        next_checkpoint_name="biopsy_preprocessing_shadow",
    ),
    PatientScientificRunnerCheckpoint(
        checkpoint_name="biopsy_preprocessing_shadow",
        pathway_name=PatientScientificPathwayName.BIOPSY_PREPROCESSING_SHADOW,
        summary="Second checkpoint: adds biopsy-facing preprocessing after anatomical products exist.",
        validation_after_run=(
            "Compare biopsy preprocessing stage statuses and emitted metadata against the legacy run log.",
            "Inspect patient manifests for preprocessing warnings or skipped uncertainty attachment.",
        ),
        next_checkpoint_name="optimization_shadow",
    ),
    PatientScientificRunnerCheckpoint(
        checkpoint_name="optimization_shadow",
        pathway_name=PatientScientificPathwayName.OPTIMIZATION_SHADOW,
        summary="Third checkpoint: adds transform generation and optimizer-v1/v2 adapters.",
        validation_after_run=(
            "Compare optimizer stage status and selected optimizer-v2 metadata against the legacy optimizer run.",
            "Re-enable optimizer-v2 validation sidecars only if optimizer behavior changed or drift is suspected.",
        ),
        next_checkpoint_name="post_optimizer_biopsy_realization_shadow",
    ),
    PatientScientificRunnerCheckpoint(
        checkpoint_name="post_optimizer_biopsy_realization_shadow",
        pathway_name=PatientScientificPathwayName.POST_OPTIMIZER_BIOPSY_REALIZATION_SHADOW,
        summary=(
            "Fourth checkpoint: adds simulated-biopsy finalization and realized targeting after optimizer output."
        ),
        validation_after_run=(
            "Inspect simulated-biopsy finalization metadata for succeeded status and expected simulated biopsy counts.",
            "Confirm realized targeting runs after finalization and no biopsy centroid fields are missing.",
        ),
        next_checkpoint_name="sampling_classification_shadow",
    ),
    PatientScientificRunnerCheckpoint(
        checkpoint_name="sampling_classification_shadow",
        pathway_name=PatientScientificPathwayName.SAMPLING_CLASSIFICATION_SHADOW,
        summary="Fifth checkpoint: adds sampled-biopsy storage and classification-adjacent patient fragments.",
        validation_after_run=(
            "Inspect sampled-biopsy stage metadata for sampled result fragments and biopsy coordinate systems.",
            "Confirm optimizer-v2 sampling audit annotation runs after sampled-biopsy processing.",
            "Keep run-level per-voxel double-sextant assembly on the legacy/oracle side until separately wired.",
        ),
        next_checkpoint_name="current_dosimetry_shadow",
    ),
    PatientScientificRunnerCheckpoint(
        checkpoint_name="current_dosimetry_shadow",
        pathway_name=PatientScientificPathwayName.CURRENT_DOSIMETRY_SHADOW,
        summary=(
            "Sixth checkpoint: extends through MC prep, dose simulation, containment simulation, "
            "MR ADC simulation, and downstream MC output tables."
        ),
        validation_after_run=(
            "Inspect MC prep, MC simulation, and MC output-table patient manifests for succeeded stage status.",
            "Run cohort CSV parity against the previous validated run output.",
            "Re-enable Phase 3B/3C output sidecars only if output schema or artifact surfaces changed.",
        ),
        next_checkpoint_name="full_current_pipeline_shadow",
    ),
    PatientScientificRunnerCheckpoint(
        checkpoint_name="full_current_pipeline_shadow",
        pathway_name=PatientScientificPathwayName.FULL_CURRENT_PIPELINE_SHADOW,
        summary=(
            "Full current live-runner checkpoint: all current executable scientific stages, "
            "including downstream MC output tables and guidance."
        ),
        validation_after_run=(
            "Run full cohort CSV parity against the previous validated oracle run.",
            "Review guidance outputs and patient/cohort assembly evidence before considering any legacy deletion.",
        ),
    ),
)

_CHECKPOINTS_BY_NAME = {
    checkpoint.checkpoint_name: checkpoint for checkpoint in DEFAULT_PATIENT_SCIENTIFIC_RUNNER_CHECKPOINTS
}


def get_patient_scientific_runner_checkpoint(checkpoint_name: str) -> PatientScientificRunnerCheckpoint:
    """Return one configured outward-validation checkpoint."""
    resolved_name = _non_empty_string(checkpoint_name, "checkpoint_name")
    if resolved_name not in _CHECKPOINTS_BY_NAME:
        raise ValueError(
            "Unsupported patient scientific runner checkpoint: "
            + resolved_name
            + ". Expected one of: "
            + ", ".join(sorted(_CHECKPOINTS_BY_NAME))
        )
    return _CHECKPOINTS_BY_NAME[resolved_name]


@dataclass(frozen=True, slots=True)
class PatientScientificRunConfig:
    """Typed config for the live per-patient scientific runner.

    The batch layer owns patient selection, output location, and scheduling. This
    layer owns only the scientific pathway and stage graph choices.
    """

    batch_config: PatientBatchRunConfig
    scientific_config: PatientRunnerScientificConfig
    pathway_name: PatientScientificPathwayName | str = PatientScientificPathwayName.CURRENT_DOSIMETRY_SHADOW
    checkpoint_name: str = "current_dosimetry_shadow"
    checkpoint_summary: str = ""
    validation_after_run: Sequence[str] = ()
    next_checkpoint_name: str = ""
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
        object.__setattr__(self, "checkpoint_name", _non_empty_string(self.checkpoint_name, "checkpoint_name"))
        object.__setattr__(self, "checkpoint_summary", str(self.checkpoint_summary).strip())
        object.__setattr__(
            self,
            "validation_after_run",
            tuple(_non_empty_string(value, "validation_after_run item") for value in self.validation_after_run),
        )
        object.__setattr__(self, "next_checkpoint_name", str(self.next_checkpoint_name).strip())
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
    checkpoint_name: str | None = None,
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
    resolved_pathway_name = resolve_patient_scientific_pathway_name(pathway_name)
    checkpoint = get_patient_scientific_runner_checkpoint(
        checkpoint_name or resolved_pathway_name.value,
    )
    if checkpoint.pathway_name != resolved_pathway_name:
        raise ValueError(
            "checkpoint_name and pathway_name disagree: "
            f"{checkpoint.checkpoint_name} maps to {checkpoint.pathway_name.value}, "
            f"but pathway_name is {resolved_pathway_name.value}"
        )
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
        pathway_name=resolved_pathway_name,
        checkpoint_name=checkpoint.checkpoint_name,
        checkpoint_summary=checkpoint.summary,
        validation_after_run=checkpoint.validation_after_run,
        next_checkpoint_name=checkpoint.next_checkpoint_name,
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
        "checkpoint_name": run_config.checkpoint_name,
        "checkpoint_summary": run_config.checkpoint_summary,
        "validation_after_run": tuple(run_config.validation_after_run),
        "next_checkpoint_name": run_config.next_checkpoint_name,
        "patient_uids": tuple(validate_patient_uids(run_config.patient_uids, "patient_uids")),
        "planned_stage_names": tuple(stage_name.value for stage_name in run_config.planned_stage_names),
        "include_artifact_writing": run_config.include_artifact_writing,
        "output_root": run_config.batch_config.output_root.as_posix(),
        "execution_backend": run_config.batch_config.execution_backend.value,
        "max_workers": run_config.batch_config.max_workers,
        "metadata": dict(run_config.metadata),
    }


def patient_scientific_run_plan_summary(run_config: PatientScientificRunConfig) -> dict[str, Any]:
    """Return the durable plan summary written before optional execution."""
    summary = summarize_patient_scientific_run_config(run_config)
    return {
        "schema_version": PATIENT_SCIENTIFIC_RUNNER_PLAN_SCHEMA_VERSION,
        "runner_boundary": "patient_scientific_runner",
        "execution_policy": "pathway_expands_to_dependency_validated_patient_stages",
        **summary,
    }


def write_patient_scientific_run_plan_summary(
    run_config: PatientScientificRunConfig,
    output_path: Path | None = None,
) -> Path:
    """Write a JSON plan for a configured patient scientific runner pass."""
    resolved_output_path = Path(output_path) if output_path is not None else run_config.batch_config.output_root.joinpath(
        "patient_scientific_runner_plan.json",
    )
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(patient_scientific_run_plan_summary(run_config), file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")
    return resolved_output_path