"""Scientific shadow execution lane for patient-runner pathways."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

from .contracts import LegacyCohortRuntimeState
from .contracts import LegacyPatientRuntimeState
from .contracts import PatientCase
from .contracts import PatientRunConfig
from .contracts import PatientRunResult
from .contracts import PatientStageName
from .contracts import PatientStageResult
from .contracts import PatientStageStatus
from .contracts import validate_patient_uids
from .legacy_bridge import carve_patient_runtime_state_by_uid
from .manifests import write_patient_run_manifest
from .runner import PatientStage
from .scientific_config import PatientRunnerScientificConfig
from .scientific_dependencies import PatientScientificPathwayName
from .scientific_dependencies import executable_patient_scientific_pathway_stage_names
from .scientific_dependencies import resolve_patient_scientific_pathway_name
from .scientific_dependencies import resolve_patient_scientific_stage_names
from .scientific_stages import build_patient_scientific_stages_for_pathway


PATIENT_SCIENTIFIC_SHADOW_SCHEMA_VERSION = "patient_scientific_shadow_v1"
DEFAULT_PATIENT_RUNNER_SCIENTIFIC_SHADOW_DIR_NAME = "patient_runner_scientific_shadow"


class PatientScientificShadowStateIsolation(str, Enum):
    """How scientific shadow execution protects the legacy oracle state."""

    DEEP_COPY_PATIENT_STATE = "deep_copy_patient_state"
    SHARED_LEGACY_VIEW = "shared_legacy_view"


@dataclass(frozen=True, slots=True)
class PatientScientificShadowConfig:
    """Configuration for a staged scientific shadow run."""

    scientific_config: PatientRunnerScientificConfig
    pathway_name: PatientScientificPathwayName | str = PatientScientificPathwayName.CURRENT_DOSIMETRY_SHADOW
    patient_uids: Sequence[str] = ()
    satisfied_stage_names: Sequence[PatientStageName | str] = ()
    include_artifact_writing: bool = False
    write_patient_run_manifests: bool = True
    write_stage_state_manifests: bool = True
    include_dataframe_snapshots: bool = True
    state_isolation: PatientScientificShadowStateIsolation | str = (
        PatientScientificShadowStateIsolation.DEEP_COPY_PATIENT_STATE
    )
    stop_on_stage_error: bool = True
    raise_on_stage_error: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.scientific_config, PatientRunnerScientificConfig):
            raise TypeError("scientific_config must be a PatientRunnerScientificConfig instance")
        object.__setattr__(self, "pathway_name", resolve_patient_scientific_pathway_name(self.pathway_name))
        object.__setattr__(self, "patient_uids", validate_patient_uids(self.patient_uids, "patient_uids"))
        object.__setattr__(
            self,
            "satisfied_stage_names",
            resolve_patient_scientific_stage_names(self.satisfied_stage_names),
        )
        object.__setattr__(self, "include_artifact_writing", bool(self.include_artifact_writing))
        object.__setattr__(self, "write_patient_run_manifests", bool(self.write_patient_run_manifests))
        object.__setattr__(self, "write_stage_state_manifests", bool(self.write_stage_state_manifests))
        object.__setattr__(self, "include_dataframe_snapshots", bool(self.include_dataframe_snapshots))
        object.__setattr__(self, "state_isolation", PatientScientificShadowStateIsolation(self.state_isolation))
        object.__setattr__(self, "stop_on_stage_error", bool(self.stop_on_stage_error))
        object.__setattr__(self, "raise_on_stage_error", bool(self.raise_on_stage_error))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class PatientScientificShadowRunResult:
    """Result bundle for one scientific shadow pathway run."""

    pathway_name: PatientScientificPathwayName
    output_root: Path
    patient_results: tuple[PatientRunResult, ...]
    stage_state_manifest_paths: tuple[Path, ...] = ()
    summary_path: Path | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pathway_name", resolve_patient_scientific_pathway_name(self.pathway_name))
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "patient_results", tuple(self.patient_results))
        object.__setattr__(self, "stage_state_manifest_paths", tuple(Path(path) for path in self.stage_state_manifest_paths))
        if self.summary_path is not None:
            object.__setattr__(self, "summary_path", Path(self.summary_path))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def patient_count(self) -> int:
        return len(self.patient_results)

    @property
    def failed_patient_results(self) -> tuple[PatientRunResult, ...]:
        return tuple(patient_result for patient_result in self.patient_results if not patient_result.succeeded)

    @property
    def failed_patient_count(self) -> int:
        return len(self.failed_patient_results)

    @property
    def status(self) -> PatientStageStatus:
        return PatientStageStatus.FAILED if self.failed_patient_results else PatientStageStatus.SUCCEEDED

    @property
    def succeeded(self) -> bool:
        return self.status == PatientStageStatus.SUCCEEDED

    @property
    def written_paths(self) -> tuple[Path, ...]:
        paths: list[Path] = []
        paths.extend(self.stage_state_manifest_paths)
        if self.summary_path is not None:
            paths.append(self.summary_path)
        return tuple(paths)


def run_patient_scientific_shadow(
    legacy_cohort_state: LegacyCohortRuntimeState,
    patient_config: PatientRunConfig,
    shadow_config: PatientScientificShadowConfig,
) -> PatientScientificShadowRunResult:
    """Run one named scientific pathway in an isolated shadow/evidence lane."""
    if not isinstance(legacy_cohort_state, LegacyCohortRuntimeState):
        raise TypeError("legacy_cohort_state must be a LegacyCohortRuntimeState instance")
    if not isinstance(patient_config, PatientRunConfig):
        raise TypeError("patient_config must be a PatientRunConfig instance")
    if not isinstance(shadow_config, PatientScientificShadowConfig):
        raise TypeError("shadow_config must be a PatientScientificShadowConfig instance")
    if legacy_cohort_state.legacy_keys != patient_config.legacy_keys:
        raise ValueError("legacy_cohort_state.legacy_keys must match patient_config.legacy_keys")

    stages = build_patient_scientific_stages_for_pathway(
        shadow_config.scientific_config,
        shadow_config.pathway_name,
        include_artifact_writing=shadow_config.include_artifact_writing,
        satisfied_stage_names=shadow_config.satisfied_stage_names,
    )
    resolved_patient_config = replace(
        patient_config,
        write_patient_run_manifest=shadow_config.write_patient_run_manifests,
        stop_on_stage_error=shadow_config.stop_on_stage_error,
        raise_on_stage_error=shadow_config.raise_on_stage_error,
    )
    patient_uids = legacy_cohort_state.resolve_patient_uids(shadow_config.patient_uids)
    planned_stage_names = tuple(stage.stage_name for stage in stages)
    executable_stage_names = executable_patient_scientific_pathway_stage_names(
        shadow_config.pathway_name,
        satisfied_stage_names=shadow_config.satisfied_stage_names,
    )
    run_metadata = _shadow_run_metadata(shadow_config, planned_stage_names, executable_stage_names)

    patient_results: list[PatientRunResult] = []
    stage_state_manifest_paths: list[Path] = []
    for patient_uid in patient_uids:
        patient_result, stage_state_manifest_path = _run_one_patient_scientific_shadow(
            patient_uid,
            legacy_cohort_state,
            resolved_patient_config,
            shadow_config,
            stages,
            run_metadata,
        )
        patient_results.append(patient_result)
        if stage_state_manifest_path is not None:
            stage_state_manifest_paths.append(stage_state_manifest_path)

    summary_path = resolved_patient_config.output_root.joinpath("patient_scientific_shadow_summary.json")
    result = PatientScientificShadowRunResult(
        pathway_name=shadow_config.pathway_name,
        output_root=resolved_patient_config.output_root,
        patient_results=tuple(patient_results),
        stage_state_manifest_paths=tuple(stage_state_manifest_paths),
        summary_path=summary_path,
        metadata=run_metadata,
    )
    write_patient_scientific_shadow_summary(result, output_path=summary_path)
    return result


def summarize_patient_scientific_shadow_run(result: PatientScientificShadowRunResult) -> dict[str, Any]:
    """Return a JSON-ready summary for a scientific shadow run."""
    stage_status_counts: dict[str, int] = {}
    stage_failure_counts: dict[str, int] = {}
    for patient_result in result.patient_results:
        for stage_result in patient_result.stage_results:
            status_key = stage_result.status.value
            stage_status_counts[status_key] = stage_status_counts.get(status_key, 0) + 1
            if stage_result.status == PatientStageStatus.FAILED:
                stage_failure_counts[stage_result.stage_name] = stage_failure_counts.get(stage_result.stage_name, 0) + 1

    return _json_safe(
        {
            "schema_version": PATIENT_SCIENTIFIC_SHADOW_SCHEMA_VERSION,
            "generated_utc": _utc_now_iso(),
            "pathway_name": result.pathway_name.value,
            "status": result.status.value,
            "succeeded": result.succeeded,
            "output_root": result.output_root.as_posix(),
            "patient_count": result.patient_count,
            "failed_patient_count": result.failed_patient_count,
            "stage_status_counts": stage_status_counts,
            "stage_failure_counts": stage_failure_counts,
            "stage_state_manifest_count": len(result.stage_state_manifest_paths),
            "stage_state_manifest_paths": [path.as_posix() for path in result.stage_state_manifest_paths],
            "written_path_count": len(result.written_paths),
            "written_paths": [path.as_posix() for path in result.written_paths],
            "metadata": result.metadata,
        }
    )


def write_patient_scientific_shadow_summary(
    result: PatientScientificShadowRunResult,
    output_path: Path | None = None,
) -> Path:
    """Write the scientific shadow run summary JSON."""
    resolved_output_path = Path(output_path) if output_path is not None else result.output_root.joinpath(
        "patient_scientific_shadow_summary.json",
    )
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(summarize_patient_scientific_shadow_run(result), file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")
    return resolved_output_path


def patient_scientific_shadow_stage_state_manifest(
    *,
    patient_result: PatientRunResult,
    pathway_name: PatientScientificPathwayName,
    shadow_config: PatientScientificShadowConfig,
    run_metadata: Mapping[str, Any],
    stage_state_records: Sequence[Mapping[str, Any]],
    final_runtime_snapshot: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return a JSON-ready stage-state evidence manifest for one patient."""
    patient_case = patient_result.patient_case
    return _json_safe(
        {
            "schema_version": PATIENT_SCIENTIFIC_SHADOW_SCHEMA_VERSION,
            "generated_utc": _utc_now_iso(),
            "pathway_name": pathway_name.value,
            "patient_uid": patient_case.patient_uid,
            "patient_label": patient_case.patient_label,
            "source_run_id": patient_case.source_run_id,
            "input_manifest_id": patient_case.input_manifest_id,
            "status": patient_result.status.value,
            "succeeded": patient_result.succeeded,
            "output_root": patient_result.output_root.as_posix(),
            "state_isolation": shadow_config.state_isolation.value,
            "oracle_mutation_policy": _oracle_mutation_policy(shadow_config.state_isolation),
            "satisfied_stage_names": [stage_name.value for stage_name in shadow_config.satisfied_stage_names],
            "include_dataframe_snapshots": shadow_config.include_dataframe_snapshots,
            "stage_count": len(patient_result.stage_results),
            "failed_stage_count": len(patient_result.failed_stage_results),
            "stages": tuple(stage_state_records),
            "final_runtime_snapshot": final_runtime_snapshot,
            "run_metadata": run_metadata,
        }
    )


def write_patient_scientific_shadow_stage_state_manifest(
    *,
    patient_result: PatientRunResult,
    pathway_name: PatientScientificPathwayName,
    shadow_config: PatientScientificShadowConfig,
    run_metadata: Mapping[str, Any],
    stage_state_records: Sequence[Mapping[str, Any]],
    final_runtime_snapshot: Mapping[str, Any] | None,
    output_path: Path | None = None,
) -> Path:
    """Write one patient scientific shadow stage-state evidence manifest."""
    resolved_output_path = Path(output_path) if output_path is not None else patient_result.output_root.joinpath(
        "scientific_shadow_stage_state_manifest.json",
    )
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = patient_scientific_shadow_stage_state_manifest(
        patient_result=patient_result,
        pathway_name=pathway_name,
        shadow_config=shadow_config,
        run_metadata=run_metadata,
        stage_state_records=stage_state_records,
        final_runtime_snapshot=final_runtime_snapshot,
    )
    with resolved_output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(manifest, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")
    return resolved_output_path


def _run_one_patient_scientific_shadow(
    patient_uid: str,
    legacy_cohort_state: LegacyCohortRuntimeState,
    patient_config: PatientRunConfig,
    shadow_config: PatientScientificShadowConfig,
    stages: Sequence[PatientStage],
    run_metadata: Mapping[str, Any],
) -> tuple[PatientRunResult, Path | None]:
    patient_start_time = perf_counter()
    stage_state_records: list[Mapping[str, Any]] = []
    final_runtime_snapshot: Mapping[str, Any] | None = None
    try:
        runtime_state = carve_patient_runtime_state_by_uid(
            patient_uid,
            legacy_cohort_state.master_structure_reference_dict,
            legacy_cohort_state.master_structure_info_dict,
            legacy_keys=legacy_cohort_state.legacy_keys,
            source_run_id=patient_config.run_id,
            metadata=run_metadata,
        )
        runtime_state = _isolate_runtime_state(runtime_state, shadow_config.state_isolation)
        stage_results, stage_state_records = _run_scientific_shadow_stages(
            runtime_state,
            patient_config,
            stages,
            include_dataframe_snapshots=shadow_config.include_dataframe_snapshots,
        )
        if shadow_config.include_dataframe_snapshots:
            final_runtime_snapshot = _runtime_state_snapshot(runtime_state)
        patient_result = PatientRunResult.from_stage_results(
            runtime_state.patient_case,
            patient_config.patient_output_dir(runtime_state.patient_case),
            stage_results,
            elapsed_seconds=perf_counter() - patient_start_time,
            metadata={**run_metadata, "run_id": patient_config.run_id},
        )
    except Exception as exc:
        if patient_config.raise_on_stage_error:
            raise
        patient_case = PatientCase(
            patient_uid=patient_uid,
            patient_label=patient_uid,
            source_run_id=patient_config.run_id,
            metadata=run_metadata,
        )
        stage_result = PatientStageResult.failure(
            PatientStageName.LEGACY_BRIDGE,
            elapsed_seconds=perf_counter() - patient_start_time,
            exception=exc,
            metadata={"patient_uid": patient_uid, "shadow_setup_failed": True},
        )
        stage_state_records = (
            _stage_state_record(stage_result, runtime_snapshot=None),
        )
        patient_result = PatientRunResult.from_stage_results(
            patient_case,
            patient_config.patient_output_dir(patient_case),
            (stage_result,),
            elapsed_seconds=perf_counter() - patient_start_time,
            metadata={**run_metadata, "run_id": patient_config.run_id},
        )

    if patient_config.write_patient_run_manifest:
        write_patient_run_manifest(patient_result)

    stage_state_manifest_path = None
    if shadow_config.write_stage_state_manifests:
        stage_state_manifest_path = write_patient_scientific_shadow_stage_state_manifest(
            patient_result=patient_result,
            pathway_name=shadow_config.pathway_name,
            shadow_config=shadow_config,
            run_metadata=run_metadata,
            stage_state_records=stage_state_records,
            final_runtime_snapshot=final_runtime_snapshot,
        )
    return patient_result, stage_state_manifest_path


def _run_scientific_shadow_stages(
    runtime_state: LegacyPatientRuntimeState,
    patient_config: PatientRunConfig,
    stages: Sequence[PatientStage],
    *,
    include_dataframe_snapshots: bool,
) -> tuple[tuple[PatientStageResult, ...], tuple[Mapping[str, Any], ...]]:
    stage_results: list[PatientStageResult] = []
    stage_state_records: list[Mapping[str, Any]] = []
    for stage in stages:
        stage_start_time = perf_counter()
        try:
            stage_result = stage.runner(runtime_state, patient_config)
            stage_result = stage_result.with_elapsed_seconds(perf_counter() - stage_start_time)
        except Exception as exc:
            elapsed_seconds = perf_counter() - stage_start_time
            if patient_config.raise_on_stage_error:
                raise
            stage_result = PatientStageResult.failure(
                stage.stage_name,
                elapsed_seconds=elapsed_seconds,
                exception=exc,
                metadata={"patient_uid": runtime_state.patient_uid},
            )
        runtime_snapshot = _runtime_state_snapshot(runtime_state) if include_dataframe_snapshots else None
        stage_results.append(stage_result)
        stage_state_records.append(_stage_state_record(stage_result, runtime_snapshot=runtime_snapshot))
        if not stage_result.succeeded and patient_config.stop_on_stage_error:
            break
    return tuple(stage_results), tuple(stage_state_records)


def _stage_state_record(
    stage_result: PatientStageResult,
    *,
    runtime_snapshot: Mapping[str, Any] | None,
) -> dict[str, Any]:
    skip_reason = stage_result.metadata.get("skip_reason", "")
    return {
        "stage_name": stage_result.stage_name,
        "status": stage_result.status.value,
        "succeeded": stage_result.succeeded,
        "elapsed_seconds": stage_result.elapsed_seconds,
        "artifact_count": stage_result.artifact_count,
        "output_paths": [path.as_posix() for path in stage_result.output_paths],
        "warning_count": len(stage_result.warnings),
        "warnings": list(stage_result.warnings),
        "skip_reason": str(skip_reason),
        "metadata_keys": sorted(str(key) for key in stage_result.metadata.keys()),
        "metadata": _json_safe(stage_result.metadata),
        "runtime_snapshot": runtime_snapshot,
    }


def _isolate_runtime_state(
    runtime_state: LegacyPatientRuntimeState,
    state_isolation: PatientScientificShadowStateIsolation,
) -> LegacyPatientRuntimeState:
    if state_isolation == PatientScientificShadowStateIsolation.SHARED_LEGACY_VIEW:
        return runtime_state
    return LegacyPatientRuntimeState(
        patient_case=runtime_state.patient_case,
        master_structure_reference_dict=deepcopy(runtime_state.master_structure_reference_dict),
        master_structure_info_dict=deepcopy(runtime_state.master_structure_info_dict),
        legacy_keys=runtime_state.legacy_keys,
        metadata={**runtime_state.metadata, "state_isolation": state_isolation.value},
    )


def _shadow_run_metadata(
    shadow_config: PatientScientificShadowConfig,
    planned_stage_names: Sequence[Any],
    executable_stage_names: Sequence[PatientStageName],
) -> dict[str, Any]:
    metadata = dict(shadow_config.metadata)
    metadata.update(
        {
            "validation_mode": "scientific_shadow",
            "pathway_name": shadow_config.pathway_name.value,
            "enabled_stage_names": shadow_config.scientific_config.enabled_stage_names,
            "planned_stage_names": tuple(_stage_name_value(stage_name) for stage_name in planned_stage_names),
            "executable_stage_names": tuple(stage_name.value for stage_name in executable_stage_names),
            "satisfied_stage_names": tuple(stage_name.value for stage_name in shadow_config.satisfied_stage_names),
            "include_artifact_writing": shadow_config.include_artifact_writing,
            "state_isolation": shadow_config.state_isolation.value,
            "oracle_mutation_policy": _oracle_mutation_policy(shadow_config.state_isolation),
        }
    )
    return metadata


def _oracle_mutation_policy(state_isolation: PatientScientificShadowStateIsolation) -> str:
    if state_isolation == PatientScientificShadowStateIsolation.DEEP_COPY_PATIENT_STATE:
        return "scientific stages run on deep-copied carved patient state; source legacy oracle dictionaries are not mutated"
    return "scientific stages run on shared carved legacy objects; use only for controlled debugging"


def _runtime_state_snapshot(runtime_state: LegacyPatientRuntimeState) -> dict[str, Any]:
    dataframe_snapshots: list[dict[str, Any]] = []
    dataframe_snapshots.extend(_collect_dataframe_snapshots("patient_reference", runtime_state.pydicom_item))
    dataframe_snapshots.extend(_collect_dataframe_snapshots("patient_info", runtime_state.master_structure_info_dict))
    return {
        "patient_uid": runtime_state.patient_uid,
        "reference_patient_keys": _sorted_mapping_keys(runtime_state.pydicom_item),
        "info_top_level_keys": _sorted_mapping_keys(runtime_state.master_structure_info_dict),
        "dataframe_count": len(dataframe_snapshots),
        "dataframes": dataframe_snapshots,
    }


def _collect_dataframe_snapshots(
    path: str,
    value: Any,
    *,
    depth: int = 0,
    max_depth: int = 2,
    max_items: int = 25,
    max_records: int = 100,
) -> list[dict[str, Any]]:
    if _is_dataframe_like(value):
        shape = getattr(value, "shape", (0, 0))
        columns = tuple(str(column) for column in getattr(value, "columns", ()))
        return [
            {
                "path": path,
                "shape": tuple(int(item) for item in shape),
                "column_count": len(columns),
                "columns": columns[:50],
            }
        ]
    if depth >= max_depth:
        return []

    records: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        for key, item in list(value.items())[:max_items]:
            records.extend(
                _collect_dataframe_snapshots(
                    f"{path}.{key}",
                    item,
                    depth=depth + 1,
                    max_depth=max_depth,
                    max_items=max_items,
                    max_records=max_records,
                )
            )
            if len(records) >= max_records:
                return records[:max_records]
    elif isinstance(value, (tuple, list)) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value[:max_items]):
            records.extend(
                _collect_dataframe_snapshots(
                    f"{path}[{index}]",
                    item,
                    depth=depth + 1,
                    max_depth=max_depth,
                    max_items=max_items,
                    max_records=max_records,
                )
            )
            if len(records) >= max_records:
                return records[:max_records]
    return records


def _is_dataframe_like(value: Any) -> bool:
    shape = getattr(value, "shape", None)
    columns = getattr(value, "columns", None)
    return isinstance(shape, tuple) and len(shape) == 2 and columns is not None


def _sorted_mapping_keys(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Mapping):
        return ()
    return tuple(sorted(str(key) for key in value.keys()))


def _stage_name_value(stage_name: PatientStageName | str) -> str:
    return stage_name.value if isinstance(stage_name, PatientStageName) else str(stage_name)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
