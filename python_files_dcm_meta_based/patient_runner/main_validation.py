"""Main-facing validation hooks for patient-runner integration."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import pandas as pd

from .batch import run_patient_batch
from .cohort_assembly import PatientBatchCohortAssemblyConfig
from .cohort_assembly import PatientBatchCohortAssemblyResult
from .cohort_assembly import run_patient_batch_cohort_assembly
from .cohort_assembly import summarize_patient_batch_cohort_assembly
from .cohort_assembly import summarize_patient_batch_cohort_validation
from .contracts import LegacyCohortRuntimeState
from .contracts import LegacyRuntimeKeys
from .contracts import PatientBatchExecutionBackend
from .contracts import PatientBatchRunConfig
from .contracts import PatientBatchRunResult
from .contracts import PatientRunConfig
from .contracts import validate_patient_uids
from .scientific_shadow import DEFAULT_PATIENT_RUNNER_SCIENTIFIC_SHADOW_DIR_NAME
from .scientific_shadow import PatientScientificShadowConfig
from .scientific_shadow import PatientScientificShadowRunResult
from .scientific_shadow import run_patient_scientific_shadow
from .scientific_shadow import summarize_patient_scientific_shadow_run


PATIENT_RUNNER_MAIN_VALIDATION_SCHEMA_VERSION = "patient_runner_main_validation_v1"
DEFAULT_PATIENT_RUNNER_SHADOW_OUTPUT_DIR_NAME = "patient_runner_shadow_output"


class PatientRunnerMainValidationMode(str, Enum):
    """Main-facing validation modes for the gated patient-runner path."""

    DISABLED = "disabled"
    SHADOW_OUTPUT = "shadow_output"
    SCIENTIFIC_SHADOW = "scientific_shadow"


@dataclass(frozen=True, slots=True)
class PatientRunnerMainValidationConfig:
    """Configuration for main-facing patient-runner validation.

    `SHADOW_OUTPUT` runs after the legacy path has produced in-memory patient and
    cohort outputs. It writes patient-runner artifacts from that completed legacy
    state, assembles selected cohort tables, and compares those assembled tables
    to the legacy final cohort dataframes.
    """

    mode: PatientRunnerMainValidationMode | str = PatientRunnerMainValidationMode.DISABLED
    patient_uids: Sequence[str] = ()
    final_table_names: Sequence[str] = ()
    source_table_names: Sequence[str] = ()
    scientific_shadow_config: PatientScientificShadowConfig | None = None
    output_dir: Path | None = None
    write_outputs: bool = True
    write_assembled_tables: bool = True
    max_workers: int = 1
    execution_backend: PatientBatchExecutionBackend | str = PatientBatchExecutionBackend.SEQUENTIAL
    run_id: str = "patient-runner-shadow-output-validation"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", PatientRunnerMainValidationMode(self.mode))
        object.__setattr__(self, "patient_uids", validate_patient_uids(self.patient_uids, "patient_uids"))
        object.__setattr__(self, "final_table_names", _validate_name_filter(self.final_table_names, "final_table_names"))
        object.__setattr__(self, "source_table_names", _validate_name_filter(self.source_table_names, "source_table_names"))
        if self.scientific_shadow_config is not None and not isinstance(
            self.scientific_shadow_config,
            PatientScientificShadowConfig,
        ):
            raise TypeError("scientific_shadow_config must be a PatientScientificShadowConfig instance")
        if self.output_dir is not None:
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "write_outputs", bool(self.write_outputs))
        object.__setattr__(self, "write_assembled_tables", bool(self.write_assembled_tables))
        max_workers = int(self.max_workers)
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        object.__setattr__(self, "max_workers", max_workers)
        object.__setattr__(self, "execution_backend", PatientBatchExecutionBackend(self.execution_backend))
        object.__setattr__(self, "run_id", str(self.run_id).strip())
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def enabled(self) -> bool:
        return self.mode != PatientRunnerMainValidationMode.DISABLED


@dataclass(frozen=True, slots=True)
class PatientRunnerMainValidationResult:
    """Result bundle from a main-facing patient-runner validation run."""

    mode: PatientRunnerMainValidationMode
    output_dir: Path
    batch_result: PatientBatchRunResult
    assembly_result: PatientBatchCohortAssemblyResult
    validation_df: pd.DataFrame | None
    written_paths: tuple[Path, ...] = ()
    summary_path: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", PatientRunnerMainValidationMode(self.mode))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "written_paths", tuple(Path(path) for path in self.written_paths))
        if self.summary_path is not None:
            object.__setattr__(self, "summary_path", Path(self.summary_path))


@dataclass(frozen=True, slots=True)
class PatientRunnerMainScientificShadowValidationResult:
    """Result bundle from a main-facing scientific shadow validation run."""

    mode: PatientRunnerMainValidationMode
    output_dir: Path
    scientific_shadow_result: PatientScientificShadowRunResult
    summary_path: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", PatientRunnerMainValidationMode(self.mode))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not isinstance(self.scientific_shadow_result, PatientScientificShadowRunResult):
            raise TypeError("scientific_shadow_result must be a PatientScientificShadowRunResult instance")
        if self.summary_path is not None:
            object.__setattr__(self, "summary_path", Path(self.summary_path))


@dataclass(frozen=True, slots=True)
class PatientRunnerMainValidationSkippedResult:
    """Result object returned when main-facing validation is disabled."""

    mode: PatientRunnerMainValidationMode
    output_dir: Path | None = None
    summary_path: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", PatientRunnerMainValidationMode(self.mode))
        if self.output_dir is not None:
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.summary_path is not None:
            object.__setattr__(self, "summary_path", Path(self.summary_path))


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


def _validate_name_filter(values: Sequence[str], source_name: str) -> tuple[str, ...]:
    resolved_values = tuple(values)
    if any(not isinstance(value, str) for value in resolved_values):
        raise TypeError(f"{source_name} entries must be strings")
    if any(value.strip() == "" for value in resolved_values):
        raise ValueError(f"{source_name} cannot contain empty values")
    if len(set(resolved_values)) != len(resolved_values):
        raise ValueError(f"{source_name} cannot contain duplicates")
    return resolved_values


def _config_or_default(config: PatientRunnerMainValidationConfig | None) -> PatientRunnerMainValidationConfig:
    return PatientRunnerMainValidationConfig() if config is None else config


def _resolve_output_dir(config: PatientRunnerMainValidationConfig, fallback_output_root: Path) -> Path:
    if config.output_dir is not None:
        return config.output_dir
    return default_patient_runner_main_validation_output_dir(fallback_output_root, config.mode)


def default_patient_runner_main_validation_output_dir(
    fallback_output_root: Path,
    mode: PatientRunnerMainValidationMode | str,
) -> Path:
    """Return the default output/evidence root for one main validation mode."""
    resolved_mode = PatientRunnerMainValidationMode(mode)
    if resolved_mode == PatientRunnerMainValidationMode.SCIENTIFIC_SHADOW:
        dir_name = DEFAULT_PATIENT_RUNNER_SCIENTIFIC_SHADOW_DIR_NAME
    else:
        dir_name = DEFAULT_PATIENT_RUNNER_SHADOW_OUTPUT_DIR_NAME
    return Path(fallback_output_root).joinpath("validation", dir_name)


def _scientific_shadow_config_for_main(config: PatientRunnerMainValidationConfig) -> PatientScientificShadowConfig:
    if config.scientific_shadow_config is None:
        raise ValueError("scientific_shadow_config is required for SCIENTIFIC_SHADOW validation mode")
    shadow_config = config.scientific_shadow_config
    if config.patient_uids:
        if shadow_config.patient_uids and shadow_config.patient_uids != config.patient_uids:
            raise ValueError("patient_uids cannot differ between main validation config and scientific_shadow_config")
        if not shadow_config.patient_uids:
            shadow_config = replace(shadow_config, patient_uids=config.patient_uids)
    return shadow_config


def _cohort_dataframes(master_cohort_patient_data_and_dataframes: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    dataframe_mapping = master_cohort_patient_data_and_dataframes.get("Dataframes", {})
    return {
        str(dataframe_name): dataframe
        for dataframe_name, dataframe in dataframe_mapping.items()
        if isinstance(dataframe, pd.DataFrame)
    }


def run_patient_runner_main_validation(
    *,
    master_structure_reference_dict: MutableMapping[str, Any],
    master_structure_info_dict: MutableMapping[str, Any],
    master_cohort_patient_data_and_dataframes: Mapping[str, Any],
    legacy_keys: LegacyRuntimeKeys,
    output_root: Path,
    config: PatientRunnerMainValidationConfig | None = None,
) -> PatientRunnerMainValidationResult | PatientRunnerMainScientificShadowValidationResult | PatientRunnerMainValidationSkippedResult:
    """Run the gated patient-runner validation hook from legacy main.

    The initial `SHADOW_OUTPUT` mode does not recompute scientific stages. It
    exports patient-runner artifacts from the completed legacy runtime state,
    assembles cohort-style outputs from those artifacts, and compares them to the
    legacy final cohort dataframes.
    """
    resolved_config = _config_or_default(config)
    resolved_output_dir = _resolve_output_dir(resolved_config, output_root)

    if not resolved_config.enabled:
        return PatientRunnerMainValidationSkippedResult(
            mode=PatientRunnerMainValidationMode.DISABLED,
            output_dir=resolved_output_dir,
        )
    metadata = dict(resolved_config.metadata)
    metadata.update({"validation_mode": resolved_config.mode.value})
    legacy_cohort_state = LegacyCohortRuntimeState(
        master_structure_reference_dict=master_structure_reference_dict,
        master_structure_info_dict=master_structure_info_dict,
        legacy_keys=legacy_keys,
        metadata=metadata,
    )
    patient_config = PatientRunConfig(
        output_root=resolved_output_dir,
        legacy_keys=legacy_keys,
        run_id=resolved_config.run_id,
    )
    if resolved_config.mode == PatientRunnerMainValidationMode.SCIENTIFIC_SHADOW:
        scientific_shadow_result = run_patient_scientific_shadow(
            legacy_cohort_state,
            patient_config,
            _scientific_shadow_config_for_main(resolved_config),
        )
        summary_path = resolved_output_dir.joinpath("patient_runner_main_validation_summary.json")
        result = PatientRunnerMainScientificShadowValidationResult(
            mode=resolved_config.mode,
            output_dir=resolved_output_dir,
            scientific_shadow_result=scientific_shadow_result,
            summary_path=summary_path,
        )
        write_patient_runner_main_validation_summary(result, output_path=summary_path)
        return result

    if resolved_config.mode != PatientRunnerMainValidationMode.SHADOW_OUTPUT:
        raise ValueError(f"Unsupported patient-runner main validation mode: {resolved_config.mode}")

    batch_config = PatientBatchRunConfig(
        patient_config=patient_config,
        patient_uids=resolved_config.patient_uids,
        max_workers=resolved_config.max_workers,
        execution_backend=resolved_config.execution_backend,
        metadata=metadata,
    )
    batch_result = run_patient_batch(
        legacy_cohort_state,
        batch_config,
    )
    assembly_config = PatientBatchCohortAssemblyConfig(
        patient_uids=resolved_config.patient_uids,
        final_table_names=resolved_config.final_table_names,
        source_table_names=resolved_config.source_table_names,
        output_dir=resolved_output_dir.joinpath("cohort_assembly"),
        write_outputs=resolved_config.write_outputs,
        write_assembled_tables=resolved_config.write_assembled_tables,
    )
    assembly_result, validation_df, written_paths = run_patient_batch_cohort_assembly(
        batch_result,
        assembly_config,
        final_cohort_dataframes=_cohort_dataframes(master_cohort_patient_data_and_dataframes),
    )
    summary_path = resolved_output_dir.joinpath("patient_runner_main_validation_summary.json")
    result = PatientRunnerMainValidationResult(
        mode=resolved_config.mode,
        output_dir=resolved_output_dir,
        batch_result=batch_result,
        assembly_result=assembly_result,
        validation_df=validation_df,
        written_paths=tuple((*written_paths, summary_path)),
        summary_path=summary_path,
    )
    write_patient_runner_main_validation_summary(result, output_path=summary_path)
    return result


def summarize_patient_runner_main_validation(
    result: PatientRunnerMainValidationResult | PatientRunnerMainScientificShadowValidationResult | PatientRunnerMainValidationSkippedResult,
) -> dict[str, Any]:
    """Return a JSON-ready summary for the main-facing validation result."""
    if isinstance(result, PatientRunnerMainValidationSkippedResult):
        return {
            "schema_version": PATIENT_RUNNER_MAIN_VALIDATION_SCHEMA_VERSION,
            "generated_utc": _utc_now_iso(),
            "mode": result.mode.value,
            "status": "skipped",
            "output_dir": result.output_dir.as_posix() if result.output_dir is not None else "",
        }

    if isinstance(result, PatientRunnerMainScientificShadowValidationResult):
        scientific_shadow_summary = summarize_patient_scientific_shadow_run(result.scientific_shadow_result)
        summary = {
            "schema_version": PATIENT_RUNNER_MAIN_VALIDATION_SCHEMA_VERSION,
            "generated_utc": _utc_now_iso(),
            "mode": result.mode.value,
            "status": scientific_shadow_summary["status"],
            "output_dir": result.output_dir.as_posix(),
            "scientific_shadow_summary": scientific_shadow_summary,
            "validation_summary": None,
            "written_path_count": len(result.scientific_shadow_result.written_paths) + (1 if result.summary_path else 0),
            "written_paths": [path.as_posix() for path in result.scientific_shadow_result.written_paths]
            + ([result.summary_path.as_posix()] if result.summary_path is not None else []),
        }
        return _json_safe(summary)

    validation_summary = None
    if result.validation_df is not None:
        validation_summary = summarize_patient_batch_cohort_validation(result.validation_df)
    patient_status_counts = dict(Counter(patient_result.status.value for patient_result in result.batch_result.patient_results))
    summary = {
        "schema_version": PATIENT_RUNNER_MAIN_VALIDATION_SCHEMA_VERSION,
        "generated_utc": _utc_now_iso(),
        "mode": result.mode.value,
        "status": result.batch_result.status.value,
        "output_dir": result.output_dir.as_posix(),
        "batch_status": result.batch_result.status.value,
        "batch_elapsed_seconds": result.batch_result.elapsed_seconds,
        "patient_count": result.batch_result.patient_count,
        "patient_status_counts": patient_status_counts,
        "batch_artifact_count": len(result.batch_result.artifact_paths),
        "assembly_summary": summarize_patient_batch_cohort_assembly(result.assembly_result),
        "validation_summary": validation_summary,
        "written_path_count": len(result.written_paths),
        "written_paths": [path.as_posix() for path in result.written_paths],
    }
    return _json_safe(summary)


def write_patient_runner_main_validation_summary(
    result: PatientRunnerMainValidationResult | PatientRunnerMainScientificShadowValidationResult | PatientRunnerMainValidationSkippedResult,
    output_path: Path | None = None,
) -> Path:
    """Write the main-facing patient-runner validation summary JSON."""
    if output_path is None:
        if result.output_dir is None:
            raise ValueError("output_path is required when result.output_dir is None")
        resolved_output_path = Path(result.output_dir).joinpath("patient_runner_main_validation_summary.json")
    else:
        resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(summarize_patient_runner_main_validation(result), file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")
    return resolved_output_path
