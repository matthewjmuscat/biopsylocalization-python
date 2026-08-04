"""JSON config loading for post-run cohort assembly jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from patient_runner.contracts import validate_patient_uids


POST_RUN_COHORT_ASSEMBLY_CONFIG_SCHEMA_VERSION = "post_run_cohort_assembly_jobs_v1"

COHORT_ASSEMBLY_PACKAGE_DIR = Path(__file__).resolve().parent
POST_RUN_ROOT = COHORT_ASSEMBLY_PACKAGE_DIR.parent
PYTHON_ROOT = POST_RUN_ROOT.parent
REPO_ROOT = PYTHON_ROOT.parent
DEFAULT_COHORT_ASSEMBLY_CONFIG_PATH = POST_RUN_ROOT / "configs" / "cohort_assembly_jobs.json"


@dataclass(frozen=True, slots=True)
class PostRunCohortAssemblyJobConfig:
    """One post-run cohort assembly job loaded from JSON or a GUI layer."""

    name: str
    patient_runner_output_dir: Path
    output_dir: Path | None = None
    patient_uids: Sequence[str] = ()
    final_table_names: Sequence[str] = ()
    source_table_names: Sequence[str] = ()
    write_outputs: bool = True
    write_assembled_tables: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("post-run cohort assembly job name cannot be empty")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "patient_runner_output_dir", Path(self.patient_runner_output_dir))
        if self.output_dir is not None:
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "patient_uids", validate_patient_uids(self.patient_uids, "patient_uids"))
        object.__setattr__(self, "final_table_names", _validate_name_filter(self.final_table_names, "final_table_names"))
        object.__setattr__(self, "source_table_names", _validate_name_filter(self.source_table_names, "source_table_names"))
        object.__setattr__(self, "write_outputs", bool(self.write_outputs))
        object.__setattr__(self, "write_assembled_tables", bool(self.write_assembled_tables))
        object.__setattr__(self, "metadata", dict(self.metadata))


def _validate_name_filter(values: Sequence[str], source_name: str) -> tuple[str, ...]:
    resolved_values = tuple(values)
    if any(not isinstance(value, str) for value in resolved_values):
        raise TypeError(f"{source_name} entries must be strings")
    if any(value.strip() == "" for value in resolved_values):
        raise ValueError(f"{source_name} cannot contain empty values")
    if len(set(resolved_values)) != len(resolved_values):
        raise ValueError(f"{source_name} cannot contain duplicates")
    return resolved_values


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise TypeError(f"JSON config root must be an object: {path}")
    return payload


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _resolve_named_or_direct_path(config: Mapping[str, Any], value: str | Path, source_name: str) -> Path:
    value_text = str(value).strip()
    if not value_text:
        raise ValueError(f"{source_name} cannot be empty")
    named_paths = config.get("paths", {})
    if isinstance(named_paths, Mapping) and value_text in named_paths:
        return _resolve_path(str(named_paths[value_text]))
    return _resolve_path(value_text)


def _job_patient_runner_output_dir(config: Mapping[str, Any], job: Mapping[str, Any]) -> Path:
    if job.get("patient_runner_output_path"):
        return _resolve_named_or_direct_path(config, str(job["patient_runner_output_path"]), "patient_runner_output_path")
    if job.get("patient_runner_output_dir"):
        return _resolve_named_or_direct_path(config, str(job["patient_runner_output_dir"]), "patient_runner_output_dir")
    raise ValueError("cohort assembly jobs require patient_runner_output_path or patient_runner_output_dir")


def _resolve_optional_output_dir(config: Mapping[str, Any], job: Mapping[str, Any]) -> Path | None:
    output_dir = job.get("output_dir")
    if not output_dir:
        return None
    return _resolve_named_or_direct_path(config, str(output_dir), "output_dir")


def _merged_job(defaults: Mapping[str, Any], job: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    merged.update(dict(job))
    return merged


def load_cohort_assembly_job_configs(config_path: str | Path = DEFAULT_COHORT_ASSEMBLY_CONFIG_PATH) -> tuple[PostRunCohortAssemblyJobConfig, ...]:
    """Load enabled post-run cohort assembly jobs from a JSON config file."""
    resolved_config_path = Path(config_path).expanduser()
    if not resolved_config_path.is_absolute():
        resolved_config_path = REPO_ROOT / resolved_config_path
    config = _read_json_object(resolved_config_path)
    schema_version = str(config.get("schema_version", "")).strip()
    if schema_version and schema_version != POST_RUN_COHORT_ASSEMBLY_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported post-run cohort assembly config schema_version "
            f"{schema_version!r}; expected {POST_RUN_COHORT_ASSEMBLY_CONFIG_SCHEMA_VERSION!r}"
        )

    defaults = config.get("defaults", {})
    if not isinstance(defaults, Mapping):
        raise TypeError("defaults must be a JSON object")
    jobs = config.get("jobs", [])
    if not isinstance(jobs, list):
        raise TypeError("jobs must be a JSON list")

    loaded_jobs: list[PostRunCohortAssemblyJobConfig] = []
    for index, raw_job in enumerate(jobs, start=1):
        if not isinstance(raw_job, Mapping):
            raise TypeError(f"job {index} must be a JSON object")
        if not bool(raw_job.get("enabled", True)):
            continue
        job = _merged_job(defaults, raw_job)
        loaded_jobs.append(
            PostRunCohortAssemblyJobConfig(
                name=str(job.get("name") or f"job_{index}"),
                patient_runner_output_dir=_job_patient_runner_output_dir(config, job),
                output_dir=_resolve_optional_output_dir(config, job),
                patient_uids=tuple(job.get("patient_uids", ())),
                final_table_names=tuple(job.get("final_table_names", ())),
                source_table_names=tuple(job.get("source_table_names", ())),
                write_outputs=bool(job.get("write_outputs", True)),
                write_assembled_tables=bool(job.get("write_assembled_tables", True)),
                metadata={
                    "config_path": resolved_config_path.as_posix(),
                    **dict(job.get("metadata", {})),
                },
            )
        )
    return tuple(loaded_jobs)