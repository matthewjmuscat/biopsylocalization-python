"""Service API for post-run cohort assembly."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from patient_runner.cohort_assembly import PatientBatchCohortAssemblyConfig
from patient_runner.cohort_assembly import PatientBatchCohortAssemblyResult
from patient_runner.cohort_assembly import run_patient_batch_cohort_assembly
from patient_runner.cohort_assembly import summarize_patient_batch_cohort_assembly

from .config import DEFAULT_COHORT_ASSEMBLY_CONFIG_PATH
from .config import PostRunCohortAssemblyJobConfig
from .config import load_cohort_assembly_job_configs
from .manifest_loader import load_patient_batch_result_from_manifest
from .manifest_loader import resolve_patient_batch_manifest_path


@dataclass(frozen=True, slots=True)
class PostRunCohortAssemblyJobResult:
    """Completed post-run cohort assembly job result."""

    job_config: PostRunCohortAssemblyJobConfig
    manifest_path: Path
    output_dir: Path
    assembly_result: PatientBatchCohortAssemblyResult
    written_paths: tuple[Path, ...]
    summary: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest_path", Path(self.manifest_path))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "written_paths", tuple(Path(path) for path in self.written_paths))
        object.__setattr__(self, "summary", dict(self.summary))


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


def _default_output_dir(batch_output_root: Path) -> Path:
    return batch_output_root / "cohort_assembly"


def _assembly_config(job_config: PostRunCohortAssemblyJobConfig, output_dir: Path) -> PatientBatchCohortAssemblyConfig:
    return PatientBatchCohortAssemblyConfig(
        patient_uids=job_config.patient_uids,
        final_table_names=job_config.final_table_names,
        source_table_names=job_config.source_table_names,
        output_dir=output_dir,
        write_outputs=job_config.write_outputs,
        write_assembled_tables=job_config.write_assembled_tables,
    )


def _build_job_summary(job_config: PostRunCohortAssemblyJobConfig,
                       manifest_path: Path,
                       output_dir: Path,
                       assembly_result: PatientBatchCohortAssemblyResult,
                       written_paths: Sequence[Path]) -> dict[str, Any]:
    assembly_summary = summarize_patient_batch_cohort_assembly(assembly_result)
    return {
        "schema_version": "post_run_cohort_assembly_job_result_v1",
        "generated_utc": _utc_now_iso(),
        "job_name": job_config.name,
        "patient_runner_output_dir": job_config.patient_runner_output_dir.as_posix(),
        "manifest_path": manifest_path.as_posix(),
        "batch_output_root": assembly_result.batch_result.output_root.as_posix(),
        "output_dir": output_dir.as_posix(),
        "patient_count": assembly_result.batch_result.patient_count,
        "artifact_count": len(assembly_result.batch_result.artifact_paths),
        "selected_patient_uids": list(job_config.patient_uids),
        "selected_final_table_names": list(job_config.final_table_names),
        "selected_source_table_names": list(job_config.source_table_names),
        "write_outputs": job_config.write_outputs,
        "write_assembled_tables": job_config.write_assembled_tables,
        "assembly_summary": assembly_summary,
        "written_paths": [Path(path).as_posix() for path in written_paths],
        "metadata": _json_safe(job_config.metadata),
    }


def _write_job_summary(summary: Mapping[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "post_run_cohort_assembly_job_summary.json"
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary_path


def run_post_run_cohort_assembly(job_config: PostRunCohortAssemblyJobConfig) -> PostRunCohortAssemblyJobResult:
    """Run cohort assembly from a completed patient-runner manifest."""
    manifest_path = resolve_patient_batch_manifest_path(job_config.patient_runner_output_dir)
    batch_result = load_patient_batch_result_from_manifest(manifest_path)
    output_dir = job_config.output_dir or _default_output_dir(batch_result.output_root)
    assembly_result, _, written_paths = run_patient_batch_cohort_assembly(
        batch_result,
        _assembly_config(job_config, output_dir),
    )
    written_paths = tuple(written_paths)
    summary = _build_job_summary(job_config, manifest_path, output_dir, assembly_result, written_paths)
    if job_config.write_outputs:
        summary_path = _write_job_summary(summary, output_dir)
        written_paths = (*written_paths, summary_path)
        summary = {**summary, "written_paths": [Path(path).as_posix() for path in written_paths]}
    return PostRunCohortAssemblyJobResult(
        job_config=job_config,
        manifest_path=manifest_path,
        output_dir=output_dir,
        assembly_result=assembly_result,
        written_paths=written_paths,
        summary=summary,
    )


def run_post_run_cohort_assembly_jobs(config_path: str | Path = DEFAULT_COHORT_ASSEMBLY_CONFIG_PATH) -> tuple[PostRunCohortAssemblyJobResult, ...]:
    """Run every enabled cohort assembly job in a JSON config file."""
    return tuple(
        run_post_run_cohort_assembly(job_config)
        for job_config in load_cohort_assembly_job_configs(config_path)
    )


def format_post_run_cohort_assembly_summary(result: PostRunCohortAssemblyJobResult) -> str:
    """Return a concise human-readable summary for terminal callers."""
    assembly_summary = result.summary.get("assembly_summary", {})
    status_counts = assembly_summary.get("assembly_status_counts", {})
    return (
        f"job={result.job_config.name} "
        f"patients={result.summary.get('patient_count', 0)} "
        f"artifacts={result.summary.get('artifact_count', 0)} "
        f"assembled_tables={assembly_summary.get('assembled_table_count', 0)} "
        f"statuses={status_counts} "
        f"output_dir={result.output_dir}"
    )