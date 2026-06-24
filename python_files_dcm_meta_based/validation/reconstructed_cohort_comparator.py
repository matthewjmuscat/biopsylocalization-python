from __future__ import annotations

"""Decompose patient-runner outputs and compare reconstructed cohort surfaces."""

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from patient_runner.cohort_assembly import PatientBatchCohortAssemblyConfig
from patient_runner.cohort_assembly import PatientBatchCohortAssemblyResult
from patient_runner.cohort_assembly import run_patient_batch_cohort_assembly
from patient_runner.cohort_assembly import summarize_patient_batch_cohort_assembly
from patient_runner.cohort_assembly import write_patient_batch_cohort_assembly_outputs
from patient_runner.contracts import PatientBatchRunResult
from patient_runner.contracts import PatientRunResult
from post_run.cohort_assembly.manifest_loader import load_patient_batch_result_from_manifest

from .cohort_csv_comparator import compare_cohort_output_dirs
from .cohort_csv_comparator import write_comparison_outputs
from .run_output_paths import DEFAULT_VALIDATION_OUTPUT_ROOT
from .run_output_paths import resolve_existing_output_dir


RECONSTRUCTED_COHORT_COMPARISON_SCHEMA_VERSION = "reconstructed_cohort_comparison_v1"


@dataclass(frozen=True, slots=True)
class ReconstructedCohortComparisonConfig:
    """Config for post-run full-vs-split reconstructed cohort comparison."""

    reference_patient_runner_output_dir: Path
    split_patient_runner_output_dirs: Sequence[Path]
    output_dir: Path | None = None
    final_table_names: Sequence[str] = ()
    abs_tol: float = 1e-8
    rel_tol: float = 1e-6
    require_patient_uid_match: bool = True
    write_outputs: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "reference_patient_runner_output_dir", Path(self.reference_patient_runner_output_dir))
        split_dirs = tuple(Path(path) for path in self.split_patient_runner_output_dirs)
        if not split_dirs:
            raise ValueError("split_patient_runner_output_dirs cannot be empty")
        object.__setattr__(self, "split_patient_runner_output_dirs", split_dirs)
        if self.output_dir is not None:
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "final_table_names", _validate_name_filter(self.final_table_names, "final_table_names"))
        abs_tol = float(self.abs_tol)
        rel_tol = float(self.rel_tol)
        if abs_tol < 0:
            raise ValueError("abs_tol cannot be negative")
        if rel_tol < 0:
            raise ValueError("rel_tol cannot be negative")
        object.__setattr__(self, "abs_tol", abs_tol)
        object.__setattr__(self, "rel_tol", rel_tol)
        object.__setattr__(self, "require_patient_uid_match", bool(self.require_patient_uid_match))
        object.__setattr__(self, "write_outputs", bool(self.write_outputs))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class ReconstructedCohortSurface:
    """One reconstructed cohort surface built from patient-runner manifests."""

    label: str
    source_output_dirs: tuple[Path, ...]
    batch_result: PatientBatchRunResult
    assembly_result: PatientBatchCohortAssemblyResult
    output_root: Path
    cohort_csv_root: Path
    written_paths: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_output_dirs", tuple(Path(path) for path in self.source_output_dirs))
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "cohort_csv_root", Path(self.cohort_csv_root))
        object.__setattr__(self, "written_paths", tuple(Path(path) for path in self.written_paths))

    @property
    def patient_uids(self) -> tuple[str, ...]:
        return tuple(patient_result.patient_case.patient_uid for patient_result in self.batch_result.patient_results)


@dataclass(frozen=True, slots=True)
class ReconstructedCohortComparisonResult:
    """Result bundle for reconstructed full-vs-split cohort comparison."""

    reference_surface: ReconstructedCohortSurface
    split_surface: ReconstructedCohortSurface
    output_dir: Path
    inventory_df: pd.DataFrame
    summary_df: pd.DataFrame
    column_df: pd.DataFrame
    summary: Mapping[str, Any]
    written_paths: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "summary", dict(self.summary))
        object.__setattr__(self, "written_paths", tuple(Path(path) for path in self.written_paths))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _safe_path_name(value: str) -> str:
    safe = str(value).strip().replace("/", "_").replace("\\", "_")
    for old, new in ((" ", "_"), (":", "_"), (",", "_")):
        safe = safe.replace(old, new)
    return safe or "unknown"


def _validate_name_filter(values: Sequence[str], source_name: str) -> tuple[str, ...]:
    resolved_values = tuple(values)
    if any(not isinstance(value, str) for value in resolved_values):
        raise TypeError(f"{source_name} entries must be strings")
    if any(value.strip() == "" for value in resolved_values):
        raise ValueError(f"{source_name} cannot contain empty values")
    if len(set(resolved_values)) != len(resolved_values):
        raise ValueError(f"{source_name} cannot contain duplicates")
    return resolved_values


def _default_output_dir(reference_output_dir: Path,
                        split_output_dirs: Sequence[Path],
                        *,
                        output_root: str | Path | None = None) -> Path:
    output_root_path = Path(output_root) if output_root is not None else DEFAULT_VALIDATION_OUTPUT_ROOT
    stem = "__vs__".join(
        [_safe_path_name(reference_output_dir.name), "split", *(_safe_path_name(path.name) for path in split_output_dirs)]
    )
    target = output_root_path.joinpath("reconstructed_cohort_comparisons", stem)
    if not target.exists():
        return target
    suffix = 2
    while True:
        candidate = output_root_path.joinpath("reconstructed_cohort_comparisons", f"{stem}__{suffix}")
        if not candidate.exists():
            return candidate
        suffix += 1


def _patient_uid(patient_result: PatientRunResult) -> str:
    return patient_result.patient_case.patient_uid


def _combine_batch_results(*,
                           label: str,
                           source_output_dirs: Sequence[Path],
                           batch_results: Sequence[PatientBatchRunResult],
                           output_root: Path) -> PatientBatchRunResult:
    patient_results_by_uid: dict[str, PatientRunResult] = {}
    duplicate_patient_uids: list[str] = []
    for batch_result in batch_results:
        for patient_result in batch_result.patient_results:
            patient_uid = _patient_uid(patient_result)
            if patient_uid in patient_results_by_uid:
                duplicate_patient_uids.append(patient_uid)
                continue
            patient_results_by_uid[patient_uid] = patient_result
    if duplicate_patient_uids:
        raise ValueError(f"Duplicate patient_uids in {label} reconstructed surface: {sorted(duplicate_patient_uids)}")

    sorted_patient_results = tuple(
        patient_results_by_uid[patient_uid]
        for patient_uid in sorted(patient_results_by_uid)
    )
    return PatientBatchRunResult.from_patient_results(
        output_root=output_root,
        patient_results=sorted_patient_results,
        elapsed_seconds=sum(float(batch_result.elapsed_seconds) for batch_result in batch_results),
        metadata={
            "schema_version": RECONSTRUCTED_COHORT_COMPARISON_SCHEMA_VERSION,
            "reconstruction_label": label,
            "source_output_dirs": [Path(path).as_posix() for path in source_output_dirs],
            "source_batch_count": len(batch_results),
        },
    )


def _resolve_batch_result(path: Path) -> tuple[Path, PatientBatchRunResult]:
    resolved_path = resolve_existing_output_dir(path)
    return resolved_path, load_patient_batch_result_from_manifest(resolved_path)


def _bool_from_report_value(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text_value = str(value).strip().lower()
    if text_value in {"true", "1", "yes", "y"}:
        return True
    if text_value in {"false", "0", "no", "n"}:
        return False
    return default


def _csv_index_for_table(assembly_result: PatientBatchCohortAssemblyResult, table_name: str) -> bool:
    assembly_df = assembly_result.assembly_df
    if assembly_df.empty or "validation_csv_index" not in assembly_df.columns:
        return True
    table_rows = assembly_df[assembly_df["final_table_name"].eq(table_name)]
    if table_rows.empty:
        return True
    return _bool_from_report_value(table_rows.iloc[0].get("validation_csv_index"), True)


def _write_reconstructed_cohort_tables(surface: ReconstructedCohortSurface) -> tuple[Path, ...]:
    cohort_dir = surface.cohort_csv_root.joinpath("Output CSVs", "Cohort")
    cohort_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []
    for table_name, dataframe in surface.assembly_result.assembled_tables.items():
        path = cohort_dir.joinpath(f"{table_name}.csv")
        dataframe.to_csv(path, index=_csv_index_for_table(surface.assembly_result, table_name))
        written_paths.append(path)
    return tuple(written_paths)


def _write_assembly_reports_without_tables(surface: ReconstructedCohortSurface) -> tuple[Path, ...]:
    return write_patient_batch_cohort_assembly_outputs(
        surface.assembly_result,
        output_dir=surface.output_root.joinpath("assembly"),
        write_assembled_tables=False,
    )


def reconstruct_patient_runner_cohort_surface(label: str,
                                              patient_runner_output_dirs: Sequence[str | Path],
                                              *,
                                              output_root: str | Path,
                                              final_table_names: Sequence[str] = (),
                                              write_outputs: bool = True) -> ReconstructedCohortSurface:
    """Load patient-runner manifests and reconstruct one cohort CSV surface."""

    resolved_sources: list[Path] = []
    batch_results: list[PatientBatchRunResult] = []
    for path in patient_runner_output_dirs:
        resolved_path, batch_result = _resolve_batch_result(Path(path))
        resolved_sources.append(resolved_path)
        batch_results.append(batch_result)

    surface_output_root = Path(output_root).joinpath(label)
    combined_batch = _combine_batch_results(
        label=label,
        source_output_dirs=resolved_sources,
        batch_results=batch_results,
        output_root=surface_output_root.joinpath("combined_patient_batch"),
    )
    assembly_config = PatientBatchCohortAssemblyConfig(
        final_table_names=final_table_names,
        output_dir=surface_output_root.joinpath("assembly"),
        write_outputs=False,
        write_assembled_tables=False,
    )
    assembly_result, _, assembly_written_paths = run_patient_batch_cohort_assembly(
        combined_batch,
        assembly_config,
    )
    surface = ReconstructedCohortSurface(
        label=label,
        source_output_dirs=tuple(resolved_sources),
        batch_result=combined_batch,
        assembly_result=assembly_result,
        output_root=surface_output_root,
        cohort_csv_root=surface_output_root.joinpath("cohort_csv_surface"),
        written_paths=tuple(assembly_written_paths),
    )
    if write_outputs:
        report_written_paths = _write_assembly_reports_without_tables(surface)
        reconstructed_written_paths = _write_reconstructed_cohort_tables(surface)
    else:
        report_written_paths = ()
        reconstructed_written_paths = ()
    return ReconstructedCohortSurface(
        label=surface.label,
        source_output_dirs=surface.source_output_dirs,
        batch_result=surface.batch_result,
        assembly_result=surface.assembly_result,
        output_root=surface.output_root,
        cohort_csv_root=surface.cohort_csv_root,
        written_paths=(*surface.written_paths, *report_written_paths, *reconstructed_written_paths),
    )


def _patient_set_summary(reference_surface: ReconstructedCohortSurface,
                         split_surface: ReconstructedCohortSurface) -> dict[str, Any]:
    reference_uids = set(reference_surface.patient_uids)
    split_uids = set(split_surface.patient_uids)
    return {
        "reference_patient_count": len(reference_uids),
        "split_patient_count": len(split_uids),
        "patient_uid_sets_match": reference_uids == split_uids,
        "reference_only_patient_uids": sorted(reference_uids - split_uids),
        "split_only_patient_uids": sorted(split_uids - reference_uids),
    }


def _comparison_status_counts(summary_df: pd.DataFrame) -> dict[str, int]:
    if summary_df.empty or "comparison_status" not in summary_df.columns:
        return {}
    return {str(key): int(value) for key, value in Counter(summary_df["comparison_status"]).items()}


def _missing_file_count(inventory_df: pd.DataFrame) -> int:
    if inventory_df.empty:
        return 0
    return int((~inventory_df["present_in_baseline"] | ~inventory_df["present_in_candidate"]).sum())


def _non_ok_file_count(summary_df: pd.DataFrame) -> int:
    status_counts = _comparison_status_counts(summary_df)
    return int(sum(count for status, count in status_counts.items() if status != "ok"))


def _build_summary(*,
                   config: ReconstructedCohortComparisonConfig,
                   output_dir: Path,
                   reference_surface: ReconstructedCohortSurface,
                   split_surface: ReconstructedCohortSurface,
                   inventory_df: pd.DataFrame,
                   summary_df: pd.DataFrame) -> dict[str, Any]:
    patient_summary = _patient_set_summary(reference_surface, split_surface)
    reference_assembly_summary = summarize_patient_batch_cohort_assembly(reference_surface.assembly_result)
    split_assembly_summary = summarize_patient_batch_cohort_assembly(split_surface.assembly_result)
    non_ok_count = _non_ok_file_count(summary_df)
    missing_file_count = _missing_file_count(inventory_df)
    reference_missing_failures = int(reference_assembly_summary.get("missing_artifact_failure_count", 0) or 0)
    split_missing_failures = int(split_assembly_summary.get("missing_artifact_failure_count", 0) or 0)
    patient_sets_ok = bool(patient_summary["patient_uid_sets_match"]) or not config.require_patient_uid_match
    overall_status = "passed" if (
        patient_sets_ok
        and non_ok_count == 0
        and missing_file_count == 0
        and reference_missing_failures == 0
        and split_missing_failures == 0
    ) else "failed"
    return {
        "schema_version": RECONSTRUCTED_COHORT_COMPARISON_SCHEMA_VERSION,
        "generated_utc": _utc_now_iso(),
        "overall_status": overall_status,
        "output_dir": output_dir.as_posix(),
        "reference_patient_runner_output_dir": config.reference_patient_runner_output_dir.as_posix(),
        "split_patient_runner_output_dirs": [Path(path).as_posix() for path in config.split_patient_runner_output_dirs],
        "require_patient_uid_match": config.require_patient_uid_match,
        "patient_sets": patient_summary,
        "compared_files": int(len(summary_df)),
        "missing_file_count": missing_file_count,
        "non_ok_file_count": non_ok_count,
        "comparison_status_counts": _comparison_status_counts(summary_df),
        "reference_assembly_summary": reference_assembly_summary,
        "split_assembly_summary": split_assembly_summary,
        "metadata": dict(config.metadata),
    }


def _write_summary(summary: Mapping[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir.joinpath("reconstructed_cohort_comparison_summary.json")
    path.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_reconstructed_cohort_comparison(config: ReconstructedCohortComparisonConfig) -> ReconstructedCohortComparisonResult:
    """Compare a full patient-runner reconstruction with split-run reconstruction."""

    reference_resolved = resolve_existing_output_dir(config.reference_patient_runner_output_dir)
    split_resolved = tuple(resolve_existing_output_dir(path) for path in config.split_patient_runner_output_dirs)
    output_dir = config.output_dir or _default_output_dir(reference_resolved, split_resolved)

    reference_surface = reconstruct_patient_runner_cohort_surface(
        "reference",
        (reference_resolved,),
        output_root=output_dir,
        final_table_names=config.final_table_names,
        write_outputs=config.write_outputs,
    )
    split_surface = reconstruct_patient_runner_cohort_surface(
        "split",
        split_resolved,
        output_root=output_dir,
        final_table_names=config.final_table_names,
        write_outputs=config.write_outputs,
    )
    patient_summary = _patient_set_summary(reference_surface, split_surface)
    if config.require_patient_uid_match and not patient_summary["patient_uid_sets_match"]:
        raise ValueError(
            "Reference and split reconstructed surfaces have different patient_uid sets: "
            f"reference_only={patient_summary['reference_only_patient_uids']} "
            f"split_only={patient_summary['split_only_patient_uids']}"
        )

    inventory_df, summary_df, column_df = compare_cohort_output_dirs(
        reference_surface.cohort_csv_root,
        split_surface.cohort_csv_root,
        abs_tol=config.abs_tol,
        rel_tol=config.rel_tol,
    )
    written_paths: list[Path] = [*reference_surface.written_paths, *split_surface.written_paths]
    comparison_output_dir = Path(output_dir).joinpath("comparison")
    if config.write_outputs:
        write_comparison_outputs(
            comparison_output_dir,
            baseline_dir=reference_surface.cohort_csv_root,
            candidate_dir=split_surface.cohort_csv_root,
            inventory_df=inventory_df,
            summary_df=summary_df,
            column_df=column_df,
        )
        written_paths.extend(
            (
                comparison_output_dir.joinpath("file_inventory.csv"),
                comparison_output_dir.joinpath("file_comparison_summary.csv"),
                comparison_output_dir.joinpath("column_drift_summary.csv"),
                comparison_output_dir.joinpath("comparison_report.md"),
            )
        )

    summary = _build_summary(
        config=config,
        output_dir=Path(output_dir),
        reference_surface=reference_surface,
        split_surface=split_surface,
        inventory_df=inventory_df,
        summary_df=summary_df,
    )
    if config.write_outputs:
        written_paths.append(_write_summary(summary, Path(output_dir)))

    return ReconstructedCohortComparisonResult(
        reference_surface=reference_surface,
        split_surface=split_surface,
        output_dir=Path(output_dir),
        inventory_df=inventory_df,
        summary_df=summary_df,
        column_df=column_df,
        summary=summary,
        written_paths=tuple(written_paths),
    )


def format_reconstructed_cohort_comparison_summary(result: ReconstructedCohortComparisonResult) -> str:
    """Return a compact console summary for reconstructed cohort comparison."""

    patient_sets = result.summary.get("patient_sets", {})
    return (
        "[reconstructed-cohort] overall={overall} compared={compared} non_ok={non_ok} "
        "missing={missing} patients_match={patients_match} reference_patients={reference_patients} "
        "split_patients={split_patients}"
    ).format(
        overall=result.summary.get("overall_status", "unknown"),
        compared=result.summary.get("compared_files", 0),
        non_ok=result.summary.get("non_ok_file_count", 0),
        missing=result.summary.get("missing_file_count", 0),
        patients_match=patient_sets.get("patient_uid_sets_match", False),
        reference_patients=patient_sets.get("reference_patient_count", 0),
        split_patients=patient_sets.get("split_patient_count", 0),
    )


__all__ = [
    "RECONSTRUCTED_COHORT_COMPARISON_SCHEMA_VERSION",
    "ReconstructedCohortComparisonConfig",
    "ReconstructedCohortComparisonResult",
    "ReconstructedCohortSurface",
    "format_reconstructed_cohort_comparison_summary",
    "reconstruct_patient_runner_cohort_surface",
    "run_reconstructed_cohort_comparison",
]