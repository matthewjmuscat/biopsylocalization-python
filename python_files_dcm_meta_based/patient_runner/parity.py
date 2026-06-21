"""Post-run parity orchestration for completed patient-runner outputs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from output_artifacts import build_shadow_stitch_pairs_from_output_assembly_plans
from output_artifacts.stitch_validation import ShadowStitchPair
from validation import DEFAULT_VALIDATION_OUTPUT_ROOT
from validation import compare_run_csv_output_dirs
from validation import resolve_existing_output_dir
from validation import write_comparison_outputs
from validation.cohort_csv_comparator import _build_column_dataframe
from validation.cohort_csv_comparator import _build_summary_dataframe
from validation.cohort_csv_comparator import _compare_common_file


PATIENT_RUNNER_POST_RUN_PARITY_SCHEMA_VERSION = "patient_runner_post_run_parity_v1"
DEFAULT_PATIENT_RUNNER_PARITY_DIR_NAME = "patient_runner_post_run_parity"
ASSEMBLED_COHORT_INVENTORY_COLUMNS = [
    "relative_path",
    "final_table_name",
    "source_table_name",
    "source_output_section",
    "candidate_relative_path",
    "present_in_baseline",
    "present_in_candidate",
    "baseline_path",
    "candidate_path",
]


class PatientRunnerParitySurface(str, Enum):
    """Completed-output surfaces that the post-run parity harness can compare."""

    ASSEMBLED_COHORT = "assembled_cohort"
    RECURSIVE_CSV = "recursive_csv"


@dataclass(frozen=True, slots=True)
class PatientRunnerPostRunParityConfig:
    """Configuration for comparing a completed legacy run with patient-runner output.

    This is an artifact-level validation surface. It does not call scientific
    stages and does not touch the frozen legacy pathway.
    """

    legacy_output_dir: Path
    patient_runner_output_dir: Path
    output_dir: Path | None = None
    assembled_table_dir: Path | None = None
    surfaces: Sequence[PatientRunnerParitySurface | str] = (PatientRunnerParitySurface.ASSEMBLED_COHORT,)
    final_table_names: Sequence[str] = ()
    abs_tol: float = 1e-8
    rel_tol: float = 1e-6
    write_outputs: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "legacy_output_dir", Path(self.legacy_output_dir))
        object.__setattr__(self, "patient_runner_output_dir", Path(self.patient_runner_output_dir))
        if self.output_dir is not None:
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.assembled_table_dir is not None:
            object.__setattr__(self, "assembled_table_dir", Path(self.assembled_table_dir))
        surfaces = tuple(PatientRunnerParitySurface(surface) for surface in self.surfaces)
        if not surfaces:
            raise ValueError("surfaces cannot be empty")
        if len(set(surfaces)) != len(surfaces):
            raise ValueError("surfaces cannot contain duplicates")
        object.__setattr__(self, "surfaces", surfaces)
        object.__setattr__(self, "final_table_names", _validate_name_filter(self.final_table_names, "final_table_names"))
        abs_tol = float(self.abs_tol)
        rel_tol = float(self.rel_tol)
        if abs_tol < 0:
            raise ValueError("abs_tol cannot be negative")
        if rel_tol < 0:
            raise ValueError("rel_tol cannot be negative")
        object.__setattr__(self, "abs_tol", abs_tol)
        object.__setattr__(self, "rel_tol", rel_tol)
        object.__setattr__(self, "write_outputs", bool(self.write_outputs))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class PatientRunnerParitySurfaceResult:
    """Comparison dataframes for one completed-output parity surface."""

    surface: PatientRunnerParitySurface | str
    baseline_dir: Path
    candidate_dir: Path
    inventory_df: pd.DataFrame
    summary_df: pd.DataFrame
    column_df: pd.DataFrame

    def __post_init__(self) -> None:
        object.__setattr__(self, "surface", PatientRunnerParitySurface(self.surface))
        object.__setattr__(self, "baseline_dir", Path(self.baseline_dir))
        object.__setattr__(self, "candidate_dir", Path(self.candidate_dir))


@dataclass(frozen=True, slots=True)
class PatientRunnerPostRunParityResult:
    """Result bundle for post-run legacy-versus-patient-runner parity."""

    legacy_output_dir: Path
    patient_runner_output_dir: Path
    output_dir: Path
    surface_results: Mapping[str, PatientRunnerParitySurfaceResult]
    summary: Mapping[str, Any]
    written_paths: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "legacy_output_dir", Path(self.legacy_output_dir))
        object.__setattr__(self, "patient_runner_output_dir", Path(self.patient_runner_output_dir))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "surface_results", dict(self.surface_results))
        object.__setattr__(self, "summary", dict(self.summary))
        object.__setattr__(self, "written_paths", tuple(Path(path) for path in self.written_paths))


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


def _selected_stitch_pairs(stitch_pairs: Sequence[ShadowStitchPair],
                           final_table_names: Sequence[str]) -> tuple[ShadowStitchPair, ...]:
    selected_final_table_names = set(final_table_names)
    return tuple(
        pair
        for pair in stitch_pairs
        if not selected_final_table_names or pair.final_table_name in selected_final_table_names
    )


def default_patient_runner_post_run_parity_output_dir(legacy_output_dir: str | Path,
                                                      patient_runner_output_dir: str | Path,
                                                      *,
                                                      output_root: str | Path | None = None) -> Path:
    """Return a unique default output directory for post-run parity reports."""
    legacy_output_dir = Path(legacy_output_dir)
    patient_runner_output_dir = Path(patient_runner_output_dir)
    output_root_path = Path(output_root) if output_root is not None else DEFAULT_VALIDATION_OUTPUT_ROOT
    parity_root = output_root_path.joinpath(DEFAULT_PATIENT_RUNNER_PARITY_DIR_NAME)
    stem = f"{_safe_path_name(legacy_output_dir.name)}__vs__{_safe_path_name(patient_runner_output_dir.name)}"
    target = parity_root.joinpath(stem)
    if not target.exists():
        return target
    suffix = 2
    while True:
        candidate = parity_root.joinpath(f"{stem}__{suffix}")
        if not candidate.exists():
            return candidate
        suffix += 1


def compare_patient_runner_assembled_cohort_tables(legacy_output_dir: str | Path,
                                                   patient_runner_output_dir: str | Path,
                                                   *,
                                                   assembled_table_dir: str | Path | None = None,
                                                   final_table_names: Sequence[str] = (),
                                                   abs_tol: float = 1e-8,
                                                   rel_tol: float = 1e-6,
                                                   stitch_pairs: Sequence[ShadowStitchPair] | None = None) -> PatientRunnerParitySurfaceResult:
    """Compare legacy final cohort CSVs with patient-runner assembled cohort CSVs."""
    legacy_output_dir = Path(legacy_output_dir)
    patient_runner_output_dir = Path(patient_runner_output_dir)
    resolved_stitch_pairs = (
        build_shadow_stitch_pairs_from_output_assembly_plans()
        if stitch_pairs is None
        else tuple(stitch_pairs)
    )
    legacy_cohort_dir = legacy_output_dir.joinpath("Output CSVs", "Cohort")
    resolved_assembled_table_dir = (
        Path(assembled_table_dir)
        if assembled_table_dir is not None
        else patient_runner_output_dir.joinpath("cohort_assembly", "assembled_tables")
    )
    if not legacy_cohort_dir.is_dir():
        raise FileNotFoundError(f"Missing legacy cohort CSV directory: {legacy_cohort_dir}")
    if not resolved_assembled_table_dir.is_dir():
        raise FileNotFoundError(f"Missing patient-runner assembled table directory: {resolved_assembled_table_dir}")

    inventory_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    column_rows: list[dict[str, object]] = []
    for pair in _selected_stitch_pairs(resolved_stitch_pairs, final_table_names):
        relative_path = f"{pair.final_table_name}{pair.file_extension}"
        baseline_path = legacy_cohort_dir.joinpath(relative_path)
        candidate_file_name = f"{_safe_path_name(pair.final_table_name)}{pair.file_extension}"
        candidate_path = resolved_assembled_table_dir.joinpath(candidate_file_name)
        baseline_present = baseline_path.is_file()
        candidate_present = candidate_path.is_file()
        inventory_rows.append(
            {
                "relative_path": relative_path,
                "final_table_name": pair.final_table_name,
                "source_table_name": pair.source_table_name,
                "source_output_section": pair.source_output_section,
                "candidate_relative_path": candidate_file_name,
                "present_in_baseline": baseline_present,
                "present_in_candidate": candidate_present,
                "baseline_path": baseline_path.as_posix() if baseline_present else "",
                "candidate_path": candidate_path.as_posix() if candidate_present else "",
            }
        )
        if not (baseline_present and candidate_present):
            continue

        result, file_column_rows = _compare_common_file(
            relative_path,
            baseline_path,
            candidate_path,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )
        summary_rows.append(result.__dict__)
        column_rows.extend(file_column_rows)

    return PatientRunnerParitySurfaceResult(
        surface=PatientRunnerParitySurface.ASSEMBLED_COHORT,
        baseline_dir=legacy_cohort_dir,
        candidate_dir=resolved_assembled_table_dir,
        inventory_df=pd.DataFrame(inventory_rows, columns=ASSEMBLED_COHORT_INVENTORY_COLUMNS),
        summary_df=_build_summary_dataframe(summary_rows),
        column_df=_build_column_dataframe(column_rows),
    )


def compare_patient_runner_recursive_csvs(legacy_output_dir: str | Path,
                                          patient_runner_output_dir: str | Path,
                                          *,
                                          abs_tol: float = 1e-8,
                                          rel_tol: float = 1e-6) -> PatientRunnerParitySurfaceResult:
    """Recursively compare CSVs in two path-compatible completed output roots."""
    legacy_output_dir = Path(legacy_output_dir)
    patient_runner_output_dir = Path(patient_runner_output_dir)
    inventory_df, summary_df, column_df = compare_run_csv_output_dirs(
        legacy_output_dir,
        patient_runner_output_dir,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )
    return PatientRunnerParitySurfaceResult(
        surface=PatientRunnerParitySurface.RECURSIVE_CSV,
        baseline_dir=legacy_output_dir,
        candidate_dir=patient_runner_output_dir,
        inventory_df=inventory_df,
        summary_df=summary_df,
        column_df=column_df,
    )


def run_patient_runner_post_run_parity(config: PatientRunnerPostRunParityConfig) -> PatientRunnerPostRunParityResult:
    """Compare completed legacy and patient-runner output surfaces after both runs finish."""
    legacy_output_dir = resolve_existing_output_dir(config.legacy_output_dir)
    patient_runner_output_dir = resolve_existing_output_dir(config.patient_runner_output_dir)
    output_dir = config.output_dir or default_patient_runner_post_run_parity_output_dir(
        legacy_output_dir,
        patient_runner_output_dir,
    )

    surface_results: dict[str, PatientRunnerParitySurfaceResult] = {}
    written_paths: list[Path] = []
    for surface in config.surfaces:
        if surface == PatientRunnerParitySurface.ASSEMBLED_COHORT:
            result = compare_patient_runner_assembled_cohort_tables(
                legacy_output_dir,
                patient_runner_output_dir,
                assembled_table_dir=config.assembled_table_dir,
                final_table_names=config.final_table_names,
                abs_tol=config.abs_tol,
                rel_tol=config.rel_tol,
            )
        elif surface == PatientRunnerParitySurface.RECURSIVE_CSV:
            result = compare_patient_runner_recursive_csvs(
                legacy_output_dir,
                patient_runner_output_dir,
                abs_tol=config.abs_tol,
                rel_tol=config.rel_tol,
            )
        else:
            raise ValueError(f"Unsupported patient-runner parity surface: {surface}")

        surface_results[surface.value] = result
        if config.write_outputs:
            surface_output_dir = Path(output_dir).joinpath(surface.value)
            written_paths.extend(_write_surface_outputs(result, surface_output_dir))

    summary = summarize_patient_runner_post_run_parity_surfaces(
        surface_results,
        legacy_output_dir=legacy_output_dir,
        patient_runner_output_dir=patient_runner_output_dir,
        output_dir=output_dir,
        metadata=config.metadata,
    )
    if config.write_outputs:
        summary_path = Path(output_dir).joinpath("patient_runner_post_run_parity_summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written_paths.append(summary_path)

    return PatientRunnerPostRunParityResult(
        legacy_output_dir=legacy_output_dir,
        patient_runner_output_dir=patient_runner_output_dir,
        output_dir=output_dir,
        surface_results=surface_results,
        summary=summary,
        written_paths=tuple(written_paths),
    )


def summarize_patient_runner_parity_surface(result: PatientRunnerParitySurfaceResult) -> dict[str, Any]:
    """Summarize one parity surface result."""
    inventory_df = result.inventory_df
    summary_df = result.summary_df
    missing_from_baseline = int((~inventory_df["present_in_baseline"]).sum()) if "present_in_baseline" in inventory_df else 0
    missing_from_candidate = int((~inventory_df["present_in_candidate"]).sum()) if "present_in_candidate" in inventory_df else 0
    status_counts = dict(Counter(summary_df["comparison_status"])) if "comparison_status" in summary_df else {}
    compared_files = int(len(summary_df))
    non_ok_count = sum(count for status, count in status_counts.items() if status != "ok")
    missing_file_count = missing_from_baseline + missing_from_candidate
    surface_status = "passed" if missing_file_count == 0 and non_ok_count == 0 else "failed"
    if inventory_df.empty and summary_df.empty:
        surface_status = "skipped"
    return {
        "schema_version": PATIENT_RUNNER_POST_RUN_PARITY_SCHEMA_VERSION,
        "generated_utc": _utc_now_iso(),
        "surface": result.surface.value,
        "status": surface_status,
        "baseline_dir": result.baseline_dir.as_posix(),
        "candidate_dir": result.candidate_dir.as_posix(),
        "inventory_rows": int(len(inventory_df)),
        "compared_files": compared_files,
        "missing_from_baseline": missing_from_baseline,
        "missing_from_candidate": missing_from_candidate,
        "missing_file_count": missing_file_count,
        "comparison_status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "non_ok_file_count": int(non_ok_count),
    }


def summarize_patient_runner_post_run_parity_surfaces(surface_results: Mapping[str, PatientRunnerParitySurfaceResult],
                                                      *,
                                                      legacy_output_dir: Path,
                                                      patient_runner_output_dir: Path,
                                                      output_dir: Path,
                                                      metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build a JSON-ready summary for all requested parity surfaces."""
    surface_summaries = {
        surface_name: summarize_patient_runner_parity_surface(surface_result)
        for surface_name, surface_result in surface_results.items()
    }
    statuses = [summary["status"] for summary in surface_summaries.values()]
    if not statuses:
        overall_status = "skipped"
    elif any(status == "failed" for status in statuses):
        overall_status = "failed"
    elif all(status == "passed" for status in statuses):
        overall_status = "passed"
    else:
        overall_status = "partial"
    return {
        "schema_version": PATIENT_RUNNER_POST_RUN_PARITY_SCHEMA_VERSION,
        "generated_utc": _utc_now_iso(),
        "overall_status": overall_status,
        "legacy_output_dir": legacy_output_dir.as_posix(),
        "patient_runner_output_dir": patient_runner_output_dir.as_posix(),
        "output_dir": Path(output_dir).as_posix(),
        "surface_count": int(len(surface_summaries)),
        "surface_status_counts": dict(Counter(statuses)),
        "surfaces": surface_summaries,
        "metadata": dict(metadata or {}),
    }


def summarize_patient_runner_post_run_parity(result: PatientRunnerPostRunParityResult) -> dict[str, Any]:
    """Return the JSON-ready summary stored on a post-run parity result."""
    return dict(result.summary)


def format_patient_runner_post_run_parity_summary(result: PatientRunnerPostRunParityResult) -> str:
    """Return a compact console summary for a post-run parity result."""
    surface_chunks = []
    for surface_name, surface_summary in result.summary.get("surfaces", {}).items():
        surface_chunks.append(
            "{surface}: status={status}, compared={compared}, missing={missing}, non_ok={non_ok}".format(
                surface=surface_name,
                status=surface_summary.get("status", "unknown"),
                compared=surface_summary.get("compared_files", 0),
                missing=surface_summary.get("missing_file_count", 0),
                non_ok=surface_summary.get("non_ok_file_count", 0),
            )
        )
    return "[patient-runner-parity] overall={overall} | {surfaces}".format(
        overall=result.summary.get("overall_status", "unknown"),
        surfaces=" | ".join(surface_chunks) if surface_chunks else "no surfaces",
    )


def _write_surface_outputs(result: PatientRunnerParitySurfaceResult, output_dir: Path) -> tuple[Path, ...]:
    write_comparison_outputs(
        output_dir,
        baseline_dir=result.baseline_dir,
        candidate_dir=result.candidate_dir,
        inventory_df=result.inventory_df,
        summary_df=result.summary_df,
        column_df=result.column_df,
    )
    return (
        output_dir.joinpath("file_inventory.csv"),
        output_dir.joinpath("file_comparison_summary.csv"),
        output_dir.joinpath("column_drift_summary.csv"),
        output_dir.joinpath("comparison_report.md"),
    )