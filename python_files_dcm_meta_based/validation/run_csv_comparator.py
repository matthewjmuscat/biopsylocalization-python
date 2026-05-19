from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .cohort_csv_comparator import (
    _build_column_dataframe,
    _build_inventory_rows,
    _build_summary_dataframe,
    _compare_common_file,
    _sanitize_path_stem,
)
from .run_output_paths import DEFAULT_VALIDATION_OUTPUT_ROOT


TIMESTAMPED_PATH_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"^uncertainties_file_auto_generated Date-.*\.csv$"),
        "uncertainties_file_auto_generated.csv",
    ),
)


def default_run_csv_comparison_output_dir(
    baseline_dir: str | Path,
    candidate_dir: str | Path,
    *,
    output_root: str | Path | None = None,
) -> Path:
    baseline_dir = Path(baseline_dir)
    candidate_dir = Path(candidate_dir)
    output_root_path = Path(output_root) if output_root is not None else DEFAULT_VALIDATION_OUTPUT_ROOT
    comparison_root = output_root_path / "run_comparisons"
    stem = (
        f"{_sanitize_path_stem(baseline_dir)}__vs__{_sanitize_path_stem(candidate_dir)}"
        "__all_csvs"
    )
    target = comparison_root / stem
    if not target.exists():
        return target

    suffix = 2
    while True:
        candidate = comparison_root / f"{stem}__{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def discover_run_csvs(output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Missing run output directory: {output_dir}")

    csvs: dict[str, Path] = {}
    for path in sorted(output_dir.rglob("*.csv")):
        relative_path = path.relative_to(output_dir)
        if relative_path.parts and relative_path.parts[0] == "logs":
            continue
        normalized_relative_path = _normalize_relative_csv_path(relative_path)
        csvs[normalized_relative_path] = path
    return csvs


def compare_run_csv_output_dirs(
    baseline_dir: str | Path,
    candidate_dir: str | Path,
    *,
    abs_tol: float = 1e-8,
    rel_tol: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline_csvs = discover_run_csvs(baseline_dir)
    candidate_csvs = discover_run_csvs(candidate_dir)

    inventory_df = pd.DataFrame(_build_inventory_rows(baseline_csvs, candidate_csvs))
    summary_rows: list[dict[str, object]] = []
    column_rows: list[dict[str, object]] = []
    for relative_path in sorted(set(baseline_csvs) & set(candidate_csvs)):
        result, file_column_rows = _compare_common_file(
            relative_path,
            baseline_csvs[relative_path],
            candidate_csvs[relative_path],
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )
        summary_rows.append(result.__dict__)
        column_rows.extend(file_column_rows)

    summary_df = _build_summary_dataframe(summary_rows)
    column_df = _build_column_dataframe(column_rows)
    return inventory_df, summary_df, column_df


def _normalize_relative_csv_path(relative_path: Path) -> str:
    relative_path_str = relative_path.as_posix()
    file_name = relative_path.name
    normalized_file_name = file_name
    for pattern, replacement in TIMESTAMPED_PATH_PATTERNS:
        if pattern.match(file_name):
            normalized_file_name = replacement
            break

    if normalized_file_name == file_name:
        return relative_path_str
    return relative_path.with_name(normalized_file_name).as_posix()
