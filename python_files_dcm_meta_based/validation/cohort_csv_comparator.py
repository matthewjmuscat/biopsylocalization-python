from __future__ import annotations

import re
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from legacy_data_keys import legacy_data_keys

from .run_output_paths import DEFAULT_VALIDATION_OUTPUT_ROOT, discover_cohort_csvs


LEGACY_PATIENT_REFERENCE_KEYS = legacy_data_keys.patient_reference
LEGACY_STRUCTURE_RECORD_KEYS = legacy_data_keys.structure_record

KNOWN_MULTIINDEX_FILES = {
    "Cohort: Tissue class - distances global results.csv": [0, 1],
}

IDENTIFIER_PRIORITY = [
    "Patient ID",
    "Base patient ID",
    "Fraction label",
    LEGACY_PATIENT_REFERENCE_KEYS.fraction_number_key,
    "Biopsy member ID",
    "Real biopsy attempt ID",
    "Attempt family ID",
    "Bx ID",
    "Bx ROI",
    "Bx refnum",
    "Bx ref #",
    "Bx index",
    LEGACY_STRUCTURE_RECORD_KEYS.simulated_bool_key,
    LEGACY_STRUCTURE_RECORD_KEYS.simulated_type_key,
    "Tissue class",
    "Struct type",
    "Relative DIL ID",
    "Relative DIL index",
    "Relative structure type",
    "Relative structure ref #",
    "Target structure type",
    "Target structure ref #",
    "Matched real biopsy ref #",
    "Matched real biopsy index",
    "Multiplicity index",
    LEGACY_STRUCTURE_RECORD_KEYS.roi_key,
    LEGACY_STRUCTURE_RECORD_KEYS.ref_number_key,
    LEGACY_STRUCTURE_RECORD_KEYS.index_number_key,
    "Trial num",
    "Voxel index",
]

COLUMN_SUMMARY_COLUMNS = [
    "relative_path",
    "column_name",
    "column_type",
    "cells_compared",
    "cells_outside_tolerance",
    "mean_abs_diff",
    "max_abs_diff",
    "mean_rel_diff",
    "max_rel_diff",
]


@dataclass(frozen=True)
class FileComparisonResult:
    relative_path: str
    header_kind: str
    comparison_status: str
    join_strategy: str
    join_columns: str
    baseline_rows: int
    candidate_rows: int
    baseline_columns: int
    candidate_columns: int
    common_columns: int
    baseline_only_columns: int
    candidate_only_columns: int
    matched_rows: int
    baseline_only_rows: int
    candidate_only_rows: int
    numeric_columns_compared: int
    numeric_cells_compared: int
    numeric_cells_outside_tolerance: int
    text_columns_compared: int
    text_cells_compared: int
    text_cells_mismatched: int
    mean_abs_diff: float
    max_abs_diff: float
    max_rel_diff: float
    note: str


def default_comparison_output_dir(
    baseline_dir: str | Path,
    candidate_dir: str | Path,
    *,
    output_root: str | Path | None = None,
) -> Path:
    baseline_dir = Path(baseline_dir)
    candidate_dir = Path(candidate_dir)
    output_root_path = Path(output_root) if output_root is not None else DEFAULT_VALIDATION_OUTPUT_ROOT
    comparison_root = output_root_path / "run_comparisons"
    stem = f"{_sanitize_path_stem(baseline_dir)}__vs__{_sanitize_path_stem(candidate_dir)}"
    target = comparison_root / stem
    if not target.exists():
        return target
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return comparison_root / f"{stem}__{timestamp}"


def compare_cohort_output_dirs(
    baseline_dir: str | Path,
    candidate_dir: str | Path,
    *,
    abs_tol: float = 1e-8,
    rel_tol: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline_csvs = discover_cohort_csvs(baseline_dir)
    candidate_csvs = discover_cohort_csvs(candidate_dir)

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


def write_comparison_outputs(
    output_dir: str | Path,
    *,
    baseline_dir: str | Path,
    candidate_dir: str | Path,
    inventory_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    column_df: pd.DataFrame,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory_df.to_csv(output_dir / "file_inventory.csv", index=False)
    summary_df.to_csv(output_dir / "file_comparison_summary.csv", index=False)
    column_df.to_csv(output_dir / "column_drift_summary.csv", index=False)

    _write_report(
        output_dir,
        baseline_dir=Path(baseline_dir),
        candidate_dir=Path(candidate_dir),
        inventory_df=inventory_df,
        summary_df=summary_df,
        column_df=column_df,
    )


def format_console_summary(summary_df: pd.DataFrame, inventory_df: pd.DataFrame) -> str:
    structural = int((summary_df["comparison_status"] == "structural_difference").sum())
    drift = int((summary_df["comparison_status"] == "drift_detected").sum())
    manual = int((summary_df["comparison_status"] == "requires_manual_review").sum())
    missing = int((~inventory_df["present_in_baseline"] | ~inventory_df["present_in_candidate"]).sum())
    return (
        "[compare] files compared = {compared} | structural differences = {structural} | "
        "drift detected = {drift} | manual review = {manual} | missing files = {missing}"
    ).format(
        compared=len(summary_df),
        structural=structural,
        drift=drift,
        manual=manual,
        missing=missing,
    )


def _sanitize_path_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", path.name).strip("_") or "run"


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        flat_cols = []
        for column_tuple in out.columns:
            parts = [str(part).strip() for part in column_tuple if str(part).strip()]
            flat_cols.append(" | ".join(parts))
        out.columns = flat_cols
    else:
        out.columns = [str(column).strip() for column in out.columns]
    return out


def _read_csv(path: Path, *, logical_name: str | None = None) -> tuple[pd.DataFrame, str]:
    multiindex_key = logical_name if logical_name in KNOWN_MULTIINDEX_FILES else path.name
    if multiindex_key in KNOWN_MULTIINDEX_FILES:
        df = pd.read_csv(path, header=KNOWN_MULTIINDEX_FILES[multiindex_key], low_memory=False)
        if isinstance(df.columns, pd.MultiIndex):
            keep_cols = [
                column
                for column in df.columns
                if not any(str(level).startswith("Unnamed:") for level in column)
            ]
            df = df.loc[:, keep_cols]
        return _flatten_columns(df), "multiindex"

    df = pd.read_csv(
        path,
        usecols=lambda column: column not in ["Unnamed: 0"],
        low_memory=False,
    )
    return _flatten_columns(df), "flat"


def _normalize_key_series(series: pd.Series) -> pd.Series:
    normalized = series.copy()
    if normalized.dtype.kind in {"b", "i", "u", "f"}:
        numeric = pd.to_numeric(normalized, errors="coerce")
        return numeric.map(lambda value: "<NA>" if pd.isna(value) else repr(float(value)))
    return normalized.astype("string").fillna("<NA>").str.strip()


def _is_unique_on(df: pd.DataFrame, columns: list[str]) -> bool:
    if not columns:
        return False
    normalized = pd.DataFrame({column: _normalize_key_series(df[column]) for column in columns})
    return not normalized.duplicated(keep=False).any()


def _choose_join_columns(left_df: pd.DataFrame, right_df: pd.DataFrame) -> list[str]:
    common_identifier_columns = [
        column
        for column in IDENTIFIER_PRIORITY
        if column in left_df.columns and column in right_df.columns
    ]
    chosen: list[str] = []
    for column in common_identifier_columns:
        chosen.append(column)
        if _is_unique_on(left_df, chosen) and _is_unique_on(right_df, chosen):
            return chosen
    return []


def _coerce_numeric_pair(
    left_series: pd.Series,
    right_series: pd.Series,
) -> tuple[pd.Series, pd.Series] | None:
    left_numeric = pd.to_numeric(left_series, errors="coerce").astype(float)
    right_numeric = pd.to_numeric(right_series, errors="coerce").astype(float)
    if left_numeric.notna().sum() == 0 or right_numeric.notna().sum() == 0:
        return None
    return left_numeric, right_numeric


def _normalize_text_pair(left_series: pd.Series, right_series: pd.Series) -> tuple[pd.Series, pd.Series]:
    left_text = left_series.astype("string").fillna("<NA>").str.strip()
    right_text = right_series.astype("string").fillna("<NA>").str.strip()
    return left_text, right_text


def _threshold_for_difference(
    left_values: pd.Series,
    right_values: pd.Series,
    abs_tol: float,
    rel_tol: float,
) -> pd.Series:
    scale = np.maximum(left_values.abs().to_numpy(dtype=float), right_values.abs().to_numpy(dtype=float))
    return pd.Series(np.maximum(abs_tol, rel_tol * scale), index=left_values.index)


def _prepare_row_order_frames(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    common_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, str, list[str]]:
    sort_columns = [column for column in IDENTIFIER_PRIORITY if column in common_columns]
    if not sort_columns:
        sort_columns = sorted(common_columns)

    left_sorted = left_df[common_columns].copy().sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    right_sorted = right_df[common_columns].copy().sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    return left_sorted, right_sorted, "row_order_after_sort", sort_columns


def _compare_common_file(
    relative_path: str,
    baseline_path: Path,
    candidate_path: Path,
    *,
    abs_tol: float,
    rel_tol: float,
) -> tuple[FileComparisonResult, list[dict[str, object]]]:
    baseline_df, baseline_header_kind = _read_csv(baseline_path, logical_name=relative_path)
    candidate_df, candidate_header_kind = _read_csv(candidate_path, logical_name=relative_path)
    header_kind = baseline_header_kind if baseline_header_kind == candidate_header_kind else "mixed"

    baseline_columns = list(baseline_df.columns)
    candidate_columns = list(candidate_df.columns)
    common_columns = [column for column in baseline_columns if column in candidate_df.columns]
    baseline_only_columns = [column for column in baseline_columns if column not in candidate_df.columns]
    candidate_only_columns = [column for column in candidate_columns if column not in baseline_df.columns]

    join_columns = _choose_join_columns(baseline_df, candidate_df)
    note_parts: list[str] = []
    column_rows: list[dict[str, object]] = []

    matched_rows = 0
    baseline_only_rows = 0
    candidate_only_rows = 0
    numeric_columns_compared = 0
    numeric_cells_compared = 0
    numeric_cells_outside_tolerance = 0
    text_columns_compared = 0
    text_cells_compared = 0
    text_cells_mismatched = 0
    all_abs_diffs: list[np.ndarray] = []
    all_rel_diffs: list[np.ndarray] = []

    value_columns = [column for column in common_columns if column not in join_columns]

    if join_columns:
        left_keyed = baseline_df.set_index(join_columns).sort_index()
        right_keyed = candidate_df.set_index(join_columns).sort_index()

        common_index = left_keyed.index.intersection(right_keyed.index)
        left_only_index = left_keyed.index.difference(right_keyed.index)
        right_only_index = right_keyed.index.difference(left_keyed.index)

        matched_rows = len(common_index)
        baseline_only_rows = len(left_only_index)
        candidate_only_rows = len(right_only_index)
        left_comp = left_keyed.loc[common_index, value_columns]
        right_comp = right_keyed.loc[common_index, value_columns]
        join_strategy = "unique_keys"
    elif baseline_df.shape[0] == candidate_df.shape[0]:
        left_comp, right_comp, join_strategy, join_columns = _prepare_row_order_frames(
            baseline_df,
            candidate_df,
            common_columns,
        )
        matched_rows = left_comp.shape[0]
    else:
        join_strategy = "unkeyed"
        note_parts.append("No unique join columns found and row counts differ; cell-wise comparison skipped.")
        result = FileComparisonResult(
            relative_path=relative_path,
            header_kind=header_kind,
            comparison_status="requires_manual_review",
            join_strategy=join_strategy,
            join_columns="",
            baseline_rows=baseline_df.shape[0],
            candidate_rows=candidate_df.shape[0],
            baseline_columns=len(baseline_columns),
            candidate_columns=len(candidate_columns),
            common_columns=len(common_columns),
            baseline_only_columns=len(baseline_only_columns),
            candidate_only_columns=len(candidate_only_columns),
            matched_rows=0,
            baseline_only_rows=baseline_df.shape[0],
            candidate_only_rows=candidate_df.shape[0],
            numeric_columns_compared=0,
            numeric_cells_compared=0,
            numeric_cells_outside_tolerance=0,
            text_columns_compared=0,
            text_cells_compared=0,
            text_cells_mismatched=0,
            mean_abs_diff=float("nan"),
            max_abs_diff=float("nan"),
            max_rel_diff=float("nan"),
            note=" | ".join(note_parts),
        )
        return result, column_rows

    for column in value_columns:
        numeric_pair = _coerce_numeric_pair(left_comp[column], right_comp[column])
        if numeric_pair is not None:
            left_numeric, right_numeric = numeric_pair
            valid_mask = left_numeric.notna() & right_numeric.notna()
            if not valid_mask.any():
                continue

            left_valid = left_numeric.loc[valid_mask]
            right_valid = right_numeric.loc[valid_mask]
            abs_diff = (left_valid - right_valid).abs()
            diff_threshold = _threshold_for_difference(left_valid, right_valid, abs_tol, rel_tol)
            rel_diff = abs_diff / np.maximum(np.maximum(left_valid.abs(), right_valid.abs()), abs_tol)
            outside_tolerance = abs_diff > diff_threshold

            numeric_columns_compared += 1
            numeric_cells_compared += int(valid_mask.sum())
            numeric_cells_outside_tolerance += int(outside_tolerance.sum())
            all_abs_diffs.append(abs_diff.to_numpy(dtype=float))
            all_rel_diffs.append(rel_diff.to_numpy(dtype=float))
            column_rows.append(
                {
                    "relative_path": relative_path,
                    "column_name": column,
                    "column_type": "numeric",
                    "cells_compared": int(valid_mask.sum()),
                    "cells_outside_tolerance": int(outside_tolerance.sum()),
                    "mean_abs_diff": float(abs_diff.mean()),
                    "max_abs_diff": float(abs_diff.max()),
                    "mean_rel_diff": float(rel_diff.mean()),
                    "max_rel_diff": float(rel_diff.max()),
                }
            )
            continue

        left_text, right_text = _normalize_text_pair(left_comp[column], right_comp[column])
        valid_mask = ~(left_text.eq("<NA>") & right_text.eq("<NA>"))
        if not valid_mask.any():
            continue

        mismatches = left_text.loc[valid_mask] != right_text.loc[valid_mask]
        text_columns_compared += 1
        text_cells_compared += int(valid_mask.sum())
        text_cells_mismatched += int(mismatches.sum())
        column_rows.append(
            {
                "relative_path": relative_path,
                "column_name": column,
                "column_type": "text",
                "cells_compared": int(valid_mask.sum()),
                "cells_outside_tolerance": int(mismatches.sum()),
                "mean_abs_diff": float("nan"),
                "max_abs_diff": float("nan"),
                "mean_rel_diff": float("nan"),
                "max_rel_diff": float("nan"),
            }
        )

    if baseline_only_columns:
        note_parts.append(f"Baseline-only columns: {baseline_only_columns}")
    if candidate_only_columns:
        note_parts.append(f"Candidate-only columns: {candidate_only_columns}")
    if baseline_only_rows or candidate_only_rows:
        note_parts.append(
            f"Unmatched rows after alignment: baseline_only={baseline_only_rows}, candidate_only={candidate_only_rows}"
        )

    comparison_status = "ok"
    if baseline_only_columns or candidate_only_columns or baseline_only_rows or candidate_only_rows:
        comparison_status = "structural_difference"
    elif numeric_cells_outside_tolerance > 0 or text_cells_mismatched > 0:
        comparison_status = "drift_detected"

    if all_abs_diffs:
        abs_concat = np.concatenate(all_abs_diffs)
        rel_concat = np.concatenate(all_rel_diffs)
        mean_abs_diff = float(abs_concat.mean())
        max_abs_diff = float(abs_concat.max())
        max_rel_diff = float(rel_concat.max())
    else:
        mean_abs_diff = float("nan")
        max_abs_diff = float("nan")
        max_rel_diff = float("nan")

    result = FileComparisonResult(
        relative_path=relative_path,
        header_kind=header_kind,
        comparison_status=comparison_status,
        join_strategy=join_strategy,
        join_columns=" | ".join(join_columns),
        baseline_rows=baseline_df.shape[0],
        candidate_rows=candidate_df.shape[0],
        baseline_columns=len(baseline_columns),
        candidate_columns=len(candidate_columns),
        common_columns=len(common_columns),
        baseline_only_columns=len(baseline_only_columns),
        candidate_only_columns=len(candidate_only_columns),
        matched_rows=matched_rows,
        baseline_only_rows=baseline_only_rows,
        candidate_only_rows=candidate_only_rows,
        numeric_columns_compared=numeric_columns_compared,
        numeric_cells_compared=numeric_cells_compared,
        numeric_cells_outside_tolerance=numeric_cells_outside_tolerance,
        text_columns_compared=text_columns_compared,
        text_cells_compared=text_cells_compared,
        text_cells_mismatched=text_cells_mismatched,
        mean_abs_diff=mean_abs_diff,
        max_abs_diff=max_abs_diff,
        max_rel_diff=max_rel_diff,
        note=" | ".join(note_parts),
    )
    return result, column_rows


def _build_inventory_rows(
    baseline_csvs: dict[str, Path],
    candidate_csvs: dict[str, Path],
) -> list[dict[str, object]]:
    rows = []
    for relative_path in sorted(set(baseline_csvs) | set(candidate_csvs)):
        rows.append(
            {
                "relative_path": relative_path,
                "present_in_baseline": relative_path in baseline_csvs,
                "present_in_candidate": relative_path in candidate_csvs,
                "baseline_path": str(baseline_csvs.get(relative_path, "")),
                "candidate_path": str(candidate_csvs.get(relative_path, "")),
            }
        )
    return rows


def _build_summary_dataframe(summary_rows: list[dict[str, object]]) -> pd.DataFrame:
    summary_columns = [field.name for field in fields(FileComparisonResult)]
    if not summary_rows:
        return pd.DataFrame(columns=summary_columns)
    return pd.DataFrame(summary_rows, columns=summary_columns).sort_values("relative_path").reset_index(drop=True)


def _build_column_dataframe(column_rows: list[dict[str, object]]) -> pd.DataFrame:
    if not column_rows:
        return pd.DataFrame(columns=COLUMN_SUMMARY_COLUMNS)
    return (
        pd.DataFrame(column_rows, columns=COLUMN_SUMMARY_COLUMNS)
        .sort_values(["relative_path", "column_type", "column_name"])
        .reset_index(drop=True)
    )


def _write_report(
    output_dir: Path,
    *,
    baseline_dir: Path,
    candidate_dir: Path,
    inventory_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    column_df: pd.DataFrame,
) -> None:
    missing_baseline = inventory_df.loc[~inventory_df["present_in_baseline"], "relative_path"].tolist()
    missing_candidate = inventory_df.loc[~inventory_df["present_in_candidate"], "relative_path"].tolist()
    structural = summary_df[summary_df["comparison_status"] == "structural_difference"]
    drift = summary_df[summary_df["comparison_status"] == "drift_detected"]
    manual = summary_df[summary_df["comparison_status"] == "requires_manual_review"]
    top_numeric = column_df[column_df["column_type"] == "numeric"].sort_values(
        ["cells_outside_tolerance", "max_abs_diff"],
        ascending=[False, False],
    ).head(10)

    lines = [
        f"Baseline: {baseline_dir}",
        f"Candidate: {candidate_dir}",
        f"Inventory rows: {len(inventory_df)}",
        f"Compared files: {len(summary_df)}",
        f"Structural differences: {len(structural)}",
        f"Drift-detected files: {len(drift)}",
        f"Manual-review files: {len(manual)}",
        "",
    ]
    if missing_baseline:
        lines.append("Files missing from baseline:")
        lines.extend(f"- {relative_path}" for relative_path in missing_baseline)
        lines.append("")
    if missing_candidate:
        lines.append("Files missing from candidate:")
        lines.extend(f"- {relative_path}" for relative_path in missing_candidate)
        lines.append("")
    if not top_numeric.empty:
        lines.append("Top numeric drift columns:")
        for row in top_numeric.itertuples(index=False):
            lines.append(
                "- {path} :: {column} | outside_tol={count} | max_abs_diff={max_abs:.6g}".format(
                    path=row.relative_path,
                    column=row.column_name,
                    count=row.cells_outside_tolerance,
                    max_abs=row.max_abs_diff,
                )
            )
        lines.append("")

    (output_dir / "comparison_report.txt").write_text("\n".join(lines), encoding="utf-8")