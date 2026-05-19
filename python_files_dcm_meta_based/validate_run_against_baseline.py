from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable

import pandas as pd

from validation import (
    compare_run_csv_output_dirs,
    default_run_csv_comparison_output_dir,
    format_console_summary,
    resolve_existing_output_dir,
    write_comparison_outputs,
)


DEFAULT_ABS_TOL = 1e-8
DEFAULT_REL_TOL = 1e-6
EXPECTED_PHASE1_MANIFEST_FILES = [
    "input_manifest_summary.json",
    "input_case_manifest.csv",
    "input_dicom_manifest.csv",
    "input_routing_profile.json",
    "input_manifest_warnings.jsonl",
]
LOG_ALERT_PATTERN = re.compile(r"Traceback|\bERROR\b|\bWARN\b|exception|error", re.IGNORECASE)
DEFAULT_DIAGNOSTIC_COLUMN_PATTERNS = [
    r"elapsed\s*seconds",
    r"elapsed_sec",
    r"wall\s*time",
    r"timing",
    r"duration",
    r"fraction",
    r"rss_mb",
    r"vms_mb",
    r"gpu_",
    r"memory",
    r"dominant interpolation subphase",
    r"dominant primary",
    r"max test structures per call budget",
    r"chunk count",
    r"local chunk index",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a candidate pipeline run against a baseline run using reusable checks: "
            "recursive CSV comparison, Phase 1 input-manifest inspection, and runtime-log scanning."
        )
    )
    parser.add_argument("baseline", type=Path, help="Baseline run output directory or unique prefix.")
    parser.add_argument("candidate", type=Path, help="Candidate run output directory or unique prefix.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for validation outputs. Defaults under validation_outputs/run_comparisons.",
    )
    parser.add_argument("--abs-tol", type=float, default=DEFAULT_ABS_TOL)
    parser.add_argument("--rel-tol", type=float, default=DEFAULT_REL_TOL)
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top drift/error rows to include in the Markdown report.",
    )
    parser.add_argument(
        "--diagnostic-column-regex",
        action="append",
        default=[],
        help=(
            "Additional case-insensitive regex for columns treated as diagnostic/performance drift. "
            "Can be provided multiple times."
        ),
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"read_error": repr(exc)}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                row = {
                    "line_number": line_number,
                    "level": "PARSE_ERROR",
                    "event_type": "jsonl_parse_error",
                    "message": repr(exc),
                    "raw_line": stripped[:1000],
                }
            rows.append(row)
    return rows


def _first_items(items: Iterable[Any], count: int) -> list[Any]:
    out = []
    for item in items:
        out.append(item)
        if len(out) >= count:
            break
    return out


def inspect_phase1_manifests(run_dir: Path) -> dict[str, Any]:
    manifest_dir = run_dir / "manifests"
    manifest_paths = {
        file_name: manifest_dir / file_name
        for file_name in EXPECTED_PHASE1_MANIFEST_FILES
    }
    present_files = [file_name for file_name, path in manifest_paths.items() if path.is_file()]
    missing_files = [file_name for file_name, path in manifest_paths.items() if not path.is_file()]

    summary = _read_json(manifest_paths["input_manifest_summary.json"])
    routing_profile = _read_json(manifest_paths["input_routing_profile.json"])
    warnings = _read_jsonl(manifest_paths["input_manifest_warnings.jsonl"])
    warning_type_counts = Counter(str(row.get("warning_type", "unknown")) for row in warnings)

    case_manifest_rows = None
    dicom_manifest_rows = None
    if manifest_paths["input_case_manifest.csv"].is_file():
        case_manifest_rows = int(pd.read_csv(manifest_paths["input_case_manifest.csv"]).shape[0])
    if manifest_paths["input_dicom_manifest.csv"].is_file():
        dicom_manifest_rows = int(pd.read_csv(manifest_paths["input_dicom_manifest.csv"]).shape[0])

    return {
        "manifest_dir": str(manifest_dir),
        "present_files": present_files,
        "missing_files": missing_files,
        "summary": summary,
        "routing_profile": {
            "schema_version": routing_profile.get("schema_version"),
            "profile_id": routing_profile.get("profile_id"),
            "display_name": routing_profile.get("display_name"),
            "num_rules": len(routing_profile.get("rules", [])) if isinstance(routing_profile.get("rules"), list) else None,
            "read_error": routing_profile.get("read_error"),
        },
        "case_manifest_rows": case_manifest_rows,
        "dicom_manifest_rows": dicom_manifest_rows,
        "warning_count": len(warnings),
        "warning_type_counts": dict(sorted(warning_type_counts.items())),
        "first_warnings": _first_items(warnings, 10),
    }


def scan_run_logs(run_dir: Path) -> dict[str, Any]:
    logs_dir = run_dir / "logs"
    run_status = _read_json(logs_dir / "run_status.json")
    events = _read_jsonl(logs_dir / "events.jsonl")

    notable_events = []
    event_level_counts = Counter()
    event_type_counts = Counter()
    phase_counts = Counter()
    for event in events:
        level = str(event.get("level", ""))
        event_type = str(event.get("event_type", ""))
        phase = str(event.get("phase", ""))
        event_level_counts[level or "unknown"] += 1
        event_type_counts[event_type or "unknown"] += 1
        if phase:
            phase_counts[phase] += 1
        message = str(event.get("message", ""))
        details = event.get("details", {})
        details_text = json.dumps(details, default=str) if isinstance(details, dict) else str(details)
        if level.upper() in {"WARN", "WARNING", "ERROR", "PARSE_ERROR"} or event_type in {"warning", "error", "exception"} or LOG_ALERT_PATTERN.search(message) or LOG_ALERT_PATTERN.search(details_text):
            notable_events.append({
                "timestamp_utc": event.get("timestamp_utc"),
                "level": level,
                "event_type": event_type,
                "phase": phase,
                "patient_uid": event.get("patient_uid"),
                "structure_id": event.get("structure_id"),
                "message": message,
                "details": details,
            })

    run_log_matches = []
    run_log_path = logs_dir / "run.log"
    if run_log_path.is_file():
        with run_log_path.open("r", encoding="utf-8", errors="replace") as file_obj:
            for line_number, line in enumerate(file_obj, start=1):
                if LOG_ALERT_PATTERN.search(line):
                    run_log_matches.append({
                        "line_number": line_number,
                        "text": line.rstrip("\n")[:2000],
                    })

    traceback_matches = [
        match for match in run_log_matches
        if "traceback" in match["text"].lower()
    ]
    exception_events = [
        event for event in notable_events
        if str(event.get("event_type", "")).lower() == "exception" or str(event.get("level", "")).upper() == "ERROR"
    ]

    return {
        "logs_dir": str(logs_dir),
        "run_status": run_status,
        "event_count": len(events),
        "event_level_counts": dict(sorted(event_level_counts.items())),
        "event_type_counts": dict(sorted(event_type_counts.items())),
        "top_phase_counts": dict(phase_counts.most_common(15)),
        "notable_event_count": len(notable_events),
        "notable_events": notable_events[:50],
        "run_log_alert_count": len(run_log_matches),
        "run_log_alerts": run_log_matches[:50],
        "traceback_match_count": len(traceback_matches),
        "traceback_matches": traceback_matches[:20],
        "exception_event_count": len(exception_events),
        "exception_events": exception_events[:20],
    }


def summarize_csv_comparison(inventory_df: pd.DataFrame, summary_df: pd.DataFrame, column_df: pd.DataFrame, top_n: int) -> dict[str, Any]:
    missing_from_baseline = inventory_df.loc[~inventory_df["present_in_baseline"], "relative_path"].tolist()
    missing_from_candidate = inventory_df.loc[~inventory_df["present_in_candidate"], "relative_path"].tolist()
    status_counts = summary_df["comparison_status"].value_counts().to_dict() if not summary_df.empty else {}
    top_numeric = pd.DataFrame()
    if not column_df.empty:
        top_numeric = (
            column_df[column_df["column_type"] == "numeric"]
            .sort_values(["cells_outside_tolerance", "max_abs_diff"], ascending=[False, False])
            .head(top_n)
        )
    top_text = pd.DataFrame()
    if not column_df.empty:
        top_text = (
            column_df[column_df["column_type"] == "text"]
            .sort_values(["cells_outside_tolerance"], ascending=[False])
            .head(top_n)
        )
    return {
        "inventory_rows": int(len(inventory_df)),
        "compared_files": int(len(summary_df)),
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "missing_from_baseline": missing_from_baseline,
        "missing_from_candidate": missing_from_candidate,
        "missing_file_count": int(len(missing_from_baseline) + len(missing_from_candidate)),
        "numeric_cells_compared": int(summary_df["numeric_cells_compared"].sum()) if "numeric_cells_compared" in summary_df else 0,
        "numeric_cells_outside_tolerance": int(summary_df["numeric_cells_outside_tolerance"].sum()) if "numeric_cells_outside_tolerance" in summary_df else 0,
        "text_cells_compared": int(summary_df["text_cells_compared"].sum()) if "text_cells_compared" in summary_df else 0,
        "text_cells_mismatched": int(summary_df["text_cells_mismatched"].sum()) if "text_cells_mismatched" in summary_df else 0,
        "top_numeric_drift": top_numeric.to_dict(orient="records"),
        "top_text_drift": top_text.to_dict(orient="records"),
    }


def _compile_diagnostic_column_pattern(extra_patterns: list[str]) -> re.Pattern[str]:
    patterns = DEFAULT_DIAGNOSTIC_COLUMN_PATTERNS + list(extra_patterns)
    return re.compile("|".join(f"(?:{pattern})" for pattern in patterns), re.IGNORECASE)


def classify_drift_columns(column_df: pd.DataFrame, diagnostic_pattern: re.Pattern[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = [
        "relative_path",
        "column_name",
        "column_type",
        "cells_compared",
        "cells_outside_tolerance",
        "mean_abs_diff",
        "max_abs_diff",
        "mean_rel_diff",
        "max_rel_diff",
        "drift_class",
    ]
    if column_df.empty:
        return pd.DataFrame(columns=columns), {
            "drift_columns": 0,
            "diagnostic_drift_columns": 0,
            "non_diagnostic_drift_columns": 0,
            "files_with_only_diagnostic_drift": 0,
            "files_with_non_diagnostic_drift": 0,
        }

    drift_df = column_df[column_df["cells_outside_tolerance"] > 0].copy()
    if drift_df.empty:
        return pd.DataFrame(columns=columns), {
            "drift_columns": 0,
            "diagnostic_drift_columns": 0,
            "non_diagnostic_drift_columns": 0,
            "files_with_only_diagnostic_drift": 0,
            "files_with_non_diagnostic_drift": 0,
        }

    drift_df["drift_class"] = drift_df["column_name"].map(
        lambda column_name: "diagnostic" if diagnostic_pattern.search(str(column_name)) else "non_diagnostic"
    )
    file_classes = drift_df.groupby("relative_path")["drift_class"].apply(set)
    files_with_non_diagnostic = [
        relative_path for relative_path, classes in file_classes.items()
        if "non_diagnostic" in classes
    ]
    files_with_only_diagnostic = [
        relative_path for relative_path, classes in file_classes.items()
        if classes == {"diagnostic"}
    ]
    summary = {
        "drift_columns": int(len(drift_df)),
        "diagnostic_drift_columns": int((drift_df["drift_class"] == "diagnostic").sum()),
        "non_diagnostic_drift_columns": int((drift_df["drift_class"] == "non_diagnostic").sum()),
        "files_with_only_diagnostic_drift": int(len(files_with_only_diagnostic)),
        "files_with_non_diagnostic_drift": int(len(files_with_non_diagnostic)),
        "non_diagnostic_drift_files": sorted(files_with_non_diagnostic),
    }
    return drift_df[columns].sort_values(["drift_class", "relative_path", "column_name"]).reset_index(drop=True), summary


def _write_json(path: Path, payload: MappingLike) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


MappingLike = dict[str, Any]


def write_validation_report(output_dir: Path, report: MappingLike, top_n: int) -> None:
    csv_summary = report["csv_comparison"]
    drift_classification = report["drift_classification"]
    manifest_summary = report["candidate_phase1_manifests"]
    log_summary = report["candidate_log_scan"]
    run_status = log_summary.get("run_status", {})

    lines = [
        "# Run Validation Report",
        "",
        f"Generated UTC: {report['generated_utc']}",
        f"Baseline: `{report['baseline_dir']}`",
        f"Candidate: `{report['candidate_dir']}`",
        "",
        "## Run Status",
        "",
        f"- Candidate status: `{run_status.get('status', 'unknown')}`",
        f"- Last completed checkpoint: `{run_status.get('last_completed_checkpoint', 'unknown')}`",
        f"- Last update UTC: `{run_status.get('last_update_utc', 'unknown')}`",
        f"- Exception events in structured log: `{log_summary.get('exception_event_count', 0)}`",
        f"- Traceback matches in text log: `{log_summary.get('traceback_match_count', 0)}`",
        f"- Text-log alert matches: `{log_summary.get('run_log_alert_count', 0)}`",
        f"- Notable structured events: `{log_summary.get('notable_event_count', 0)}`",
        "",
        "## Phase 1 Manifest Check",
        "",
        f"- Present files: `{len(manifest_summary['present_files'])}` / `{len(EXPECTED_PHASE1_MANIFEST_FILES)}`",
        f"- Missing files: `{manifest_summary['missing_files']}`",
        f"- Routing profile: `{manifest_summary['routing_profile'].get('profile_id')}`",
        f"- Case manifest rows: `{manifest_summary.get('case_manifest_rows')}`",
        f"- DICOM manifest rows: `{manifest_summary.get('dicom_manifest_rows')}`",
        f"- Manifest warning count: `{manifest_summary.get('warning_count')}`",
        f"- Manifest warning types: `{manifest_summary.get('warning_type_counts')}`",
        "",
        "## CSV Comparison",
        "",
        f"- Inventory rows: `{csv_summary['inventory_rows']}`",
        f"- Compared files: `{csv_summary['compared_files']}`",
        f"- Status counts: `{csv_summary['status_counts']}`",
        f"- Missing file count: `{csv_summary['missing_file_count']}`",
        f"- Numeric cells outside tolerance: `{csv_summary['numeric_cells_outside_tolerance']}` / `{csv_summary['numeric_cells_compared']}`",
        f"- Text cells mismatched: `{csv_summary['text_cells_mismatched']}` / `{csv_summary['text_cells_compared']}`",
        f"- Drift columns classified diagnostic: `{drift_classification['diagnostic_drift_columns']}`",
        f"- Drift columns classified non-diagnostic: `{drift_classification['non_diagnostic_drift_columns']}`",
        f"- Files with only diagnostic drift: `{drift_classification['files_with_only_diagnostic_drift']}`",
        f"- Files with non-diagnostic drift: `{drift_classification['files_with_non_diagnostic_drift']}`",
        "",
    ]

    if csv_summary["missing_from_baseline"]:
        lines.append("### Candidate-Only CSVs")
        lines.append("")
        lines.extend(f"- `{path}`" for path in csv_summary["missing_from_baseline"][:top_n])
        lines.append("")
    if csv_summary["missing_from_candidate"]:
        lines.append("### Baseline-Only CSVs")
        lines.append("")
        lines.extend(f"- `{path}`" for path in csv_summary["missing_from_candidate"][:top_n])
        lines.append("")
    if csv_summary["top_numeric_drift"]:
        lines.append("### Top Numeric Drift")
        lines.append("")
    if drift_classification.get("non_diagnostic_drift_files"):
        lines.append("### Non-Diagnostic Drift Files")
        lines.append("")
        lines.extend(f"- `{path}`" for path in drift_classification["non_diagnostic_drift_files"][:top_n])
        lines.append("")
        for row in csv_summary["top_numeric_drift"][:top_n]:
            lines.append(
                "- `{relative_path}` :: `{column_name}` | outside_tol=`{cells_outside_tolerance}` | max_abs_diff=`{max_abs_diff}`".format(**row)
            )
        lines.append("")
    if csv_summary["top_text_drift"]:
        lines.append("### Top Text Drift")
        lines.append("")
        for row in csv_summary["top_text_drift"][:top_n]:
            lines.append(
                "- `{relative_path}` :: `{column_name}` | mismatches=`{cells_outside_tolerance}`".format(**row)
            )
        lines.append("")
    if log_summary.get("traceback_matches"):
        lines.append("### Traceback Matches")
        lines.append("")
        for row in log_summary["traceback_matches"][:top_n]:
            lines.append(f"- line `{row['line_number']}`: `{row['text']}`")
        lines.append("")
    if log_summary.get("exception_events"):
        lines.append("### Exception Events")
        lines.append("")
        for event in log_summary["exception_events"][:top_n]:
            lines.append(
                "- `{timestamp}` `{phase}` `{level}` `{event_type}`: {message}".format(
                    timestamp=event.get("timestamp_utc"),
                    phase=event.get("phase"),
                    level=event.get("level"),
                    event_type=event.get("event_type"),
                    message=event.get("message"),
                )
            )
        lines.append("")

    (output_dir / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    baseline_dir = resolve_existing_output_dir(args.baseline)
    candidate_dir = resolve_existing_output_dir(args.candidate)
    output_dir = args.output_dir or default_run_csv_comparison_output_dir(baseline_dir, candidate_dir)

    inventory_df, summary_df, column_df = compare_run_csv_output_dirs(
        baseline_dir,
        candidate_dir,
        abs_tol=args.abs_tol,
        rel_tol=args.rel_tol,
    )
    write_comparison_outputs(
        output_dir,
        baseline_dir=baseline_dir,
        candidate_dir=candidate_dir,
        inventory_df=inventory_df,
        summary_df=summary_df,
        column_df=column_df,
    )
    diagnostic_pattern = _compile_diagnostic_column_pattern(args.diagnostic_column_regex)
    drift_classification_df, drift_classification_summary = classify_drift_columns(column_df, diagnostic_pattern)
    drift_classification_df.to_csv(output_dir / "drift_classification.csv", index=False)

    report: MappingLike = {
        "generated_utc": _utc_now_iso(),
        "baseline_dir": str(baseline_dir),
        "candidate_dir": str(candidate_dir),
        "output_dir": str(output_dir),
        "csv_comparison": summarize_csv_comparison(inventory_df, summary_df, column_df, args.top_n),
        "drift_classification": drift_classification_summary,
        "candidate_phase1_manifests": inspect_phase1_manifests(candidate_dir),
        "candidate_log_scan": scan_run_logs(candidate_dir),
    }
    _write_json(output_dir / "validation_summary.json", report)
    write_validation_report(output_dir, report, args.top_n)

    print(f"[validate] wrote outputs to {output_dir}")
    print(format_console_summary(summary_df, inventory_df))
    print(
        "[validate] run_status={status} | tracebacks={tracebacks} | exception_events={exceptions} | manifest_warnings={manifest_warnings}".format(
            status=report["candidate_log_scan"].get("run_status", {}).get("status", "unknown"),
            tracebacks=report["candidate_log_scan"].get("traceback_match_count", 0),
            exceptions=report["candidate_log_scan"].get("exception_event_count", 0),
            manifest_warnings=report["candidate_phase1_manifests"].get("warning_count", 0),
        )
    )


if __name__ == "__main__":
    main()
