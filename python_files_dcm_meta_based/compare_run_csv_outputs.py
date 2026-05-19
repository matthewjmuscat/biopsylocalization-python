from __future__ import annotations

import argparse
from pathlib import Path

from validation import (
    compare_run_csv_output_dirs,
    default_run_csv_comparison_output_dir,
    format_console_summary,
    resolve_existing_output_dir,
    write_comparison_outputs,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two main-algorithm output directories across every exported CSV and "
            "summarize schema mismatches versus numeric drift."
        )
    )
    parser.add_argument("baseline", type=Path, help="Baseline main output directory or unique prefix.")
    parser.add_argument("candidate", type=Path, help="Candidate main output directory or unique prefix.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for comparison outputs. Defaults to "
            "validation_outputs/run_comparisons/<baseline>__vs__<candidate>__all_csvs."
        ),
    )
    parser.add_argument(
        "--abs-tol",
        type=float,
        default=1e-8,
        help="Absolute tolerance for numeric cell comparisons.",
    )
    parser.add_argument(
        "--rel-tol",
        type=float,
        default=1e-6,
        help="Relative tolerance for numeric cell comparisons.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    baseline_dir = resolve_existing_output_dir(args.baseline)
    candidate_dir = resolve_existing_output_dir(args.candidate)
    output_dir = args.output_dir or default_run_csv_comparison_output_dir(
        baseline_dir,
        candidate_dir,
    )

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

    print(f"[compare] wrote outputs to {output_dir}")
    print(format_console_summary(summary_df, inventory_df))


if __name__ == "__main__":
    main()
