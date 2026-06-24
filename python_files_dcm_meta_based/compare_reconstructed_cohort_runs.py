from __future__ import annotations

import argparse
from pathlib import Path

from validation.reconstructed_cohort_comparator import ReconstructedCohortComparisonConfig
from validation.reconstructed_cohort_comparator import format_reconstructed_cohort_comparison_summary
from validation.reconstructed_cohort_comparator import run_reconstructed_cohort_comparison


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a full patient-runner run with one-or-more split patient-runner runs after "
            "decomposing patient artifacts and reconstructing both cohort surfaces."
        )
    )
    parser.add_argument(
        "reference_patient_runner_output_dir",
        type=Path,
        help="Patient-runner output directory, batch manifest, or parent run directory for the full/reference run.",
    )
    parser.add_argument(
        "split_patient_runner_output_dirs",
        type=Path,
        nargs="+",
        help="One or more patient-runner output directories, batch manifests, or parent run directories for split runs.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for reconstructed comparison outputs.")
    parser.add_argument(
        "--final-table-name",
        action="append",
        default=[],
        help="Restrict reconstruction/comparison to a final cohort table name. Can be repeated.",
    )
    parser.add_argument("--abs-tol", type=float, default=1e-8)
    parser.add_argument("--rel-tol", type=float, default=1e-6)
    parser.add_argument(
        "--allow-patient-set-mismatch",
        action="store_true",
        help="Do not raise when reference and split patient UID sets differ; summary will still report the mismatch.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = run_reconstructed_cohort_comparison(
        ReconstructedCohortComparisonConfig(
            reference_patient_runner_output_dir=args.reference_patient_runner_output_dir,
            split_patient_runner_output_dirs=tuple(args.split_patient_runner_output_dirs),
            output_dir=args.output_dir,
            final_table_names=tuple(args.final_table_name),
            abs_tol=args.abs_tol,
            rel_tol=args.rel_tol,
            require_patient_uid_match=not args.allow_patient_set_mismatch,
        )
    )
    print(f"[reconstructed-cohort] wrote outputs to {result.output_dir}")
    print(format_reconstructed_cohort_comparison_summary(result))
    return 0 if result.summary.get("overall_status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())