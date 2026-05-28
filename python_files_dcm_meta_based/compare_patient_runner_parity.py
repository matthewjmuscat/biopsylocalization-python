from __future__ import annotations

import argparse
from pathlib import Path

from patient_runner import PatientRunnerParitySurface
from patient_runner import PatientRunnerPostRunParityConfig
from patient_runner import format_patient_runner_post_run_parity_summary
from patient_runner import run_patient_runner_post_run_parity


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a completed legacy output run with completed patient-runner outputs. "
            "By default this compares legacy final cohort CSVs with patient-runner assembled cohort tables."
        )
    )
    parser.add_argument("legacy_output_dir", type=Path, help="Completed legacy output directory or unique prefix.")
    parser.add_argument("patient_runner_output_dir", type=Path, help="Completed patient-runner output directory or unique prefix.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for parity reports.")
    parser.add_argument(
        "--assembled-table-dir",
        type=Path,
        default=None,
        help="Override the patient-runner assembled table directory.",
    )
    parser.add_argument(
        "--final-table-name",
        action="append",
        default=[],
        help="Restrict assembled-cohort comparison to a final cohort table name. Can be repeated.",
    )
    parser.add_argument("--abs-tol", type=float, default=1e-8)
    parser.add_argument("--rel-tol", type=float, default=1e-6)
    parser.add_argument(
        "--recursive-csv",
        action="store_true",
        help="Also compare all recursive CSV paths. Useful only when both roots are path-compatible.",
    )
    parser.add_argument(
        "--skip-assembled-cohort",
        action="store_true",
        help="Skip the default assembled-cohort comparison.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    surfaces: list[PatientRunnerParitySurface] = []
    if not args.skip_assembled_cohort:
        surfaces.append(PatientRunnerParitySurface.ASSEMBLED_COHORT)
    if args.recursive_csv:
        surfaces.append(PatientRunnerParitySurface.RECURSIVE_CSV)
    if not surfaces:
        raise ValueError("At least one parity surface must be selected")

    result = run_patient_runner_post_run_parity(
        PatientRunnerPostRunParityConfig(
            legacy_output_dir=args.legacy_output_dir,
            patient_runner_output_dir=args.patient_runner_output_dir,
            output_dir=args.output_dir,
            assembled_table_dir=args.assembled_table_dir,
            surfaces=tuple(surfaces),
            final_table_names=tuple(args.final_table_name),
            abs_tol=args.abs_tol,
            rel_tol=args.rel_tol,
        )
    )
    print(f"[patient-runner-parity] wrote outputs to {result.output_dir}")
    print(format_patient_runner_post_run_parity_summary(result))


if __name__ == "__main__":
    main()