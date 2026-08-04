from __future__ import annotations

import argparse
from pathlib import Path

from post_run.cohort_assembly.config import PostRunCohortAssemblyJobConfig
from post_run.cohort_assembly.service import format_post_run_cohort_assembly_summary
from post_run.cohort_assembly.service import run_post_run_cohort_assembly


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble cohort CSV tables from a completed patient-runner output directory."
    )
    parser.add_argument("patient_runner_output_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--patient-uid", action="append", default=[])
    parser.add_argument("--final-table-name", action="append", default=[])
    parser.add_argument("--source-table-name", action="append", default=[])
    parser.add_argument("--no-write-outputs", action="store_true")
    parser.add_argument("--no-write-assembled-tables", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = run_post_run_cohort_assembly(
        PostRunCohortAssemblyJobConfig(
            name="post_run_cohort_assembly",
            patient_runner_output_dir=args.patient_runner_output_dir,
            output_dir=args.output_dir,
            patient_uids=tuple(args.patient_uid),
            final_table_names=tuple(args.final_table_name),
            source_table_names=tuple(args.source_table_name),
            write_outputs=not args.no_write_outputs,
            write_assembled_tables=not args.no_write_assembled_tables,
            metadata={"source": "assemble_patient_runner_cohort.py"},
        )
    )
    print(f"[post-run-cohort-assembly] {format_post_run_cohort_assembly_summary(result)}")
    return 0 if result.summary.get("overall_status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())