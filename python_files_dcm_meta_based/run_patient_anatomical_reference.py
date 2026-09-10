"""Validation-only worker using independent legacy singleton input construction."""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_path", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    from patient_runner.process_runner import load_patient_worker_job, run_patient_worker_job, write_patient_worker_result
    from validation.anatomical_execution import build_legacy_input_anatomical_runtime

    job = load_patient_worker_job(args.job_path)
    if job.pathway_name != "anatomical_qa":
        raise ValueError("reference worker supports anatomical_qa only")
    result = run_patient_worker_job(job, dry_run=args.dry_run, runtime_builder=build_legacy_input_anatomical_runtime)
    write_patient_worker_result(result)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())