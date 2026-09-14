"""Reference worker for anatomical and biopsy-preprocessing migration validation.

Retains the historical CLI name for existing callers. Each invocation builds
independent legacy singleton inputs; the ordinary worker owns preflight, current
scientific stages and result writing. Unsupported boundaries fail before dispatch.
"""

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
    from validation.preprocessing_boundary import preprocessing_boundary

    job = load_patient_worker_job(args.job_path)
    boundary = preprocessing_boundary(job.pathway_name)
    if job.checkpoint_name != boundary.name:
        raise ValueError("reference worker requires matching pathway/checkpoint: " + boundary.name)
    result = run_patient_worker_job(job, dry_run=args.dry_run, runtime_builder=build_legacy_input_anatomical_runtime)
    write_patient_worker_result(result)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
