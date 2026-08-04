from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from patient_runner.process_runner import run_worker_job_file


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one standalone patient-scientific worker job packet."
    )
    parser.add_argument(
        "job_path",
        type=Path,
        help="Path to a patient worker job JSON packet.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate job loading/result writing without building runtime state or running scientific stages.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_worker_job_file(args.job_path, dry_run=args.dry_run)
    print(
        "[patient-worker] patient_uid={} status={} exit_code={} dry_run={}".format(
            result.worker_job.patient_case.patient_uid,
            result.status.value,
            result.exit_code,
            result.dry_run,
        )
    )
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())