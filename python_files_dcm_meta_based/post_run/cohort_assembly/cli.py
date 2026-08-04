"""CLI entrypoint for post-run cohort assembly."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .config import DEFAULT_COHORT_ASSEMBLY_CONFIG_PATH
from .config import load_cohort_assembly_job_configs
from .service import format_post_run_cohort_assembly_summary
from .service import run_post_run_cohort_assembly


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run post-run cohort assembly jobs from a JSON config file."
    )
    parser.add_argument(
        "config_path",
        nargs="?",
        type=Path,
        default=DEFAULT_COHORT_ASSEMBLY_CONFIG_PATH,
        help="JSON config path. Defaults to the bundled post-run cohort assembly config.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    job_configs = load_cohort_assembly_job_configs(args.config_path)
    if not job_configs:
        print(f"[post-run-cohort-assembly] no enabled jobs in {args.config_path}")
        return 0
    for job_config in job_configs:
        result = run_post_run_cohort_assembly(job_config)
        print(f"[post-run-cohort-assembly] {format_post_run_cohort_assembly_summary(result)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())