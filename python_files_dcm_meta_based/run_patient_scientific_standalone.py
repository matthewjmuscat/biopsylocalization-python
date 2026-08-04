from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from patient_runner.process_runner import PatientProcessFailurePolicy
from patient_runner.process_runner import build_patient_process_run_plan
from patient_runner.process_runner import run_patient_process_plan
from patient_runner.process_runner import write_patient_process_run_plan
from patient_runner.process_runner import write_patient_worker_job_packets


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan or launch the standalone patient-scientific process runner."
    )
    parser.add_argument(
        "--input-case-manifest",
        required=True,
        type=Path,
        help="Path to manifests/input_case_manifest.csv from a completed input discovery run.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Standalone patient-runner output root.",
    )
    parser.add_argument(
        "--pathway-name",
        default="full_current_pipeline_shadow",
        help="Patient scientific pathway name to record in worker jobs.",
    )
    parser.add_argument(
        "--checkpoint-name",
        default="full_current_pipeline_shadow",
        help="Patient scientific checkpoint name to record in worker jobs.",
    )
    parser.add_argument(
        "--run-id",
        default="patient-process-runner",
        help="Run ID recorded in the parent plan and worker jobs.",
    )
    parser.add_argument(
        "--patient-uid",
        action="append",
        default=[],
        help="Optional patient UID to include. Repeat for multiple patients. Defaults to all manifest patients.",
    )
    parser.add_argument(
        "--failure-policy",
        choices=[policy.value for policy in PatientProcessFailurePolicy],
        default=PatientProcessFailurePolicy.STOP_ON_FAILURE.value,
        help="Parent failure policy for worker subprocesses.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Recorded max worker count. The current executable backend is sequential.",
    )
    parser.add_argument(
        "--launch-dry-run-workers",
        action="store_true",
        help="Launch one subprocess per worker job in dry-run mode after writing the plan.",
    )
    parser.add_argument(
        "--launch-workers",
        action="store_true",
        help="Launch worker subprocesses. This currently returns a controlled not-implemented result until the one-patient runtime builder is wired.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="Optional timeout per worker subprocess.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    plan = build_patient_process_run_plan(
        input_case_manifest_path=args.input_case_manifest,
        output_root=args.output_root,
        pathway_name=args.pathway_name,
        checkpoint_name=args.checkpoint_name,
        patient_uids=tuple(args.patient_uid),
        run_id=args.run_id,
        failure_policy=args.failure_policy,
        max_workers=args.max_workers,
        metadata={"source": "run_patient_scientific_standalone.py"},
    )
    plan_path = write_patient_process_run_plan(plan)
    job_paths = write_patient_worker_job_packets(plan)
    print(f"[patient-process-runner] wrote plan: {plan_path}")
    print(f"[patient-process-runner] wrote worker jobs: {len(job_paths)}")
    print(f"[patient-process-runner] patient count: {len(plan.worker_jobs)}")

    if args.launch_dry_run_workers or args.launch_workers:
        results = run_patient_process_plan(
            plan,
            dry_run_workers=args.launch_dry_run_workers,
            timeout_seconds=args.timeout_seconds,
        )
        failed_results = [result for result in results if not result.succeeded]
        print(
            "[patient-process-runner] completed worker subprocesses: "
            f"{len(results)} | failed: {len(failed_results)}"
        )
        print(json.dumps([result.as_mapping() for result in results], indent=2, sort_keys=True))
        return 1 if failed_results else 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())