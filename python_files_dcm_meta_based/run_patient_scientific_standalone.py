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
from patient_runner.run_profile import load_patient_orchestration_profile


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan or launch the standalone patient-scientific process runner."
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=None,
        help="Orchestration-only TOML profile. Cannot be combined with manual run arguments.",
    )
    parser.add_argument(
        "--input-case-manifest",
        type=Path,
        default=None,
        help="Path to manifests/input_case_manifest.csv from a completed input discovery run.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Standalone patient-runner output root.",
    )
    parser.add_argument(
        "--pathway-name",
        default=None,
        help="Patient scientific pathway name to record in worker jobs.",
    )
    parser.add_argument(
        "--checkpoint-name",
        default=None,
        help="Patient scientific checkpoint name to record in worker jobs.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run ID recorded in the parent plan and worker jobs.",
    )
    parser.add_argument(
        "--scientific-config-snapshot",
        type=Path,
        default=None,
        help="Verified resolved_scientific_config.json consumed by live patient workers.",
    )
    parser.add_argument(
        "--run-compatibility-identity",
        type=Path,
        default=None,
        help="Verified run_compatibility_identity.json required by live patient workers.",
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
        default=None,
        help="Parent failure policy for worker subprocesses.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
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
        help="Launch workers for anatomical_qa, biopsy_preprocessing_shadow or optimization_shadow with a matching checkpoint.",
    )
    parser.add_argument("--capture-input-content", action="store_true",
                        help="Bind declared input file bytes to jobs and verify them before/after execution.")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="Optional timeout per worker subprocess.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.launch_dry_run_workers and args.launch_workers:
        raise ValueError("choose either --launch-dry-run-workers or --launch-workers, not both")

    if args.profile is not None:
        if args.capture_input_content:
            raise ValueError("--capture-input-content currently belongs to manual planning, not a profile override")
        _reject_manual_args_with_profile(args)
        profile = load_patient_orchestration_profile(args.profile)
        if not profile.enabled:
            print("[patient-process-runner] profile is disabled: {}".format(profile.source_path))
            return 0
        plan = profile.build_process_run_plan()
        execution_mode = profile.execution_mode
    else:
        if args.input_case_manifest is None or args.output_root is None:
            raise ValueError("manual mode requires --input-case-manifest and --output-root")
        execution_mode = _manual_execution_mode(args)
        if execution_mode == "live_workers" and args.scientific_config_snapshot is None:
            raise ValueError("manual live workers require --scientific-config-snapshot")
        if execution_mode == "live_workers" and args.run_compatibility_identity is None:
            raise ValueError("manual live workers require --run-compatibility-identity")
        plan = build_patient_process_run_plan(
            input_case_manifest_path=args.input_case_manifest,
            output_root=args.output_root,
            pathway_name=args.pathway_name or "anatomical_qa",
            checkpoint_name=args.checkpoint_name or "anatomical_qa",
            patient_uids=tuple(args.patient_uid),
            run_id=args.run_id or "patient-process-runner",
            failure_policy=args.failure_policy or PatientProcessFailurePolicy.STOP_ON_FAILURE.value,
            max_workers=1 if args.max_workers is None else args.max_workers,
            timeout_seconds=args.timeout_seconds,
            execution_mode=execution_mode,
            scientific_config_snapshot_path=args.scientific_config_snapshot,
            run_compatibility_identity_path=args.run_compatibility_identity,
            capture_input_content=args.capture_input_content,
            metadata={"source": "run_patient_scientific_standalone.py", "source_mode": "manual_cli"},
        )

    if execution_mode == "plan_only":
        plan_path = write_patient_process_run_plan(plan)
        job_paths = write_patient_worker_job_packets(plan)
        results = ()
    else:
        results = run_patient_process_plan(
            plan,
            dry_run_workers=execution_mode == "dry_run_workers",
        )
        plan_path = plan.plan_path
        job_paths = tuple(worker_job.job_path for worker_job in plan.worker_jobs)
    print(f"[patient-process-runner] wrote plan: {plan_path}")
    print(f"[patient-process-runner] wrote worker jobs: {len(job_paths)}")
    print(f"[patient-process-runner] patient count: {len(plan.worker_jobs)}")
    print(f"[patient-process-runner] execution mode: {execution_mode}")

    if results:
        failed_results = [result for result in results if not result.succeeded]
        print(
            "[patient-process-runner] completed worker subprocesses: "
            f"{len(results)} | failed: {len(failed_results)}"
        )
        print(json.dumps([result.as_mapping() for result in results], indent=2, sort_keys=True))
        return 1 if failed_results else 0

    return 0


def _manual_execution_mode(args: argparse.Namespace) -> str:
    if bool(args.launch_workers):
        return "live_workers"
    if bool(args.launch_dry_run_workers):
        return "dry_run_workers"
    return "plan_only"


def _reject_manual_args_with_profile(args: argparse.Namespace) -> None:
    conflicting_values = {
        "--input-case-manifest": args.input_case_manifest,
        "--output-root": args.output_root,
        "--pathway-name": args.pathway_name,
        "--checkpoint-name": args.checkpoint_name,
        "--run-id": args.run_id,
        "--scientific-config-snapshot": args.scientific_config_snapshot,
        "--run-compatibility-identity": args.run_compatibility_identity,
        "--patient-uid": tuple(args.patient_uid),
        "--failure-policy": args.failure_policy,
        "--max-workers": args.max_workers,
        "--launch-dry-run-workers": bool(args.launch_dry_run_workers),
        "--launch-workers": bool(args.launch_workers),
        "--timeout-seconds": args.timeout_seconds,
    }
    conflicts = [name for name, value in conflicting_values.items() if value not in (None, (), False)]
    if conflicts:
        raise ValueError("--profile cannot be combined with manual run arguments: {}".format(", ".join(conflicts)))


if __name__ == "__main__":
    raise SystemExit(main())
