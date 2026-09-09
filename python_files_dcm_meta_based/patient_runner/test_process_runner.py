from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import process_runner as process_runner_module
from .contracts import PatientStageStatus
from .process_runner import PatientProcessFailurePolicy
from .process_runner import PatientWorkerResult
from .process_runner import build_patient_process_run_plan
from .process_runner import launch_worker_job_file
from .process_runner import load_patient_worker_job
from .process_runner import run_patient_process_plan
from .process_runner import run_patient_worker_job
from .process_runner import write_patient_worker_job_packets


_CASE_MANIFEST_COLUMNS = (
    "Patient UID (generated)",
    "Patient Name",
    "Patient ID (from dicom)",
    "Fraction number (legacy parsed)",
    "Has RTSTRUCT",
    "Has RTDOSE",
    "Has RTPLAN",
    "Core RTSTRUCT/RTDOSE/RTPLAN complete",
    "RTSTRUCT path",
    "RTDOSE path",
    "RTPLAN path",
    "Num US files",
    "Num MR T2 files",
    "Num MR ADC files",
    "US paths",
    "MR T2 paths",
    "MR ADC paths",
)


class PatientProcessRunnerTests(unittest.TestCase):
    def test_build_plan_preserves_requested_patient_order_and_job_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = _write_case_manifest(root, ("P001", "P002"), create_core_files=True)
            plan = build_patient_process_run_plan(
                input_case_manifest_path=manifest_path,
                output_root=root.joinpath("output"),
                pathway_name="anatomical_qa",
                checkpoint_name="anatomical_qa",
                patient_uids=("P002", "P001"),
                run_id="synthetic",
            )

            job_paths = write_patient_worker_job_packets(plan)
            loaded_job = load_patient_worker_job(job_paths[0])

        self.assertEqual(tuple(job.patient_case.patient_uid for job in plan.worker_jobs), ("P002", "P001"))
        self.assertEqual(loaded_job, plan.worker_jobs[0])
        self.assertTrue(loaded_job.patient_case.metadata["core_input_paths_all_present"])

    def test_dry_run_reports_missing_inputs_without_failing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = _write_case_manifest(root, ("P001",), create_core_files=False)
            job = build_patient_process_run_plan(
                input_case_manifest_path=manifest_path,
                output_root=root.joinpath("output"),
                pathway_name="anatomical_qa",
                checkpoint_name="anatomical_qa",
            ).worker_jobs[0]

            result = run_patient_worker_job(job, dry_run=True)

        self.assertTrue(result.succeeded)
        self.assertTrue(result.dry_run)
        self.assertEqual(result.status, PatientStageStatus.SKIPPED)
        self.assertEqual(
            result.metadata["input_preflight"]["missing_core_input_roles"],
            ("rtstruct", "rtdose", "rtplan"),
        )

    def test_live_worker_fails_at_core_input_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = _write_case_manifest(root, ("P001",), create_core_files=False)
            job = build_patient_process_run_plan(
                input_case_manifest_path=manifest_path,
                output_root=root.joinpath("output"),
                pathway_name="anatomical_qa",
                checkpoint_name="anatomical_qa",
            ).worker_jobs[0]

            result = run_patient_worker_job(job)

        self.assertFalse(result.succeeded)
        self.assertEqual(result.metadata["failed_boundary"], "core_input_path_preflight")

    def test_live_worker_fails_closed_at_unimplemented_runtime_builder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = _write_case_manifest(root, ("P001",), create_core_files=True)
            job = build_patient_process_run_plan(
                input_case_manifest_path=manifest_path,
                output_root=root.joinpath("output"),
                pathway_name="anatomical_qa",
                checkpoint_name="anatomical_qa",
            ).worker_jobs[0]

            result = run_patient_worker_job(job)

        self.assertFalse(result.succeeded)
        self.assertEqual(result.metadata["missing_boundary"], "one_patient_runtime_state_builder")

    def test_process_plan_launches_cpu_only_dry_run_worker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = _write_case_manifest(root, ("P001",), create_core_files=True)
            plan = build_patient_process_run_plan(
                input_case_manifest_path=manifest_path,
                output_root=root.joinpath("output"),
                pathway_name="anatomical_qa",
                checkpoint_name="anatomical_qa",
                execution_mode="dry_run_workers",
            )

            results = run_patient_process_plan(plan, dry_run_workers=True)

            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].succeeded)
            self.assertTrue(plan.plan_path.is_file())
            self.assertTrue(plan.worker_jobs[0].result_path.is_file())

    def test_process_plan_respects_failure_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = _write_case_manifest(root, ("P001", "P002"), create_core_files=True)
            plan = build_patient_process_run_plan(
                input_case_manifest_path=manifest_path,
                output_root=root.joinpath("output"),
                pathway_name="anatomical_qa",
                checkpoint_name="anatomical_qa",
                failure_policy=PatientProcessFailurePolicy.STOP_ON_FAILURE,
                execution_mode="live_workers",
            )
            failed_result = PatientWorkerResult(
                worker_job=plan.worker_jobs[0],
                status=PatientStageStatus.FAILED,
                elapsed_seconds=0.0,
                exit_code=2,
            )
            succeeding_result = PatientWorkerResult(
                worker_job=plan.worker_jobs[1],
                status=PatientStageStatus.SUCCEEDED,
                elapsed_seconds=0.0,
                exit_code=0,
            )

            with patch.object(
                process_runner_module,
                "launch_worker_job_file",
                side_effect=(failed_result, succeeding_result),
            ) as launch_mock:
                stopped_results = run_patient_process_plan(plan)

            continuing_plan = build_patient_process_run_plan(
                input_case_manifest_path=manifest_path,
                output_root=root.joinpath("continue_output"),
                pathway_name="anatomical_qa",
                checkpoint_name="anatomical_qa",
                failure_policy=PatientProcessFailurePolicy.CONTINUE_ON_FAILURE,
                execution_mode="live_workers",
            )
            continuing_results = (
                PatientWorkerResult(
                    worker_job=continuing_plan.worker_jobs[0],
                    status=PatientStageStatus.FAILED,
                    elapsed_seconds=0.0,
                    exit_code=2,
                ),
                PatientWorkerResult(
                    worker_job=continuing_plan.worker_jobs[1],
                    status=PatientStageStatus.SUCCEEDED,
                    elapsed_seconds=0.0,
                    exit_code=0,
                ),
            )
            with patch.object(
                process_runner_module,
                "launch_worker_job_file",
                side_effect=continuing_results,
            ) as continue_launch_mock:
                completed_results = run_patient_process_plan(continuing_plan)

        self.assertEqual(len(stopped_results), 1)
        self.assertEqual(launch_mock.call_count, 1)
        self.assertEqual(len(completed_results), 2)
        self.assertEqual(continue_launch_mock.call_count, 2)

    def test_plan_only_cannot_launch_workers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = _write_case_manifest(root, ("P001",), create_core_files=True)
            plan = build_patient_process_run_plan(
                input_case_manifest_path=manifest_path,
                output_root=root.joinpath("output"),
                pathway_name="anatomical_qa",
                checkpoint_name="anatomical_qa",
            )

            with self.assertRaisesRegex(ValueError, "plan_only"):
                run_patient_process_plan(plan)

    def test_worker_timeout_becomes_durable_failed_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = _write_case_manifest(root, ("P001",), create_core_files=True)
            plan = build_patient_process_run_plan(
                input_case_manifest_path=manifest_path,
                output_root=root.joinpath("output"),
                pathway_name="anatomical_qa",
                checkpoint_name="anatomical_qa",
                execution_mode="live_workers",
            )
            job_path = write_patient_worker_job_packets(plan)[0]

            with patch.object(
                process_runner_module.subprocess,
                "run",
                side_effect=subprocess.TimeoutExpired(("worker",), timeout=2.0),
            ):
                result = launch_worker_job_file(job_path, timeout_seconds=2.0)

        self.assertFalse(result.succeeded)
        self.assertTrue(result.timed_out)
        self.assertEqual(result.exit_code, 124)

    def test_worker_cli_help_does_not_require_scientific_execution(self) -> None:
        script_path = Path(__file__).resolve().parents[1].joinpath("run_patient_scientific_worker.py")

        completed = subprocess.run(
            (sys.executable, str(script_path), "--help"),
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        self.assertIn("--dry-run", completed.stdout)


def _write_case_manifest(
    root: Path,
    patient_uids: tuple[str, ...],
    *,
    create_core_files: bool,
) -> Path:
    manifest_path = root.joinpath("input_case_manifest.csv")
    rows = []
    for patient_uid in patient_uids:
        core_paths = {
            role: root.joinpath("inputs", patient_uid, "{}.dcm".format(role))
            for role in ("rtstruct", "rtdose", "rtplan")
        }
        if create_core_files:
            for core_path in core_paths.values():
                core_path.parent.mkdir(parents=True, exist_ok=True)
                core_path.touch()
        rows.append(
            {
                "Patient UID (generated)": patient_uid,
                "Patient Name": patient_uid,
                "Patient ID (from dicom)": patient_uid,
                "Fraction number (legacy parsed)": "1",
                "Has RTSTRUCT": "true",
                "Has RTDOSE": "true",
                "Has RTPLAN": "true",
                "Core RTSTRUCT/RTDOSE/RTPLAN complete": "true",
                "RTSTRUCT path": core_paths["rtstruct"],
                "RTDOSE path": core_paths["rtdose"],
                "RTPLAN path": core_paths["rtplan"],
                "Num US files": "0",
                "Num MR T2 files": "0",
                "Num MR ADC files": "0",
                "US paths": "",
                "MR T2 paths": "",
                "MR ADC paths": "",
            }
        )
    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=_CASE_MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


if __name__ == "__main__":
    unittest.main()
