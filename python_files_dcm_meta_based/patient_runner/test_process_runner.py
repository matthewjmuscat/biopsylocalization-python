from __future__ import annotations

import csv
from importlib import import_module
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from config.snapshots import PipelineConfigSnapshot
from config.snapshots import canonical_sha256
from config.snapshots import read_pipeline_config_snapshot
from config.snapshots import write_pipeline_config_snapshot
from output_artifacts.run_compatibility import RunCompatibilityIdentity
from output_artifacts.run_compatibility import write_run_compatibility_identity
from output_artifacts.schema_registry import OUTPUT_SCHEMA_REGISTRY_VERSION
from startup.code_identity import capture_code_identity
from startup.runtime_environment import capture_runtime_environment_identity
from . import process_runner as process_runner_module
from .contracts import PatientStageStatus
from .contracts import PatientRunResult
from .contracts import PatientStageName
from .contracts import PatientStageResult
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
            inputs_present = loaded_job.patient_inputs.core_paths_all_present
            input_identity = loaded_job.patient_inputs.manifest_identity_sha256

        self.assertEqual(tuple(job.patient_case.patient_uid for job in plan.worker_jobs), ("P002", "P001"))
        self.assertEqual(loaded_job, plan.worker_jobs[0])
        self.assertTrue(inputs_present)
        self.assertEqual(len(input_identity), 64)

    def test_legacy_v1_job_hydrates_explicit_patient_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = _write_case_manifest(root, ("P001",), create_core_files=True)
            job = build_patient_process_run_plan(
                input_case_manifest_path=manifest_path,
                output_root=root.joinpath("output"),
                pathway_name="anatomical_qa",
                checkpoint_name="anatomical_qa",
            ).worker_jobs[0]
            payload = job.as_mapping()
            payload["schema_version"] = "patient_worker_job_v1"
            payload.pop("patient_inputs")
            payload["patient_case"]["metadata"]["core_input_paths"] = {
                role: path.as_posix() if path is not None else ""
                for role, path in job.patient_inputs.core_paths.items()
            }

            loaded = process_runner_module.PatientWorkerJob.from_mapping(payload)

        self.assertEqual(loaded.patient_inputs, job.patient_inputs)

    def test_legacy_v2_job_is_readable_but_fails_live_compatibility_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = _write_case_manifest(root, ("P001",), create_core_files=True)
            config_snapshot_path = _write_config_snapshot(root)
            compatibility_identity_path = _write_compatibility_identity(root, config_snapshot_path)
            job = build_patient_process_run_plan(
                input_case_manifest_path=manifest_path,
                output_root=root.joinpath("output"),
                pathway_name="anatomical_qa",
                checkpoint_name="anatomical_qa",
                scientific_config_snapshot_path=config_snapshot_path,
                run_compatibility_identity_path=compatibility_identity_path,
            ).worker_jobs[0]
            payload = job.as_mapping()
            payload["schema_version"] = "patient_worker_job_v2"
            payload.pop("run_compatibility_identity_path")
            payload["metadata"].pop("run_compatibility_identity")
            payload["metadata"].pop("run_compatibility_identity_file_sha256")
            loaded = process_runner_module.PatientWorkerJob.from_mapping(payload)

            result = run_patient_worker_job(loaded)

        self.assertEqual(loaded.run_compatibility_identity_path, None)
        self.assertFalse(result.succeeded)
        self.assertEqual(result.metadata["failed_boundary"], "run_compatibility_identity_preflight")

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

    def test_live_worker_reports_runtime_builder_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = _write_case_manifest(root, ("P001",), create_core_files=True)
            config_snapshot_path = _write_config_snapshot(root)
            compatibility_identity_path = _write_compatibility_identity(root, config_snapshot_path)
            job = build_patient_process_run_plan(
                input_case_manifest_path=manifest_path,
                output_root=root.joinpath("output"),
                pathway_name="anatomical_qa",
                checkpoint_name="anatomical_qa",
                scientific_config_snapshot_path=config_snapshot_path,
                run_compatibility_identity_path=compatibility_identity_path,
            ).worker_jobs[0]

            runtime_builder_module = import_module(
                "{}.runtime_builder".format(process_runner_module.__package__)
            )
            with patch(
                "config.rehydration.rehydrate_pipeline_scientific_config_snapshot",
                return_value=SimpleNamespace(),
            ), patch.object(
                process_runner_module,
                "_validate_worker_compatibility_identity",
                return_value={"run_compatibility_identity_sha256": "synthetic"},
            ), patch.object(
                runtime_builder_module,
                "build_standalone_patient_runtime",
                side_effect=RuntimeError("synthetic runtime failure"),
            ):
                result = run_patient_worker_job(job)
            failure_manifest_exists = job.patient_output_root.joinpath("patient_run_manifest.json").is_file()
            failure_manifest = json.loads(
                job.patient_output_root.joinpath("patient_run_manifest.json").read_text(encoding="utf-8")
            )

        self.assertFalse(result.succeeded)
        self.assertEqual(result.metadata["failed_boundary"], "one_patient_runtime_state_builder")
        self.assertTrue(failure_manifest_exists)
        self.assertEqual(failure_manifest["metadata"]["pathway_name"], "anatomical_qa")
        self.assertEqual(
            failure_manifest["metadata"]["planned_stage_names"],
            ["grid_preprocessing", "anatomical_preprocessing"],
        )
        self.assertIn("run_compatibility_identity", failure_manifest["metadata"])

    def test_live_worker_runs_anatomical_checkpoint_through_patient_runner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = _write_case_manifest(root, ("P001",), create_core_files=True)
            config_snapshot_path = _write_config_snapshot(root)
            compatibility_identity_path = _write_compatibility_identity(root, config_snapshot_path)
            job = build_patient_process_run_plan(
                input_case_manifest_path=manifest_path,
                output_root=root.joinpath("output"),
                pathway_name="anatomical_qa",
                checkpoint_name="anatomical_qa",
                scientific_config_snapshot_path=config_snapshot_path,
                run_compatibility_identity_path=compatibility_identity_path,
            ).worker_jobs[0]
            pipeline_config = SimpleNamespace()
            runtime_state = SimpleNamespace(metadata={})
            standalone_runtime = SimpleNamespace(
                runtime_state=runtime_state,
                config_build_context=object(),
            )
            patient_config = object()
            scientific_run_config = SimpleNamespace(
                batch_config=SimpleNamespace(
                    patient_config=patient_config,
                    metadata={"run_compatibility_identity": {"identity_sha256": "compatibility"}},
                ),
                metadata={"random_seed_policy": {"schema_version": "seed-policy"}},
                pathway_name=SimpleNamespace(value="anatomical_qa"),
                planned_stage_names=(
                    PatientStageName.GRID_PREPROCESSING,
                    PatientStageName.ANATOMICAL_PREPROCESSING,
                ),
            )
            patient_result = PatientRunResult.from_stage_results(
                job.patient_case,
                job.patient_output_root,
                (
                    PatientStageResult.success(PatientStageName.GRID_PREPROCESSING),
                    PatientStageResult.success(PatientStageName.ANATOMICAL_PREPROCESSING),
                ),
            )
            runtime_builder_module = import_module(
                "{}.runtime_builder".format(process_runner_module.__package__)
            )
            scientific_runner_module = import_module(
                "{}.scientific_runner".format(process_runner_module.__package__)
            )
            runner_module = import_module(
                "{}.runner".format(process_runner_module.__package__)
            )

            with patch(
                "config.rehydration.rehydrate_pipeline_scientific_config_snapshot",
                return_value=pipeline_config,
            ), patch.object(
                process_runner_module,
                "_validate_worker_compatibility_identity",
                return_value={"run_compatibility_identity_sha256": "synthetic"},
            ), patch.object(
                runtime_builder_module,
                "build_standalone_patient_runtime",
                return_value=standalone_runtime,
            ), patch.object(
                scientific_runner_module,
                "build_patient_scientific_run_config_from_pipeline",
                return_value=scientific_run_config,
            ), patch.object(
                scientific_runner_module,
                "build_patient_scientific_runner_stages",
                return_value=(object(), object()),
            ), patch.object(
                runner_module,
                "run_patient_case",
                return_value=patient_result,
            ) as run_patient:
                result = run_patient_worker_job(job)

        run_patient.assert_called_once()
        self.assertTrue(result.succeeded)
        self.assertEqual(result.metadata["executed_boundary"], "anatomical_qa")
        self.assertEqual(runtime_state.metadata["pathway_name"], "anatomical_qa")
        self.assertEqual(
            runtime_state.metadata["planned_stage_names"],
            ("grid_preprocessing", "anatomical_preprocessing"),
        )
        self.assertIn("random_seed_policy", runtime_state.metadata)
        self.assertIn("run_compatibility_identity", runtime_state.metadata)
        self.assertEqual(
            result.metadata["stage_statuses"],
            {
                PatientStageName.GRID_PREPROCESSING.value: PatientStageStatus.SUCCEEDED.value,
                PatientStageName.ANATOMICAL_PREPROCESSING.value: PatientStageStatus.SUCCEEDED.value,
            },
        )

    def test_live_worker_rejects_snapshot_changed_after_planning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = _write_case_manifest(root, ("P001",), create_core_files=True)
            config_snapshot_path = _write_config_snapshot(root)
            compatibility_identity_path = _write_compatibility_identity(root, config_snapshot_path)
            job = build_patient_process_run_plan(
                input_case_manifest_path=manifest_path,
                output_root=root.joinpath("output"),
                pathway_name="anatomical_qa",
                checkpoint_name="anatomical_qa",
                scientific_config_snapshot_path=config_snapshot_path,
                run_compatibility_identity_path=compatibility_identity_path,
            ).worker_jobs[0]
            config_snapshot_path.write_text(config_snapshot_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

            result = run_patient_worker_job(job)

        self.assertFalse(result.succeeded)
        self.assertEqual(
            result.metadata["failed_boundary"],
            "scientific_config_snapshot_identity_preflight",
        )

    def test_live_worker_rejects_compatibility_identity_changed_after_planning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = _write_case_manifest(root, ("P001",), create_core_files=True)
            config_snapshot_path = _write_config_snapshot(root)
            compatibility_identity_path = _write_compatibility_identity(root, config_snapshot_path)
            job = build_patient_process_run_plan(
                input_case_manifest_path=manifest_path,
                output_root=root.joinpath("output"),
                pathway_name="anatomical_qa",
                checkpoint_name="anatomical_qa",
                scientific_config_snapshot_path=config_snapshot_path,
                run_compatibility_identity_path=compatibility_identity_path,
            ).worker_jobs[0]
            compatibility_identity_path.write_text(
                compatibility_identity_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )

            result = run_patient_worker_job(job)

        self.assertFalse(result.succeeded)
        self.assertEqual(result.metadata["failed_boundary"], "run_compatibility_identity_preflight")

    def test_worker_accepts_current_strict_compatibility_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = _write_case_manifest(root, ("P001",), create_core_files=True)
            config_snapshot_path = _write_config_snapshot(root)
            compatibility_identity_path = _write_current_compatibility_identity(root, config_snapshot_path)
            job = build_patient_process_run_plan(
                input_case_manifest_path=manifest_path,
                output_root=root.joinpath("output"),
                pathway_name="anatomical_qa",
                checkpoint_name="anatomical_qa",
                scientific_config_snapshot_path=config_snapshot_path,
                run_compatibility_identity_path=compatibility_identity_path,
            ).worker_jobs[0]
            snapshot = read_pipeline_config_snapshot(config_snapshot_path)

            preflight = process_runner_module._validate_worker_compatibility_identity(
                job,
                scientific_config_sha256=snapshot.config_sha256,
            )

        self.assertEqual(preflight["output_schema_registry_version"], OUTPUT_SCHEMA_REGISTRY_VERSION)
        self.assertEqual(len(preflight["run_compatibility_identity_sha256"]), 64)

    def test_live_worker_requires_scientific_config_snapshot(self) -> None:
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
        self.assertEqual(result.metadata["failed_boundary"], "scientific_config_snapshot_preflight")

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
            config_snapshot_path = _write_config_snapshot(root)
            compatibility_identity_path = _write_compatibility_identity(root, config_snapshot_path)
            plan = build_patient_process_run_plan(
                input_case_manifest_path=manifest_path,
                output_root=root.joinpath("output"),
                pathway_name="anatomical_qa",
                checkpoint_name="anatomical_qa",
                failure_policy=PatientProcessFailurePolicy.STOP_ON_FAILURE,
                execution_mode="live_workers",
                scientific_config_snapshot_path=config_snapshot_path,
                run_compatibility_identity_path=compatibility_identity_path,
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
                scientific_config_snapshot_path=config_snapshot_path,
                run_compatibility_identity_path=compatibility_identity_path,
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
            config_snapshot_path = _write_config_snapshot(root)
            compatibility_identity_path = _write_compatibility_identity(root, config_snapshot_path)
            plan = build_patient_process_run_plan(
                input_case_manifest_path=manifest_path,
                output_root=root.joinpath("output"),
                pathway_name="anatomical_qa",
                checkpoint_name="anatomical_qa",
                execution_mode="live_workers",
                scientific_config_snapshot_path=config_snapshot_path,
                run_compatibility_identity_path=compatibility_identity_path,
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

    def test_worker_launch_does_not_reuse_stale_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = _write_case_manifest(root, ("P001",), create_core_files=True)
            config_snapshot_path = _write_config_snapshot(root)
            compatibility_identity_path = _write_compatibility_identity(root, config_snapshot_path)
            plan = build_patient_process_run_plan(
                input_case_manifest_path=manifest_path,
                output_root=root.joinpath("output"),
                pathway_name="anatomical_qa",
                checkpoint_name="anatomical_qa",
                execution_mode="live_workers",
                scientific_config_snapshot_path=config_snapshot_path,
                run_compatibility_identity_path=compatibility_identity_path,
            )
            job_path = write_patient_worker_job_packets(plan)[0]
            stale_result = PatientWorkerResult(
                worker_job=plan.worker_jobs[0],
                status=PatientStageStatus.SUCCEEDED,
                elapsed_seconds=0.0,
                exit_code=0,
            )
            process_runner_module.write_patient_worker_result(stale_result)

            with patch.object(
                process_runner_module.subprocess,
                "run",
                return_value=subprocess.CompletedProcess(("worker",), 1),
            ):
                result = launch_worker_job_file(job_path)

        self.assertFalse(result.succeeded)
        self.assertEqual(result.warnings, ("worker did not write a result JSON",))
        self.assertNotEqual(result.exit_code, 0)

    def test_worker_launch_converts_invalid_result_to_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = _write_case_manifest(root, ("P001",), create_core_files=True)
            config_snapshot_path = _write_config_snapshot(root)
            compatibility_identity_path = _write_compatibility_identity(root, config_snapshot_path)
            plan = build_patient_process_run_plan(
                input_case_manifest_path=manifest_path,
                output_root=root.joinpath("output"),
                pathway_name="anatomical_qa",
                checkpoint_name="anatomical_qa",
                execution_mode="live_workers",
                scientific_config_snapshot_path=config_snapshot_path,
                run_compatibility_identity_path=compatibility_identity_path,
            )
            job_path = write_patient_worker_job_packets(plan)[0]

            def write_invalid_result(*args, **kwargs):
                del args, kwargs
                plan.worker_jobs[0].result_path.parent.mkdir(parents=True, exist_ok=True)
                plan.worker_jobs[0].result_path.write_text("{invalid", encoding="utf-8")
                return subprocess.CompletedProcess(("worker",), 1)

            with patch.object(
                process_runner_module.subprocess,
                "run",
                side_effect=write_invalid_result,
            ):
                result = launch_worker_job_file(job_path)

        self.assertFalse(result.succeeded)
        self.assertEqual(result.metadata["failed_boundary"], "worker_result_validation")
        self.assertNotEqual(result.exit_code, 0)

    def test_worker_launch_converts_invalid_result_metadata_to_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = _write_case_manifest(root, ("P001",), create_core_files=True)
            config_snapshot_path = _write_config_snapshot(root)
            compatibility_identity_path = _write_compatibility_identity(root, config_snapshot_path)
            plan = build_patient_process_run_plan(
                input_case_manifest_path=manifest_path,
                output_root=root.joinpath("output"),
                pathway_name="anatomical_qa",
                checkpoint_name="anatomical_qa",
                execution_mode="live_workers",
                scientific_config_snapshot_path=config_snapshot_path,
                run_compatibility_identity_path=compatibility_identity_path,
            )
            job_path = write_patient_worker_job_packets(plan)[0]

            def write_invalid_result(*args, **kwargs):
                del args, kwargs
                invalid_payload = PatientWorkerResult(
                    worker_job=plan.worker_jobs[0],
                    status=PatientStageStatus.SUCCEEDED,
                    elapsed_seconds=0.0,
                    exit_code=0,
                ).as_mapping()
                invalid_payload["metadata"] = []
                plan.worker_jobs[0].result_path.parent.mkdir(parents=True, exist_ok=True)
                plan.worker_jobs[0].result_path.write_text(
                    json.dumps(invalid_payload),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(("worker",), 0)

            with patch.object(
                process_runner_module.subprocess,
                "run",
                side_effect=write_invalid_result,
            ):
                result = launch_worker_job_file(job_path)

        self.assertFalse(result.succeeded)
        self.assertEqual(result.metadata["failed_boundary"], "worker_result_validation")
        self.assertNotEqual(result.exit_code, 0)

    def test_live_plan_rejects_missing_provenance_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = _write_case_manifest(root, ("P001",), create_core_files=True)

            with self.assertRaises(FileNotFoundError):
                build_patient_process_run_plan(
                    input_case_manifest_path=manifest_path,
                    output_root=root.joinpath("output"),
                    pathway_name="anatomical_qa",
                    checkpoint_name="anatomical_qa",
                    execution_mode="live_workers",
                    scientific_config_snapshot_path=root.joinpath("missing-config.json"),
                    run_compatibility_identity_path=root.joinpath("missing-compatibility.json"),
                )

    def test_live_plan_rejects_unsupported_pathway_before_writing_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = _write_case_manifest(root, ("P001",), create_core_files=True)

            with self.assertRaisesRegex(ValueError, "supports anatomical_qa"):
                build_patient_process_run_plan(
                    input_case_manifest_path=manifest_path,
                    output_root=root.joinpath("output"),
                    pathway_name="full_current_pipeline_shadow",
                    checkpoint_name="full_current_pipeline_shadow",
                    execution_mode="live_workers",
                )

    def test_manifest_patient_uid_is_preserved_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            patient_uid = " P001 "
            manifest_path = _write_case_manifest(root, (patient_uid,), create_core_files=True)

            plan = build_patient_process_run_plan(
                input_case_manifest_path=manifest_path,
                output_root=root.joinpath("output"),
                pathway_name="anatomical_qa",
                checkpoint_name="anatomical_qa",
            )

        self.assertEqual(plan.worker_jobs[0].patient_case.patient_uid, patient_uid)
        self.assertEqual(plan.worker_jobs[0].patient_inputs.patient_uid, patient_uid)

    def test_manifest_duplicate_patient_uid_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = _write_case_manifest(root, ("P001", "P001"), create_core_files=True)

            with self.assertRaisesRegex(ValueError, "duplicate patient UID"):
                build_patient_process_run_plan(
                    input_case_manifest_path=manifest_path,
                    output_root=root.joinpath("output"),
                    pathway_name="anatomical_qa",
                    checkpoint_name="anatomical_qa",
                )

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

    def test_standalone_cli_exposes_live_scientific_snapshot_input(self) -> None:
        script_path = Path(__file__).resolve().parents[1].joinpath("run_patient_scientific_standalone.py")

        completed = subprocess.run(
            (sys.executable, str(script_path), "--help"),
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        self.assertIn("--scientific-config-snapshot", completed.stdout)
        self.assertIn("--run-compatibility-identity", completed.stdout)
        self.assertIn("anatomical_qa", completed.stdout)

    def test_process_runner_import_does_not_load_gpu_libraries(self) -> None:
        python_root = Path(__file__).resolve().parents[1]
        command = (
            "import sys; "
            "sys.path.insert(0, {!r}); "
            "import patient_runner.process_runner; "
            "print(','.join(name for name in ('cupy', 'cudf', 'cuspatial', 'rmm') if name in sys.modules))"
        ).format(str(python_root))

        completed = subprocess.run(
            (sys.executable, "-c", command),
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        self.assertEqual(completed.stdout.strip(), "")


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


def _write_config_snapshot(root: Path) -> Path:
    payload = {"pathway": "anatomical_qa"}
    path = root.joinpath("resolved_scientific_config.json")
    write_pipeline_config_snapshot(
        PipelineConfigSnapshot(
            config_type="config.PipelineConfig.scientific",
            config=payload,
            config_sha256=canonical_sha256(payload),
        ),
        path,
    )
    return path


def _write_compatibility_identity(root: Path, config_snapshot_path: Path) -> Path:
    snapshot = read_pipeline_config_snapshot(config_snapshot_path)
    path = root.joinpath("run_compatibility_identity.json")
    write_run_compatibility_identity(
        RunCompatibilityIdentity(
            scientific_config_sha256=snapshot.config_sha256,
            code_source_sha256="synthetic-code-source",
            input_policy_sha256=canonical_sha256({"policy": "synthetic"}),
            runtime_environment_sha256="synthetic-runtime-environment",
            output_schema_registry_version="synthetic-output-schema",
        ),
        path,
    )
    return path


def _write_current_compatibility_identity(root: Path, config_snapshot_path: Path) -> Path:
    snapshot = read_pipeline_config_snapshot(config_snapshot_path)
    repository_root = Path(__file__).resolve().parents[2]
    code_identity = capture_code_identity(repository_root)
    environment_identity = capture_runtime_environment_identity(repository_root)
    path = root.joinpath("run_compatibility_identity.json")
    write_run_compatibility_identity(
        RunCompatibilityIdentity(
            scientific_config_sha256=snapshot.config_sha256,
            code_source_sha256=code_identity.source_tree_sha256,
            input_policy_sha256=canonical_sha256({"policy": "synthetic"}),
            runtime_environment_sha256=environment_identity.identity_sha256,
            output_schema_registry_version=OUTPUT_SCHEMA_REGISTRY_VERSION,
            code_commit=code_identity.commit,
            code_dirty=code_identity.dirty,
        ),
        path,
    )
    return path


if __name__ == "__main__":
    unittest.main()
