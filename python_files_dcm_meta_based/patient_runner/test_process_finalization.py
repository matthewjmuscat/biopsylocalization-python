"""Synthetic persisted-result tests for the standalone parent handoff."""

from pathlib import Path
from tempfile import TemporaryDirectory
from dataclasses import replace
import json
import unittest

from patient_runner.contracts import PatientCase, PatientRunResult, PatientStageResult, PatientStageStatus
from patient_runner.manifests import write_patient_run_manifest
from patient_runner.process_runner import build_patient_process_run_plan, PatientWorkerResult, run_patient_process_plan
from patient_runner.process_finalization import finalize_patient_process_run
from patient_runner.test_process_runner import _write_case_manifest, _write_config_snapshot, _write_compatibility_identity
from post_run.cohort_assembly.manifest_loader import load_patient_result_from_manifest, load_patient_batch_result_from_manifest


class PatientManifestReadbackTests(unittest.TestCase):
    def test_single_patient_round_trip_preserves_exact_identity(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            result = PatientRunResult.from_stage_results(
                PatientCase(patient_uid=" Synthetic "),
                Path(temporary_directory),
                (PatientStageResult.success("anatomical_preprocessing"),),
                metadata={"run_id": "synthetic"},
            )
            loaded = load_patient_result_from_manifest(write_patient_run_manifest(result))
            self.assertEqual(loaded, result)

    def test_complete_batch_loads_through_existing_assembly_reader(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            plan = _plan(Path(temporary_directory))
            workers = tuple(_success(job) for job in plan.worker_jobs)
            finalized = finalize_patient_process_run(plan, workers)
            loaded = load_patient_batch_result_from_manifest(plan.output_root)
            self.assertTrue(loaded.succeeded)
            self.assertEqual([patient.patient_case.patient_uid for patient in loaded.patient_results], ["P001", "P002"])
            self.assertTrue(finalized.index_path.is_file())
            self.assertTrue(loaded.metadata["assembly_ready"])

    def test_timeout_and_unlaunched_patients_block_assembly(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            plan = _plan(Path(temporary_directory))
            worker = PatientWorkerResult(plan.worker_jobs[0], PatientStageStatus.FAILED, 1.0, 124, timed_out=True)
            finalized = finalize_patient_process_run(plan, (worker,))
            self.assertEqual([patient.status for patient in finalized.batch_result.patient_results], [PatientStageStatus.FAILED, PatientStageStatus.NOT_STARTED])
            self.assertTrue(worker.worker_job.result_path.is_file())
            with self.assertRaisesRegex(ValueError, "not assembly-ready"):
                load_patient_batch_result_from_manifest(plan.output_root)

    def test_missing_artifact_converts_success_to_failure(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            plan = _plan(Path(temporary_directory))
            workers = tuple(_success(job) for job in plan.worker_jobs)
            manifest = plan.worker_jobs[0].patient_output_root / "patient_run_manifest.json"
            payload = json.loads(manifest.read_text())
            payload["artifact_paths"] = [str(manifest.parent / "missing.csv")]
            manifest.write_text(json.dumps(payload))
            finalized = finalize_patient_process_run(plan, workers)
            self.assertFalse(finalized.worker_results[0].succeeded)
            self.assertFalse(finalized.batch_result.metadata["assembly_ready"])

    def test_dry_run_is_not_scientific_completion(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            plan = replace(_plan(Path(temporary_directory)), execution_mode="dry_run_workers")
            workers = tuple(PatientWorkerResult(job, PatientStageStatus.SKIPPED, 0.0, 0, dry_run=True) for job in plan.worker_jobs)
            finalized = finalize_patient_process_run(plan, workers)
            self.assertFalse(finalized.batch_result.metadata["assembly_ready"])
            self.assertFalse(finalized.batch_result.metadata["scientific_execution"])

    def test_existing_run_index_is_not_overwritten(self):
        with TemporaryDirectory() as temporary_directory:
            plan = _plan(Path(temporary_directory))
            index = plan.output_root / "manifests" / "run_manifest_index.json"
            index.parent.mkdir(parents=True)
            index.write_text("existing evidence")
            with self.assertRaises(FileExistsError):
                run_patient_process_plan(plan)
            self.assertEqual(index.read_text(), "existing evidence")

    def test_reordered_or_duplicate_stages_fail_validation(self):
        with TemporaryDirectory() as temporary_directory:
            plan = _plan(Path(temporary_directory))
            workers = tuple(_success(job) for job in plan.worker_jobs)
            path = plan.worker_jobs[0].patient_output_root / "patient_run_manifest.json"
            payload = json.loads(path.read_text())
            payload["stages"].reverse()
            path.write_text(json.dumps(payload))
            result = finalize_patient_process_run(plan, workers)
            self.assertFalse(result.worker_results[0].succeeded)


def _plan(root):
    snapshot = _write_config_snapshot(root)
    return build_patient_process_run_plan(
        input_case_manifest_path=_write_case_manifest(root, ("P001", "P002"), create_core_files=True),
        output_root=root / "output", pathway_name="anatomical_qa", checkpoint_name="anatomical_qa",
        scientific_config_snapshot_path=snapshot,
        run_compatibility_identity_path=_write_compatibility_identity(root, snapshot),
        execution_mode="live_workers",
    )


def _success(job):
    patient = PatientRunResult.from_stage_results(
        job.patient_case, job.patient_output_root,
        tuple(PatientStageResult.success(name) for name in (*job.metadata["planned_stage_names"], "patient_artifact_writing")),
        metadata={**job.metadata, "run_id": job.run_id, "worker_job_id": job.job_id},
    )
    write_patient_run_manifest(patient)
    return PatientWorkerResult(job, PatientStageStatus.SUCCEEDED, 0.0, 0, metadata={"artifact_paths": []})


if __name__ == "__main__":
    unittest.main()