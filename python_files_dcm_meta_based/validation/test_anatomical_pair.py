"""CPU-only tests of paired orchestration and the real checkpoint hook."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from patient_runner.contracts import PatientRunResult, PatientStageResult, PatientStageStatus
from patient_runner.manifests import write_patient_run_manifest
from patient_runner.process_runner import load_patient_worker_job, PatientWorkerResult, write_patient_worker_job_packets
from patient_runner.test_process_finalization import _plan
from config.snapshots import build_pipeline_scientific_config_snapshot
from patient_runner.runner import PatientStage
from validation.anatomical_checkpoint import write_anatomical_checkpoint
from validation.anatomical_execution import with_anatomical_checkpoint
from validation.anatomical_pair import run_anatomical_pair
from validation.test_anatomical_checkpoint import _config, _runtime


class AnatomicalPairTests(unittest.TestCase):
    def test_hook_captures_after_success_without_changing_artifact_inventory(self):
        with TemporaryDirectory() as temporary_directory:
            runtime = _runtime()
            runtime.metadata = {}
            config = SimpleNamespace(patient_output_dir=lambda case: Path(temporary_directory))
            stage = PatientStage("anatomical_preprocessing", lambda state, config: PatientStageResult.success("anatomical_preprocessing"))
            wrapped = with_anatomical_checkpoint((stage,), _config())
            result = wrapped[0].runner(runtime, config)
            self.assertTrue(Path(result.metadata["anatomical_checkpoint_path"]).is_file())
            self.assertEqual(result.output_paths, ())

    def test_paired_service_compares_real_evidence_and_keeps_lane_commands_distinct(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            plan = _plan(root)
            from dataclasses import replace

            plan = replace(plan, worker_jobs=tuple(replace(job, metadata={**job.metadata, "scientific_config_snapshot_fingerprint_sha256": build_pipeline_scientific_config_snapshot(_config()).config_sha256}) for job in plan.worker_jobs))
            job_path = write_patient_worker_job_packets(plan)[0]
            scripts = []

            def launch(path, *, timeout_seconds, worker_script_path, log_path):
                scripts.append(worker_script_path)
                job = load_patient_worker_job(path)
                runtime = _runtime()
                runtime.master_structure_reference_dict = {job.patient_case.patient_uid: runtime.master_structure_reference_dict["synthetic-001"]}
                runtime.patient_case = job.patient_case
                checkpoint = write_anatomical_checkpoint(
                    runtime_state=runtime, pipeline_config=_config(),
                    output_dir=job.patient_output_root / "validation" / "anatomical",
                    metadata={**job.metadata, "worker_job_id": job.job_id, "run_id": job.run_id},
                )
                stages = tuple(PatientStageResult.success(name, metadata={"anatomical_checkpoint_path": str(checkpoint)} if name == "anatomical_preprocessing" else {}) for name in ("grid_preprocessing", "anatomical_preprocessing", "patient_artifact_writing"))
                write_patient_run_manifest(PatientRunResult.from_stage_results(
                    job.patient_case, job.patient_output_root, stages,
                    metadata={**job.metadata, "worker_job_id": job.job_id, "run_id": job.run_id},
                ))
                return PatientWorkerResult(job, PatientStageStatus.SUCCEEDED, 0.0, 0, metadata={"artifact_paths": []})

            with patch("patient_runner.process_runner.launch_worker_job_file", side_effect=launch):
                result = run_anatomical_pair(job_path=job_path, output_dir=root / "pair", abs_tol=0.0, rel_tol=0.0, timeout_seconds=10.0)
            self.assertTrue(result["passed"], result)
            self.assertIsNone(scripts[0])
            self.assertEqual(scripts[1].name, "run_patient_anatomical_reference.py")

    def test_failed_lane_writes_failed_summary_without_comparison(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = write_patient_worker_job_packets(_plan(root))[0]
            with patch("patient_runner.process_runner.launch_worker_job_file", side_effect=RuntimeError("synthetic failure")):
                report = run_anatomical_pair(job_path=path, output_dir=root / "pair", abs_tol=0.0, rel_tol=0.0, timeout_seconds=1.0)
            self.assertFalse(report["passed"])
            self.assertNotIn("comparison_path", report)
            self.assertFalse(json.loads((root / "pair" / "anatomical_pair_summary.json").read_text())["passed"])

    def test_invalid_tolerance_fails_before_patient_launch(self):
        with self.assertRaises(ValueError):
            run_anatomical_pair(job_path=Path("absent.json"), output_dir=Path("unused"), abs_tol=float("nan"), rel_tol=0.0, timeout_seconds=1.0)


if __name__ == "__main__":
    unittest.main()