"""Exercise the actual reference CLI guard, dispatch and fresh dry-run process."""

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_patient_anatomical_reference as reference
from patient_runner.process_runner import build_patient_process_run_plan, write_patient_worker_job_packets
from patient_runner.test_process_runner import _write_case_manifest
from validation.anatomical_execution import build_legacy_input_anatomical_runtime


class ReferenceWorkerTests(unittest.TestCase):
    def test_live_entrypoint_dispatches_supported_boundaries_to_independent_builder(self):
        for boundary in ("anatomical_qa", "biopsy_preprocessing_shadow", "optimization_shadow"):
            with self.subTest(boundary=boundary):
                job = SimpleNamespace(pathway_name=boundary, checkpoint_name=boundary)
                result = SimpleNamespace(exit_code=0)
                with patch("patient_runner.process_runner.load_patient_worker_job", return_value=job), \
                     patch("patient_runner.process_runner.run_patient_worker_job", return_value=result) as run, \
                     patch("patient_runner.process_runner.write_patient_worker_result") as write:
                    self.assertEqual(reference.main(["synthetic-job.json"]), 0)
                run.assert_called_once_with(job, dry_run=False, runtime_builder=build_legacy_input_anatomical_runtime)
                write.assert_called_once_with(result)

    def test_unsupported_pathway_and_mismatched_checkpoint_fail_before_dispatch(self):
        for pathway, checkpoint in (("full", "full"), ("unknown", "unknown"),
                                    ("optimization_shadow", "biopsy_preprocessing_shadow"),
                                    ("post_optimizer_biopsy_realization_shadow", "post_optimizer_biopsy_realization_shadow"),
                                    ("biopsy_preprocessing_shadow", "anatomical_qa"),
                                    ("anatomical_qa", "biopsy_preprocessing_shadow")):
            with self.subTest(pathway=pathway, checkpoint=checkpoint):
                job = SimpleNamespace(pathway_name=pathway, checkpoint_name=checkpoint)
                with patch("patient_runner.process_runner.load_patient_worker_job", return_value=job), \
                     patch("patient_runner.process_runner.run_patient_worker_job") as run:
                    with self.assertRaisesRegex(ValueError, "unsupported preprocessing checkpoint|matching pathway/checkpoint"):
                        reference.main(["synthetic-job.json"])
                    run.assert_not_called()

    def test_actual_reference_subprocess_accepts_supported_jobs(self):
        script = Path(reference.__file__).resolve()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _write_case_manifest(root, ("SYNTHETIC",), create_core_files=False)
            for boundary in ("anatomical_qa", "biopsy_preprocessing_shadow", "optimization_shadow"):
                with self.subTest(boundary=boundary):
                    plan = build_patient_process_run_plan(input_case_manifest_path=manifest,
                        output_root=root / boundary, pathway_name=boundary, checkpoint_name=boundary)
                    job_path = write_patient_worker_job_packets(plan)[0]
                    completed = subprocess.run([sys.executable, str(script), str(job_path), "--dry-run"],
                                               capture_output=True, text=True, timeout=30)
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    result = json.loads(plan.worker_jobs[0].result_path.read_text())
                    self.assertTrue(result["dry_run"])
                    self.assertEqual(result["exit_code"], 0)
