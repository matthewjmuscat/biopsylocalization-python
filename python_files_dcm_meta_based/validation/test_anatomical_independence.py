"""CPU synthetic process, provenance and numerical independence qualification.

The fixture worker writes deterministic synthetic geometry, not anatomical
science. Real subprocesses, jobs, finalization and checkpoint comparison execute.
"""

from dataclasses import replace
import itertools
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from config.snapshots import build_pipeline_scientific_config_snapshot, write_pipeline_config_snapshot
from input_data.content_identity import capture_patient_input_content, verify_patient_input_content
from output_artifacts.run_compatibility import RunCompatibilityIdentity
from patient_runner.contracts import PatientRunResult, PatientStageResult
from patient_runner.manifests import write_patient_run_manifest
from patient_runner.process_runner import (
    build_patient_process_run_plan, launch_worker_job_file, load_patient_worker_job,
    PatientWorkerResult, run_patient_process_plan, write_patient_worker_job_packets, write_patient_worker_result,
)
from patient_runner.test_process_runner import _write_case_manifest, _write_compatibility_identity
from post_run.completed_patients import match_completed_patients, read_completed_patient
from validation.anatomical_checkpoint import write_anatomical_checkpoint
from validation.anatomical_independence import (
    _arm_plan, compare_completed_anatomical_jobs, qualification_orders, run_anatomical_independence, write_new_json,
)
from validation.test_anatomical_checkpoint import _config, _runtime


def _base(root, uids=("A", "B", "C")):
    root.mkdir(parents=True, exist_ok=True)
    snapshot = root / "config.json"
    write_pipeline_config_snapshot(build_pipeline_scientific_config_snapshot(_config()), snapshot)
    return build_patient_process_run_plan(
        input_case_manifest_path=_write_case_manifest(root, uids, create_core_files=True),
        scientific_config_snapshot_path=snapshot,
        run_compatibility_identity_path=_write_compatibility_identity(root, snapshot),
        output_root=root / "source", patient_uids=uids, pathway_name="anatomical_qa",
        checkpoint_name="anatomical_qa", execution_mode="live_workers", capture_input_content=True,
        metadata={"capture_anatomical_checkpoint": True},
    )


def _emit_completed(job_path, *, volume=1.25):
    job = load_patient_worker_job(Path(job_path))
    verify_patient_input_content(job.metadata["input_content_identity"], job.patient_inputs)
    runtime = _runtime()
    patient = runtime.master_structure_reference_dict.pop("synthetic-001")
    patient["OAR ref"][0]["Structure volume"] = volume
    runtime.patient_case = job.patient_case
    runtime.master_structure_reference_dict[job.patient_case.patient_uid] = patient
    metadata = {**job.metadata, "worker_job_id": job.job_id, "run_id": job.run_id,
                "patient_input_manifest_identity_sha256": job.patient_inputs.manifest_identity_sha256,
                "input_content_verified_before": True, "input_content_verified_after": True}
    checkpoint = write_anatomical_checkpoint(runtime_state=runtime, pipeline_config=_config(),
        output_dir=job.patient_output_root / "validation/anatomical", metadata=metadata)
    stages = (
        PatientStageResult.success("grid_preprocessing", metadata={"resolved_scientific_state": {
            "schema_version": "patient_grid_state_v1", "dose": {"dose_present": False}, "mr_adc": {"present": False},
        }}),
        PatientStageResult.success("anatomical_preprocessing", metadata={"anatomical_checkpoint_path": str(checkpoint)}),
        PatientStageResult.success("patient_artifact_writing"),
    )
    write_patient_run_manifest(PatientRunResult.from_stage_results(job.patient_case, job.patient_output_root, stages, metadata=metadata))
    result = PatientWorkerResult(job, "succeeded", 0.0, 0, metadata={"artifact_paths": []})
    write_patient_worker_result(result)
    return result


class IndependenceTests(unittest.TestCase):
    def test_entire_recipe_reuses_pair_and_parent_finalization_and_requires_all_comparisons(self):
        from validation.anatomical_pair import run_anatomical_pair

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = _base(root)
            launched = []

            def launch(path, **kwargs):
                launched.append(load_patient_worker_job(path).patient_case.patient_uid)
                return _emit_completed(path)

            def child(arguments, log_path, *, timeout):
                if Path(arguments[0]).name == "validate_patient_anatomical.py":
                    values = dict(zip(arguments[1::2], arguments[2::2]))
                    result = run_anatomical_pair(job_path=Path(values["--job"]), output_dir=Path(values["--output-dir"]), abs_tol=0, rel_tol=0, timeout_seconds=1)
                else:
                    result = compare_completed_anatomical_jobs(Path(arguments[-1]))
                self.assertTrue(result["passed"], result)

            with patch("patient_runner.process_runner.launch_worker_job_file", side_effect=launch), patch("validation.anatomical_independence._child", side_effect=child):
                report = run_anatomical_independence(input_case_manifest=base.input_case_manifest_path,
                    scientific_config_snapshot=base.scientific_config_snapshot_path,
                    run_compatibility_identity=base.run_compatibility_identity_path,
                    output_dir=root / "qualification", patient_uids=("A", "B", "C"), split_a=("A",))
            self.assertTrue(report["passed"], report)
            self.assertEqual(len(launched), 15)
            self.assertEqual(set(report["comparisons"]), {"forward", "reverse", "split_union"})
            for name, run in report["runs"].items():
                self.assertEqual(run["planned_order"], run["actual_order"])
                self.assertTrue((Path(run["output_root"]) / "manifests/run_manifest_index.json").is_file())

    def test_small_schedule_preserves_forward_reverse_and_disjoint_split(self):
        orders = qualification_orders(("B", "A", "C"), ("A",))
        self.assertEqual(orders, {"forward": ("B", "A", "C"), "reverse": ("C", "A", "B"), "split_b": ("B", "C"), "split_a": ("A",)})
        for first in ((), ("B", "A", "C"), ("A", "A"), ("absent",)):
            with self.assertRaises(ValueError):
                qualification_orders(("B", "A", "C"), first)

    def test_all_small_cohort_orders_and_splits_use_real_subprocesses(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = _base(root)
            worker = root / "synthetic_worker.py"
            worker.write_text("import sys\nfrom validation.test_anatomical_independence import _emit_completed\n_emit_completed(sys.argv[1])\n")

            def launch(path, **kwargs):
                return launch_worker_job_file(path, worker_script_path=worker, **kwargs)

            surfaces = []
            with patch("patient_runner.process_runner.launch_worker_job_file", side_effect=launch):
                for index, order in enumerate(itertools.permutations(("A", "B", "C"))):
                    plan = _arm_plan(base, str(index), order, root / "runs")
                    results = run_patient_process_plan(plan)
                    self.assertTrue(all(r.succeeded for r in results), results)
                    self.assertEqual(tuple(r.worker_job.patient_case.patient_uid for r in results), order)
                    surfaces.append([job.job_path for job in plan.worker_jobs])
                split = []
                for name, order in (("split_b", ("C", "B")), ("split_a", ("A",))):
                    plan = _arm_plan(base, name, order, root / "runs")
                    run_patient_process_plan(plan)
                    split.extend(job.job_path for job in plan.worker_jobs)
                surfaces.append(split)
            for index, paths in enumerate(surfaces[1:], 1):
                request = root / "compare" / str(index) / "request.json"
                write_new_json(request, {"reference_jobs": list(map(str, surfaces[0])), "candidate_jobs": list(map(str, paths))})
                report = compare_completed_anatomical_jobs(request)
                self.assertTrue(report["passed"], report)

    def test_matching_rejects_duplicates_missing_patients_and_wrong_attempt(self):
        with TemporaryDirectory() as temporary:
            base = _base(Path(temporary))
            paths = write_patient_worker_job_packets(base)
            for path in paths:
                _emit_completed(path)
            for candidate in (paths[:1], (paths[0], paths[0], paths[1])):
                with self.assertRaises(ValueError):
                    match_completed_patients(paths, candidate)
            result = base.worker_jobs[0].result_path
            payload = json.loads(result.read_text())
            payload["job"]["attempt_number"] += 1
            result.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                read_completed_patient(paths[0])

    def test_actual_numeric_and_stage_state_drift_fail(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = _base(root)
            left = _arm_plan(base, "left", ("A",), root)
            right = _arm_plan(base, "right", ("A",), root)
            paths = [write_patient_worker_job_packets(p)[0] for p in (left, right)]
            _emit_completed(paths[0])
            _emit_completed(paths[1], volume=2.0)
            request = root / "comparison/request.json"
            write_new_json(request, {"reference_jobs": [str(paths[0])], "candidate_jobs": [str(paths[1])]})
            report = compare_completed_anatomical_jobs(request)
            self.assertFalse(report["passed"], report)
            self.assertFalse(report["patients"][0]["numerical_passed"])
            manifest = right.worker_jobs[0].patient_output_root / "patient_run_manifest.json"
            payload = json.loads(manifest.read_text())
            payload["stages"][0]["metadata"]["resolved_scientific_state"]["dose"]["effective_lower_bound"] = 99
            manifest.write_text(json.dumps(payload))
            request2 = root / "state/request.json"
            write_new_json(request2, {"reference_jobs": [str(paths[0])], "candidate_jobs": [str(paths[1])]})
            self.assertFalse(compare_completed_anatomical_jobs(request2)["patients"][0]["state_passed"])

    def test_incompatible_environment_and_changed_input_content_fail_matching(self):
        for field in ("environment", "content"):
            with self.subTest(field=field), TemporaryDirectory() as temporary:
                root = Path(temporary)
                base = _base(root)
                left = _arm_plan(base, "left", ("A",), root)
                right = _arm_plan(base, "right", ("A",), root)
                left_path = write_patient_worker_job_packets(left)[0]
                _emit_completed(left_path)
                job = right.worker_jobs[0]
                metadata = dict(job.metadata)
                if field == "environment":
                    identity = RunCompatibilityIdentity.from_dict(metadata["run_compatibility_identity"])
                    metadata["run_compatibility_identity"] = replace(identity, runtime_environment_sha256="different", identity_sha256="").to_dict()
                else:
                    job.patient_inputs.rtstruct.write_bytes(b"changed synthetic content")
                    metadata["input_content_identity"] = capture_patient_input_content(job.patient_inputs)
                right = replace(right, worker_jobs=(replace(job, metadata=metadata),))
                right_path = write_patient_worker_job_packets(right)[0]
                _emit_completed(right_path)
                with self.assertRaises(ValueError):
                    match_completed_patients((left_path,), (right_path,))

    def test_worker_rejects_changed_content_before_runtime_construction(self):
        from patient_runner.process_runner import run_patient_worker_job

        with TemporaryDirectory() as temporary:
            base = _base(Path(temporary))
            job = base.worker_jobs[0]
            job.patient_inputs.rtstruct.write_bytes(b"changed after planning")
            with patch("config.rehydration.rehydrate_pipeline_scientific_config_snapshot", return_value=None), \
                    patch("patient_runner.process_runner._validate_worker_compatibility_identity", return_value={}), \
                    patch("patient_runner.runtime_builder.build_standalone_patient_runtime") as build:
                result = run_patient_worker_job(job)
            self.assertFalse(result.succeeded)
            self.assertEqual(result.metadata["failed_boundary"], "input_content_preflight")
            build.assert_not_called()

    def test_missing_checkpoint_and_missing_content_confirmation_fail_closed(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = _base(root)
            paths = write_patient_worker_job_packets(base)
            for path in paths:
                _emit_completed(path)
            checkpoint = base.worker_jobs[0].patient_output_root / "validation/anatomical/anatomical_checkpoint.json"
            checkpoint.rename(checkpoint.with_suffix(".missing"))
            request = root / "comparison/request.json"
            write_new_json(request, {"reference_jobs": list(map(str, paths)), "candidate_jobs": list(map(str, paths))})
            self.assertFalse(compare_completed_anatomical_jobs(request)["passed"])
            manifest = base.worker_jobs[1].patient_output_root / "patient_run_manifest.json"
            payload = json.loads(manifest.read_text())
            payload["metadata"].pop("input_content_verified_after")
            manifest.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "verification"):
                read_completed_patient(paths[1])

    def test_failed_pair_or_timeout_retains_failed_experiment_summary(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = _base(root)
            with patch("validation.anatomical_independence._child", side_effect=RuntimeError("synthetic timeout")):
                report = run_anatomical_independence(
                    input_case_manifest=base.input_case_manifest_path,
                    scientific_config_snapshot=base.scientific_config_snapshot_path,
                    run_compatibility_identity=base.run_compatibility_identity_path,
                    output_dir=root / "qualification", patient_uids=("A", "B", "C"), split_a=("A",),
                )
            self.assertFalse(report["passed"])
            self.assertTrue((root / "qualification/anatomical_independence_summary.json").is_file())
            self.assertEqual(report["runs"], {})


if __name__ == "__main__":
    unittest.main()
