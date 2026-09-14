"""Synthetic biopsy adapter/checkpoint parity with bounded geometry substitutes.

Actual patient target/multiplicity/length preparation and planning orchestration
run. Native reconstruction and containment are deterministic substitutes, not
claimed algorithm truth. The separate geometry characterization tests the real
reconstruction helper's existing allocation defect.
"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, make_dataclass, replace
import importlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from config.snapshots import build_pipeline_scientific_config_snapshot, write_pipeline_config_snapshot
from patient_runner.contracts import PatientRunResult, PatientStageResult
from patient_runner.manifests import write_patient_run_manifest
from patient_runner.process_runner import (
    build_patient_process_run_plan, load_patient_worker_job, PatientWorkerResult,
    write_patient_worker_job_packets,
)
from patient_runner.scientific_config import (
    PatientPreprocessingScientificConfig, PatientRealBiopsyProcessingStageConfig,
    PatientSimulatedBiopsyPlanningStageConfig, PatientSimulatedBiopsyPreparationStageConfig,
    PatientRunnerScientificConfig, PatientScientificStageResources,
)
from patient_runner.scientific_stages import run_patient_preprocessing_scientific_stage
from patient_runner.test_process_runner import _write_case_manifest, _write_compatibility_identity
from patient_runner.worker_resources import SequentialWorkerPool
from input_data.content_identity import verify_patient_input_content
from validation.anatomical_checkpoint import write_anatomical_checkpoint, compare_anatomical_checkpoints
from validation.anatomical_pair import run_anatomical_pair
from validation.test_anatomical_checkpoint import _config, _runtime, _patient, _record


BOUNDARY = "biopsy_preprocessing_shadow"


def biopsy_config():
    config = _config()
    values = {**asdict(config.legacy_refs), "bx_ref": "Bx ref"}
    config.legacy_refs = make_dataclass("BiopsyRefs", [(key, str) for key in values])(**values)
    return config


def _model(slices, radius=0.5):
    points = np.vstack(slices)
    centers = np.asarray([row.mean(axis=0) for row in slices])
    line = centers[[0, -1]]
    vector = line[1] - line[0]
    length = float(np.linalg.norm(vector))
    return {
        "Raw contour pts": points, "Structure centroid pts": centers,
        "Structure global centroid": centers.mean(axis=0, keepdims=True),
        "Reconstructed biopsy cylinder length (from contour data)": length,
        "Best fit line of centroid pts": line,
        "Centroid line unit vec (bx needle base to bx needle tip)": vector / length,
        "Centroid line vec (bx needle base to bx needle tip)": vector,
        "Centroid line vec length (bx needle base to bx needle tip)": length,
        "Centroid line sample pts": np.linspace(line[0], line[1], 4),
        "Reconstructed structure pts arr": points,
        "Reconstructed structure delaunay global": SimpleNamespace(delaunay_triangulation=SimpleNamespace(
            points=points, simplices=np.array([[0, 1, 2, 3]], dtype=np.int32))),
        "Centroid variation arr": np.zeros(len(slices)), "Mean centroid variation": 0.,
        "Maximum projected distance between original centroids": 0.,
        "Distance between centroid sample rings": length / (len(slices) - 1),
        "Rotated reconstructed structure pts arr rounded": points,
        "Rotated reconstructed structure z values": centers[:, 2],
        "Rotated reconstructed structure zslice list": slices,
        "Biopsy coord sys origin translation vec": -line[0],
        "Centroid line to z axis rotation matrix": np.eye(3),
    }


def biopsy_runtime():
    runtime = _runtime()
    patient = _patient(runtime)
    ring = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    real = {"ROI": "real-1", "Ref #": 1, "Index number": 0, "Struct type": "Bx ref",
            "Simulated bool": False, "Simulated type": "Real",
            "Raw contour pts zslice list": [ring, ring + [0., 0., 2.]]}
    patient["Bx ref"] = [real, {"ROI": "planned-1", "Ref #": "sim-1", "Index number": 1,
        "Struct type": "Bx ref", "Simulated bool": True, "Simulated type": "DIL centroid",
        "Relative structure type": "DIL ref", "Relative structure ref #": 2}]
    dil = deepcopy(_record(runtime))
    dil.update({"ROI": "target-2", "Ref #": 2, "Struct type": "DIL ref", "Structure global centroid": np.zeros((1, 3))})
    patient["DIL ref"] = [dil]
    patient["All ref"] = {"Multi-structure pre-processing output dataframes dict": {}}
    runtime.patient_uid = runtime.patient_case.patient_uid
    runtime.pydicom_item = patient
    runtime.bx_ref, runtime.all_ref_key = "Bx ref", "All ref"
    runtime.master_structure_info_dict = {"By patient": {runtime.patient_uid: {"Bx ref": {}}}}
    runtime.metadata = {}
    return runtime


@contextmanager
def biopsy_science():
    """Keep real per-patient functions, replacing only expensive leaf producers."""
    package_name = "preprocessing.biopsy_processing.per_patient"
    package = ModuleType(package_name)
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "preprocessing/biopsy_processing/per_patient")]

    def finalize(reference, uid, bx_ref, index, record, slices, *args, **kwargs):
        assert reference[uid][bx_ref][index] is record
        record.update(_model(slices))
        record["Structure volume"] = 2.
        record["Equal num zslice contour pts"] = slices

    def rings(direction, origin, count, step, radius, plot):
        ring = np.array([[0., 0., 0.], [radius, 0., 0.], [0., radius, 0.]])
        return [ring + np.asarray(origin) + i * step * np.asarray(direction) for i in range(count)]

    sampler = SimpleNamespace(sample_biopsy_points_from_reconstructed_global_delaunay_convex_structure=
        lambda spacing, triangulation, points, uid, ref, index, rotation:
            (points[:2].copy(), points.copy(), 2, {"Patient UID": uid, "Structure type": ref, "Specific structure index": index}))
    modules = {package_name: package,
        "preprocessing.biopsy_processing.biopsy_geometry_helper": SimpleNamespace(
            finalize_biopsy_geometry_from_zslice_list=finalize,
            build_reconstructed_biopsy_model_for_sampling_from_zslice_list=_model),
        "biopsy_creator": SimpleNamespace(biopsy_points_creater_by_transport_for_sim_bxs=rings),
        "sampling": SimpleNamespace(biopsy_point_sampler=sampler)}
    with patch.dict(sys.modules, modules):
        # Do not let a previous import bind a different synthetic producer set.
        for name in (package_name + ".real_biopsy_processing", package_name + ".simulated_biopsy_preparation",
                     package_name + ".simulated_biopsy_planning", "preprocessing.biopsy_processing.simulated_biopsy_planner"):
            sys.modules.pop(name, None)
        real = importlib.import_module(package_name + ".real_biopsy_processing")
        prep = importlib.import_module(package_name + ".simulated_biopsy_preparation")
        plan = importlib.import_module(package_name + ".simulated_biopsy_planning")
        package.process_patient_real_biopsies = real.process_patient_real_biopsies
        package.prepare_patient_simulated_biopsies = prep.prepare_patient_simulated_biopsies
        package.plan_patient_simulated_biopsies = plan.plan_patient_simulated_biopsies
        with patch.object(prep, "finalize_patient_simulated_biopsy_preparation_dataframe_for_export",
                          side_effect=lambda **kw: kw["pydicom_item"]["All ref"]["Multi-structure pre-processing output dataframes dict"]["Simulated biopsy preparation dataframe"]):
            yield package


def stage_config(method="match real"):
    real = PatientRealBiopsyProcessingStageConfig({}, 1., 1., 1., 0.5, False, 1., 1., 100, 100,
                                                 False, "synthetic", True, False, "synthetic", False, False, False)
    return PatientRunnerScientificConfig(
        resources=PatientScientificStageResources(parallel_pool=SequentialWorkerPool()),
        preprocessing=PatientPreprocessingScientificConfig(real_biopsy_processing=real,
            simulated_biopsy_preparation=PatientSimulatedBiopsyPreparationStageConfig("DIL ref", method, 8.),
            simulated_biopsy_planning=PatientSimulatedBiopsyPlanningStageConfig(1., 0.5, num_centroids_for_sim_bxs=3)))


def prepared_runtime():
    runtime = biopsy_runtime()
    with biopsy_science():
        run_patient_preprocessing_scientific_stage(runtime, None, scientific_config=stage_config())
    return runtime


class BiopsyPreprocessingTests(unittest.TestCase):
    def capture(self, root, name, runtime):
        return write_anatomical_checkpoint(runtime_state=runtime, pipeline_config=biopsy_config(),
                                          output_dir=root / name, checkpoint_name=BOUNDARY)

    def test_real_patient_functions_match_adapter_and_capture_complete_products(self):
        left, right = biopsy_runtime(), biopsy_runtime()
        with biopsy_science() as functions:
            config = stage_config()
            run_patient_preprocessing_scientific_stage(left, None, scientific_config=config)
            real = asdict(config.preprocessing.real_biopsy_processing)
            real.pop("metadata")
            real["cupy_array_upper_limit_NxN_size_input"] = real.pop("cupy_array_upper_limit_nxn_size_input")
            functions.process_patient_real_biopsies(patient_uid=right.patient_uid, pydicom_item=right.pydicom_item,
                master_structure_reference_dict=right.master_structure_reference_dict, bx_ref="Bx ref",
                parallel_pool=SequentialWorkerPool(), **real)
            functions.prepare_patient_simulated_biopsies(patient_uid=right.patient_uid, pydicom_item=right.pydicom_item,
                bx_ref="Bx ref", dil_ref="DIL ref", all_ref_key="All ref", simulated_biopsy_length_method="match real",
                biopsy_needle_compartment_length=8., master_structure_info_dict=right.master_structure_info_dict)
            planning = asdict(config.preprocessing.simulated_biopsy_planning)
            planning.pop("metadata")
            functions.plan_patient_simulated_biopsies(patient_uid=right.patient_uid, pydicom_item=right.pydicom_item,
                                                     bx_ref="Bx ref", parallel_pool=SequentialWorkerPool(), **planning)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            report = compare_anatomical_checkpoints(self.capture(root, "left", left), self.capture(root, "right", right),
                                                   abs_tol=0, rel_tol=0, checkpoint_name=BOUNDARY)
            self.assertTrue(report["passed"], report)
            coverage = report["coverage"]["reference"]["biopsies"]
            self.assertEqual((coverage["real_count"], coverage["simulated_count"]), (1, 1))
            self.assertTrue(coverage["complete"])

    def test_patient_local_multiplicity_lengths_and_removed_cohort_methods(self):
        with biopsy_science():
            for method, expected in (("match real", [2., 5.]), ("full", [8., 8.])):
                runtime = biopsy_runtime()
                extra = deepcopy(runtime.pydicom_item["Bx ref"][0])
                extra.update({"ROI": "real-2", "Ref #": 3, "Index number": 2})
                extra["Raw contour pts zslice list"][-1][:, 2] = 5.
                runtime.pydicom_item["Bx ref"].append(extra)
                run_patient_preprocessing_scientific_stage(runtime, None, scientific_config=stage_config(method))
                simulated = [r for r in runtime.pydicom_item["Bx ref"] if r["Simulated bool"]]
                self.assertEqual([r["Simulated biopsy preparation dict"]["Nominal length mm"] for r in simulated], expected)
                self.assertEqual(len(simulated), 2)
            for method in ("real mean", "real normal"):
                with self.assertRaisesRegex(ValueError, "all-patient"):
                    run_patient_preprocessing_scientific_stage(biopsy_runtime(), None, scientific_config=stage_config(method))

    def test_changed_real_geometry_planned_samples_and_decisions_are_detected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            reference = self.capture(root, "reference", prepared_runtime())
            for index, mutate in enumerate((
                lambda records: records[0]["Reconstructed structure pts arr"].__setitem__((0, 0), 0.2),
                lambda records: records[1]["Simulated biopsy planning dict"]["Planned sampled volume pts arr"].__setitem__((0, 0), 0.2),
                lambda records: records[1]["Simulated biopsy preparation dict"].__setitem__("Length source", "changed"),
            )):
                candidate = prepared_runtime()
                mutate(candidate.pydicom_item["Bx ref"])
                report = compare_anatomical_checkpoints(reference, self.capture(root, str(index), candidate),
                                                       abs_tol=0, rel_tol=0, checkpoint_name=BOUNDARY)
                self.assertFalse(report["passed"])
            self.assertFalse(compare_anatomical_checkpoints(reference, reference, abs_tol=0, rel_tol=0)["passed"])

    def test_missing_or_unfinished_biopsy_products_fail_even_in_both_lanes(self):
        with TemporaryDirectory() as directory:
            for index, mutate in enumerate((
                lambda records: records.clear(),
                lambda records: records[0].pop("Reconstructed structure pts arr"),
                lambda records: records[1]["Simulated biopsy planning dict"].__setitem__("Planning complete", False),
                lambda records: records[1]["Simulated biopsy planning dict"].__setitem__("Planned sampled point count", 999),
            )):
                runtime = prepared_runtime()
                mutate(runtime.pydicom_item["Bx ref"])
                with self.assertRaisesRegex(ValueError, "incomplete biopsy"):
                    self.capture(Path(directory), str(index), runtime)
            runtime = prepared_runtime()
            tables = runtime.pydicom_item["All ref"]["Multi-structure pre-processing output dataframes dict"]
            tables["Simulated biopsy preparation dataframe"] = tables["Simulated biopsy preparation dataframe"].iloc[:0]
            with self.assertRaisesRegex(ValueError, "incomplete biopsy"):
                self.capture(Path(directory), "empty_table", runtime)

    def test_shared_paired_service_requires_bytes_completed_attempts_and_three_scientific_stages(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "config.json"
            write_pipeline_config_snapshot(build_pipeline_scientific_config_snapshot(biopsy_config()), snapshot)
            plan = build_patient_process_run_plan(input_case_manifest_path=_write_case_manifest(root, ("A",), create_core_files=True),
                scientific_config_snapshot_path=snapshot, run_compatibility_identity_path=_write_compatibility_identity(root, snapshot),
                output_root=root / "source", pathway_name=BOUNDARY, checkpoint_name=BOUNDARY,
                execution_mode="live_workers", capture_input_content=True)
            job_path = write_patient_worker_job_packets(plan)[0]
            self.assertEqual(plan.worker_jobs[0].metadata["planned_stage_names"],
                             ["grid_preprocessing", "anatomical_preprocessing", "preprocessing"])

            def launch(path, **kwargs):
                job = load_patient_worker_job(path)
                verify_patient_input_content(job.metadata["input_content_identity"], job.patient_inputs)
                runtime = prepared_runtime()
                runtime.master_structure_reference_dict = {job.patient_case.patient_uid: runtime.pydicom_item}
                runtime.patient_case = job.patient_case
                metadata = {**job.metadata, "worker_job_id": job.job_id, "run_id": job.run_id,
                    "patient_input_manifest_identity_sha256": job.patient_inputs.manifest_identity_sha256,
                    "input_content_verified_before": True, "input_content_verified_after": True}
                write_anatomical_checkpoint(runtime_state=runtime, pipeline_config=biopsy_config(),
                    output_dir=job.patient_output_root / "validation/biopsy_preprocessing",
                    metadata=metadata, checkpoint_name=BOUNDARY)
                stages = tuple(PatientStageResult.success(name, metadata={"resolved_scientific_state":
                    {"schema_version": "patient_grid_state_v1", "dose": {}, "mr_adc": {}}} if name == "grid_preprocessing" else {})
                    for name in (*job.metadata["planned_stage_names"], "patient_artifact_writing"))
                write_patient_run_manifest(PatientRunResult.from_stage_results(job.patient_case, job.patient_output_root, stages, metadata=metadata))
                return PatientWorkerResult(job, "succeeded", 0., 0, metadata={"artifact_paths": []})

            with patch("patient_runner.process_runner.launch_worker_job_file", side_effect=launch):
                report = run_anatomical_pair(job_path=job_path, output_dir=root / "pair", abs_tol=0, rel_tol=0,
                                             timeout_seconds=10., checkpoint_name=BOUNDARY)
            self.assertTrue(report["passed"], report)
            self.assertTrue(report["resolved_state_passed"])
            without = dict(plan.worker_jobs[0].metadata)
            without.pop("input_content_identity")
            job = replace(plan.worker_jobs[0], metadata=without)
            job.job_path.write_text(json.dumps(job.as_mapping()))
            with self.assertRaisesRegex(ValueError, "input content"):
                run_anatomical_pair(job_path=job_path, output_dir=root / "missing", abs_tol=0, rel_tol=0,
                                    timeout_seconds=10., checkpoint_name=BOUNDARY)
