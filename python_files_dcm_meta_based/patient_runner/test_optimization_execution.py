"""CPU contracts for optimizer admission, explicit RNG and worker/reference dispatch."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch, Mock

import numpy as np

from config.production import build_production_pipeline_config
from config.snapshots import build_pipeline_scientific_config_snapshot, write_pipeline_config_snapshot
from patient_runner.contracts import PatientRunResult, PatientStageResult
from patient_runner.manifests import write_patient_run_manifest
from patient_runner.optimization_execution import resolve_fixed_optimization_execution
from patient_runner.process_runner import (
    build_patient_process_run_plan, write_patient_worker_job_packets, load_patient_worker_job,
    run_patient_worker_job, PatientWorkerResult,
)
from patient_runner.scientific_dependencies import (
    standalone_pathway_supported, executable_patient_scientific_pathway_stage_names,
)
from patient_runner.scientific_config_builder import _build_preprocessing_config, PatientRunnerScientificConfigBuildContext
from patient_runner.test_process_runner import _write_case_manifest, _write_current_compatibility_identity
from preprocessing.test_patient_uncertainty import load_functions, ROOT, uncertainty_science, uncertainty_patient
from preprocessing.patient_uncertainty import prepare_patient_uncertainty_data
from random_seed_policy import build_transform_generation_patient_rng, build_optimizer_v1_patient_rng
from validation.anatomical_checkpoint import write_anatomical_checkpoint
from validation.anatomical_pair import run_anatomical_pair
from validation.preprocessing_boundary import requested_checkpoint_captures
from validation.test_optimization_checkpoint import optimization_fixture, TRANSFORM_FIELDS

BOUNDARY = 'optimization_shadow'


class OptimizationExecutionTests(unittest.TestCase):
    def test_dependency_slice_and_live_allowlist(self):
        self.assertEqual(tuple(s.value for s in executable_patient_scientific_pathway_stage_names(BOUNDARY)),
                         ('grid_preprocessing', 'anatomical_preprocessing', 'preprocessing', 'transform_generation', 'optimization'))
        for name in ('anatomical_qa', 'biopsy_preprocessing_shadow', BOUNDARY):
            self.assertTrue(standalone_pathway_supported(name, name))
            self.assertFalse(standalone_pathway_supported(name, 'other'))
        for name in ('post_optimizer_biopsy_realization_shadow', 'sampling_classification_shadow',
                     'current_dosimetry_shadow', 'full_current_pipeline_shadow', 'unknown'):
            self.assertFalse(standalone_pathway_supported(name, name))
        self.assertEqual(requested_checkpoint_captures({'capture_validation_checkpoint': True}, BOUNDARY), (BOUNDARY,))
        self.assertEqual(requested_checkpoint_captures({'capture_anatomical_checkpoint': True}, 'anatomical_qa'), ('anatomical_qa',))
        with self.assertRaises(TypeError):
            requested_checkpoint_captures({'capture_validation_checkpoint': 'yes'}, BOUNDARY)
        with self.assertRaises(ValueError):
            requested_checkpoint_captures({'capture_biopsy_preprocessing_checkpoint': True}, BOUNDARY)

    def test_fixed_capacity_is_required_and_can_change_adaptive_trial_prefix(self):
        production = build_production_pipeline_config()
        with self.assertRaisesRegex(ValueError, 'explicit positive integer'):
            resolve_fixed_optimization_execution(production)
        config, _, _ = optimization_fixture()
        budget, chunk = resolve_fixed_optimization_execution(config)
        self.assertGreater(budget, 0)
        self.assertGreater(chunk, 0)
        for seed_name in ('transform_generation_random_seed', 'optimizer_v1_random_seed'):
            unseeded = replace(config, random_seeds=replace(config.random_seeds, **{seed_name: None}))
            with self.assertRaisesRegex(ValueError, 'explicit transform and optimizer-v1 seeds'):
                resolve_fixed_optimization_execution(unseeded)
        adaptive = production.optimizer.optimizer_v2.search_config.adaptive_block_config
        args = dict(current_trial_prefix=0, active_candidate_count=10, max_candidates_per_chunk=10, include_nominal=True)
        first = adaptive.resolve_capacity_packed_trial_prefix(**args, max_test_structures_per_call=170)
        second = adaptive.resolve_capacity_packed_trial_prefix(**args, max_test_structures_per_call=330)
        self.assertEqual((first, second), (16, 32))

    def test_generated_uncertainty_is_opt_in_to_transform_path_and_preserves_resolved_attachment(self):
        production = build_production_pipeline_config()
        context = PatientRunnerScientificConfigBuildContext()
        self.assertIsNone(_build_preprocessing_config(production, context).uncertainty_preparation)
        prepared = _build_preprocessing_config(production, context, generate_uncertainty=True)
        self.assertEqual(prepared.uncertainty_preparation.policy, production.preprocessing.uncertainty)
        explicit = replace(context, read_uncertainties_dataframe=object(), uncertainty_data_cls=object)
        resolved = _build_preprocessing_config(production, explicit, generate_uncertainty=True)
        self.assertIsNone(resolved.uncertainty_preparation)
        self.assertIs(resolved.uncertainty_attachment.read_uncertainties_dataframe, explicit.read_uncertainties_dataframe)

    def test_real_transform_generator_uses_patient_streams_and_preserves_global_rng(self):
        # Real generator bodies; a NumPy backend substitutes for CUDA, not for the
        # algorithm. This proves stream ownership, not NumPy/CuPy bit equivalence.
        config = build_production_pipeline_config()
        cp = SimpleNamespace(array=np.array, asnumpy=np.asarray,
                             random=SimpleNamespace(RandomState=np.random.RandomState))
        from preprocessing.biopsy_processing.simulated_biopsy_preparation import get_biopsy_length_for_mc_preparation_mm

        functions = load_functions('cupy_functions', {
            '_get_rng', 'MC_simulator_all_structs_dilations_generator_cupy',
            'MC_simulator_all_structs_rotations_generator_cupy',
            'MC_simulator_shift_all_structures_generator_cupy',
            'MC_simulator_shift_biopsy_structures_uniform_generator_cupy',
        }, {'cp': cp, 'get_biopsy_length_for_mc_preparation_mm': get_biopsy_length_for_mc_preparation_mm})
        spec = importlib.util.spec_from_file_location('synthetic_transform_generator', ROOT / 'mc/prep/per_patient/transform_generation.py')
        generator = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'cupy': cp, 'cupy_functions': functions}):
            spec.loader.exec_module(generator)
        def run(uid):
            patient = uncertainty_patient(config)
            patient[config.legacy_refs.bx_ref][0]['Reconstructed biopsy cylinder length (from contour data)'] = 17.0
            with uncertainty_science():
                prepare_patient_uncertainty_data(patient_uid=uid, pydicom_item=patient,
                    master_structure_info_dict={}, structs_referenced_list=config.structure_registry.structs_referenced_list,
                    structs_referenced_dict=config.structure_registry.structs_referenced_dict, policy=config.preprocessing.uncertainty)
            with patch('random_seed_policy._cupy_module', return_value=cp):
                rng, metadata = build_transform_generation_patient_rng({}, uid,
                    transform_generation_random_seed=config.random_seeds.transform_generation_random_seed)
                v1, v1_metadata = build_optimizer_v1_patient_rng({}, uid,
                    optimizer_v1_random_seed=config.random_seeds.optimizer_v1_random_seed)
            generator.generate_transformations_for_patient(
                patient_uid=uid, pydicom_item=patient,
                simulate_uniform_bx_shifts_due_to_bx_needle_compartment=True,
                bx_ref=config.legacy_refs.bx_ref, biopsy_needle_compartment_length=20.,
                num_generated_transform_samples=7,
                structs_referenced_list=config.structure_registry.structs_referenced_list, rng=rng)
            arrays = [patient[f][0][spec.key] for f in config.structure_registry.structs_referenced_list for spec in TRANSFORM_FIELDS[:3]]
            uniform = patient[config.legacy_refs.bx_ref][0][TRANSFORM_FIELDS[3].key]
            self.assertTrue(((uniform >= 0) & (uniform < 3)).all())
            arrays.append(uniform)
            return arrays, metadata, v1.normal(size=7), v1_metadata
        before = np.random.get_state()
        a, b = run('synthetic-A'), run('synthetic-B')
        b_again, a_again = run('synthetic-B'), run('synthetic-A')
        for first, second in ((a, a_again), (b, b_again)):
            for left, right in zip(first[0], second[0]):
                np.testing.assert_array_equal(left, right)
                self.assertEqual(left.shape[0], 7)
            self.assertEqual(first[1], second[1])
            np.testing.assert_array_equal(first[2], second[2])
            self.assertEqual(first[3], second[3])
        self.assertNotEqual(a[1]['transform_generation_resolved_patient_seed'], b[1]['transform_generation_resolved_patient_seed'])
        after = np.random.get_state()
        for left, right in zip(before, after):
            np.testing.assert_array_equal(left, right)

    def make_job(self, root, config):
        snapshot = root / 'config.json'
        write_pipeline_config_snapshot(build_pipeline_scientific_config_snapshot(config), snapshot)
        plan = build_patient_process_run_plan(
            input_case_manifest_path=_write_case_manifest(root, ('synthetic-001',), create_core_files=True),
            scientific_config_snapshot_path=snapshot, run_compatibility_identity_path=_write_current_compatibility_identity(root, snapshot),
            output_root=root / 'source', pathway_name=BOUNDARY, checkpoint_name=BOUNDARY,
            execution_mode='live_workers', capture_input_content=True)
        path = write_patient_worker_job_packets(plan)[0]
        return plan.worker_jobs[0], path

    def test_unfixed_capacity_fails_before_runtime_and_reference_dispatch_uses_independent_builder(self):
        with TemporaryDirectory() as directory:
            job, path = self.make_job(Path(directory), build_production_pipeline_config())
            runtime_builder = Mock(side_effect=AssertionError('science must not run'))
            result = run_patient_worker_job(job, runtime_builder=runtime_builder)
            self.assertFalse(result.succeeded)
            self.assertEqual(result.metadata['failed_boundary'], 'scientific_execution_contract_preflight')
            runtime_builder.assert_not_called()
            from run_patient_anatomical_reference import main
            from validation.anatomical_execution import build_legacy_input_anatomical_runtime
            with patch('patient_runner.process_runner.run_patient_worker_job', return_value=SimpleNamespace(exit_code=0)) as run, \
                    patch('patient_runner.process_runner.write_patient_worker_result'):
                self.assertEqual(main([str(path)]), 0)
            self.assertIs(run.call_args.kwargs['runtime_builder'], build_legacy_input_anatomical_runtime)

    def test_paired_optimization_reuses_completed_manifest_and_input_seals(self):
        config, _, _ = optimization_fixture()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, path = self.make_job(root, config)
            lanes = []
            def launch(path, **kwargs):
                job = load_patient_worker_job(path)
                self.assertTrue(job.metadata['capture_validation_checkpoint'])
                lanes.append(kwargs.get('worker_script_path'))
                _, runtime, state = optimization_fixture()
                metadata = {**job.metadata, 'worker_job_id': job.job_id, 'run_id': job.run_id,
                    'patient_input_manifest_identity_sha256': job.patient_inputs.manifest_identity_sha256,
                    'input_content_verified_before': True, 'input_content_verified_after': True}
                write_anatomical_checkpoint(runtime_state=runtime, pipeline_config=config,
                    output_dir=job.patient_output_root / 'validation/optimization', metadata=metadata,
                    checkpoint_name=BOUNDARY, optimization_state=state)
                stages = tuple(PatientStageResult.success(name, metadata={'resolved_scientific_state':
                    {'schema_version': 'patient_grid_state_v1', 'dose': {}, 'mr_adc': {}}} if name == 'grid_preprocessing' else {})
                    for name in (*job.metadata['planned_stage_names'], 'patient_artifact_writing'))
                write_patient_run_manifest(PatientRunResult.from_stage_results(job.patient_case, job.patient_output_root, stages, metadata=metadata))
                return PatientWorkerResult(job, 'succeeded', 0., 0, metadata={'artifact_paths': []})
            with patch('patient_runner.process_runner.launch_worker_job_file', side_effect=launch):
                report = run_anatomical_pair(job_path=path, output_dir=root / 'pair', abs_tol=0, rel_tol=0,
                                             timeout_seconds=10., checkpoint_name=BOUNDARY)
            self.assertTrue(report['passed'], report)
            self.assertTrue(report['resolved_state_passed'])
            self.assertIsNone(lanes[0])
            self.assertEqual(lanes[1].name, 'run_patient_anatomical_reference.py')
            source.patient_inputs.rtstruct.write_bytes(b'changed synthetic bytes')
            with patch('patient_runner.process_runner.launch_worker_job_file') as run:
                with self.assertRaisesRegex(ValueError, 'content'):
                    run_anatomical_pair(job_path=path, output_dir=root / 'changed-input', abs_tol=0, rel_tol=0,
                                         timeout_seconds=10., checkpoint_name=BOUNDARY)
                run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
