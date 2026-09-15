"""Synthetic exact optimization checkpoint and executed-state integrity tests."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from config.production import build_production_pipeline_config
from patient_runner.optimization_execution import resolve_fixed_optimization_execution
from patient_runner.scientific_config_builder import _resolve_explicit_transform_counts
from patient_runner.scientific_stages import run_patient_preprocessing_scientific_stage
from random_seed_policy import build_transform_generation_patient_rng, build_optimizer_v1_patient_rng
from validation.anatomical_checkpoint import write_anatomical_checkpoint, compare_anatomical_checkpoints
from validation.optimization_checkpoint_fields import (
    TRANSFORM_FIELDS, UNCERTAINTY_FIELDS, V2_TABLES, V2_TIMING_COLUMNS,
    OPTIMIZER_V1_DIL_OUTPUT_KEYS, OPTIMIZER_V1_MULTI_STRUCTURE_INFORMATION_KEYS,
    OPTIMIZER_V1_MULTI_STRUCTURE_PREPROCESSING_KEYS,
)
from validation.test_biopsy_preprocessing import biopsy_runtime, biopsy_science, stage_config

BOUNDARY = 'optimization_shadow'


def optimization_fixture():
    config = build_production_pipeline_config()
    v2 = config.optimizer.optimizer_v2
    config = replace(config, optimizer=replace(config.optimizer, optimizer_v2=replace(v2,
        capacity=replace(v2.capacity, max_test_structures_per_call=100000))),
        mc=replace(config.mc, counts=replace(config.mc.counts,
            num_mc_containment_simulations_input=4, num_mc_dose_simulations_input=4,
            num_mc_mr_simulations_input=4)))
    runtime = biopsy_runtime()
    with biopsy_science():
        run_patient_preprocessing_scientific_stage(runtime, None, scientific_config=stage_config())
    patient = runtime.pydicom_item
    for family in config.structure_registry.structs_referenced_list:
        patient.setdefault(family, [])
    patient[config.legacy_refs.bx_ref][1]['Simulated type'] = config.biopsy.simulated.optimizer_simulated_type
    patient[config.legacy_refs.bx_ref][1]['Transport family'] = 'identity'
    n = _resolve_explicit_transform_counts(config)[1]
    for family in config.structure_registry.structs_referenced_list:
        for index, record in enumerate(patient[family]):
            record.update({'Index number': index, 'Struct type': family})
            for spec in TRANSFORM_FIELDS:
                if len(spec.shape) == 1 and family != config.legacy_refs.bx_ref:
                    continue
                shape = tuple(n if size is None else size for size in spec.shape)
                record[spec.key] = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) / 100
            record['Uncertainty data'] = SimpleNamespace(
                **{spec.key: np.zeros(spec.shape) for spec in UNCERTAINTY_FIELDS},
                uncertainty_data_info_dict={'Frame of reference': 'Biopsy' if family == config.legacy_refs.bx_ref else 'Lab',
                                            'Distribution': 'Normal'})
    table = pd.DataFrame({'ROI': pd.Categorical(['target']), 'score': [0.75], 'rank': np.array([1], dtype=np.int32)})
    for record in patient[config.legacy_refs.dil_ref]:
        for key in OPTIMIZER_V1_DIL_OUTPUT_KEYS:
            record[key] = np.ones((2, 3)) if key.endswith('only in dil') else table.copy(deep=True)
    all_ref = patient[config.legacy_refs.all_ref_key]
    all_ref['Multi-structure information dict (not for csv output)'] = {
        key: table.copy(deep=True) for key in OPTIMIZER_V1_MULTI_STRUCTURE_INFORMATION_KEYS}
    tables = all_ref['Multi-structure pre-processing output dataframes dict']
    tables.update({key: table.copy(deep=True) for key in OPTIMIZER_V1_MULTI_STRUCTURE_PREPROCESSING_KEYS})
    selection = {'Target optimizer selected X': 1.0, 'Target optimizer selected Y': 2.0,
                 'Target optimizer selected Z': 3.0, 'Target optimizer score': 0.875,
                 'Target optimizer final stage total elapsed seconds': 12.5}
    tables.update({key: pd.DataFrame([selection]) for key in V2_TABLES})
    patient[config.legacy_refs.bx_ref][1]['Simulated biopsy transport request dict'] = {
        'Transport family': 'identity', 'Target vector': np.array([1., 2., 3.]),
        'Transport source': 'target_dil_optimizer_v2', 'Selection metadata': dict(selection)}
    with patch('random_seed_policy._cupy_module', return_value=np):
        _, transform = build_transform_generation_patient_rng({}, runtime.patient_uid,
            transform_generation_random_seed=config.random_seeds.transform_generation_random_seed)
        _, v1 = build_optimizer_v1_patient_rng({}, runtime.patient_uid,
            optimizer_v1_random_seed=config.random_seeds.optimizer_v1_random_seed)
    budget, chunk = resolve_fixed_optimization_execution(config)
    state = {'transform_generation': {**transform, 'num_generated_transform_samples': n},
             'optimization': {**{'optimizer_v1_' + key: value for key, value in v1.items()},
                 'optimizer_v1_dil_count': len(patient[config.legacy_refs.dil_ref]),
                 'optimizer_v2_target_structure_count': 1,
                 'optimizer_v2_resolved_max_test_structures_per_call': budget,
                 'optimizer_v2_resolved_max_candidates_per_chunk': chunk}}
    return config, runtime, state


class OptimizationCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.config, self.runtime, self.state = optimization_fixture()
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def capture(self, name, runtime=None, state=None):
        return write_anatomical_checkpoint(
            runtime_state=self.runtime if runtime is None else runtime,
            pipeline_config=self.config, output_dir=self.root / name,
            checkpoint_name=BOUNDARY, optimization_state=self.state if state is None else state)

    def compare(self, left, right):
        return compare_anatomical_checkpoints(left, right, abs_tol=0, rel_tol=0, checkpoint_name=BOUNDARY)

    def test_complete_exact_copy_and_numpy_dataframes_round_trip(self):
        left = self.capture('left')
        report = self.compare(left, self.capture('right', deepcopy(self.runtime)))
        self.assertTrue(report['passed'], report.get('errors'))
        self.assertTrue(report['coverage']['reference']['optimization']['complete'])
        self.assertEqual(json.loads(left.read_text())['schema_version'], 'optimization_checkpoint_v1')
        self.assertTrue(report['coverage']['reference']['biopsies']['complete'])

    def test_transform_v1_v2_request_and_rank_perturbations_fail(self):
        left = self.capture('left')
        refs = self.config.legacy_refs
        for number, mutation in enumerate((
            lambda p: p[refs.bx_ref][0][TRANSFORM_FIELDS[0].key].__setitem__((0, 0), 0.001),
            lambda p: p[refs.dil_ref][0][OPTIMIZER_V1_DIL_OUTPUT_KEYS[0]].loc.__setitem__((0, 'score'), .5),
            lambda p: p[refs.bx_ref][1]['Simulated biopsy transport request dict']['Target vector'].__setitem__(0, 9.),
            lambda p: p[refs.all_ref_key]['Multi-structure pre-processing output dataframes dict'][V2_TABLES[1]].loc.__setitem__((0, 'Target optimizer score'), .25),
        )):
            changed = deepcopy(self.runtime)
            mutation(changed.pydicom_item)
            self.assertFalse(self.compare(left, self.capture('changed-' + str(number), changed))['passed'])

    def test_timing_and_runtime_caches_do_not_affect_scientific_equality(self):
        left = self.capture('left')
        patient = self.runtime.pydicom_item
        patient['cache'] = object()
        refs = self.config.legacy_refs
        tables = patient[refs.all_ref_key]['Multi-structure pre-processing output dataframes dict']
        for key in V2_TABLES:
            tables[key]['Target optimizer final stage total elapsed seconds'] = 999.0
        request = patient[refs.bx_ref][1]['Simulated biopsy transport request dict']
        request['Selection metadata']['Target optimizer final stage total elapsed seconds'] = 999.0
        report = self.compare(left, self.capture('right'))
        self.assertTrue(report['passed'], report.get('errors'))
        self.assertEqual(request['Selection metadata']['Target optimizer final stage total elapsed seconds'], 999.)

    def test_missing_transform_optimizer_outputs_and_unsealed_execution_fail(self):
        refs = self.config.legacy_refs
        for number, mutation in enumerate((
            lambda p: p[refs.bx_ref][0].pop(TRANSFORM_FIELDS[1].key),
            lambda p: p[refs.dil_ref][0].pop(OPTIMIZER_V1_DIL_OUTPUT_KEYS[0]),
            lambda p: p[refs.bx_ref][1].pop('Simulated biopsy transport request dict'),
            lambda p: p[refs.all_ref_key]['Multi-structure pre-processing output dataframes dict'].pop(V2_TABLES[2]),
        )):
            changed = deepcopy(self.runtime)
            mutation(changed.pydicom_item)
            with self.assertRaisesRegex(ValueError, 'incomplete optimization'):
                self.capture('missing-' + str(number), changed)
        changed = deepcopy(self.state)
        changed['optimization']['optimizer_v2_resolved_max_candidates_per_chunk'] += 1
        with self.assertRaisesRegex(ValueError, 'incomplete optimization'):
            self.capture('changed-budget', state=changed)
        with self.assertRaisesRegex(ValueError, 'incomplete optimization'):
            self.capture('missing-execution', state={})

    def test_gpu_array_copy_happens_only_at_capture_boundary(self):
        array = self.runtime.pydicom_item[self.config.legacy_refs.bx_ref][0][TRANSFORM_FIELDS[0].key]
        device = SimpleNamespace(__cuda_array_interface__={'shape': array.shape}, host=array)
        self.runtime.pydicom_item[self.config.legacy_refs.bx_ref][0][TRANSFORM_FIELDS[0].key] = device
        with patch.dict('sys.modules', {'cupy': SimpleNamespace(asnumpy=lambda value: value.host.copy())}):
            left = self.capture('gpu')
        self.assertIs(self.runtime.pydicom_item[self.config.legacy_refs.bx_ref][0][TRANSFORM_FIELDS[0].key], device)
        self.runtime.pydicom_item[self.config.legacy_refs.bx_ref][0][TRANSFORM_FIELDS[0].key] = array
        self.assertTrue(self.compare(left, self.capture('host'))['passed'])

    def test_tolerance_cannot_be_relaxed(self):
        with self.assertRaisesRegex(ValueError, 'exact'):
            compare_anatomical_checkpoints(self.root / 'missing', self.root / 'missing',
                                           abs_tol=1e-9, rel_tol=0, checkpoint_name=BOUNDARY)

    def test_table_row_order_is_semantic_and_hook_records_actual_execution(self):
        from patient_runner.contracts import PatientStageResult
        from patient_runner.runner import PatientStage
        from validation.anatomical_execution import with_preprocessing_checkpoint

        stages = (
            PatientStage('transform_generation', lambda runtime, config: PatientStageResult.success(
                'transform_generation', metadata=self.state['transform_generation'])),
            PatientStage('optimization', lambda runtime, config: PatientStageResult.success(
                'optimization', metadata={**self.state['optimization'], 'steps': ['optimizer_v1', 'optimizer_v2']})),
        )
        wrapped = with_preprocessing_checkpoint(stages, self.config, checkpoint_name=BOUNDARY)
        execution_config = SimpleNamespace(patient_output_dir=lambda case: self.root / 'hook')
        for stage in wrapped:
            result = stage.runner(self.runtime, execution_config)
        left = Path(result.metadata['optimization_checkpoint_path'])
        self.assertTrue(self.compare(left, self.capture('direct'))['passed'])
        tables = self.runtime.pydicom_item[self.config.legacy_refs.all_ref_key]['Multi-structure pre-processing output dataframes dict']
        tables[V2_TABLES[1]] = pd.concat([tables[V2_TABLES[1]], tables[V2_TABLES[1]]], ignore_index=True)
        tables[V2_TABLES[1]].loc[1, 'Target optimizer score'] = .25
        first = self.capture('ranked')
        tables[V2_TABLES[1]] = tables[V2_TABLES[1]].iloc[::-1].reset_index(drop=True)
        self.assertFalse(self.compare(first, self.capture('reordered'))['passed'])


if __name__ == '__main__':
    unittest.main()
