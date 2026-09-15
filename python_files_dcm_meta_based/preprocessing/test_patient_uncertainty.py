"""CPU synthetic parity for the generated uncertainty producer and typed policy.

Legacy calculation/attachment bodies run unchanged. Only unrelated monolithic
imports are omitted; no fabricated patient files or GPU execution are needed.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import ast
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from config.production import build_production_pipeline_config
from config.uncertainty import UncertaintyPreparationConfig
from config.snapshots import build_pipeline_scientific_config_snapshot
from config.rehydration import rehydrate_pipeline_scientific_config_snapshot
from preprocessing.patient_uncertainty import prepare_patient_uncertainty_data

ROOT = Path(__file__).resolve().parents[1]


def load_functions(module_name, names, namespace=None):
    """Execute real selected legacy definitions without unrelated eager imports."""
    path = ROOT / (module_name + '.py')
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(nodes) != len(names):
        raise AssertionError('missing legacy function in ' + module_name)
    module = ModuleType(module_name)
    module.__dict__.update(namespace or {})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), module.__dict__)
    return module


@contextmanager
def uncertainty_science():
    modules = {
        'loading_tools': ModuleType('loading_tools'),
        'misc_tools': load_functions('misc_tools', {'specific_structure_info_dict_creator'}),
        'math_funcs': load_functions('math_funcs', {'add_in_quadrature'}, {'np': np}),
    }
    with patch.dict(sys.modules, modules):
        for name in ('uncertainty_file_writer', 'preprocessing.uncertainty_attachment'):
            spec = importlib.util.spec_from_file_location(name, ROOT / (name.replace('.', '/') + '.py'))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        yield sys.modules['preprocessing.uncertainty_attachment']


def uncertainty_patient(config):
    patient = {family: [] for family in config.structure_registry.structs_referenced_list}
    for family in patient:
        patient[family].append({'ROI': 'synthetic-' + family, 'Struct type': family,
                               'Ref #': 1, 'Index number': 0, 'Simulated bool': False,
                               'Simulated type': 'Real', 'Mean centroid variation': 0.73,
                               'Maximum projected distance between original centroids': 1.23})
    return patient


class PatientUncertaintyTests(unittest.TestCase):
    def test_exact_legacy_file_roundtrip_and_all_supported_biopsy_modes(self):
        config = build_production_pipeline_config()
        for mode in ('Default only', 'Per biopsy mean', 'Per biopsy max'):
            policy = replace(config.preprocessing.uncertainty, biopsy_variation_uncertainty_setting=mode)
            left, right = uncertainty_patient(config), uncertainty_patient(config)
            with uncertainty_science() as legacy, TemporaryDirectory() as directory:
                resolved = prepare_patient_uncertainty_data(
                    patient_uid='synthetic-patient', pydicom_item=left, master_structure_info_dict={},
                    structs_referenced_list=config.structure_registry.structs_referenced_list,
                    structs_referenced_dict=config.structure_registry.structs_referenced_dict, policy=policy)
                view = SimpleNamespace(start=lambda: None)
                _, _, expected, _ = legacy.prepare_and_attach_uncertainty_data(
                    {'synthetic-patient': right}, {}, {'Dataframes': {}},
                    config.structure_registry.structs_referenced_list,
                    config.structure_registry.structs_referenced_dict,
                    policy.biopsy_variation_uncertainty_setting, policy.non_biopsy_variation_uncertainty_setting,
                    policy.use_added_in_quad_errors_as, Path(directory), 'synthetic', '.csv',
                    False, Path(directory), None, None, view, legacy.uncertainty_data)
                pd.testing.assert_frame_equal(resolved, expected, check_exact=True)
                for family in left:
                    for attr, value in vars(left[family][0]['Uncertainty data']).items():
                        other = getattr(right[family][0]['Uncertainty data'], attr)
                        if isinstance(value, np.ndarray):
                            np.testing.assert_array_equal(value, other)
                        else:
                            self.assertEqual(value, other)

    def test_policy_is_sealed_and_illegal_choices_fail(self):
        config = build_production_pipeline_config()
        snapshot = build_pipeline_scientific_config_snapshot(config)
        restored = rehydrate_pipeline_scientific_config_snapshot(snapshot)
        self.assertEqual(restored.preprocessing.uncertainty, config.preprocessing.uncertainty)
        changed = replace(config, preprocessing=replace(config.preprocessing,
            uncertainty=replace(config.preprocessing.uncertainty, use_added_in_quad_errors_as='sigma')))
        self.assertNotEqual(snapshot.config_sha256, build_pipeline_scientific_config_snapshot(changed).config_sha256)
        for key, value in (('biopsy_variation_uncertainty_setting', 'Global mean'),
                           ('non_biopsy_variation_uncertainty_setting', 'arbitrary'),
                           ('use_added_in_quad_errors_as', 'three sigma')):
            with self.assertRaises(ValueError):
                UncertaintyPreparationConfig(**{key: value})

    def test_biopsy_variation_uses_planned_geometry_and_anatomy_defaults(self):
        config = build_production_pipeline_config()
        patient = uncertainty_patient(config)
        biopsy = patient[config.legacy_refs.bx_ref][0]
        biopsy.update({'Simulated bool': True, 'Simulated biopsy planning dict': {'Planned mean centroid variation': 3.0}})
        with uncertainty_science():
            prepare_patient_uncertainty_data(
                patient_uid='synthetic-patient', pydicom_item=patient, master_structure_info_dict={},
                structs_referenced_list=config.structure_registry.structs_referenced_list,
                structs_referenced_dict=config.structure_registry.structs_referenced_dict,
                policy=config.preprocessing.uncertainty)
        for family, record in patient.items():
            base = config.structure_registry.structs_referenced_dict[family]['Default sigma X']
            extra = [3.0] if family == config.legacy_refs.bx_ref else []
            expected = np.sqrt(np.square(base + extra).sum()) / 2
            self.assertAlmostEqual(record[0]['Uncertainty data'].uncertainty_data_sigma_arr[0], expected, places=14)

    def test_csv_patient_identity_coercion_is_explicit_failure(self):
        config = build_production_pipeline_config()
        with uncertainty_science(), self.assertRaisesRegex(ValueError, 'patient identity'):
            prepare_patient_uncertainty_data(
                patient_uid='001', pydicom_item=uncertainty_patient(config), master_structure_info_dict={},
                structs_referenced_list=config.structure_registry.structs_referenced_list,
                structs_referenced_dict=config.structure_registry.structs_referenced_dict,
                policy=config.preprocessing.uncertainty)


if __name__ == '__main__':
    unittest.main()
