"""Production config ownership contracts, without a second golden config file."""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from collections.abc import Mapping

from config.pipeline import (
    ArtifactConfig,
    BiopsyRuntimeConfig,
    GridPreprocessingConfig,
    LegacyReferenceConfig,
    MCCountsConfig,
    OptimizerV1RuntimeConfig,
    PipelineConfig,
    RuntimeUIConfig,
)
from config.production import build_production_pipeline_config
from config.rehydration import rehydrate_pipeline_scientific_config_snapshot
from config.snapshots import (
    build_pipeline_config_snapshot,
    build_pipeline_scientific_config_snapshot,
    read_pipeline_config_snapshot,
    write_pipeline_config_snapshot,
)


class ProductionConfigTests(unittest.TestCase):
    def test_deterministic_existing_root_and_generic_defaults(self):
        first = build_production_pipeline_config()
        second = build_production_pipeline_config()
        self.assertIsInstance(first, PipelineConfig)
        self.assertEqual(
            build_pipeline_config_snapshot(first).to_dict(),
            build_pipeline_config_snapshot(second).to_dict(),
        )
        for actual, default in (
            (first.ui, RuntimeUIConfig()),
            (first.artifacts, ArtifactConfig()),
            (first.legacy_refs, LegacyReferenceConfig()),
            (first.biopsy, BiopsyRuntimeConfig()),
            (first.grid_preprocessing, GridPreprocessingConfig()),
            (first.mc.counts, MCCountsConfig()),
            (first.optimizer.optimizer_v1, OptimizerV1RuntimeConfig()),
        ):
            self.assertEqual(actual, default)

    def test_tree_contains_only_pure_config_data(self):
        def check(value, path):
            if is_dataclass(value) and not isinstance(value, type):
                self.assertTrue(type(value).__dataclass_params__.frozen, path)
                for field in fields(value):
                    check(getattr(value, field.name), f'{path}.{field.name}')
            elif isinstance(value, Mapping):
                for key, item in value.items():
                    self.assertIsInstance(key, str, path)
                    check(item, f'{path}.{key}')
            elif isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    check(item, f'{path}[{index}]')
            else:
                self.assertIn(type(value), (str, int, float, bool, type(None)), path)
        check(build_production_pipeline_config(), 'PipelineConfig')

    def test_builds_do_not_share_mutable_compatibility_records(self):
        first = build_production_pipeline_config()
        second = build_production_pipeline_config()
        expected = build_pipeline_config_snapshot(second).to_dict()
        first.structure_registry.structs_referenced_dict[first.legacy_refs.bx_ref]['Default mu X'][0] = 9
        first.bootstrap.simulated_biopsies.locations.clear()
        first.optimizer.optimizer_v2.rendering.render_layer_style_by_name.clear()
        self.assertEqual(build_pipeline_config_snapshot(second).to_dict(), expected)
        self.assertEqual(build_pipeline_config_snapshot(build_production_pipeline_config()).to_dict(), expected)

    def test_assembly_preserves_shared_policy_and_registry_order(self):
        config = build_production_pipeline_config()
        self.assertIs(config.optimizer.optimizer_v2_search_config, config.optimizer.optimizer_v2.search_config)
        kernel = config.preprocessing.kernel_execution
        optimizer = config.optimizer.optimizer_v1
        for field in fields(kernel):
            self.assertEqual(getattr(kernel, field.name), getattr(optimizer, field.name))
        registry = config.structure_registry
        refs = config.legacy_refs
        self.assertEqual(tuple(registry.structs_referenced_dict), registry.structs_referenced_list_generalized)
        self.assertEqual(registry.structs_referenced_list[0], refs.bx_ref)
        self.assertEqual(
            registry.structs_referenced_list_generalized_unique_structs,
            tuple(key for key in registry.structs_referenced_list_generalized if key not in (refs.bx_ref, refs.dil_ref)),
        )
        for key, names in (
            (refs.bx_ref, config.bootstrap.contours.biopsy),
            (refs.oar_ref, config.bootstrap.contours.oar),
            (refs.dil_ref, config.bootstrap.contours.dil),
        ):
            self.assertEqual(tuple(registry.structs_referenced_dict[key]['Contour names']), names)

    def test_production_derivations_follow_the_typed_source_defaults(self):
        biopsy = BiopsyRuntimeConfig()
        biopsy = replace(biopsy, geometry=replace(biopsy.geometry, biopsy_radius=0.75))
        counts = MCCountsConfig(num_mc_dose_simulations_input=12, num_mc_mr_simulations_input=3)
        with patch('config.production.BiopsyRuntimeConfig', return_value=biopsy), patch(
            'config.production.MCCountsConfig', return_value=counts,
        ):
            config = build_production_pipeline_config()
        self.assertEqual(config.biopsy.geometry.simulated_biopsy_planning_radius_mm, 0.75)
        self.assertEqual(config.mc.counts.num_mc_mr_simulations_input, 12)
        self.assertEqual(config.mc.counts.num_mc_containment_simulations_input, counts.num_mc_containment_simulations_input)

    def test_snapshot_file_rehydration_is_exact(self):
        snapshot = build_pipeline_scientific_config_snapshot(build_production_pipeline_config())
        with tempfile.TemporaryDirectory() as directory:
            path = write_pipeline_config_snapshot(snapshot, Path(directory) / 'scientific.json')
            restored = rehydrate_pipeline_scientific_config_snapshot(read_pipeline_config_snapshot(path))
        self.assertEqual(build_pipeline_scientific_config_snapshot(restored).to_dict(), snapshot.to_dict())

    def test_explicit_typed_overrides_preserve_identity_boundaries(self):
        config = build_production_pipeline_config()
        original = build_pipeline_scientific_config_snapshot(config)
        session = replace(
            config,
            ui=replace(config.ui, rich_live_display_bool=False),
            artifacts=replace(config.artifacts, output_folder_name='synthetic-session'),
            patient_scientific_runner=replace(config.patient_scientific_runner, mode='plan_only'),
        )
        self.assertEqual(build_pipeline_scientific_config_snapshot(session).to_dict(), original.to_dict())
        changed = replace(session, replay=replace(session.replay, lower_bound_dose_value=7))
        snapshot = build_pipeline_scientific_config_snapshot(changed)
        self.assertNotEqual(snapshot.config_sha256, original.config_sha256)
        restored = rehydrate_pipeline_scientific_config_snapshot(snapshot)
        self.assertEqual(build_pipeline_scientific_config_snapshot(restored).to_dict(), snapshot.to_dict())
        self.assertEqual(build_pipeline_scientific_config_snapshot(config).to_dict(), original.to_dict())

    def test_construction_imports_no_main_or_scientific_execution(self):
        script = '''
import importlib.abc
import sys
class ConfigImportBoundary(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        forbidden = (
            "biopsy_localization_convex_main", "preprocessing", "startup", "patient_runner",
            "biopsy_optimizer.v2.biopsy_optimizer_module_v2", "guidance_maps.planning",
            "MC_simulator_convex", "pandas", "cupy", "open3d", "matplotlib",
        )
        if any(fullname == name or fullname.startswith(name + ".") for name in forbidden):
            raise AssertionError("Config imported execution dependency: " + fullname)
boundary = ConfigImportBoundary()
sys.meta_path.insert(0, boundary)
from config.production import build_production_pipeline_config
from config.snapshots import build_pipeline_scientific_config_snapshot
build_pipeline_scientific_config_snapshot(build_production_pipeline_config())
assert "guidance_maps.planning" not in sys.modules
sys.meta_path.remove(boundary)
import guidance_maps
from guidance_maps import planning
for name in guidance_maps.__all__:
    assert getattr(guidance_maps, name) is getattr(planning, name), name
assert set(guidance_maps.__all__) <= set(dir(guidance_maps))
try:
    guidance_maps.nonexistent_export
except AttributeError:
    pass
else:
    raise AssertionError("Unknown guidance export must fail")
'''
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
        result = subprocess.run([sys.executable, '-c', script], env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
