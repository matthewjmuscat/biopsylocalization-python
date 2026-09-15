"""Exercise main's config selection/export branch without importing GPU science."""

import ast
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from config.production import build_production_pipeline_config
from config.snapshots import read_pipeline_config_snapshot, build_pipeline_scientific_config_snapshot


def _isolated_export_prefix(builder):
    """Compile the actual config selection and immediate return, excluding runtime."""
    path = Path(__file__).resolve().parents[1] / 'biopsy_localization_convex_main.py'
    module = ast.parse(path.read_text())
    main = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    config_index = next(
        index for index, node in enumerate(main.body)
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'pipeline_config' for target in node.targets)
    )
    selection = main.body[config_index]
    branch = main.body[config_index + 1]
    assert isinstance(branch, ast.If)
    assert isinstance(branch.test, ast.Compare)
    assert isinstance(branch.test.left, ast.Name) and branch.test.left.id == 'scientific_config_export_path'
    assert isinstance(branch.body[-1], ast.Return)
    # Root construction belongs to the config layer, never another inline main tree.
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'PipelineConfig' for node in ast.walk(main))
    export_function = ast.FunctionDef(
        name='export_only', args=main.args,
        body=[selection, branch, ast.Raise(exc=ast.Call(func=ast.Name(id='RuntimeError', ctx=ast.Load()), args=[ast.Constant('export did not return')], keywords=[]))],
        decorator_list=[],
    )
    isolated = ast.fix_missing_locations(ast.Module(body=[export_function], type_ignores=[]))
    namespace = {'build_production_pipeline_config': builder}
    exec(compile(isolated, '<main-config-export-test>', 'exec'), namespace)
    return namespace['export_only']


class LegacyConfigExportTests(unittest.TestCase):
    def test_export_uses_production_owner_and_returns_before_execution(self):
        config = build_production_pipeline_config()
        builder = Mock(return_value=config)
        export = _isolated_export_prefix(builder)
        with TemporaryDirectory() as directory:
            output = Path(directory) / 'scientific.json'
            export(scientific_config_export_path=output)
            builder.assert_called_once_with()
            self.assertEqual(read_pipeline_config_snapshot(output).to_dict(), build_pipeline_scientific_config_snapshot(config).to_dict())
            with self.assertRaises(FileExistsError):
                export(scientific_config_export_path=output)

    def test_explicit_resolved_config_bypasses_default_construction(self):
        config = build_production_pipeline_config()
        config = replace(config, replay=replace(config.replay, lower_bound_dose_value=7))
        builder = Mock(side_effect=AssertionError('Explicit config must not rebuild defaults'))
        export = _isolated_export_prefix(builder)
        with TemporaryDirectory() as directory:
            output = Path(directory) / 'scientific.json'
            export(pipeline_config=config, scientific_config_export_path=output)
            self.assertEqual(read_pipeline_config_snapshot(output).to_dict(), build_pipeline_scientific_config_snapshot(config).to_dict())
        builder.assert_not_called()


if __name__ == '__main__':
    unittest.main()
