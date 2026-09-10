"""Guard the transitional config-only exit without importing GPU legacy main."""

import ast
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from config.test_snapshots import _build_real_pipeline_config
from config.snapshots import read_pipeline_config_snapshot, build_pipeline_scientific_config_snapshot


class LegacyConfigExportTests(unittest.TestCase):
    def test_export_branch_immediately_follows_config_construction_and_returns(self):
        path = Path(__file__).resolve().parents[1] / "biopsy_localization_convex_main.py"
        module = ast.parse(path.read_text())
        main = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        config_index = next(index for index, node in enumerate(main.body) if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "pipeline_config" for target in node.targets))
        branch = main.body[config_index + 1]
        self.assertIsInstance(branch, ast.If)
        self.assertIn("scientific_config_export_path", ast.unparse(branch.test))
        self.assertIsInstance(branch.body[-1], ast.Return)
        export_function = ast.FunctionDef(
            name="export_only", args=main.args, body=[branch, ast.Raise(exc=ast.Call(func=ast.Name(id="RuntimeError", ctx=ast.Load()), args=[ast.Constant("export did not return")], keywords=[]))],
            decorator_list=[],
        )
        isolated = ast.fix_missing_locations(ast.Module(body=[export_function], type_ignores=[]))
        config = _build_real_pipeline_config()
        namespace = {"pipeline_config": config}
        exec(compile(isolated, "<export-branch-test>", "exec"), namespace)
        with TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "scientific.json"
            namespace["export_only"](scientific_config_export_path=output)
            self.assertEqual(read_pipeline_config_snapshot(output).config_sha256, build_pipeline_scientific_config_snapshot(config).config_sha256)
            with self.assertRaises(FileExistsError):
                namespace["export_only"](scientific_config_export_path=output)


if __name__ == "__main__":
    unittest.main()