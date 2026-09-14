from __future__ import annotations

from pathlib import Path
from dataclasses import replace
import sys
import tempfile
import unittest
from unittest.mock import patch

from .runtime_environment import capture_runtime_environment_identity
from .runtime_environment import read_runtime_environment_identity
from .runtime_environment import write_runtime_environment_identity
from . import runtime_environment as environment


class RuntimeEnvironmentIdentityTests(unittest.TestCase):
    def test_arbitrary_sys_path_metadata_does_not_change_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vendor = root / "arbitrary_vendor"
            metadata = vendor / "imaginary_dependency-9.8.dist-info"
            metadata.mkdir(parents=True)
            (metadata / "METADATA").write_text("Name: imaginary-dependency\nVersion: 9.8\n")
            before = capture_runtime_environment_identity(root)
            with patch.object(sys, "path", [*sys.path, str(vendor)]):
                self.assertTrue(any(d.metadata["Name"] == "imaginary-dependency"
                                    for d in environment.importlib_metadata.distributions()))
                self.assertEqual(before, capture_runtime_environment_identity(root))

    def test_installed_roots_deduplicate_records_and_detect_package_and_lock_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sites = [root / "purelib", root / "platlib"]
            (root / ".git").mkdir()
            for name, site_path in zip(("Actual_Dependency", "actual-dependency"), sites):
                metadata = site_path / "actual_dependency-1.0.dist-info"
                metadata.mkdir(parents=True)
                (metadata / "METADATA").write_text(f"Name: {name}\nVersion: 1.0\n")
            with patch.object(environment, "installed_distribution_roots", return_value=tuple(map(str, sites))):
                before = capture_runtime_environment_identity(root)
                self.assertEqual(before.installed_distributions_sha256,
                                 environment.canonical_sha256(["actual-dependency==1.0"]))
                metadata.joinpath("METADATA").write_text("Name: actual-dependency\nVersion: 2.0\n")
                after = capture_runtime_environment_identity(root)
                self.assertNotEqual(before.identity_sha256, after.identity_sha256)
                root.joinpath("Pipfile.lock").write_text('{"changed": true}')
                self.assertNotEqual(after.identity_sha256, capture_runtime_environment_identity(root).identity_sha256)

    def test_active_root_policy_includes_enabled_sites_without_using_sys_path(self):
        with patch.object(environment.sysconfig, "get_path", side_effect=lambda key: "/env/" + key), \
                patch.object(environment.site, "getsitepackages", return_value=["/system/site", "/env/purelib"]), \
                patch.object(environment.site, "getusersitepackages", return_value="/user/site"), \
                patch.object(environment.site, "ENABLE_USER_SITE", True):
            self.assertEqual(environment.installed_distribution_roots(),
                             ("/env/platlib", "/env/purelib", "/system/site", "/user/site"))
            with patch.object(environment.site, "ENABLE_USER_SITE", False):
                self.assertNotIn("/user/site", environment.installed_distribution_roots())

    def test_python_platform_dimensions_and_historical_schema_remain_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            identity = capture_runtime_environment_identity(directory)
            for field in ("python_version", "python_implementation", "platform"):
                with self.subTest(field=field):
                    changed = replace(identity, identity_sha256="", **{field: "different"})
                    self.assertNotEqual(changed.identity_sha256, identity.identity_sha256)
            historical = replace(identity, schema_version="runtime_environment_identity_v1", identity_sha256="")
            path = Path(directory) / "v1.json"
            write_runtime_environment_identity(historical, path)
            self.assertEqual(read_runtime_environment_identity(path), historical)
            self.assertNotEqual(historical.identity_sha256, identity.identity_sha256)

    def test_environment_identity_records_lockfile_and_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = Path(temporary_directory)
            repository.joinpath(".git").mkdir()
            repository.joinpath("Pipfile.lock").write_text('{"default": {}}\n', encoding="utf-8")

            identity = capture_runtime_environment_identity(repository)
            output_path = repository.joinpath("artifacts", "runtime_environment_identity.json")
            write_runtime_environment_identity(identity, output_path, overwrite=True)
            loaded = read_runtime_environment_identity(output_path)

        self.assertEqual(loaded, identity)
        self.assertEqual(len(identity.installed_distributions_sha256), 64)
        self.assertEqual(len(identity.dependency_lock_sha256), 64)
        self.assertEqual(len(identity.identity_sha256), 64)


if __name__ == "__main__":
    unittest.main()
