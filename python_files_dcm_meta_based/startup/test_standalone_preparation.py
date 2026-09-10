"""Synthetic contract/integration tests for standalone provenance preparation.

Tests reuse the real config snapshot fixture without reading patient data or
executing science. Temporary packaged-build identities exercise the production
provenance writer without creating commits or inspecting a research worktree.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

from config.rehydration import rehydrate_pipeline_scientific_config_snapshot
from config.snapshots import build_pipeline_scientific_config_snapshot
from config.snapshots import canonical_sha256
from config.snapshots import read_pipeline_config_snapshot
from config.snapshots import write_pipeline_config_snapshot
from config.test_snapshots import _build_real_pipeline_config
from output_artifacts.run_compatibility import read_run_compatibility_identity
from startup.code_identity import PACKAGED_SOURCE_IDENTITY_ENV_VAR
from startup.code_identity import capture_code_identity
from startup.code_identity import read_code_identity
from startup.runtime_environment import capture_runtime_environment_identity
from startup.runtime_environment import read_runtime_environment_identity
from startup.standalone_preparation import DEFAULT_REPOSITORY_PATH
from startup.standalone_preparation import PREPARATION_RECORD_SCHEMA_VERSION
from startup.standalone_preparation import StandalonePreparationResult
from startup.standalone_preparation import prepare_patient_scientific_run


CLI_PATH = Path(__file__).resolve().parents[1].joinpath("prepare_patient_scientific_run.py")
_CPU_SAFE_CLI = textwrap.dedent("""\
    import importlib.abc
    from pathlib import Path
    import runpy
    import sys

    forbidden = {
        'main', 'cupy', 'cupyx', 'cudf', 'cuspatial', 'rmm', 'numba',
        'tkinter', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
        'open3d', 'pyvista', 'vtk', 'matplotlib',
    }

    class RejectExecutionImports(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in forbidden:
                raise AssertionError('forbidden preparation import: ' + fullname)
            return None

    sys.meta_path.insert(0, RejectExecutionImports())
    sys.argv = sys.argv[1:]
    sys.path.insert(0, str(Path(sys.argv[0]).parent))
    import startup.standalone_preparation
    runpy.run_path(sys.argv[0], run_name='__main__')
""")


class StandalonePreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)
        self.repository = self.root.joinpath("synthetic-build")
        self.repository.mkdir()
        self.repository.joinpath("Pipfile.lock").write_text("{}\n", encoding="utf-8")
        environment = patch.dict(os.environ, {
            PACKAGED_SOURCE_IDENTITY_ENV_VAR: "standalone-preparation-synthetic-build-v1",
        })
        environment.start()
        self.addCleanup(environment.stop)
        self.source_snapshot = build_pipeline_scientific_config_snapshot(_build_real_pipeline_config())
        self.snapshot_path = write_pipeline_config_snapshot(
            self.source_snapshot,
            self.root.joinpath("historical-provenance", "resolved_scientific_config.json"),
        )
        for filename in ("code_identity.json", "runtime_environment_identity.json", "run_compatibility_identity.json"):
            self.snapshot_path.parent.joinpath(filename).write_text(
                '{"historical_identity": "must-not-be-relabelled"}\n', encoding="utf-8",
            )
        self.routing_path = self.root.joinpath("shared-policy", "input_routing_profile.json")
        self.routing_path.parent.mkdir()
        self.routing_path.write_text(
            json.dumps({"schema_version": 1, "profile_id": "synthetic"}), encoding="utf-8",
        )
        self.destination = self.root.joinpath("new-provenance")
        self.source_bytes = {
            path: path.read_bytes()
            for path in (*self.snapshot_path.parent.iterdir(), self.routing_path)
        }

    def _prepare(self, destination: Path | None = None) -> StandalonePreparationResult:
        return prepare_patient_scientific_run(
            scientific_config_snapshot_path=self.snapshot_path,
            routing_profile_path=self.routing_path,
            output_dir=self.destination if destination is None else destination,
            repository_path=self.repository,
        )

    def _assert_sources_preserved(self) -> None:
        for path, original_bytes in self.source_bytes.items():
            self.assertEqual(path.read_bytes(), original_bytes, path)
        self.assertEqual(
            set(self.snapshot_path.parent.iterdir()),
            set(self.source_bytes).difference({self.routing_path}),
        )

    def _run_cli(self, *args: str, stdlib_only: bool = False) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            (sys.executable, *(("-S",) if stdlib_only else ()), "-c", _CPU_SAFE_CLI, str(CLI_PATH), *args),
            cwd=self.root,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def _cli_args(self) -> tuple[str, ...]:
        return (
            "--scientific-config-snapshot", str(self.snapshot_path),
            "--routing-profile", str(self.routing_path),
            "--output-dir", str(self.destination),
            "--repository-path", str(self.repository),
        )

    def test_real_fixture_roundtrip_new_provenance_and_source_preservation(self) -> None:
        rehydrated = rehydrate_pipeline_scientific_config_snapshot(self.snapshot_path)
        self.assertEqual(
            build_pipeline_scientific_config_snapshot(rehydrated).config_sha256,
            self.source_snapshot.config_sha256,
        )

        result = self._prepare()
        snapshot = read_pipeline_config_snapshot(result.provenance.scientific_config_snapshot_path)
        code = read_code_identity(result.provenance.code_identity_path)
        environment = read_runtime_environment_identity(result.provenance.runtime_environment_identity_path)
        compatibility = read_run_compatibility_identity(result.provenance.compatibility_identity_path)
        record = json.loads(result.preparation_record_path.read_text(encoding="utf-8"))

        self.assertEqual(snapshot, self.source_snapshot)
        self.assertEqual(compatibility.scientific_config_sha256, self.source_snapshot.config_sha256)
        self.assertEqual(code, capture_code_identity(self.repository))
        self.assertEqual(environment, capture_runtime_environment_identity(self.repository))
        self.assertEqual(compatibility.code_source_sha256, code.source_tree_sha256)
        self.assertEqual(compatibility.runtime_environment_sha256, environment.identity_sha256)
        self.assertEqual(record["schema_version"], PREPARATION_RECORD_SCHEMA_VERSION)
        self.assertEqual(record["scientific_configuration"], "reused_verified_snapshot")
        self.assertEqual(record["execution_provenance"], "captured_current")
        self.assertEqual(record["source_snapshot"], {
            "path": self.snapshot_path.as_posix(),
            "config_sha256": self.source_snapshot.config_sha256,
            "file_sha256": hashlib.sha256(self.source_bytes[self.snapshot_path]).hexdigest(),
        })
        self.assertEqual(record["source_routing_profile"], {
            "path": self.routing_path.as_posix(),
            "canonical_sha256": canonical_sha256(json.loads(self.source_bytes[self.routing_path])),
            "file_sha256": hashlib.sha256(self.source_bytes[self.routing_path]).hexdigest(),
        })
        self.assertEqual(record["repository_path"], self.repository.as_posix())
        self.assertEqual(record["new_run_provenance"], result.provenance.manifest_metadata())
        self.assertEqual({path.name for path in self.destination.iterdir()}, {
            "resolved_scientific_config.json", "code_identity.json",
            "runtime_environment_identity.json", "run_compatibility_identity.json",
            "preparation_record.json",
        })
        self._assert_sources_preserved()

    def test_new_execution_identity_does_not_relabel_previous_preparation(self) -> None:
        first = self._prepare()
        previous_bytes = {path: path.read_bytes() for path in self.destination.iterdir()}
        with patch.dict(os.environ, {PACKAGED_SOURCE_IDENTITY_ENV_VAR: "synthetic-build-v2"}):
            second = self._prepare(self.root.joinpath("another-new-provenance"))
        self.assertEqual(
            first.provenance.compatibility_identity.scientific_config_sha256,
            second.provenance.compatibility_identity.scientific_config_sha256,
        )
        self.assertNotEqual(
            first.provenance.compatibility_identity.code_source_sha256,
            second.provenance.compatibility_identity.code_source_sha256,
        )
        for path, original_bytes in previous_bytes.items():
            self.assertEqual(path.read_bytes(), original_bytes)
        self._assert_sources_preserved()

    def test_default_repository_is_module_relative(self) -> None:
        self.assertEqual(DEFAULT_REPOSITORY_PATH, CLI_PATH.parents[1])
        self.assertEqual(
            inspect.signature(prepare_patient_scientific_run).parameters["repository_path"].default,
            DEFAULT_REPOSITORY_PATH,
        )

    def test_existing_empty_destination_is_supported(self) -> None:
        self.destination.mkdir()
        self.assertTrue(self._prepare().preparation_record_path.is_file())

    def test_nonempty_destination_is_rejected_without_changes(self) -> None:
        self.destination.mkdir()
        marker = self.destination.joinpath(".existing-output")
        marker.write_bytes(b"preserve me")
        with self.assertRaisesRegex(FileExistsError, "must be empty"):
            self._prepare()
        self.assertEqual(list(self.destination.iterdir()), [marker])
        self.assertEqual(marker.read_bytes(), b"preserve me")
        self._assert_sources_preserved()

    def test_file_destination_is_rejected_without_changes(self) -> None:
        self.destination.write_bytes(b"not a directory")
        with self.assertRaises(NotADirectoryError):
            self._prepare()
        self.assertEqual(self.destination.read_bytes(), b"not a directory")
        self._assert_sources_preserved()

    def test_source_destination_overlap_including_symlinks_is_rejected(self) -> None:
        alias = self.root.joinpath("historical-alias")
        alias.symlink_to(self.snapshot_path.parent, target_is_directory=True)
        for destination in (
            self.snapshot_path, self.snapshot_path.parent,
            self.snapshot_path.parent.joinpath("nested-output"), self.root,
            alias, alias.joinpath("nested-output"),
            self.routing_path, self.routing_path.parent,
        ):
            with self.subTest(destination=destination):
                with self.assertRaisesRegex(ValueError, "overlap"):
                    self._prepare(destination)
        self._assert_sources_preserved()

    def test_tampered_snapshot_fails_before_output_creation(self) -> None:
        payload = self.source_snapshot.to_dict()
        payload["config"]["random_seeds"]["master_seed"] = 999
        tampered_bytes = json.dumps(payload).encode("utf-8")
        self.snapshot_path.write_bytes(tampered_bytes)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self._prepare()
        self.assertFalse(self.destination.exists())
        self.assertEqual(self.snapshot_path.read_bytes(), tampered_bytes)

    def test_rehydration_contract_failure_precedes_output_creation(self) -> None:
        payload = self.source_snapshot.to_dict()
        payload["config"].pop("bootstrap")
        payload["config_sha256"] = canonical_sha256(payload["config"])
        self.snapshot_path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "supported contract"):
            self._prepare()
        self.assertFalse(self.destination.exists())

    def test_invalid_routing_json_fails_before_output_creation(self) -> None:
        for contents in ("[]", "null", "{invalid", '{"value": NaN}'):
            with self.subTest(contents=contents):
                self.routing_path.write_text(contents, encoding="utf-8")
                with self.assertRaises((TypeError, ValueError)):
                    self._prepare()
                self.assertFalse(self.destination.exists())
                self.assertEqual(self.routing_path.read_text(encoding="utf-8"), contents)

    def test_provenance_failure_has_no_completion_record(self) -> None:
        with patch(
            "startup.run_provenance.write_run_provenance_artifacts",
            side_effect=RuntimeError("synthetic provenance failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic provenance failure"):
                self._prepare()
        self.assertFalse(self.destination.joinpath("preparation_record.json").exists())
        self._assert_sources_preserved()

    def test_cli_help_and_service_import_are_stdlib_only(self) -> None:
        result = self._run_cli("--help", stdlib_only=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--scientific-config-snapshot", result.stdout)
        self.assertIn("--routing-profile", result.stdout)
        self.assertIn("--output-dir", result.stdout)
        self.assertIn("--repository-path", result.stdout)
        self.assertIn("NOT patient data", result.stdout)
        self.assertFalse(self.destination.exists())

    def test_cli_prepares_real_fixture_with_execution_imports_blocked(self) -> None:
        result = self._run_cli(*self._cli_args())
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads(result.stdout)
        self.assertEqual(
            summary["preparation_record_path"], self.destination.joinpath("preparation_record.json").as_posix(),
        )
        self.assertEqual(
            summary["run_compatibility_identity"]["scientific_config_sha256"],
            self.source_snapshot.config_sha256,
        )
        self._assert_sources_preserved()

    def test_cli_requires_explicit_routing_profile(self) -> None:
        result = self._run_cli(
            "--scientific-config-snapshot", str(self.snapshot_path),
            "--output-dir", str(self.destination),
        )
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("--routing-profile", result.stderr)
        self.assertFalse(self.destination.exists())

    def test_cli_tamper_failure_is_nonzero_without_traceback_or_output(self) -> None:
        payload = self.source_snapshot.to_dict()
        payload["config_sha256"] = "0" * 64
        self.snapshot_path.write_text(json.dumps(payload), encoding="utf-8")
        result = self._run_cli(*self._cli_args())
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("fingerprint", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse(self.destination.exists())


if __name__ == "__main__":
    unittest.main()