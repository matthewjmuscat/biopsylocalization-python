from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
import subprocess
import tempfile
import unittest

from .code_identity import capture_code_identity
from .code_identity import PACKAGED_SOURCE_IDENTITY_ENV_VAR
from .code_identity import read_code_identity
from .code_identity import write_code_identity


class CodeIdentityTests(unittest.TestCase):
    def test_clean_and_dirty_source_states_have_distinct_identities(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = _initialize_repository(Path(temporary_directory))
            clean_identity = capture_code_identity(repository)

            tracked_path = repository.joinpath("module.py")
            tracked_path.write_text("VALUE = 2\n", encoding="utf-8")
            tracked_dirty_identity = capture_code_identity(repository)

            untracked_path = repository.joinpath("new_module.py")
            untracked_path.write_text("NEW_VALUE = 3\n", encoding="utf-8")
            tracked_and_untracked_identity = capture_code_identity(repository)
            repository.joinpath("untracked_patient_data.dcm").write_bytes(b"not-source")
            identity_with_untracked_data = capture_code_identity(repository)

        self.assertFalse(clean_identity.dirty)
        self.assertTrue(tracked_dirty_identity.dirty)
        self.assertNotEqual(clean_identity.source_tree_sha256, tracked_dirty_identity.source_tree_sha256)
        self.assertNotEqual(tracked_dirty_identity.source_tree_sha256, tracked_and_untracked_identity.source_tree_sha256)
        self.assertEqual(
            tracked_and_untracked_identity.source_tree_sha256,
            identity_with_untracked_data.source_tree_sha256,
        )
        self.assertEqual(tracked_and_untracked_identity.untracked_source_files, ("new_module.py",))

    def test_code_identity_round_trip_verifies_component_fingerprints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = _initialize_repository(Path(temporary_directory))
            identity = capture_code_identity(repository)
            output_path = repository.joinpath("identity.json")
            write_code_identity(identity, output_path)
            loaded = read_code_identity(output_path)

        self.assertEqual(loaded, identity)

    def test_packaged_build_requires_explicit_source_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            package_root = Path(temporary_directory)
            with patch.dict("os.environ", {PACKAGED_SOURCE_IDENTITY_ENV_VAR: "release-1.2.3"}):
                identity = capture_code_identity(package_root)

        self.assertEqual(identity.source_kind, "packaged_override")
        self.assertEqual(identity.commit, "release-1.2.3")
        self.assertFalse(identity.dirty)


def _initialize_repository(repository: Path) -> Path:
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "tests@example.invalid")
    _git(repository, "config", "user.name", "Test User")
    repository.joinpath("module.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repository, "add", "module.py")
    _git(repository, "commit", "-q", "-m", "initial")
    return repository


def _git(repository: Path, *args: str) -> None:
    subprocess.run(("git", *args), cwd=repository, check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
