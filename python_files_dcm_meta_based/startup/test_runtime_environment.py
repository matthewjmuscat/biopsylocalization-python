from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from .runtime_environment import capture_runtime_environment_identity
from .runtime_environment import read_runtime_environment_identity
from .runtime_environment import write_runtime_environment_identity


class RuntimeEnvironmentIdentityTests(unittest.TestCase):
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
