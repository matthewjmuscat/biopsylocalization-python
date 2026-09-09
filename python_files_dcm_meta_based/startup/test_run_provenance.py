from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from config.bootstrap import PatientBootstrapConfig
from config.snapshots import read_pipeline_config_snapshot
from output_artifacts.run_compatibility import read_run_compatibility_identity
from startup.code_identity import read_code_identity
from startup.runtime_environment import read_runtime_environment_identity
from startup.run_provenance import write_run_provenance_artifacts


@dataclass(frozen=True)
class _PipelineLikeConfig:
    preprocessing: str = "preprocessing"
    replay: str = "replay"
    guidance_maps: str = "guidance"
    optimizer: str = "optimizer"
    random_seeds: str = "seeds"
    mc: str = "mc"
    legacy_refs: str = "refs"
    structure_registry: str = "registry"
    bootstrap: PatientBootstrapConfig = PatientBootstrapConfig()
    grid_preprocessing: str = "grid"
    biopsy: str = "biopsy"


class RunProvenanceArtifactsTests(unittest.TestCase):
    def test_service_writes_linked_config_code_and_compatibility_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            repository = root.joinpath("repo")
            repository.mkdir()
            _git(repository, "init", "-q")
            _git(repository, "config", "user.email", "tests@example.invalid")
            _git(repository, "config", "user.name", "Test User")
            repository.joinpath("module.py").write_text("VALUE = 1\n", encoding="utf-8")
            _git(repository, "add", "module.py")
            _git(repository, "commit", "-q", "-m", "initial")
            routing_profile_path = root.joinpath("input_routing_profile.json")
            routing_profile_path.write_text(
                json.dumps({"schema_version": 1, "profile_id": "synthetic"}),
                encoding="utf-8",
            )

            result = write_run_provenance_artifacts(
                pipeline_config=_PipelineLikeConfig(),
                routing_profile_path=routing_profile_path,
                manifest_dir=root.joinpath("manifests"),
                repository_path=repository,
            )
            config_snapshot = read_pipeline_config_snapshot(result.scientific_config_snapshot_path)
            code_identity = read_code_identity(result.code_identity_path)
            runtime_environment_identity = read_runtime_environment_identity(
                result.runtime_environment_identity_path
            )
            compatibility_identity = read_run_compatibility_identity(result.compatibility_identity_path)

        self.assertEqual(compatibility_identity.scientific_config_sha256, config_snapshot.config_sha256)
        self.assertEqual(compatibility_identity.code_source_sha256, code_identity.source_tree_sha256)
        self.assertEqual(
            compatibility_identity.runtime_environment_sha256,
            runtime_environment_identity.identity_sha256,
        )
        self.assertFalse(code_identity.dirty)
        self.assertEqual(
            result.manifest_metadata()["run_compatibility_identity"]["identity_sha256"],
            compatibility_identity.identity_sha256,
        )


def _git(repository: Path, *args: str) -> None:
    subprocess.run(("git", *args), cwd=repository, check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
