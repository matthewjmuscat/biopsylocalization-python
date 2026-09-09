from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from .run_compatibility import COMPATIBILITY_DIMENSIONS
from .run_compatibility import IncompatibleRunArtifactsError
from .run_compatibility import RUN_COMPATIBILITY_METADATA_KEY
from .run_compatibility import RunCompatibilityIdentity
from .run_compatibility import compatibility_identity_from_manifest
from .run_compatibility import build_run_compatibility_identity
from .run_compatibility import require_compatible_run_identities
from .run_compatibility import read_run_compatibility_identity
from .run_compatibility import write_run_compatibility_identity
from .run_compatibility import validate_run_metadata_compatibility


class RunCompatibilityIdentityTests(unittest.TestCase):
    def test_identical_scientific_identities_are_compatible(self) -> None:
        identity = _identity()

        result = require_compatible_run_identities(
            (identity, identity),
            source_labels=("split_a", "split_b"),
        )

        self.assertTrue(result.compatible)
        self.assertEqual(result.mismatches, {})

    def test_each_scientific_dimension_is_strictly_enforced(self) -> None:
        identity = _identity()
        for field_name in COMPATIBILITY_DIMENSIONS:
            with self.subTest(field_name=field_name):
                changed = replace(identity, **{field_name: "different"}, identity_sha256="")
                with self.assertRaisesRegex(IncompatibleRunArtifactsError, field_name):
                    require_compatible_run_identities(
                        (identity, changed),
                        source_labels=("reference", "candidate"),
                    )

    def test_manifest_identity_round_trip_verifies_fingerprint(self) -> None:
        identity = _identity()
        manifest = {"metadata": {RUN_COMPATIBILITY_METADATA_KEY: identity.to_dict()}}

        self.assertEqual(compatibility_identity_from_manifest(manifest), identity)

    def test_manifest_without_identity_fails_strictly(self) -> None:
        with self.assertRaisesRegex(IncompatibleRunArtifactsError, RUN_COMPATIBILITY_METADATA_KEY):
            compatibility_identity_from_manifest({"metadata": {}})

    def test_identity_json_round_trip(self) -> None:
        identity = _identity()
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory).joinpath("run_compatibility_identity.json")
            write_run_compatibility_identity(identity, path)
            loaded = read_run_compatibility_identity(path)

        self.assertEqual(loaded, identity)

    def test_builder_uses_config_code_and_input_policy_identities(self) -> None:
        from config.snapshots import PipelineConfigSnapshot
        from config.snapshots import canonical_sha256
        from startup.code_identity import CodeIdentity
        from startup.runtime_environment import RuntimeEnvironmentIdentity

        config_payload = {"mc": {"trials": 10}}
        config_snapshot = PipelineConfigSnapshot(
            config_type="config.PipelineConfig.scientific",
            config=config_payload,
            config_sha256=canonical_sha256(config_payload),
        )
        code_identity = CodeIdentity(
            repository_root="/repo",
            commit="commit-a",
            branch="main",
            dirty=False,
            tracked_diff_sha256="tracked",
            untracked_files_sha256="untracked",
            source_tree_sha256=canonical_sha256(
                {
                    "schema_version": "code_identity_v1",
                    "commit": "commit-a",
                    "source_kind": "git_worktree",
                    "tracked_diff_sha256": "tracked",
                    "untracked_files_sha256": "untracked",
                }
            ),
        )
        runtime_environment_identity = RuntimeEnvironmentIdentity(
            python_version="3.11.0",
            python_implementation="CPython",
            platform="synthetic-platform",
            installed_distributions_sha256="packages-sha",
            dependency_lock_sha256="lock-sha",
        )

        identity = build_run_compatibility_identity(
            scientific_config_snapshot=config_snapshot,
            code_identity=code_identity,
            runtime_environment_identity=runtime_environment_identity,
            input_policy={"routing_profile": "legacy", "bootstrap": "v1"},
            output_schema_registry_version="registry-v1",
        )

        self.assertEqual(identity.scientific_config_sha256, config_snapshot.config_sha256)
        self.assertEqual(identity.code_source_sha256, code_identity.source_tree_sha256)
        self.assertEqual(identity.code_commit, "commit-a")
        self.assertEqual(identity.runtime_environment_sha256, runtime_environment_identity.identity_sha256)

    def test_strict_metadata_validation_rejects_missing_identity(self) -> None:
        with self.assertRaisesRegex(IncompatibleRunArtifactsError, "requires provenance"):
            validate_run_metadata_compatibility({"old_run": {}}, mode="strict")

    def test_legacy_mode_allows_only_uniformly_unidentified_historical_runs(self) -> None:
        result = validate_run_metadata_compatibility(
            {"old_a": {}, "old_b": {}},
            mode="legacy_allow_missing",
        )
        self.assertTrue(result.compatible)
        self.assertEqual(result.status, "legacy_missing_allowed")

        with self.assertRaisesRegex(IncompatibleRunArtifactsError, "identified and unidentified"):
            validate_run_metadata_compatibility(
                {
                    "new": {RUN_COMPATIBILITY_METADATA_KEY: _identity().to_dict()},
                    "old": {},
                },
                mode="legacy_allow_missing",
            )


def _identity() -> RunCompatibilityIdentity:
    return RunCompatibilityIdentity(
        scientific_config_sha256="config-sha",
        code_source_sha256="code-sha",
        input_policy_sha256="input-policy-sha",
        runtime_environment_sha256="environment-sha",
        output_schema_registry_version="registry-v1",
        code_commit="commit-a",
        code_dirty=False,
    )


if __name__ == "__main__":
    unittest.main()
