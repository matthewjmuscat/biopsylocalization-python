from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from output_artifacts.run_compatibility import IncompatibleRunArtifactsError
from output_artifacts.run_compatibility import RUN_COMPATIBILITY_METADATA_KEY
from output_artifacts.run_compatibility import RunCompatibilityIdentity
from patient_runner.contracts import PatientBatchRunResult
from validation import reconstructed_cohort_comparator as comparator


class ReconstructedCohortCompatibilityTests(unittest.TestCase):
    def test_reconstruction_rejects_mismatched_run_identity_before_assembly(self) -> None:
        first = _batch_result("first", _identity())
        second = _batch_result(
            "second",
            replace(_identity(), scientific_config_sha256="other-config", identity_sha256=""),
        )

        with tempfile.TemporaryDirectory() as temporary_directory, patch.object(
            comparator,
            "_resolve_batch_result",
            side_effect=((Path("first"), first), (Path("second"), second)),
        ), patch.object(comparator, "run_patient_batch_cohort_assembly") as assembly_mock:
            with self.assertRaisesRegex(IncompatibleRunArtifactsError, "scientific_config_sha256"):
                comparator.reconstruct_patient_runner_cohort_surface(
                    "split",
                    ("first", "second"),
                    output_root=temporary_directory,
                    compatibility_mode="strict",
                    write_outputs=False,
                )

        assembly_mock.assert_not_called()

    def test_strict_reconstruction_rejects_missing_identity(self) -> None:
        batch_result = _batch_result("legacy", None)

        with tempfile.TemporaryDirectory() as temporary_directory, patch.object(
            comparator,
            "_resolve_batch_result",
            return_value=(Path("legacy"), batch_result),
        ):
            with self.assertRaisesRegex(IncompatibleRunArtifactsError, "requires provenance"):
                comparator.reconstruct_patient_runner_cohort_surface(
                    "reference",
                    ("legacy",),
                    output_root=temporary_directory,
                    compatibility_mode="strict",
                    write_outputs=False,
                )

    def test_explicit_legacy_mode_records_missing_identity(self) -> None:
        batch_result = _batch_result("legacy", None)
        assembly_result = SimpleNamespace(assembled_tables={})

        with tempfile.TemporaryDirectory() as temporary_directory, patch.object(
            comparator,
            "_resolve_batch_result",
            return_value=(Path("legacy"), batch_result),
        ), patch.object(
            comparator,
            "run_patient_batch_cohort_assembly",
            return_value=(assembly_result, None, ()),
        ):
            surface = comparator.reconstruct_patient_runner_cohort_surface(
                "reference",
                ("legacy",),
                output_root=temporary_directory,
                compatibility_mode="legacy_allow_missing",
                write_outputs=False,
            )

        self.assertEqual(surface.compatibility_validation.status, "legacy_missing_allowed")
        self.assertNotIn(RUN_COMPATIBILITY_METADATA_KEY, surface.batch_result.metadata)


def _batch_result(label: str, identity: RunCompatibilityIdentity | None) -> PatientBatchRunResult:
    metadata = {"label": label}
    if identity is not None:
        metadata[RUN_COMPATIBILITY_METADATA_KEY] = identity.to_dict()
    return PatientBatchRunResult.from_patient_results(
        output_root=Path(label),
        patient_results=(),
        metadata=metadata,
    )


def _identity() -> RunCompatibilityIdentity:
    return RunCompatibilityIdentity(
        scientific_config_sha256="config-sha",
        code_source_sha256="code-sha",
        input_policy_sha256="input-policy-sha",
        runtime_environment_sha256="environment-sha",
        output_schema_registry_version="registry-v1",
        code_commit="commit-a",
    )


if __name__ == "__main__":
    unittest.main()
