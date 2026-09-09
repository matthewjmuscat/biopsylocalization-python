"""Run-start provenance service for strict scientific artifact compatibility.

This module owns provenance assembly and file writing. It does not build
PipelineConfig, discover DICOM files, execute scientific stages, or decide which
runs may be merged. Those responsibilities remain with config, input, runner,
and post-run compatibility boundaries respectively.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from config.snapshots import build_pipeline_scientific_config_snapshot
from config.snapshots import canonical_json_value
from config.snapshots import write_pipeline_config_snapshot
from output_artifacts.run_compatibility import RunCompatibilityIdentity
from output_artifacts.run_compatibility import build_run_compatibility_identity
from output_artifacts.run_compatibility import write_run_compatibility_identity
from output_artifacts.schema_registry import OUTPUT_SCHEMA_REGISTRY_VERSION
from startup.code_identity import capture_code_identity
from startup.code_identity import write_code_identity
from startup.runtime_environment import capture_runtime_environment_identity
from startup.runtime_environment import write_runtime_environment_identity


PATIENT_INPUT_POLICY_SCHEMA_VERSION = "patient_input_policy_v1"


@dataclass(frozen=True, slots=True)
class RunProvenanceArtifacts:
    """Paths and strict identity produced at the run-start provenance boundary."""

    scientific_config_snapshot_path: Path
    code_identity_path: Path
    runtime_environment_identity_path: Path
    compatibility_identity_path: Path
    compatibility_identity: RunCompatibilityIdentity

    def __post_init__(self) -> None:
        for field_name in (
            "scientific_config_snapshot_path",
            "code_identity_path",
            "runtime_environment_identity_path",
            "compatibility_identity_path",
        ):
            object.__setattr__(self, field_name, Path(getattr(self, field_name)))
        if not isinstance(self.compatibility_identity, RunCompatibilityIdentity):
            raise TypeError("compatibility_identity must be a RunCompatibilityIdentity")

    def manifest_metadata(self) -> dict[str, Any]:
        """Return metadata propagated into patient, batch, and index manifests."""
        return {
            "run_compatibility_identity": self.compatibility_identity.to_dict(),
            "provenance_paths": {
                "scientific_config_snapshot": self.scientific_config_snapshot_path.as_posix(),
                "code_identity": self.code_identity_path.as_posix(),
                "runtime_environment_identity": self.runtime_environment_identity_path.as_posix(),
                "run_compatibility_identity": self.compatibility_identity_path.as_posix(),
            },
        }


def write_run_provenance_artifacts(
    *,
    pipeline_config: Any,
    routing_profile_path: Path | str,
    manifest_dir: Path | str,
    repository_path: Path | str,
    overwrite: bool = False,
) -> RunProvenanceArtifacts:
    """Write config, code, and strict compatibility identities for one run."""
    resolved_manifest_dir = Path(manifest_dir)
    scientific_config_snapshot = build_pipeline_scientific_config_snapshot(pipeline_config)
    code_identity = capture_code_identity(repository_path)
    runtime_environment_identity = capture_runtime_environment_identity(repository_path)
    scientific_config_snapshot_path = write_pipeline_config_snapshot(
        scientific_config_snapshot,
        resolved_manifest_dir.joinpath("resolved_scientific_config.json"),
        overwrite=overwrite,
    )
    code_identity_path = write_code_identity(
        code_identity,
        resolved_manifest_dir.joinpath("code_identity.json"),
        overwrite=overwrite,
    )
    runtime_environment_identity_path = write_runtime_environment_identity(
        runtime_environment_identity,
        resolved_manifest_dir.joinpath("runtime_environment_identity.json"),
        overwrite=overwrite,
    )
    routing_profile_payload = _read_json_object(Path(routing_profile_path))
    input_policy_payload = {
        "schema_version": PATIENT_INPUT_POLICY_SCHEMA_VERSION,
        "routing_profile": routing_profile_payload,
        "bootstrap": canonical_json_value(pipeline_config.bootstrap),
    }
    compatibility_identity = build_run_compatibility_identity(
        scientific_config_snapshot=scientific_config_snapshot,
        code_identity=code_identity,
        runtime_environment_identity=runtime_environment_identity,
        input_policy=input_policy_payload,
        output_schema_registry_version=OUTPUT_SCHEMA_REGISTRY_VERSION,
    )
    compatibility_identity_path = write_run_compatibility_identity(
        compatibility_identity,
        resolved_manifest_dir.joinpath("run_compatibility_identity.json"),
        overwrite=overwrite,
    )
    return RunProvenanceArtifacts(
        scientific_config_snapshot_path=scientific_config_snapshot_path,
        code_identity_path=code_identity_path,
        runtime_environment_identity_path=runtime_environment_identity_path,
        compatibility_identity_path=compatibility_identity_path,
        compatibility_identity=compatibility_identity,
    )


def _read_json_object(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, Mapping):
        raise TypeError("routing profile root must be an object: {}".format(path))
    return payload


__all__ = [
    "PATIENT_INPUT_POLICY_SCHEMA_VERSION",
    "RunProvenanceArtifacts",
    "write_run_provenance_artifacts",
]
