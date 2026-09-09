"""Strict compatibility identities for combining patient-run artifacts.

The initial policy is intentionally conservative: artifacts from separate runs
may be combined only when scientific configuration, effective source code,
input/bootstrap policy, and output schema registry version are identical. Patient
selection is not part of this identity because split runs intentionally contain
different, disjoint patients.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from config.snapshots import PipelineConfigSnapshot
from config.snapshots import canonical_sha256
from startup.code_identity import CodeIdentity
from startup.runtime_environment import RuntimeEnvironmentIdentity


RUN_COMPATIBILITY_IDENTITY_SCHEMA_VERSION = "run_compatibility_identity_v1"
RUN_COMPATIBILITY_METADATA_KEY = "run_compatibility_identity"
STRICT_RUN_COMPATIBILITY_POLICY = "strict_exact_v1"
RUN_COMPATIBILITY_MODES = frozenset({"strict", "legacy_allow_missing"})
COMPATIBILITY_DIMENSIONS = (
    "scientific_config_sha256",
    "code_source_sha256",
    "input_policy_sha256",
    "runtime_environment_sha256",
    "output_schema_registry_version",
)


class IncompatibleRunArtifactsError(ValueError):
    """Raised when run outputs fail the strict artifact-combination policy."""


@dataclass(frozen=True, slots=True)
class RunCompatibilityIdentity:
    """Scientific identity that must match before cross-run artifact merging."""

    scientific_config_sha256: str
    code_source_sha256: str
    input_policy_sha256: str
    runtime_environment_sha256: str
    output_schema_registry_version: str
    code_commit: str = ""
    code_dirty: bool = False
    policy: str = STRICT_RUN_COMPATIBILITY_POLICY
    schema_version: str = RUN_COMPATIBILITY_IDENTITY_SCHEMA_VERSION
    identity_sha256: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != RUN_COMPATIBILITY_IDENTITY_SCHEMA_VERSION:
            raise ValueError("unsupported run compatibility schema_version: {}".format(self.schema_version))
        if self.policy != STRICT_RUN_COMPATIBILITY_POLICY:
            raise ValueError("unsupported run compatibility policy: {}".format(self.policy))
        for field_name in COMPATIBILITY_DIMENSIONS:
            if str(getattr(self, field_name)).strip() == "":
                raise ValueError("{} cannot be empty under strict compatibility policy".format(field_name))
        expected_identity_sha256 = canonical_sha256(self._identity_payload())
        if self.identity_sha256 and self.identity_sha256 != expected_identity_sha256:
            raise ValueError("run compatibility identity_sha256 does not match its dimensions")
        object.__setattr__(self, "code_commit", str(self.code_commit).strip())
        object.__setattr__(self, "code_dirty", bool(self.code_dirty))
        object.__setattr__(self, "identity_sha256", expected_identity_sha256)

    def _identity_payload(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "policy": self.policy,
            **{field_name: str(getattr(self, field_name)) for field_name in COMPATIBILITY_DIMENSIONS},
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the manifest-ready compatibility identity."""
        return {
            **self._identity_payload(),
            "identity_sha256": self.identity_sha256,
            "code_commit": self.code_commit,
            "code_dirty": self.code_dirty,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunCompatibilityIdentity":
        """Read and verify a compatibility identity from manifest metadata."""
        return cls(
            schema_version=str(payload.get("schema_version", "")),
            policy=str(payload.get("policy", "")),
            scientific_config_sha256=str(payload.get("scientific_config_sha256", "")),
            code_source_sha256=str(payload.get("code_source_sha256", "")),
            input_policy_sha256=str(payload.get("input_policy_sha256", "")),
            runtime_environment_sha256=str(payload.get("runtime_environment_sha256", "")),
            output_schema_registry_version=str(payload.get("output_schema_registry_version", "")),
            code_commit=str(payload.get("code_commit", "")),
            code_dirty=bool(payload.get("code_dirty", False)),
            identity_sha256=str(payload.get("identity_sha256", "")),
        )


def build_run_compatibility_identity(
    *,
    scientific_config_snapshot: PipelineConfigSnapshot,
    code_identity: CodeIdentity,
    runtime_environment_identity: RuntimeEnvironmentIdentity,
    input_policy: Any,
    output_schema_registry_version: str,
) -> RunCompatibilityIdentity:
    """Build the strict identity recorded in patient and batch manifests."""
    if not isinstance(scientific_config_snapshot, PipelineConfigSnapshot):
        raise TypeError("scientific_config_snapshot must be a PipelineConfigSnapshot")
    if not isinstance(code_identity, CodeIdentity):
        raise TypeError("code_identity must be a CodeIdentity")
    if not isinstance(runtime_environment_identity, RuntimeEnvironmentIdentity):
        raise TypeError("runtime_environment_identity must be a RuntimeEnvironmentIdentity")
    return RunCompatibilityIdentity(
        scientific_config_sha256=scientific_config_snapshot.config_sha256,
        code_source_sha256=code_identity.source_tree_sha256,
        input_policy_sha256=canonical_sha256(input_policy),
        runtime_environment_sha256=runtime_environment_identity.identity_sha256,
        output_schema_registry_version=str(output_schema_registry_version),
        code_commit=code_identity.commit,
        code_dirty=code_identity.dirty,
    )


def write_run_compatibility_identity(
    identity: RunCompatibilityIdentity,
    output_path: Path | str,
    *,
    overwrite: bool = False,
) -> Path:
    """Write one strict compatibility identity JSON artifact."""
    if not isinstance(identity, RunCompatibilityIdentity):
        raise TypeError("identity must be a RunCompatibilityIdentity")
    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError("run compatibility identity already exists: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(identity.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def read_run_compatibility_identity(input_path: Path | str) -> RunCompatibilityIdentity:
    """Read and verify one strict compatibility identity JSON artifact."""
    path = Path(input_path)
    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, Mapping):
        raise TypeError("run compatibility identity root must be an object")
    return RunCompatibilityIdentity.from_dict(payload)


@dataclass(frozen=True, slots=True)
class RunCompatibilityCheck:
    """Detailed result of comparing multiple run identities."""

    compatible: bool
    reference_identity_sha256: str
    identity_sha256_by_source: Mapping[str, str]
    mismatches: Mapping[str, Mapping[str, str]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "identity_sha256_by_source", dict(self.identity_sha256_by_source))
        object.__setattr__(self, "mismatches", {key: dict(value) for key, value in self.mismatches.items()})

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RUN_COMPATIBILITY_IDENTITY_SCHEMA_VERSION,
            "policy": STRICT_RUN_COMPATIBILITY_POLICY,
            "compatible": self.compatible,
            "reference_identity_sha256": self.reference_identity_sha256,
            "identity_sha256_by_source": dict(self.identity_sha256_by_source),
            "mismatches": {key: dict(value) for key, value in self.mismatches.items()},
        }


@dataclass(frozen=True, slots=True)
class RunCompatibilityValidation:
    """Compatibility decision including explicit legacy missing-provenance state."""

    compatible: bool
    mode: str
    status: str
    missing_identity_sources: tuple[str, ...] = ()
    check: RunCompatibilityCheck | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RUN_COMPATIBILITY_IDENTITY_SCHEMA_VERSION,
            "policy": STRICT_RUN_COMPATIBILITY_POLICY,
            "mode": self.mode,
            "status": self.status,
            "compatible": self.compatible,
            "missing_identity_sources": list(self.missing_identity_sources),
            "check": None if self.check is None else self.check.to_dict(),
        }


def compare_run_compatibility_identities(
    identities: Sequence[RunCompatibilityIdentity],
    *,
    source_labels: Sequence[str] = (),
) -> RunCompatibilityCheck:
    """Compare run identities and report every mismatched scientific dimension."""
    resolved_identities = tuple(identities)
    if len(resolved_identities) == 0:
        raise ValueError("at least one run compatibility identity is required")
    if source_labels:
        labels = tuple(str(label) for label in source_labels)
        if len(labels) != len(resolved_identities):
            raise ValueError("source_labels must match identities length")
    else:
        labels = tuple("run_{}".format(index) for index in range(len(resolved_identities)))
    if len(set(labels)) != len(labels):
        raise ValueError("source_labels cannot contain duplicates")

    reference = resolved_identities[0]
    mismatches: dict[str, dict[str, str]] = {}
    for field_name in COMPATIBILITY_DIMENSIONS:
        values_by_source = {
            label: str(getattr(identity, field_name))
            for label, identity in zip(labels, resolved_identities)
        }
        if len(set(values_by_source.values())) != 1:
            mismatches[field_name] = values_by_source
    return RunCompatibilityCheck(
        compatible=len(mismatches) == 0,
        reference_identity_sha256=reference.identity_sha256,
        identity_sha256_by_source={
            label: identity.identity_sha256
            for label, identity in zip(labels, resolved_identities)
        },
        mismatches=mismatches,
    )


def require_compatible_run_identities(
    identities: Sequence[RunCompatibilityIdentity],
    *,
    source_labels: Sequence[str] = (),
) -> RunCompatibilityCheck:
    """Return a compatibility report or raise before any artifacts are combined."""
    result = compare_run_compatibility_identities(identities, source_labels=source_labels)
    if not result.compatible:
        raise IncompatibleRunArtifactsError(
            "run artifacts are incompatible under {}: {}".format(
                STRICT_RUN_COMPATIBILITY_POLICY,
                json.dumps(result.to_dict()["mismatches"], sort_keys=True),
            )
        )
    return result


def compatibility_identity_from_manifest(manifest: Mapping[str, Any]) -> RunCompatibilityIdentity:
    """Load the strict compatibility identity embedded in a run manifest."""
    metadata = manifest.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise TypeError("run manifest metadata must be an object")
    identity_payload = metadata.get(RUN_COMPATIBILITY_METADATA_KEY)
    if not isinstance(identity_payload, Mapping):
        raise IncompatibleRunArtifactsError(
            "run manifest is missing strict compatibility metadata: {}".format(RUN_COMPATIBILITY_METADATA_KEY)
        )
    return RunCompatibilityIdentity.from_dict(identity_payload)


def validate_run_metadata_compatibility(
    metadata_by_source: Mapping[str, Mapping[str, Any]],
    *,
    mode: str = "strict",
) -> RunCompatibilityValidation:
    """Validate manifest metadata before combining artifacts from multiple runs.

    ``legacy_allow_missing`` is restricted to the case where every source lacks
    an identity. A mixture of identified and unidentified runs always fails.
    """
    resolved_mode = str(mode).strip().lower()
    if resolved_mode not in RUN_COMPATIBILITY_MODES:
        raise ValueError("compatibility mode must be one of: {}".format(", ".join(sorted(RUN_COMPATIBILITY_MODES))))
    if len(metadata_by_source) == 0:
        raise ValueError("metadata_by_source cannot be empty")

    identities: list[RunCompatibilityIdentity] = []
    identity_sources: list[str] = []
    missing_sources: list[str] = []
    for source, metadata in metadata_by_source.items():
        payload = metadata.get(RUN_COMPATIBILITY_METADATA_KEY) if isinstance(metadata, Mapping) else None
        if isinstance(payload, Mapping):
            identities.append(RunCompatibilityIdentity.from_dict(payload))
            identity_sources.append(str(source))
        else:
            missing_sources.append(str(source))

    if missing_sources:
        if len(missing_sources) != len(metadata_by_source):
            raise IncompatibleRunArtifactsError(
                "cannot combine identified and unidentified run artifacts; missing identities: {}".format(
                    sorted(missing_sources)
                )
            )
        if resolved_mode != "legacy_allow_missing":
            raise IncompatibleRunArtifactsError(
                "strict compatibility requires provenance identities; missing from: {}".format(sorted(missing_sources))
            )
        return RunCompatibilityValidation(
            compatible=True,
            mode=resolved_mode,
            status="legacy_missing_allowed",
            missing_identity_sources=tuple(sorted(missing_sources)),
        )

    check = require_compatible_run_identities(identities, source_labels=identity_sources)
    return RunCompatibilityValidation(
        compatible=True,
        mode=resolved_mode,
        status="compatible",
        check=check,
    )


__all__ = [
    "COMPATIBILITY_DIMENSIONS",
    "IncompatibleRunArtifactsError",
    "RUN_COMPATIBILITY_IDENTITY_SCHEMA_VERSION",
    "RUN_COMPATIBILITY_MODES",
    "RUN_COMPATIBILITY_METADATA_KEY",
    "RunCompatibilityCheck",
    "RunCompatibilityIdentity",
    "RunCompatibilityValidation",
    "STRICT_RUN_COMPATIBILITY_POLICY",
    "build_run_compatibility_identity",
    "compare_run_compatibility_identities",
    "compatibility_identity_from_manifest",
    "read_run_compatibility_identity",
    "require_compatible_run_identities",
    "validate_run_metadata_compatibility",
    "write_run_compatibility_identity",
]
