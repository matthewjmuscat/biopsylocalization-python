from __future__ import annotations

"""Expected-artifact policy for validation and post-run assembly.

The schema registry defines table contracts. This module interprets those
contracts for a concrete validation surface so disabled sidecars and optional
products are not treated the same way as missing required artifacts.
"""

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Iterable, Sequence

from .assembly_planner import OutputAssemblyPlan
from .schema_registry import OutputTableSpec


EXPECTED_ARTIFACT_POLICY_SCHEMA_VERSION = "expected_artifact_policy_v1"


class ExpectedArtifactStatus(str, Enum):
    """Validation expectation for an output artifact in a selected run profile."""

    REQUIRED = "required"
    OPTIONAL = "optional"
    VALIDATION_ONLY = "validation_only"
    DISABLED_BY_CONFIG = "disabled_by_config"
    NOT_APPLICABLE = "not_applicable"
    DEPRECATED = "deprecated"
    DOWNSTREAM_CALCULABLE = "downstream_calculable"
    UNKNOWN = "unknown"


class MissingArtifactSeverity(str, Enum):
    """How a missing artifact should affect validation status."""

    NONE = "none"
    INFO = "info"
    WARNING = "warning"
    FAILURE = "failure"


@dataclass(frozen=True, slots=True)
class ExpectedArtifactDecision:
    """Policy decision for one artifact on one validation surface."""

    table_id: str
    table_name: str
    artifact_scope: str
    table_family: str
    retention_policy: str
    validation_status: str
    expected_artifact_status: str
    missing_artifact_severity: str
    missing_artifact_is_failure: bool
    expected_artifact_notes: str
    schema_version: str = EXPECTED_ARTIFACT_POLICY_SCHEMA_VERSION

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_identifiers(values: Iterable[str]) -> frozenset[str]:
    return frozenset(str(value).strip() for value in values if str(value).strip())


def _known_presence_values(*values: bool | None) -> tuple[bool, ...]:
    return tuple(value for value in values if value is not None)


def _any_missing(values: Sequence[bool]) -> bool:
    return any(value is False for value in values)


def _any_present(values: Sequence[bool]) -> bool:
    return any(value is True for value in values)


def _all_known_absent(values: Sequence[bool]) -> bool:
    return bool(values) and all(value is False for value in values)


def _status_for_retention_policy(retention_policy: str) -> ExpectedArtifactStatus:
    if retention_policy == "validation_only":
        return ExpectedArtifactStatus.VALIDATION_ONLY
    if retention_policy == "retain_validation_only":
        return ExpectedArtifactStatus.VALIDATION_ONLY
    if retention_policy == "downstream_calculable":
        return ExpectedArtifactStatus.DOWNSTREAM_CALCULABLE
    if retention_policy == "migrate_to_manifest":
        return ExpectedArtifactStatus.OPTIONAL
    if retention_policy == "deprecated":
        return ExpectedArtifactStatus.DEPRECATED
    if retention_policy in {"retain_core", "reimplement_later"}:
        return ExpectedArtifactStatus.REQUIRED
    if not retention_policy:
        return ExpectedArtifactStatus.UNKNOWN
    return ExpectedArtifactStatus.REQUIRED


def _missing_severity(expected_status: ExpectedArtifactStatus,
                      presence_values: Sequence[bool]) -> MissingArtifactSeverity:
    if not _any_missing(presence_values):
        return MissingArtifactSeverity.NONE
    if expected_status == ExpectedArtifactStatus.DISABLED_BY_CONFIG:
        return MissingArtifactSeverity.NONE
    if expected_status in {
        ExpectedArtifactStatus.DEPRECATED,
        ExpectedArtifactStatus.DOWNSTREAM_CALCULABLE,
        ExpectedArtifactStatus.NOT_APPLICABLE,
        ExpectedArtifactStatus.OPTIONAL,
    } and not _any_present(presence_values):
        return MissingArtifactSeverity.INFO
    if expected_status in {ExpectedArtifactStatus.REQUIRED, ExpectedArtifactStatus.VALIDATION_ONLY}:
        return MissingArtifactSeverity.FAILURE
    if _any_present(presence_values):
        return MissingArtifactSeverity.FAILURE
    return MissingArtifactSeverity.WARNING


def classify_expected_artifact(*,
                               table_id: str,
                               table_name: str,
                               artifact_scope: str,
                               table_family: str,
                               retention_policy: str,
                               validation_status: str,
                               present_in_baseline: bool | None = None,
                               present_in_candidate: bool | None = None,
                               active_validation_artifact_ids: Iterable[str] = (),
                               active_validation_artifact_names: Iterable[str] = ()) -> ExpectedArtifactDecision:
    """Classify one artifact for a validation surface.

    Presence values are optional so callers can use the same function for
    one-sided assembly checks and two-sided parity checks.
    """

    identifiers = _normalize_identifiers((table_id, table_name))
    active_identifiers = _normalize_identifiers(active_validation_artifact_ids)
    active_names = _normalize_identifiers(active_validation_artifact_names)
    presence_values = _known_presence_values(present_in_baseline, present_in_candidate)
    expected_status = _status_for_retention_policy(retention_policy)
    active_validation_artifact = bool(identifiers & (active_identifiers | active_names))

    notes = "Registry retention policy determines this artifact expectation."
    if expected_status == ExpectedArtifactStatus.VALIDATION_ONLY:
        if not active_validation_artifact and _all_known_absent(presence_values):
            expected_status = ExpectedArtifactStatus.DISABLED_BY_CONFIG
            notes = "Validation-only artifact is absent on every checked surface and is treated as disabled for this run."
        elif active_validation_artifact:
            notes = "Validation-only artifact is active for this run profile."
        else:
            notes = "Validation-only artifact appeared on at least one checked surface; compare it when both sides exist."
    elif expected_status == ExpectedArtifactStatus.DOWNSTREAM_CALCULABLE:
        notes = "Artifact can be regenerated from lower-level outputs; absence on every checked surface is non-failing."
    elif expected_status == ExpectedArtifactStatus.OPTIONAL:
        notes = "Artifact is optional or migrating to manifest metadata for this validation surface."
    elif expected_status == ExpectedArtifactStatus.DEPRECATED:
        notes = "Deprecated artifact is retained only for historical compatibility."
    elif expected_status == ExpectedArtifactStatus.UNKNOWN:
        notes = "Artifact expectation is unknown; validation treats one-sided absence conservatively."

    severity = _missing_severity(expected_status, presence_values)
    return ExpectedArtifactDecision(
        table_id=table_id,
        table_name=table_name,
        artifact_scope=artifact_scope,
        table_family=table_family,
        retention_policy=retention_policy,
        validation_status=validation_status,
        expected_artifact_status=expected_status.value,
        missing_artifact_severity=severity.value,
        missing_artifact_is_failure=severity == MissingArtifactSeverity.FAILURE,
        expected_artifact_notes=notes,
    )


def classify_expected_table_spec(spec: OutputTableSpec,
                                 *,
                                 table_name: str | None = None,
                                 present_in_baseline: bool | None = None,
                                 present_in_candidate: bool | None = None,
                                 active_validation_artifact_ids: Iterable[str] = (),
                                 active_validation_artifact_names: Iterable[str] = ()) -> ExpectedArtifactDecision:
    """Classify a registry table spec for a concrete validation surface."""

    return classify_expected_artifact(
        table_id=spec.table_id,
        table_name=table_name or spec.legacy_table_name,
        artifact_scope=spec.artifact_scope,
        table_family=spec.table_family,
        retention_policy=spec.retention_policy,
        validation_status=spec.validation_status,
        present_in_baseline=present_in_baseline,
        present_in_candidate=present_in_candidate,
        active_validation_artifact_ids=active_validation_artifact_ids,
        active_validation_artifact_names=active_validation_artifact_names,
    )


def classify_expected_assembly_plan(plan: OutputAssemblyPlan,
                                    *,
                                    present_in_baseline: bool | None = None,
                                    present_in_candidate: bool | None = None,
                                    active_validation_artifact_ids: Iterable[str] = (),
                                    active_validation_artifact_names: Iterable[str] = ()) -> ExpectedArtifactDecision:
    """Classify an assembly plan's final cohort artifact expectation."""

    return classify_expected_artifact(
        table_id=plan.final_table_id,
        table_name=plan.final_table_name,
        artifact_scope="cohort",
        table_family="",
        retention_policy=plan.retention_policy,
        validation_status=plan.validation_status,
        present_in_baseline=present_in_baseline,
        present_in_candidate=present_in_candidate,
        active_validation_artifact_ids=active_validation_artifact_ids,
        active_validation_artifact_names=active_validation_artifact_names,
    )


def expected_artifact_decision_report_fields(decision: ExpectedArtifactDecision) -> dict[str, Any]:
    """Return policy fields safe to merge into another report row."""

    return {
        "expected_artifact_policy_schema_version": decision.schema_version,
        "expected_artifact_table_id": decision.table_id,
        "expected_artifact_table_name": decision.table_name,
        "expected_artifact_scope": decision.artifact_scope,
        "expected_artifact_table_family": decision.table_family,
        "expected_artifact_retention_policy": decision.retention_policy,
        "expected_artifact_registry_validation_status": decision.validation_status,
        "expected_artifact_status": decision.expected_artifact_status,
        "missing_artifact_severity": decision.missing_artifact_severity,
        "missing_artifact_is_failure": decision.missing_artifact_is_failure,
        "expected_artifact_notes": decision.expected_artifact_notes,
    }


__all__ = [
    "EXPECTED_ARTIFACT_POLICY_SCHEMA_VERSION",
    "ExpectedArtifactDecision",
    "ExpectedArtifactStatus",
    "MissingArtifactSeverity",
    "classify_expected_artifact",
    "classify_expected_assembly_plan",
    "classify_expected_table_spec",
    "expected_artifact_decision_report_fields",
]