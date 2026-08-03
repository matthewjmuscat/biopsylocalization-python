from __future__ import annotations

"""Shared manifest contract metadata primitives."""

from dataclasses import asdict, dataclass
from typing import Any, Sequence


MANIFEST_CATALOG_SCHEMA_VERSION = "manifest_catalog_v1"


@dataclass(frozen=True, slots=True)
class ManifestContract:
    """Contract metadata for one produced or planned manifest surface."""

    manifest_key: str
    title: str
    scope: str
    artifact_data_class: str
    lifecycle_status: str
    default_relative_paths: tuple[str, ...]
    payload_format: str
    schema_version_source: str
    producer: str
    purpose: str
    tracks: tuple[str, ...]
    reader: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        _validate_non_empty(self.manifest_key, "manifest_key")
        _validate_non_empty(self.title, "title")
        _validate_non_empty(self.scope, "scope")
        _validate_non_empty(self.artifact_data_class, "artifact_data_class")
        _validate_non_empty(self.lifecycle_status, "lifecycle_status")
        _validate_non_empty(self.payload_format, "payload_format")
        _validate_non_empty(self.schema_version_source, "schema_version_source")
        _validate_non_empty(self.producer, "producer")
        _validate_non_empty(self.purpose, "purpose")
        _validate_non_empty_sequence(self.default_relative_paths, "default_relative_paths")
        _validate_non_empty_sequence(self.tracks, "tracks")
        object.__setattr__(self, "default_relative_paths", tuple(self.default_relative_paths))
        object.__setattr__(self, "tracks", tuple(self.tracks))

    def to_row(self) -> dict[str, Any]:
        """Return a CSV-friendly row for generated catalog reports."""
        row = asdict(self)
        for key, value in list(row.items()):
            if isinstance(value, tuple):
                row[key] = " | ".join(str(item) for item in value)
        row["schema_version"] = MANIFEST_CATALOG_SCHEMA_VERSION
        return row


def manifest_contract(
    manifest_key: str,
    title: str,
    scope: str,
    artifact_data_class: str,
    lifecycle_status: str,
    default_relative_paths: tuple[str, ...],
    payload_format: str,
    schema_version_source: str,
    producer: str,
    purpose: str,
    tracks: tuple[str, ...],
    *,
    reader: str = "",
    notes: str = "",
) -> ManifestContract:
    """Build one manifest contract with common validation."""
    return ManifestContract(
        manifest_key=manifest_key,
        title=title,
        scope=scope,
        artifact_data_class=artifact_data_class,
        lifecycle_status=lifecycle_status,
        default_relative_paths=default_relative_paths,
        payload_format=payload_format,
        schema_version_source=schema_version_source,
        producer=producer,
        reader=reader,
        purpose=purpose,
        tracks=tracks,
        notes=notes,
    )


def _validate_non_empty(value: str, field_name: str) -> None:
    if str(value).strip() == "":
        raise ValueError(f"{field_name} cannot be empty")


def _validate_non_empty_sequence(values: Sequence[str], field_name: str) -> None:
    if not values:
        raise ValueError(f"{field_name} cannot be empty")
    for value in values:
        if str(value).strip() == "":
            raise ValueError(f"{field_name} cannot contain empty values")