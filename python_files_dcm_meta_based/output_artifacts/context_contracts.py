"""Shared contracts for retained patient scientific context artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence
import json


SCIENTIFIC_CONTEXT_CONTRACTS_SCHEMA_VERSION = "scientific_context_contracts_v1"
PATIENT_ARTIFACT_INDEX_FILENAME = "manifest.json"
SUPPORTED_CONTEXT_ARTIFACT_FORMATS = frozenset({"json", "parquet", "zarr", "npz"})


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """Storage reference for one retained scientific context artifact."""

    artifact_id: str
    title: str
    artifact_family: str
    relative_path: str
    storage_format: str
    schema_version: str
    patient_uid: str = ""
    stage_name: str = ""
    retention_level: str = ""
    producer: str = ""
    reader: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty(self.artifact_id, "artifact_id")
        _validate_non_empty(self.artifact_family, "artifact_family")
        _validate_non_empty(self.relative_path, "relative_path")
        _validate_non_empty(self.storage_format, "storage_format")
        _validate_non_empty(self.schema_version, "schema_version")
        storage_format = _normalize_storage_format(self.storage_format)
        object.__setattr__(self, "artifact_id", str(self.artifact_id).strip())
        object.__setattr__(self, "title", str(self.title).strip())
        object.__setattr__(self, "artifact_family", str(self.artifact_family).strip())
        object.__setattr__(self, "relative_path", _normalize_relative_path(self.relative_path))
        object.__setattr__(self, "storage_format", storage_format)
        object.__setattr__(self, "schema_version", str(self.schema_version).strip())
        object.__setattr__(self, "patient_uid", str(self.patient_uid).strip())
        object.__setattr__(self, "stage_name", str(self.stage_name).strip())
        object.__setattr__(self, "retention_level", str(self.retention_level).strip())
        object.__setattr__(self, "producer", str(self.producer).strip())
        object.__setattr__(self, "reader", str(self.reader).strip())
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready artifact reference dictionary."""
        return _json_ready(asdict(self))


@dataclass(frozen=True, slots=True)
class ArrayArtifactSpec:
    """Shape, storage, and meaning of one retained array dataset."""

    artifact_ref: ArtifactRef
    dataset_name: str
    symbolic_shape: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    units: str = ""
    coordinate_frame: str = ""
    dimension_names: tuple[str, ...] = ()
    chunk_shape: tuple[int, ...] | None = None
    compressor: str = ""
    filters: tuple[str, ...] = ()
    fill_value: Any | None = None
    checksum: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_ref, ArtifactRef):
            raise TypeError("artifact_ref must be an ArtifactRef")
        _validate_non_empty(self.dataset_name, "dataset_name")
        _validate_non_empty(self.dtype, "dtype")
        shape = tuple(int(dimension) for dimension in tuple(self.shape))
        if any(dimension < 0 for dimension in shape):
            raise ValueError("shape dimensions cannot be negative")
        symbolic_shape = tuple(str(dimension).strip() for dimension in tuple(self.symbolic_shape))
        if len(symbolic_shape) != len(shape):
            raise ValueError("symbolic_shape must have the same rank as shape")
        dimension_names = tuple(str(name).strip() for name in tuple(self.dimension_names))
        if len(dimension_names) not in (0, len(shape)):
            raise ValueError("dimension_names must be empty or have the same rank as shape")
        chunk_shape = self.chunk_shape
        if chunk_shape is not None:
            chunk_shape = tuple(int(dimension) for dimension in tuple(chunk_shape))
            if len(chunk_shape) != len(shape):
                raise ValueError("chunk_shape must have the same rank as shape")
            if any(dimension <= 0 for dimension in chunk_shape):
                raise ValueError("chunk_shape dimensions must be positive")
        object.__setattr__(self, "dataset_name", str(self.dataset_name).strip())
        object.__setattr__(self, "symbolic_shape", symbolic_shape)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", str(self.dtype).strip())
        object.__setattr__(self, "units", str(self.units).strip())
        object.__setattr__(self, "coordinate_frame", str(self.coordinate_frame).strip())
        object.__setattr__(self, "dimension_names", dimension_names)
        object.__setattr__(self, "chunk_shape", chunk_shape)
        object.__setattr__(self, "compressor", str(self.compressor).strip())
        object.__setattr__(self, "filters", tuple(str(item).strip() for item in tuple(self.filters)))
        object.__setattr__(self, "checksum", str(self.checksum).strip())
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready array spec dictionary."""
        return _json_ready(asdict(self))


@dataclass(frozen=True, slots=True)
class TableArtifactSpec:
    """Shape and identity contract for one retained table artifact."""

    artifact_ref: ArtifactRef
    table_name: str
    columns: tuple[str, ...]
    row_count: int | None = None
    primary_key_columns: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_ref, ArtifactRef):
            raise TypeError("artifact_ref must be an ArtifactRef")
        _validate_non_empty(self.table_name, "table_name")
        columns = tuple(str(column).strip() for column in tuple(self.columns))
        if len(columns) == 0:
            raise ValueError("columns cannot be empty")
        if any(column == "" for column in columns):
            raise ValueError("columns cannot contain empty names")
        row_count = self.row_count
        if row_count is not None:
            row_count = int(row_count)
            if row_count < 0:
                raise ValueError("row_count cannot be negative")
        object.__setattr__(self, "table_name", str(self.table_name).strip())
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "row_count", row_count)
        object.__setattr__(self, "primary_key_columns", tuple(str(column).strip() for column in self.primary_key_columns))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready table spec dictionary."""
        return _json_ready(asdict(self))


@dataclass(frozen=True, slots=True)
class PatientArtifactIndex:
    """Lightweight patient/run index of retained scientific context artifacts."""

    patient_uid: str
    run_id: str = ""
    schema_version: str = SCIENTIFIC_CONTEXT_CONTRACTS_SCHEMA_VERSION
    artifacts: tuple[ArtifactRef, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty(self.patient_uid, "patient_uid")
        artifacts = tuple(self.artifacts)
        if any(not isinstance(artifact, ArtifactRef) for artifact in artifacts):
            raise TypeError("artifacts must contain ArtifactRef objects")
        artifact_ids = [artifact.artifact_id for artifact in artifacts]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("PatientArtifactIndex cannot contain duplicate artifact IDs")
        object.__setattr__(self, "patient_uid", str(self.patient_uid).strip())
        object.__setattr__(self, "run_id", str(self.run_id).strip())
        object.__setattr__(self, "schema_version", str(self.schema_version).strip())
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def artifacts_by_id(self) -> dict[str, ArtifactRef]:
        """Return artifact references keyed by stable artifact ID."""
        return {artifact.artifact_id: artifact for artifact in self.artifacts}

    def add_artifact(self, artifact: ArtifactRef) -> "PatientArtifactIndex":
        """Return a new index with one additional artifact reference."""
        if artifact.artifact_id in self.artifacts_by_id:
            raise ValueError("duplicate artifact ID: {}".format(artifact.artifact_id))
        return replace(self, artifacts=(*self.artifacts, artifact))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready patient artifact index manifest."""
        return {
            "schema_version": self.schema_version,
            "patient_uid": self.patient_uid,
            "run_id": self.run_id,
            "artifact_count": len(self.artifacts),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "metadata": _json_ready(dict(self.metadata)),
        }


def artifact_ref_from_dict(payload: Mapping[str, Any]) -> ArtifactRef:
    """Rebuild an ArtifactRef from a manifest dictionary."""
    return ArtifactRef(
        artifact_id=str(payload["artifact_id"]),
        title=str(payload.get("title", "")),
        artifact_family=str(payload["artifact_family"]),
        relative_path=str(payload["relative_path"]),
        storage_format=str(payload["storage_format"]),
        schema_version=str(payload["schema_version"]),
        patient_uid=str(payload.get("patient_uid", "")),
        stage_name=str(payload.get("stage_name", "")),
        retention_level=str(payload.get("retention_level", "")),
        producer=str(payload.get("producer", "")),
        reader=str(payload.get("reader", "")),
        metadata=dict(payload.get("metadata", {})),
    )


def patient_artifact_index_from_dict(payload: Mapping[str, Any]) -> PatientArtifactIndex:
    """Rebuild a PatientArtifactIndex from a manifest dictionary."""
    return PatientArtifactIndex(
        patient_uid=str(payload["patient_uid"]),
        run_id=str(payload.get("run_id", "")),
        schema_version=str(payload.get("schema_version", SCIENTIFIC_CONTEXT_CONTRACTS_SCHEMA_VERSION)),
        artifacts=tuple(artifact_ref_from_dict(artifact) for artifact in tuple(payload.get("artifacts", ()))),
        metadata=dict(payload.get("metadata", {})),
    )


def write_patient_artifact_index(
    index: PatientArtifactIndex,
    output_path: Path | str,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a patient scientific context artifact index JSON file."""
    if not isinstance(index, PatientArtifactIndex):
        raise TypeError("index must be a PatientArtifactIndex")
    resolved_output_path = Path(output_path)
    if resolved_output_path.exists() and not overwrite:
        raise FileExistsError("patient artifact index already exists: {}".format(resolved_output_path))
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_output_path.open("w", encoding="utf-8") as output_file:
        json.dump(index.to_dict(), output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    return resolved_output_path


def read_patient_artifact_index(input_path: Path | str) -> PatientArtifactIndex:
    """Read a patient scientific context artifact index JSON file."""
    resolved_input_path = Path(input_path)
    with resolved_input_path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    return patient_artifact_index_from_dict(payload)


def _normalize_relative_path(value: str) -> str:
    path = Path(str(value).strip())
    if path.is_absolute():
        raise ValueError("relative_path must be relative")
    normalized = path.as_posix().strip()
    if normalized in ("", "."):
        raise ValueError("relative_path cannot be empty")
    return normalized


def _normalize_storage_format(value: str) -> str:
    storage_format = str(value).strip().lower().lstrip(".")
    if storage_format not in SUPPORTED_CONTEXT_ARTIFACT_FORMATS:
        raise ValueError("unsupported context artifact storage format: {}".format(value))
    return storage_format


def _validate_non_empty(value: str, field_name: str) -> None:
    if str(value).strip() == "":
        raise ValueError("{} cannot be empty".format(field_name))


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value