from __future__ import annotations

"""Run-level index of manifest artifacts produced during execution.

The manifest catalog answers which manifest contracts the codebase knows about.
This module answers which manifest objects a concrete run produced, skipped,
failed to produce, or constructed without writing to disk.
"""

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .manifest_catalog import manifest_contracts_by_key


RUN_MANIFEST_INDEX_SCHEMA_VERSION = "run_manifest_index_v1"
RUN_MANIFEST_INDEX_FILENAME = "run_manifest_index.json"
RUN_MANIFEST_INDEX_DIR_NAME = "manifests"

MANIFEST_STATUS_WRITTEN = "written"
MANIFEST_STATUS_CONSTRUCTED_NOT_WRITTEN = "constructed_not_written"
MANIFEST_STATUS_SKIPPED = "skipped"
MANIFEST_STATUS_FAILED = "failed"
MANIFEST_PRODUCED_STATUSES = frozenset(
    {
        MANIFEST_STATUS_WRITTEN,
        MANIFEST_STATUS_CONSTRUCTED_NOT_WRITTEN,
        MANIFEST_STATUS_SKIPPED,
        MANIFEST_STATUS_FAILED,
    }
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class ManifestIndexEntry:
    """One manifest-related event recorded for a concrete run."""

    manifest_key: str
    produced_status: str
    manifest_path: str = ""
    path_is_relative_to_run_root: bool = True
    path_exists_at_index_write: bool = False
    catalog_status: str = "unknown_contract"
    title: str = ""
    scope: str = ""
    artifact_data_class: str = ""
    payload_format: str = ""
    manifest_schema_version: str = ""
    producer: str = ""
    patient_uid: str = ""
    stage_name: str = ""
    generated_utc: str = ""
    notes: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty(self.manifest_key, "manifest_key")
        if self.produced_status not in MANIFEST_PRODUCED_STATUSES:
            raise ValueError(
                "produced_status must be one of {}; got {!r}".format(
                    sorted(MANIFEST_PRODUCED_STATUSES),
                    self.produced_status,
                )
            )
        if self.produced_status == MANIFEST_STATUS_WRITTEN and str(self.manifest_path).strip() == "":
            raise ValueError("written manifest index entries must include manifest_path")
        object.__setattr__(self, "manifest_key", str(self.manifest_key).strip())
        object.__setattr__(self, "produced_status", str(self.produced_status).strip())
        object.__setattr__(self, "manifest_path", str(self.manifest_path).strip())
        object.__setattr__(self, "catalog_status", str(self.catalog_status).strip())
        object.__setattr__(self, "metadata", dict(self.metadata))
        if str(self.generated_utc).strip() == "":
            object.__setattr__(self, "generated_utc", _utc_now_iso())

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready entry dictionary."""
        row = asdict(self)
        row["schema_version"] = RUN_MANIFEST_INDEX_SCHEMA_VERSION
        return _json_safe(row)


class ManifestIndexRecorder:
    """Accumulate manifest index entries for one run and write the run index."""

    def __init__(
        self,
        run_root: Path | str,
        *,
        run_id: str = "",
        output_path: Path | str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.run_root = Path(run_root)
        self.run_id = str(run_id).strip()
        self.output_path = Path(output_path) if output_path is not None else default_run_manifest_index_path(self.run_root)
        self.metadata = {} if metadata is None else dict(metadata)
        self._entries: list[ManifestIndexEntry] = []

    @property
    def entries(self) -> tuple[ManifestIndexEntry, ...]:
        return tuple(self._entries)

    def record_entry(self, entry: ManifestIndexEntry) -> ManifestIndexEntry:
        self._entries.append(entry)
        return entry

    def record_written_manifest(
        self,
        manifest_key: str,
        manifest_path: Path | str,
        *,
        manifest_schema_version: str = "",
        patient_uid: str = "",
        stage_name: str = "",
        metadata: Mapping[str, Any] | None = None,
        notes: str = "",
    ) -> ManifestIndexEntry:
        entry = manifest_index_entry(
            manifest_key,
            MANIFEST_STATUS_WRITTEN,
            manifest_path=manifest_path,
            run_root=self.run_root,
            manifest_schema_version=manifest_schema_version,
            patient_uid=patient_uid,
            stage_name=stage_name,
            metadata=metadata,
            notes=notes,
        )
        return self.record_entry(entry)

    def record_constructed_manifest(
        self,
        manifest_key: str,
        *,
        manifest_schema_version: str = "",
        patient_uid: str = "",
        stage_name: str = "",
        metadata: Mapping[str, Any] | None = None,
        notes: str = "",
    ) -> ManifestIndexEntry:
        entry = manifest_index_entry(
            manifest_key,
            MANIFEST_STATUS_CONSTRUCTED_NOT_WRITTEN,
            run_root=self.run_root,
            manifest_schema_version=manifest_schema_version,
            patient_uid=patient_uid,
            stage_name=stage_name,
            metadata=metadata,
            notes=notes,
        )
        return self.record_entry(entry)

    def record_skipped_manifest(
        self,
        manifest_key: str,
        *,
        patient_uid: str = "",
        stage_name: str = "",
        metadata: Mapping[str, Any] | None = None,
        notes: str = "",
    ) -> ManifestIndexEntry:
        entry = manifest_index_entry(
            manifest_key,
            MANIFEST_STATUS_SKIPPED,
            run_root=self.run_root,
            patient_uid=patient_uid,
            stage_name=stage_name,
            metadata=metadata,
            notes=notes,
        )
        return self.record_entry(entry)

    def record_failed_manifest(
        self,
        manifest_key: str,
        *,
        manifest_path: Path | str | None = None,
        patient_uid: str = "",
        stage_name: str = "",
        metadata: Mapping[str, Any] | None = None,
        notes: str = "",
    ) -> ManifestIndexEntry:
        entry = manifest_index_entry(
            manifest_key,
            MANIFEST_STATUS_FAILED,
            manifest_path=manifest_path,
            run_root=self.run_root,
            patient_uid=patient_uid,
            stage_name=stage_name,
            metadata=metadata,
            notes=notes,
        )
        return self.record_entry(entry)

    def write(self, *, overwrite: bool = True) -> Path:
        return write_run_manifest_index(
            self.entries,
            self.output_path,
            run_id=self.run_id,
            run_root=self.run_root,
            metadata=self.metadata,
            overwrite=overwrite,
        )


def default_run_manifest_index_path(run_root: Path | str) -> Path:
    """Return the preferred run-level manifest index path."""
    return Path(run_root).joinpath(RUN_MANIFEST_INDEX_DIR_NAME, RUN_MANIFEST_INDEX_FILENAME)


def manifest_index_entry(
    manifest_key: str,
    produced_status: str,
    *,
    manifest_path: Path | str | None = None,
    run_root: Path | str | None = None,
    manifest_schema_version: str = "",
    producer: str = "",
    scope: str = "",
    artifact_data_class: str = "",
    payload_format: str = "",
    patient_uid: str = "",
    stage_name: str = "",
    metadata: Mapping[str, Any] | None = None,
    notes: str = "",
) -> ManifestIndexEntry:
    """Build a manifest index entry, enriching it from the catalog when known."""
    contract = manifest_contracts_by_key().get(str(manifest_key).strip())
    manifest_path_text, path_is_relative, path_exists = _manifest_path_info(manifest_path, run_root)
    if contract is None:
        catalog_status = "unknown_contract"
        title = ""
        resolved_scope = scope
        resolved_artifact_data_class = artifact_data_class
        resolved_payload_format = payload_format
        resolved_producer = producer
    else:
        catalog_status = "cataloged"
        title = contract.title
        resolved_scope = scope or contract.scope
        resolved_artifact_data_class = artifact_data_class or contract.artifact_data_class
        resolved_payload_format = payload_format or contract.payload_format
        resolved_producer = producer or contract.producer

    return ManifestIndexEntry(
        manifest_key=str(manifest_key).strip(),
        produced_status=produced_status,
        manifest_path=manifest_path_text,
        path_is_relative_to_run_root=path_is_relative,
        path_exists_at_index_write=path_exists,
        catalog_status=catalog_status,
        title=title,
        scope=resolved_scope,
        artifact_data_class=resolved_artifact_data_class,
        payload_format=resolved_payload_format,
        manifest_schema_version=str(manifest_schema_version).strip(),
        producer=resolved_producer,
        patient_uid=str(patient_uid).strip(),
        stage_name=str(stage_name).strip(),
        notes=str(notes).strip(),
        metadata={} if metadata is None else dict(metadata),
    )


def build_run_manifest_index(
    entries: Sequence[ManifestIndexEntry],
    *,
    run_id: str = "",
    run_root: Path | str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-ready run manifest index payload."""
    entry_dicts = [entry.to_dict() for entry in entries]
    summary = summarize_manifest_index_entries(entries)
    return {
        "schema_version": RUN_MANIFEST_INDEX_SCHEMA_VERSION,
        "generated_utc": _utc_now_iso(),
        "run_id": str(run_id).strip(),
        "run_root": "" if run_root is None else Path(run_root).as_posix(),
        "metadata": _json_safe({} if metadata is None else dict(metadata)),
        "manifest_count": len(entry_dicts),
        "summary": summary,
        "manifests": entry_dicts,
    }


def summarize_manifest_index_entries(entries: Sequence[ManifestIndexEntry]) -> dict[str, Any]:
    """Summarize manifest index entries by status, key, and scope."""
    return {
        "schema_version": RUN_MANIFEST_INDEX_SCHEMA_VERSION,
        "manifest_count": len(entries),
        "produced_status_counts": dict(Counter(entry.produced_status for entry in entries)),
        "manifest_key_counts": dict(Counter(entry.manifest_key for entry in entries)),
        "scope_counts": dict(Counter(entry.scope for entry in entries)),
        "catalog_status_counts": dict(Counter(entry.catalog_status for entry in entries)),
        "written_manifest_count": sum(1 for entry in entries if entry.produced_status == MANIFEST_STATUS_WRITTEN),
        "existing_path_count": sum(1 for entry in entries if entry.path_exists_at_index_write),
    }


def write_run_manifest_index(
    entries: Sequence[ManifestIndexEntry],
    output_path: Path | str,
    *,
    run_id: str = "",
    run_root: Path | str | None = None,
    metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Write the run-level manifest index JSON artifact."""
    resolved_output_path = Path(output_path)
    if resolved_output_path.exists() and not overwrite:
        raise FileExistsError(f"run manifest index already exists: {resolved_output_path}")
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_run_manifest_index(entries, run_id=run_id, run_root=run_root, metadata=metadata)
    with resolved_output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")
    return resolved_output_path


def read_run_manifest_index(path: Path | str) -> dict[str, Any]:
    """Read and minimally validate a run manifest index JSON artifact."""
    with Path(path).open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise TypeError(f"run manifest index root must be a JSON object: {path}")
    schema_version = str(payload.get("schema_version", ""))
    if schema_version != RUN_MANIFEST_INDEX_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported run manifest index schema_version {schema_version!r}; "
            f"expected {RUN_MANIFEST_INDEX_SCHEMA_VERSION!r}"
        )
    manifests = payload.get("manifests", [])
    if not isinstance(manifests, list):
        raise TypeError(f"run manifest index manifests must be a list: {path}")
    return payload


def manifest_index_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return CSV-friendly rows from a run manifest index payload."""
    run_id = str(payload.get("run_id", ""))
    run_root = str(payload.get("run_root", ""))
    rows: list[dict[str, Any]] = []
    for entry in payload.get("manifests", []):
        if not isinstance(entry, Mapping):
            continue
        rows.append(
            {
                "schema_version": RUN_MANIFEST_INDEX_SCHEMA_VERSION,
                "run_id": run_id,
                "run_root": run_root,
                "manifest_key": entry.get("manifest_key", ""),
                "produced_status": entry.get("produced_status", ""),
                "manifest_path": entry.get("manifest_path", ""),
                "path_is_relative_to_run_root": entry.get("path_is_relative_to_run_root", ""),
                "path_exists_at_index_write": entry.get("path_exists_at_index_write", ""),
                "catalog_status": entry.get("catalog_status", ""),
                "scope": entry.get("scope", ""),
                "artifact_data_class": entry.get("artifact_data_class", ""),
                "payload_format": entry.get("payload_format", ""),
                "manifest_schema_version": entry.get("manifest_schema_version", ""),
                "producer": entry.get("producer", ""),
                "patient_uid": entry.get("patient_uid", ""),
                "stage_name": entry.get("stage_name", ""),
                "generated_utc": entry.get("generated_utc", ""),
                "notes": entry.get("notes", ""),
            }
        )
    return rows


def _manifest_path_info(manifest_path: Path | str | None, run_root: Path | str | None) -> tuple[str, bool, bool]:
    if manifest_path is None or str(manifest_path).strip() == "":
        return "", True, False

    path = Path(manifest_path).expanduser()
    if run_root is None:
        return path.as_posix(), not path.is_absolute(), path.exists()

    resolved_run_root = Path(run_root).expanduser().resolve(strict=False)
    resolved_path = path if path.is_absolute() else resolved_run_root / path
    path_exists = resolved_path.exists()
    try:
        relative_path = resolved_path.resolve(strict=False).relative_to(resolved_run_root)
        return relative_path.as_posix(), True, path_exists
    except ValueError:
        return resolved_path.as_posix(), False, path_exists


def _validate_non_empty(value: str, field_name: str) -> None:
    if str(value).strip() == "":
        raise ValueError(f"{field_name} cannot be empty")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)