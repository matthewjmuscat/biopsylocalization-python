"""Deterministic snapshots and fingerprints for typed scientific configuration.

This module is deliberately independent of the concrete PipelineConfig classes.
It accepts pure-data dataclasses and converts them to canonical JSON so run
manifests can identify the exact resolved scientific configuration. It is a
provenance boundary, not a second configuration authority.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PIPELINE_CONFIG_SNAPSHOT_SCHEMA_VERSION = "pipeline_config_snapshot_v1"
PIPELINE_SCIENTIFIC_CONFIG_FIELD_NAMES = (
    "preprocessing",
    "replay",
    "guidance_maps",
    "optimizer",
    "random_seeds",
    "mc",
    "legacy_refs",
    "structure_registry",
    "bootstrap",
    "grid_preprocessing",
    "biopsy",
)


@dataclass(frozen=True, slots=True)
class PipelineConfigSnapshot:
    """Canonical resolved scientific-config payload and its SHA-256 identity."""

    config_type: str
    config: Mapping[str, Any]
    config_sha256: str
    schema_version: str = PIPELINE_CONFIG_SNAPSHOT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PIPELINE_CONFIG_SNAPSHOT_SCHEMA_VERSION:
            raise ValueError("unsupported pipeline config snapshot schema_version: {}".format(self.schema_version))
        config_type = str(self.config_type).strip()
        if config_type == "":
            raise ValueError("config_type cannot be empty")
        config = canonical_json_value(self.config)
        expected_sha256 = canonical_sha256(config)
        if str(self.config_sha256).strip() != expected_sha256:
            raise ValueError("pipeline config snapshot fingerprint does not match its config payload")
        object.__setattr__(self, "config_type", config_type)
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "config_sha256", expected_sha256)

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON representation written to run provenance."""
        return {
            "schema_version": self.schema_version,
            "config_type": self.config_type,
            "config_sha256": self.config_sha256,
            "config": dict(self.config),
        }


def build_pipeline_config_snapshot(config: Any) -> PipelineConfigSnapshot:
    """Build a deterministic snapshot from a pure-data config dataclass."""
    if not is_dataclass(config) or isinstance(config, type):
        raise TypeError("config must be a dataclass instance")
    config_payload = canonical_json_value(config)
    return PipelineConfigSnapshot(
        config_type="{}.{}".format(type(config).__module__, type(config).__qualname__),
        config=config_payload,
        config_sha256=canonical_sha256(config_payload),
    )


def build_pipeline_scientific_config_snapshot(config: Any) -> PipelineConfigSnapshot:
    """Snapshot only fields that define scientific behavior or data semantics.

    UI state, output paths, validation toggles, patient selection, and execution
    scheduling are intentionally excluded. This allows disjoint split runs to
    share a scientific identity while retaining strict checks on algorithms,
    parameters, seeds, structure policy, and preprocessing semantics.
    """
    if not is_dataclass(config) or isinstance(config, type):
        raise TypeError("config must be a dataclass instance")
    missing_fields = [field_name for field_name in PIPELINE_SCIENTIFIC_CONFIG_FIELD_NAMES if not hasattr(config, field_name)]
    if missing_fields:
        raise ValueError("config is missing scientific fields: {}".format(missing_fields))
    config_payload = canonical_json_value(
        {
            field_name: getattr(config, field_name)
            for field_name in PIPELINE_SCIENTIFIC_CONFIG_FIELD_NAMES
        }
    )
    return PipelineConfigSnapshot(
        config_type="{}.{}.scientific".format(type(config).__module__, type(config).__qualname__),
        config=config_payload,
        config_sha256=canonical_sha256(config_payload),
    )


def write_pipeline_config_snapshot(
    snapshot: PipelineConfigSnapshot,
    output_path: Path | str,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a resolved config snapshot as canonical, inspectable JSON."""
    if not isinstance(snapshot, PipelineConfigSnapshot):
        raise TypeError("snapshot must be a PipelineConfigSnapshot")
    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError("pipeline config snapshot already exists: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def read_pipeline_config_snapshot(input_path: Path | str) -> PipelineConfigSnapshot:
    """Read and verify a resolved config snapshot without constructing runtime config."""
    path = Path(input_path)
    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, Mapping):
        raise TypeError("pipeline config snapshot root must be an object")
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise TypeError("pipeline config snapshot config must be an object")
    return PipelineConfigSnapshot(
        schema_version=str(payload.get("schema_version", "")),
        config_type=str(payload.get("config_type", "")),
        config=config,
        config_sha256=str(payload.get("config_sha256", "")),
    )


def canonical_sha256(value: Any) -> str:
    """Return SHA-256 over the canonical JSON encoding of a pure-data value."""
    payload = canonical_json_value(value)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_json_value(value: Any, *, _path: str = "config") -> Any:
    """Convert supported pure-data values to deterministic JSON-compatible data.

    Unsupported runtime objects fail closed. This prevents file handles, pools,
    GUI objects, arrays, or arbitrary object repr strings from contaminating a
    scientific configuration fingerprint.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: canonical_json_value(getattr(value, field.name), _path="{}.{}".format(_path, field.name))
            for field in fields(value)
        }
    if isinstance(value, Enum):
        return canonical_json_value(value.value, _path=_path)
    if isinstance(value, Path):
        return value.as_posix()
    if type(value).__module__.split(".")[0] == "numpy":
        if hasattr(value, "tolist"):
            return canonical_json_value(value.tolist(), _path=_path)
        if hasattr(value, "item"):
            return canonical_json_value(value.item(), _path=_path)
    if isinstance(value, Mapping):
        canonical_mapping: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("{} contains non-string mapping key {!r}".format(_path, key))
            canonical_mapping[key] = canonical_json_value(item, _path="{}.{}".format(_path, key))
        return canonical_mapping
    if isinstance(value, (tuple, list)):
        return [canonical_json_value(item, _path="{}[]".format(_path)) for item in value]
    if isinstance(value, (set, frozenset)):
        canonical_items = [canonical_json_value(item, _path="{}[]".format(_path)) for item in value]
        return sorted(canonical_items, key=lambda item: json.dumps(item, sort_keys=True, allow_nan=False))
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(
        "{} contains unsupported config value of type {}.{}".format(
            _path,
            type(value).__module__,
            type(value).__qualname__,
        )
    )


__all__ = [
    "PIPELINE_CONFIG_SNAPSHOT_SCHEMA_VERSION",
    "PIPELINE_SCIENTIFIC_CONFIG_FIELD_NAMES",
    "PipelineConfigSnapshot",
    "build_pipeline_config_snapshot",
    "build_pipeline_scientific_config_snapshot",
    "canonical_json_value",
    "canonical_sha256",
    "read_pipeline_config_snapshot",
    "write_pipeline_config_snapshot",
]
