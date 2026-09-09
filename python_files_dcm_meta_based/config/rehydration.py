"""Typed rehydration of verified scientific PipelineConfig snapshots.

The snapshot is generated provenance, not a second config authority. Rehydration
reconstructs the nested dataclasses needed by a standalone worker and verifies
that re-serializing them yields the original scientific configuration SHA.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import NoneType, UnionType
from typing import Any, Mapping, Union, get_args, get_origin, get_type_hints

from .snapshots import PIPELINE_SCIENTIFIC_CONFIG_FIELD_NAMES
from .snapshots import PipelineConfigSnapshot
from .snapshots import build_pipeline_scientific_config_snapshot
from .snapshots import read_pipeline_config_snapshot


EXPECTED_PIPELINE_SCIENTIFIC_CONFIG_TYPE_SUFFIX = ".PipelineConfig.scientific"


def rehydrate_pipeline_scientific_config_snapshot(
    snapshot_or_path: PipelineConfigSnapshot | Path | str,
) -> Any:
    """Rebuild PipelineConfig from a verified scientific snapshot.

    Fields deliberately excluded from scientific identity receive inert defaults:
    UI, output paths, validation sidecars, and runner scheduling. The reconstructed
    config is re-snapshotted and must reproduce the source SHA exactly.
    """
    snapshot = (
        snapshot_or_path
        if isinstance(snapshot_or_path, PipelineConfigSnapshot)
        else read_pipeline_config_snapshot(snapshot_or_path)
    )
    if not snapshot.config_type.endswith(EXPECTED_PIPELINE_SCIENTIFIC_CONFIG_TYPE_SUFFIX):
        raise ValueError("snapshot is not a PipelineConfig scientific snapshot: {}".format(snapshot.config_type))

    from .pipeline import ArtifactConfig
    from .pipeline import PipelineConfig
    from .pipeline import RuntimeUIConfig

    expected_fields = set(PIPELINE_SCIENTIFIC_CONFIG_FIELD_NAMES)
    payload_fields = set(snapshot.config)
    missing_fields = sorted(expected_fields.difference(payload_fields))
    unknown_fields = sorted(payload_fields.difference(expected_fields))
    if missing_fields or unknown_fields:
        raise ValueError(
            "scientific config snapshot fields differ from the supported contract: missing={} unknown={}".format(
                missing_fields,
                unknown_fields,
            )
        )

    pipeline_hints = get_type_hints(PipelineConfig)
    hydrated_fields = {
        field_name: _hydrate_value(
            snapshot.config[field_name],
            pipeline_hints[field_name],
            path="PipelineConfig.{}".format(field_name),
        )
        for field_name in PIPELINE_SCIENTIFIC_CONFIG_FIELD_NAMES
    }
    config = PipelineConfig(
        ui=RuntimeUIConfig(),
        artifacts=ArtifactConfig(),
        **hydrated_fields,
    )
    rehydrated_snapshot = build_pipeline_scientific_config_snapshot(config)
    if rehydrated_snapshot.config_sha256 != snapshot.config_sha256:
        raise ValueError(
            "rehydrated PipelineConfig does not reproduce source scientific config SHA: {} != {}".format(
                rehydrated_snapshot.config_sha256,
                snapshot.config_sha256,
            )
        )
    return config


def _hydrate_value(value: Any, annotation: Any, *, path: str) -> Any:
    if annotation is Any:
        return value
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin in (Union, UnionType):
        if value is None and NoneType in args:
            return None
        errors: list[Exception] = []
        for candidate in args:
            if candidate is NoneType:
                continue
            try:
                return _hydrate_value(value, candidate, path=path)
            except (TypeError, ValueError) as exc:
                errors.append(exc)
        raise TypeError("{} does not match any supported union member: {}".format(path, errors))

    if isinstance(annotation, type) and is_dataclass(annotation):
        if not isinstance(value, Mapping):
            raise TypeError("{} must be an object for {}".format(path, annotation.__name__))
        type_hints = get_type_hints(annotation)
        known_fields = {field.name for field in fields(annotation)}
        unknown_fields = sorted(set(value).difference(known_fields))
        if unknown_fields:
            raise ValueError("{} contains unknown fields: {}".format(path, unknown_fields))
        kwargs = {
            field.name: _hydrate_value(
                value[field.name],
                type_hints.get(field.name, Any),
                path="{}.{}".format(path, field.name),
            )
            for field in fields(annotation)
            if field.name in value
        }
        return annotation(**kwargs)

    if origin in (dict, Mapping, MappingABC):
        if not isinstance(value, Mapping):
            raise TypeError("{} must be an object".format(path))
        key_type, value_type = args if len(args) == 2 else (Any, Any)
        return {
            _hydrate_value(key, key_type, path="{}.<key>".format(path)): _hydrate_value(
                item,
                value_type,
                path="{}.{}".format(path, key),
            )
            for key, item in value.items()
        }

    if origin in (tuple, list, set, frozenset, SequenceABC):
        if isinstance(value, (str, bytes)) or not isinstance(value, SequenceABC):
            raise TypeError("{} must be an array".format(path))
        if origin is tuple and len(args) > 1 and args[-1] is not Ellipsis:
            if len(value) != len(args):
                raise ValueError("{} tuple length mismatch".format(path))
            hydrated_items = tuple(
                _hydrate_value(item, item_type, path="{}[{}]".format(path, index))
                for index, (item, item_type) in enumerate(zip(value, args))
            )
        else:
            item_type = args[0] if args else Any
            hydrated_items = tuple(
                _hydrate_value(item, item_type, path="{}[{}]".format(path, index))
                for index, item in enumerate(value)
            )
        if origin is list:
            return list(hydrated_items)
        if origin is set:
            return set(hydrated_items)
        if origin is frozenset:
            return frozenset(hydrated_items)
        return hydrated_items

    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation(value)
    if annotation is Path:
        return Path(str(value))
    if annotation is bool:
        if not isinstance(value, bool):
            raise TypeError("{} must be a boolean".format(path))
        return value
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("{} must be an integer".format(path))
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("{} must be numeric".format(path))
        return value
    if annotation is str:
        if not isinstance(value, str):
            raise TypeError("{} must be a string".format(path))
        return value
    if isinstance(annotation, type) and isinstance(value, annotation):
        return value
    return value


__all__ = [
    "EXPECTED_PIPELINE_SCIENTIFIC_CONFIG_TYPE_SUFFIX",
    "rehydrate_pipeline_scientific_config_snapshot",
]
