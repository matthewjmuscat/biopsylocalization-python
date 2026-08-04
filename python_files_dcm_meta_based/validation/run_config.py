from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback when tomli is installed.
    try:
        import tomli as tomllib
    except ModuleNotFoundError:  # pragma: no cover - handled at load time.
        tomllib = None


VALIDATION_RUN_CONFIG_SCHEMA_VERSION = "validation_run_config_v1"
LEGACY_VALIDATION_JOBS_SCHEMA_VERSION = "validation_jobs_v1"

VALIDATION_DIR = Path(__file__).resolve().parent
PYTHON_ROOT = VALIDATION_DIR.parent
REPO_ROOT = PYTHON_ROOT.parent
DEFAULT_TOML_CONFIG_PATH = VALIDATION_DIR / "configs" / "validation_jobs.toml"
DEFAULT_JSON_CONFIG_PATH = VALIDATION_DIR / "configs" / "validation_jobs.json"
DEFAULT_CONFIG_PATH = DEFAULT_TOML_CONFIG_PATH if DEFAULT_TOML_CONFIG_PATH.is_file() else DEFAULT_JSON_CONFIG_PATH


@dataclass(frozen=True, slots=True)
class ValidationRunConfig:
    """Human-authored validation run profile after TOML/JSON parsing.

    TOML is the preferred hand-authored profile format. JSON remains supported
    for existing job files and generated provenance snapshots.
    """

    source_path: Path
    source_format: str
    schema_version: str
    description: str = ""
    defaults: Mapping[str, Any] = field(default_factory=dict)
    runs: Mapping[str, str] = field(default_factory=dict)
    paths: Mapping[str, str] = field(default_factory=dict)
    run_groups: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version not in {
            VALIDATION_RUN_CONFIG_SCHEMA_VERSION,
            LEGACY_VALIDATION_JOBS_SCHEMA_VERSION,
        }:
            raise ValueError(
                "Unsupported validation config schema_version "
                f"{self.schema_version!r}; expected {VALIDATION_RUN_CONFIG_SCHEMA_VERSION!r}"
            )
        object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(self, "source_format", str(self.source_format).strip().lower())
        object.__setattr__(self, "description", str(self.description or ""))
        object.__setattr__(self, "defaults", _mapping_copy(self.defaults, "defaults"))
        object.__setattr__(self, "runs", _string_mapping_copy(self.runs, "runs"))
        object.__setattr__(self, "paths", _string_mapping_copy(self.paths, "paths"))
        object.__setattr__(self, "run_groups", _run_groups_copy(self.run_groups))

    def as_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "description": self.description,
            "defaults": dict(self.defaults),
            "runs": dict(self.runs),
            "paths": dict(self.paths),
            "run_groups": {group_name: dict(group_config) for group_name, group_config in self.run_groups.items()},
        }


def _mapping_copy(value: Mapping[str, Any], source_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{source_name} must be a table/object")
    return dict(value)


def _string_mapping_copy(value: Mapping[str, Any], source_name: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{source_name} must be a table/object")
    copied: dict[str, str] = {}
    for key, raw_value in value.items():
        if not isinstance(raw_value, str):
            raise TypeError(f"{source_name}.{key} must be a string path")
        copied[str(key)] = raw_value
    return copied


def _run_groups_copy(value: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        raise TypeError("run_groups must be a table/object")
    copied: dict[str, Mapping[str, Any]] = {}
    for group_name, group_config in value.items():
        if not isinstance(group_config, Mapping):
            raise TypeError(f"run group {group_name!r} must be a table/object")
        jobs = group_config.get("jobs", [])
        if not isinstance(jobs, list):
            raise TypeError(f"run group {group_name!r} jobs must be a list")
        copied[str(group_name)] = dict(group_config)
    return copied


def _read_json_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        config = json.load(file_obj)
    if not isinstance(config, dict):
        raise TypeError(f"JSON validation config root must be an object: {path}")
    return config


def _read_toml_config(path: Path) -> dict[str, Any]:
    if tomllib is None:
        raise RuntimeError("TOML validation configs require Python 3.11+ tomllib support or the tomli package")
    with path.open("rb") as file_obj:
        config = tomllib.load(file_obj)
    if not isinstance(config, dict):
        raise TypeError(f"TOML validation config root must be a table: {path}")
    return config


def resolve_validation_config_path(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Path:
    resolved_path = Path(config_path).expanduser()
    if not resolved_path.is_absolute():
        resolved_path = REPO_ROOT / resolved_path
    return resolved_path


def load_validation_run_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> ValidationRunConfig:
    resolved_path = resolve_validation_config_path(config_path)
    suffix = resolved_path.suffix.lower()
    if suffix == ".toml":
        payload = _read_toml_config(resolved_path)
        source_format = "toml"
    elif suffix == ".json":
        payload = _read_json_config(resolved_path)
        source_format = "json"
    else:
        raise ValueError(f"Unsupported validation config extension {resolved_path.suffix!r}; use .toml or .json")

    return ValidationRunConfig(
        source_path=resolved_path,
        source_format=source_format,
        schema_version=str(payload.get("schema_version", "")).strip(),
        description=str(payload.get("description", "")),
        defaults=payload.get("defaults", {}),
        runs=payload.get("runs", {}),
        paths=payload.get("paths", {}),
        run_groups=payload.get("run_groups", {}),
    )


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_JSON_CONFIG_PATH",
    "DEFAULT_TOML_CONFIG_PATH",
    "LEGACY_VALIDATION_JOBS_SCHEMA_VERSION",
    "REPO_ROOT",
    "VALIDATION_RUN_CONFIG_SCHEMA_VERSION",
    "ValidationRunConfig",
    "load_validation_run_config",
    "resolve_validation_config_path",
]