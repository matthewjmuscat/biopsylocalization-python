"""Runtime environment identity for reproducible scientific execution."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import platform
import re
import site
import sysconfig
from typing import Any, Mapping

from config.snapshots import canonical_sha256


RUNTIME_ENVIRONMENT_IDENTITY_SCHEMA_VERSION = "runtime_environment_identity_v2"
_READABLE_SCHEMAS = {"runtime_environment_identity_v1", RUNTIME_ENVIRONMENT_IDENTITY_SCHEMA_VERSION}


@dataclass(frozen=True, slots=True)
class RuntimeEnvironmentIdentity:
    """Python, platform, dependency, and lockfile identity for one run."""

    python_version: str
    python_implementation: str
    platform: str
    installed_distributions_sha256: str
    dependency_lock_sha256: str
    identity_sha256: str = ""
    schema_version: str = RUNTIME_ENVIRONMENT_IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version not in _READABLE_SCHEMAS:
            raise ValueError("unsupported runtime environment schema_version: {}".format(self.schema_version))
        for field_name in (
            "python_version",
            "python_implementation",
            "platform",
            "installed_distributions_sha256",
            "dependency_lock_sha256",
        ):
            if str(getattr(self, field_name)).strip() == "":
                raise ValueError("{} cannot be empty".format(field_name))
        expected_identity_sha256 = canonical_sha256(self._identity_payload())
        if self.identity_sha256 and self.identity_sha256 != expected_identity_sha256:
            raise ValueError("runtime environment identity_sha256 does not match its dimensions")
        object.__setattr__(self, "identity_sha256", expected_identity_sha256)

    def _identity_payload(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "python_version": self.python_version,
            "python_implementation": self.python_implementation,
            "platform": self.platform,
            "installed_distributions_sha256": self.installed_distributions_sha256,
            "dependency_lock_sha256": self.dependency_lock_sha256,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "identity_sha256": self.identity_sha256}


def installed_distribution_roots() -> tuple[str, ...]:
    """Return interpreter installation roots, independent of runtime ``sys.path``.

    ``site`` establishes its prefixes and user-site policy at interpreter startup,
    including system sites explicitly enabled by a virtual environment. Combine
    those roots with this interpreter's purelib/platlib installation scheme.
    Arbitrary import paths (including nested vendor directories) are not installed
    dependency roots. Paths select metadata; their locations are not hash inputs.
    """
    paths = [sysconfig.get_path("purelib"), sysconfig.get_path("platlib"), *site.getsitepackages()]
    if site.ENABLE_USER_SITE:
        paths.append(site.getusersitepackages())
    roots = set()
    for value in paths:
        path = Path(value).expanduser()
        if not path.is_absolute():
            raise ValueError("interpreter installation root must be absolute: {}".format(path))
        roots.add(str(path.resolve()))
    return tuple(sorted(roots))


def capture_runtime_environment_identity(repository_path: Path | str) -> RuntimeEnvironmentIdentity:
    """Capture v2 installed dependencies, Python/platform, and lockfile identity.

    Historical v1 artifacts remain readable and retain their original hash, but
    cannot match a newly captured v2 identity under strict compatibility.
    """
    repository_root = _resolve_repository_root(Path(repository_path))
    distributions = sorted({
        "{}=={}".format(
            re.sub(r"[-_.]+", "-", distribution.metadata["Name"].strip()).lower(),
            distribution.version.strip(),
        )
        for distribution in importlib_metadata.distributions(path=installed_distribution_roots())
    })
    installed_distributions_sha256 = canonical_sha256(distributions)
    lock_path = repository_root.joinpath("Pipfile.lock")
    dependency_lock_sha256 = _file_sha256(lock_path) if lock_path.is_file() else canonical_sha256([])
    return RuntimeEnvironmentIdentity(
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        platform=platform.platform(),
        installed_distributions_sha256=installed_distributions_sha256,
        dependency_lock_sha256=dependency_lock_sha256,
    )


def write_runtime_environment_identity(
    identity: RuntimeEnvironmentIdentity,
    output_path: Path | str,
    *,
    overwrite: bool = False,
) -> Path:
    """Write one runtime environment identity JSON artifact."""
    if not isinstance(identity, RuntimeEnvironmentIdentity):
        raise TypeError("identity must be a RuntimeEnvironmentIdentity")
    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError("runtime environment identity already exists: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(identity.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def read_runtime_environment_identity(input_path: Path | str) -> RuntimeEnvironmentIdentity:
    """Read and verify one runtime environment identity JSON artifact."""
    path = Path(input_path)
    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, Mapping):
        raise TypeError("runtime environment identity root must be an object")
    return RuntimeEnvironmentIdentity(
        schema_version=str(payload.get("schema_version", "")),
        python_version=str(payload.get("python_version", "")),
        python_implementation=str(payload.get("python_implementation", "")),
        platform=str(payload.get("platform", "")),
        installed_distributions_sha256=str(payload.get("installed_distributions_sha256", "")),
        dependency_lock_sha256=str(payload.get("dependency_lock_sha256", "")),
        identity_sha256=str(payload.get("identity_sha256", "")),
    )


def _resolve_repository_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.is_file():
        resolved = resolved.parent
    for candidate in (resolved, *resolved.parents):
        if candidate.joinpath(".git").exists():
            return candidate
    return resolved


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "RUNTIME_ENVIRONMENT_IDENTITY_SCHEMA_VERSION",
    "RuntimeEnvironmentIdentity",
    "capture_runtime_environment_identity",
    "read_runtime_environment_identity",
    "write_runtime_environment_identity",
]
