"""Headless preparation of new provenance from resolved scientific configuration.

This transitional startup boundary reuses config rehydration and the provenance
writer used by legacy startup. It neither imports main nor resolves scientific
defaults, discovers patient inputs, plans jobs, or executes scientific stages.
Historical execution identities are not reused or rewritten.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

from config.rehydration import rehydrate_pipeline_scientific_config_snapshot
from config.snapshots import canonical_sha256
from config.snapshots import read_pipeline_config_snapshot

if TYPE_CHECKING:
    from startup.run_provenance import RunProvenanceArtifacts


PREPARATION_RECORD_SCHEMA_VERSION = "standalone_scientific_preparation_v1"
DEFAULT_REPOSITORY_PATH = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class StandalonePreparationResult:
    """New run provenance and its source-reuse record, all in the destination.

    Returned only after successful preparation. These paths describe preparation,
    not evidence that a patient job or scientific validation has run.
    """

    provenance: RunProvenanceArtifacts
    preparation_record_path: Path


def prepare_patient_scientific_run(
    *,
    scientific_config_snapshot_path: Path | str,
    routing_profile_path: Path | str,
    output_dir: Path | str,
    repository_path: Path | str = DEFAULT_REPOSITORY_PATH,
) -> StandalonePreparationResult:
    """Prepare current execution provenance using an existing scientific snapshot.

    The snapshot must pass the existing reader and exact typed rehydration SHA
    round-trip. Routing JSON is explicit shared input policy, never a patient
    manifest; only the existing provenance writer defines its identity semantics.
    Repository identity defaults to this module's repository, independent of cwd.

    The destination must be absent or empty, outside the snapshot's containing
    directory, and must not contain either source. Resolved paths enforce these
    checks through symlinks. Inputs are read only. On success the destination
    contains four standard provenance artifacts plus preparation_record.json,
    which distinguishes reused science from newly captured execution identity.

    Callers must exclusively own the destination and keep inputs immutable during
    preparation. Invalid inputs/overlap raise ValueError or TypeError; occupied
    destinations raise FileExistsError. I/O and provenance errors propagate.
    A late failure can leave partial new artifacts but no completion record;
    retry with a fresh destination, never by overwriting historical provenance.
    """
    snapshot_path = Path(scientific_config_snapshot_path).expanduser().resolve(strict=True)
    routing_path = Path(routing_profile_path).expanduser().resolve(strict=True)
    destination = Path(output_dir).expanduser().resolve()
    repository = Path(repository_path).expanduser().resolve(strict=True)
    _validate_destination(destination, snapshot_path, routing_path)

    snapshot_bytes = snapshot_path.read_bytes()
    source_snapshot = read_pipeline_config_snapshot(snapshot_path)
    pipeline_config = rehydrate_pipeline_scientific_config_snapshot(source_snapshot)
    routing_bytes = routing_path.read_bytes()
    routing_payload = json.loads(routing_bytes)
    if not isinstance(routing_payload, dict):
        raise TypeError("routing profile root must be an object: {}".format(routing_path))
    routing_sha256 = canonical_sha256(routing_payload)
    _verify_unchanged(snapshot_path, snapshot_bytes)

    from startup.run_provenance import write_run_provenance_artifacts

    _validate_destination(destination, snapshot_path, routing_path)
    destination.mkdir(parents=True, exist_ok=True)
    provenance = write_run_provenance_artifacts(
        pipeline_config=pipeline_config,
        routing_profile_path=routing_path,
        manifest_dir=destination,
        repository_path=repository,
        overwrite=False,
    )
    _verify_unchanged(snapshot_path, snapshot_bytes)
    _verify_unchanged(routing_path, routing_bytes)
    if provenance.compatibility_identity.scientific_config_sha256 != source_snapshot.config_sha256:
        raise ValueError("prepared scientific configuration SHA differs from the source snapshot")

    preparation_record_path = destination.joinpath("preparation_record.json")
    record = {
        "schema_version": PREPARATION_RECORD_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_configuration": "reused_verified_snapshot",
        "execution_provenance": "captured_current",
        "source_snapshot": {
            "path": snapshot_path.as_posix(),
            "config_sha256": source_snapshot.config_sha256,
            "file_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
        },
        "source_routing_profile": {
            "path": routing_path.as_posix(),
            "canonical_sha256": routing_sha256,
            "file_sha256": hashlib.sha256(routing_bytes).hexdigest(),
        },
        "repository_path": repository.as_posix(),
        "new_run_provenance": provenance.manifest_metadata(),
    }
    with preparation_record_path.open("x", encoding="utf-8") as output_file:
        json.dump(record, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")
    return StandalonePreparationResult(
        provenance=provenance,
        preparation_record_path=preparation_record_path,
    )


def _validate_destination(destination: Path, snapshot_path: Path, routing_path: Path) -> None:
    for source in (snapshot_path.parent, routing_path):
        if destination.is_relative_to(source) or source.is_relative_to(destination):
            raise ValueError("preparation source/destination overlap: {} and {}".format(source, destination))
    if destination.exists():
        if not destination.is_dir():
            raise NotADirectoryError("preparation destination is not a directory: {}".format(destination))
        if any(destination.iterdir()):
            raise FileExistsError("preparation destination must be empty: {}".format(destination))


def _verify_unchanged(path: Path, original_bytes: bytes) -> None:
    if path.read_bytes() != original_bytes:
        raise ValueError("preparation source changed while being read: {}".format(path))


__all__ = [
    "DEFAULT_REPOSITORY_PATH",
    "PREPARATION_RECORD_SCHEMA_VERSION",
    "StandalonePreparationResult",
    "prepare_patient_scientific_run",
]