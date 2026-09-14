"""Patient input byte identity, independent of discovery and scientific execution.

This reusable provenance boundary streams local files without decoding DICOM.
It preserves role order and path identity; it does not anonymize inputs or claim
that matching bytes imply scientifically correct routing. Callers must keep
inputs immutable during execution. Before/after checks detect drift, not every
possible transient modification between checks.
This ledger binds physical locations and bytes for strict local execution. It
does not identify or deduplicate logical DICOM objects (SOPInstanceUID/SOPClassUID).
See DICOM_INPUT_SHAPE.md for the separate planned discovery identity policy.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING

from config.snapshots import canonical_sha256

if TYPE_CHECKING:
    from patient_runner.inputs import PatientInputPaths


INPUT_CONTENT_SCHEMA = "patient_input_content_v1"
INPUT_CONTENT_KEY = "input_content_identity"


def fingerprint_file(path: Path) -> dict[str, Any]:
    """Stream one regular file; reject a detectable change while hashing.

    Paths stay local in the ledger. SHA-256 and byte length describe exact file
    contents, including DICOM headers. Missing/nonregular files and concurrent
    changes raise; no patient values or file contents are returned.
    """
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("input is not a regular file: " + str(path))
    before = resolved.stat()
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = resolved.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns
    ):
        raise ValueError("input changed while hashing: " + str(path))
    return {"resolved_path": str(resolved), "size_bytes": after.st_size, "sha256": digest.hexdigest()}


def capture_patient_input_content(inputs: PatientInputPaths) -> dict[str, Any]:
    """Capture every declared role, including optional modalities, into metadata.

    No files are written. Empty optional roles remain explicit. The digest is
    patient-scoped so disjoint runs need not share a cohort-wide input digest.
    """
    roles = {}
    for role, value in inputs.to_dict(include_identity=False).items():
        if role == "patient_uid":
            continue
        paths = value if isinstance(value, list) else [value] if value else []
        roles[role] = [{"path": path, **fingerprint_file(Path(path))} for path in paths]
    payload = {
        "schema_version": INPUT_CONTENT_SCHEMA,
        "patient_uid": inputs.patient_uid,
        "role_paths_sha256": inputs.manifest_identity_sha256,
        "roles": roles,
    }
    return {**payload, "identity_sha256": canonical_sha256(payload)}


def validate_patient_input_content(identity: Mapping[str, Any], inputs: PatientInputPaths) -> None:
    """Validate a retained ledger's schema, digest, and role bindings without I/O.

    Used by completed-run readers; it does not re-open historical source files.
    """
    payload = dict(identity)
    digest = payload.pop("identity_sha256", None)
    if (set(payload) != {"schema_version", "patient_uid", "role_paths_sha256", "roles"}
            or payload.get("schema_version") != INPUT_CONTENT_SCHEMA or canonical_sha256(payload) != digest):
        raise ValueError("invalid patient input content identity")
    if payload.get("patient_uid") != inputs.patient_uid or payload.get("role_paths_sha256") != inputs.manifest_identity_sha256:
        raise ValueError("input content identity differs from patient role paths")
    expected_roles = inputs.to_dict(include_identity=False)
    expected_roles.pop("patient_uid")
    if set(payload.get("roles", {})) != set(expected_roles):
        raise ValueError("input content role inventory mismatch")
    for role, value in expected_roles.items():
        paths = value if isinstance(value, list) else [value] if value else []
        records = payload["roles"][role]
        if not isinstance(records, list) or [item["path"] for item in records] != paths:
            raise ValueError("input content role order/path mismatch")
        for item in records:
            sha = item.get("sha256", "")
            if (set(item) != {"path", "resolved_path", "size_bytes", "sha256"}
                    or not Path(item["resolved_path"]).is_absolute()
                    or type(item["size_bytes"]) is not int or item["size_bytes"] < 0
                    or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)):
                raise ValueError("invalid file content record")


def verify_patient_input_content(identity: Mapping[str, Any], inputs: PatientInputPaths) -> None:
    """Re-read all declared bytes and fail if they differ from the planned ledger."""
    validate_patient_input_content(identity, inputs)
    if capture_patient_input_content(inputs) != identity:
        raise ValueError("patient input content changed since planning")
