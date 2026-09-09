"""Exact Git-backed source identity for auditable scientific runs.

A commit hash alone is insufficient when a run uses an uncommitted worktree.
The source identity therefore fingerprints HEAD, the tracked diff from HEAD, and
all non-ignored untracked files. Generated/ignored outputs do not affect it.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping


CODE_IDENTITY_SCHEMA_VERSION = "code_identity_v1"
PACKAGED_SOURCE_IDENTITY_ENV_VAR = "BIOPSYLOCALIZATION_SOURCE_IDENTITY"
SOURCE_IDENTITY_SUFFIXES = frozenset({".py", ".toml", ".md", ".txt", ".yaml", ".yml", ".ini", ".cfg", ".sh"})
SOURCE_IDENTITY_ROOT_FILES = frozenset({"Pipfile", "Pipfile.lock", ".gitignore"})


@dataclass(frozen=True, slots=True)
class CodeIdentity:
    """Git revision and exact effective source-tree fingerprint for one run."""

    repository_root: str
    commit: str
    branch: str
    dirty: bool
    tracked_diff_sha256: str
    untracked_files_sha256: str
    source_tree_sha256: str
    source_kind: str = "git_worktree"
    untracked_source_files: tuple[str, ...] = ()
    schema_version: str = CODE_IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CODE_IDENTITY_SCHEMA_VERSION:
            raise ValueError("unsupported code identity schema_version: {}".format(self.schema_version))
        for field_name in (
            "repository_root",
            "commit",
            "branch",
            "tracked_diff_sha256",
            "untracked_files_sha256",
            "source_tree_sha256",
            "source_kind",
        ):
            if str(getattr(self, field_name)).strip() == "":
                raise ValueError("{} cannot be empty".format(field_name))
        expected_source_sha256 = _source_tree_sha256(
            commit=self.commit,
            tracked_diff_sha256=self.tracked_diff_sha256,
            untracked_files_sha256=self.untracked_files_sha256,
            source_kind=self.source_kind,
        )
        if self.source_tree_sha256 != expected_source_sha256:
            raise ValueError("code source_tree_sha256 does not match its component fingerprints")
        object.__setattr__(self, "dirty", bool(self.dirty))
        object.__setattr__(self, "untracked_source_files", tuple(str(path) for path in self.untracked_source_files))

    def to_dict(self) -> dict[str, Any]:
        """Return the inspectable JSON representation embedded in run identity."""
        return {
            "schema_version": self.schema_version,
            "repository_root": self.repository_root,
            "commit": self.commit,
            "branch": self.branch,
            "dirty": self.dirty,
            "source_kind": self.source_kind,
            "untracked_source_files": list(self.untracked_source_files),
            "tracked_diff_sha256": self.tracked_diff_sha256,
            "untracked_files_sha256": self.untracked_files_sha256,
            "source_tree_sha256": self.source_tree_sha256,
        }


def capture_code_identity(path_within_repository: Path | str) -> CodeIdentity:
    """Capture an exact identity for the Git worktree containing ``path``."""
    working_path = Path(path_within_repository).expanduser().resolve()
    if working_path.is_file():
        working_path = working_path.parent
    try:
        repository_root = Path(_git_text(working_path, "rev-parse", "--show-toplevel"))
    except RuntimeError:
        packaged_identity = str(os.environ.get(PACKAGED_SOURCE_IDENTITY_ENV_VAR, "")).strip()
        if packaged_identity == "":
            raise RuntimeError(
                "code identity requires a Git worktree or {} for a packaged build".format(
                    PACKAGED_SOURCE_IDENTITY_ENV_VAR
                )
            )
        empty_sha256 = hashlib.sha256(b"").hexdigest()
        return CodeIdentity(
            repository_root=working_path.as_posix(),
            commit=packaged_identity,
            branch="packaged",
            dirty=False,
            tracked_diff_sha256=empty_sha256,
            untracked_files_sha256=empty_sha256,
            source_kind="packaged_override",
            source_tree_sha256=_source_tree_sha256(
                commit=packaged_identity,
                tracked_diff_sha256=empty_sha256,
                untracked_files_sha256=empty_sha256,
                source_kind="packaged_override",
            ),
        )
    commit = _git_text(repository_root, "rev-parse", "HEAD")
    branch = _git_text(repository_root, "rev-parse", "--abbrev-ref", "HEAD")
    tracked_diff = _git_bytes(repository_root, "diff", "--binary", "HEAD", "--")
    untracked_listing = _git_bytes(repository_root, "ls-files", "--others", "--exclude-standard", "-z")
    untracked_paths = tuple(
        Path(path_bytes.decode("utf-8", errors="strict"))
        for path_bytes in untracked_listing.split(b"\0")
        if path_bytes and _is_source_identity_path(Path(path_bytes.decode("utf-8", errors="strict")))
    )
    untracked_digest = hashlib.sha256()
    for relative_path in sorted(untracked_paths, key=lambda path: path.as_posix()):
        absolute_path = repository_root.joinpath(relative_path)
        untracked_digest.update(relative_path.as_posix().encode("utf-8"))
        untracked_digest.update(b"\0")
        untracked_digest.update(absolute_path.read_bytes())
        untracked_digest.update(b"\0")
    tracked_diff_sha256 = hashlib.sha256(tracked_diff).hexdigest()
    untracked_files_sha256 = untracked_digest.hexdigest()
    return CodeIdentity(
        repository_root=repository_root.as_posix(),
        commit=commit,
        branch=branch,
        dirty=bool(tracked_diff or untracked_paths),
        tracked_diff_sha256=tracked_diff_sha256,
        untracked_files_sha256=untracked_files_sha256,
        source_tree_sha256=_source_tree_sha256(
            commit=commit,
            tracked_diff_sha256=tracked_diff_sha256,
            untracked_files_sha256=untracked_files_sha256,
            source_kind="git_worktree",
        ),
        untracked_source_files=tuple(path.as_posix() for path in sorted(untracked_paths, key=lambda path: path.as_posix())),
    )


def write_code_identity(identity: CodeIdentity, output_path: Path | str, *, overwrite: bool = False) -> Path:
    """Write one code identity JSON artifact."""
    if not isinstance(identity, CodeIdentity):
        raise TypeError("identity must be a CodeIdentity")
    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError("code identity already exists: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(identity.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def read_code_identity(input_path: Path | str) -> CodeIdentity:
    """Read and verify one code identity JSON artifact."""
    path = Path(input_path)
    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, Mapping):
        raise TypeError("code identity root must be an object")
    return CodeIdentity(
        schema_version=str(payload.get("schema_version", "")),
        repository_root=str(payload.get("repository_root", "")),
        commit=str(payload.get("commit", "")),
        branch=str(payload.get("branch", "")),
        dirty=bool(payload.get("dirty", False)),
        tracked_diff_sha256=str(payload.get("tracked_diff_sha256", "")),
        untracked_files_sha256=str(payload.get("untracked_files_sha256", "")),
        source_tree_sha256=str(payload.get("source_tree_sha256", "")),
        source_kind=str(payload.get("source_kind", "git_worktree")),
        untracked_source_files=tuple(str(path) for path in payload.get("untracked_source_files", ())),
    )


def _source_tree_sha256(
    *,
    commit: str,
    tracked_diff_sha256: str,
    untracked_files_sha256: str,
    source_kind: str,
) -> str:
    payload = {
        "schema_version": CODE_IDENTITY_SCHEMA_VERSION,
        "commit": str(commit),
        "source_kind": str(source_kind),
        "tracked_diff_sha256": str(tracked_diff_sha256),
        "untracked_files_sha256": str(untracked_files_sha256),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_text(working_dir: Path, *args: str) -> str:
    output = _git_bytes(working_dir, *args).decode("utf-8", errors="strict").strip()
    if output == "":
        raise RuntimeError("git command returned empty output: git {}".format(" ".join(args)))
    return output


def _is_source_identity_path(path: Path) -> bool:
    return path.name in SOURCE_IDENTITY_ROOT_FILES or path.suffix.lower() in SOURCE_IDENTITY_SUFFIXES


def _git_bytes(working_dir: Path, *args: str) -> bytes:
    try:
        return subprocess.check_output(
            ("git", *args),
            cwd=str(working_dir),
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("unable to capture Git code identity with: git {}".format(" ".join(args))) from exc


__all__ = [
    "CODE_IDENTITY_SCHEMA_VERSION",
    "PACKAGED_SOURCE_IDENTITY_ENV_VAR",
    "SOURCE_IDENTITY_ROOT_FILES",
    "SOURCE_IDENTITY_SUFFIXES",
    "CodeIdentity",
    "capture_code_identity",
    "read_code_identity",
    "write_code_identity",
]
