"""Fail-closed isolation helpers for legacy-backed patient-runner execution."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, MutableMapping


POST_DISCOVERY_DEEPCOPY_SOURCE = "post_discovery_deepcopy"


@dataclass(frozen=True, slots=True)
class IsolatedLegacyRuntimeState:
    """Independent legacy-shaped state selected for one modular execution lane."""

    master_structure_reference_dict: MutableMapping[str, Any]
    master_structure_info_dict: MutableMapping[str, Any]
    source: str = POST_DISCOVERY_DEEPCOPY_SOURCE


def copy_isolated_legacy_runtime_state_from_snapshot(
    snapshot_reference_dict: MutableMapping[str, Any] | None,
    snapshot_info_dict: MutableMapping[str, Any] | None,
    *,
    operation_name: str,
) -> IsolatedLegacyRuntimeState:
    """Return deep-copied pristine state or fail before modular science runs."""
    has_reference_snapshot = snapshot_reference_dict is not None
    has_info_snapshot = snapshot_info_dict is not None
    if has_reference_snapshot != has_info_snapshot:
        raise RuntimeError(
            "{} requires both post-discovery snapshot dictionaries; found only one".format(operation_name)
        )
    if not has_reference_snapshot:
        raise RuntimeError(
            "{} requires a pristine post-discovery snapshot and will not fall back to mutated legacy runtime state".format(
                operation_name
            )
        )
    return IsolatedLegacyRuntimeState(
        master_structure_reference_dict=deepcopy(snapshot_reference_dict),
        master_structure_info_dict=deepcopy(snapshot_info_dict),
    )


__all__ = [
    "IsolatedLegacyRuntimeState",
    "POST_DISCOVERY_DEEPCOPY_SOURCE",
    "copy_isolated_legacy_runtime_state_from_snapshot",
]
