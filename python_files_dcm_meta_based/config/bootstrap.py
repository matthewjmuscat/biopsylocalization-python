"""Typed input/bootstrap policy for constructing patient-local runtime state.

These contracts describe how discovered DICOM roles become the initial
legacy-compatible patient structure shell. They contain policy only: no DICOM
objects, pools, GUI state, or scientific runtime arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class StructureDataRemovalPolicy:
    """Explicit patient/ROI exclusions applied during structure bootstrap."""

    biopsy: Mapping[str, Sequence[str]] = field(default_factory=dict)
    prostate: Mapping[str, Sequence[str]] = field(default_factory=dict)
    dil: Mapping[str, Sequence[str]] = field(default_factory=dict)
    urethra: Mapping[str, Sequence[str]] = field(default_factory=dict)
    rectum: Mapping[str, Sequence[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("biopsy", "prostate", "dil", "urethra", "rectum"):
            object.__setattr__(self, field_name, _normalize_removal_mapping(getattr(self, field_name), field_name))


@dataclass(frozen=True, slots=True)
class StructureContourPolicy:
    """Ordered contour-name tokens used to identify each structure family."""

    oar: Sequence[str] = ("Prostate",)
    dil: Sequence[str] = ("DIL",)
    biopsy: Sequence[str] = ("Bx",)
    rectum: Sequence[str] = ("Rectum",)
    urethra: Sequence[str] = ("Urethra",)

    def __post_init__(self) -> None:
        for field_name in ("oar", "dil", "biopsy", "rectum", "urethra"):
            object.__setattr__(self, field_name, _non_empty_string_tuple(getattr(self, field_name), field_name))


@dataclass(frozen=True, slots=True)
class SimulatedBiopsyBootstrapPolicy:
    """Rules for creating initial simulated-biopsy records from structures."""

    locations: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    fraction_numbers_to_create: str | int | Sequence[int] = "all"
    fraction_prefixes: Sequence[str] = ("f", "fraction", "")

    def __post_init__(self) -> None:
        normalized_locations: dict[str, dict[str, Any]] = {}
        for biopsy_type, raw_config in self.locations.items():
            resolved_type = str(biopsy_type).strip()
            if resolved_type == "":
                raise ValueError("simulated biopsy type cannot be empty")
            if not isinstance(raw_config, Mapping):
                raise TypeError("simulated biopsy location config must be a mapping")
            location_config = dict(raw_config)
            for required_field in ("Create", "Relative to struct type", "Identifier string"):
                if required_field not in location_config:
                    raise ValueError(
                        "simulated biopsy location {!r} is missing {!r}".format(resolved_type, required_field)
                    )
            location_config["Create"] = bool(location_config["Create"])
            location_config["Relative to struct type"] = _non_empty_string(
                location_config["Relative to struct type"],
                "Relative to struct type",
            )
            location_config["Identifier string"] = _non_empty_string(
                location_config["Identifier string"],
                "Identifier string",
            )
            location_config["Transport family"] = str(location_config.get("Transport family", "identity")).strip()
            normalized_locations[resolved_type] = location_config
        object.__setattr__(self, "locations", normalized_locations)

        fraction_policy = self.fraction_numbers_to_create
        if isinstance(fraction_policy, str):
            if fraction_policy.strip().lower() != "all":
                raise ValueError("fraction_numbers_to_create string value must be 'all'")
            normalized_fraction_policy: str | int | tuple[int, ...] = "all"
        elif isinstance(fraction_policy, int):
            normalized_fraction_policy = int(fraction_policy)
        elif isinstance(fraction_policy, Sequence):
            normalized_fraction_policy = tuple(int(value) for value in fraction_policy)
            if len(normalized_fraction_policy) == 0:
                raise ValueError("fraction_numbers_to_create cannot be empty")
        else:
            raise TypeError("fraction_numbers_to_create must be 'all', an integer, or a sequence of integers")
        object.__setattr__(self, "fraction_numbers_to_create", normalized_fraction_policy)
        object.__setattr__(self, "fraction_prefixes", tuple(str(value) for value in self.fraction_prefixes))


@dataclass(frozen=True, slots=True)
class PatientBootstrapConfig:
    """Complete reusable policy for one-patient structure-reference bootstrap."""

    removals: StructureDataRemovalPolicy = field(default_factory=StructureDataRemovalPolicy)
    contours: StructureContourPolicy = field(default_factory=StructureContourPolicy)
    simulated_biopsies: SimulatedBiopsyBootstrapPolicy = field(default_factory=SimulatedBiopsyBootstrapPolicy)
    mr_global_structure_table_name: str = "Global MR ADC statistics"
    mr_global_voxel_table_name: str = "Global by voxel MR ADC statistics"

    def __post_init__(self) -> None:
        if not isinstance(self.removals, StructureDataRemovalPolicy):
            raise TypeError("removals must be a StructureDataRemovalPolicy")
        if not isinstance(self.contours, StructureContourPolicy):
            raise TypeError("contours must be a StructureContourPolicy")
        if not isinstance(self.simulated_biopsies, SimulatedBiopsyBootstrapPolicy):
            raise TypeError("simulated_biopsies must be a SimulatedBiopsyBootstrapPolicy")
        object.__setattr__(
            self,
            "mr_global_structure_table_name",
            _non_empty_string(self.mr_global_structure_table_name, "mr_global_structure_table_name"),
        )
        object.__setattr__(
            self,
            "mr_global_voxel_table_name",
            _non_empty_string(self.mr_global_voxel_table_name, "mr_global_voxel_table_name"),
        )


def _normalize_removal_mapping(value: Mapping[str, Sequence[str]], field_name: str) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, Mapping):
        raise TypeError("{} removals must be a mapping".format(field_name))
    normalized: dict[str, tuple[str, ...]] = {}
    for patient_uid, roi_names in value.items():
        resolved_patient_uid = _non_empty_string(patient_uid, "{} patient_uid".format(field_name))
        if isinstance(roi_names, str):
            raise TypeError("{} removals for {} must be a sequence, not a string".format(field_name, patient_uid))
        normalized[resolved_patient_uid] = _non_empty_string_tuple(
            roi_names,
            "{} removals for {}".format(field_name, patient_uid),
            allow_empty=True,
        )
    return normalized


def _non_empty_string(value: Any, field_name: str) -> str:
    resolved_value = str(value).strip()
    if resolved_value == "":
        raise ValueError("{} cannot be empty".format(field_name))
    return resolved_value


def _non_empty_string_tuple(
    values: Sequence[Any],
    field_name: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError("{} must be a sequence, not a string".format(field_name))
    resolved_values = tuple(str(value).strip() for value in values)
    if not allow_empty and len(resolved_values) == 0:
        raise ValueError("{} cannot be empty".format(field_name))
    if any(value == "" for value in resolved_values):
        raise ValueError("{} cannot contain empty values".format(field_name))
    return resolved_values


__all__ = [
    "PatientBootstrapConfig",
    "SimulatedBiopsyBootstrapPolicy",
    "StructureContourPolicy",
    "StructureDataRemovalPolicy",
]
