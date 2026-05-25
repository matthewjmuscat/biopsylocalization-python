"""Patient-local MC relative-structure inventory helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from legacy_data_keys import legacy_data_keys

from .legacy_keys import legacy_mc_keys

RelativeStructureInfo = tuple[Any, str, Any, int]


@dataclass(slots=True)
class PatientRelativeStructureInventory:
    """Patient-local non-biopsy structures and legacy count metadata."""

    patient_uid: str
    relative_structure_template: dict[RelativeStructureInfo, None]
    total_num_structures: int
    total_num_biopsies: int
    total_num_non_biopsies: int

    @property
    def relative_structure_infos(self) -> tuple[RelativeStructureInfo, ...]:
        return tuple(self.relative_structure_template.keys())


def resolve_patient_info(patient_uid: str, patient_info_dict: Mapping[str, Any]) -> Mapping[str, Any]:
    """Accept either a whole master-info dictionary or one patient info dictionary."""
    master_keys = legacy_data_keys.master_info
    if master_keys.by_patient_key in patient_info_dict:
        by_patient = patient_info_dict[master_keys.by_patient_key]
        if patient_uid in by_patient:
            return by_patient[patient_uid]
        return by_patient[str(patient_uid)]
    return patient_info_dict


def build_patient_relative_structure_inventory(patient_uid: str,
                                               patient_reference_dict: Mapping[str, Any],
                                               patient_info_dict: Mapping[str, Any],
                                               *,
                                               structs_referenced_list: Sequence[str],
                                               bx_ref: str,
                                               all_ref_key: str) -> PatientRelativeStructureInventory:
    """Mirror the oracle's patient-specific non-biopsy structure inventory."""
    identity_keys = legacy_mc_keys.biopsy_identity
    structure_info_keys = legacy_data_keys.structure_info
    resolved_patient_info = resolve_patient_info(patient_uid, patient_info_dict)

    relative_structure_template: dict[RelativeStructureInfo, None] = {}
    for non_bx_struct_type in tuple(structs_referenced_list)[1:]:
        for structure_index, specific_non_bx_structure in enumerate(patient_reference_dict[non_bx_struct_type]):
            structure_info = (
                specific_non_bx_structure[identity_keys.roi_key],
                non_bx_struct_type,
                specific_non_bx_structure[identity_keys.ref_number_key],
                structure_index,
            )
            relative_structure_template[structure_info] = None

    total_num_structures = int(resolved_patient_info[all_ref_key][structure_info_keys.total_num_structs_key])
    total_num_biopsies = int(resolved_patient_info[bx_ref][structure_info_keys.num_structs_key])
    return PatientRelativeStructureInventory(
        patient_uid=str(patient_uid),
        relative_structure_template=relative_structure_template,
        total_num_structures=total_num_structures,
        total_num_biopsies=total_num_biopsies,
        total_num_non_biopsies=total_num_structures - total_num_biopsies,
    )