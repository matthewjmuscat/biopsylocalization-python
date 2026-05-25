"""Patient-level MC containment output contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .legacy_keys import legacy_mc_keys

MC_CONTAINMENT_BIOPSY_OUTPUT_KEYS = legacy_mc_keys.biopsy_outputs.containment_output_keys


@dataclass(slots=True)
class PatientContainmentOutputs:
    """Containment outputs collected from one patient's biopsy records."""

    patient_uid: str
    biopsy_outputs: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "patient_uid": self.patient_uid,
            "biopsy_outputs": self.biopsy_outputs,
        }


def collect_patient_containment_outputs(patient_uid: str,
                                        patient_reference_dict: Mapping[str, Any],
                                        *,
                                        bx_ref: str) -> PatientContainmentOutputs:
    """Collect containment artifacts written into one patient dictionary."""
    identity_keys = legacy_mc_keys.biopsy_identity
    biopsy_outputs: list[dict[str, Any]] = []
    for biopsy_index, biopsy_structure in enumerate(patient_reference_dict.get(bx_ref, ())):
        outputs = {
            output_key: biopsy_structure.get(output_key)
            for output_key in MC_CONTAINMENT_BIOPSY_OUTPUT_KEYS
            if output_key in biopsy_structure
        }
        if outputs:
            biopsy_outputs.append(
                {
                    identity_keys.roi_key: biopsy_structure.get(identity_keys.roi_key),
                    identity_keys.ref_number_key: biopsy_structure.get(identity_keys.ref_number_key),
                    identity_keys.index_number_key: biopsy_structure.get(identity_keys.index_number_key, biopsy_index),
                    identity_keys.simulated_bool_key: biopsy_structure.get(identity_keys.simulated_bool_key),
                    identity_keys.simulated_type_key: biopsy_structure.get(identity_keys.simulated_type_key),
                    "outputs": outputs,
                }
            )
    return PatientContainmentOutputs(
        patient_uid=str(patient_uid),
        biopsy_outputs=biopsy_outputs,
    )
