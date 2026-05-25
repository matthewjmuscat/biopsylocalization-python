"""Patient-level MC containment output contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


MC_CONTAINMENT_BIOPSY_OUTPUT_KEYS = (
    "MC data: compiled sim results dataframe",
    "MC data: compiled sim sum-to-one results dataframe",
    "MC data: compiled sim results",
    "MC data: MC sim compiled distances global dataframe",
    "MC data: MC sim compiled distances point-wise dataframe",
    "MC data: MC sim compiled distances voxel-wise dataframe",
    "MC data: MC sim containment and distance all trials dataframe (light)",
)


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
                    "ROI": biopsy_structure.get("ROI"),
                    "Ref #": biopsy_structure.get("Ref #"),
                    "Index number": biopsy_structure.get("Index number", biopsy_index),
                    "Simulated bool": biopsy_structure.get("Simulated bool"),
                    "Simulated type": biopsy_structure.get("Simulated type"),
                    "outputs": outputs,
                }
            )
    return PatientContainmentOutputs(
        patient_uid=str(patient_uid),
        biopsy_outputs=biopsy_outputs,
    )
