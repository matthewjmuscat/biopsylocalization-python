"""Patient-level MC dose output contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


MC_DOSE_BIOPSY_OUTPUT_KEYS = (
    "MC data: Dose vals for each sampled bx pt arr (nominal & all MC trials)",
    "MC data: Dose gradient vals for each sampled bx pt arr (nominal & all MC trials)",
    "MC data: Differential DVH dict",
    "MC data: Cumulative DVH dict",
    "MC data: dose volume metrics dict",
    "MC data: Dose statistics (MLE) for each sampled bx pt list (mean, std)",
    "MC data: Dose statistics for each sampled bx pt list (mean, std, quantiles)",
    "MC data: voxelized dose results list",
    "MC data: voxelized dose results dict (dict of lists)",
)


@dataclass(slots=True)
class PatientDoseOutputs:
    """Dose outputs collected from one patient's biopsy records."""

    patient_uid: str
    biopsy_outputs: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "patient_uid": self.patient_uid,
            "biopsy_outputs": self.biopsy_outputs,
        }


def collect_patient_dose_outputs(patient_uid: str,
                                 patient_reference_dict: Mapping[str, Any],
                                 *,
                                 bx_ref: str) -> PatientDoseOutputs:
    """Collect dose artifacts written into one patient dictionary."""
    biopsy_outputs: list[dict[str, Any]] = []
    for biopsy_index, biopsy_structure in enumerate(patient_reference_dict.get(bx_ref, ())):
        outputs = {
            output_key: biopsy_structure.get(output_key)
            for output_key in MC_DOSE_BIOPSY_OUTPUT_KEYS
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
    return PatientDoseOutputs(
        patient_uid=str(patient_uid),
        biopsy_outputs=biopsy_outputs,
    )
