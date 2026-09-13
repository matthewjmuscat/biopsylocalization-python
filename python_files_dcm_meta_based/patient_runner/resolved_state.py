"""Small scientific-state evidence for existing patient stage manifests.

These records describe actual stage inputs and returned decisions, not new
scientific defaults or a second runtime store. Native dose units are retained.
Legacy probe callers may record inherited values using the same field contract.
"""

from __future__ import annotations

from typing import Any, Mapping


def dose_threshold_inputs(patient: Mapping[str, Any], config: Any) -> dict[str, Any]:
    """Observe dose-helper inputs without resolving or changing its threshold.

    The prescription lookup mirrors the helper's accepted legacy storage shape;
    the helper remains the authority for the effective value recorded afterward.
    """
    present = config.dose_ref in patient
    try:
        prescription = patient[config.plan_ref]["Prescription doses dict"]["TARGET"]
        available = True
    except Exception:
        prescription, available = None, False
    configured = config.lower_bound_dose_value
    return {
        "dose_present": present,
        "dose_units": patient[config.dose_ref].get("Dose units") if present else None,
        "configured_lower_bound": configured,
        "configured_gradient_lower_bound": config.lower_bound_dose_gradient_value,
        "target_prescription_available": available,
        "target_prescription": prescription,
        "resolution_source": (
            "not_applicable" if not present else "configured" if configured is not None
            else "patient_prescription" if available else "fallback_zero"
        ),
    }
