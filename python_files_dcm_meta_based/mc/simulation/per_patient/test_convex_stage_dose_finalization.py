"""Synthetic checks for MC dose-localization finalization callbacks."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from mc.simulation.per_patient import convex_stage


class ConvexStageDoseFinalizationTests(unittest.TestCase):
    def test_finalization_callback_runs_after_legacy_update(self) -> None:
        events: list[str] = []
        lattice_context = SimpleNamespace(localization_kind="dose")
        biopsy_context = SimpleNamespace(biopsy_index=0)
        localization_outputs = SimpleNamespace(localization_kind="dose")

        originals = {
            "build_patient_dose_lattice_context": convex_stage.build_patient_dose_lattice_context,
            "build_patient_dose_biopsy_context": convex_stage.build_patient_dose_biopsy_context,
            "run_patient_dose_localization_for_biopsy": convex_stage.run_patient_dose_localization_for_biopsy,
            "write_patient_dose_localization_outputs_to_legacy_record": (
                convex_stage.write_patient_dose_localization_outputs_to_legacy_record
            ),
        }
        try:
            convex_stage.build_patient_dose_lattice_context = lambda *args, **kwargs: lattice_context
            convex_stage.build_patient_dose_biopsy_context = lambda *args, **kwargs: biopsy_context
            convex_stage.run_patient_dose_localization_for_biopsy = lambda *args, **kwargs: localization_outputs

            def fake_write(_biopsy_structure, _localization_outputs):
                events.append("legacy_write")
                return _biopsy_structure

            convex_stage.write_patient_dose_localization_outputs_to_legacy_record = fake_write
            callback_payloads = []

            biopsy_count = convex_stage._run_patient_dose_localization_kind(
                patient_uid="synthetic_patient",
                patient_reference_dict={"Bx ref": [{}], "Dose ref": {}},
                config=SimpleNamespace(
                    keys=SimpleNamespace(bx_ref="Bx ref", dose_ref="Dose ref"),
                    dose=SimpleNamespace(),
                ),
                num_mc_dose_simulations=1,
                localization_kind="dose",
                dose_localization_finalization_callback=lambda payload: (
                    events.append("callback"),
                    callback_payloads.append(payload),
                ),
            )
        finally:
            for name, original in originals.items():
                setattr(convex_stage, name, original)

        self.assertEqual(biopsy_count, 1)
        self.assertEqual(events, ["legacy_write", "callback"])
        self.assertEqual(callback_payloads[0].patient_uid, "synthetic_patient")
        self.assertEqual(callback_payloads[0].biopsy_index, 0)
        self.assertIs(callback_payloads[0].lattice_context, lattice_context)
        self.assertIs(callback_payloads[0].biopsy_context, biopsy_context)
        self.assertIs(callback_payloads[0].localization_outputs, localization_outputs)


if __name__ == "__main__":
    unittest.main()