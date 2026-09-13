"""Synthetic sensitivity controls executing the unchanged wrapper AND helper.

Only low-level grid/render producers are replaced with deterministic arrays.
No DICOM, scientific GPU kernels, or patient files are used. The observed
threshold transition is always the actual legacy helper/wrapper behavior.
"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import importlib
import itertools
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from validation.legacy_dose_order import capture_legacy_dose_order, compare_legacy_dose_orders, _Progress


@contextmanager
def _legacy_fixture():
    def grid(**kwargs):
        result = np.zeros((2, 4, 14), dtype=np.float64)
        result[:, :, 3:6] = np.arange(24).reshape(2, 4, 3)
        result[:, :, 6] = np.arange(8).reshape(2, 4)
        return result

    def cloud(values, *, truncate_below_dose=None, **kwargs):
        flat = values.reshape(-1, 14)
        selected = flat if truncate_below_dose is None else flat[flat[:, 6] >= truncate_below_dose]
        return SimpleNamespace(points=selected[:, 3:6].copy()), None

    numerical = SimpleNamespace(build_dose_grid=grid, calculate_gradient_lattices=lambda *args: (None, None, None),
        map_gradient_to_physical_space=lambda **kwargs: kwargs["phys_space_dose_map_3d_arr"])
    rendering = SimpleNamespace(create_dose_point_cloud_with_gradients=cloud)
    with patch.dict("sys.modules", {"dose_lattice_helper_funcs": numerical, "plotting_funcs": rendering}):
        legacy = importlib.import_module("preprocessing.dose_grid_processing")
        with patch.object(importlib.import_module("preprocessing"), "dose_grid_processing", legacy, create=True), \
                patch.object(legacy, "dose_lattice_helper_funcs", numerical), \
                patch.object(legacy, "plotting_funcs", rendering):
            yield legacy


def _patient(prescription=2, *, dose=True, missing_prescription=False):
    result = {"Plan": {"Prescription doses dict": {} if missing_prescription else {"TARGET": prescription}}}
    if dose:
        result["Dose"] = {"Dose units": "GY", "Dose pixel arr": np.ones((2, 2, 2)), "Dose grid scaling": 1,
                          "Pixel spacing": [1, 1], "Grid frame offset vector": [0, 1],
                          "Image orientation patient": [1, 0, 0, 0, 1, 0], "Image position patient": [0, 0, 0]}
    return result


class LegacyDoseOrderTests(unittest.TestCase):
    def test_previous_patient_arrays_are_released_before_loading_successor(self):
        with TemporaryDirectory() as temporary, _legacy_fixture() as legacy:
            references = []
            config = legacy.DoseGridProcessingConfig("Dose", "Plan", None, 0, False, False)
            def load(uid):
                self.assertTrue(all(reference() is None for reference in references))
                patient = _patient()
                references.append(weakref.ref(patient["Dose"]["Dose pixel arr"]))
                return patient
            capture_legacy_dose_order(patient_uids=("A", "B", "C"), load_patient=load, config=config, output_dir=Path(temporary) / "probe")
            self.assertTrue(all(reference() is None for reference in references))

    def test_unequal_prescriptions_expose_actual_carry_without_changing_full_grid(self):
        with TemporaryDirectory() as temporary, _legacy_fixture() as legacy:
            root = Path(temporary)
            config = legacy.DoseGridProcessingConfig("Dose", "Plan", None, 0, False, False)
            patients = {"A": _patient(2), "B": _patient(6), "F1": _patient(dose=False)}
            for index, order in enumerate(itertools.permutations(patients)):
                report = capture_legacy_dose_order(patient_uids=order, load_patient=lambda uid: deepcopy(patients[uid]),
                    config=config, output_dir=root / str(index))
                first_dose = next(uid for uid in order if uid != "F1")
                expected = patients[first_dose]["Plan"]["Prescription doses dict"]["TARGET"]
                self.assertEqual(report["final_lower_bound"], expected)
                for patient in report["patients"]:
                    if patient["state"]["dose_present"]:
                        self.assertEqual(patient["state"]["effective_lower_bound"], expected)
            # permutations 0 and 2 put A and B first, respectively.
            comparison = compare_legacy_dose_orders(root / "0", root / "2")
            self.assertTrue(comparison["characterization_complete"])
            for patient in comparison["patients"]:
                if patient["patient_uid"] != "F1":
                    self.assertEqual(patient["classification"], "dependence_observed")
                    self.assertTrue(all(p["passed"] for p in patient["products"] if p["field"] != "thresholded_points"))

    def test_fallback_zero_and_explicit_override_are_characterized(self):
        with TemporaryDirectory() as temporary, _legacy_fixture() as legacy:
            root = Path(temporary)
            config = legacy.DoseGridProcessingConfig("Dose", "Plan", None, 0, False, False)
            patients = {"missing": _patient(missing_prescription=True), "B": _patient(6)}
            report = capture_legacy_dose_order(patient_uids=patients, load_patient=lambda uid: deepcopy(patients[uid]), config=config, output_dir=root / "fallback")
            self.assertEqual(report["final_lower_bound"], 0)
            self.assertEqual(report["patients"][0]["state"]["resolution_source"], "fallback_zero")
            self.assertEqual(report["patients"][1]["state"]["configured_lower_bound"], 0)
            explicit = replace(config, lower_bound_dose_value=4)
            report = capture_legacy_dose_order(patient_uids=patients, load_patient=lambda uid: deepcopy(patients[uid]), config=explicit, output_dir=root / "explicit")
            self.assertEqual([p["state"]["effective_lower_bound"] for p in report["patients"]], [4, 4])

    def test_observer_matches_uninstrumented_actual_wrapper_and_restores_helper(self):
        with TemporaryDirectory() as temporary, _legacy_fixture() as legacy:
            config = legacy.DoseGridProcessingConfig("Dose", "Plan", None, 0, False, False)
            patients = {"A": _patient(2), "B": _patient(6)}
            original = legacy.build_dose_grid_runtime_objects_for_patient
            direct = deepcopy(patients)
            progress = _Progress()
            legacy.build_dose_grids_for_cohort(direct, {"Global": {"Num cases": 2}}, config, progress, progress, None)
            report = capture_legacy_dose_order(patient_uids=patients, load_patient=lambda uid: deepcopy(patients[uid]), config=config, output_dir=Path(temporary) / "probe")
            self.assertIs(original, legacy.build_dose_grid_runtime_objects_for_patient)
            for record in report["patients"]:
                with np.load(Path(temporary) / "probe" / record["arrays_file"], allow_pickle=False) as arrays:
                    self.assertTrue(np.array_equal(arrays["thresholded_points"], direct[record["patient_uid"]]["Dose"]["Dose grid point cloud thresholded"].points))

    def test_patient_adapter_restarts_threshold_for_each_patient_and_records_state(self):
        from patient_runner.scientific_stages import run_patient_grid_preprocessing_scientific_stage

        with _legacy_fixture() as legacy:
            dose_config = legacy.DoseGridProcessingConfig("Dose", "Plan", None, 0, False, False)
            scientific = SimpleNamespace(grid_preprocessing=SimpleNamespace(enabled=True, dose_grid_config=dose_config,
                mr_adc_input_normalization=None, mr_adc_grid_config=None))
            for order in itertools.permutations((2, 6, 4)):
                for prescription in order:
                    runtime = SimpleNamespace(patient_uid=str(prescription), pydicom_item=_patient(prescription))
                    result = run_patient_grid_preprocessing_scientific_stage(runtime, None, scientific_config=scientific)
                    state = result.metadata["resolved_scientific_state"]["dose"]
                    self.assertEqual(state["effective_lower_bound"], prescription)
                    self.assertIsNone(state["configured_lower_bound"])
                    self.assertEqual(state["resolution_source"], "patient_prescription")

    def test_verification_error_never_writes_completed_probe_report(self):
        with TemporaryDirectory() as temporary, _legacy_fixture() as legacy:
            root = Path(temporary) / "probe"
            config = legacy.DoseGridProcessingConfig("Dose", "Plan", None, 0, False, False)
            def fail(uid):
                raise ValueError("synthetic input drift")
            original = legacy.build_dose_grid_runtime_objects_for_patient
            with self.assertRaisesRegex(ValueError, "drift"):
                capture_legacy_dose_order(patient_uids=("A",), load_patient=lambda uid: _patient(), config=config, output_dir=root, verify_after=fail)
            self.assertFalse((root / "legacy_dose_order.json").exists())
            self.assertIs(original, legacy.build_dose_grid_runtime_objects_for_patient)


if __name__ == "__main__":
    unittest.main()
