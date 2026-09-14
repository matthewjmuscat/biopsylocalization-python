"""Numerical regression using actual CPU scientific implementations.

PCA, cylinder transport, Open3D, SciPy Delaunay, the lattice sampler and biopsy
coordinate transforms run on fabricated contours. Only allocation contents and
the old sample recurrence are substituted. GPU volume integration and tissue
classification do not run; their geometry/coordinate inputs are checked.
"""

from contextlib import contextmanager
import importlib
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial import ConvexHull


SAMPLES = "Centroid line sample pts"
MODEL = "Planned reconstructed biopsy model dict"


def synthetic_slices():
    ring = np.array([[.35, 0., 0.], [-.35, 0., 0.], [0., .35, 0.], [0., -.35, 0.]])
    direction = np.array([.3, .4, np.sqrt(.75)])
    oblique = np.array([10., -2., 5.]) + np.array([0., .617, 1.234])[:, None] * direction
    centers = {
        "axial": np.array([[0., 0., 0.], [0., 0., .5], [0., 0., 1.]]),
        "oblique": oblique,
        "reversed_slices": oblique[::-1],
        "asymmetric_bent": np.array([[0., 0., 0.], [.15, 0., .2], [.1, .1, 1.]]),
    }
    return {name: [ring + point for point in points] for name, points in centers.items()}


@contextmanager
def allocation_contents(helper, residue, *, historical=False):
    """Replay the old N-row allocation/last-row recurrence as a negative control.

    Only the builder's NumPy binding changes. Its other empty allocations are
    also poisoned; actual dependencies retain their normal NumPy bindings.
    """
    proxy = SimpleNamespace(**{key: getattr(np, key) for key in dir(np)})
    proxy.empty = lambda shape, dtype=float: np.full(shape, residue, dtype=dtype)
    if historical:
        def old_samples(start, end, count):
            intervals = count - 1
            samples = np.full((intervals, 3), residue, dtype=float)
            samples[0] = start
            travel = np.array([end - start]) * 1 / intervals
            for index in range(1, intervals):
                samples[index] = samples[-1] + travel
            return samples
        proxy.linspace = old_samples
    with patch.object(helper, "np", proxy):
        yield


class BiopsyGeometryCharacterizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helper = importlib.import_module("preprocessing.biopsy_processing.biopsy_geometry_helper")
        cls.sampling = importlib.import_module("preprocessing.biopsy_processing.sampled_biopsy_processing")
        cls.planner = importlib.import_module("preprocessing.biopsy_processing.simulated_biopsy_planner")
        cls.point_sampler = importlib.import_module("sampling.biopsy_point_sampler")

    def build(self, slices, residue=29., *, historical=False):
        with allocation_contents(self.helper, residue, historical=historical):
            return self.helper.build_reconstructed_biopsy_model_for_sampling_from_zslice_list(slices, .35)

    def assert_same_geometry(self, left, right):
        self.assertEqual(left.keys(), right.keys())
        for key in left:
            if key == SAMPLES:
                continue
            with self.subTest(field=key):
                if key == "Reconstructed structure point cloud":
                    for attr in ("points", "colors"):
                        np.testing.assert_array_equal(np.asarray(getattr(left[key], attr)), np.asarray(getattr(right[key], attr)))
                elif key == "Reconstructed structure delaunay global":
                    for attr in ("points", "simplices", "neighbors", "convex_hull", "transform"):
                        np.testing.assert_array_equal(getattr(left[key].delaunay_triangulation, attr),
                                                      getattr(right[key].delaunay_triangulation, attr))
                elif key == "Rotated reconstructed structure zslice list":
                    self.assertEqual(len(left[key]), len(right[key]))
                    for a, b in zip(left[key], right[key]):
                        np.testing.assert_array_equal(a, b)
                else:
                    np.testing.assert_array_equal(left[key], right[key])

    def test_closed_segment_contract_and_allocation_independence(self):
        for name, slices in synthetic_slices().items():
            reference = self.build(slices)
            line = reference["Best fit line of centroid pts"]
            delta = line[1] - line[0]
            length = np.linalg.norm(delta)
            intervals = math.ceil(length / .1)
            for residue in (87., -29., np.nan, 1e200):
                with self.subTest(case=name, residue=residue):
                    model = self.build(slices, residue)
                    samples = model[SAMPLES]
                    self.assertEqual(samples.shape, (intervals + 1, 3))
                    self.assertTrue(np.isfinite(samples).all())
                    np.testing.assert_array_equal(samples[[0, -1]], line)
                    t = (samples - line[0]) @ delta / length**2
                    np.testing.assert_allclose(t, np.arange(intervals + 1) / intervals, atol=2e-14, rtol=0)
                    np.testing.assert_allclose(samples, line[0] + t[:, None] * delta, atol=2e-14, rtol=0)
                    distances = np.linalg.norm(np.diff(samples, axis=0), axis=1)
                    np.testing.assert_allclose(distances, length / intervals, atol=2e-14, rtol=0)
                    self.assertLessEqual(distances.max(), .1 + 2e-14)
                    np.testing.assert_array_equal(reference[SAMPLES], samples)
                    self.assert_same_geometry(reference, model)

    def test_old_residue_changes_samples_only_and_cylinder_extent_is_preserved(self):
        for name, slices in synthetic_slices().items():
            with self.subTest(case=name):
                fixed = self.build(slices)
                old = [self.build(slices, value, historical=True) for value in (29., 87.)]
                self.assertFalse(np.array_equal(old[0][SAMPLES], old[1][SAMPLES]))
                self.assertEqual(len(fixed[SAMPLES]), len(old[0][SAMPLES]) + 1)
                for baseline in old:
                    self.assert_same_geometry(baseline, fixed)
                line = fixed["Best fit line of centroid pts"]
                length = np.linalg.norm(line[1] - line[0])
                intervals = math.ceil(length / .1)
                points = fixed["Reconstructed structure pts arr"]
                self.assertEqual(points.shape, (20 * intervals, 3))
                centers = points.reshape(intervals, 20, 3).mean(axis=1)
                np.testing.assert_allclose(centers[0], line[0], atol=2e-14, rtol=0)
                np.testing.assert_allclose(centers[-1], line[1] - (line[1] - line[0]) / intervals, atol=2e-14, rtol=0)
                self.assertAlmostEqual(np.linalg.norm(centers[-1] - centers[0]), length - length / intervals)
                self.assertEqual(ConvexHull(points).volume, ConvexHull(old[0]["Reconstructed structure pts arr"]).volume)

    def sample_real(self, model):
        record = {**model, "ROI": "synthetic-real"}
        patient = {"Bx ref": [record]}
        args = self.sampling.build_patient_sampled_biopsy_sampling_args(
            patient_uid="synthetic", pydicom_item=patient, bx_ref="Bx ref", bx_sample_pts_lattice_spacing=.2)
        result = self.point_sampler.sample_biopsy_points_from_reconstructed_global_delaunay_convex_structure(*args[0])
        self.assertGreater(result[2], 0)
        self.sampling.store_patient_sampled_biopsy_results(patient_uid="synthetic", pydicom_item=patient,
            bx_ref="Bx ref", parallel_results_sampled_bx_points_from_global_delaunay_arr_and_bounding_box_arr=[result])
        self.sampling.create_patient_biopsy_oriented_coordinate_system(
            patient_uid="synthetic", pydicom_item=patient, bx_ref="Bx ref")
        return result, record["Random uniformly sampled volume pts bx coord sys arr"]

    def test_actual_real_sampling_and_classification_coordinate_inputs_are_unchanged(self):
        for name, slices in synthetic_slices().items():
            with self.subTest(case=name):
                fixed_result, fixed_local = self.sample_real(self.build(slices))
                old_result, old_local = self.sample_real(self.build(slices, historical=True))
                for index in (0, 1):
                    np.testing.assert_array_equal(fixed_result[index], old_result[index])
                self.assertEqual(fixed_result[2:], old_result[2:])
                np.testing.assert_array_equal(fixed_local, old_local)

    def planned(self, *, historical=False):
        record = {"ROI": "synthetic-planned", "Simulated biopsy preparation dict": {"Nominal length mm": 1.234}}
        with allocation_contents(self.helper, 29., historical=historical):
            self.planner.build_simulated_biopsy_planning_state(record, [0., 0., 1.], [0., 0., 0.], 5, .35, False)
        return self.planner.build_simulated_biopsy_planning_sample_state(record, .2)

    def test_actual_planned_geometry_and_sampling_are_unchanged(self):
        old, fixed = self.planned(historical=True), self.planned()
        self.assert_same_geometry(old[MODEL], fixed[MODEL])
        self.assertEqual(len(fixed[MODEL][SAMPLES]), len(old[MODEL][SAMPLES]) + 1)
        for key in ("Planned sampled volume pts arr", "Planned sample bounding box pts arr",
                    "Planned sampled point count", "Planned biopsy cylinder length mm",
                    "Planned centroid count", "Planned centroid separation mm"):
            np.testing.assert_array_equal(old[key], fixed[key])
        self.assertGreater(fixed["Planned sampled point count"], 0)

    def test_pca_extent_is_symmetric_radius_not_observed_tip_projection(self):
        model = self.build(synthetic_slices()["asymmetric_bent"])
        centers = model["Structure centroid pts"]
        origin = centers.mean(axis=0)
        line = model["Best fit line of centroid pts"]
        length = np.linalg.norm(line[1] - line[0])
        np.testing.assert_allclose(line.mean(axis=0), origin, atol=1e-15, rtol=0)
        self.assertAlmostEqual(length, 2 * np.linalg.norm(centers - origin, axis=1).max())
        projected = (centers - origin) @ ((line[1] - line[0]) / length)
        self.assertGreater(length, np.ptp(projected))

    def test_checkpoint_keeps_actual_corrected_real_and_planned_samples(self):
        from validation.anatomical_checkpoint import _load_checkpoint, write_anatomical_checkpoint
        from validation.test_biopsy_preprocessing import BOUNDARY, biopsy_config, prepared_runtime

        real = self.build(synthetic_slices()["axial"])
        planned = self.planned()
        runtime = prepared_runtime()  # Substitute unrelated anatomy/volume/preparation only.
        records = runtime.pydicom_item["Bx ref"]
        records[0].update(real)
        records[1]["Simulated biopsy planning dict"].update(planned)
        with TemporaryDirectory() as directory:
            path = write_anatomical_checkpoint(runtime_state=runtime, pipeline_config=biopsy_config(),
                                              output_dir=Path(directory) / "checkpoint", checkpoint_name=BOUNDARY)
            document, arrays = _load_checkpoint(path, checkpoint_name=BOUNDARY)
            captured = [arrays[item["payload"]] for item in document["items"]
                        if item["path"][-1] == SAMPLES and item["state"] == "present"]
            self.assertEqual(len(captured), 2)
            for expected in (real[SAMPLES], planned[MODEL][SAMPLES]):
                self.assertTrue(any(np.array_equal(expected, actual) for actual in captured))
