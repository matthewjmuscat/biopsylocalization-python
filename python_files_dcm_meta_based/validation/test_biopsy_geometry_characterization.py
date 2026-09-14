"""Permanent numerical invariants for projected-centroid biopsy reconstruction.

Fabricated contours exercise actual PCA, cylinder rings, SciPy Delaunay,
sampling and biopsy-frame transforms. Patient data and GPU volume integration
do not run. Historical-vs-current reporting belongs in the disposable utility.
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

from preprocessing.biopsy_processing.fitted_segment import fit_biopsy_segment


SAMPLES = "Centroid line sample pts"
MODEL = "Planned reconstructed biopsy model dict"


def synthetic_slices():
    ring = np.array([[.35, 0., 0.], [-.35, 0., 0.], [0., .35, 0.], [0., -.35, 0.]])
    direction = np.array([.3, .4, np.sqrt(.75)])
    oblique = np.array([10., -2., 5.]) + np.array([0., .617, 1.234])[:, None] * direction
    centers = {
        "axial": np.array([[0., 0., 0.], [0., 0., .5], [0., 0., 1.]]),
        "axial_asymmetric_sampling": np.array([[0., 0., 0.], [0., 0., .2], [0., 0., 1.]]),
        "crosses_one_mm": np.array([[0., 0., 0.], [0., 0., .525], [0., 0., 1.05]]),
        "oblique": oblique,
        "reversed_slices": oblique[::-1],
        "asymmetric_bent": np.array([[0., 0., 0.], [.15, 0., .2], [.1, .1, 1.]]),
        "short": np.array([[0., 0., 0.], [0., 0., .05]]),
    }
    return {name: [ring + point for point in points] for name, points in centers.items()}


@contextmanager
def allocation_contents(helper, residue):
    """Poison the builder's empty allocations without altering native dependencies."""
    proxy = SimpleNamespace(**{key: getattr(np, key) for key in dir(np)})
    proxy.empty = lambda shape, dtype=float: np.full(shape, residue, dtype=dtype)
    segment_module = importlib.import_module("preprocessing.biopsy_processing.fitted_segment")
    with patch.object(helper, "np", proxy), patch.object(segment_module, "np", proxy):
        yield


class BiopsyGeometryCharacterizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helper = importlib.import_module("preprocessing.biopsy_processing.biopsy_geometry_helper")
        cls.sampling = importlib.import_module("preprocessing.biopsy_processing.sampled_biopsy_processing")
        cls.planner = importlib.import_module("preprocessing.biopsy_processing.simulated_biopsy_planner")
        cls.point_sampler = importlib.import_module("sampling.biopsy_point_sampler")

    def build(self, slices, residue=29.):
        with allocation_contents(self.helper, residue):
            return self.helper.build_reconstructed_biopsy_model_for_sampling_from_zslice_list(slices, .35)

    def assert_same_geometry(self, left, right):
        self.assertEqual(left.keys(), right.keys())
        for key in left:
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

    def test_projected_segment_and_cylinder_invariants(self):
        for name, slices in synthetic_slices().items():
            with self.subTest(case=name):
                model = self.build(slices)
                centers = np.asarray([s.mean(axis=0) for s in slices])
                segment = fit_biopsy_segment(centers)
                line = model["Best fit line of centroid pts"]
                length = model["Reconstructed biopsy cylinder length (from contour data)"]
                intervals = math.ceil(length / .1)
                projections = (centers - centers.mean(axis=0)) @ segment.axis_direction
                expected = centers.mean(axis=0) + np.array([projections.min(), projections.max()])[:, None] * segment.axis_direction
                np.testing.assert_allclose(line, expected, atol=2e-14, rtol=0)
                self.assertAlmostEqual(length, float(np.ptp(projections)))
                self.assertAlmostEqual(length, np.linalg.norm(line[1] - line[0]))
                self.assertTrue(np.all(projections >= segment.projection_bounds_mm[0] - 2e-14))
                self.assertTrue(np.all(projections <= segment.projection_bounds_mm[1] + 2e-14))
                samples = model[SAMPLES]
                self.assertEqual(samples.shape, (intervals + 1, 3))
                self.assertTrue(np.isfinite(samples).all())
                np.testing.assert_array_equal(samples[[0, -1]], line)
                t = (samples - line[0]) @ segment.axis_direction
                np.testing.assert_allclose(t, np.arange(intervals + 1) * length / intervals, atol=2e-14, rtol=0)
                np.testing.assert_allclose(samples, line[0] + t[:, None] * segment.axis_direction, atol=2e-14, rtol=0)
                steps = np.linalg.norm(np.diff(samples, axis=0), axis=1)
                np.testing.assert_allclose(steps, length / intervals, atol=2e-14, rtol=0)
                self.assertLessEqual(steps.max(), .1 + 2e-14)
                points = model["Reconstructed structure pts arr"]
                self.assertEqual(points.shape, (20 * (intervals + 1), 3))
                rings = points.reshape(intervals + 1, 20, 3)
                np.testing.assert_allclose(rings.mean(axis=1), samples, atol=2e-14, rtol=0)
                np.testing.assert_allclose(rings.mean(axis=1)[[0, -1]], line, atol=2e-14, rtol=0)
                self.assertAlmostEqual(np.ptp(points @ segment.axis_direction), length)
                np.testing.assert_allclose(np.linalg.norm(rings - samples[:, None, :], axis=2), .35, atol=2e-14, rtol=0)
                np.testing.assert_array_equal(model["Reconstructed structure delaunay global"].delaunay_triangulation.points, points)

    def test_allocation_residue_cannot_change_line_samples_or_geometry(self):
        for name, slices in synthetic_slices().items():
            reference = self.build(slices)
            for residue in (87., -29., np.nan, 1e200):
                with self.subTest(case=name, residue=residue):
                    self.assert_same_geometry(reference, self.build(slices, residue))

    def sample_real(self, model):
        record = {**model, "ROI": "synthetic-real"}
        patient = {"Bx ref": [record]}
        args = self.sampling.build_patient_sampled_biopsy_sampling_args(
            patient_uid="synthetic", pydicom_item=patient, bx_ref="Bx ref", bx_sample_pts_lattice_spacing=.2)
        self.assertIs(args[0][1], model["Reconstructed structure delaunay global"].delaunay_triangulation)
        self.assertIs(args[0][2], model["Reconstructed structure pts arr"])
        result = self.point_sampler.sample_biopsy_points_from_reconstructed_global_delaunay_convex_structure(*args[0])
        self.assertGreater(result[2], 0)
        self.sampling.store_patient_sampled_biopsy_results(patient_uid="synthetic", pydicom_item=patient,
            bx_ref="Bx ref", parallel_results_sampled_bx_points_from_global_delaunay_arr_and_bounding_box_arr=[result])
        self.sampling.create_patient_biopsy_oriented_coordinate_system(
            patient_uid="synthetic", pydicom_item=patient, bx_ref="Bx ref")
        return result, record["Random uniformly sampled volume pts bx coord sys arr"]

    def test_actual_sampling_and_biopsy_frame_are_deterministic(self):
        for name, slices in synthetic_slices().items():
            with self.subTest(case=name):
                left, left_local = self.sample_real(self.build(slices))
                right, right_local = self.sample_real(self.build(slices, np.nan))
                for index in (0, 1):
                    np.testing.assert_array_equal(left[index], right[index])
                self.assertEqual(left[2:], right[2:])
                np.testing.assert_array_equal(left_local, right_local)
                np.testing.assert_allclose(left_local[:, 2] / .2, np.round(left_local[:, 2] / .2), atol=1e-12, rtol=0)

    def planned(self):
        record = {"ROI": "synthetic-planned", "Simulated biopsy preparation dict": {"Nominal length mm": 1.234}}
        self.planner.build_simulated_biopsy_planning_state(record, [0., 0., 1.], [0., 0., 0.], 5, .35, False)
        return self.planner.build_simulated_biopsy_planning_sample_state(record, .2)

    def test_actual_planning_uses_the_same_closed_segment(self):
        left, right = self.planned(), self.planned()
        self.assert_same_geometry(left[MODEL], right[MODEL])
        self.assertAlmostEqual(left["Planned biopsy cylinder length mm"], 1.234)
        points = left[MODEL]["Reconstructed structure pts arr"]
        self.assertAlmostEqual(np.ptp(points[:, 2]), 1.234)
        self.assertEqual(points.shape[0], 20 * len(left[MODEL][SAMPLES]))
        np.testing.assert_array_equal(left["Planned sampled volume pts arr"], right["Planned sampled volume pts arr"])
        self.assertGreater(left["Planned sampled point count"], 0)

    def test_axis_matches_generic_pca_but_extent_uses_projections(self):
        pca = importlib.import_module("pca")
        for name, slices in synthetic_slices().items():
            with self.subTest(case=name):
                centers = np.asarray([s.mean(axis=0) for s in slices])
                fitted = fit_biopsy_segment(centers)
                generic = pca.linear_fitter(centers.T)
                direction = generic[1] - generic[0]
                np.testing.assert_allclose(fitted.axis_direction, direction / np.linalg.norm(direction), atol=2e-14, rtol=0)
                self.assertAlmostEqual(np.linalg.norm(direction), 2 * np.linalg.norm(centers - centers.mean(axis=0), axis=1).max())
                if name in ("asymmetric_bent", "axial_asymmetric_sampling"):
                    self.assertLess(fitted.length_mm, np.linalg.norm(direction))

    def test_reversed_oblique_input_retains_geometry_and_patient_z_orientation(self):
        cases = synthetic_slices()
        left, right = (self.build(cases[name]) for name in ("oblique", "reversed_slices"))
        for model in (left, right):
            axis = model["Centroid line unit vec (bx needle base to bx needle tip)"]
            self.assertGreater(axis[2], 0)
            endpoints = model["Best fit line of centroid pts"]
            np.testing.assert_array_equal(model["Biopsy coord sys origin translation vec"], -endpoints[endpoints[:, 2].argmin()])
        np.testing.assert_allclose(left["Best fit line of centroid pts"], right["Best fit line of centroid pts"], atol=2e-14, rtol=0)

    def test_invalid_extents_fail_clearly_and_horizontal_orientation_is_not_guessed(self):
        for points in ([], [[0, 0, 0]], [[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 1e-12]],
                       [[0, 0, 0], [0, 0, np.nan]], [[0, 0], [1, 1]]):
            with self.subTest(points=points):
                with self.assertRaisesRegex(ValueError, "finite XYZ|near zero"):
                    fit_biopsy_segment(points)
        for spacing in (0., -1., np.nan, np.inf):
            with self.assertRaisesRegex(ValueError, "spacing"):
                fit_biopsy_segment([[0, 0, 0], [0, 0, 1]], max_ring_spacing_mm=spacing)
        horizontal = fit_biopsy_segment([[0, 0, 0], [1, 0, 0]])
        self.assertAlmostEqual(horizontal.length_mm, 1.)
        with self.assertRaisesRegex(ValueError, "horizontal biopsy axis"):
            self.build([np.array([[0., 0., 0.]]), np.array([[1., 0., 0.]])])

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
