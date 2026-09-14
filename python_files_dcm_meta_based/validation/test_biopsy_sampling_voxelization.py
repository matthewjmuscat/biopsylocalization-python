"""Preserve the existing terminal-exclusive sample lattice and voxel semantics.

Closed reconstructed geometry, sampled axial planes and labelled voxel bounds
are different products. These tests use actual scientific helpers on fabricated
contours; they do not redefine physical length from samples or voxel labels.
"""

import importlib
import unittest

import numpy as np
import pandas as pd

from config.pipeline import MCPrepConfig, MCSimulationCoreConfig


class BiopsySamplingVoxelizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry = importlib.import_module("preprocessing.biopsy_processing.biopsy_geometry_helper")
        cls.sampling = importlib.import_module("preprocessing.biopsy_processing.sampled_biopsy_processing")
        cls.sampler = importlib.import_module("sampling.biopsy_point_sampler")
        cls.frames = importlib.import_module("dataframe_builders")
        cls.sextants = importlib.import_module("preprocessing.biopsy_processing.biopsy_double_sextant")

    def sampled_record(self, length, *, oblique=False):
        direction = np.array([.3, .4, np.sqrt(.75)]) if oblique else np.array([0., 0., 1.])
        origin = np.array([12., -5., 7.]) if oblique else np.zeros(3)
        ring = np.array([[.35, 0., 0.], [-.35, 0., 0.], [0., .35, 0.], [0., -.35, 0.]])
        slices = [ring + origin + z * direction for z in (0., length / 2, length)]
        record = self.geometry.build_reconstructed_biopsy_model_for_sampling_from_zslice_list(slices, .35)
        record.update({"ROI": "fabricated", "Ref #": 1, "Simulated bool": False, "Simulated type": "Real"})
        patient = {"Bx ref": [record]}
        args = self.sampling.build_patient_sampled_biopsy_sampling_args(
            patient_uid="synthetic", pydicom_item=patient, bx_ref="Bx ref", bx_sample_pts_lattice_spacing=1.)
        result = self.sampler.sample_biopsy_points_from_reconstructed_global_delaunay_convex_structure(*args[0])
        self.sampling.store_patient_sampled_biopsy_results(patient_uid="synthetic", pydicom_item=patient,
            bx_ref="Bx ref", parallel_results_sampled_bx_points_from_global_delaunay_arr_and_bounding_box_arr=[result])
        self.sampling.create_patient_biopsy_oriented_coordinate_system(
            patient_uid="synthetic", pydicom_item=patient, bx_ref="Bx ref")
        return record

    def test_production_config_defaults_use_matching_one_mm_resolutions(self):
        self.assertEqual(MCPrepConfig().bx_sample_pts_lattice_spacing, 1.)
        self.assertEqual(MCSimulationCoreConfig().biopsy_z_voxel_length, 1.)

    def test_first_plane_present_and_exact_terminal_plane_excluded(self):
        for oblique in (False, True):
            for length, planes in ((1., [0.]), (2., [0., 1.]), (2.25, [0., 1., 2.])):
                with self.subTest(length=length, oblique=oblique):
                    record = self.sampled_record(length, oblique=oblique)
                    np.testing.assert_allclose(record["Random uniformly sampled volume pts bx coord sys arr"][:, 2],
                                               planes, atol=1e-12, rtol=0)
                    self.assertAlmostEqual(record["Reconstructed biopsy cylinder length (from contour data)"], length)
                    axis = record["Centroid line unit vec (bx needle base to bx needle tip)"]
                    self.assertAlmostEqual(np.ptp(record["Reconstructed structure pts arr"] @ axis), length)

    def test_existing_probe_margin_is_distinct_from_physical_extent(self):
        # The unchanged sampler tests z + 1e-4 mm. An interior plane within this
        # margin of the terminal endpoint is also excluded; document the limit.
        for length, planes in ((1.00005, [0.]), (1.0002, [0., 1.])):
            with self.subTest(length=length):
                record = self.sampled_record(length)
                np.testing.assert_allclose(record["Random uniformly sampled volume pts bx coord sys arr"][:, 2],
                                           planes, atol=1e-12, rtol=0)

    def test_one_mm_boundaries_snap_and_go_up_deterministically(self):
        values = np.array([0., np.nextafter(1., -np.inf), 1., np.nextafter(1., np.inf),
                           1. - 1e-7, 1. + 1e-7, 2., 2.25])
        source = pd.DataFrame({"z": values})
        first = self.frames.add_voxel_columns_helper_func(source, 1., "z")
        second = self.frames.add_voxel_columns_helper_func(source, 1., "z")
        pd.testing.assert_frame_equal(first, second, check_exact=True)
        np.testing.assert_array_equal(first["Voxel index"], [1, 2, 2, 2, 2, 2, 3, 3])
        np.testing.assert_array_equal(first["Voxel begin (Z)"], [0, 1, 1, 1, 1, 1, 2, 2])
        self.assertEqual(list(source.columns), ["z"])
        self.assertEqual(first.iloc[-1]["Voxel end (Z)"], 3.)

    def test_voxelizer_snaps_off_lattice_values_rather_than_continuous_floor_binning(self):
        table = self.frames.add_voxel_columns_helper_func(pd.DataFrame({"z": [.49, .51]}), 1., "z")
        np.testing.assert_array_equal(table["Voxel index"], [1, 2])

    def test_corrected_extent_propagates_into_point_and_voxel_tables(self):
        for length, expected_indices in ((2., [1, 2]), (2.25, [1, 2, 3])):
            with self.subTest(length=length):
                record = self.sampled_record(length)
                patient = {"Bx ref": [record], "All ref": {"Multi-structure pre-processing output dataframes dict": {
                    "Selected structures": pd.DataFrame([{"Struct ref type": "OAR ref", "Struct found bool": False}])}}}
                points = self.sextants._build_patient_per_sample_point_double_sextant_dataframe(
                    "synthetic", patient, "All ref", "Bx ref", "OAR ref", 1.)
                voxels = self.sextants._build_per_voxel_double_sextant_dataframe(points)
                np.testing.assert_array_equal(points["Voxel index"], expected_indices)
                np.testing.assert_array_equal(voxels["Voxel index"], expected_indices)
                self.assertEqual(points["Voxel end (Z)"].max(), float(expected_indices[-1]))
                self.assertAlmostEqual(record["Reconstructed biopsy cylinder length (from contour data)"], length)
