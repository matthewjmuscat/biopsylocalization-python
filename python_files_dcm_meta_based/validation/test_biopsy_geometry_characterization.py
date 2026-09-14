"""Characterize an existing uninitialized diagnostic array without changing science.

Only native/geometry dependencies are replaced. The actual reconstruction helper
runs twice with different deterministic contents in newly allocated NumPy memory.
This test records a migration blocker, not a scientific acceptance assertion.
"""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


class BiopsyGeometryCharacterizationTests(unittest.TestCase):
    def test_centroid_samples_depend_on_uninitialized_memory_but_cylinder_does_not(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        modules = {
            "open3d": SimpleNamespace(), "misc_tools": SimpleNamespace(), "plotting_funcs": SimpleNamespace(),
            "centroid_finder": SimpleNamespace(centeroidfinder_numpy_3D=lambda pts: pts.mean(axis=0, keepdims=True)),
            "pca": SimpleNamespace(linear_fitter=lambda pts: np.array([[0., 0., 0.], [0., 0., 1.]])),
            "math_funcs": SimpleNamespace(rotation_matrix_from_vectors=lambda a, b: np.eye(3)),
            "biopsy_creator": SimpleNamespace(
                point_to_line_segment_distance=lambda *args: (0., np.zeros(3)),
                distance_of_most_distant_points_2d_projection=lambda *args: 0.,
                biopsy_points_creater_by_transport=lambda *args: points.T.copy()),
            "point_containment_tools": SimpleNamespace(
                create_point_cloud=lambda *args: SimpleNamespace(points=points.copy()),
                delaunay_obj=lambda *args: SimpleNamespace(generate_lineset=lambda: None)),
        }
        path = Path(__file__).resolve().parents[1] / "preprocessing/biopsy_processing/biopsy_geometry_helper.py"
        spec = importlib.util.spec_from_file_location("_geometry_characterization", path)
        helper = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(helper)
        results = []
        for residue in (29., 87.):
            numpy_proxy = SimpleNamespace(**{key: getattr(np, key) for key in dir(np)})
            numpy_proxy.empty = lambda shape, dtype=float, fill=residue: np.full(shape, fill, dtype=dtype)
            with patch.object(helper, "np", numpy_proxy):
                results.append(helper.build_reconstructed_biopsy_model_for_sampling_from_zslice_list(
                    [points, points + [0., 0., 1.]], 0.5))
        self.assertFalse(np.array_equal(results[0]["Centroid line sample pts"], results[1]["Centroid line sample pts"]))
        np.testing.assert_array_equal(results[0]["Centroid line sample pts"][1:, 0], 29.)
        np.testing.assert_array_equal(results[1]["Centroid line sample pts"][1:, 0], 87.)
        for field in ("Reconstructed structure pts arr", "Best fit line of centroid pts",
                      "Structure global centroid", "Rotated reconstructed structure pts arr rounded"):
            np.testing.assert_array_equal(results[0][field], results[1][field])
