"""Synthetic checks for dose NN render scene contracts."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from mc.simulation.per_patient.dose import (
    MC_DOSE_ORIGINAL_POINT_INDEX_COLUMN,
    MC_DOSE_TRIAL_COLUMN,
    MC_DOSE_VALUE_COLUMN,
)
from mc.visualization.dose_nn_scene import (
    DOSE_NN_NEAREST_DISTANCES_COLUMN,
    DOSE_NN_NEAREST_DOSES_COLUMN,
    DOSE_NN_NEAREST_POINTS_COLUMN,
    DOSE_NN_QUERY_POINT_COLUMN,
    DoseNNRenderConfig,
    DoseNNSceneMetadata,
    build_dose_nn_render_scene_from_dataframe,
    prepare_dose_nn_render_scene,
)


class DoseNNRenderSceneTests(unittest.TestCase):
    def test_build_scene_from_dataframe_preserves_nn_shape(self):
        scene = build_dose_nn_render_scene_from_dataframe(
            _synthetic_nn_dataframe(),
            lattice_points=_synthetic_lattice_points(),
            lattice_doses=_synthetic_lattice_doses(),
            metadata=DoseNNSceneMetadata(patient_uid="synthetic", biopsy_roi="Bx 1"),
        )

        self.assertEqual(scene.metadata.patient_uid, "synthetic")
        self.assertEqual(scene.available_trials, (0, 1))
        self.assertEqual(scene.num_query_points, 6)
        self.assertEqual(scene.num_nearest_neighbours, 2)
        self.assertEqual(scene.nearest_lattice_points.shape, (6, 2, 3))
        self.assertEqual(scene.nearest_lattice_doses.shape, (6, 2))

    def test_prepare_scene_filters_trials_lattice_points_and_vectors(self):
        scene = build_dose_nn_render_scene_from_dataframe(
            _synthetic_nn_dataframe(),
            lattice_points=_synthetic_lattice_points(),
            lattice_doses=_synthetic_lattice_doses(),
        )
        prepared_scene = prepare_dose_nn_render_scene(
            scene,
            DoseNNRenderConfig(
                selected_trials=(1,),
                dose_threshold_min=20.0,
                max_lattice_points=2,
                biopsy_point_stride=2,
                vector_stride=2,
            ),
        )

        self.assertTrue(np.all(prepared_scene.trial_numbers == 1))
        self.assertEqual(prepared_scene.biopsy_points.shape, (2, 3))
        self.assertEqual(prepared_scene.lattice_points.shape, (2, 3))
        self.assertTrue(np.all(prepared_scene.lattice_doses >= 20.0))
        self.assertEqual(prepared_scene.num_vectors, 2)

    def test_missing_required_dataframe_column_fails_closed(self):
        dataframe = _synthetic_nn_dataframe().drop(columns=[DOSE_NN_NEAREST_DISTANCES_COLUMN])

        with self.assertRaisesRegex(ValueError, DOSE_NN_NEAREST_DISTANCES_COLUMN):
            build_dose_nn_render_scene_from_dataframe(
                dataframe,
                lattice_points=_synthetic_lattice_points(),
                lattice_doses=_synthetic_lattice_doses(),
            )

    def test_invalid_threshold_order_fails_closed(self):
        scene = build_dose_nn_render_scene_from_dataframe(
            _synthetic_nn_dataframe(),
            lattice_points=_synthetic_lattice_points(),
            lattice_doses=_synthetic_lattice_doses(),
        )

        with self.assertRaisesRegex(ValueError, "dose_threshold_min"):
            prepare_dose_nn_render_scene(
                scene,
                DoseNNRenderConfig(dose_threshold_min=30.0, dose_threshold_max=10.0),
            )


def _synthetic_nn_dataframe() -> pd.DataFrame:
    rows = []
    for trial_number in (0, 1):
        for point_index in range(3):
            query_point = [float(point_index), float(trial_number), 0.0]
            rows.append(
                {
                    MC_DOSE_ORIGINAL_POINT_INDEX_COLUMN: point_index,
                    MC_DOSE_TRIAL_COLUMN: trial_number,
                    DOSE_NN_QUERY_POINT_COLUMN: query_point,
                    MC_DOSE_VALUE_COLUMN: float(10 + trial_number + point_index),
                    DOSE_NN_NEAREST_POINTS_COLUMN: [
                        [query_point[0], query_point[1], 1.0],
                        [query_point[0] + 1.0, query_point[1], 1.0],
                    ],
                    DOSE_NN_NEAREST_DOSES_COLUMN: [20.0 + point_index, 30.0 + point_index],
                    DOSE_NN_NEAREST_DISTANCES_COLUMN: [1.0, 2.0],
                }
            )
    return pd.DataFrame(rows)


def _synthetic_lattice_points() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=float,
    )


def _synthetic_lattice_doses() -> np.ndarray:
    return np.array([5.0, 15.0, 25.0, 35.0, 45.0], dtype=float)


if __name__ == "__main__":
    unittest.main()
