"""Render-agnostic dosimetric nearest-neighbour scene contracts."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import numpy as np

from ..simulation.per_patient.dose import (
    MC_DOSE_ORIGINAL_POINT_INDEX_COLUMN,
    MC_DOSE_TRIAL_COLUMN,
    MC_DOSE_VALUE_COLUMN,
)

DOSE_NN_QUERY_POINT_COLUMN = "Struct test pt vec"
DOSE_NN_NEAREST_POINTS_COLUMN = "Nearest phys space points"
DOSE_NN_NEAREST_DOSES_COLUMN = "Nearest doses"
DOSE_NN_NEAREST_DISTANCES_COLUMN = "Nearest distances"


@dataclass(frozen=True, slots=True)
class DoseNNSceneMetadata:
    """Identity and provenance for a dose nearest-neighbour render scene."""

    patient_uid: str = ""
    biopsy_roi: str = ""
    biopsy_index: int | None = None
    localization_kind: str = "dose"
    result_column: str = MC_DOSE_VALUE_COLUMN
    source_label: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DoseNNRenderScene:
    """Renderer-neutral arrays for one biopsy/dose nearest-neighbour scene."""

    metadata: DoseNNSceneMetadata
    lattice_points: np.ndarray
    lattice_doses: np.ndarray
    original_point_indices: np.ndarray
    trial_numbers: np.ndarray
    biopsy_points: np.ndarray
    interpolated_biopsy_doses: np.ndarray
    nearest_lattice_points: np.ndarray
    nearest_lattice_doses: np.ndarray
    nearest_distances: np.ndarray

    @property
    def num_query_points(self) -> int:
        return int(self.biopsy_points.shape[0])

    @property
    def num_nearest_neighbours(self) -> int:
        return int(self.nearest_lattice_points.shape[1])

    @property
    def available_trials(self) -> tuple[int, ...]:
        return tuple(int(trial_number) for trial_number in np.unique(self.trial_numbers))


@dataclass(frozen=True, slots=True)
class DoseNNRenderConfig:
    """Renderer-neutral display filters for a dose nearest-neighbour scene."""

    selected_trials: tuple[int, ...] | None = None
    reference_trial_numbers: tuple[int, ...] | None = None
    dose_threshold_min: float | None = None
    dose_threshold_max: float | None = None
    max_lattice_points: int | None = None
    spatial_radius_mm: float | None = None
    biopsy_point_stride: int = 1
    vector_stride: int = 1
    show_biopsy_points: bool = True
    show_reference_biopsy_points: bool = False
    show_lattice_points: bool = True
    show_dose_colorwash: bool = False
    show_nearest_neighbour_points: bool = True
    show_nearest_neighbour_vectors: bool = True


@dataclass(frozen=True, slots=True)
class DoseNNPreparedScene:
    """A scene after trial, lattice, point, and vector filters are applied."""

    metadata: DoseNNSceneMetadata
    config: DoseNNRenderConfig
    lattice_points: np.ndarray
    lattice_doses: np.ndarray
    original_point_indices: np.ndarray
    trial_numbers: np.ndarray
    biopsy_points: np.ndarray
    reference_biopsy_points: np.ndarray
    reference_trial_numbers: np.ndarray
    interpolated_biopsy_doses: np.ndarray
    nearest_lattice_points: np.ndarray
    nearest_lattice_doses: np.ndarray
    nearest_distances: np.ndarray
    vector_start_points: np.ndarray
    vector_end_points: np.ndarray

    @property
    def num_vectors(self) -> int:
        return int(self.vector_start_points.shape[0])


def build_dose_nn_render_scene_from_dataframe(
    localization_dataframe: Any,
    *,
    lattice_points: Any,
    lattice_doses: Any,
    metadata: DoseNNSceneMetadata | None = None,
    result_column: str = MC_DOSE_VALUE_COLUMN,
) -> DoseNNRenderScene:
    """Build a renderer-neutral scene from the existing dose NN dataframe."""
    required_columns = {
        MC_DOSE_TRIAL_COLUMN,
        DOSE_NN_QUERY_POINT_COLUMN,
        result_column,
        DOSE_NN_NEAREST_POINTS_COLUMN,
        DOSE_NN_NEAREST_DOSES_COLUMN,
        DOSE_NN_NEAREST_DISTANCES_COLUMN,
    }
    missing_columns = sorted(required_columns.difference(set(localization_dataframe.columns)))
    if missing_columns:
        raise ValueError("dose NN dataframe is missing required columns: {}".format(missing_columns))

    resolved_metadata = metadata or DoseNNSceneMetadata()
    resolved_metadata = replace(resolved_metadata, result_column=str(result_column))

    if MC_DOSE_ORIGINAL_POINT_INDEX_COLUMN in localization_dataframe.columns:
        original_point_indices = localization_dataframe[MC_DOSE_ORIGINAL_POINT_INDEX_COLUMN].to_numpy(dtype=int)
    else:
        original_point_indices = np.arange(len(localization_dataframe), dtype=int)

    return build_dose_nn_render_scene(
        metadata=resolved_metadata,
        lattice_points=lattice_points,
        lattice_doses=lattice_doses,
        original_point_indices=original_point_indices,
        trial_numbers=localization_dataframe[MC_DOSE_TRIAL_COLUMN].to_numpy(dtype=int),
        biopsy_points=localization_dataframe[DOSE_NN_QUERY_POINT_COLUMN].tolist(),
        interpolated_biopsy_doses=localization_dataframe[result_column].to_numpy(dtype=float),
        nearest_lattice_points=localization_dataframe[DOSE_NN_NEAREST_POINTS_COLUMN].tolist(),
        nearest_lattice_doses=localization_dataframe[DOSE_NN_NEAREST_DOSES_COLUMN].tolist(),
        nearest_distances=localization_dataframe[DOSE_NN_NEAREST_DISTANCES_COLUMN].tolist(),
    )


def build_dose_nn_render_scene(
    *,
    metadata: DoseNNSceneMetadata | None = None,
    lattice_points: Any,
    lattice_doses: Any,
    original_point_indices: Any,
    trial_numbers: Any,
    biopsy_points: Any,
    interpolated_biopsy_doses: Any,
    nearest_lattice_points: Any,
    nearest_lattice_doses: Any,
    nearest_distances: Any,
) -> DoseNNRenderScene:
    """Build a renderer-neutral scene from explicit arrays."""
    resolved_lattice_points = _as_points_array("lattice_points", lattice_points)
    resolved_lattice_doses = _as_scalar_array("lattice_doses", lattice_doses)
    if resolved_lattice_points.shape[0] != resolved_lattice_doses.shape[0]:
        raise ValueError("lattice_points and lattice_doses must have the same length")

    resolved_biopsy_points = _as_points_array("biopsy_points", biopsy_points)
    resolved_trial_numbers = _as_index_array("trial_numbers", trial_numbers)
    resolved_original_point_indices = _as_index_array("original_point_indices", original_point_indices)
    resolved_interpolated_biopsy_doses = _as_scalar_array(
        "interpolated_biopsy_doses",
        interpolated_biopsy_doses,
    )
    resolved_nearest_lattice_points = _as_nearest_points_array("nearest_lattice_points", nearest_lattice_points)
    resolved_nearest_lattice_doses = _as_nearest_scalar_array("nearest_lattice_doses", nearest_lattice_doses)
    resolved_nearest_distances = _as_nearest_scalar_array("nearest_distances", nearest_distances)

    num_query_points = resolved_biopsy_points.shape[0]
    for name, array in (
        ("trial_numbers", resolved_trial_numbers),
        ("original_point_indices", resolved_original_point_indices),
        ("interpolated_biopsy_doses", resolved_interpolated_biopsy_doses),
        ("nearest_lattice_points", resolved_nearest_lattice_points),
        ("nearest_lattice_doses", resolved_nearest_lattice_doses),
        ("nearest_distances", resolved_nearest_distances),
    ):
        if array.shape[0] != num_query_points:
            raise ValueError("{} must have one row per biopsy query point".format(name))

    nearest_shape = resolved_nearest_lattice_points.shape[:2]
    if resolved_nearest_lattice_doses.shape != nearest_shape:
        raise ValueError("nearest_lattice_doses must match nearest_lattice_points row/neighbour shape")
    if resolved_nearest_distances.shape != nearest_shape:
        raise ValueError("nearest_distances must match nearest_lattice_points row/neighbour shape")

    return DoseNNRenderScene(
        metadata=metadata or DoseNNSceneMetadata(),
        lattice_points=resolved_lattice_points,
        lattice_doses=resolved_lattice_doses,
        original_point_indices=resolved_original_point_indices,
        trial_numbers=resolved_trial_numbers,
        biopsy_points=resolved_biopsy_points,
        interpolated_biopsy_doses=resolved_interpolated_biopsy_doses,
        nearest_lattice_points=resolved_nearest_lattice_points,
        nearest_lattice_doses=resolved_nearest_lattice_doses,
        nearest_distances=resolved_nearest_distances,
    )


def prepare_dose_nn_render_scene(
    scene: DoseNNRenderScene,
    config: DoseNNRenderConfig | None = None,
) -> DoseNNPreparedScene:
    """Apply renderer-neutral display filters without mutating the scene."""
    resolved_config = normalize_dose_nn_render_config(config or DoseNNRenderConfig())
    row_indices = _select_query_row_indices(scene, resolved_config)
    reference_row_indices = _select_reference_query_row_indices(scene, resolved_config)
    biopsy_points = scene.biopsy_points[row_indices]
    nearest_lattice_points = scene.nearest_lattice_points[row_indices]

    lattice_indices = select_lattice_point_indices(scene, resolved_config, centers=biopsy_points)
    vector_start_points, vector_end_points = build_dose_nn_vector_points(
        biopsy_points,
        nearest_lattice_points,
        vector_stride=resolved_config.vector_stride,
        show_vectors=resolved_config.show_nearest_neighbour_vectors,
    )

    return DoseNNPreparedScene(
        metadata=scene.metadata,
        config=resolved_config,
        lattice_points=scene.lattice_points[lattice_indices],
        lattice_doses=scene.lattice_doses[lattice_indices],
        original_point_indices=scene.original_point_indices[row_indices],
        trial_numbers=scene.trial_numbers[row_indices],
        biopsy_points=biopsy_points,
        reference_biopsy_points=scene.biopsy_points[reference_row_indices],
        reference_trial_numbers=scene.trial_numbers[reference_row_indices],
        interpolated_biopsy_doses=scene.interpolated_biopsy_doses[row_indices],
        nearest_lattice_points=nearest_lattice_points,
        nearest_lattice_doses=scene.nearest_lattice_doses[row_indices],
        nearest_distances=scene.nearest_distances[row_indices],
        vector_start_points=vector_start_points,
        vector_end_points=vector_end_points,
    )


def normalize_dose_nn_render_config(config: DoseNNRenderConfig) -> DoseNNRenderConfig:
    """Normalize and validate renderer-neutral dose scene filters."""
    selected_trials = _normalize_trial_numbers(config.selected_trials)
    reference_trial_numbers = _normalize_trial_numbers(config.reference_trial_numbers)

    biopsy_point_stride = int(config.biopsy_point_stride)
    if biopsy_point_stride <= 0:
        raise ValueError("biopsy_point_stride must be positive")
    vector_stride = int(config.vector_stride)
    if vector_stride <= 0:
        raise ValueError("vector_stride must be positive")

    max_lattice_points = config.max_lattice_points
    if max_lattice_points is not None:
        max_lattice_points = int(max_lattice_points)
        if max_lattice_points <= 0:
            raise ValueError("max_lattice_points must be positive when provided")

    spatial_radius_mm = config.spatial_radius_mm
    if spatial_radius_mm is not None:
        spatial_radius_mm = float(spatial_radius_mm)
        if spatial_radius_mm <= 0:
            spatial_radius_mm = None

    dose_threshold_min = None if config.dose_threshold_min is None else float(config.dose_threshold_min)
    dose_threshold_max = None if config.dose_threshold_max is None else float(config.dose_threshold_max)
    if dose_threshold_min is not None and dose_threshold_max is not None:
        if dose_threshold_min > dose_threshold_max:
            raise ValueError("dose_threshold_min cannot exceed dose_threshold_max")

    return DoseNNRenderConfig(
        selected_trials=selected_trials,
        reference_trial_numbers=reference_trial_numbers,
        dose_threshold_min=dose_threshold_min,
        dose_threshold_max=dose_threshold_max,
        max_lattice_points=max_lattice_points,
        spatial_radius_mm=spatial_radius_mm,
        biopsy_point_stride=biopsy_point_stride,
        vector_stride=vector_stride,
        show_biopsy_points=bool(config.show_biopsy_points),
        show_reference_biopsy_points=bool(config.show_reference_biopsy_points),
        show_lattice_points=bool(config.show_lattice_points),
        show_dose_colorwash=bool(config.show_dose_colorwash),
        show_nearest_neighbour_points=bool(config.show_nearest_neighbour_points),
        show_nearest_neighbour_vectors=bool(config.show_nearest_neighbour_vectors),
    )


def select_lattice_point_indices(
    scene: DoseNNRenderScene,
    config: DoseNNRenderConfig,
    *,
    centers: Any | None = None,
) -> np.ndarray:
    """Return deterministic lattice point indices for display."""
    resolved_config = normalize_dose_nn_render_config(config)
    mask = np.ones(scene.lattice_points.shape[0], dtype=bool)
    if resolved_config.dose_threshold_min is not None:
        mask &= scene.lattice_doses >= resolved_config.dose_threshold_min
    if resolved_config.dose_threshold_max is not None:
        mask &= scene.lattice_doses <= resolved_config.dose_threshold_max
    if resolved_config.spatial_radius_mm is not None and centers is not None:
        center_points = _as_points_array("centers", centers)
        if center_points.shape[0] > 0:
            mask &= _points_within_radius_mask(
                scene.lattice_points,
                center_points,
                resolved_config.spatial_radius_mm,
            )

    indices = np.flatnonzero(mask)
    if resolved_config.max_lattice_points is not None and indices.shape[0] > resolved_config.max_lattice_points:
        sampled_positions = np.linspace(
            0,
            indices.shape[0] - 1,
            num=resolved_config.max_lattice_points,
            dtype=int,
        )
        indices = indices[sampled_positions]
    return indices


def build_dose_nn_vector_points(
    biopsy_points: Any,
    nearest_lattice_points: Any,
    *,
    vector_stride: int = 1,
    show_vectors: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Build vector start/end point arrays from biopsy points to NN lattice points."""
    resolved_vector_stride = int(vector_stride)
    if resolved_vector_stride <= 0:
        raise ValueError("vector_stride must be positive")
    if not bool(show_vectors):
        return _empty_points_array(), _empty_points_array()

    resolved_biopsy_points = _as_points_array("biopsy_points", biopsy_points)
    resolved_nearest_lattice_points = _as_nearest_points_array("nearest_lattice_points", nearest_lattice_points)
    if resolved_biopsy_points.shape[0] != resolved_nearest_lattice_points.shape[0]:
        raise ValueError("nearest_lattice_points must have one row per biopsy point")
    if resolved_biopsy_points.shape[0] == 0:
        return _empty_points_array(), _empty_points_array()

    num_neighbours = resolved_nearest_lattice_points.shape[1]
    vector_start_points = np.repeat(resolved_biopsy_points, num_neighbours, axis=0)
    vector_end_points = np.reshape(resolved_nearest_lattice_points, (-1, 3), order="C")
    return vector_start_points[::resolved_vector_stride], vector_end_points[::resolved_vector_stride]


def _select_query_row_indices(scene: DoseNNRenderScene, config: DoseNNRenderConfig) -> np.ndarray:
    mask = np.ones(scene.num_query_points, dtype=bool)
    if config.selected_trials is not None:
        mask &= np.isin(scene.trial_numbers, np.asarray(config.selected_trials, dtype=int))
    row_indices = np.flatnonzero(mask)
    return row_indices[::config.biopsy_point_stride]


def _select_reference_query_row_indices(scene: DoseNNRenderScene, config: DoseNNRenderConfig) -> np.ndarray:
    if not bool(config.show_reference_biopsy_points) or config.reference_trial_numbers is None:
        return np.asarray([], dtype=int)
    mask = np.isin(scene.trial_numbers, np.asarray(config.reference_trial_numbers, dtype=int))
    row_indices = np.flatnonzero(mask)
    return row_indices[::config.biopsy_point_stride]


def _normalize_trial_numbers(trial_numbers: tuple[int, ...] | None) -> tuple[int, ...] | None:
    if trial_numbers is None:
        return None
    resolved_trial_numbers = tuple(int(trial_number) for trial_number in tuple(trial_numbers))
    if len(resolved_trial_numbers) == 0:
        return None
    return resolved_trial_numbers


def _points_within_radius_mask(points: np.ndarray, centers: np.ndarray, radius: float) -> np.ndarray:
    radius_squared = float(radius) ** 2
    mask = np.zeros(points.shape[0], dtype=bool)
    chunk_size = 100_000
    for start_index in range(0, points.shape[0], chunk_size):
        stop_index = min(start_index + chunk_size, points.shape[0])
        deltas = points[start_index:stop_index, None, :] - centers[None, :, :]
        distances_squared = np.sum(deltas * deltas, axis=2)
        mask[start_index:stop_index] = np.any(distances_squared <= radius_squared, axis=1)
    return mask


def _as_points_array(name: str, values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return _empty_points_array()
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("{} must be a two-dimensional array with shape (n, 3)".format(name))
    return array


def _as_nearest_points_array(name: str, values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return np.empty((0, 0, 3), dtype=float)
    if array.ndim == 2 and array.shape[1] == 3:
        array = array[:, None, :]
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("{} must have shape (n, k, 3)".format(name))
    return array


def _as_scalar_array(name: str, values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("{} must be a one-dimensional array".format(name))
    return array


def _as_nearest_scalar_array(name: str, values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return np.empty((0, 0), dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError("{} must have shape (n, k)".format(name))
    return array


def _as_index_array(name: str, values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=int)
    if array.ndim != 1:
        raise ValueError("{} must be a one-dimensional array".format(name))
    return array


def _empty_points_array() -> np.ndarray:
    return np.empty((0, 3), dtype=float)
