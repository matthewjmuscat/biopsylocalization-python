"""Biopsy-specific straight-axis fit and projected centroid extent, in millimetres.

Contour-slice centroids are the observations; PCA supplies direction and their
projection extrema supply extent. This is not curved tissue length or measured
needle-tip localization. The generic pca.linear_fitter contract is unchanged.
"""

from dataclasses import dataclass
import math

import numpy as np
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class BiopsyFittedSegment:
    """One fitted segment in the input coordinate frame, including both ends.

    PCA's sign is retained; anatomical/needle orientation is not inferred here.
    The N + 1 ring centers also define the stored centroid-line sample points.
    """

    centroid_mean: np.ndarray
    axis_direction: np.ndarray
    projection_bounds_mm: tuple[float, float]
    endpoints: np.ndarray
    ring_centers: np.ndarray

    @property
    def length_mm(self) -> float:
        return self.projection_bounds_mm[1] - self.projection_bounds_mm[0]

    @property
    def interval_count(self) -> int:
        return len(self.ring_centers) - 1

    @property
    def spacing_mm(self) -> float:
        return self.length_mm / self.interval_count

    @property
    def travel_vector(self) -> np.ndarray:
        return (self.endpoints[1] - self.endpoints[0]) / self.interval_count


def fit_biopsy_segment(centroids, *, max_ring_spacing_mm: float = 0.1) -> BiopsyFittedSegment:
    """Fit finite (slice, xyz) centroids and include projected extrema as rings.

    Require at least two centroids and positive finite spacing. Extents at or
    below max(1e-8 mm, 64 coordinate-scale floating-point epsilons) fail clearly
    as numerically degenerate before transport/division/Delaunay construction.
    The floor is a numerical guard, not a clinical length threshold.
    """
    points = np.asarray(centroids, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("biopsy fit requires at least two finite XYZ slice centroids")
    if not math.isfinite(max_ring_spacing_mm) or max_ring_spacing_mm <= 0:
        raise ValueError("maximum biopsy ring spacing must be finite and positive")
    origin = points.mean(axis=0)
    centered = points - origin
    minimum_extent = max(1e-8, 64 * np.finfo(float).eps * max(1., float(np.abs(points).max())))
    if np.linalg.norm(centered, axis=1).max() <= minimum_extent:
        raise ValueError("biopsy centroid extent is zero or numerically near zero")
    # Same PCA estimator/settings as the generic fitter, without its radius-based endpoints.
    direction = PCA(n_components=1).fit(points).components_[0]
    direction = direction / np.linalg.norm(direction)
    projections = centered @ direction
    bounds = (float(projections.min()), float(projections.max()))
    length = bounds[1] - bounds[0]
    if not math.isfinite(length) or length <= minimum_extent:
        raise ValueError("projected biopsy extent is zero or numerically near zero")
    endpoints = origin + np.asarray(bounds)[:, None] * direction
    intervals = int(math.ceil(length / max_ring_spacing_mm))
    centers = np.linspace(endpoints[0], endpoints[1], intervals + 1)
    return BiopsyFittedSegment(origin, direction, bounds, endpoints, centers)
