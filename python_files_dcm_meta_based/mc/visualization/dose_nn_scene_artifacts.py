"""Read and write compact dosimetric nearest-neighbour render scene artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
import hashlib
import json

import numpy as np

from .dose_nn_scene import DoseNNRenderScene, DoseNNSceneMetadata, build_dose_nn_render_scene

DOSE_NN_RENDER_SCENE_ARTIFACT_SCHEMA_VERSION = "dose_nn_render_scene_artifact_v1"
DOSE_NN_RENDER_SCENE_MANIFEST_FILENAME = "manifest.json"
DOSE_NN_RENDER_SCENE_ARRAYS_FILENAME = "scene_arrays.npz"

DOSE_NN_RENDER_SCENE_ARRAY_NAMES = (
    "lattice_points",
    "lattice_doses",
    "original_point_indices",
    "trial_numbers",
    "biopsy_points",
    "interpolated_biopsy_doses",
    "nearest_lattice_points",
    "nearest_lattice_doses",
    "nearest_distances",
)


@dataclass(frozen=True, slots=True)
class DoseNNArrayArtifactSpec:
    """Manifest entry for one array stored in a scene NPZ artifact."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    sha256: str


@dataclass(frozen=True, slots=True)
class DoseNNRenderSceneArtifactManifest:
    """Manifest for one compact selected dose NN render scene artifact."""

    schema_version: str
    scene_id: str
    arrays_filename: str
    metadata: dict[str, Any]
    arrays: tuple[DoseNNArrayArtifactSpec, ...]
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def array_specs_by_name(self) -> dict[str, DoseNNArrayArtifactSpec]:
        return {array_spec.name: array_spec for array_spec in self.arrays}


def write_dose_nn_render_scene_artifact(
    scene: DoseNNRenderScene,
    artifact_dir: Path | str,
    *,
    scene_id: str,
    overwrite: bool = False,
) -> DoseNNRenderSceneArtifactManifest:
    """Write one selected render scene as JSON metadata plus compressed arrays."""
    resolved_scene_id = str(scene_id).strip()
    if resolved_scene_id == "":
        raise ValueError("scene_id cannot be empty")

    resolved_artifact_dir = Path(artifact_dir)
    manifest_path = resolved_artifact_dir.joinpath(DOSE_NN_RENDER_SCENE_MANIFEST_FILENAME)
    arrays_path = resolved_artifact_dir.joinpath(DOSE_NN_RENDER_SCENE_ARRAYS_FILENAME)
    if not overwrite and (manifest_path.exists() or arrays_path.exists()):
        raise FileExistsError(
            "dose NN render scene artifact already exists; pass overwrite=True to replace it: {}".format(
                resolved_artifact_dir
            )
        )

    arrays = _scene_arrays(scene)
    array_specs = tuple(_array_spec(name, array) for name, array in arrays.items())
    metadata_dict = _metadata_to_manifest_dict(scene.metadata)
    summary_dict = _scene_summary_to_manifest_dict(scene)
    _assert_json_serializable(metadata_dict, "scene metadata")
    _assert_json_serializable(summary_dict, "scene summary")
    manifest = DoseNNRenderSceneArtifactManifest(
        schema_version=DOSE_NN_RENDER_SCENE_ARTIFACT_SCHEMA_VERSION,
        scene_id=resolved_scene_id,
        arrays_filename=DOSE_NN_RENDER_SCENE_ARRAYS_FILENAME,
        metadata=metadata_dict,
        arrays=array_specs,
        summary=summary_dict,
    )

    resolved_artifact_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **arrays)
    _write_json(manifest_path, _manifest_to_dict(manifest))
    return manifest


def read_dose_nn_render_scene_artifact(artifact_dir: Path | str) -> DoseNNRenderScene:
    """Read a selected render scene artifact and validate all array specs."""
    resolved_artifact_dir = Path(artifact_dir)
    manifest = read_dose_nn_render_scene_artifact_manifest(resolved_artifact_dir)
    if manifest.schema_version != DOSE_NN_RENDER_SCENE_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            "unsupported dose NN render scene artifact schema version: {}".format(manifest.schema_version)
        )

    arrays_path = resolved_artifact_dir.joinpath(manifest.arrays_filename)
    if not arrays_path.exists():
        raise FileNotFoundError("dose NN render scene arrays file is missing: {}".format(arrays_path))

    expected_specs = manifest.array_specs_by_name
    missing_specs = sorted(set(DOSE_NN_RENDER_SCENE_ARRAY_NAMES).difference(expected_specs))
    if missing_specs:
        raise ValueError("dose NN render scene manifest is missing array specs: {}".format(missing_specs))

    with np.load(arrays_path, allow_pickle=False) as loaded_arrays:
        missing_arrays = sorted(set(DOSE_NN_RENDER_SCENE_ARRAY_NAMES).difference(set(loaded_arrays.files)))
        if missing_arrays:
            raise ValueError("dose NN render scene arrays file is missing arrays: {}".format(missing_arrays))
        arrays = {
            array_name: _validated_loaded_array(array_name, loaded_arrays[array_name], expected_specs[array_name])
            for array_name in DOSE_NN_RENDER_SCENE_ARRAY_NAMES
        }

    return build_dose_nn_render_scene(
        metadata=_metadata_from_manifest_dict(manifest.metadata),
        lattice_points=arrays["lattice_points"],
        lattice_doses=arrays["lattice_doses"],
        original_point_indices=arrays["original_point_indices"],
        trial_numbers=arrays["trial_numbers"],
        biopsy_points=arrays["biopsy_points"],
        interpolated_biopsy_doses=arrays["interpolated_biopsy_doses"],
        nearest_lattice_points=arrays["nearest_lattice_points"],
        nearest_lattice_doses=arrays["nearest_lattice_doses"],
        nearest_distances=arrays["nearest_distances"],
    )


def read_dose_nn_render_scene_artifact_manifest(
    artifact_dir: Path | str,
) -> DoseNNRenderSceneArtifactManifest:
    """Read only the JSON manifest for a selected render scene artifact."""
    manifest_path = Path(artifact_dir).joinpath(DOSE_NN_RENDER_SCENE_MANIFEST_FILENAME)
    if not manifest_path.exists():
        raise FileNotFoundError("dose NN render scene manifest is missing: {}".format(manifest_path))
    with open(manifest_path, "r", encoding="utf-8") as manifest_file:
        manifest_dict = json.load(manifest_file)
    return _manifest_from_dict(manifest_dict)


def _scene_arrays(scene: DoseNNRenderScene) -> dict[str, np.ndarray]:
    return {
        "lattice_points": np.asarray(scene.lattice_points),
        "lattice_doses": np.asarray(scene.lattice_doses),
        "original_point_indices": np.asarray(scene.original_point_indices),
        "trial_numbers": np.asarray(scene.trial_numbers),
        "biopsy_points": np.asarray(scene.biopsy_points),
        "interpolated_biopsy_doses": np.asarray(scene.interpolated_biopsy_doses),
        "nearest_lattice_points": np.asarray(scene.nearest_lattice_points),
        "nearest_lattice_doses": np.asarray(scene.nearest_lattice_doses),
        "nearest_distances": np.asarray(scene.nearest_distances),
    }


def _array_spec(name: str, array: np.ndarray) -> DoseNNArrayArtifactSpec:
    return DoseNNArrayArtifactSpec(
        name=str(name),
        shape=tuple(int(dimension) for dimension in array.shape),
        dtype=str(array.dtype),
        sha256=_array_sha256(array),
    )


def _validated_loaded_array(
    array_name: str,
    array: np.ndarray,
    expected_spec: DoseNNArrayArtifactSpec,
) -> np.ndarray:
    if tuple(array.shape) != tuple(expected_spec.shape):
        raise ValueError(
            "dose NN render scene array '{}' has shape {}, expected {}".format(
                array_name,
                tuple(array.shape),
                tuple(expected_spec.shape),
            )
        )
    if str(array.dtype) != str(expected_spec.dtype):
        raise ValueError(
            "dose NN render scene array '{}' has dtype {}, expected {}".format(
                array_name,
                array.dtype,
                expected_spec.dtype,
            )
        )
    actual_sha256 = _array_sha256(array)
    if actual_sha256 != expected_spec.sha256:
        raise ValueError(
            "dose NN render scene array '{}' failed checksum validation".format(array_name)
        )
    return array


def _array_sha256(array: np.ndarray) -> str:
    contiguous_array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous_array.dtype).encode("utf-8"))
    digest.update(str(tuple(contiguous_array.shape)).encode("utf-8"))
    digest.update(contiguous_array.tobytes(order="C"))
    return digest.hexdigest()


def _metadata_to_manifest_dict(metadata: DoseNNSceneMetadata) -> dict[str, Any]:
    return {
        "patient_uid": str(metadata.patient_uid),
        "biopsy_roi": str(metadata.biopsy_roi),
        "biopsy_index": metadata.biopsy_index,
        "localization_kind": str(metadata.localization_kind),
        "result_column": str(metadata.result_column),
        "source_label": str(metadata.source_label),
        "extra": dict(metadata.extra),
    }


def _metadata_from_manifest_dict(metadata: Mapping[str, Any]) -> DoseNNSceneMetadata:
    required_keys = {
        "patient_uid",
        "biopsy_roi",
        "biopsy_index",
        "localization_kind",
        "result_column",
        "source_label",
        "extra",
    }
    missing_keys = sorted(required_keys.difference(set(metadata)))
    if missing_keys:
        raise ValueError("dose NN render scene metadata is missing keys: {}".format(missing_keys))
    biopsy_index = metadata["biopsy_index"]
    if biopsy_index is not None:
        biopsy_index = int(biopsy_index)
    return DoseNNSceneMetadata(
        patient_uid=str(metadata["patient_uid"]),
        biopsy_roi=str(metadata["biopsy_roi"]),
        biopsy_index=biopsy_index,
        localization_kind=str(metadata["localization_kind"]),
        result_column=str(metadata["result_column"]),
        source_label=str(metadata["source_label"]),
        extra=dict(metadata["extra"]),
    )


def _scene_summary_to_manifest_dict(scene: DoseNNRenderScene) -> dict[str, Any]:
    trial_numbers, trial_counts = np.unique(scene.trial_numbers, return_counts=True)
    return {
        "available_trials": [int(trial_number) for trial_number in trial_numbers],
        "trial_query_counts": {
            str(int(trial_number)): int(trial_count)
            for trial_number, trial_count in zip(trial_numbers, trial_counts)
        },
        "num_lattice_points": int(scene.lattice_points.shape[0]),
        "num_query_points": int(scene.biopsy_points.shape[0]),
        "num_nearest_neighbours": int(scene.num_nearest_neighbours),
        "lattice_bounds": _points_bounds(scene.lattice_points),
        "biopsy_bounds": _points_bounds(scene.biopsy_points),
        "nearest_lattice_bounds": _points_bounds(np.reshape(scene.nearest_lattice_points, (-1, 3))),
        "lattice_dose_range": _scalar_range(scene.lattice_doses),
        "interpolated_biopsy_dose_range": _scalar_range(scene.interpolated_biopsy_doses),
        "nearest_lattice_dose_range": _scalar_range(np.reshape(scene.nearest_lattice_doses, (-1,))),
        "nearest_distance_range": _scalar_range(np.reshape(scene.nearest_distances, (-1,))),
    }


def _points_bounds(points: Any) -> dict[str, list[float]] | None:
    point_array = np.asarray(points, dtype=float)
    if point_array.size == 0:
        return None
    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ValueError("point bounds require an array with shape (n, 3)")
    return {
        "min": [float(value) for value in np.min(point_array, axis=0)],
        "max": [float(value) for value in np.max(point_array, axis=0)],
    }


def _scalar_range(values: Any) -> dict[str, float] | None:
    value_array = np.asarray(values, dtype=float)
    if value_array.size == 0:
        return None
    return {
        "min": float(np.min(value_array)),
        "max": float(np.max(value_array)),
    }


def _manifest_to_dict(manifest: DoseNNRenderSceneArtifactManifest) -> dict[str, Any]:
    return {
        "schema_version": manifest.schema_version,
        "scene_id": manifest.scene_id,
        "arrays_filename": manifest.arrays_filename,
        "metadata": manifest.metadata,
        "arrays": [
            {
                "name": array_spec.name,
                "shape": list(array_spec.shape),
                "dtype": array_spec.dtype,
                "sha256": array_spec.sha256,
            }
            for array_spec in manifest.arrays
        ],
        "summary": manifest.summary,
    }


def _manifest_from_dict(manifest: Mapping[str, Any]) -> DoseNNRenderSceneArtifactManifest:
    required_keys = {"schema_version", "scene_id", "arrays_filename", "metadata", "arrays"}
    missing_keys = sorted(required_keys.difference(set(manifest)))
    if missing_keys:
        raise ValueError("dose NN render scene artifact manifest is missing keys: {}".format(missing_keys))

    arrays = tuple(
        DoseNNArrayArtifactSpec(
            name=str(array_spec["name"]),
            shape=tuple(int(dimension) for dimension in array_spec["shape"]),
            dtype=str(array_spec["dtype"]),
            sha256=str(array_spec["sha256"]),
        )
        for array_spec in manifest["arrays"]
    )
    array_names = [array_spec.name for array_spec in arrays]
    if len(array_names) != len(set(array_names)):
        raise ValueError("dose NN render scene artifact manifest contains duplicate array specs")
    return DoseNNRenderSceneArtifactManifest(
        schema_version=str(manifest["schema_version"]),
        scene_id=str(manifest["scene_id"]),
        arrays_filename=str(manifest["arrays_filename"]),
        metadata=dict(manifest["metadata"]),
        arrays=arrays,
        summary=dict(manifest.get("summary", {})),
    )


def _write_json(output_path: Path, payload: Mapping[str, Any]) -> None:
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def _assert_json_serializable(payload: Mapping[str, Any], payload_name: str) -> None:
    try:
        json.dumps(payload)
    except TypeError as exc:
        raise TypeError("{} must be JSON serializable".format(payload_name)) from exc
