"""Bridge retained dose context artifacts into renderer-neutral NN scenes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from output_artifacts.context_contracts import ArtifactRef
from output_artifacts.context_contracts import ArrayArtifactSpec

from ..simulation.per_patient.dose_context_artifacts import DOSE_CONTEXT_ARTIFACT_RELATIVE_ROOT
from ..simulation.per_patient.dose_context_artifacts import DOSE_CONTEXT_ARTIFACT_MODULE
from ..simulation.per_patient.dose_context_artifacts import DOSE_CONTEXT_RETENTION_LEVEL
from ..simulation.per_patient.dose_context_artifacts import DOSE_CONTEXT_STAGE_NAME
from ..simulation.per_patient.dose_context_artifacts import PatientDoseContextArtifactPlan
from ..simulation.per_patient.dose_context_artifacts import open_patient_dose_zarr_array_artifact
from ..simulation.per_patient.dose_context_artifacts import write_patient_dose_context_zarr_arrays
from .dose_nn_scene import DoseNNRenderScene
from .dose_nn_scene import DoseNNSceneMetadata
from .dose_nn_scene import build_dose_nn_render_scene
from .dose_nn_scene_artifacts import DoseNNRenderSceneArtifactManifest
from .dose_nn_scene_artifacts import write_dose_nn_render_scene_artifact


DOSE_NN_RENDER_CONTEXT_ARTIFACT_SCHEMA_VERSION = "dose_nn_render_context_artifact_v1"
DOSE_NN_RENDER_CONTEXT_ARTIFACT_MODULE = "mc.visualization.dose_nn_context_bridge"


def build_dose_nn_render_context_artifact_plan(
    scene: DoseNNRenderScene,
    *,
    relative_root: str = DOSE_CONTEXT_ARTIFACT_RELATIVE_ROOT,
) -> PatientDoseContextArtifactPlan:
    """Build a Zarr artifact plan for renderer-ready NN query arrays."""
    patient_uid = _non_empty_patient_uid(scene.metadata.patient_uid)
    localization_kind = _normal_token(scene.metadata.localization_kind or "dose")
    biopsy_token = _biopsy_token(scene.metadata.biopsy_index)
    artifact_ref = ArtifactRef(
        artifact_id="{}_{}_render_context".format(localization_kind, biopsy_token),
        title="{} {} render context".format(localization_kind.replace("_", " ").title(), biopsy_token.replace("_", " ")),
        artifact_family="dose_nn_render_context",
        relative_path="{}/{}/{}/render_context.zarr".format(
            _normalize_relative_root(relative_root),
            localization_kind,
            biopsy_token,
        ),
        storage_format="zarr",
        schema_version=DOSE_NN_RENDER_CONTEXT_ARTIFACT_SCHEMA_VERSION,
        patient_uid=patient_uid,
        stage_name=DOSE_CONTEXT_STAGE_NAME,
        retention_level=DOSE_CONTEXT_RETENTION_LEVEL,
        producer=DOSE_NN_RENDER_CONTEXT_ARTIFACT_MODULE,
        reader=DOSE_NN_RENDER_CONTEXT_ARTIFACT_MODULE,
        metadata={
            "context_kind": "dose_nn_render_context",
            "localization_kind": localization_kind,
            "biopsy_index": scene.metadata.biopsy_index,
            "biopsy_roi": str(scene.metadata.biopsy_roi),
            "result_column": str(scene.metadata.result_column),
            "source_label": str(scene.metadata.source_label),
            "source_context_module": DOSE_CONTEXT_ARTIFACT_MODULE,
        },
    )
    array_specs = (
        _array_spec(artifact_ref, "original_point_indices", scene.original_point_indices, ("n_query_rows",), ("query_row",), "", "query_index"),
        _array_spec(artifact_ref, "trial_numbers", scene.trial_numbers, ("n_query_rows",), ("query_row",), "", "trial_index"),
        _array_spec(artifact_ref, "biopsy_points", scene.biopsy_points, ("n_query_rows", "xyz"), ("query_row", "xyz"), "mm", "patient_physical_mm"),
        _array_spec(artifact_ref, "interpolated_biopsy_doses", scene.interpolated_biopsy_doses, ("n_query_rows",), ("query_row",), "Gy", "biopsy_sample_point_index"),
        _array_spec(artifact_ref, "nearest_lattice_points", scene.nearest_lattice_points, ("n_query_rows", "k_nearest", "xyz"), ("query_row", "nearest_neighbour", "xyz"), "mm", "patient_physical_mm"),
        _array_spec(artifact_ref, "nearest_lattice_doses", scene.nearest_lattice_doses, ("n_query_rows", "k_nearest"), ("query_row", "nearest_neighbour"), "Gy", "dose_lattice_index"),
        _array_spec(artifact_ref, "nearest_distances", scene.nearest_distances, ("n_query_rows", "k_nearest"), ("query_row", "nearest_neighbour"), "mm", "patient_physical_mm"),
    )
    return PatientDoseContextArtifactPlan(
        patient_uid=patient_uid,
        artifact_refs=(artifact_ref,),
        array_specs=array_specs,
        metadata={
            "context_kind": "dose_nn_render_context",
            "localization_kind": localization_kind,
            "biopsy_index": scene.metadata.biopsy_index,
        },
    )


def dose_nn_render_context_array_payload(scene: DoseNNRenderScene) -> dict[str, Any]:
    """Return renderer-ready NN arrays keyed by retained dataset name."""
    return {
        "original_point_indices": scene.original_point_indices,
        "trial_numbers": scene.trial_numbers,
        "biopsy_points": scene.biopsy_points,
        "interpolated_biopsy_doses": scene.interpolated_biopsy_doses,
        "nearest_lattice_points": scene.nearest_lattice_points,
        "nearest_lattice_doses": scene.nearest_lattice_doses,
        "nearest_distances": scene.nearest_distances,
    }


def write_dose_nn_render_context_zarr_artifact(
    scene: DoseNNRenderScene,
    output_root: Path | str,
    *,
    relative_root: str = DOSE_CONTEXT_ARTIFACT_RELATIVE_ROOT,
    overwrite: bool = False,
) -> tuple[PatientDoseContextArtifactPlan, dict[str, Path]]:
    """Write renderer-ready NN query arrays as a Zarr context artifact."""
    plan = build_dose_nn_render_context_artifact_plan(scene, relative_root=relative_root)
    written_paths = write_patient_dose_context_zarr_arrays(
        plan,
        dose_nn_render_context_array_payload(scene),
        output_root,
        overwrite=overwrite,
    )
    return plan, written_paths


def build_dose_nn_render_scene_from_context_artifacts(
    *,
    lattice_artifact_ref: ArtifactRef,
    render_context_artifact_ref: ArtifactRef,
    output_root: Path | str,
    metadata: DoseNNSceneMetadata | None = None,
) -> DoseNNRenderScene:
    """Rebuild a renderer-neutral scene from retained dose context artifacts."""
    lattice_reader = open_patient_dose_zarr_array_artifact(lattice_artifact_ref, output_root)
    render_reader = open_patient_dose_zarr_array_artifact(render_context_artifact_ref, output_root)
    return build_dose_nn_render_scene(
        metadata=metadata or _metadata_from_render_context_ref(render_context_artifact_ref),
        lattice_points=lattice_reader.read_array("physical_coordinates"),
        lattice_doses=lattice_reader.read_array("sampled_values"),
        original_point_indices=render_reader.read_array("original_point_indices"),
        trial_numbers=render_reader.read_array("trial_numbers"),
        biopsy_points=render_reader.read_array("biopsy_points"),
        interpolated_biopsy_doses=render_reader.read_array("interpolated_biopsy_doses"),
        nearest_lattice_points=render_reader.read_array("nearest_lattice_points"),
        nearest_lattice_doses=render_reader.read_array("nearest_lattice_doses"),
        nearest_distances=render_reader.read_array("nearest_distances"),
    )


def assert_dose_nn_render_context_artifacts_match_scene(
    scene: DoseNNRenderScene,
    *,
    lattice_artifact_ref: ArtifactRef,
    render_context_artifact_ref: ArtifactRef,
    output_root: Path | str,
) -> DoseNNRenderScene:
    """Rebuild a scene from retained context artifacts and compare it to a runtime scene."""
    rebuilt_scene = build_dose_nn_render_scene_from_context_artifacts(
        lattice_artifact_ref=lattice_artifact_ref,
        render_context_artifact_ref=render_context_artifact_ref,
        output_root=output_root,
        metadata=scene.metadata,
    )
    for array_name in (
        "lattice_points",
        "lattice_doses",
        "original_point_indices",
        "trial_numbers",
        "biopsy_points",
        "interpolated_biopsy_doses",
        "nearest_lattice_points",
        "nearest_lattice_doses",
        "nearest_distances",
    ):
        _assert_same_array(array_name, getattr(scene, array_name), getattr(rebuilt_scene, array_name))
    return rebuilt_scene


def materialize_dose_nn_saved_scene_artifact_from_context(
    *,
    lattice_artifact_ref: ArtifactRef,
    render_context_artifact_ref: ArtifactRef,
    output_root: Path | str,
    scene_artifact_dir: Path | str,
    scene_id: str,
    metadata: DoseNNSceneMetadata | None = None,
    overwrite: bool = False,
) -> DoseNNRenderSceneArtifactManifest:
    """Write a standard saved-scene artifact from retained Zarr context artifacts."""
    scene = build_dose_nn_render_scene_from_context_artifacts(
        lattice_artifact_ref=lattice_artifact_ref,
        render_context_artifact_ref=render_context_artifact_ref,
        output_root=output_root,
        metadata=metadata,
    )
    return write_dose_nn_render_scene_artifact(
        scene,
        scene_artifact_dir,
        scene_id=scene_id,
        overwrite=overwrite,
    )


def _array_spec(
    artifact_ref: ArtifactRef,
    dataset_name: str,
    array_like: Any,
    symbolic_shape: tuple[str, ...],
    dimension_names: tuple[str, ...],
    units: str,
    coordinate_frame: str,
) -> ArrayArtifactSpec:
    array = np.asarray(array_like)
    return ArrayArtifactSpec(
        artifact_ref=artifact_ref,
        dataset_name=dataset_name,
        symbolic_shape=symbolic_shape,
        shape=tuple(int(dimension) for dimension in array.shape),
        dtype=str(array.dtype),
        units=units,
        coordinate_frame=coordinate_frame,
        dimension_names=dimension_names,
        metadata={"chunking_strategy": "writer_selected"},
    )


def _metadata_from_render_context_ref(artifact_ref: ArtifactRef) -> DoseNNSceneMetadata:
    metadata = dict(artifact_ref.metadata)
    biopsy_index = metadata.get("biopsy_index")
    if biopsy_index is not None:
        biopsy_index = int(biopsy_index)
    return DoseNNSceneMetadata(
        patient_uid=artifact_ref.patient_uid,
        biopsy_roi=str(metadata.get("biopsy_roi", "")),
        biopsy_index=biopsy_index,
        localization_kind=str(metadata.get("localization_kind", "dose")),
        result_column=str(metadata.get("result_column", "Dose val (interpolated)")),
        source_label=str(metadata.get("source_label", "retained_context")),
        extra={"render_context_artifact_id": artifact_ref.artifact_id},
    )


def _assert_same_array(array_name: str, expected: Any, actual: Any) -> None:
    expected_array = np.asarray(expected)
    actual_array = np.asarray(actual)
    if expected_array.shape != actual_array.shape:
        raise ValueError(
            "dose NN context artifact array '{}' has shape {}, expected {}".format(
                array_name,
                actual_array.shape,
                expected_array.shape,
            )
        )
    if str(expected_array.dtype) != str(actual_array.dtype):
        raise ValueError(
            "dose NN context artifact array '{}' has dtype {}, expected {}".format(
                array_name,
                actual_array.dtype,
                expected_array.dtype,
            )
        )
    if not np.array_equal(expected_array, actual_array):
        raise ValueError("dose NN context artifact array '{}' does not match runtime scene".format(array_name))


def _non_empty_patient_uid(patient_uid: str) -> str:
    normalized = str(patient_uid).strip()
    if normalized == "":
        raise ValueError("Dose NN render context artifact requires scene.metadata.patient_uid")
    return normalized


def _biopsy_token(biopsy_index: int | None) -> str:
    if biopsy_index is None:
        return "biopsy_unknown"
    normalized = int(biopsy_index)
    if normalized < 0:
        raise ValueError("biopsy_index cannot be negative")
    return "biopsy_{:03d}".format(normalized)


def _normal_token(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized == "":
        raise ValueError("token value cannot be empty")
    return normalized


def _normalize_relative_root(relative_root: str) -> str:
    normalized = str(relative_root).strip().strip("/")
    if normalized == "":
        raise ValueError("relative_root cannot be empty")
    return normalized