"""Dose-owned artifact specs for retained per-patient scientific context."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from output_artifacts.context_contracts import ArtifactRef
from output_artifacts.context_contracts import ArrayArtifactSpec
from output_artifacts.context_contracts import PatientArtifactIndex
from output_artifacts.context_contracts import TableArtifactSpec

from .dose import MC_DOSE_LOCALIZATION_KIND_DOSE
from .dose import MC_DOSE_LOCALIZATION_KIND_GRADIENT


DOSE_CONTEXT_ARTIFACT_SCHEMA_VERSION = "patient_dose_context_artifacts_v1"
DOSE_CONTEXT_ARTIFACT_RELATIVE_ROOT = "context/dosimetry"
DOSE_CONTEXT_STAGE_NAME = "mc_dosimetry"
DOSE_CONTEXT_RETENTION_LEVEL = "context"
DOSE_CONTEXT_ARTIFACT_MODULE = "mc.simulation.per_patient.dose_context_artifacts"


@dataclass(frozen=True, slots=True)
class PatientDoseContextArtifactPlan:
    """Manifest-ready dose context artifacts and dataset specs for one patient slice."""

    patient_uid: str
    artifact_refs: tuple[ArtifactRef, ...]
    array_specs: tuple[ArrayArtifactSpec, ...] = ()
    table_specs: tuple[TableArtifactSpec, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        patient_uid = str(self.patient_uid).strip()
        if patient_uid == "":
            raise ValueError("patient_uid cannot be empty")
        artifact_refs = tuple(self.artifact_refs)
        array_specs = tuple(self.array_specs)
        table_specs = tuple(self.table_specs)
        if any(not isinstance(artifact_ref, ArtifactRef) for artifact_ref in artifact_refs):
            raise TypeError("artifact_refs must contain ArtifactRef objects")
        if any(not isinstance(array_spec, ArrayArtifactSpec) for array_spec in array_specs):
            raise TypeError("array_specs must contain ArrayArtifactSpec objects")
        if any(not isinstance(table_spec, TableArtifactSpec) for table_spec in table_specs):
            raise TypeError("table_specs must contain TableArtifactSpec objects")
        artifact_ids = [artifact_ref.artifact_id for artifact_ref in artifact_refs]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("PatientDoseContextArtifactPlan cannot contain duplicate artifact IDs")
        object.__setattr__(self, "patient_uid", patient_uid)
        object.__setattr__(self, "artifact_refs", artifact_refs)
        object.__setattr__(self, "array_specs", array_specs)
        object.__setattr__(self, "table_specs", table_specs)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def array_specs_by_dataset(self) -> dict[str, ArrayArtifactSpec]:
        """Return array specs keyed by dataset name."""
        return {array_spec.dataset_name: array_spec for array_spec in self.array_specs}

    @property
    def table_specs_by_name(self) -> dict[str, TableArtifactSpec]:
        """Return table specs keyed by table name."""
        return {table_spec.table_name: table_spec for table_spec in self.table_specs}

    def to_patient_artifact_index(self, *, run_id: str = "") -> PatientArtifactIndex:
        """Return the patient artifact index entries implied by this dose plan."""
        return PatientArtifactIndex(
            patient_uid=self.patient_uid,
            run_id=str(run_id),
            artifacts=self.artifact_refs,
            metadata=dict(self.metadata),
        )


def build_patient_dose_lattice_context_artifact_plan(
    lattice_context: Any,
    *,
    relative_root: str = DOSE_CONTEXT_ARTIFACT_RELATIVE_ROOT,
) -> PatientDoseContextArtifactPlan:
    """Build storage specs for one patient's retained dose lattice context."""
    patient_uid = str(lattice_context.patient_uid)
    localization_kind = _normalize_localization_kind(lattice_context.localization_kind)
    artifact_ref = _artifact_ref(
        patient_uid=patient_uid,
        artifact_id="{}_lattice_context".format(localization_kind),
        title="{} lattice context".format(localization_kind.replace("_", " ").title()),
        artifact_family="dose_lattice_context",
        relative_path="{}/{}/lattice.zarr".format(_normalize_relative_root(relative_root), localization_kind),
        storage_format="zarr",
        metadata={"localization_kind": localization_kind, "kdtree_persisted": False},
    )
    array_specs = (
        _array_spec(
            artifact_ref,
            "source_dose_and_gradient_array",
            lattice_context.source_dose_and_gradient_array,
            symbolic_shape=_generic_symbolic_shape(lattice_context.source_dose_and_gradient_array, "source_component"),
            dimension_names=_generic_dimension_names(lattice_context.source_dose_and_gradient_array, "source_component"),
            units="mixed",
            coordinate_frame="dose_grid_index_space",
        ),
        _array_spec(
            artifact_ref,
            "localization_map_array",
            lattice_context.localization_map_array,
            symbolic_shape=_generic_symbolic_shape(lattice_context.localization_map_array, "localization_component"),
            dimension_names=_generic_dimension_names(lattice_context.localization_map_array, "localization_component"),
            units="mixed",
            coordinate_frame="patient_physical_mm",
        ),
        _array_spec(
            artifact_ref,
            "localization_map_flattened",
            lattice_context.localization_map_flattened,
            symbolic_shape=_ranked_names(lattice_context.localization_map_flattened, ("n_lattice_points", "localization_component")),
            dimension_names=_ranked_names(lattice_context.localization_map_flattened, ("lattice_point", "localization_component")),
            units="mixed",
            coordinate_frame="patient_physical_mm",
        ),
        _array_spec(
            artifact_ref,
            "physical_coordinates",
            lattice_context.physical_coordinates,
            symbolic_shape=_ranked_names(lattice_context.physical_coordinates, ("n_lattice_points", "xyz")),
            dimension_names=_ranked_names(lattice_context.physical_coordinates, ("lattice_point", "xyz")),
            units="mm",
            coordinate_frame="patient_physical_mm",
        ),
        _array_spec(
            artifact_ref,
            "sampled_values",
            lattice_context.sampled_values,
            symbolic_shape=_ranked_names(lattice_context.sampled_values, ("n_lattice_points",)),
            dimension_names=_ranked_names(lattice_context.sampled_values, ("lattice_point",)),
            units=_sampled_value_units(localization_kind),
            coordinate_frame="patient_physical_mm",
        ),
    )
    return PatientDoseContextArtifactPlan(
        patient_uid=patient_uid,
        artifact_refs=(artifact_ref,),
        array_specs=array_specs,
        metadata={"localization_kind": localization_kind, "context_kind": "dose_lattice"},
    )


def build_patient_dose_biopsy_query_context_artifact_plan(
    biopsy_context: Any,
    *,
    relative_root: str = DOSE_CONTEXT_ARTIFACT_RELATIVE_ROOT,
) -> PatientDoseContextArtifactPlan:
    """Build storage specs for one biopsy's retained query-point context."""
    patient_uid = str(biopsy_context.patient_uid)
    biopsy_index = _normalize_biopsy_index(biopsy_context.biopsy_index)
    biopsy_token = _biopsy_token(biopsy_index)
    artifact_ref = _artifact_ref(
        patient_uid=patient_uid,
        artifact_id="{}_query_context".format(biopsy_token),
        title="Biopsy {} query context".format(biopsy_index),
        artifact_family="dose_biopsy_query_context",
        relative_path="{}/{}/query_points.zarr".format(_normalize_relative_root(relative_root), biopsy_token),
        storage_format="zarr",
        metadata={
            "biopsy_index": biopsy_index,
            "roi": str(getattr(biopsy_context, "roi", "")),
            "ref_number": str(getattr(biopsy_context, "ref_number", "")),
        },
    )
    array_specs = tuple(
        _array_spec(
            artifact_ref,
            dataset_name,
            getattr(biopsy_context, attribute_name),
            symbolic_shape=symbolic_shape,
            dimension_names=dimension_names,
            units="mm",
            coordinate_frame=coordinate_frame,
        )
        for dataset_name, attribute_name, symbolic_shape, dimension_names, coordinate_frame in (
            (
                "unshifted_sampled_points",
                "unshifted_sampled_points",
                _ranked_names(biopsy_context.unshifted_sampled_points, ("n_sample_points", "xyz")),
                _ranked_names(biopsy_context.unshifted_sampled_points, ("sample_point", "xyz")),
                "patient_physical_mm",
            ),
            (
                "sampled_points_bx_coord_sys",
                "sampled_points_bx_coord_sys",
                _ranked_names(biopsy_context.sampled_points_bx_coord_sys, ("n_sample_points", "xyz")),
                _ranked_names(biopsy_context.sampled_points_bx_coord_sys, ("sample_point", "xyz")),
                "biopsy_coordinate_system_mm",
            ),
            (
                "bx_only_shifted_points",
                "bx_only_shifted_points",
                _ranked_names(biopsy_context.bx_only_shifted_points, ("n_trials", "n_sample_points", "xyz")),
                _ranked_names(biopsy_context.bx_only_shifted_points, ("trial", "sample_point", "xyz")),
                "patient_physical_mm",
            ),
            (
                "bx_only_shifted_points_cutoff",
                "bx_only_shifted_points_cutoff",
                _ranked_names(biopsy_context.bx_only_shifted_points_cutoff, ("n_mc_trials", "n_sample_points", "xyz")),
                _ranked_names(biopsy_context.bx_only_shifted_points_cutoff, ("trial", "sample_point", "xyz")),
                "patient_physical_mm",
            ),
            (
                "nominal_and_shifted_points",
                "nominal_and_shifted_points",
                _ranked_names(biopsy_context.nominal_and_shifted_points, ("n_nominal_and_trials", "n_sample_points", "xyz")),
                _ranked_names(biopsy_context.nominal_and_shifted_points, ("trial", "sample_point", "xyz")),
                "patient_physical_mm",
            ),
            (
                "stacked_nominal_and_shifted_points",
                "stacked_nominal_and_shifted_points",
                _ranked_names(biopsy_context.stacked_nominal_and_shifted_points, ("n_stacked_query_points", "xyz")),
                _ranked_names(biopsy_context.stacked_nominal_and_shifted_points, ("stacked_query_point", "xyz")),
                "patient_physical_mm",
            ),
        )
    )
    return PatientDoseContextArtifactPlan(
        patient_uid=patient_uid,
        artifact_refs=(artifact_ref,),
        array_specs=array_specs,
        metadata={"biopsy_index": biopsy_index, "context_kind": "dose_biopsy_query"},
    )


def build_patient_dose_localization_context_artifact_plan(
    localization_outputs: Any,
    *,
    patient_uid: str,
    biopsy_index: int,
    relative_root: str = DOSE_CONTEXT_ARTIFACT_RELATIVE_ROOT,
) -> PatientDoseContextArtifactPlan:
    """Build storage specs for one biopsy's retained dose localization outputs."""
    normalized_patient_uid = str(patient_uid).strip()
    if normalized_patient_uid == "":
        raise ValueError("patient_uid cannot be empty")
    normalized_biopsy_index = _normalize_biopsy_index(biopsy_index)
    biopsy_token = _biopsy_token(normalized_biopsy_index)
    localization_kind = _normalize_localization_kind(localization_outputs.localization_kind)
    relative_root = _normalize_relative_root(relative_root)
    values_ref = _artifact_ref(
        patient_uid=normalized_patient_uid,
        artifact_id="{}_{}_localization_values".format(localization_kind, biopsy_token),
        title="{} biopsy {} localization values".format(localization_kind.replace("_", " ").title(), normalized_biopsy_index),
        artifact_family="dose_localization_values_context",
        relative_path="{}/{}/{}/localization_values.zarr".format(relative_root, localization_kind, biopsy_token),
        storage_format="zarr",
        metadata={"localization_kind": localization_kind, "biopsy_index": normalized_biopsy_index},
    )
    rows_ref = _artifact_ref(
        patient_uid=normalized_patient_uid,
        artifact_id="{}_{}_nearest_neighbour_rows".format(localization_kind, biopsy_token),
        title="{} biopsy {} nearest-neighbour rows".format(localization_kind.replace("_", " ").title(), normalized_biopsy_index),
        artifact_family="dose_nearest_neighbour_rows",
        relative_path="{}/{}/{}/nearest_neighbour_rows.parquet".format(relative_root, localization_kind, biopsy_token),
        storage_format="parquet",
        metadata={"localization_kind": localization_kind, "biopsy_index": normalized_biopsy_index},
    )
    values_spec = _array_spec(
        values_ref,
        "values_by_point_nominal_and_trials",
        localization_outputs.values_by_point_nominal_and_trials,
        symbolic_shape=_ranked_names(
            localization_outputs.values_by_point_nominal_and_trials,
            ("n_sample_points", "n_nominal_and_trials"),
        ),
        dimension_names=_ranked_names(
            localization_outputs.values_by_point_nominal_and_trials,
            ("sample_point", "trial"),
        ),
        units=_sampled_value_units(localization_kind),
        coordinate_frame="biopsy_sample_point_index",
    )
    table_spec = TableArtifactSpec(
        artifact_ref=rows_ref,
        table_name="nearest_neighbour_rows",
        columns=_table_columns(localization_outputs.nearest_neighbour_dataframe),
        row_count=_table_row_count(localization_outputs.nearest_neighbour_dataframe),
        metadata={"localization_kind": localization_kind, "biopsy_index": normalized_biopsy_index},
    )
    return PatientDoseContextArtifactPlan(
        patient_uid=normalized_patient_uid,
        artifact_refs=(values_ref, rows_ref),
        array_specs=(values_spec,),
        table_specs=(table_spec,),
        metadata={
            "localization_kind": localization_kind,
            "biopsy_index": normalized_biopsy_index,
            "context_kind": "dose_localization",
        },
    )


def _artifact_ref(
    *,
    patient_uid: str,
    artifact_id: str,
    title: str,
    artifact_family: str,
    relative_path: str,
    storage_format: str,
    metadata: Mapping[str, Any] | None = None,
) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=artifact_id,
        title=title,
        artifact_family=artifact_family,
        relative_path=relative_path,
        storage_format=storage_format,
        schema_version=DOSE_CONTEXT_ARTIFACT_SCHEMA_VERSION,
        patient_uid=patient_uid,
        stage_name=DOSE_CONTEXT_STAGE_NAME,
        retention_level=DOSE_CONTEXT_RETENTION_LEVEL,
        producer=DOSE_CONTEXT_ARTIFACT_MODULE,
        reader=DOSE_CONTEXT_ARTIFACT_MODULE,
        metadata=dict(metadata or {}),
    )


def _array_spec(
    artifact_ref: ArtifactRef,
    dataset_name: str,
    array_like: Any,
    *,
    symbolic_shape: tuple[str, ...],
    dimension_names: tuple[str, ...],
    units: str,
    coordinate_frame: str,
) -> ArrayArtifactSpec:
    return ArrayArtifactSpec(
        artifact_ref=artifact_ref,
        dataset_name=dataset_name,
        symbolic_shape=symbolic_shape,
        shape=_array_shape(array_like),
        dtype=_dtype_name(array_like),
        units=units,
        coordinate_frame=coordinate_frame,
        dimension_names=dimension_names,
        metadata={"chunking_strategy": "writer_selected"},
    )


def _normalize_localization_kind(localization_kind: str) -> str:
    normalized = str(localization_kind).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"dose", "dosimetry"}:
        return MC_DOSE_LOCALIZATION_KIND_DOSE
    if normalized in {"gradient", "dose_gradient", "dosegrad"}:
        return MC_DOSE_LOCALIZATION_KIND_GRADIENT
    raise ValueError("Unsupported dose localization kind: {!r}".format(localization_kind))


def _normalize_biopsy_index(biopsy_index: int) -> int:
    normalized = int(biopsy_index)
    if normalized < 0:
        raise ValueError("biopsy_index cannot be negative")
    return normalized


def _biopsy_token(biopsy_index: int) -> str:
    return "biopsy_{:03d}".format(_normalize_biopsy_index(biopsy_index))


def _normalize_relative_root(relative_root: str) -> str:
    normalized = str(relative_root).strip().strip("/")
    if normalized == "":
        raise ValueError("relative_root cannot be empty")
    return normalized


def _array_shape(array_like: Any) -> tuple[int, ...]:
    shape = getattr(array_like, "shape", None)
    if shape is None:
        try:
            return (len(array_like),)
        except TypeError as exc:
            raise ValueError("array_like must expose shape or length") from exc
    return tuple(int(dimension) for dimension in tuple(shape))


def _dtype_name(array_like: Any) -> str:
    dtype = getattr(array_like, "dtype", None)
    if dtype is None:
        return type(array_like).__name__
    return str(dtype)


def _ranked_names(array_like: Any, names: tuple[str, ...]) -> tuple[str, ...]:
    shape = _array_shape(array_like)
    if len(shape) == len(names):
        return names
    return tuple("axis_{}".format(axis_index) for axis_index in range(len(shape)))


def _generic_symbolic_shape(array_like: Any, final_dimension_name: str) -> tuple[str, ...]:
    shape = _array_shape(array_like)
    if len(shape) == 0:
        return ()
    names = ["axis_{}".format(axis_index) for axis_index in range(len(shape))]
    names[-1] = final_dimension_name
    return tuple(names)


def _generic_dimension_names(array_like: Any, final_dimension_name: str) -> tuple[str, ...]:
    shape = _array_shape(array_like)
    if len(shape) == 0:
        return ()
    names = ["axis_{}".format(axis_index) for axis_index in range(len(shape))]
    names[-1] = final_dimension_name
    return tuple(names)


def _sampled_value_units(localization_kind: str) -> str:
    normalized = _normalize_localization_kind(localization_kind)
    if normalized == MC_DOSE_LOCALIZATION_KIND_DOSE:
        return "Gy"
    return "Gy_per_mm"


def _table_columns(table_like: Any) -> tuple[str, ...]:
    columns = getattr(table_like, "columns", None)
    if columns is None:
        raise ValueError("nearest_neighbour_dataframe must expose columns")
    return tuple(str(column) for column in tuple(columns))


def _table_row_count(table_like: Any) -> int | None:
    try:
        return int(len(table_like))
    except TypeError:
        return None