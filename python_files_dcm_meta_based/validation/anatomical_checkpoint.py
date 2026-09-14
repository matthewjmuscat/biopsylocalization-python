"""Patient-scoped anatomical and biopsy-preprocessing validation artifacts.

This transitional post-preprocessing boundary reads explicitly named legacy
fields without invoking science, loading DICOM, or serializing runtime objects.
It owns NPZ/JSON evidence and comparison, not orchestration or parity approval.
Coordinates retain DICOM patient millimetres; other quantities retain the units
specified by each field. No alignment, sorting of points, or rescaling occurs.
The historical anatomical entrypoint names remain compatible. An explicit
``checkpoint_name`` selects the second bounded biopsy schema, which adds fields
through biopsy_checkpoint_fields while reusing integrity and comparison logic.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zipfile import BadZipFile

import numpy as np
import pandas as pd

from config.snapshots import (
    PipelineConfigSnapshot,
    build_pipeline_scientific_config_snapshot,
    canonical_json_value,
    canonical_sha256,
)
from legacy_data_keys import legacy_data_keys
from .preprocessing_boundary import preprocessing_boundary


SCHEMA_VERSION = "anatomical_checkpoint_v1"
_MANIFEST_NAME = "anatomical_checkpoint.json"
_ARRAYS_NAME = "anatomical_arrays.npz"
_FAMILY_ATTRIBUTES = ("oar_ref", "dil_ref", "rectum_ref_key", "urethra_ref_key")
_KEYS = legacy_data_keys


@dataclass(frozen=True)
class AnatomicalFieldSpec:
    """Bounded legacy field contract, including payload axes and comparison role.

    ``geometry`` distinguishes raw from reconstructed patient-space evidence.
    ``exact`` makes discrete numeric values ineligible for tolerance relaxation.
    A ``None`` axis extent is variable; an empty axes tuple denotes a scalar.
    """

    key: str
    kind: str = "array"
    axes: tuple[str, ...] = ()
    shape: tuple[int | None, ...] = ()
    units: str = "legacy native units; unconverted"
    geometry: str = ""
    exact: bool = False
    exact_columns: tuple[int, ...] = ()


_STRUCTURE_FIELDS = (
    AnatomicalFieldSpec(_KEYS.structure_record.roi_key, "label"),
    AnatomicalFieldSpec(_KEYS.structure_record.ref_number_key, "label"),
    AnatomicalFieldSpec(_KEYS.structure_record.struct_type_key, "label"),
    AnatomicalFieldSpec(_KEYS.structure_record.index_number_key, "label"),
    AnatomicalFieldSpec("Raw contour pts", axes=("point", "xyz"), shape=(None, 3), units="mm", geometry="raw"),
    AnatomicalFieldSpec("Raw contour pts zslice list", "slices", ("point", "xyz"), (None, 3), "mm", "raw"),
    AnatomicalFieldSpec(_KEYS.structure_geometry.equal_num_zslice_contour_points_key, "slices", ("point", "xyz"), (None, 3), "mm", "reconstructed"),
    AnatomicalFieldSpec("Structure volume", units="mm^3"),
    AnatomicalFieldSpec("Structure surface area", units="mm^2"),
    AnatomicalFieldSpec("Maximum pairwise distance", units="mm"),
    AnatomicalFieldSpec("Voxel size for structure volume calc", units="mm"),
    AnatomicalFieldSpec("Voxel size for structure dimension calc", units="mm"),
    AnatomicalFieldSpec("Structure global centroid", axes=("singleton", "xyz"), shape=(1, 3), units="mm"),
    AnatomicalFieldSpec("Structure centroid pts", axes=("slice", "xyz"), shape=(None, 3), units="mm"),
    AnatomicalFieldSpec("Structure features dataframe", "table"),
)

_DOSE_FIELDS = (
    AnatomicalFieldSpec("Dose units", "label"),
    AnatomicalFieldSpec("Dose type", "label"),
    AnatomicalFieldSpec("Dose pixel arr", axes=("slice", "row", "column"), shape=(None, None, None), units="stored dose pixels", exact=True),
    AnatomicalFieldSpec("Dose grid scaling", units="Dose units / stored pixel"),
    AnatomicalFieldSpec("Pixel spacing", axes=("row_column",), shape=(2,), units="mm"),
    AnatomicalFieldSpec("Grid frame offset vector", axes=("slice",), shape=(None,), units="mm"),
    AnatomicalFieldSpec("Image orientation patient", axes=("direction_cosine",), shape=(6,), units="dimensionless"),
    AnatomicalFieldSpec("Image position patient", axes=("xyz",), shape=(3,), units="mm"),
    AnatomicalFieldSpec("Dose and gradient phys space and pixel 3d arr", axes=("slice", "voxel", "slice_row_column_X_Y_Z_dose_gradXYZ_norm_normalizedXYZ"), shape=(None, None, 14), units="indices; mm; Dose units; Dose units/mm; dimensionless", exact_columns=(0, 1, 2)),
)
_MR_FIELDS = (
    AnatomicalFieldSpec("Units", "label"),
    AnatomicalFieldSpec("RWV Units", "label"),
    AnatomicalFieldSpec("Series instance UID", "label"),
    AnatomicalFieldSpec("Pixel arr (all slices)", axes=("row", "column", "slice"), shape=(None, None, None), units="retained current values in Units; no new scaling"),
    AnatomicalFieldSpec("RWVSlope (all slices)", axes=("slice",), shape=(None,), units="real-world mapping slope"),
    AnatomicalFieldSpec("RWVIntercept (all slices)", axes=("slice",), shape=(None,), units="real-world mapping intercept"),
    AnatomicalFieldSpec("Pixel spacing", axes=("row_column",), shape=(2,), units="mm"),
    AnatomicalFieldSpec("Slice thickness", units="mm"),
    AnatomicalFieldSpec("Image orientation patient", axes=("direction_cosine",), shape=(6,), units="dimensionless"),
    AnatomicalFieldSpec("Image position patient (all slices)", axes=("slice", "xyz"), shape=(None, 3), units="mm"),
    AnatomicalFieldSpec("MR ADC phys space Nx4 arr", axes=("voxel", "X_Y_Z_ADC"), shape=(None, 4), units="mm; Units"),
    AnatomicalFieldSpec("MR ADC phys space Nx4 arr (filtered, non-negative)", axes=("voxel", "X_Y_Z_ADC"), shape=(None, 4), units="mm; Units"),
)
_PATIENT_TABLES = (
    "Selected structures",
    "MR - ADC - summary statistics by structure dataframe",
    "Prostate only points MR ADC dataframe (temporary for pre-processing)",
    "Prostate only MR ADC validation",
)
_INTERPOLATION_MEMBERS = {
    "Inter-slice interpolation information": ("interpolated_pts_np_arr", "interpolated_pts_list", "zslice_vals_after_interpolation_list"),
    "Intra-slice interpolation information": ("interpolated_pts_np_arr", "interpolated_pts_with_end_caps_np_arr"),
}


def _mapping(value: Any) -> Mapping:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("expected a legacy mapping")
    return value


def _points_spec(key: str, geometry: str = "") -> AnatomicalFieldSpec:
    return AnatomicalFieldSpec(key, axes=("point", "xyz"), shape=(None, 3), units="mm", geometry=geometry)


def _member(value: Any, name: str) -> Any:
    if value is None:
        return None
    if not hasattr(value, name):
        raise TypeError("retained geometry object lacks numeric member: " + name)
    return getattr(value, name)


def _dtype(dtype: Any) -> dict[str, Any]:
    if isinstance(dtype, pd.CategoricalDtype):
        return {"type": "category", "ordered": dtype.ordered, "categories": _axis(dtype.categories)}
    if isinstance(dtype, np.dtype):
        return {"type": "numpy", "dtype": dtype.str}
    if isinstance(dtype, pd.StringDtype):
        return {"type": "string", "storage": dtype.storage, "dtype": str(dtype)}
    if hasattr(dtype, "numpy_dtype") and pd.api.types.is_numeric_dtype(dtype):
        return {"type": "nullable", "dtype": str(dtype), "numpy_dtype": dtype.numpy_dtype.str}
    raise TypeError("unsupported table dtype: " + type(dtype).__name__)


def _axis(index: pd.Index) -> dict[str, Any]:
    result = {"type": type(index).__name__, "names": [_label(name) for name in index.names]}
    if isinstance(index, pd.MultiIndex):
        result.update(levels=[_axis(level) for level in index.levels], codes=[code.tolist() for code in index.codes], sortorder=index.sortorder)
    else:
        result.update(dtype=_dtype(index.dtype), labels=[_label(value) for value in index])
        if isinstance(index, pd.RangeIndex):
            result.update(start=index.start, stop=index.stop, step=index.step)
    return result


def _contract(spec: AnatomicalFieldSpec) -> dict[str, Any]:
    return canonical_json_value(asdict(spec))


def _label(value: Any) -> dict[str, Any]:
    if value is pd.NA:
        return {"type": "pandas.NA"}
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return {"type": "none"}
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [_label(item) for item in value]}
    if isinstance(value, bool):
        return {"type": "bool", "value": bool(value)}
    if isinstance(value, int):
        return {"type": "int", "value": int(value)}
    if isinstance(value, float):
        if not math.isfinite(value):
            return {"type": "float", "value": "nan" if math.isnan(value) else ("+inf" if value > 0 else "-inf")}
        return {"type": "float", "value": float(value)}
    if isinstance(value, str):
        return {"type": "str", "value": str(value)}
    raise TypeError("unsupported label type: " + type(value).__name__)


class _Capture:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []
        self.arrays: dict[str, np.ndarray] = {}

    def field(self, path: list[str], spec: AnatomicalFieldSpec, value: Any) -> None:
        item: dict[str, Any] = {
            "path": path, "contract": _contract(spec),
            "state": "absent" if value is None else "present",
        }
        self.items.append(item)
        if value is None:
            return
        if spec.kind == "label":
            item["value"] = _label(value)
        elif spec.kind == "table":
            self.table(path, item, value)
        elif spec.kind == "slices":
            if not isinstance(value, (list, tuple)):
                raise TypeError("contour slices must be a list or tuple")
            item["length"] = len(value)
            for index, points in enumerate(value):
                child = AnatomicalFieldSpec(spec.key, "array", spec.axes, spec.shape, spec.units, spec.geometry, spec.exact)
                if points is None:
                    raise ValueError("contour slices cannot contain missing arrays")
                self.field([*path, str(index)], child, points)
        elif spec.kind == "array":
            if hasattr(value, "__cuda_array_interface__"):
                raise TypeError("checkpoint requires host arrays")
            array = np.array(value, copy=True)
            _validate_array(array, item["contract"])
            key = "array_" + str(len(self.arrays))
            self.arrays[key] = array
            item.update(payload=key, dtype=array.dtype.str, shape=list(array.shape))
        else:
            raise ValueError("unsupported field kind: " + spec.kind)

    def table(self, path: list[str], item: dict[str, Any], table: Any) -> None:
        if not isinstance(table, pd.DataFrame):
            raise TypeError("preprocessing table must be a pandas DataFrame")
        item["table_schema"] = {
            "row_grain": "legacy dataframe row, original order retained",
            "shape": list(table.shape), "index": _axis(table.index),
            "columns": _axis(table.columns), "dtypes": [_dtype(dtype) for dtype in table.dtypes],
        }
        for column_index in range(table.shape[1]):
            series = table.iloc[:, column_index]
            column_path = [*path, "columns", str(column_index)]
            if isinstance(series.dtype, pd.CategoricalDtype):
                self.field(column_path, AnatomicalFieldSpec("category codes", axes=("row",), shape=(len(table),), exact=True), series.cat.codes.to_numpy())
            elif pd.api.types.is_numeric_dtype(series.dtype):
                if isinstance(series.dtype, np.dtype):
                    values = series.to_numpy()
                else:
                    values = series.to_numpy(dtype=series.dtype.numpy_dtype, na_value=0)
                    self.field([*column_path, "missing"], AnatomicalFieldSpec("nullable mask", axes=("row",), shape=(len(table),), exact=True), series.isna().to_numpy())
                self.field(column_path, AnatomicalFieldSpec("table numeric values", axes=("row",), shape=(len(table),)), values)
            else:
                for row_index, value in enumerate(series):
                    cell_path = [*column_path, "rows", str(row_index)]
                    if isinstance(value, (tuple, list, int, float, np.number)) and not isinstance(value, (bool, np.bool_)):
                        array = np.asarray(value)
                        if array.dtype.kind in "iuf":
                            cell_kind = "tuple" if isinstance(value, tuple) else "list" if isinstance(value, list) else "scalar"
                            self.field(cell_path, AnatomicalFieldSpec("table numeric " + cell_kind, axes=tuple("component" for _ in array.shape), shape=array.shape), array)
                            continue
                    self.field(cell_path, AnatomicalFieldSpec("table discrete cell", "label"), value)


def _capture_structure(capture: _Capture, path: list[str], record: Mapping) -> None:
    for spec in _STRUCTURE_FIELDS:
        capture.field([*path, spec.key], spec, record.get(spec.key))
    dimensions = _mapping(record.get("Structure dimension at centroid dict"))
    for axis in ("X", "Y", "Z"):
        key = axis + " dimension length at centroid"
        capture.field([*path, "Structure dimension at centroid dict", key], AnatomicalFieldSpec(key, units="mm"), dimensions.get(key))
    for key, attributes in _INTERPOLATION_MEMBERS.items():
        interpolation = record.get(key)
        for attribute in attributes:
            value = _member(interpolation, attribute)
            if attribute == "zslice_vals_after_interpolation_list":
                spec = AnatomicalFieldSpec(attribute, axes=("slice",), shape=(None,), units="mm")
            elif attribute == "interpolated_pts_list":
                spec = AnatomicalFieldSpec(attribute, "slices", ("point", "xyz"), (None, 3), "mm", "reconstructed")
            else:
                spec = _points_spec(attribute, "reconstructed")
            capture.field([*path, key, attribute], spec, value)
    clouds = _mapping(record.get("Interpolated structure point cloud dict"))
    for name in ("Interslice", "Full", "Full with end caps"):
        capture.field([*path, "Interpolated structure point cloud dict", name], _points_spec(name, "reconstructed"), _member(clouds.get(name), "points"))
    capture.field([*path, "Point cloud raw"], _points_spec("Point cloud raw", "raw"), _member(record.get("Point cloud raw"), "points"))
    mesh = record.get("Structure OPEN3D triangle mesh object")
    capture.field([*path, "mesh", "vertices"], _points_spec("vertices", "reconstructed"), _member(mesh, "vertices"))
    capture.field([*path, "mesh", "triangles"], AnatomicalFieldSpec("triangles", axes=("triangle", "vertex_index"), shape=(None, 3), units="vertex indices", exact=True), _member(mesh, "triangles"))


def _capture_modalities(capture: _Capture, patient: Mapping, refs: Any) -> None:
    for role, key, fields in (("dose", refs.dose_ref, _DOSE_FIELDS), ("mr_adc", refs.mr_adc_ref, _MR_FIELDS)):
        value = patient.get(key)
        capture.field([role, "present"], AnatomicalFieldSpec(key, "label"), value is not None)
        store = _mapping(value)
        if role == "mr_adc" and store and "Pixel arr (all slices)" not in store:
            raise ValueError("MR series mapping is not a completed selected/scaled MR grid")
        for spec in fields:
            capture.field([role, spec.key], spec, store.get(spec.key))
        for cloud_key in (("Dose grid point cloud", "Dose grid point cloud thresholded") if role == "dose" else ("MR ADC grid point cloud", "MR ADC grid point cloud thresholded")):
            capture.field([role, cloud_key], _points_spec(cloud_key), _member(store.get(cloud_key), "points"))


def _capture_tables(capture: _Capture, patient: Mapping, refs: Any, *, include_biopsies: bool = False) -> None:
    all_ref = _mapping(patient.get(refs.all_ref_key))
    tables = _mapping(all_ref.get(_KEYS.patient_all_reference.preprocessing_output_dataframes_key))
    for name in _PATIENT_TABLES:
        capture.field(["patient_tables", name], AnatomicalFieldSpec(name, "table"), tables.get(name))
    if any(not isinstance(name, str) for name in tables):
        raise TypeError("patient preprocessing table names must be strings")
    included = set(_PATIENT_TABLES) | ({"Simulated biopsy preparation dataframe"} if include_biopsies else set())
    excluded = tuple(sorted(set(tables) - included))
    capture.field(["table_exclusions"], AnatomicalFieldSpec("unlisted patient table names", "label"), excluded)


def _modality_coverage(items: list[dict[str, Any]], arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    by_path = {tuple(item["path"]): item for item in items}
    coverage = {}
    for role, specs in (("dose", _DOSE_FIELDS), ("mr_adc", _MR_FIELDS)):
        marker = by_path.get((role, "present"), {})
        present = marker.get("value") == _label(True)
        missing = []
        if present:
            for spec in specs:
                if spec.key.startswith("MR ADC phys space") or spec.key in ("RWV Units", "Series instance UID"):
                    continue
                item = by_path.get((role, spec.key), {})
                if item.get("state") != "present" or ("payload" in item and arrays[item["payload"]].size == 0):
                    missing.append(spec.key)
        coverage[role] = {"present": present, "missing_required": missing, "complete": not missing}
    return coverage


def _validate_array(array: np.ndarray, contract: Mapping[str, Any]) -> None:
    if array.dtype.kind not in "biuf":
        raise TypeError("only real numeric and boolean arrays are supported")
    expected_shape = contract["shape"]
    if array.ndim != len(expected_shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(array.shape, expected_shape)
    ):
        raise ValueError("array dimensions do not match field contract: " + contract["key"])
    if contract["key"] == "triangles" and array.dtype.kind not in "iu":
        raise TypeError("mesh triangle indices must have integer dtype")


def _coverage(items: list[dict[str, Any]], arrays: Mapping[str, np.ndarray], *, include_biopsies: bool = False) -> dict[str, Any]:
    structures: dict[tuple[str, ...], dict[str, Any]] = {}
    invalid_geometry = []
    for item in items:
        path = item["path"]
        if path[0] != "structures":
            continue
        structure = structures.setdefault(tuple(path[:3]), {"raw": 0, "reconstructed": 0, "identity": []})
        if len(path) == 4 and path[-1] in ("ROI", "Ref #") and item["state"] == "present" and item.get("value", {}).get("value") not in (None, ""):
            structure["identity"].append(path[-1])
        role = item["contract"]["geometry"]
        if role and item.get("payload") in arrays:
            points = arrays[item["payload"]]
            finite_points = points[np.isfinite(points).all(axis=1)]
            if len(finite_points) >= 3 and np.any(finite_points != finite_points[0]):
                structure[role] += 1
            else:
                invalid_geometry.append(path)
        elif role and item["state"] == "present" and item.get("length") == 0:
            invalid_geometry.append(path)
    geometry = [
        {"path": list(path), **counts,
         "complete": counts["raw"] > 0 and counts["reconstructed"] > 0 and len(counts["identity"]) == 2}
        for path, counts in structures.items()
    ]
    result = {
        "geometry": geometry,
        "geometry_complete": bool(geometry) and not invalid_geometry and all(item["complete"] for item in geometry),
        "invalid_geometry": invalid_geometry,
        "modalities": _modality_coverage(items, arrays),
        "numeric_arrays": len(arrays),
        "numeric_values": sum(int(array.size) for array in arrays.values()),
        "present": [item["path"] for item in items if item["state"] == "present"],
        "absent": [item["path"] for item in items if item["state"] == "absent"],
        "excluded_patient_tables": next((item["value"] for item in items if item["path"] == ["table_exclusions"]), _label(())),
        "exclusions": ["biopsies", "runtime resources", "timings", "DICOM bytes", "unlisted legacy fields", "dose gradient arrow render objects", "cloud colors/normals", "discarded MR lattices (not recomputed)", "unlisted patient tables", "table attrs", "source/environment/input file content verification"],
    }
    if include_biopsies:
        from .biopsy_checkpoint_fields import biopsy_coverage

        result["biopsies"] = biopsy_coverage(items, arrays)
        result["exclusions"].remove("biopsies")
        result["exclusions"].extend(["optimizer/realization/classification/MC/guidance products", "uncertainty spreadsheet attachment"])
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def _validate_inventory(manifest: Mapping[str, Any]) -> None:
    families = manifest["families"]
    refs = manifest["reference_keys"]
    include_biopsies = manifest["schema_version"] == "biopsy_preprocessing_checkpoint_v1"
    reference_names = {"dose_ref", "mr_adc_ref", "all_ref_key"} | ({"bx_ref"} if include_biopsies else set())
    if set(families) != set(_FAMILY_ATTRIBUTES) or set(refs) != reference_names:
        raise ValueError("reference key schema mismatch")
    if not all(isinstance(value, str) and value for value in (*families.values(), *refs.values())):
        raise ValueError("reference keys must be nonempty strings")
    if len(set(families.values())) != len(families):
        raise ValueError("duplicate structure families")
    config_refs = manifest["scientific_config"]["config"]["legacy_refs"]
    if any(config_refs.get(key) != value for key, value in {**families, **refs}.items()):
        raise ValueError("reference keys differ from scientific configuration")
    if not isinstance(manifest["patient_uid"], str) or not manifest["patient_uid"]:
        raise ValueError("invalid patient identity")
    items = {tuple(item["path"]): item for item in manifest["items"]}
    expected = _Capture()
    for family in families.values():
        path = ("families", family)
        item = items[path]
        count = None if item["state"] == "absent" else item["value"]["value"]
        if count is not None and (type(count) is not int or count < 0 or count > len(items)):
            raise ValueError("invalid structure count")
        expected.field(list(path), AnatomicalFieldSpec(family, "label"), count)
        for index in range(count or 0):
            _capture_structure(expected, ["structures", family, str(index)], {})
    _capture_modalities(expected, {}, SimpleNamespace(**refs))
    _capture_tables(expected, {}, SimpleNamespace(**refs))
    if include_biopsies:
        from .biopsy_checkpoint_fields import expected_biopsy_inventory

        expected_biopsy_inventory(expected, items)
    contracts = {tuple(item["path"]): item["contract"] for item in expected.items}
    for path, contract in list(contracts.items()):
        item = items[path]
        if item["state"] != "present":
            continue
        if contract["kind"] == "slices":
            length = item["length"]
            if type(length) is not int or length < 0 or length > len(items):
                raise ValueError("invalid contour slice count")
            for index in range(length):
                contracts[(*path, str(index))] = {**contract, "kind": "array"}
        elif contract["kind"] == "table":
            _table_inventory(path, item, items, contracts)
    if set(contracts) != set(items):
        raise ValueError("item inventory differs from bounded checkpoint schema")
    for path, contract in contracts.items():
        item = items[path]
        if item["contract"] != contract:
            raise ValueError("item contract differs from bounded checkpoint schema")
        fields = {"path", "contract", "state"}
        if item["state"] == "present":
            fields.update({"array": ("payload", "dtype", "shape"), "label": ("value",), "table": ("table_schema",), "slices": ("length",)}[contract["kind"]])
        if set(item) != fields:
            raise ValueError("unexpected or missing item descriptor fields")
    for role in ("dose", "mr_adc"):
        if items[(role, "present")].get("value") not in (_label(True), _label(False)):
            raise ValueError("invalid modality presence marker")
        if items[(role, "present")]["value"] == _label(False) and any(item["state"] == "present" for path, item in items.items() if path[0] == role and path[1] != "present"):
            raise ValueError("absent modality contains payloads")


def _table_inventory(path: tuple[str, ...], item: Mapping, items: Mapping, contracts: dict) -> None:
    schema = item["table_schema"]
    rows, columns = schema["shape"]
    if type(rows) is not int or type(columns) is not int or min(rows, columns) < 0 or columns != len(schema["dtypes"]):
        raise ValueError("invalid table dimensions/dtypes")
    for column_index, dtype in enumerate(schema["dtypes"]):
        column_path = (*path, "columns", str(column_index))
        dtype_kind = dtype["type"]
        if dtype_kind == "category":
            spec = AnatomicalFieldSpec("category codes", axes=("row",), shape=(rows,), exact=True)
            contracts[column_path] = _contract(spec)
        elif dtype_kind == "nullable" or (dtype_kind == "numpy" and np.dtype(dtype["dtype"]).kind in "biuf"):
            contracts[column_path] = _contract(AnatomicalFieldSpec("table numeric values", axes=("row",), shape=(rows,)))
            if dtype_kind == "nullable":
                contracts[(*column_path, "missing")] = _contract(AnatomicalFieldSpec("nullable mask", axes=("row",), shape=(rows,), exact=True))
        elif dtype_kind == "string" or (dtype_kind == "numpy" and np.dtype(dtype["dtype"]).kind == "O"):
            for row_index in range(rows):
                cell_path = (*column_path, "rows", str(row_index))
                cell = items[cell_path]
                if cell["contract"]["kind"] == "label":
                    spec = AnatomicalFieldSpec("table discrete cell", "label")
                else:
                    key = cell["contract"]["key"]
                    if key not in ("table numeric scalar", "table numeric tuple", "table numeric list"):
                        raise ValueError("unsupported table cell contract")
                    shape = tuple(cell["shape"])
                    spec = AnatomicalFieldSpec(key, axes=tuple("component" for _ in shape), shape=shape)
                contracts[cell_path] = _contract(spec)
        else:
            raise ValueError("unsupported table dtype schema")


def write_anatomical_checkpoint(
    *, runtime_state: Any, pipeline_config: Any, output_dir: Path,
    metadata: Mapping | None = None,
    checkpoint_name: str = "anatomical_qa",
) -> Path:
    """Capture one completed patient into a fresh directory; return its manifest.

    ``runtime_state.patient_case.patient_uid`` selects a patient in the legacy
    master reference dictionary. ``pipeline_config`` is a pure-data scientific
    config accepted by the existing snapshot API, with ``legacy_refs`` fields.
    No runtime state is mutated. Unsupported values or insufficient geometry
    raise TypeError/ValueError before output creation. Existing destinations
    raise FileExistsError, even when empty. I/O failures may leave a partial new
    destination, which must not be reused. Metadata is caller-supplied provenance,
    not a claim that source/environment/input fingerprints were independently
    established by this writer. Missing optional fields and explicit None share
    an absent marker. Every retained structure needs ROI/ref identity plus usable
    raw and reconstructed XYZ evidence. Present modalities require their grid
    arrays and physical metadata; missing modalities are allowed. The embedded
    field contracts define axes/units, and coverage lists every absent field.
    ``biopsy_preprocessing_shadow`` additionally requires completed biopsy
    reconstruction/planning products and a patient preparation table. Planned
    coordinates retain their canonical local frame, not the DICOM patient frame.
    """
    boundary = preprocessing_boundary(checkpoint_name)
    include_biopsies = checkpoint_name == "biopsy_preprocessing_shadow"
    snapshot = build_pipeline_scientific_config_snapshot(pipeline_config)
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    stored_metadata = canonical_json_value(dict(metadata or {}))
    patient_uid = runtime_state.patient_case.patient_uid
    if not isinstance(patient_uid, str) or not patient_uid:
        raise ValueError("patient_uid must be a nonempty string")
    patient = runtime_state.master_structure_reference_dict[patient_uid]
    refs = pipeline_config.legacy_refs
    capture = _Capture()
    families = {name: getattr(refs, name) for name in _FAMILY_ATTRIBUTES}
    if len(set(families.values())) != len(families):
        raise ValueError("structure reference families must be distinct")
    for family in families.values():
        records = patient.get(family)
        capture.field(["families", family], AnatomicalFieldSpec(family, "label"), None if records is None else len(records))
        if records is None:
            continue
        if not isinstance(records, (list, tuple)):
            raise TypeError("structure family must contain a list of records")
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise TypeError("structure record must be a mapping")
            _capture_structure(capture, ["structures", family, str(index)], record)
    _capture_modalities(capture, patient, refs)
    _capture_tables(capture, patient, refs, include_biopsies=include_biopsies)
    if include_biopsies:
        from .biopsy_checkpoint_fields import capture_biopsies

        capture_biopsies(capture, patient, refs)
    coverage = _coverage(capture.items, capture.arrays, include_biopsies=include_biopsies)
    if include_biopsies and not coverage["biopsies"]["complete"]:
        raise ValueError("incomplete biopsy preprocessing evidence: " + str(coverage["biopsies"]))
    if not coverage["geometry_complete"]:
        raise ValueError("checkpoint requires ROI identity and nonempty raw and reconstructed geometry for every structure")
    if not all(modality["complete"] for modality in coverage["modalities"].values()):
        raise ValueError("present modality has incomplete grid evidence")
    manifest = {
        "schema_version": boundary.schema_version,
        "patient_uid": patient_uid,
        "families": families,
        "reference_keys": {name: getattr(refs, name) for name in
                           ("dose_ref", "mr_adc_ref", "all_ref_key", *(["bx_ref"] if include_biopsies else []))},
        "scientific_config": snapshot.to_dict(),
        "metadata": stored_metadata,
        "metadata_sha256": canonical_sha256(stored_metadata),
        "items": capture.items,
        "coverage": coverage,
        "arrays_file": boundary.arrays_name,
    }
    _validate_inventory(manifest)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    arrays_path = output_dir / boundary.arrays_name
    with arrays_path.open("xb") as stream:
        np.savez_compressed(stream, **capture.arrays)
    manifest["arrays_sha256"] = _sha256_file(arrays_path)
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_path = output_dir / boundary.manifest_name
    _write_json(manifest_path, manifest)
    return manifest_path


def _load_checkpoint(path: Path, *, checkpoint_name: str = "anatomical_qa") -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    boundary = preprocessing_boundary(checkpoint_name)
    include_biopsies = checkpoint_name == "biopsy_preprocessing_shadow"
    with Path(path).open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != boundary.schema_version:
        raise ValueError("unsupported checkpoint schema")
    unsigned = dict(manifest)
    digest = unsigned.pop("manifest_sha256", None)
    if canonical_sha256(unsigned) != digest:
        raise ValueError("manifest digest mismatch")
    if canonical_sha256(manifest["metadata"]) != manifest["metadata_sha256"]:
        raise ValueError("metadata digest mismatch")
    if not isinstance(manifest["metadata"], dict):
        raise ValueError("metadata must be a JSON object")
    PipelineConfigSnapshot(**manifest["scientific_config"])
    if manifest["arrays_file"] != boundary.arrays_name:
        raise ValueError("arrays_file must be the checkpoint-local NPZ filename")
    arrays_path = Path(path).parent / boundary.arrays_name
    if arrays_path.is_symlink():
        raise ValueError("NPZ symlinks are not supported")
    if _sha256_file(arrays_path) != manifest["arrays_sha256"]:
        raise ValueError("NPZ digest mismatch")
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    seen_paths: set[tuple[str, ...]] = set()
    payloads: list[str] = []
    for item in manifest["items"]:
        if not isinstance(item, dict) or not isinstance(item.get("path"), list) or not all(isinstance(part, str) for part in item["path"]):
            raise ValueError("invalid item path/schema")
        item_path = tuple(item["path"])
        if not item_path or item_path in seen_paths:
            raise ValueError("empty or duplicate item path")
        seen_paths.add(item_path)
        if item["state"] not in ("present", "absent"):
            raise ValueError("invalid item presence state")
        if item.get("payload") is not None:
            key = item["payload"]
            array = arrays[key]
            _validate_array(array, item["contract"])
            if item["state"] != "present" or array.dtype.str != item["dtype"] or list(array.shape) != item["shape"]:
                raise ValueError("NPZ array does not match manifest dtype/shape/state")
            payloads.append(key)
        elif item["state"] == "present" and item["contract"]["kind"] == "array":
            raise ValueError("present numeric field is missing its payload")
    if len(set(payloads)) != len(payloads) or set(payloads) != set(arrays):
        raise ValueError("NPZ inventory does not match manifest")
    _validate_inventory(manifest)
    coverage = _coverage(manifest["items"], arrays, include_biopsies=include_biopsies)
    if include_biopsies and not coverage["biopsies"]["complete"]:
        raise ValueError("incomplete biopsy preprocessing evidence")
    if coverage != manifest["coverage"] or not coverage["geometry_complete"]:
        raise ValueError("incomplete or inconsistent geometry coverage")
    if not all(modality["complete"] for modality in coverage["modalities"].values()):
        raise ValueError("incomplete modality coverage")
    return manifest, arrays


def validate_anatomical_checkpoint_identity(
    path: Path, *, patient_uid: str, scientific_config_sha256: str,
    expected_metadata: Mapping[str, Any],
    checkpoint_name: str = "anatomical_qa",
) -> None:
    """Verify checkpoint integrity and bind retained evidence to a requesting job.

    Loads the numeric archive with the same integrity checks as comparison. The
    caller supplies authoritative metadata fields; extra capture metadata is
    allowed, but the supplied fields must match exactly.
    """
    manifest, _arrays = _load_checkpoint(path, checkpoint_name=checkpoint_name)
    if manifest["patient_uid"] != patient_uid:
        raise ValueError("checkpoint patient UID differs from requested job")
    if manifest["scientific_config"]["config_sha256"] != scientific_config_sha256:
        raise ValueError("checkpoint scientific config differs from requested job")
    if any(manifest["metadata"].get(key) != value for key, value in expected_metadata.items()):
        raise ValueError("checkpoint metadata differs from requested job")


def _compare_array(reference: np.ndarray, candidate: np.ndarray, *, abs_tol: float, rel_tol: float, exact: bool, exact_columns: tuple[int, ...] = ()) -> dict[str, Any]:
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        return {"passed": False, "reason": "shape or dtype mismatch"}
    finite = np.isfinite(reference)
    same_masks = (
        (finite == np.isfinite(candidate))
        & (np.isnan(reference) == np.isnan(candidate))
        & (np.isposinf(reference) == np.isposinf(candidate))
        & (np.isneginf(reference) == np.isneginf(candidate))
    )
    shared_finite = finite & np.isfinite(candidate)
    reference_values = reference[shared_finite].astype(np.longdouble)
    candidate_values = candidate[shared_finite].astype(np.longdouble)
    with np.errstate(over="ignore", invalid="ignore"):
        differences = np.abs(candidate_values - reference_values)
        limits = np.longdouble(abs_tol) + np.longdouble(rel_tol) * np.abs(reference_values)
    exact = exact or reference.dtype.kind in "biu"
    mismatches = reference[shared_finite] != candidate[shared_finite] if exact else differences > limits
    if not exact:
        overflowed = ~np.isfinite(differences) | ~np.isfinite(limits)
        scale = np.maximum(np.abs(reference_values[overflowed]), np.abs(candidate_values[overflowed]))
        mismatches[overflowed] = (
            np.abs(candidate_values[overflowed] / scale - reference_values[overflowed] / scale)
            > np.longdouble(abs_tol) / scale + np.longdouble(rel_tol) * (np.abs(reference_values[overflowed]) / scale)
        )
    if exact_columns:
        discrete_mask = np.zeros(reference.shape, dtype=bool)
        discrete_mask[..., list(exact_columns)] = True
        mismatches |= discrete_mask[shared_finite] & (reference[shared_finite] != candidate[shared_finite])
    maximum = float(np.max(differences, initial=np.longdouble(0)))
    mask_count = int(np.count_nonzero(~same_masks))
    numeric_count = int(np.count_nonzero(mismatches))
    return {
        "passed": mask_count == 0 and numeric_count == 0,
        "values": int(reference.size), "finite_pairs": int(shared_finite.sum()),
        "nonfinite_mask_mismatches": mask_count,
        "numeric_mismatches": numeric_count,
        "max_absolute_difference": maximum if math.isfinite(maximum) else "overflow",
        "exact": exact,
        "exact_columns": list(exact_columns),
    }


def compare_anatomical_checkpoints(
    reference_path: Path, candidate_path: Path, *, abs_tol: float,
    rel_tol: float, output_path: Path | None = None,
    checkpoint_name: str = "anatomical_qa",
) -> dict:
    """Compare verified artifacts without running scientific algorithms.

    Identity, schema, shapes, dtypes, integer/boolean values and nonfinite masks
    are exact. Finite reals satisfy abs(candidate-reference) <= abs_tol +
    rel_tol*abs(reference), elementwise. Nonnegative finite tolerances are
    required (ValueError otherwise). Missing/corrupt evidence returns a failed
    JSON-compatible report, never a vacuous pass. Optional output is written
    exclusively; no existing file is overwritten. Metadata fingerprints are
    verified internally and retained in the report, not required to match across
    independently captured pathways. These unkeyed hashes detect corruption,
    not authenticity or source/input/environment content identity.
    """
    boundary = preprocessing_boundary(checkpoint_name)
    for tolerance in (abs_tol, rel_tol):
        if isinstance(tolerance, bool) or not isinstance(tolerance, Real) or not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerances must be finite nonnegative numbers")
    result: dict[str, Any] = {
        "schema_version": boundary.directory + "_checkpoint_comparison_v1", "passed": False,
        "abs_tol": float(abs_tol), "rel_tol": float(rel_tol),
        "items": [], "coverage": {}, "errors": [],
        "metadata_policy": "verify internal fingerprints; do not require cross-pathway equality",
    }
    loaded = {}
    for role, path in (("reference", reference_path), ("candidate", candidate_path)):
        try:
            manifest, arrays = _load_checkpoint(Path(path), checkpoint_name=checkpoint_name)
            loaded[role] = (manifest, arrays)
            result["coverage"][role] = manifest["coverage"]
            result[role + "_metadata"] = manifest["metadata"]
        except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, EOFError, BadZipFile) as error:
            result["errors"].append({"role": role, "message": str(error)})
            result["coverage"][role] = {"verified": False}
    if len(loaded) == 2:
        reference_manifest, reference_arrays = loaded["reference"]
        candidate_manifest, candidate_arrays = loaded["candidate"]
        for key in ("schema_version", "patient_uid", "families", "reference_keys", "scientific_config"):
            result["items"].append({"path": [key], "passed": reference_manifest[key] == candidate_manifest[key], "reason": "exact identity"})
        reference_items = {tuple(item["path"]): item for item in reference_manifest["items"]}
        candidate_items = {tuple(item["path"]): item for item in candidate_manifest["items"]}
        for path in sorted(reference_items.keys() | candidate_items.keys()):
            reference_item = reference_items.get(path)
            candidate_item = candidate_items.get(path)
            report: dict[str, Any] = {"path": list(path)}
            if reference_item is None or candidate_item is None:
                report.update(passed=False, reason="item coverage mismatch")
            else:
                reference_schema = {key: value for key, value in reference_item.items() if key != "payload"}
                candidate_schema = {key: value for key, value in candidate_item.items() if key != "payload"}
                if reference_schema != candidate_schema:
                    report.update(passed=False, reason="identity, schema, label, or presence mismatch")
                elif "payload" in reference_item:
                    report.update(_compare_array(reference_arrays[reference_item["payload"]], candidate_arrays[candidate_item["payload"]], abs_tol=abs_tol, rel_tol=rel_tol, exact=reference_item["contract"]["exact"], exact_columns=tuple(reference_item["contract"]["exact_columns"])))
                else:
                    report.update(passed=True, reason="exact labels/schema or symmetric explicit absence")
            result["items"].append(report)
        result["passed"] = all(item["passed"] for item in result["items"])
    result["coverage"]["comparison"] = {
        "arrays_compared": sum("finite_pairs" in item for item in result["items"]),
        "finite_values_compared": sum(item.get("finite_pairs", 0) for item in result["items"]),
        "numeric_mismatches": sum(item.get("numeric_mismatches", 0) for item in result["items"]),
        "nonfinite_mask_mismatches": sum(item.get("nonfinite_mask_mismatches", 0) for item in result["items"]),
        "failed_items": sum(not item["passed"] for item in result["items"]),
        "artifact_errors": len(result["errors"]),
    }
    if output_path is not None:
        _write_json(Path(output_path), result)
    return result


__all__ = ["AnatomicalFieldSpec", "write_anatomical_checkpoint", "compare_anatomical_checkpoints", "validate_anatomical_checkpoint_identity"]
