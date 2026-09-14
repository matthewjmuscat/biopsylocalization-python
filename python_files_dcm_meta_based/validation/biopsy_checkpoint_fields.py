"""Bounded biopsy products appended to the existing anatomical evidence engine.

Transitional dictionary reader only: no preprocessing or DICOM I/O. Real biopsy
coordinates retain the DICOM patient frame; planned simulated coordinates retain
the canonical local biopsy frame. Native numeric arrays are neither recomputed
nor reordered. Runtime dictionaries, native objects, and resources are not saved.
"""

from collections.abc import Mapping

import numpy as np

from .anatomical_checkpoint import (
    AnatomicalFieldSpec as Field, _capture_structure, _mapping, _member, _points_spec,
)


_PREPARATION_LABELS = (
    "Length determined", "Length source", "Target determined", "Target source",
    "Target structure type", "Target structure ref #", "Target structure index", "Target structure ID",
    "Multiplicity", "Multiplicity index", "Real matched biopsy count", "Matched real biopsy ROI",
    "Matched real biopsy ref #", "Matched real biopsy index", "Extra biopsy bool", "Family source",
    "Multiplicity base ROI", "Multiplicity base ref #", "Preparation complete",
)
_MODEL_FIELDS = (
    Field("Reconstructed biopsy cylinder length (from contour data)", units="mm"),
    Field("Best fit line of centroid pts", axes=("endpoint", "xyz"), shape=(2, 3), units="mm"),
    Field("Centroid line unit vec (bx needle base to bx needle tip)", axes=("xyz",), shape=(3,), units="dimensionless"),
    Field("Centroid line vec (bx needle base to bx needle tip)", axes=("xyz",), shape=(3,), units="mm"),
    Field("Centroid line vec length (bx needle base to bx needle tip)", units="mm"),
    _points_spec("Centroid line sample pts"),
    _points_spec("Reconstructed structure pts arr"),
    Field("Centroid variation arr", axes=("slice",), shape=(None,), units="mm"),
    Field("Mean centroid variation", units="mm"),
    Field("Maximum projected distance between original centroids", units="mm"),
)
_PLANNING_LABELS = ("Planning complete", "Planning frame", "Planned centroid count",
                    "Planned sampled point count", "Planning source")
_PLANNING_LENGTHS = ("Nominal length mm", "Planned biopsy radius mm", "Planned centroid separation mm",
                     "Planned biopsy cylinder length mm", "Planned sample lattice spacing mm",
                     "Planned mean centroid variation", "Planned maximum projected distance between original centroids")


def _fields(capture, path, record, specs):
    for spec in specs:
        capture.field([*path, spec.key], spec, record.get(spec.key))


def _model(capture, path, record):
    _fields(capture, path, record, _MODEL_FIELDS)
    triangulation = _member(record.get("Reconstructed structure delaunay global"), "delaunay_triangulation")
    capture.field([*path, "delaunay", "points"], _points_spec("points"), _member(triangulation, "points"))
    capture.field([*path, "delaunay", "simplices"], Field("simplices", axes=("tetrahedron", "vertex_index"),
                  shape=(None, 4), units="vertex indices", exact=True), _member(triangulation, "simplices"))


def capture_biopsy_record(capture, path, record):
    """Read a fixed inventory, including explicit absent markers for either kind."""
    _capture_structure(capture, [*path, "real_geometry"], record)
    _fields(capture, path, record, [Field(key, "label") for key in
            ("ROI", "Ref #", "Index number", "Struct type", "Simulated bool", "Simulated type",
             "Relative structure type", "Relative structure ref #")])
    _model(capture, [*path, "real_model"], record)
    preparation = _mapping(record.get("Simulated biopsy preparation dict"))
    _fields(capture, [*path, "preparation"], preparation,
            [Field(key, "label") for key in _PREPARATION_LABELS] +
            [Field(key, units="mm") for key in ("Contour length mm", "Centroid line length mm", "Nominal length mm")])
    planning = _mapping(record.get("Simulated biopsy planning dict"))
    planning_path = [*path, "planning"]
    _fields(capture, planning_path, planning,
            [Field(key, "label") for key in _PLANNING_LABELS] +
            [Field(key, units="mm; canonical local biopsy frame") for key in _PLANNING_LENGTHS] + [
                Field("Planned raw contour pts zslice list", "slices", ("point", "xyz"), (None, 3), "mm; canonical local biopsy frame"),
                Field("Planned structure global centroid", axes=("singleton", "xyz"), shape=(1, 3), units="mm; canonical local biopsy frame"),
                Field("Planned centroid variation arr", axes=("slice",), shape=(None,), units="mm"),
                _points_spec("Planned sampled volume pts arr"), _points_spec("Planned sample bounding box pts arr"),
            ])
    sampling = _mapping(planning.get("Planned sampling metadata"))
    _fields(capture, [*planning_path, "sampling"], sampling,
            [Field(key, "label") for key in ("Patient UID", "Structure type", "Specific structure index")])
    model = _mapping(planning.get("Planned reconstructed biopsy model dict"))
    model_path = [*planning_path, "model"]
    _model(capture, model_path, model)
    _fields(capture, model_path, model, [
        _points_spec("Raw contour pts"), _points_spec("Structure centroid pts"),
        Field("Structure global centroid", axes=("singleton", "xyz"), shape=(1, 3), units="mm"),
        Field("Distance between centroid sample rings", units="mm"),
        _points_spec("Rotated reconstructed structure pts arr rounded"),
        Field("Rotated reconstructed structure z values", axes=("slice",), shape=(None,), units="mm"),
        Field("Rotated reconstructed structure zslice list", "slices", ("point", "xyz"), (None, 3), "mm"),
        Field("Biopsy coord sys origin translation vec", axes=("xyz",), shape=(3,), units="mm"),
        Field("Centroid line to z axis rotation matrix", axes=("row", "column"), shape=(3, 3), units="dimensionless"),
    ])


def capture_biopsies(capture, patient, refs):
    """Capture all biopsy records and the patient preparation dataframe."""
    records = patient.get(refs.bx_ref)
    if records is not None and not isinstance(records, (list, tuple)):
        raise TypeError("biopsy family must be a list of records")
    capture.field(["biopsies", "count"], Field("count", "label"), None if records is None else len(records))
    for index, record in enumerate(records or ()):
        if not isinstance(record, Mapping):
            raise TypeError("biopsy record must be a mapping")
        capture_biopsy_record(capture, ["biopsies", str(index)], record)
    table = _mapping(_mapping(patient.get(refs.all_ref_key)).get("Multi-structure pre-processing output dataframes dict"))
    capture.field(["biopsy_preparation_table"], Field("Simulated biopsy preparation dataframe", "table"),
                  table.get("Simulated biopsy preparation dataframe"))


def expected_biopsy_inventory(capture, items):
    """Build schema descriptors without trusting producer-supplied contracts."""
    count_item = items[("biopsies", "count")]
    count = count_item.get("value", {}).get("value")
    if type(count) is not int or count < 1 or count > len(items):
        raise ValueError("biopsy checkpoint requires at least one biopsy")
    capture.field(["biopsies", "count"], Field("count", "label"), count)
    for index in range(count):
        capture_biopsy_record(capture, ["biopsies", str(index)], {})
    capture.field(["biopsy_preparation_table"], Field("Simulated biopsy preparation dataframe", "table"), None)


def biopsy_coverage(items, arrays):
    """Reject vacuous/unfinished evidence and report coverage by biopsy kind.

    This checks presence and structural consistency, not independent algorithm
    correctness. Resolved decisions, including legitimate incomplete targets,
    remain exact labels. Zero planning samples are retained with their true count.
    """
    entries = {tuple(item["path"]): item for item in items}
    count = entries[("biopsies", "count")].get("value", {}).get("value", 0)
    records = []
    for index in range(count or 0):
        base = ("biopsies", str(index))
        missing = []

        def require(path, *, numeric=False, nonempty=True):
            item = entries[base + path]
            if item["state"] != "present":
                missing.append(list(path))
                return None
            if numeric:
                value = arrays.get(item.get("payload"))
                if value is None or (nonempty and not value.size) or not np.isfinite(value).all():
                    missing.append(list(path))
                return value
            return item.get("value", {}).get("value")

        for key in ("ROI", "Ref #", "Index number", "Simulated type"):
            require((key,))
        simulated = require(("Simulated bool",))
        if type(simulated) is not bool:
            missing.append(["Simulated bool"])
        require(("preparation", "Nominal length mm"), numeric=True)
        require(("preparation", "Length source"))
        if require(("preparation", "Length determined")) is not True:
            missing.append(["preparation", "Length determined"])
        model = ("planning", "model") if simulated else ("real_model",)
        for spec in _MODEL_FIELDS:
            require((*model, spec.key), numeric=True)
        require((*model, "delaunay", "points"), numeric=True)
        require((*model, "delaunay", "simplices"), numeric=True)
        if simulated:
            if require(("planning", "Planning complete")) is not True:
                missing.append(["planning", "Planning complete"])
            if require(("planning", "Planning frame")) != "Canonical local biopsy frame":
                missing.append(["planning", "Planning frame"])
            for key in _PLANNING_LENGTHS:
                require(("planning", key), numeric=True)
            samples = require(("planning", "Planned sampled volume pts arr"), numeric=True, nonempty=False)
            sample_count = require(("planning", "Planned sampled point count"))
            if type(sample_count) is not int or samples is None or sample_count != len(samples):
                missing.append(["planning", "Planned sampled point count"])
            require(("planning", "Planned sample bounding box pts arr"), numeric=True)
            require(("planning", "model", "Centroid line to z axis rotation matrix"), numeric=True)
        else:
            require(("real_geometry", "Raw contour pts"), numeric=True)
            require(("real_geometry", "Structure volume"), numeric=True)
            require(("real_geometry", "Structure centroid pts"), numeric=True)
        records.append({"index": index, "simulated": simulated, "missing_required": missing, "complete": not missing})
    table = entries[("biopsy_preparation_table",)]
    table_present = table["state"] == "present"
    table_rows = table.get("table_schema", {}).get("shape", [None])[0]
    table_complete = table_present and table_rows == len(records)
    return {"records": records, "real_count": sum(r["simulated"] is False for r in records),
            "simulated_count": sum(r["simulated"] is True for r in records),
            "preparation_table_present": table_present,
            "preparation_table_rows": table_rows,
            "complete": bool(records) and table_complete and all(r["complete"] for r in records)}
