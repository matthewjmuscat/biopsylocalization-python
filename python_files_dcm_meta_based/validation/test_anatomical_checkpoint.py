"""Synthetic CPU-only evidence for independent anatomical checkpoint contracts.

No patient files, scientific preprocessing, or GPU implementations are used.
"""

from dataclasses import make_dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from config.snapshots import PIPELINE_SCIENTIFIC_CONFIG_FIELD_NAMES, canonical_sha256, build_pipeline_scientific_config_snapshot
from validation.anatomical_checkpoint import compare_anatomical_checkpoints, write_anatomical_checkpoint, validate_anatomical_checkpoint_identity


def _config():
    refs = make_dataclass("CheckpointRefs", [(name, str) for name in (
        "oar_ref", "dil_ref", "rectum_ref_key", "urethra_ref_key", "all_ref_key", "dose_ref", "mr_adc_ref",
    )])("OAR ref", "DIL ref", "Rectum ref", "Urethra ref", "All ref", "Dose ref", "MR ADC ref")
    config_type = make_dataclass("CheckpointConfig", [(name, object) for name in PIPELINE_SCIENTIFIC_CONFIG_FIELD_NAMES])
    return config_type(**{name: refs if name == "legacy_refs" else {} for name in PIPELINE_SCIENTIFIC_CONFIG_FIELD_NAMES})


def _runtime():
    points = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.], [0., 0., 2.]])
    record = {
        "ROI": "synthetic-prostate", "Ref #": 1, "Index number": 0, "Struct type": "OAR ref",
        "Raw contour pts": points.copy(), "Raw contour pts zslice list": [points.copy()],
        "Equal num zslice contour pts": [points.copy()], "Structure volume": 1.25,
    }
    return SimpleNamespace(
        patient_case=SimpleNamespace(patient_uid="synthetic-001"),
        master_structure_reference_dict={"synthetic-001": {"OAR ref": [record], "DIL ref": [], "Rectum ref": [], "Urethra ref": []}},
    )


def _record(runtime):
    return runtime.master_structure_reference_dict["synthetic-001"]["OAR ref"][0]


def _patient(runtime):
    return runtime.master_structure_reference_dict["synthetic-001"]


def _modalities(runtime):
    _patient(runtime)["Dose ref"] = {
        "Dose units": "GY", "Dose type": "PHYSICAL", "Dose pixel arr": np.ones((2, 2, 2), dtype=np.uint16),
        "Dose grid scaling": 0.1, "Pixel spacing": [1., 1.], "Grid frame offset vector": [0., 2.],
        "Image orientation patient": [1., 0., 0., 0., 1., 0.], "Image position patient": [0., 0., 0.],
        "Dose and gradient phys space and pixel 3d arr": np.arange(112., dtype=np.float64).reshape(2, 4, 14),
    }
    _patient(runtime)["MR ADC ref"] = {
        "Units": "mm2/s", "RWV Units": "ADC", "Series instance UID": "synthetic-series",
        "Pixel arr (all slices)": np.full((2, 2, 2), 0.001),
        "RWVSlope (all slices)": np.array([1e-6, 1e-6]), "RWVIntercept (all slices)": np.zeros(2),
        "Pixel spacing": np.ones(2), "Slice thickness": 2.,
        "Image orientation patient": np.array([1., 0., 0., 0., 1., 0.]),
        "Image position patient (all slices)": np.array([[0., 0., 0.], [0., 0., 2.]]),
    }
    return runtime


def _tables(runtime):
    columns = pd.MultiIndex.from_tuples([("ROI", "label"), ("ADC", "mean")], names=["quantity", "statistic"])
    table = pd.DataFrame([["synthetic-prostate", 0.001], ["synthetic-dil", 0.002]], columns=columns)
    table.index = pd.MultiIndex.from_tuples([("synthetic-001", 1), ("synthetic-001", 2)], names=["patient", "roi"])
    _patient(runtime)["All ref"] = {"Multi-structure pre-processing output dataframes dict": {"MR - ADC - summary statistics by structure dataframe": table}}
    _record(runtime)["Structure features dataframe"] = pd.DataFrame({
        "ROI": pd.Categorical(["synthetic-prostate"]), "Volume": [1.25],
        "PCA eigenvector major": [(1., 0., 0.)], "Missing": pd.Series([None], dtype="Float64"),
    })
    return table


class AnatomicalCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = _config()

    def capture(self, name, runtime=None, **kwargs):
        return write_anatomical_checkpoint(runtime_state=runtime or _runtime(), pipeline_config=self.config, output_dir=self.root / name, **kwargs)

    def compare(self, reference, candidate, **kwargs):
        return compare_anatomical_checkpoints(reference, candidate, abs_tol=kwargs.pop("abs_tol", 0.), rel_tol=kwargs.pop("rel_tol", 0.), **kwargs)

    def test_equal_numeric_roundtrip(self):
        reference = self.capture("reference", metadata={"source_sha256": "caller-supplied"})
        candidate = self.capture("candidate")
        result = self.compare(reference, candidate)
        self.assertTrue(result["passed"], result)
        self.assertTrue(result["coverage"]["reference"]["geometry_complete"])
        manifest = json.loads(reference.read_text())
        with np.load(reference.parent / manifest["arrays_file"], allow_pickle=False) as arrays:
            self.assertGreater(len(arrays.files), 0)
            self.assertTrue(all(not arrays[key].dtype.hasobject for key in arrays.files))
        self.assertEqual(result["reference_metadata"], {"source_sha256": "caller-supplied"})

    def test_checkpoint_is_bound_to_requested_patient_config_and_job(self):
        path = self.capture("identity", metadata={"worker_job_id": "expected-job"})
        kwargs = {
            "patient_uid": "synthetic-001",
            "scientific_config_sha256": build_pipeline_scientific_config_snapshot(self.config).config_sha256,
            "expected_metadata": {"worker_job_id": "expected-job"},
        }
        validate_anatomical_checkpoint_identity(path, **kwargs)
        for changed in (
            {"patient_uid": "other-patient"},
            {"scientific_config_sha256": "other-config"},
            {"expected_metadata": {"worker_job_id": "other-job"}},
        ):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_anatomical_checkpoint_identity(path, **{**kwargs, **changed})

    def test_single_coordinate_difference(self):
        runtime = _runtime()
        _record(runtime)["Equal num zslice contour pts"][0][0, 0] += 0.25
        result = self.compare(self.capture("reference"), self.capture("candidate", runtime))
        self.assertFalse(result["passed"])
        differences = [item for item in result["items"] if item.get("numeric_mismatches", 0)]
        self.assertEqual(len(differences), 1)
        self.assertEqual(differences[0]["numeric_mismatches"], 1)
        self.assertEqual(differences[0]["max_absolute_difference"], 0.25)

    def test_modalities_and_tables_equal(self):
        runtime = _modalities(_runtime())
        _tables(runtime)
        points = _record(runtime)["Raw contour pts"]
        _record(runtime)["Interpolated structure point cloud dict"] = {"Full": SimpleNamespace(points=points)}
        _record(runtime)["Structure OPEN3D triangle mesh object"] = SimpleNamespace(vertices=points, triangles=np.array([[0, 1, 2]], dtype=np.int32))
        result = self.compare(self.capture("reference", runtime), self.capture("candidate", runtime))
        self.assertTrue(result["passed"], result)
        self.assertTrue(result["coverage"]["reference"]["modalities"]["mr_adc"]["present"])

    def test_tables_numeric_change(self):
        runtime = _runtime()
        table = _tables(runtime)
        reference = self.capture("reference", runtime)
        table.iloc[0, 1] += 0.01
        result = self.compare(reference, self.capture("candidate", runtime))
        self.assertFalse(result["passed"])
        self.assertTrue(any(item.get("numeric_mismatches") == 1 and item["path"][0] == "patient_tables" for item in result["items"]))

    def test_identity_units_and_config_exact(self):
        for field in ("ROI", "Ref #", "Struct type", "Index number", "dose units", "mr units", "config"):
            with self.subTest(field=field):
                runtime = _modalities(_runtime())
                reference = self.capture("ref-" + field, runtime)
                if field in ("ROI", "Struct type"):
                    _record(runtime)[field] = "changed"
                elif field in ("Ref #", "Index number"):
                    _record(runtime)[field] += 1
                elif field == "dose units":
                    _patient(runtime)["Dose ref"]["Dose units"] = "CGY"
                elif field == "mr units":
                    _patient(runtime)["MR ADC ref"]["Units"] = "stored"
                else:
                    self.config.preprocessing = {"spacing": 2.0}
                self.assertFalse(self.compare(reference, self.capture("cand-" + field, runtime), abs_tol=100., rel_tol=100.)["passed"])

    def test_optional_modality_absence_asymmetry(self):
        for role in ("Dose ref", "MR ADC ref"):
            with self.subTest(role=role):
                runtime = _modalities(_runtime())
                reference = self.capture("ref-" + role, runtime)
                del _patient(runtime)[role]
                candidate = self.capture("cand-" + role, runtime)
                self.assertFalse(self.compare(reference, candidate)["passed"])
                self.assertFalse(self.compare(candidate, reference)["passed"])

    def test_nan_and_infinity_masks(self):
        runtime = _runtime()
        _record(runtime)["Structure features dataframe"] = pd.DataFrame({"value": [np.nan, np.inf, -np.inf, 4.]})
        reference = self.capture("reference", runtime)
        self.assertTrue(self.compare(reference, self.capture("same", runtime))["passed"])
        for index, value in enumerate((np.inf, -np.inf, 0., np.nan)):
            candidate_runtime = _runtime()
            values = [np.nan, np.inf, -np.inf, 4.]
            values[index] = value
            _record(candidate_runtime)["Structure features dataframe"] = pd.DataFrame({"value": values})
            result = self.compare(reference, self.capture("candidate-" + str(index), candidate_runtime), abs_tol=100.)
            self.assertFalse(result["passed"])
            self.assertEqual(sum(item.get("nonfinite_mask_mismatches", 0) for item in result["items"]), 1)
            json.dumps(result, allow_nan=False)

    def test_invalid_tolerances(self):
        for value in (-1., float("nan"), float("inf"), True, "0.1", None, 1j):
            for parameter in ("abs_tol", "rel_tol"):
                with self.subTest(value=value, parameter=parameter), self.assertRaises(ValueError):
                    self.compare(self.root / "missing", self.root / "missing", **{parameter: value})

    def test_missing_npz(self):
        reference = self.capture("reference")
        candidate = self.capture("candidate")
        (candidate.parent / "anatomical_arrays.npz").unlink()
        result = self.compare(reference, candidate)
        self.assertFalse(result["passed"])
        self.assertFalse(result["coverage"]["candidate"]["verified"])

    def test_empty_or_incomplete_geometry(self):
        for mode in ("no structures", "raw only", "missing identity", "empty", "nonfinite", "degenerate", "second incomplete"):
            with self.subTest(mode=mode):
                runtime = _runtime()
                record = _record(runtime)
                if mode == "no structures":
                    _patient(runtime)["OAR ref"] = []
                elif mode == "raw only":
                    del record["Equal num zslice contour pts"]
                elif mode == "missing identity":
                    del record["ROI"]
                elif mode == "second incomplete":
                    _patient(runtime)["DIL ref"] = [{"ROI": "empty", "Ref #": 2}]
                else:
                    record["Equal num zslice contour pts"] = [np.empty((0, 3)) if mode == "empty" else np.full((4, 3), np.nan if mode == "nonfinite" else 0.)]
                with self.assertRaises(ValueError):
                    self.capture(mode, runtime)
                self.assertFalse((self.root / mode).exists())

    def test_present_incomplete_modality(self):
        runtime = _modalities(_runtime())
        del _patient(runtime)["Dose ref"]["Dose and gradient phys space and pixel 3d arr"]
        with self.assertRaisesRegex(ValueError, "incomplete grid"):
            self.capture("candidate", runtime)

    def test_actual_tolerances_and_exact_dose_indices(self):
        runtime = _modalities(_runtime())
        reference = self.capture("reference", runtime)
        mapped = _patient(runtime)["Dose ref"]["Dose and gradient phys space and pixel 3d arr"]
        mapped[0, 0, 6] += 0.25
        candidate = self.capture("tolerated", runtime)
        self.assertFalse(self.compare(reference, candidate, abs_tol=0.24)["passed"])
        self.assertTrue(self.compare(reference, candidate, abs_tol=0.25)["passed"])
        self.assertTrue(self.compare(reference, candidate, rel_tol=0.05)["passed"])
        mapped[0, 0, 0] += 1
        self.assertFalse(self.compare(reference, self.capture("indices", runtime), abs_tol=1000.)["passed"])

    def test_no_overwrite_and_no_runtime_mutation(self):
        runtime = _modalities(_runtime())
        before = _patient(runtime)["MR ADC ref"]["Pixel arr (all slices)"].copy()
        reference = self.capture("reference", runtime)
        original_bytes = reference.read_bytes()
        with self.assertRaises(FileExistsError):
            self.capture("reference", runtime)
        self.assertEqual(reference.read_bytes(), original_bytes)
        self.root.joinpath("empty").mkdir()
        with self.assertRaises(FileExistsError):
            self.capture("empty", runtime)
        output = self.root / "comparison.json"
        result = self.compare(reference, reference, output_path=output)
        self.assertEqual(json.loads(output.read_text()), result)
        with self.assertRaises(FileExistsError):
            self.compare(reference, reference, output_path=output)
        np.testing.assert_array_equal(before, _patient(runtime)["MR ADC ref"]["Pixel arr (all slices)"])

    def test_unsupported_objects_fail_closed(self):
        with self.assertRaises(TypeError):
            self.capture("metadata", metadata={"unsupported": object()})
        runtime = _runtime()
        _record(runtime)["Structure features dataframe"] = pd.DataFrame({"object": [object()]})
        with self.assertRaises(TypeError):
            self.capture("table", runtime)
        runtime = _runtime()
        _record(runtime)["Interpolated structure point cloud dict"] = {"Full": object()}
        with self.assertRaises(TypeError):
            self.capture("cloud", runtime)

    def rewrite_manifest(self, path, change):
        manifest = json.loads(path.read_text())
        change(manifest)
        manifest.pop("manifest_sha256")
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        path.write_text(json.dumps(manifest))

    def test_metadata_and_config_fingerprints_verified(self):
        for name in ("metadata", "scientific_config"):
            reference = self.capture("ref-" + name)
            candidate = self.capture("cand-" + name)
            def change(manifest):
                if name == "metadata":
                    manifest["metadata"]["changed"] = True
                else:
                    manifest["scientific_config"]["config"]["preprocessing"] = {"changed": True}
            self.rewrite_manifest(candidate, change)
            self.assertFalse(self.compare(reference, candidate)["passed"])

    def test_manifest_and_npz_tampering(self):
        reference = self.capture("reference")
        candidate = self.capture("manifest")
        payload = json.loads(candidate.read_text())
        payload["patient_uid"] = "tampered"
        candidate.write_text(json.dumps(payload))
        self.assertFalse(self.compare(reference, candidate)["passed"])
        candidate = self.capture("npz")
        arrays_path = candidate.parent / "anatomical_arrays.npz"
        arrays_path.write_bytes(arrays_path.read_bytes() + b"tampered")
        self.assertFalse(self.compare(reference, candidate)["passed"])

    def test_resigned_incomplete_inventory_rejected(self):
        candidate = self.capture("candidate")
        def change(manifest):
            manifest["items"] = [item for item in manifest["items"] if item["path"] != ["dose", "Dose units"]]
            manifest["coverage"]["absent"].remove(["dose", "Dose units"])
        self.rewrite_manifest(candidate, change)
        self.assertFalse(self.compare(candidate, candidate)["passed"])

    def test_pickle_payload_is_never_loaded(self):
        candidate = self.capture("candidate")
        arrays_path = candidate.parent / "anatomical_arrays.npz"
        with arrays_path.open("wb") as stream:
            np.savez(stream, array_0=np.array([object()], dtype=object))
        self.rewrite_manifest(candidate, lambda manifest: manifest.update(arrays_sha256=hashlib.sha256(arrays_path.read_bytes()).hexdigest()))
        with patch("pickle.load", side_effect=AssertionError("pickle forbidden")):
            self.assertFalse(self.compare(candidate, candidate)["passed"])

    def test_geometry_objects_are_numeric_not_repr(self):
        runtime = _runtime()
        record = _record(runtime)
        points = record["Raw contour pts"].copy()
        record["Inter-slice interpolation information"] = SimpleNamespace(
            interpolated_pts_np_arr=points.copy(), interpolated_pts_list=[points.copy()],
            zslice_vals_after_interpolation_list=[0., 2.],
        )
        record["Intra-slice interpolation information"] = SimpleNamespace(
            interpolated_pts_np_arr=points.copy(), interpolated_pts_with_end_caps_np_arr=points.copy(),
        )
        record["Interpolated structure point cloud dict"] = {"Full": SimpleNamespace(points=points.copy())}
        record["Structure OPEN3D triangle mesh object"] = SimpleNamespace(vertices=points.copy(), triangles=np.array([[0, 1, 2]], dtype=np.int32))
        record["Structure global centroid"] = np.array([[0.5, 0.5, 0.5]])
        reference = self.capture("reference", runtime)
        targets = (
            record["Inter-slice interpolation information"].interpolated_pts_np_arr,
            record["Intra-slice interpolation information"].interpolated_pts_with_end_caps_np_arr,
            record["Interpolated structure point cloud dict"]["Full"].points,
            record["Structure OPEN3D triangle mesh object"].vertices,
            record["Structure OPEN3D triangle mesh object"].triangles,
        )
        for index, array in enumerate(targets):
            array[0, 0] += 1
            result = self.compare(reference, self.capture("candidate-" + str(index), runtime))
            self.assertFalse(result["passed"])
            self.assertEqual(result["coverage"]["comparison"]["numeric_mismatches"], 1)
            array[0, 0] -= 1

    def test_array_shape_and_dtype_exact(self):
        reference = self.capture("reference")
        for mode in ("shape", "dtype"):
            runtime = _runtime()
            points = _record(runtime)["Equal num zslice contour pts"][0]
            _record(runtime)["Equal num zslice contour pts"] = [points[:3] if mode == "shape" else points.astype(np.float32)]
            self.assertFalse(self.compare(reference, self.capture(mode, runtime), abs_tol=100.)["passed"])

    def test_table_labels_dtypes_categories_and_order_exact(self):
        for mode in ("column", "index", "dtype", "category", "row order", "name"):
            runtime = _runtime()
            table = _tables(runtime)
            reference = self.capture("reference-" + mode, runtime)
            if mode == "column":
                table.columns = pd.MultiIndex.from_tuples([("ROI", "label"), ("ADC", "median")])
            elif mode == "index":
                table.index = pd.MultiIndex.from_tuples([("synthetic-001", 8), ("synthetic-001", 2)])
            elif mode == "dtype":
                table[("ADC", "mean")] = table[("ADC", "mean")].astype(np.float32)
            elif mode == "category":
                features = _record(runtime)["Structure features dataframe"]
                features["ROI"] = features["ROI"].cat.add_categories(["unused-category"])
            elif mode == "name":
                table.columns.names = ["renamed", "statistic"]
            else:
                table.sort_index(ascending=False, inplace=True)
            self.assertFalse(self.compare(reference, self.capture("candidate-" + mode, runtime), abs_tol=100.)["passed"])

    def test_numeric_tuple_and_nullable_cells(self):
        runtime = _runtime()
        _tables(runtime)
        features = _record(runtime)["Structure features dataframe"]
        reference = self.capture("reference", runtime)
        features.at[0, "PCA eigenvector major"] = (1., 0.125, 0.)
        candidate = self.capture("tuple", runtime)
        self.assertFalse(self.compare(reference, candidate)["passed"])
        self.assertTrue(self.compare(reference, candidate, abs_tol=0.125)["passed"])
        features.at[0, "Missing"] = 0.
        self.assertFalse(self.compare(reference, self.capture("missing", runtime), abs_tol=100.)["passed"])

    def test_table_name_exclusions_are_explicit(self):
        runtime = _runtime()
        _patient(runtime)["All ref"] = {"Multi-structure pre-processing output dataframes dict": {"Structure preprocessing timings": object()}}
        path = self.capture("reference", runtime)
        coverage = self.compare(path, path)["coverage"]["reference"]
        self.assertEqual(coverage["excluded_patient_tables"]["items"][0]["value"], "Structure preprocessing timings")

    def test_npz_shape_manifest_is_verified_even_after_rehash(self):
        path = self.capture("candidate")
        arrays_path = path.parent / "anatomical_arrays.npz"
        with np.load(arrays_path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        first_key = next(iter(arrays))
        arrays[first_key] = np.ones((2, 2))
        with arrays_path.open("wb") as stream:
            np.savez(stream, **arrays)
        self.rewrite_manifest(path, lambda manifest: manifest.update(arrays_sha256=hashlib.sha256(arrays_path.read_bytes()).hexdigest()))
        self.assertFalse(self.compare(path, path)["passed"])

    def test_invalid_resigned_geometry_is_rejected(self):
        path = self.capture("candidate")
        arrays_path = path.parent / "anatomical_arrays.npz"
        with np.load(arrays_path, allow_pickle=False) as archive:
            arrays = {key: np.full_like(archive[key], np.nan) if archive[key].ndim == 2 else archive[key] for key in archive.files}
        with arrays_path.open("wb") as stream:
            np.savez(stream, **arrays)
        self.rewrite_manifest(path, lambda manifest: manifest.update(arrays_sha256=hashlib.sha256(arrays_path.read_bytes()).hexdigest()))
        self.assertFalse(self.compare(path, path)["passed"])

    def test_cpu_only_import_in_fresh_process(self):
        script = """
import importlib.abc
import sys
forbidden = {'cupy', 'cudf', 'cuspatial', 'rmm', 'open3d', 'torch', 'pycuda'}
class BlockGPU(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in forbidden:
            raise AssertionError('GPU/render import attempted: ' + fullname)
sys.meta_path.insert(0, BlockGPU())
from validation.anatomical_checkpoint import write_anatomical_checkpoint, compare_anatomical_checkpoints
assert not forbidden.intersection(sys.modules)
"""
        completed = subprocess.run(
            [sys.executable, "-B", "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()