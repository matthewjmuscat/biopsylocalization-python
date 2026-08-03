from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from output_artifacts.manifest_index import MANIFEST_STATUS_CONSTRUCTED_NOT_WRITTEN
from output_artifacts.manifest_index import MANIFEST_STATUS_WRITTEN
from output_artifacts.manifest_index import RUN_MANIFEST_INDEX_SCHEMA_VERSION
from output_artifacts.manifest_index import ManifestIndexRecorder
from output_artifacts.manifest_index import build_run_manifest_index
from output_artifacts.manifest_index import default_run_manifest_index_path
from output_artifacts.manifest_index import manifest_index_entry
from output_artifacts.manifest_index import manifest_index_rows
from output_artifacts.manifest_index import read_run_manifest_index
from output_artifacts.manifest_index import summarize_manifest_index_entries
from output_artifacts.manifest_index import write_run_manifest_index


class ManifestIndexTests(unittest.TestCase):
    def test_written_manifest_entry_uses_catalog_and_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir)
            manifest_path = run_root / "manifests" / "input_manifest_summary.json"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text("{}\n", encoding="utf-8")

            entry = manifest_index_entry(
                "input_manifest_summary",
                MANIFEST_STATUS_WRITTEN,
                manifest_path=manifest_path,
                run_root=run_root,
                manifest_schema_version="1",
            )

            self.assertEqual(entry.catalog_status, "cataloged")
            self.assertEqual(entry.manifest_path, "manifests/input_manifest_summary.json")
            self.assertTrue(entry.path_is_relative_to_run_root)
            self.assertTrue(entry.path_exists_at_index_write)
            self.assertEqual(entry.scope, "run_input")
            self.assertEqual(entry.producer, "input_data.dicom_manifest.write_input_manifest_files")

    def test_constructed_not_written_entry_does_not_require_path(self) -> None:
        entry = manifest_index_entry(
            "patient_batch_run_manifest",
            MANIFEST_STATUS_CONSTRUCTED_NOT_WRITTEN,
            manifest_schema_version="patient_batch_run_manifest_v1",
            metadata={"reason": "dry_run"},
        )

        self.assertEqual(entry.manifest_path, "")
        self.assertFalse(entry.path_exists_at_index_write)
        self.assertEqual(entry.metadata["reason"], "dry_run")

    def test_written_entry_without_path_fails(self) -> None:
        with self.assertRaises(ValueError):
            manifest_index_entry("input_manifest_summary", MANIFEST_STATUS_WRITTEN)

    def test_manifest_index_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir)
            manifest_path = run_root / "manifests" / "input_manifest_summary.json"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text("{}\n", encoding="utf-8")
            entry = manifest_index_entry(
                "input_manifest_summary",
                MANIFEST_STATUS_WRITTEN,
                manifest_path=manifest_path,
                run_root=run_root,
            )
            output_path = run_root / "manifests" / "run_manifest_index.json"

            written_path = write_run_manifest_index(
                (entry,),
                output_path,
                run_id="synthetic_run",
                run_root=run_root,
            )
            payload = read_run_manifest_index(written_path)
            rows = manifest_index_rows(payload)

            self.assertEqual(payload["schema_version"], RUN_MANIFEST_INDEX_SCHEMA_VERSION)
            self.assertEqual(payload["manifest_count"], 1)
            self.assertEqual(payload["summary"]["written_manifest_count"], 1)
            self.assertEqual(rows[0]["manifest_key"], "input_manifest_summary")

    def test_manifest_index_recorder_accumulates_and_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir)
            manifest_path = run_root / "patient_batch_run_manifest.json"
            manifest_path.write_text("{}\n", encoding="utf-8")
            recorder = ManifestIndexRecorder(run_root, run_id="synthetic_run")

            recorder.record_written_manifest("patient_batch_run_manifest", manifest_path)
            recorder.record_constructed_manifest("patient_scientific_context_manifest", notes="not emitted in this run")
            output_path = recorder.write()

            self.assertEqual(output_path, default_run_manifest_index_path(run_root))
            payload = read_run_manifest_index(output_path)
            self.assertEqual(payload["manifest_count"], 3)
            self.assertEqual(payload["summary"]["produced_status_counts"][MANIFEST_STATUS_WRITTEN], 2)
            self.assertEqual(
                payload["summary"]["produced_status_counts"][MANIFEST_STATUS_CONSTRUCTED_NOT_WRITTEN],
                1,
            )
            self_entry = next(entry for entry in payload["manifests"] if entry["manifest_key"] == "run_manifest_index")
            self.assertEqual(self_entry["manifest_path"], "manifests/run_manifest_index.json")
            self.assertTrue(self_entry["path_exists_at_index_write"])

    def test_summary_counts_unknown_contracts(self) -> None:
        entry = manifest_index_entry(
            "not_yet_cataloged_manifest",
            MANIFEST_STATUS_CONSTRUCTED_NOT_WRITTEN,
            scope="test_scope",
        )

        summary = summarize_manifest_index_entries((entry,))

        self.assertEqual(summary["catalog_status_counts"]["unknown_contract"], 1)
        self.assertEqual(summary["scope_counts"]["test_scope"], 1)

    def test_build_manifest_index_payload_contains_metadata(self) -> None:
        entry = manifest_index_entry("input_manifest_summary", MANIFEST_STATUS_CONSTRUCTED_NOT_WRITTEN)

        payload = build_run_manifest_index(
            (entry,),
            run_id="synthetic_run",
            run_root="/tmp/run_root",
            metadata={"retention_level": "context"},
        )

        self.assertEqual(payload["run_id"], "synthetic_run")
        self.assertEqual(payload["metadata"]["retention_level"], "context")
        self.assertEqual(payload["manifests"][0]["schema_version"], RUN_MANIFEST_INDEX_SCHEMA_VERSION)


if __name__ == "__main__":
    unittest.main()