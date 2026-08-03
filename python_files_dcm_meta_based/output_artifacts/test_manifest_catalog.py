from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from output_artifacts.manifest_catalog import MANIFEST_CATALOG_SCHEMA_VERSION
from output_artifacts.manifest_catalog import ManifestContract
from output_artifacts.manifest_catalog import iter_manifest_contracts
from output_artifacts.manifest_catalog import manifest_catalog_rows
from output_artifacts.manifest_catalog import manifest_contracts_by_key
from output_artifacts.manifest_catalog import render_manifest_catalog_markdown
from output_artifacts.manifest_catalog import summarize_manifest_catalog
from output_artifacts.manifest_catalog import write_manifest_catalog


class ManifestCatalogTests(unittest.TestCase):
    def test_catalog_keys_are_unique(self) -> None:
        contracts_by_key = manifest_contracts_by_key()

        self.assertEqual(len(contracts_by_key), len(iter_manifest_contracts()))

    def test_catalog_includes_core_manifest_surfaces(self) -> None:
        contracts_by_key = manifest_contracts_by_key()

        expected_keys = {
            "input_manifest_summary",
            "input_case_manifest",
            "input_dicom_manifest",
            "patient_run_manifest",
            "patient_batch_run_manifest",
            "dose_nn_render_scene_artifact_manifest",
            "patient_scientific_context_manifest",
        }
        self.assertTrue(expected_keys.issubset(contracts_by_key))
        self.assertEqual(
            contracts_by_key["dose_nn_render_scene_artifact_manifest"].artifact_data_class,
            "scene_manifest",
        )
        self.assertEqual(
            contracts_by_key["patient_scientific_context_manifest"].lifecycle_status,
            "planned",
        )

    def test_catalog_rows_are_report_ready(self) -> None:
        rows = manifest_catalog_rows()

        self.assertGreater(len(rows), 0)
        self.assertTrue(all(row["schema_version"] == MANIFEST_CATALOG_SCHEMA_VERSION for row in rows))
        self.assertTrue(all(isinstance(row["tracks"], str) and row["tracks"] for row in rows))
        self.assertTrue(all(" | " not in row["manifest_key"] for row in rows))

    def test_duplicate_catalog_keys_fail_closed(self) -> None:
        contract = ManifestContract(
            manifest_key="duplicate_key",
            title="Synthetic manifest",
            scope="test",
            artifact_data_class="manifest",
            lifecycle_status="test",
            default_relative_paths=("manifest.json",),
            payload_format="json",
            schema_version_source="test.SCHEMA_VERSION",
            producer="test.write_manifest",
            purpose="Exercise duplicate-key validation.",
            tracks=("synthetic field",),
        )

        with self.assertRaises(ValueError):
            manifest_contracts_by_key((contract, contract))

    def test_summary_counts_catalog_statuses(self) -> None:
        summary = summarize_manifest_catalog()

        self.assertEqual(summary["schema_version"], MANIFEST_CATALOG_SCHEMA_VERSION)
        self.assertEqual(summary["manifest_contract_count"], len(iter_manifest_contracts()))
        self.assertGreater(summary["lifecycle_status_counts"].get("current_durable", 0), 0)
        self.assertGreater(summary["lifecycle_status_counts"].get("planned", 0), 0)

    def test_markdown_renders_catalog_table(self) -> None:
        markdown = render_manifest_catalog_markdown()

        self.assertIn("# Manifest Catalog", markdown)
        self.assertIn("dose_nn_render_scene_artifact_manifest", markdown)
        self.assertIn("| Manifest key | Scope | Status | Format | Producer | Tracks |", markdown)

    def test_write_manifest_catalog_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog_path, summary_path, markdown_path = write_manifest_catalog(Path(temp_dir))

            self.assertTrue(catalog_path.is_file())
            self.assertTrue(summary_path.is_file())
            self.assertTrue(markdown_path.is_file())
            with catalog_path.open("r", encoding="utf-8", newline="") as file_obj:
                rows = list(csv.DictReader(file_obj))
            with summary_path.open("r", encoding="utf-8") as file_obj:
                summary = json.load(file_obj)
            self.assertEqual(len(rows), len(iter_manifest_contracts()))
            self.assertEqual(summary["schema_version"], MANIFEST_CATALOG_SCHEMA_VERSION)
            self.assertIn("patient_batch_run_manifest", markdown_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()