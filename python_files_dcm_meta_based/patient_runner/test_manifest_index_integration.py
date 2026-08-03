from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from output_artifacts.manifest_index import MANIFEST_STATUS_SKIPPED
from output_artifacts.manifest_index import MANIFEST_STATUS_WRITTEN
from output_artifacts.manifest_index import default_run_manifest_index_path
from output_artifacts.manifest_index import read_run_manifest_index

from .batch import run_patient_batch
from .contracts import LegacyCohortRuntimeState
from .contracts import LegacyRuntimeKeys
from .contracts import PatientBatchRunConfig
from .contracts import PatientRunConfig


class PatientRunnerManifestIndexIntegrationTests(unittest.TestCase):
    def test_patient_batch_writes_run_manifest_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir)
            legacy_keys = _legacy_keys()
            batch_result = run_patient_batch(
                _legacy_cohort_state(legacy_keys),
                PatientBatchRunConfig(
                    patient_config=PatientRunConfig(
                        output_root=run_root,
                        legacy_keys=legacy_keys,
                        run_id="synthetic_run",
                    ),
                    patient_uids=("P001",),
                ),
                stages=(),
            )

            index_payload = read_run_manifest_index(default_run_manifest_index_path(run_root))
            entries_by_key = {entry["manifest_key"]: entry for entry in index_payload["manifests"]}

            self.assertTrue(batch_result.succeeded)
            self.assertEqual(index_payload["run_id"], "synthetic_run")
            self.assertEqual(index_payload["manifest_count"], 3)
            self.assertEqual(index_payload["summary"]["produced_status_counts"][MANIFEST_STATUS_WRITTEN], 3)
            self.assertEqual(entries_by_key["patient_run_manifest"]["produced_status"], MANIFEST_STATUS_WRITTEN)
            self.assertEqual(entries_by_key["patient_run_manifest"]["patient_uid"], "P001")
            self.assertEqual(
                entries_by_key["patient_run_manifest"]["manifest_path"],
                "patients/P001/patient_run_manifest.json",
            )
            self.assertTrue(entries_by_key["patient_run_manifest"]["path_exists_at_index_write"])
            self.assertEqual(entries_by_key["patient_batch_run_manifest"]["produced_status"], MANIFEST_STATUS_WRITTEN)
            self.assertEqual(entries_by_key["patient_batch_run_manifest"]["manifest_path"], "patient_batch_run_manifest.json")
            self.assertEqual(entries_by_key["run_manifest_index"]["produced_status"], MANIFEST_STATUS_WRITTEN)
            self.assertEqual(entries_by_key["run_manifest_index"]["manifest_path"], "manifests/run_manifest_index.json")

    def test_patient_batch_index_records_disabled_manifest_writers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir)
            legacy_keys = _legacy_keys()
            run_patient_batch(
                _legacy_cohort_state(legacy_keys),
                PatientBatchRunConfig(
                    patient_config=PatientRunConfig(
                        output_root=run_root,
                        legacy_keys=legacy_keys,
                        run_id="synthetic_run",
                        write_patient_run_manifest=False,
                    ),
                    patient_uids=("P001",),
                    write_batch_run_manifest=False,
                ),
                stages=(),
            )

            index_payload = read_run_manifest_index(default_run_manifest_index_path(run_root))
            entries_by_key = {entry["manifest_key"]: entry for entry in index_payload["manifests"]}

            self.assertFalse(run_root.joinpath("patients", "P001", "patient_run_manifest.json").exists())
            self.assertFalse(run_root.joinpath("patient_batch_run_manifest.json").exists())
            self.assertEqual(index_payload["summary"]["produced_status_counts"][MANIFEST_STATUS_SKIPPED], 2)
            self.assertEqual(index_payload["summary"]["produced_status_counts"][MANIFEST_STATUS_WRITTEN], 1)
            self.assertEqual(entries_by_key["patient_run_manifest"]["produced_status"], MANIFEST_STATUS_SKIPPED)
            self.assertEqual(entries_by_key["patient_batch_run_manifest"]["produced_status"], MANIFEST_STATUS_SKIPPED)
            self.assertEqual(entries_by_key["run_manifest_index"]["produced_status"], MANIFEST_STATUS_WRITTEN)


def _legacy_keys() -> LegacyRuntimeKeys:
    return LegacyRuntimeKeys(
        all_ref_key="all_ref",
        bx_ref="biopsies",
        by_patient_key="by_patient",
        global_key="global",
        global_num_cases_key="num_cases",
    )


def _legacy_cohort_state(legacy_keys: LegacyRuntimeKeys) -> LegacyCohortRuntimeState:
    return LegacyCohortRuntimeState(
        master_structure_reference_dict={"P001": {legacy_keys.bx_ref: []}},
        master_structure_info_dict={},
        legacy_keys=legacy_keys,
    )


if __name__ == "__main__":
    unittest.main()