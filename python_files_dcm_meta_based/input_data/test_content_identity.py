"""Synthetic byte identity and worker sealing checks; no DICOM decoding."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from config.snapshots import canonical_sha256
from input_data.content_identity import capture_patient_input_content, verify_patient_input_content, validate_patient_input_content
from patient_runner.inputs import PatientInputPaths
from patient_runner.process_runner import _with_input_content_verification
from patient_runner.runner import PatientStage, run_patient_stages
from patient_runner.contracts import PatientStageResult


class ContentIdentityTests(unittest.TestCase):
    def test_all_roles_and_same_size_same_mtime_byte_changes(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "synthetic.dcm"
            path.write_bytes(b"abcd")
            inputs = PatientInputPaths("synthetic", rtstruct=path, mr_adc=(path,))
            ledger = capture_patient_input_content(inputs)
            self.assertEqual(ledger["roles"]["mr_adc"][0]["size_bytes"], 4)
            self.assertEqual(ledger["roles"]["mr_t2"], [])
            verify_patient_input_content(ledger, inputs)
            timestamp = path.stat().st_mtime_ns
            path.write_bytes(b"dcba")
            os.utime(path, ns=(timestamp, timestamp))
            with self.assertRaisesRegex(ValueError, "changed"):
                verify_patient_input_content(ledger, inputs)

    def test_declared_missing_adc_fails_and_ledger_role_forgery_fails(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "absent.dcm"
            with self.assertRaises(FileNotFoundError):
                capture_patient_input_content(PatientInputPaths("synthetic", mr_adc=(path,)))
            path.write_bytes(b"test")
            inputs = PatientInputPaths("synthetic", mr_adc=(path,))
            ledger = capture_patient_input_content(inputs)
            ledger["roles"]["mr_adc"] = []
            ledger.pop("identity_sha256")
            ledger["identity_sha256"] = canonical_sha256(ledger)
            with self.assertRaises(ValueError):
                validate_patient_input_content(ledger, inputs)

    def test_completed_artifact_stage_cannot_seal_success_after_input_drift(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "synthetic.dcm"
            path.write_bytes(b"before")
            inputs = PatientInputPaths("synthetic", rtstruct=path)
            job = SimpleNamespace(patient_inputs=inputs, metadata={"input_content_identity": capture_patient_input_content(inputs)})
            runtime = SimpleNamespace(metadata={}, patient_uid="synthetic")

            def export(state, config):
                path.write_bytes(b"after")
                return PatientStageResult.success("patient_artifact_writing")

            stages = _with_input_content_verification((PatientStage("patient_artifact_writing", export),), job)
            results = run_patient_stages(runtime, SimpleNamespace(raise_on_stage_error=False, stop_on_stage_error=True), stages)
            self.assertFalse(results[-1].succeeded)
            self.assertNotIn("input_content_verified_after", runtime.metadata)


if __name__ == "__main__":
    unittest.main()
