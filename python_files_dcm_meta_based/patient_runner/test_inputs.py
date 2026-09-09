from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from .inputs import PatientInputPaths


class PatientInputPathsTests(unittest.TestCase):
    def test_case_manifest_row_parses_scalar_and_multi_file_roles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            core_paths = [root.joinpath(name) for name in ("struct.dcm", "dose.dcm", "plan.dcm")]
            for path in core_paths:
                path.touch()
            row = {
                "Patient UID (generated)": "P001",
                "RTSTRUCT path": core_paths[0].as_posix(),
                "RTDOSE path": core_paths[1].as_posix(),
                "RTPLAN path": core_paths[2].as_posix(),
                "US paths": "{} | {}".format(root.joinpath("us1.dcm"), root.joinpath("us2.dcm")),
                "MR T2 paths": "",
                "MR ADC paths": root.joinpath("adc.dcm").as_posix(),
            }

            inputs = PatientInputPaths.from_case_manifest_row(row)
            core_paths_present = inputs.core_paths_all_present

        self.assertEqual(inputs.patient_uid, "P001")
        self.assertEqual(inputs.us, (root.joinpath("us1.dcm"), root.joinpath("us2.dcm")))
        self.assertEqual(inputs.mr_t2, ())
        self.assertEqual(inputs.mr_adc, (root.joinpath("adc.dcm"),))
        self.assertTrue(core_paths_present)
        self.assertEqual(len(inputs.manifest_identity_sha256), 64)

    def test_manifest_identity_changes_when_role_assignment_changes(self) -> None:
        first = PatientInputPaths(patient_uid="P001", rtstruct=Path("first.dcm"))
        second = PatientInputPaths(patient_uid="P001", rtstruct=Path("second.dcm"))

        self.assertNotEqual(first.manifest_identity_sha256, second.manifest_identity_sha256)
        self.assertEqual(first.missing_core_roles, ("rtstruct", "rtdose", "rtplan"))

    def test_case_manifest_row_rejects_missing_role_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            PatientInputPaths.from_case_manifest_row({"Patient UID (generated)": "P001"})

    def test_serialized_input_identity_detects_tampered_paths(self) -> None:
        inputs = PatientInputPaths(patient_uid="P001", rtstruct=Path("first.dcm"))
        payload = inputs.to_dict()
        payload["rtstruct"] = "changed.dcm"

        with self.assertRaisesRegex(ValueError, "identity"):
            PatientInputPaths.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
