"""Test local copy construction using wholly synthetic DICOM, never patient data."""

from dataclasses import replace
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, RTDoseStorage, RTPlanStorage, RTStructureSetStorage, UltrasoundImageStorage, generate_uid

from input_data.content_identity import capture_patient_input_content
from patient_runner.process_runner import write_patient_worker_job_packets
from preprocessing.structure_reference_bootstrap import attach_patient_plan_reference_from_path
from validation.synthetic_dose_fixture import build_synthetic_dose_fixture
from validation.test_anatomical_independence import _base
from validation.test_legacy_dose_order import _legacy_fixture, _patient
from validation.legacy_dose_order import capture_legacy_dose_order, compare_legacy_dose_orders


def _source(root):
    plan = _base(root, ("Source (F2)",))
    job = plan.worker_jobs[0]
    study, frame = generate_uid(), generate_uid()
    instances = {role: generate_uid() for role in ("rtstruct", "rtplan", "rtdose")}
    for role, sop_class in (("rtstruct", RTStructureSetStorage), ("rtplan", RTPlanStorage), ("rtdose", RTDoseStorage)):
        path = getattr(job.patient_inputs, role)
        meta = FileMetaDataset()
        meta.TransferSyntaxUID, meta.MediaStorageSOPClassUID = ExplicitVRLittleEndian, sop_class
        meta.MediaStorageSOPInstanceUID = instances[role]
        dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
        dataset.SOPInstanceUID, dataset.SOPClassUID = instances[role], sop_class
        dataset.StudyInstanceUID, dataset.SeriesInstanceUID, dataset.FrameOfReferenceUID = study, generate_uid(), frame
        dataset.PatientName, dataset.PatientID, dataset.Modality = "Source", "F2", role.upper()
        dataset.StudyDate = "20000101"
        if role == "rtplan":
            target = Dataset()
            target.DoseReferenceType, target.TargetPrescriptionDose = "TARGET", 13.5
            dataset.DoseReferenceSequence = [target]
            reference = Dataset()
            reference.ReferencedSOPClassUID, reference.ReferencedSOPInstanceUID = RTStructureSetStorage, instances["rtstruct"]
            dataset.ReferencedStructureSetSequence = [reference]
        if role == "rtdose":
            reference = Dataset()
            reference.ReferencedSOPClassUID, reference.ReferencedSOPInstanceUID = RTPlanStorage, instances["rtplan"]
            dataset.ReferencedRTPlanSequence = [reference]
            dataset.Rows, dataset.Columns, dataset.NumberOfFrames = 2, 2, 2
            dataset.BitsAllocated, dataset.BitsStored, dataset.HighBit = 16, 16, 15
            dataset.PixelRepresentation, dataset.SamplesPerPixel = 0, 1
            dataset.PhotometricInterpretation = "MONOCHROME2"
            dataset.PixelData = np.arange(8, dtype=np.uint16).tobytes()
        dataset.save_as(path, enforce_file_format=True)
    job = replace(job, metadata={**job.metadata, "input_content_identity": capture_patient_input_content(job.patient_inputs)})
    return write_patient_worker_job_packets(replace(plan, worker_jobs=(job,)))[0], job


class SyntheticDoseFixtureTests(unittest.TestCase):
    def test_source_aliases_collapse_but_conflicting_bytes_under_same_sop_uid_fail(self):
        for conflict in (False, True):
            with self.subTest(conflict=conflict), TemporaryDirectory() as directory:
                root = Path(directory)
                source_path, job = _source(root)
                dataset = pydicom.dcmread(job.patient_inputs.rtstruct)
                dataset.Modality = "US"
                dataset.SOPClassUID = dataset.file_meta.MediaStorageSOPClassUID = UltrasoundImageStorage
                dataset.SOPInstanceUID = dataset.file_meta.MediaStorageSOPInstanceUID = generate_uid()
                first, second = root / "first.dcm", root / "alias.dcm"
                dataset.save_as(first, enforce_file_format=True)
                second.write_bytes(first.read_bytes())
                if conflict:
                    dataset.PatientComments = "Different content, same object UID"
                    dataset.save_as(second, enforce_file_format=True)
                inputs = replace(job.patient_inputs, us=(first, second))
                job = replace(job, patient_inputs=inputs, metadata={**job.metadata,
                    "input_content_identity": capture_patient_input_content(inputs)})
                source_path.write_text(json.dumps(job.as_mapping()))
                if conflict:
                    with self.assertRaisesRegex(ValueError, "conflicting copies"):
                        build_synthetic_dose_fixture(source_job=source_path, output_dir=root / "copies")
                    self.assertFalse((root / "copies").exists())
                else:
                    report = build_synthetic_dose_fixture(source_job=source_path, output_dir=root / "copies",
                                                          case_prefix="SYNTHETIC_ALIAS_TEST")
                    self.assertTrue(all(subject["patient_uid"].startswith("SYNTHETIC_ALIAS_TEST_") for subject in report["subjects"]))
                    self.assertEqual(len(report["source_objects"]), 4)
                    self.assertEqual(len(list((root / "copies").glob("**/*.dcm"))), 8)
                    aliases = [obj for obj in report["source_objects"] if len(obj["locations"]) == 2]
                    self.assertEqual(len(aliases), 1)

    def test_copy_identity_references_pixels_manifests_and_real_threshold_parser(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path, job = _source(root)
            original = {role: getattr(job.patient_inputs, role).read_bytes() for role in ("rtstruct", "rtplan", "rtdose")}
            result = build_synthetic_dose_fixture(source_job=source_path, output_dir=root / "synthetic set")
            with Path(result["input_case_manifest"]).open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual(len(list((root / "synthetic set").glob("**/*.dcm"))), 6)
            observed, seen_uids = {}, set()
            for row in rows:
                uid = row["Patient UID (generated)"]
                dose = pydicom.dcmread(row["RTDOSE path"])
                plan = pydicom.dcmread(row["RTPLAN path"])
                structure = pydicom.dcmread(row["RTSTRUCT path"])
                self.assertTrue(uid.startswith("SYNTHETIC_DOSE_"))
                self.assertEqual(dose.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID, plan.SOPInstanceUID)
                self.assertEqual(plan.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID, structure.SOPInstanceUID)
                self.assertEqual(dose.PixelData, pydicom.dcmread(job.patient_inputs.rtdose).PixelData)
                for dataset in (plan, dose, structure):
                    self.assertEqual(dataset.file_meta.MediaStorageSOPInstanceUID, dataset.SOPInstanceUID)
                    self.assertNotIn(dataset.SOPInstanceUID, seen_uids)
                    seen_uids.add(dataset.SOPInstanceUID)
                parsed = {}
                attach_patient_plan_reference_from_path(parsed, patient_uid=uid, plan_item_path=row["RTPLAN path"], pln_ref="Plan")
                observed[uid] = float(parsed["Plan"]["Prescription doses dict"]["TARGET"])
            self.assertEqual(sorted(observed.values()), [10., 13.5])
            for role, content in original.items():
                self.assertEqual(getattr(job.patient_inputs, role).read_bytes(), content)
            for record in result["source_objects"]:
                self.assertEqual(set(record), {"dicom", "content", "locations"})
                self.assertEqual(record["content"]["sha256"], hashlib.sha256(Path(record["locations"][0]["path"]).read_bytes()).hexdigest())
            with _legacy_fixture() as legacy:
                config = legacy.DoseGridProcessingConfig("Dose", "Plan", None, 0, False, False)
                order = tuple(observed)
                for name, uids in (("forward", order), ("reverse", order[::-1])):
                    capture_legacy_dose_order(patient_uids=uids, load_patient=lambda uid: _patient(observed[uid]),
                                             config=config, output_dir=root / name)
                comparison = compare_legacy_dose_orders(root / "forward", root / "reverse")
                self.assertTrue(all(row["classification"] == "dependence_observed" for row in comparison["patients"]))

    def test_existing_destination_and_changed_source_fail_closed(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path, job = _source(root)
            with self.assertRaises(FileExistsError):
                build_synthetic_dose_fixture(source_job=source_path, output_dir=root)
            plan = pydicom.dcmread(job.patient_inputs.rtplan)
            plan.DoseReferenceSequence[0].TargetPrescriptionDose = 11.
            plan.save_as(job.patient_inputs.rtplan, enforce_file_format=True)
            with self.assertRaisesRegex(ValueError, "content"):
                build_synthetic_dose_fixture(source_job=source_path, output_dir=root / "invalid")
            self.assertFalse((root / "invalid").exists())
