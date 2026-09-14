"""User-operated, patient-derived unequal-prescription DICOM fixture creation.

Copies one explicitly selected F2 job's inputs into two synthetic subjects. It
changes TARGET prescription and case/object identities, not pixel data or geometry.
This is a local validation tool, not discovery, de-identification, or a DICOM export
conformance validator. Patient execution and source files are never modified.
"""

import json
from pathlib import Path
import re

import pydicom
from pydicom.uid import generate_uid

from input_data.content_identity import capture_patient_input_content, verify_patient_input_content
from input_data.dicom_manifest import write_input_manifest_files
from patient_runner.process_runner import load_patient_worker_job


_IDENTIFIER_FIELDS = ("SOPInstanceUID", "StudyInstanceUID", "SeriesInstanceUID", "FrameOfReferenceUID")


def _logical_identity(dataset):
    if not dataset.get("SOPInstanceUID") or not dataset.get("SOPClassUID"):
        raise ValueError("fixture source lacks logical DICOM identity")
    if (str(dataset.file_meta.get("MediaStorageSOPInstanceUID", "")) != str(dataset.SOPInstanceUID)
            or str(dataset.file_meta.get("MediaStorageSOPClassUID", "")) != str(dataset.SOPClassUID)):
        raise ValueError("fixture source file meta disagrees with logical DICOM identity")
    return {key: str(dataset.get(key, "")) for key in ("SOPInstanceUID", "SOPClassUID", "StudyInstanceUID", "SeriesInstanceUID")}


def _remap_identifiers(dataset, replacements):
    for element in dataset.iterall():
        if element.VR == "UI":
            if element.VM == 1:
                element.value = replacements.get(str(element.value), element.value)
            elif element.VM > 1:
                element.value = [replacements.get(str(value), value) for value in element.value]
    dataset.file_meta.MediaStorageSOPInstanceUID = dataset.SOPInstanceUID


def build_synthetic_dose_fixture(*, source_job: Path, output_dir: Path,
                                 case_prefix: str = "SYNTHETIC_DOSE") -> dict:
    """Write fresh 10.0/13.5 Gy subjects, manifests, and a source/transform ledger.

    A source ledger, when present, must still match. All source bytes are verified
    again after copying. Logical objects with identical bytes are copied once and
    retain path aliases in the fixture ledger; conflicting copies fail. This local
    fixture policy does not alter production discovery or strict job identity.
    Existing destinations fail, including empty directories; partial failures
    remain incomplete and must not be reused. All copied metadata stays private.
    """
    if not re.fullmatch(r"SYNTHETIC_[A-Z0-9_]{1,44}", case_prefix):
        raise ValueError("case_prefix must start SYNTHETIC_ and use at most 44 further uppercase letters, digits or underscores")
    job = load_patient_worker_job(source_job)
    inputs = job.patient_inputs
    if inputs is None:
        raise ValueError("fixture requires explicit patient inputs")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError("synthetic fixture destination must be absent")
    if "input_content_identity" in job.metadata:
        verify_patient_input_content(job.metadata["input_content_identity"], inputs)
    ledger = capture_patient_input_content(inputs)
    objects, roles, identity_uids = {}, {}, set()
    for role, records in ledger["roles"].items():
        roles[role] = []
        for record in records:
            header = pydicom.dcmread(record["resolved_path"], stop_before_pixels=True)
            if str(header.get("PatientID", "")) != "F2":
                raise ValueError("dose fixture requires a consistently labelled F2 source case")
            if f"{header.PatientName} ({header.PatientID})" != inputs.patient_uid:
                raise ValueError("source DICOM case identity differs from the selected job")
            identity = _logical_identity(header)
            uid = identity["SOPInstanceUID"]
            existing = objects.get(uid)
            content = {key: record[key] for key in ("sha256", "size_bytes")}
            if existing is not None and (existing["content"] != content or existing["dicom"] != identity):
                raise ValueError("conflicting copies of logical DICOM object: " + uid)
            if existing is None:
                objects[uid] = {"dicom": identity, "content": content, "locations": [], "header": header}
            objects[uid]["locations"].append({key: record[key] for key in ("path", "resolved_path")})
            if uid not in roles[role]:
                roles[role].append(uid)
            for element in header.iterall():
                if element.keyword in _IDENTIFIER_FIELDS and element.value:
                    identity_uids.add(str(element.value))
    for role in ("rtstruct", "rtplan", "rtdose"):
        if len(roles[role]) != 1:
            raise ValueError("fixture needs exactly one logical object for " + role)
        if objects[roles[role][0]]["header"].Modality != role.upper():
            raise ValueError("source role and DICOM modality disagree: " + role)
    plan_uid = roles["rtplan"][0]
    plan = objects[plan_uid]["header"]
    targets = [item for item in plan.get("DoseReferenceSequence", ()) if item.get("DoseReferenceType") == "TARGET"]
    if len(targets) != 1 or "TargetPrescriptionDose" not in targets[0]:
        raise ValueError("fixture requires exactly one TARGET prescription in RTPLAN")
    dose = objects[roles["rtdose"][0]]["header"]
    if not any(str(item.get("ReferencedSOPInstanceUID", "")) == plan_uid
               for item in dose.get("ReferencedRTPlanSequence", ())):
        raise ValueError("RTDOSE must reference the selected source RTPLAN")
    destination.mkdir(parents=True, exist_ok=False)
    role_paths = {role: {} for role in roles}
    all_paths, subjects = [], []
    for token, prescription in (("10", 10.0), ("13_5", 13.5)):
        name = case_prefix + "_" + token
        case_uid = name + " (F2)"
        replacements = {uid: generate_uid() for uid in sorted(identity_uids)}
        copies = {}
        for index, (uid, source) in enumerate(sorted(objects.items()), 1):
            path = Path(source["locations"][0]["resolved_path"])
            copied = pydicom.dcmread(path)
            _remap_identifiers(copied, replacements)
            copied.PatientName, copied.PatientID = name, "F2"
            copied.PatientComments = "SYNTHETIC VALIDATION FIXTURE; NOT A CLINICAL SUBJECT"
            if uid == plan_uid:
                for item in copied.DoseReferenceSequence:
                    if item.DoseReferenceType == "TARGET":
                        item.TargetPrescriptionDose = prescription
            target = destination / name / f"{index:04d}_{copied.Modality}.dcm"
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                copied.save_as(stream, enforce_file_format=True)
            copies[uid] = target
            all_paths.append(target)
        for role, uids in roles.items():
            paths = [copies[uid] for uid in uids]
            if paths:
                role_paths[role][case_uid] = paths[0] if role in ("rtstruct", "rtplan", "rtdose") else paths
        subjects.append({"patient_uid": case_uid, "target_prescription_gy": prescription, "uid_map": replacements})
    verify_patient_input_content(ledger, inputs)
    manifests = write_input_manifest_files(output_dir=destination, dicom_paths=all_paths,
        rtstruct_dcms_dict=role_paths["rtstruct"], rtdose_dcms_dict=role_paths["rtdose"],
        rtplan_dcms_dict=role_paths["rtplan"], us_dcms_dict=role_paths["us"],
        mr_t2_dcms_dict=role_paths["mr_t2"], mr_adc_dcms_dict=role_paths["mr_adc"], fraction_prefixes=("F",))
    report = {"schema_version": "synthetic_dose_fixture_v1", "complete": True,
        "source_job": str(Path(source_job).resolve()), "source_input_content": ledger,
        "source_objects": [{key: value for key, value in item.items() if key != "header"} for item in objects.values()],
        "subjects": subjects, "input_case_manifest": str(manifests.case_manifest_path),
        "scope": "Patient-derived private fixtures; preserved pixels/geometry; remapped known copied-object identities/references. External references remain external; not de-identified."}
    with (destination / "synthetic_fixture.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return report
