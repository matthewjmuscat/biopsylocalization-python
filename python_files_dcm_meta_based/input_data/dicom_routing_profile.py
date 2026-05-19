from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


ROUTING_PROFILE_SCHEMA_VERSION = 1
LEGACY_VARISEED_MIM_PROFILE_ID = "legacy_variseed_mim_v1"


@dataclass(frozen=True)
class DicomRoutingRule:
    rule_id: str
    selected_role: str
    priority: int
    match_all: Mapping[str, str]
    description: str
    fallback: bool = False
    notes: Sequence[str] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "selected_role": self.selected_role,
            "priority": self.priority,
            "match_all": dict(self.match_all),
            "description": self.description,
            "fallback": self.fallback,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class DicomRoutingProfile:
    schema_version: int
    profile_id: str
    display_name: str
    description: str
    dicom_fields: Mapping[str, Mapping[str, str]]
    patient_uid_components: Sequence[str]
    fraction_identifier: Mapping[str, Any]
    required_core_roles: Sequence[str]
    rules: Sequence[DicomRoutingRule]
    known_export_artifacts: Sequence[str]
    future_user_configurable_fields: Sequence[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "display_name": self.display_name,
            "description": self.description,
            "dicom_fields": {
                field_name: dict(field_metadata)
                for field_name, field_metadata in self.dicom_fields.items()
            },
            "patient_uid_components": list(self.patient_uid_components),
            "fraction_identifier": dict(self.fraction_identifier),
            "required_core_roles": list(self.required_core_roles),
            "rules": [rule.to_dict() for rule in self.rules],
            "known_export_artifacts": list(self.known_export_artifacts),
            "future_user_configurable_fields": list(self.future_user_configurable_fields),
        }

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "display_name": self.display_name,
            "num_rules": len(self.rules),
            "required_core_roles": list(self.required_core_roles),
        }


def build_legacy_variseed_mim_routing_profile(fraction_prefixes: Sequence[str]) -> DicomRoutingProfile:
    return DicomRoutingProfile(
        schema_version=ROUTING_PROFILE_SCHEMA_VERSION,
        profile_id=LEGACY_VARISEED_MIM_PROFILE_ID,
        display_name="Legacy Variseed/MIM DICOM routing",
        description=(
            "Current fixed DICOM input shape used by the biopsy localization pipeline. "
            "This profile documents the existing role-detection behavior; it does not "
            "change role assignment."
        ),
        dicom_fields={
            "Modality": {
                "keyword": "Modality",
                "tag": "(0008,0060)",
                "purpose": "Primary RTSTRUCT, RTDOSE, RTPLAN, US, and MR discriminator.",
            },
            "SeriesDescription": {
                "keyword": "SeriesDescription",
                "tag": "(0008,103E)",
                "purpose": "MR subtype discriminator for T2 and ADC series.",
            },
            "MRAcquisitionType": {
                "keyword": "MRAcquisitionType",
                "tag": "(0018,0023)",
                "purpose": "Legacy fallback discriminator for US files exported with MR modality.",
            },
            "PatientName": {
                "keyword": "PatientName",
                "tag": "(0010,0010)",
                "purpose": "First component of the legacy generated patient UID.",
            },
            "PatientID": {
                "keyword": "PatientID",
                "tag": "(0010,0020)",
                "purpose": "Second component of the legacy generated patient UID and current fraction parsing source.",
            },
            "SeriesInstanceUID": {
                "keyword": "SeriesInstanceUID",
                "tag": "(0020,000E)",
                "purpose": "DICOM-file manifest identity and audit field.",
            },
            "StudyInstanceUID": {
                "keyword": "StudyInstanceUID",
                "tag": "(0020,000D)",
                "purpose": "DICOM-file manifest identity and audit field.",
            },
            "SOPInstanceUID": {
                "keyword": "SOPInstanceUID",
                "tag": "(0008,0018)",
                "purpose": "DICOM-file manifest identity and audit field.",
            },
            "StudyDate": {
                "keyword": "StudyDate",
                "tag": "(0008,0020)",
                "purpose": "DICOM-file manifest audit field.",
            },
        },
        patient_uid_components=["PatientName", "PatientID"],
        fraction_identifier={
            "source": "PatientID",
            "method": "legacy_prefix_number_regex",
            "allowed_prefixes": list(fraction_prefixes),
            "notes": [
                "This is a legacy parser only; future profiles should support explicit fraction identifiers when available.",
            ],
        },
        required_core_roles=["RTSTRUCT", "RTDOSE", "RTPLAN"],
        rules=[
            DicomRoutingRule(
                rule_id="legacy_rtstruct_modality",
                selected_role="RTSTRUCT",
                priority=10,
                match_all={"Modality": "RTSTRUCT"},
                description="RT structure file when Modality is RTSTRUCT.",
            ),
            DicomRoutingRule(
                rule_id="legacy_rtdose_modality",
                selected_role="RTDOSE",
                priority=20,
                match_all={"Modality": "RTDOSE"},
                description="RT dose file when Modality is RTDOSE.",
            ),
            DicomRoutingRule(
                rule_id="legacy_rtplan_modality",
                selected_role="RTPLAN",
                priority=30,
                match_all={"Modality": "RTPLAN"},
                description="RT plan file when Modality is RTPLAN.",
            ),
            DicomRoutingRule(
                rule_id="legacy_us_modality",
                selected_role="US",
                priority=40,
                match_all={"Modality": "US"},
                description="Ultrasound file when Modality is US.",
            ),
            DicomRoutingRule(
                rule_id="legacy_mr_t2_series_description",
                selected_role="MR_T2",
                priority=50,
                match_all={"Modality": "MR", "SeriesDescription": "T2"},
                description="MR T2 file when Modality is MR and SeriesDescription is T2.",
            ),
            DicomRoutingRule(
                rule_id="legacy_mr_adc_series_description",
                selected_role="MR_ADC",
                priority=60,
                match_all={"Modality": "MR", "SeriesDescription": "ADC"},
                description="MR ADC file when Modality is MR and SeriesDescription is ADC.",
            ),
            DicomRoutingRule(
                rule_id="legacy_us_mr_empty_acquisition_type",
                selected_role="US",
                priority=70,
                match_all={"Modality": "MR", "MRAcquisitionType": ""},
                description="US fallback for Variseed/MIM exports where ultrasound files are labeled as MR.",
                fallback=True,
                notes=[
                    "This fallback is intentionally documented as a legacy export artifact.",
                    "It should become user-configurable in a future routing profile layer before supporting broader datasets.",
                ],
            ),
        ],
        known_export_artifacts=[
            "Some ultrasound files are exported with Modality equal to MR.",
            "The current dataset identifies those ultrasound files by an empty MRAcquisitionType after T2/ADC checks.",
            "MR T2 and MR ADC are currently identified by exact SeriesDescription values: T2 and ADC.",
        ],
        future_user_configurable_fields=[
            "role rule field selectors and expected values",
            "rule priority and fallback behavior",
            "fraction identifier source field and parser",
            "patient/case UID composition",
            "required versus optional role set",
            "whether role ambiguity is warning-only or fatal",
        ],
    )
