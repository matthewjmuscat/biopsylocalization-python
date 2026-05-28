"""Patient-local structure reference/bootstrap helpers.

These functions are additive patient-runner surfaces for the dictionary shell
currently built inside ``structure_referencer(...)``. The legacy cohort function
remains the oracle and is not routed through this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pydicom

import misc_tools
from biopsy_optimizer.v2.live_integration import (
    TARGET_DIL_OPTIMIZER_V2_RANKED_DF_KEY,
    TARGET_DIL_OPTIMIZER_V2_SUMMARY_DF_KEY,
)
from legacy_data_keys import legacy_data_keys
from presentation import ProgressEvent
from presentation import ProgressSink
from presentation import coerce_progress_sink


StructureRecord = dict[str, Any]
StructureInfoRecord = dict[str, Any]
LEGACY_MASTER_INFO_KEYS = legacy_data_keys.master_info
LEGACY_PATIENT_REFERENCE_KEYS = legacy_data_keys.patient_reference
LEGACY_STRUCTURE_RECORD_KEYS = legacy_data_keys.structure_record
LEGACY_STRUCTURE_INFO_KEYS = legacy_data_keys.structure_info
LEGACY_PATIENT_ALL_REFERENCE_KEYS = legacy_data_keys.patient_all_reference
LEGACY_BIOPSY_RUNTIME_KEYS = legacy_data_keys.biopsy_runtime


@dataclass(frozen=True, slots=True)
class PatientStructureReferenceKeys:
    """Legacy key names used by the patient structure reference boundary."""

    biopsy_ref: str
    oar_ref: str
    dil_ref: str
    rectum_ref: str
    urethra_ref: str
    all_ref_key: str

    def __post_init__(self) -> None:
        for field_name in (
            "biopsy_ref",
            "oar_ref",
            "dil_ref",
            "rectum_ref",
            "urethra_ref",
            "all_ref_key",
        ):
            field_value = str(getattr(self, field_name)).strip()
            if field_value == "":
                raise ValueError(f"{field_name} cannot be empty")
            object.__setattr__(self, field_name, field_value)

    @classmethod
    def from_legacy_refs(cls,
                         *,
                         st_ref_list: Sequence[str],
                         all_ref_key: str) -> "PatientStructureReferenceKeys":
        if len(st_ref_list) < 5:
            raise ValueError("st_ref_list must include biopsy, OAR, DIL, rectum, and urethra refs")
        return cls(
            biopsy_ref=st_ref_list[0],
            oar_ref=st_ref_list[1],
            dil_ref=st_ref_list[2],
            rectum_ref=st_ref_list[3],
            urethra_ref=st_ref_list[4],
            all_ref_key=all_ref_key,
        )

    def as_legacy_st_ref_list(self) -> list[str]:
        return [
            self.biopsy_ref,
            self.oar_ref,
            self.dil_ref,
            self.rectum_ref,
            self.urethra_ref,
        ]


@dataclass(frozen=True, slots=True)
class PatientStructureReferenceState:
    """Typed patient-local boundary around the legacy structure dictionary."""

    keys: PatientStructureReferenceKeys
    patient_uid: str
    patient_id_from_dicom: str
    patient_name: str
    fraction_number: Any
    biopsies: Sequence[StructureRecord]
    oars: Sequence[StructureRecord]
    dils: Sequence[StructureRecord]
    rectums: Sequence[StructureRecord]
    urethras: Sequence[StructureRecord]
    all_reference: Mapping[str, Any]
    ready_to_plot_data: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.keys, PatientStructureReferenceKeys):
            raise TypeError("keys must be a PatientStructureReferenceKeys instance")
        object.__setattr__(self, "patient_uid", str(self.patient_uid))
        object.__setattr__(self, "patient_id_from_dicom", str(self.patient_id_from_dicom))
        object.__setattr__(self, "patient_name", str(self.patient_name))
        object.__setattr__(self, "biopsies", list(self.biopsies))
        object.__setattr__(self, "oars", list(self.oars))
        object.__setattr__(self, "dils", list(self.dils))
        object.__setattr__(self, "rectums", list(self.rectums))
        object.__setattr__(self, "urethras", list(self.urethras))
        object.__setattr__(self, "all_reference", dict(self.all_reference))

    def to_legacy_dict(self) -> dict[str, Any]:
        return {
            LEGACY_PATIENT_REFERENCE_KEYS.patient_uid_generated_key: self.patient_uid,
            LEGACY_PATIENT_REFERENCE_KEYS.patient_id_from_dicom_key: self.patient_id_from_dicom,
            LEGACY_PATIENT_REFERENCE_KEYS.patient_name_key: self.patient_name,
            LEGACY_PATIENT_REFERENCE_KEYS.fraction_number_key: self.fraction_number,
            self.keys.biopsy_ref: list(self.biopsies),
            self.keys.oar_ref: list(self.oars),
            self.keys.dil_ref: list(self.dils),
            self.keys.rectum_ref: list(self.rectums),
            self.keys.urethra_ref: list(self.urethras),
            self.keys.all_ref_key: dict(self.all_reference),
            LEGACY_PATIENT_REFERENCE_KEYS.ready_to_plot_data_list_key: self.ready_to_plot_data,
        }

    @classmethod
    def from_legacy_dict(cls,
                         patient_reference_dict: Mapping[str, Any],
                         *,
                         keys: PatientStructureReferenceKeys) -> "PatientStructureReferenceState":
        return cls(
            keys=keys,
            patient_uid=patient_reference_dict[LEGACY_PATIENT_REFERENCE_KEYS.patient_uid_generated_key],
            patient_id_from_dicom=patient_reference_dict[LEGACY_PATIENT_REFERENCE_KEYS.patient_id_from_dicom_key],
            patient_name=patient_reference_dict[LEGACY_PATIENT_REFERENCE_KEYS.patient_name_key],
            fraction_number=patient_reference_dict[LEGACY_PATIENT_REFERENCE_KEYS.fraction_number_key],
            biopsies=patient_reference_dict.get(keys.biopsy_ref, ()),
            oars=patient_reference_dict.get(keys.oar_ref, ()),
            dils=patient_reference_dict.get(keys.dil_ref, ()),
            rectums=patient_reference_dict.get(keys.rectum_ref, ()),
            urethras=patient_reference_dict.get(keys.urethra_ref, ()),
            all_reference=patient_reference_dict.get(keys.all_ref_key, {}),
            ready_to_plot_data=patient_reference_dict.get(
                LEGACY_PATIENT_REFERENCE_KEYS.ready_to_plot_data_list_key
            ),
        )


@dataclass(frozen=True, slots=True)
class PatientStructureInfoState:
    """Typed patient-local boundary around one patient's structure counts."""

    keys: PatientStructureReferenceKeys
    patient_uid: str
    patient_id_from_dicom: str
    patient_name: str
    fraction_number: Any
    biopsy_info: StructureInfoRecord
    oar_info: StructureInfoRecord
    dil_info: StructureInfoRecord
    rectum_info: StructureInfoRecord
    urethra_info: StructureInfoRecord
    all_structures_info: StructureInfoRecord

    def __post_init__(self) -> None:
        if not isinstance(self.keys, PatientStructureReferenceKeys):
            raise TypeError("keys must be a PatientStructureReferenceKeys instance")
        object.__setattr__(self, "patient_uid", str(self.patient_uid))
        object.__setattr__(self, "patient_id_from_dicom", str(self.patient_id_from_dicom))
        object.__setattr__(self, "patient_name", str(self.patient_name))
        object.__setattr__(self, "biopsy_info", dict(self.biopsy_info))
        object.__setattr__(self, "oar_info", dict(self.oar_info))
        object.__setattr__(self, "dil_info", dict(self.dil_info))
        object.__setattr__(self, "rectum_info", dict(self.rectum_info))
        object.__setattr__(self, "urethra_info", dict(self.urethra_info))
        object.__setattr__(self, "all_structures_info", dict(self.all_structures_info))

    def to_legacy_dict(self) -> dict[str, Any]:
        return {
            LEGACY_PATIENT_REFERENCE_KEYS.patient_uid_generated_key: self.patient_uid,
            LEGACY_PATIENT_REFERENCE_KEYS.patient_id_from_dicom_key: self.patient_id_from_dicom,
            LEGACY_PATIENT_REFERENCE_KEYS.patient_name_key: self.patient_name,
            LEGACY_PATIENT_REFERENCE_KEYS.fraction_number_key: self.fraction_number,
            self.keys.biopsy_ref: dict(self.biopsy_info),
            self.keys.oar_ref: dict(self.oar_info),
            self.keys.dil_ref: dict(self.dil_info),
            self.keys.rectum_ref: dict(self.rectum_info),
            self.keys.urethra_ref: dict(self.urethra_info),
            self.keys.all_ref_key: dict(self.all_structures_info),
        }

    @classmethod
    def from_legacy_dict(cls,
                         patient_info_dict: Mapping[str, Any],
                         *,
                         keys: PatientStructureReferenceKeys) -> "PatientStructureInfoState":
        return cls(
            keys=keys,
            patient_uid=patient_info_dict[LEGACY_PATIENT_REFERENCE_KEYS.patient_uid_generated_key],
            patient_id_from_dicom=patient_info_dict[LEGACY_PATIENT_REFERENCE_KEYS.patient_id_from_dicom_key],
            patient_name=patient_info_dict[LEGACY_PATIENT_REFERENCE_KEYS.patient_name_key],
            fraction_number=patient_info_dict[LEGACY_PATIENT_REFERENCE_KEYS.fraction_number_key],
            biopsy_info=patient_info_dict.get(keys.biopsy_ref, {}),
            oar_info=patient_info_dict.get(keys.oar_ref, {}),
            dil_info=patient_info_dict.get(keys.dil_ref, {}),
            rectum_info=patient_info_dict.get(keys.rectum_ref, {}),
            urethra_info=patient_info_dict.get(keys.urethra_ref, {}),
            all_structures_info=patient_info_dict.get(keys.all_ref_key, {}),
        )


@dataclass(frozen=True, slots=True)
class PatientStructureReferenceBootstrapFragment:
    """One patient's typed bootstrap state plus legacy-shaped dictionaries."""

    patient_uid: str
    patient_reference_dict: dict[str, Any]
    patient_info_dict: dict[str, Any]
    patient_structure_reference: PatientStructureReferenceState | None = None
    patient_structure_info: PatientStructureInfoState | None = None
    messages: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "patient_uid", str(self.patient_uid))
        object.__setattr__(self, "patient_reference_dict", dict(self.patient_reference_dict))
        object.__setattr__(self, "patient_info_dict", dict(self.patient_info_dict))
        object.__setattr__(self, "messages", tuple(str(message) for message in self.messages))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


def _patient_dicom_value(structure_item: Any, tag: tuple[int, int]) -> str:
    return str(structure_item[tag].value)


def _emit(progress_sink: ProgressSink,
          patient_uid: str,
          message: str,
          **details: Any) -> None:
    progress_sink.emit(
        ProgressEvent(
            "structure_reference_bootstrap.notice",
            message=message,
            patient_uid=patient_uid,
            stage_name="structure_reference_bootstrap",
            details=details,
        )
    )


def _matching_rois(structure_item: Any, contour_names: Sequence[str]) -> list[Any]:
    return [
        roi
        for roi in structure_item.StructureSetROISequence
        if any(str(contour).lower() in roi.ROIName.lower() for contour in contour_names)
    ]


def _filter_removed_rois(rois: Sequence[Any],
                         removal_names: Sequence[str],
                         *,
                         patient_uid: str,
                         removal_label: str,
                         progress_sink: ProgressSink) -> list[Any]:
    filtered_rois = list(rois)
    for roi_name_to_remove in removal_names:
        retained_rois = [roi for roi in filtered_rois if roi.ROIName != roi_name_to_remove]
        if len(retained_rois) != len(filtered_rois):
            _emit(
                progress_sink,
                patient_uid,
                "Removed data-point (Pt: {}, {}: {})) ".format(
                    patient_uid,
                    removal_label,
                    roi_name_to_remove,
                ),
                removed_roi_name=str(roi_name_to_remove),
                removal_label=removal_label,
            )
        filtered_rois = retained_rois
    return filtered_rois


def _build_non_biopsy_structure_record(roi: Any,
                                       index_number: int,
                                       struct_type: str) -> dict[str, Any]:
    return {
    LEGACY_STRUCTURE_RECORD_KEYS.roi_key: roi.ROIName,
    LEGACY_STRUCTURE_RECORD_KEYS.ref_number_key: roi.ROINumber,
    LEGACY_STRUCTURE_RECORD_KEYS.index_number_key: index_number,
        "Struct type": struct_type,
        "Raw contour pts zslice list": None,
        "Raw contour pts": None,
        "Equal num zslice contour pts": None,
        "Intra-slice interpolation information": None,
        "Inter-slice interpolation information": None,
        "Point cloud raw": None,
        "Delaunay triangulation global structure": None,
        "Delaunay triangulation zslice-wise list": None,
        "Structure centroid pts": None,
        "Best fit line of centroid pts": None,
        "Centroid line sample pts": None,
        "Structure global centroid": None,
        "Reconstructed structure pts arr": None,
        "Interpolated structure point cloud dict": None,
        "Reconstructed structure delaunay global": None,
        "Maximum pairwise distance": None,
        "Structure volume": None,
        "Structure OPEN3D triangle mesh object": None,
        "Voxel size for structure volume calc": None,
        "Uncertainty data": None,
        "MC data: Generated normal dist random samples arr": None,
        "KDtree": None,
        "Nearest neighbours objects": [],
    }


def _build_biopsy_structure_record(roi_name: str,
                                   ref_number: Any,
                                   index_number: int,
                                   struct_type: str,
                                   simulated_bool: bool,
                                   simulated_type: str,
                                   simulated_metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    record = {
        LEGACY_STRUCTURE_RECORD_KEYS.roi_key: roi_name,
        LEGACY_STRUCTURE_RECORD_KEYS.ref_number_key: ref_number,
        LEGACY_STRUCTURE_RECORD_KEYS.index_number_key: index_number,
        "Struct type": struct_type,
        LEGACY_STRUCTURE_RECORD_KEYS.simulated_bool_key: bool(simulated_bool),
        LEGACY_STRUCTURE_RECORD_KEYS.simulated_type_key: simulated_type,
    }
    if simulated_metadata:
        record.update(dict(simulated_metadata))
    record.update(
        {
            "Reconstructed biopsy cylinder length (from contour data)": None,
            "Raw contour pts zslice list": None,
            "Raw contour pts": None,
            "Centroid variation arr": None,
            "Mean centroid variation": None,
            "Maximum projected distance between original centroids": None,
            "Equal num zslice contour pts": None,
            "Intra-slice interpolation information": None,
            "Inter-slice interpolation information": None,
            "Point cloud raw": None,
            "Delaunay triangulation global structure": None,
            "Delaunay triangulation zslice-wise list": None,
            "Structure global centroid": None,
            "Structure centroid pts": None,
            "Best fit line of centroid pts": None,
            "Centroid line sample pts": None,
            "Centroid line unit vec (bx needle base to bx needle tip)": None,
            "Interpolated structure point cloud dict": None,
            "Reconstructed structure pts arr": None,
            "Reconstructed structure point cloud": None,
            "Reconstructed structure delaunay global": None,
            "Maximum pairwise distance": None,
            "Structure volume": None,
            "Voxel size for structure volume calc": None,
            "Target DIL dict": None,
            "Random uniformly sampled volume pts arr": None,
            "Random uniformly sampled volume pts pcd": None,
            "Random uniformly sampled volume pts bx coord sys arr": None,
            "Random uniformly sampled volume pts bx coord sys pcd": None,
            "Bounding box for random uniformly sampled volume pts": None,
            "Num sampled bx pts": None,
            "Uncertainty data": None,
            "MC data: Generated uniform dist (biopsy needle compartment) random distance (z_needle) samples arr": None,
            "MC data: Generated uniform (biopsy needle compartment) random vectors (z_needle) samples arr": None,
            "MC data: Generated normal dist random samples arr": None,
            "MC data: Total rigid shift vectors arr": None,
            "MC data: bx only shifted 3darr": None,
            "MC data: bx and structure shifted dict": None,
            "MC data: MC sim translation results dict": None,
            "MC data: MC sim containment raw results dataframe": None,
            "MC data: MC sim compiled distances global dataframe": None,
            "MC data: MC sim compiled distances point-wise dataframe": None,
            "MC data: MC sim compiled distances voxel-wise dataframe": None,
            "MC data: MC sim containment and distance all trials dataframe (light)": None,
            "MC data: compiled sim results dataframe": None,
            "MC data: compiled sim sum-to-one results dataframe": None,
            "MC data: compiled sim results": None,
            "MC data: mutual compiled sim results": None,
            "MC data: voxelized containment results dict": None,
            "MC data: voxelized containment results dict (dict of lists)": None,
            "MC data: bx to dose NN search objects list": None,
            "MC data: Dose vals for each sampled bx pt arr (nominal & all MC trials)": None,
            "MC data: Dose gradient vals for each sampled bx pt arr (nominal & all MC trials)": None,
            "MC data: Differential DVH dict": None,
            "MC data: Cumulative DVH dict": None,
            "MC data: dose volume metrics dict": None,
            "MC data: Dose statistics for each sampled bx pt list (mean, std, quantiles)": None,
            "MC data: Dose statistics (MLE) for each sampled bx pt list (mean, std)": None,
            "MC data: voxelized dose results list": None,
            "MC data: voxelized dose results dict (dict of lists)": None,
            "Output csv file paths dict": {},
            "Output dicts for data frames": {},
            "KDtree": None,
            "Nearest neighbours objects": [],
        }
    )
    if simulated_bool:
        record[LEGACY_BIOPSY_RUNTIME_KEYS.simulated_biopsy_transport_request_key] = None
        record[LEGACY_BIOPSY_RUNTIME_KEYS.output_dataframes_key] = {
            "Dose output Z and radius": None,
            "Dose output voxelized": None,
            "Point-wise dose output by MC trial number": None,
            "Voxel-wise dose output by MC trial number": None,
            "Differential DVH by MC trial": None,
        }
    else:
        record[LEGACY_BIOPSY_RUNTIME_KEYS.output_dataframes_key] = {
            "Dose output Z and radius": None,
            "Dose output voxelized": None,
            "Point-wise dose output by MC trial number": None,
            "Voxel-wise dose output by MC trial number": None,
            "Differential DVH by MC trial": None,
            "Cumulative DVH by MC trial": None,
        }
    return record


def _build_all_ref_dict(mr_global_multi_structure_output_dataframe_str: str,
                        mr_global_by_voxel_multi_structure_output_dataframe_str: str) -> dict[str, Any]:
    return {
        LEGACY_PATIENT_ALL_REFERENCE_KEYS.multi_structure_information_key: {
            "Biopsy optimization: Optimal biopsy location (entire cubic lattice) dataframe": None,
        },
        LEGACY_PATIENT_ALL_REFERENCE_KEYS.preprocessing_output_dataframes_key: {
            "Selected structures": None,
            "Biopsy basic spatial features dataframe": None,
            "Simulated biopsy preparation dataframe": None,
            "Nearest DILs info dataframe": None,
            "Biopsy optimization - Cumulative projection (all points within prostate) dataframe": None,
            "Biopsy optimization - DIL centroids optimal targeting dataframe": None,
            "Biopsy optimization - Optimal DIL targeting dataframe": None,
            "Biopsy optimization - Optimal DIL targeting entire lattice dataframe": None,
            TARGET_DIL_OPTIMIZER_V2_SUMMARY_DF_KEY: None,
            TARGET_DIL_OPTIMIZER_V2_RANKED_DF_KEY: None,
            "Biopsy optimization - Guidance-map firing depth recommendations dataframe": None,
            "3D radiomic features all OAR and DIL structures": None,
            "Per sample point prostate double sextant classification": None,
            "Per voxel prostate double sextant classification": None,
            "Simulated biopsy planned vs realized centroid variation validation": None,
            "Prostate only points MR ADC dataframe (temporary for pre-processing)": None,
            "MR - ADC - summary statistics by structure dataframe": None,
        },
        LEGACY_PATIENT_ALL_REFERENCE_KEYS.mc_output_dataframes_key: {
            "All MC structure transformation values": None,
            "Tissue class - Global tissue class statistics": None,
            "Tissue class - Global tissue by structure statistics": None,
            "Tissue class - Tissue length above threshold": None,
            "Tissue class - sum-to-one mc results": None,
            "Tissue class - distances global results": None,
            "Tissue class - distances pt-wise results": None,
            "Tissue class - distances voxel-wise results": None,
            "Tissue class - containment and distances (light) results": None,
            "Tissue class - Pt wise structure specific results": None,
            "DVH metrics (Dx, Vx) statistics": None,
            "MR - " + str(mr_global_multi_structure_output_dataframe_str): None,
            "MR - " + str(mr_global_by_voxel_multi_structure_output_dataframe_str): None,
        },
    }


def _should_create_simulated_biopsies(simulated_biopsy_fraction_numbers_to_create: Any,
                                      patient_fraction_number: Any) -> bool:
    if simulated_biopsy_fraction_numbers_to_create == "all":
        return True
    if isinstance(simulated_biopsy_fraction_numbers_to_create, (list, tuple, set)):
        return patient_fraction_number in simulated_biopsy_fraction_numbers_to_create
    return patient_fraction_number == simulated_biopsy_fraction_numbers_to_create


def build_patient_structure_reference_bootstrap_fragment(
    *,
    patient_uid: str,
    structure_item: Any,
    data_removals_dict_bx: Mapping[str, Sequence[str]],
    data_removals_dict_prostate: Mapping[str, Sequence[str]],
    data_removals_dict_dil: Mapping[str, Sequence[str]],
    data_removals_dict_urethra: Mapping[str, Sequence[str]],
    data_removals_dict_rectum: Mapping[str, Sequence[str]],
    OAR_list: Sequence[str],
    DIL_list: Sequence[str],
    Bx_list: Sequence[str],
    st_ref_list: Sequence[str],
    structs_referenced_dict: Mapping[str, Any],
    all_ref_key: str,
    mr_global_multi_structure_output_dataframe_str: str,
    mr_global_by_voxel_multi_structure_output_dataframe_str: str,
    bx_sim_locations_dict: Mapping[str, Mapping[str, Any]],
    rectum_list: Sequence[str],
    urethra_list: Sequence[str],
    simulated_biopsy_fraction_numbers_to_create: Any,
    fraction_prefixes: Sequence[str],
    progress_sink: ProgressSink | None = None,
) -> PatientStructureReferenceBootstrapFragment:
    """Build the one-patient reference/info shell from an RTSTRUCT dataset."""
    progress_sink = coerce_progress_sink(progress_sink)
    patient_uid = str(patient_uid)
    messages_before = tuple(getattr(progress_sink, "emitted_events", ()))

    filtered_oars = _filter_removed_rois(
        _matching_rois(structure_item, OAR_list),
        data_removals_dict_prostate.get(patient_uid, ()),
        patient_uid=patient_uid,
        removal_label="Prostate",
        progress_sink=progress_sink,
    )
    filtered_dils = _filter_removed_rois(
        _matching_rois(structure_item, DIL_list),
        data_removals_dict_dil.get(patient_uid, ()),
        patient_uid=patient_uid,
        removal_label="DIL",
        progress_sink=progress_sink,
    )
    filtered_rectums = _filter_removed_rois(
        _matching_rois(structure_item, rectum_list),
        data_removals_dict_rectum.get(patient_uid, ()),
        patient_uid=patient_uid,
        removal_label="Rect",
        progress_sink=progress_sink,
    )
    filtered_urethras = _filter_removed_rois(
        _matching_rois(structure_item, urethra_list),
        data_removals_dict_urethra.get(patient_uid, ()),
        patient_uid=patient_uid,
        removal_label="Uret",
        progress_sink=progress_sink,
    )
    filtered_biopsies = _filter_removed_rois(
        _matching_rois(structure_item, Bx_list),
        data_removals_dict_bx.get(patient_uid, ()),
        patient_uid=patient_uid,
        removal_label="Bx",
        progress_sink=progress_sink,
    )

    oar_ref = [
        _build_non_biopsy_structure_record(roi, index, st_ref_list[1])
        for index, roi in enumerate(filtered_oars)
    ]
    dil_ref = [
        _build_non_biopsy_structure_record(roi, index, st_ref_list[2])
        for index, roi in enumerate(filtered_dils)
    ]
    rectum_ref = [
        _build_non_biopsy_structure_record(roi, index, st_ref_list[3])
        for index, roi in enumerate(filtered_rectums)
    ]
    urethra_ref = [
        _build_non_biopsy_structure_record(roi, index, st_ref_list[4])
        for index, roi in enumerate(filtered_urethras)
    ]
    biopsy_ref = [
        _build_biopsy_structure_record(
            roi.ROIName,
            roi.ROINumber,
            index,
            st_ref_list[0],
            False,
            "Real",
        )
        for index, roi in enumerate(filtered_biopsies)
    ]

    biopsy_ref_index_start = len(biopsy_ref)
    patient_id_from_dicom = _patient_dicom_value(structure_item, (0x0010, 0x0020))
    patient_name_from_dicom = _patient_dicom_value(structure_item, (0x0010, 0x0010))
    patient_fraction_number = misc_tools.extract_number_from_string(patient_id_from_dicom, fraction_prefixes)
    create_simulated_for_fraction = _should_create_simulated_biopsies(
        simulated_biopsy_fraction_numbers_to_create,
        patient_fraction_number,
    )

    simulated_biopsy_ref_index_start = 0
    simulated_biopsy_refs_total: list[dict[str, Any]] = []
    for biopsy_sim_type, biopsy_sim_config in bx_sim_locations_dict.items():
        if not bool(biopsy_sim_config.get("Create", False)) or not create_simulated_for_fraction:
            continue

        simulated_relative_struct_type = biopsy_sim_config["Relative to struct type"]
        simulated_ref_identifier = biopsy_sim_config["Identifier string"]
        simulated_relative_contour_names = structs_referenced_dict[simulated_relative_struct_type]["Contour names"]

        if simulated_relative_struct_type == st_ref_list[2]:
            removal_list = data_removals_dict_dil.get(patient_uid, ())
        elif simulated_relative_struct_type == st_ref_list[0]:
            removal_list = data_removals_dict_bx.get(patient_uid, ())
        elif simulated_relative_struct_type == st_ref_list[1]:
            removal_list = data_removals_dict_prostate.get(patient_uid, ())
        elif simulated_relative_struct_type == st_ref_list[3]:
            removal_list = data_removals_dict_rectum.get(patient_uid, ())
        elif simulated_relative_struct_type == st_ref_list[4]:
            removal_list = data_removals_dict_urethra.get(patient_uid, ())
        else:
            removal_list = ()

        filtered_simulated_biopsies = [
            roi
            for roi in _matching_rois(structure_item, simulated_relative_contour_names)
            if roi.ROIName not in removal_list
        ]
        simulated_biopsy_refs = [
            _build_biopsy_structure_record(
                "Bx_Tr_" + str(simulated_ref_identifier) + " " + roi.ROIName,
                str(simulated_ref_identifier) + " " + roi.ROIName,
                biopsy_ref_index_start + simulated_biopsy_ref_index_start + index,
                st_ref_list[0],
                True,
                str(biopsy_sim_type),
                {
                    "Transport family": biopsy_sim_config.get("Transport family", "identity"),
                    "Relative structure type": simulated_relative_struct_type,
                    "Relative structure name": roi.ROIName,
                    "Relative structure ref #": roi.ROINumber,
                },
            )
            for index, roi in enumerate(filtered_simulated_biopsies)
        ]
        simulated_biopsy_ref_index_start = len(simulated_biopsy_refs)
        simulated_biopsy_refs_total.extend(simulated_biopsy_refs)

    biopsy_ref.extend(simulated_biopsy_refs_total)
    all_ref = _build_all_ref_dict(
        mr_global_multi_structure_output_dataframe_str,
        mr_global_by_voxel_multi_structure_output_dataframe_str,
    )

    biopsy_type_counts = {
        item[LEGACY_STRUCTURE_RECORD_KEYS.simulated_type_key]: sum(
            1
            for record in biopsy_ref
            if record[LEGACY_STRUCTURE_RECORD_KEYS.simulated_type_key]
            == item[LEGACY_STRUCTURE_RECORD_KEYS.simulated_type_key]
        )
        for item in biopsy_ref
    }
    biopsy_info = {
        LEGACY_STRUCTURE_INFO_KEYS.num_structs_key: len(biopsy_ref),
        LEGACY_STRUCTURE_INFO_KEYS.num_sim_structs_key: len(simulated_biopsy_refs_total),
        LEGACY_STRUCTURE_INFO_KEYS.num_real_structs_key: len(biopsy_ref) - len(simulated_biopsy_refs_total),
        LEGACY_STRUCTURE_INFO_KEYS.biopsy_type_counts_key: biopsy_type_counts,
    }
    oar_info = {LEGACY_STRUCTURE_INFO_KEYS.num_structs_key: len(oar_ref)}
    dil_info = {LEGACY_STRUCTURE_INFO_KEYS.num_structs_key: len(dil_ref)}
    rectum_info = {LEGACY_STRUCTURE_INFO_KEYS.num_structs_key: len(rectum_ref)}
    urethra_info = {LEGACY_STRUCTURE_INFO_KEYS.num_structs_key: len(urethra_ref)}
    patient_total_num_structs = (
        biopsy_info[LEGACY_STRUCTURE_INFO_KEYS.num_structs_key]
        + oar_info[LEGACY_STRUCTURE_INFO_KEYS.num_structs_key]
        + dil_info[LEGACY_STRUCTURE_INFO_KEYS.num_structs_key]
        + rectum_info[LEGACY_STRUCTURE_INFO_KEYS.num_structs_key]
        + urethra_info[LEGACY_STRUCTURE_INFO_KEYS.num_structs_key]
    )
    all_structs_info = {LEGACY_STRUCTURE_INFO_KEYS.total_num_structs_key: patient_total_num_structs}

    reference_keys = PatientStructureReferenceKeys.from_legacy_refs(
        st_ref_list=st_ref_list,
        all_ref_key=all_ref_key,
    )
    patient_structure_reference = PatientStructureReferenceState(
        keys=reference_keys,
        patient_uid=patient_uid,
        patient_id_from_dicom=patient_id_from_dicom,
        patient_name=patient_name_from_dicom,
        fraction_number=patient_fraction_number,
        biopsies=biopsy_ref,
        oars=oar_ref,
        dils=dil_ref,
        rectums=rectum_ref,
        urethras=urethra_ref,
        all_reference=all_ref,
    )
    patient_structure_info = PatientStructureInfoState(
        keys=reference_keys,
        patient_uid=patient_uid,
        patient_id_from_dicom=patient_id_from_dicom,
        patient_name=patient_name_from_dicom,
        fraction_number=patient_fraction_number,
        biopsy_info=biopsy_info,
        oar_info=oar_info,
        dil_info=dil_info,
        rectum_info=rectum_info,
        urethra_info=urethra_info,
        all_structures_info=all_structs_info,
    )
    patient_reference_dict = patient_structure_reference.to_legacy_dict()
    patient_info_dict = patient_structure_info.to_legacy_dict()

    emitted_events = tuple(getattr(progress_sink, "emitted_events", ()))
    new_events = emitted_events[len(messages_before):]
    messages = tuple(event.message for event in new_events if getattr(event, "message", ""))
    return PatientStructureReferenceBootstrapFragment(
        patient_uid=patient_uid,
        patient_reference_dict=patient_reference_dict,
        patient_info_dict=patient_info_dict,
        patient_structure_reference=patient_structure_reference,
        patient_structure_info=patient_structure_info,
        messages=messages,
        metadata={
            "num_real_biopsies": biopsy_info[LEGACY_STRUCTURE_INFO_KEYS.num_real_structs_key],
            "num_simulated_biopsies": biopsy_info[LEGACY_STRUCTURE_INFO_KEYS.num_sim_structs_key],
            "num_total_structures": patient_total_num_structs,
        },
    )


def build_patient_structure_reference_bootstrap_fragment_from_path(
    *,
    patient_uid: str,
    structure_item_path: str | Path,
    **kwargs: Any,
) -> PatientStructureReferenceBootstrapFragment:
    """Read one RTSTRUCT file and build its patient reference fragment."""
    with pydicom.dcmread(structure_item_path, defer_size="2 MB") as structure_item:
        return build_patient_structure_reference_bootstrap_fragment(
            patient_uid=patient_uid,
            structure_item=structure_item,
            **kwargs,
        )


def attach_patient_dose_reference_from_path(patient_reference_dict: dict[str, Any],
                                            *,
                                            patient_uid: str,
                                            dose_item_path: str | Path,
                                            ds_ref: str) -> bool:
    """Attach the legacy-shaped dose dictionary for one patient when applicable."""
    if patient_reference_dict[LEGACY_PATIENT_REFERENCE_KEYS.fraction_number_key] == 1:
        return False
    with pydicom.dcmread(dose_item_path, defer_size="2 MB") as dose_item:
        dose_id = str(patient_uid) + dose_item.StudyDate
        patient_reference_dict[ds_ref] = {
            "Dose ID": dose_id,
            "Study date": dose_item.StudyDate,
            "Dose pixel data": dose_item.PixelData,
            "Dose pixel arr": dose_item.pixel_array,
            "Pixel spacing": [float(item) for item in dose_item.PixelSpacing],
            "Dose grid scaling": float(dose_item.DoseGridScaling),
            "Dose units": dose_item.DoseUnits,
            "Dose type": dose_item.DoseType,
            "Grid frame offset vector": [float(item) for item in dose_item.GridFrameOffsetVector],
            "Image orientation patient": [float(item) for item in dose_item.ImageOrientationPatient],
            "Image position patient": [float(item) for item in dose_item.ImagePositionPatient],
            "Dose and gradient phys space and pixel 3d arr": None,
            "Dose grid point cloud": None,
            "Dose grid point cloud thresholded": None,
            "Dose grid gradient point cloud": None,
            "Dose grid gradient point cloud thresholded": None,
            "KDtree": None,
            "KDtree gradient": None,
        }
    return True


def attach_patient_plan_reference_from_path(patient_reference_dict: dict[str, Any],
                                            *,
                                            patient_uid: str,
                                            plan_item_path: str | Path,
                                            pln_ref: str) -> None:
    """Attach the legacy-shaped treatment-plan dictionary for one patient."""
    with pydicom.dcmread(plan_item_path, defer_size="2 MB") as plan_item:
        plan_id = str(patient_uid) + plan_item.StudyDate
        plan_ref_dict = {
            "Plan ID": plan_id,
            "Study date": plan_item.StudyDate,
            "Dose units": "Gy",
            "Prescription doses dict": {},
        }
        for dose_ref_seq_index in range(len(plan_item.DoseReferenceSequence)):
            dose_reference = plan_item.DoseReferenceSequence[dose_ref_seq_index]
            plan_ref_dict["Prescription doses dict"][dose_reference["DoseReferenceType"].value] = (
                dose_reference["TargetPrescriptionDose"].value
            )
        patient_reference_dict[pln_ref] = plan_ref_dict


def attach_patient_mr_adc_references_from_paths(patient_reference_dict: dict[str, Any],
                                                *,
                                                patient_uid: str,
                                                mr_adc_item_paths: Sequence[str | Path],
                                                mr_adc_ref: str,
                                                progress_sink: ProgressSink | None = None) -> None:
    """Attach the legacy-shaped MR ADC dictionary for one patient."""
    progress_sink = coerce_progress_sink(progress_sink)
    mr_adc_ref_dict: dict[str, Any] = {}
    for mr_adc_item_path in mr_adc_item_paths:
        with pydicom.dcmread(mr_adc_item_path, defer_size="2 MB") as mr_adc_item:
            series_instance_uid = mr_adc_item.SeriesInstanceUID
            mr_adc_id = str(patient_uid) + mr_adc_item.StudyDate
            rwvm = getattr(mr_adc_item, "RealWorldValueMappingSequence", None)
            if rwvm and len(rwvm) > 1:
                _emit(
                    progress_sink,
                    str(patient_uid),
                    "Multiple real world value mappings detected for ({}, {})".format(patient_uid, mr_adc_id),
                    mr_adc_id=mr_adc_id,
                )
            if rwvm and len(rwvm) > 0:
                rwv = rwvm[0]
                units = str(getattr(rwv.MeasurementUnitsCodeSequence[0], "CodeMeaning", "unknown"))
                slope = np.array(rwv.RealWorldValueSlope)
                intercept = np.array(rwv.RealWorldValueIntercept)
                rwv_units = getattr(rwv, "LUTLabel", "unknown")
            else:
                units = "mm\u00B2/s (assumed)"
                slope = np.array([1e-6])
                intercept = np.array([0.0])
                rwv_units = "mm\u00B2/s (assumed)"
                _emit(
                    progress_sink,
                    str(patient_uid),
                    "No RealWorldValueMappingSequence found for ({}, {}) - using defaults.".format(patient_uid, mr_adc_id),
                    mr_adc_id=mr_adc_id,
                )

            if series_instance_uid not in mr_adc_ref_dict:
                mr_adc_ref_dict[series_instance_uid] = {
                    "MR ADC ID": mr_adc_id,
                    "Series instance UID": series_instance_uid,
                    "Study date": mr_adc_item.StudyDate,
                    "Pixel arr (all slices)": mr_adc_item.pixel_array,
                    "Pixel spacing": np.array(mr_adc_item.PixelSpacing),
                    "Units": units,
                    "RWVSlope (all slices)": slope,
                    "RWVIntercept (all slices)": intercept,
                    "RWV Units": rwv_units,
                    "Slice thickness": getattr(mr_adc_item, "SliceThickness", -1),
                    "Image orientation patient": np.array(mr_adc_item.ImageOrientationPatient),
                    "Image position patient (all slices)": np.array(mr_adc_item.ImagePositionPatient),
                    "MR ADC phys space Nx4 arr": None,
                    "MR ADC phys space Nx4 arr (filtered, non-negative)": None,
                    "MR ADC grid point cloud": None,
                    "MR ADC grid point cloud thresholded": None,
                    "KDtree": None,
                }
            else:
                mr_adc_ref_subdict = mr_adc_ref_dict[series_instance_uid]
                mr_adc_ref_subdict["Pixel arr (all slices)"] = np.dstack(
                    (mr_adc_ref_subdict["Pixel arr (all slices)"], mr_adc_item.pixel_array)
                )
                mr_adc_ref_subdict["RWVSlope (all slices)"] = np.hstack(
                    (mr_adc_ref_subdict["RWVSlope (all slices)"], slope)
                )
                mr_adc_ref_subdict["RWVIntercept (all slices)"] = np.hstack(
                    (mr_adc_ref_subdict["RWVIntercept (all slices)"], intercept)
                )
                mr_adc_ref_subdict["Image position patient (all slices)"] = np.vstack(
                    (
                        mr_adc_ref_subdict["Image position patient (all slices)"],
                        np.array(mr_adc_item.ImagePositionPatient),
                    )
                )
    patient_reference_dict[mr_adc_ref] = mr_adc_ref_dict


def assemble_master_structure_reference_from_patient_fragments(
    fragments: Sequence[PatientStructureReferenceBootstrapFragment],
) -> dict[str, Any]:
    """Assemble a legacy-shaped patient dictionary from patient fragments."""
    return {
        fragment.patient_uid: fragment.patient_reference_dict
        for fragment in fragments
    }


def assemble_patient_structure_reference_state_registry(
    fragments: Sequence[PatientStructureReferenceBootstrapFragment],
) -> dict[str, PatientStructureReferenceState]:
    """Assemble the typed patient-reference registry for future runner use."""
    registry: dict[str, PatientStructureReferenceState] = {}
    for fragment in fragments:
        if fragment.patient_structure_reference is None:
            raise ValueError(
                "fragment is missing patient_structure_reference: "
                f"{fragment.patient_uid}"
            )
        registry[fragment.patient_uid] = fragment.patient_structure_reference
    return registry


def assemble_patient_structure_info_state_registry(
    fragments: Sequence[PatientStructureReferenceBootstrapFragment],
) -> dict[str, PatientStructureInfoState]:
    """Assemble the typed patient-info registry for future runner use."""
    registry: dict[str, PatientStructureInfoState] = {}
    for fragment in fragments:
        if fragment.patient_structure_info is None:
            raise ValueError(
                "fragment is missing patient_structure_info: "
                f"{fragment.patient_uid}"
            )
        registry[fragment.patient_uid] = fragment.patient_structure_info
    return registry


def assemble_structure_reference_info_for_run(
    fragments: Sequence[PatientStructureReferenceBootstrapFragment],
    *,
    st_ref_list: Sequence[str],
    all_ref_key: str,
    bx_sim_locations_dict: Mapping[str, Mapping[str, Any]],
    interp_inter_slice_dist: float,
    interp_intra_slice_dist: float,
) -> dict[str, Any]:
    """Build run-level structure info from patient-local info fragments."""
    by_patient_info = {
        fragment.patient_uid: fragment.patient_info_dict
        for fragment in fragments
    }
    global_num_cases = len(fragments)
    global_unique_patient_names = []
    for fragment in fragments:
        patient_name = str(fragment.patient_info_dict[LEGACY_PATIENT_REFERENCE_KEYS.patient_name_key])
        if patient_name not in global_unique_patient_names:
            global_unique_patient_names.append(patient_name)
    global_num_biopsies = sum(
        info[st_ref_list[0]][LEGACY_STRUCTURE_INFO_KEYS.num_structs_key]
        for info in by_patient_info.values()
    )
    global_num_oar = sum(
        info[st_ref_list[1]][LEGACY_STRUCTURE_INFO_KEYS.num_structs_key]
        for info in by_patient_info.values()
    )
    global_num_dil = sum(
        info[st_ref_list[2]][LEGACY_STRUCTURE_INFO_KEYS.num_structs_key]
        for info in by_patient_info.values()
    )
    global_total_num_structs = sum(
        info[all_ref_key][LEGACY_STRUCTURE_INFO_KEYS.total_num_structs_key]
        for info in by_patient_info.values()
    )
    biopsy_type_counts_by_patient = [
        info[st_ref_list[0]][LEGACY_STRUCTURE_INFO_KEYS.biopsy_type_counts_key]
        for info in by_patient_info.values()
    ]
    global_num_biopsies_by_type = {
        key: sum(counts[key] for counts in biopsy_type_counts_by_patient if key in counts)
        for counts in biopsy_type_counts_by_patient
        for key in counts
    }
    bx_types_list = ["Real"] + [
        key
        for key, value in bx_sim_locations_dict.items()
        if value.get("Create", False)
    ]
    preprocessing_info = {
        "Interslice interp dist": interp_inter_slice_dist,
        "Intraslice interp dist": interp_intra_slice_dist,
        "Preprocessing performed": False,
    }
    mc_info = {
        "Num MC containment simulations": None,
        "Num MC dose simulations": None,
        "Num MC MR simulations": None,
        "Num optimizer v2 transform samples": None,
        "Num stochastic targeting transform samples": None,
        "Num sample pts per BX core": None,
        "BX sample pt lattice spacing (mm)": None,
        "BX sample pt volume element (mm^3)": None,
        "Max of num MC simulations": None,
        "Max of generated transform samples": None,
        "MC sim performed": False,
        "MC containment sim performed": False,
        "MC dose sim performed": False,
        "MC MR sim performed": False,
    }
    random_info = {
        "Transform generation random seed": None,
        "Optimizer v1 random seed": None,
    }
    return {
        LEGACY_MASTER_INFO_KEYS.global_key: {
            LEGACY_MASTER_INFO_KEYS.num_cases_key: global_num_cases,
            LEGACY_MASTER_INFO_KEYS.num_unique_patient_names_key: len(global_unique_patient_names),
            LEGACY_MASTER_INFO_KEYS.num_structures_key: global_total_num_structs,
            LEGACY_MASTER_INFO_KEYS.num_biopsies_key: global_num_biopsies,
            LEGACY_MASTER_INFO_KEYS.num_biopsies_by_type_key: global_num_biopsies_by_type,
            LEGACY_MASTER_INFO_KEYS.num_dils_key: global_num_dil,
            LEGACY_MASTER_INFO_KEYS.bx_types_list_key: bx_types_list,
            LEGACY_MASTER_INFO_KEYS.preprocessing_info_key: preprocessing_info,
            LEGACY_MASTER_INFO_KEYS.mc_info_key: mc_info,
            "Random info": random_info,
            "Patient specific guidance map figures directory dict": None,
            "Guidance map figures dir": None,
            LEGACY_MASTER_INFO_KEYS.specific_output_dir_key: None,
        },
        LEGACY_MASTER_INFO_KEYS.by_patient_key: by_patient_info,
    }