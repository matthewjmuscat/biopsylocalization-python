"""Patient-level MC containment output contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from legacy_data_keys import legacy_data_keys

from .contracts import MCContainmentSimulationConfig
from .legacy_keys import legacy_mc_keys

MC_CONTAINMENT_BIOPSY_OUTPUT_KEYS = legacy_mc_keys.biopsy_outputs.containment_output_keys
MC_STRUCTURE_SPECIFIC_RESULT_KEYS = (
    "Total successes (containment) list",
    "Binomial estimator list",
    "Confidence interval 95 (containment) list",
    "Standard error (containment) list",
    "Nominal containment list",
)


@dataclass(slots=True)
class PatientContainmentStructureInventory:
    """Patient-local non-biopsy structures and legacy count metadata for containment."""

    patient_uid: str
    relative_structure_template: dict[tuple[Any, str, Any, int], None]
    total_num_structures: int
    total_num_biopsies: int
    total_num_non_biopsies: int

    @property
    def relative_structure_infos(self) -> tuple[tuple[Any, str, Any, int], ...]:
        return tuple(self.relative_structure_template.keys())


@dataclass(slots=True)
class PatientContainmentDilatedStructureBank:
    """Patient-local relative-structure dilation state reused for each biopsy."""

    patient_uid: str
    dilated_structures_by_structure: dict[tuple[Any, str, Any, int], list[Any]] = field(default_factory=dict)
    centroids_by_structure: dict[tuple[Any, str, Any, int], Any] = field(default_factory=dict)
    relative_structure_mapping_by_structure: dict[tuple[Any, str, Any, int], Any] = field(default_factory=dict)

    @property
    def relative_structure_infos(self) -> tuple[tuple[Any, str, Any, int], ...]:
        return tuple(self.dilated_structures_by_structure.keys())


@dataclass(slots=True)
class PatientContainmentOutputs:
    """Containment outputs collected from one patient's biopsy records."""

    patient_uid: str
    biopsy_outputs: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "patient_uid": self.patient_uid,
            "biopsy_outputs": self.biopsy_outputs,
        }


def build_structure_specific_results_template() -> dict[str, None]:
    """Return the legacy per-structure containment result template."""
    return {key: None for key in MC_STRUCTURE_SPECIFIC_RESULT_KEYS}


def build_mutual_structure_specific_results_template() -> dict[str, None]:
    """Return the legacy mutual-containment result template."""
    return {key: None for key in MC_STRUCTURE_SPECIFIC_RESULT_KEYS}


def _resolve_patient_info(patient_uid: str, patient_info_dict: Mapping[str, Any]) -> Mapping[str, Any]:
    master_keys = legacy_data_keys.master_info
    if master_keys.by_patient_key in patient_info_dict:
        by_patient = patient_info_dict[master_keys.by_patient_key]
        if patient_uid in by_patient:
            return by_patient[patient_uid]
        return by_patient[str(patient_uid)]
    return patient_info_dict


def build_patient_relative_structure_inventory(patient_uid: str,
                                               patient_reference_dict: Mapping[str, Any],
                                               patient_info_dict: Mapping[str, Any],
                                               *,
                                               structs_referenced_list: Sequence[str],
                                               bx_ref: str,
                                               all_ref_key: str) -> PatientContainmentStructureInventory:
    """Mirror the oracle's patient-specific non-biopsy structure inventory."""
    identity_keys = legacy_mc_keys.biopsy_identity
    structure_info_keys = legacy_data_keys.structure_info
    resolved_patient_info = _resolve_patient_info(patient_uid, patient_info_dict)

    relative_structure_template: dict[tuple[Any, str, Any, int], None] = {}
    for non_bx_struct_type in tuple(structs_referenced_list)[1:]:
        for structure_index, specific_non_bx_structure in enumerate(patient_reference_dict[non_bx_struct_type]):
            structure_info = (
                specific_non_bx_structure[identity_keys.roi_key],
                non_bx_struct_type,
                specific_non_bx_structure[identity_keys.ref_number_key],
                structure_index,
            )
            relative_structure_template[structure_info] = None

    total_num_structures = int(resolved_patient_info[all_ref_key][structure_info_keys.total_num_structs_key])
    total_num_biopsies = int(resolved_patient_info[bx_ref][structure_info_keys.num_structs_key])
    return PatientContainmentStructureInventory(
        patient_uid=str(patient_uid),
        relative_structure_template=relative_structure_template,
        total_num_structures=total_num_structures,
        total_num_biopsies=total_num_biopsies,
        total_num_non_biopsies=total_num_structures - total_num_biopsies,
    )


def _structure_zslices_for_containment(patient_reference_dict: Mapping[str, Any],
                                       *,
                                       non_bx_structure_type: str,
                                       structure_index: int,
                                       oar_ref: str,
                                       rectum_ref: str,
                                       urethra_ref: str) -> Any:
    geometry_keys = legacy_data_keys.structure_geometry
    if non_bx_structure_type in {oar_ref, rectum_ref, urethra_ref}:
        return patient_reference_dict[non_bx_structure_type][structure_index][
            geometry_keys.equal_num_zslice_contour_points_key
        ]
    interslice_information = patient_reference_dict[non_bx_structure_type][structure_index][
        geometry_keys.interslice_interpolation_information_key
    ]
    return interslice_information.interpolated_pts_list


def build_patient_containment_dilated_structure_bank(
    patient_uid: str,
    patient_reference_dict: Mapping[str, Any],
    inventory: PatientContainmentStructureInventory,
    *,
    num_mc_containment_simulations: int,
    oar_ref: str,
    rectum_ref: str,
    urethra_ref: str,
    containment_config: MCContainmentSimulationConfig,
    parallel_pool: Any,
) -> PatientContainmentDilatedStructureBank:
    """Build the patient-level dilated relative-structure bank used by containment."""
    import cupy as cp
    import numpy as np
    import polygon_dilation_helpers_numpy

    intermediate_keys = legacy_mc_keys.containment_intermediates
    geometry_keys = legacy_data_keys.structure_geometry
    dilated_structures_by_structure: dict[tuple[Any, str, Any, int], list[Any]] = {}
    centroids_by_structure: dict[tuple[Any, str, Any, int], Any] = {}
    relative_structure_mapping_by_structure: dict[tuple[Any, str, Any, int], Any] = {}

    for structure_info in inventory.relative_structure_infos:
        non_bx_structure_type = structure_info[1]
        structure_index = structure_info[3]
        non_bx_struct_zslices_list = _structure_zslices_for_containment(
            patient_reference_dict,
            non_bx_structure_type=non_bx_structure_type,
            structure_index=structure_index,
            oar_ref=oar_ref,
            rectum_ref=rectum_ref,
            urethra_ref=urethra_ref,
        )
        dilation_samples = cp.asnumpy(
            patient_reference_dict[non_bx_structure_type][structure_index][
                intermediate_keys.normal_dist_dilations_samples_array_key
            ]
        )
        nominal_centroid = patient_reference_dict[non_bx_structure_type][structure_index][
            geometry_keys.structure_global_centroid_key
        ].copy()

        if not dilation_samples.any():
            dilated_structures_by_structure[structure_info] = [non_bx_struct_zslices_list]
            centroids_by_structure[structure_info] = np.reshape(nominal_centroid, (1, 3))
            relative_structure_mapping_by_structure[structure_info] = np.zeros(
                int(num_mc_containment_simulations) + 1,
                dtype=int,
            )
            continue

        org_config_2d_arr, org_config_indices_slices_arr = (
            polygon_dilation_helpers_numpy.convert_to_2d_array_and_indices_numpy(non_bx_struct_zslices_list)
        )
        dilated_structures_list, dilated_structures_slices_indices_list = (
            polygon_dilation_helpers_numpy.generate_dilated_structures_parallelized(
                org_config_2d_arr,
                org_config_indices_slices_arr,
                dilation_samples,
                containment_config.show_non_bx_relative_structure_z_dilation_bool,
                containment_config.show_non_bx_relative_structure_xy_dilation_bool,
                parallel_pool,
            )
        )

        reconstructed_dilated_structures = []
        centroids_of_each_dilated_structure = np.empty([len(dilated_structures_list), 3])
        for dilated_structure_index, dilated_structure_2d_arr in enumerate(dilated_structures_list):
            reconstructed_dilated_structures.append(
                polygon_dilation_helpers_numpy.reconstruct_list_from_2d_array(
                    dilated_structure_2d_arr,
                    dilated_structures_slices_indices_list[dilated_structure_index],
                )
            )
            centroids_of_each_dilated_structure[dilated_structure_index, :] = np.mean(
                dilated_structure_2d_arr,
                axis=0,
            )

        nominal_and_dilated_structures = [non_bx_struct_zslices_list] + reconstructed_dilated_structures
        dilated_structures_by_structure[structure_info] = nominal_and_dilated_structures
        centroids_by_structure[structure_info] = np.vstack((nominal_centroid, centroids_of_each_dilated_structure))
        relative_structure_mapping_by_structure[structure_info] = np.arange(0, len(nominal_and_dilated_structures))

        del dilated_structures_list
        del dilated_structures_slices_indices_list

    return PatientContainmentDilatedStructureBank(
        patient_uid=str(patient_uid),
        dilated_structures_by_structure=dilated_structures_by_structure,
        centroids_by_structure=centroids_by_structure,
        relative_structure_mapping_by_structure=relative_structure_mapping_by_structure,
    )


def collect_patient_containment_outputs(patient_uid: str,
                                        patient_reference_dict: Mapping[str, Any],
                                        *,
                                        bx_ref: str) -> PatientContainmentOutputs:
    """Collect containment artifacts written into one patient dictionary."""
    identity_keys = legacy_mc_keys.biopsy_identity
    biopsy_outputs: list[dict[str, Any]] = []
    for biopsy_index, biopsy_structure in enumerate(patient_reference_dict.get(bx_ref, ())):
        outputs = {
            output_key: biopsy_structure.get(output_key)
            for output_key in MC_CONTAINMENT_BIOPSY_OUTPUT_KEYS
            if output_key in biopsy_structure
        }
        if outputs:
            biopsy_outputs.append(
                {
                    identity_keys.roi_key: biopsy_structure.get(identity_keys.roi_key),
                    identity_keys.ref_number_key: biopsy_structure.get(identity_keys.ref_number_key),
                    identity_keys.index_number_key: biopsy_structure.get(identity_keys.index_number_key, biopsy_index),
                    identity_keys.simulated_bool_key: biopsy_structure.get(identity_keys.simulated_bool_key),
                    identity_keys.simulated_type_key: biopsy_structure.get(identity_keys.simulated_type_key),
                    "outputs": outputs,
                }
            )
    return PatientContainmentOutputs(
        patient_uid=str(patient_uid),
        biopsy_outputs=biopsy_outputs,
    )
