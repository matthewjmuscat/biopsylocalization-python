from collections import defaultdict

import numpy as np

from target_dil_v2.state import MC_PREP_BIOPSY_CYLINDER_LENGTH_KEY
from target_dil_v2.state import MC_PREP_BIOPSY_CYLINDER_LENGTH_METHOD_KEY
from target_dil_v2.state import TARGET_DIL_V2_DICT_KEY
from target_dil_v2.state import TARGET_DIL_V2_INFO_DICT_KEY


def _set_length_fields(specific_structure,
                       length_mm,
                       length_source,
                       enable_target_dil_v2
                       ):
    specific_structure[MC_PREP_BIOPSY_CYLINDER_LENGTH_KEY] = float(length_mm)
    specific_structure[MC_PREP_BIOPSY_CYLINDER_LENGTH_METHOD_KEY] = length_source

    target_dil_v2_dict = specific_structure[TARGET_DIL_V2_DICT_KEY]
    target_dil_v2_dict["Enabled"] = enable_target_dil_v2
    target_dil_v2_dict["Length sidecar source"] = length_source
    target_dil_v2_dict["Nominal length mm"] = float(length_mm)


def _find_nearest_dil_refnum(bx_centroid,
                             dil_centroids_by_ref
                             ):
    best_refnum = None
    best_dist2 = None
    for dil_refnum, dil_centroid in dil_centroids_by_ref.items():
        dist2 = np.sum((bx_centroid - dil_centroid) ** 2)
        if best_dist2 is None or dist2 < best_dist2:
            best_dist2 = dist2
            best_refnum = dil_refnum

    return best_refnum


def _determine_simulated_length(patientUID,
                                specific_structure,
                                simulated_biopsy_length_method,
                                biopsy_needle_compartment_length,
                                mean_of_real_biopsy_lengths,
                                std_of_real_biopsy_lengths,
                                real_bx_lengths_by_dil,
                                dil_ref
                                ):
    if simulated_biopsy_length_method == 'full':
        return float(biopsy_needle_compartment_length), 'full'

    if simulated_biopsy_length_method == 'real normal':
        within_bounds = False
        while within_bounds == False:
            sampled_length = np.random.normal(loc=mean_of_real_biopsy_lengths,
                                              scale=std_of_real_biopsy_lengths)
            if std_of_real_biopsy_lengths == 0:
                within_bounds = True
            elif (sampled_length >= mean_of_real_biopsy_lengths - 2 * std_of_real_biopsy_lengths) and (sampled_length <= mean_of_real_biopsy_lengths + 2 * std_of_real_biopsy_lengths):
                within_bounds = True

        return float(sampled_length), 'real normal'

    if simulated_biopsy_length_method == 'real mean':
        return float(mean_of_real_biopsy_lengths), 'real mean'

    if simulated_biopsy_length_method == 'match real':
        sampled_length = float(mean_of_real_biopsy_lengths)
        relative_structure_type = specific_structure["Relative structure type"]
        relative_structure_refnum = specific_structure["Relative structure ref #"]

        if relative_structure_type == dil_ref:
            lengths_for_this_dil = real_bx_lengths_by_dil.get(patientUID, {}).get(relative_structure_refnum, [])
            if lengths_for_this_dil:
                sampled_length = float(np.mean(lengths_for_this_dil))

        return sampled_length, 'match real'

    return float(biopsy_needle_compartment_length), 'full'


def run_length_sidecar(master_structure_reference_dict,
                       bx_ref,
                       dil_ref,
                       all_ref_key,
                       simulated_biopsy_length_method,
                       biopsy_needle_compartment_length,
                       enable_target_dil_v2
                       ):
    real_biopsy_lengths_list = []
    real_bx_lengths_by_dil = defaultdict(lambda: defaultdict(list))

    for patientUID, pydicom_item in master_structure_reference_dict.items():
        dil_centroids_by_ref = {}
        if dil_ref in pydicom_item:
            for specific_dil_structure in pydicom_item[dil_ref]:
                dil_refnum = specific_dil_structure["Ref #"]
                dil_centroid = np.array(specific_dil_structure["Structure global centroid"]).reshape(3)
                dil_centroids_by_ref[dil_refnum] = dil_centroid

        for specific_structure in pydicom_item[bx_ref]:
            if specific_structure["Simulated bool"] == True:
                continue

            length_mm = specific_structure["Reconstructed biopsy cylinder length (from contour data)"]
            if length_mm is None:
                continue

            _set_length_fields(specific_structure,
                               length_mm,
                               'real contour reconstruction',
                               enable_target_dil_v2)

            real_biopsy_lengths_list.append(float(length_mm))

            if dil_centroids_by_ref:
                bx_centroid = np.array(specific_structure["Structure global centroid"]).reshape(3)
                best_refnum = _find_nearest_dil_refnum(bx_centroid, dil_centroids_by_ref)
                if best_refnum is not None:
                    real_bx_lengths_by_dil[patientUID][best_refnum].append(float(length_mm))

    if len(real_biopsy_lengths_list) >= 1:
        real_biopsy_lengths_arr = np.array(real_biopsy_lengths_list, dtype=float)
        mean_of_real_biopsy_lengths = float(np.mean(real_biopsy_lengths_arr))
        std_of_real_biopsy_lengths = float(np.std(real_biopsy_lengths_arr))
    else:
        mean_of_real_biopsy_lengths = float(biopsy_needle_compartment_length)
        std_of_real_biopsy_lengths = 0.0

    for patientUID, pydicom_item in master_structure_reference_dict.items():
        patient_v2_info_dict = pydicom_item[all_ref_key]["Multi-structure information dict (not for csv output)"][TARGET_DIL_V2_INFO_DICT_KEY]
        patient_v2_info_dict["Enabled"] = enable_target_dil_v2
        patient_v2_info_dict["Length sidecar complete"] = True
        patient_v2_info_dict["Length sidecar method"] = simulated_biopsy_length_method
        patient_v2_info_dict["Real biopsy length mean mm"] = mean_of_real_biopsy_lengths
        patient_v2_info_dict["Real biopsy length std mm"] = std_of_real_biopsy_lengths

        for specific_structure in pydicom_item[bx_ref]:
            if specific_structure["Simulated bool"] == False:
                continue

            length_mm, length_source = _determine_simulated_length(patientUID,
                                                                   specific_structure,
                                                                   simulated_biopsy_length_method,
                                                                   biopsy_needle_compartment_length,
                                                                   mean_of_real_biopsy_lengths,
                                                                   std_of_real_biopsy_lengths,
                                                                   real_bx_lengths_by_dil,
                                                                   dil_ref)

            _set_length_fields(specific_structure,
                               length_mm,
                               length_source,
                               enable_target_dil_v2)

    real_bx_lengths_by_dil_standard_dict = {}
    for patientUID, patient_lengths_dict in real_bx_lengths_by_dil.items():
        real_bx_lengths_by_dil_standard_dict[patientUID] = dict(patient_lengths_dict)

    return {
        "real_biopsy_lengths_list": real_biopsy_lengths_list,
        "real_bx_lengths_by_dil": real_bx_lengths_by_dil_standard_dict,
        "mean_of_real_biopsy_lengths": mean_of_real_biopsy_lengths,
        "std_of_real_biopsy_lengths": std_of_real_biopsy_lengths,
    }