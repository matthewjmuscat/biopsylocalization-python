import cupy as cp
import cupy_functions


def generate_transformations_for_patient(*,
                                         patient_uid,
                                         pydicom_item,
                                         simulate_uniform_bx_shifts_due_to_bx_needle_compartment,
                                         bx_ref,
                                         biopsy_needle_compartment_length,
                                         num_generated_transform_samples,
                                         structs_referenced_list,
                                         rng=None):
    """Generate and attach one patient's MC transform-bank samples."""
    sp_bx_structure_uniform_dist_shift_samples_and_structure_reference_list = cupy_functions.MC_simulator_all_structs_dilations_generator_cupy(
        pydicom_item,
        structs_referenced_list,
        num_generated_transform_samples,
        rng=rng,
    )
    for generated_shifts_info_list in sp_bx_structure_uniform_dist_shift_samples_and_structure_reference_list:
        structure_type = generated_shifts_info_list[0]
        specific_structure_index = generated_shifts_info_list[1]
        specific_structure_normal_dist_dilations_samples_arr = generated_shifts_info_list[2]
        pydicom_item[structure_type][specific_structure_index]["MC data: Generated normal dist random samples dilations arr"] = specific_structure_normal_dist_dilations_samples_arr

    sp_bx_structure_uniform_dist_shift_samples_and_structure_reference_list = cupy_functions.MC_simulator_all_structs_rotations_generator_cupy(
        pydicom_item,
        structs_referenced_list,
        num_generated_transform_samples,
        rng=rng,
    )
    for generated_shifts_info_list in sp_bx_structure_uniform_dist_shift_samples_and_structure_reference_list:
        structure_type = generated_shifts_info_list[0]
        specific_structure_index = generated_shifts_info_list[1]
        specific_structure_normal_dist_rotations_samples_arr = generated_shifts_info_list[2]
        pydicom_item[structure_type][specific_structure_index]["MC data: Generated normal dist random samples rotations arr"] = specific_structure_normal_dist_rotations_samples_arr

    if simulate_uniform_bx_shifts_due_to_bx_needle_compartment == True:
        sp_bx_structure_uniform_dist_shift_samples_and_structure_reference_list = cupy_functions.MC_simulator_shift_biopsy_structures_uniform_generator_cupy(
            pydicom_item,
            bx_ref,
            biopsy_needle_compartment_length,
            num_generated_transform_samples,
            rng=rng,
        )
        for generated_shifts_info_list in sp_bx_structure_uniform_dist_shift_samples_and_structure_reference_list:
            structure_type = generated_shifts_info_list[0]
            specific_structure_index = generated_shifts_info_list[1]
            specific_structure_structure_uniform_dist_shift_samples_arr = generated_shifts_info_list[2]
            pydicom_item[structure_type][specific_structure_index]["MC data: Generated uniform dist (biopsy needle compartment) random distance (z_needle) samples arr"] = specific_structure_structure_uniform_dist_shift_samples_arr

    sp_structure_normal_dist_shift_samples_and_structure_reference_list = cupy_functions.MC_simulator_shift_all_structures_generator_cupy(
        pydicom_item,
        structs_referenced_list,
        num_generated_transform_samples,
        rng=rng,
    )
    for generated_shifts_info_list in sp_structure_normal_dist_shift_samples_and_structure_reference_list:
        structure_type = generated_shifts_info_list[0]
        specific_structure_index = generated_shifts_info_list[1]
        specific_structure_structure_normal_dist_shift_samples_arr = generated_shifts_info_list[2]
        pydicom_item[structure_type][specific_structure_index]["MC data: Generated normal dist random samples arr"] = cp.asnumpy(specific_structure_structure_normal_dist_shift_samples_arr)

    return {
        "patient_uid": patient_uid,
        "pydicom_item": pydicom_item,
    }