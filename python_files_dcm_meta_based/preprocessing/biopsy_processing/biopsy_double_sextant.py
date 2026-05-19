import pandas

import dataframe_builders
import misc_tools


PER_SAMPLE_POINT_DOUBLE_SEXTANT_DF_KEY = "Per sample point prostate double sextant classification"
PER_VOXEL_DOUBLE_SEXTANT_DF_KEY = "Per voxel prostate double sextant classification"


def _concat_dataframes(dataframes):
    if not dataframes:
        return pandas.DataFrame()
    return pandas.concat(dataframes, ignore_index=True)


def _build_patient_per_sample_point_double_sextant_dataframe(patient_uid,
                                                             pydicom_item,
                                                             all_ref_key,
                                                             bx_ref,
                                                             oar_ref,
                                                             biopsy_z_voxel_length):
    """Build the per-patient sample-point prostate double-sextant table."""

    patient_per_sample_point_double_sextant_prostate_location_df = pandas.DataFrame()

    for specific_structure_index, specific_structure in enumerate(pydicom_item[bx_ref]):
        simulated_type = specific_structure["Simulated type"]
        simulated_bool = specific_structure["Simulated bool"]
        bx_refnum = specific_structure["Ref #"]
        specific_bx_structure_roi = specific_structure["ROI"]
        sampled_bx_points_arr = specific_structure["Random uniformly sampled volume pts arr"]
        bx_points_bx_coords_sys_arr = specific_structure["Random uniformly sampled volume pts bx coord sys arr"]

        n_pts = len(sampled_bx_points_arr)
        biopsy_per_voxel_double_sextant_df = pandas.DataFrame(
            {
                "Patient ID": [patient_uid] * n_pts,
                "Structure type": [bx_ref] * n_pts,
                "Bx index": [specific_structure_index] * n_pts,
                "Bx ID": [specific_bx_structure_roi] * n_pts,
                "Simulated type": [simulated_type] * n_pts,
                "Simulated bool": [simulated_bool] * n_pts,
                "Bx refnum": [bx_refnum] * n_pts,
                "Original pt index": list(range(n_pts)),
                "X (global)": [pt[0] for pt in sampled_bx_points_arr],
                "Y (global)": [pt[1] for pt in sampled_bx_points_arr],
                "Z (global)": [pt[2] for pt in sampled_bx_points_arr],
            }
        )

        biopsy_per_voxel_double_sextant_with_bx_frame_coords_df = misc_tools.include_vector_columns_in_dataframe(
            biopsy_per_voxel_double_sextant_df,
            bx_points_bx_coords_sys_arr,
            reference_column_name='Original pt index',
            new_column_name_x="X (Bx frame)",
            new_column_name_y="Y (Bx frame)",
            new_column_name_z="Z (Bx frame)",
            in_place=False,
        )

        biopsy_per_voxel_double_sextant_with_bx_frame_coords_and_voxel_index_df = dataframe_builders.add_voxel_columns_helper_func(
            biopsy_per_voxel_double_sextant_with_bx_frame_coords_df,
            biopsy_z_voxel_length,
            "Z (Bx frame)",
            in_place=False,
        )

        sp_patient_selected_structure_info_dataframe = pydicom_item[all_ref_key]["Multi-structure pre-processing output dataframes dict"]["Selected structures"]
        specific_prostate_info_df = sp_patient_selected_structure_info_dataframe[sp_patient_selected_structure_info_dataframe["Struct ref type"] == oar_ref]
        selected_prostate_info = specific_prostate_info_df.to_dict('records')[0]
        prostate_found_bool = selected_prostate_info["Struct found bool"]

        if prostate_found_bool:
            prostate_structure_index = selected_prostate_info["Index number"]
            prostate_structure = pydicom_item[oar_ref][prostate_structure_index]
            biopsy_per_voxel_double_sextant_with_sextants_df = misc_tools.classify_voxels_in_prostate_frame_sextant(
                biopsy_per_voxel_double_sextant_with_bx_frame_coords_and_voxel_index_df,
                prostate_structure=prostate_structure,
                global_coord_cols=("X (global)", "Y (global)", "Z (global)"),
                lr_col="Bx voxel prostate sextant (LR)",
                ap_col="Bx voxel prostate sextant (AP)",
                si_col="Bx voxel prostate sextant (SI)",
            )
        else:
            biopsy_per_voxel_double_sextant_with_sextants_df = biopsy_per_voxel_double_sextant_with_bx_frame_coords_and_voxel_index_df.copy()
            biopsy_per_voxel_double_sextant_with_sextants_df["Bx voxel prostate sextant (LR)"] = None
            biopsy_per_voxel_double_sextant_with_sextants_df["Bx voxel prostate sextant (AP)"] = None
            biopsy_per_voxel_double_sextant_with_sextants_df["Bx voxel prostate sextant (SI)"] = None

        patient_per_sample_point_double_sextant_prostate_location_df = pandas.concat(
            [
                patient_per_sample_point_double_sextant_prostate_location_df,
                biopsy_per_voxel_double_sextant_with_sextants_df,
            ],
            ignore_index=True,
        )

    return patient_per_sample_point_double_sextant_prostate_location_df


def _build_per_voxel_double_sextant_dataframe(per_sample_point_double_sextant_df):
    if per_sample_point_double_sextant_df.empty:
        return pandas.DataFrame()

    return (
        per_sample_point_double_sextant_df
        .groupby(
            ["Patient ID", "Bx ID", "Bx index", "Voxel index", "Simulated type", "Simulated bool", "Bx refnum"],
            as_index=False,
        )
        .agg(
            {
                "Bx voxel prostate sextant (LR)": misc_tools.majority_with_random_tie,
                "Bx voxel prostate sextant (AP)": misc_tools.majority_with_random_tie,
                "Bx voxel prostate sextant (SI)": misc_tools.majority_with_random_tie,
            }
        )
    )


def _build_per_sample_point_double_sextant_dataframe(master_structure_reference_dict,
                                                     all_ref_key,
                                                     bx_ref,
                                                     oar_ref,
                                                     biopsy_z_voxel_length):
    patient_dataframes = []
    for patient_uid, pydicom_item in master_structure_reference_dict.items():
        patient_dataframes.append(
            _build_patient_per_sample_point_double_sextant_dataframe(
                patient_uid,
                pydicom_item,
                all_ref_key,
                bx_ref,
                oar_ref,
                biopsy_z_voxel_length,
            )
        )
    return _concat_dataframes(patient_dataframes)


def biopsy_double_sextant_processer(master_structure_reference_dict,
                                    master_cohort_patient_data_and_dataframes,
                                    all_ref_key,
                                    bx_ref,
                                    oar_ref,
                                    biopsy_z_voxel_length,
                                    live_display):
    live_display.stop()

    patient_sample_point_dataframes = []
    for patient_uid, pydicom_item in master_structure_reference_dict.items():
        patient_sample_point_df = _build_patient_per_sample_point_double_sextant_dataframe(
            patient_uid,
            pydicom_item,
            all_ref_key,
            bx_ref,
            oar_ref,
            biopsy_z_voxel_length,
        )
        pydicom_item[all_ref_key]["Multi-structure pre-processing output dataframes dict"][PER_SAMPLE_POINT_DOUBLE_SEXTANT_DF_KEY] = patient_sample_point_df
        patient_sample_point_dataframes.append(patient_sample_point_df)

    all_biopsies_per_sample_point_double_sextant_prostate_location_df = _concat_dataframes(patient_sample_point_dataframes)
    all_biopsies_per_voxel_double_sextant_prostate_location_df = _build_per_voxel_double_sextant_dataframe(
        all_biopsies_per_sample_point_double_sextant_prostate_location_df
    )

    for patient_uid, pydicom_item in master_structure_reference_dict.items():
        if "Patient ID" in all_biopsies_per_voxel_double_sextant_prostate_location_df.columns:
            patient_voxel_df = all_biopsies_per_voxel_double_sextant_prostate_location_df[
                all_biopsies_per_voxel_double_sextant_prostate_location_df["Patient ID"].eq(patient_uid)
            ].reset_index(drop=True)
        else:
            patient_voxel_df = pandas.DataFrame()
        pydicom_item[all_ref_key]["Multi-structure pre-processing output dataframes dict"][PER_VOXEL_DOUBLE_SEXTANT_DF_KEY] = patient_voxel_df

    master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: Per sample point prostate double sextant classification"] = all_biopsies_per_sample_point_double_sextant_prostate_location_df
    master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: Per voxel prostate double sextant classification"] = all_biopsies_per_voxel_double_sextant_prostate_location_df

    live_display.start()
    return live_display