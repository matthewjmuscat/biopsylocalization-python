import pandas

import dataframe_builders
import plotting_funcs


PROSTATE_ONLY_MR_ADC_TEMP_DF_KEY = "Prostate only points MR ADC dataframe (temporary for pre-processing)"
MR_ADC_SUMMARY_BY_STRUCTURE_DF_KEY = "MR - ADC - summary statistics by structure dataframe"
SELECTED_STRUCTURES_DF_KEY = "Selected structures"
MULTI_STRUCTURE_PREPROCESSING_OUTPUT_DF_DICT_KEY = "Multi-structure pre-processing output dataframes dict"


def collect_patient_selected_structure_point_clouds(
        *,
        pydicom_item,
        all_ref_key,
        dil_ref):
    selected_structure_info_dataframe = pydicom_item[all_ref_key][
        MULTI_STRUCTURE_PREPROCESSING_OUTPUT_DF_DICT_KEY
    ][SELECTED_STRUCTURES_DF_KEY]

    additional_point_clouds = []
    for _row_index, row in selected_structure_info_dataframe.iterrows():
        struct_type = row["Struct ref type"]
        struct_found_bool = row["Struct found bool"]
        if struct_found_bool == True:
            struct_index = row["Index number"]
            interpolated_pcd_dict = pydicom_item[struct_type][struct_index]["Interpolated structure point cloud dict"]
            additional_point_clouds.append(interpolated_pcd_dict['Full with end caps'])

    for _specific_dil_structure_index, specific_dil_structure in enumerate(pydicom_item[dil_ref]):
        interpolated_pcd_dict = specific_dil_structure["Interpolated structure point cloud dict"]
        additional_point_clouds.append(interpolated_pcd_dict['Full with end caps'])

    return additional_point_clouds


def append_patient_prostate_only_mr_adc_summary(
        *,
        pydicom_item,
        all_ref_key,
        prostate_only_mr_adc_dataframe):
    mr_adc_value_column_name_str = "MR ADC value"

    prostate_only_summary_dataframe = dataframe_builders.dataframe_mr_summary_statistics(
        prostate_only_mr_adc_dataframe,
        mr_adc_value_column_name_str,
        filter_column="Pt contained bool",
        filter_value=True,
        id_cols=("Relative structure ROI", "Relative structure type", "Relative structure index"),
        id_values=("Prostate_excluding_UDR", "custom_P-UDR", 0),
    )

    output_dataframes_dict = pydicom_item[all_ref_key][MULTI_STRUCTURE_PREPROCESSING_OUTPUT_DF_DICT_KEY]
    if output_dataframes_dict[MR_ADC_SUMMARY_BY_STRUCTURE_DF_KEY] is not None:
        mr_adc_value_summary_statistics_specific_structure_master = output_dataframes_dict[
            MR_ADC_SUMMARY_BY_STRUCTURE_DF_KEY
        ]
        mr_adc_value_summary_statistics_specific_structure_master = pandas.concat(
            [
                mr_adc_value_summary_statistics_specific_structure_master,
                prostate_only_summary_dataframe,
            ],
            ignore_index=True,
        )
        output_dataframes_dict[MR_ADC_SUMMARY_BY_STRUCTURE_DF_KEY] = (
            mr_adc_value_summary_statistics_specific_structure_master
        )

    elif output_dataframes_dict[MR_ADC_SUMMARY_BY_STRUCTURE_DF_KEY] is None:
        output_dataframes_dict[MR_ADC_SUMMARY_BY_STRUCTURE_DF_KEY] = prostate_only_summary_dataframe


def process_patient_prostate_only_mr_adc(
        *,
        patient_uid,
        pydicom_item,
        all_ref_key,
        dil_ref,
        mr_adc_ref,
        demonstrate_mr_adc_pcd_containment_correctness_prostate_only_all_other_structures_removed_bool,
        indeterminate_progress_sub):
    if mr_adc_ref not in pydicom_item:
        return

    additional_point_clouds = collect_patient_selected_structure_point_clouds(
        pydicom_item=pydicom_item,
        all_ref_key=all_ref_key,
        dil_ref=dil_ref,
    )

    indeterminate_task = indeterminate_progress_sub.add_task(
        "[cyan]~~Demonstrating prostate only correctness",
        total=None,
    )

    output_dataframes_dict = pydicom_item[all_ref_key][MULTI_STRUCTURE_PREPROCESSING_OUTPUT_DF_DICT_KEY]
    prostate_only_mr_adc_dataframe = output_dataframes_dict[PROSTATE_ONLY_MR_ADC_TEMP_DF_KEY]

    if demonstrate_mr_adc_pcd_containment_correctness_prostate_only_all_other_structures_removed_bool == True:
        plotting_funcs.plot_containment_info_dataframe_to_point_cloud_plus_other_clouds(
            prostate_only_mr_adc_dataframe,
            "Test pt X",
            "Test pt Y",
            "Test pt Z",
            "Pt clr R",
            "Pt clr G",
            "Pt clr B",
            additional_point_clouds=additional_point_clouds,
        )

    indeterminate_progress_sub.update(indeterminate_task, visible=False)

    indeterminate_task = indeterminate_progress_sub.add_task(
        "[cyan]~~Calculating MR ADC statistics (Prostate - UDR)",
        total=None,
    )

    append_patient_prostate_only_mr_adc_summary(
        pydicom_item=pydicom_item,
        all_ref_key=all_ref_key,
        prostate_only_mr_adc_dataframe=prostate_only_mr_adc_dataframe,
    )

    del prostate_only_mr_adc_dataframe
    del output_dataframes_dict[PROSTATE_ONLY_MR_ADC_TEMP_DF_KEY]

    indeterminate_progress_sub.update(indeterminate_task, visible=False)


def prostate_only_mr_adc_processer(master_structure_reference_dict,
                                   master_structure_info_dict,
                                   all_ref_key,
                                   dil_ref,
                                   mr_adc_ref,
                                   demonstrate_mr_adc_pcd_containment_correctness_prostate_only_all_other_structures_removed_bool,
                                   patients_progress,
                                   completed_progress,
                                   indeterminate_progress_sub,
                                   live_display):
    patientUID_default = "Initializing"
    processing_patients_task_main_description = "[red]Computing Prostate Only MR Values [{}]...".format(patientUID_default)
    processing_patients_task_completed_main_description = "[green]Computing Prostate Only MR Values "
    processing_patients_task = patients_progress.add_task(processing_patients_task_main_description, total=master_structure_info_dict["Global"]["Num cases"])
    processing_patients_task_completed = completed_progress.add_task(processing_patients_task_completed_main_description, total=master_structure_info_dict["Global"]["Num cases"], visible=False)

    for patientUID, pydicom_item in master_structure_reference_dict.items():
        processing_patients_task_main_description = "[red]Computing Prostate Only MR Values  [{}]...".format(patientUID)
        patients_progress.update(processing_patients_task, description=processing_patients_task_main_description)

        process_patient_prostate_only_mr_adc(
            patient_uid=patientUID,
            pydicom_item=pydicom_item,
            all_ref_key=all_ref_key,
            dil_ref=dil_ref,
            mr_adc_ref=mr_adc_ref,
            demonstrate_mr_adc_pcd_containment_correctness_prostate_only_all_other_structures_removed_bool=(
                demonstrate_mr_adc_pcd_containment_correctness_prostate_only_all_other_structures_removed_bool
            ),
            indeterminate_progress_sub=indeterminate_progress_sub,
        )

        patients_progress.update(processing_patients_task, advance=1)
        completed_progress.update(processing_patients_task_completed, advance=1)
    patients_progress.update(processing_patients_task, visible=False)
    completed_progress.update(processing_patients_task_completed, visible=True)

    return live_display