import copy

import pandas
from pandas.testing import assert_frame_equal

import dataframe_builders
import plotting_funcs


PROSTATE_ONLY_MR_ADC_TEMP_DF_KEY = "Prostate only points MR ADC dataframe (temporary for pre-processing)"
MR_ADC_SUMMARY_BY_STRUCTURE_DF_KEY = "MR - ADC - summary statistics by structure dataframe"
SELECTED_STRUCTURES_DF_KEY = "Selected structures"
PROSTATE_ONLY_MR_ADC_VALIDATION_DF_KEY = "Prostate only MR ADC validation"
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


def process_patient_prostate_only_mr_adc_legacy(
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

    sp_patient_selected_structure_info_dataframe = pydicom_item[all_ref_key][
        MULTI_STRUCTURE_PREPROCESSING_OUTPUT_DF_DICT_KEY
    ][SELECTED_STRUCTURES_DF_KEY]

    list_of_additional_pcds = []
    for _, row in sp_patient_selected_structure_info_dataframe.iterrows():
        struct_type = row["Struct ref type"]
        struct_found_bool = row["Struct found bool"]
        if struct_found_bool == True:
            struct_index = row["Index number"]
            interpolated_pcd_dict = pydicom_item[struct_type][struct_index]["Interpolated structure point cloud dict"]
            list_of_additional_pcds.append(interpolated_pcd_dict['Full with end caps'])

    for _specific_dil_structure_index, specific_dil_structure in enumerate(pydicom_item[dil_ref]):
        interpolated_pcd_dict = specific_dil_structure["Interpolated structure point cloud dict"]
        list_of_additional_pcds.append(interpolated_pcd_dict['Full with end caps'])

    indeterminate_task = indeterminate_progress_sub.add_task(
        "[cyan]~~Demonstrating prostate only correctness",
        total=None,
    )

    containment_info_for_all_lattice_points_grand_pandas_dataframe_prostate_only = pydicom_item[all_ref_key][
        MULTI_STRUCTURE_PREPROCESSING_OUTPUT_DF_DICT_KEY
    ][PROSTATE_ONLY_MR_ADC_TEMP_DF_KEY]

    if demonstrate_mr_adc_pcd_containment_correctness_prostate_only_all_other_structures_removed_bool == True:
        plotting_funcs.plot_containment_info_dataframe_to_point_cloud_plus_other_clouds(
            containment_info_for_all_lattice_points_grand_pandas_dataframe_prostate_only,
            "Test pt X",
            "Test pt Y",
            "Test pt Z",
            "Pt clr R",
            "Pt clr G",
            "Pt clr B",
            additional_point_clouds=list_of_additional_pcds,
        )

    indeterminate_progress_sub.update(indeterminate_task, visible=False)

    indeterminate_task = indeterminate_progress_sub.add_task(
        "[cyan]~~Calculating MR ADC statistics (Prostate - UDR)",
        total=None,
    )

    mr_adc_value_column_name_str = "MR ADC value"

    mr_adc_value_summary_statistics_prostate_only_excluding_UDR = dataframe_builders.dataframe_mr_summary_statistics(
        containment_info_for_all_lattice_points_grand_pandas_dataframe_prostate_only,
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
                mr_adc_value_summary_statistics_prostate_only_excluding_UDR,
            ],
            ignore_index=True,
        )
        output_dataframes_dict[MR_ADC_SUMMARY_BY_STRUCTURE_DF_KEY] = (
            mr_adc_value_summary_statistics_specific_structure_master
        )

    elif output_dataframes_dict[MR_ADC_SUMMARY_BY_STRUCTURE_DF_KEY] is None:
        output_dataframes_dict[MR_ADC_SUMMARY_BY_STRUCTURE_DF_KEY] = (
            mr_adc_value_summary_statistics_prostate_only_excluding_UDR
        )

    del containment_info_for_all_lattice_points_grand_pandas_dataframe_prostate_only
    del output_dataframes_dict[PROSTATE_ONLY_MR_ADC_TEMP_DF_KEY]

    indeterminate_progress_sub.update(indeterminate_task, visible=False)


def capture_patient_prostate_only_mr_adc_live_state(
        *,
        pydicom_item,
        all_ref_key):
    return copy.deepcopy(
        pydicom_item[all_ref_key][MULTI_STRUCTURE_PREPROCESSING_OUTPUT_DF_DICT_KEY]
    )


def restore_patient_prostate_only_mr_adc_live_state(
        *,
        pydicom_item,
        all_ref_key,
        saved_state):
    if saved_state is None:
        return
    pydicom_item[all_ref_key][MULTI_STRUCTURE_PREPROCESSING_OUTPUT_DF_DICT_KEY] = saved_state


def capture_patient_prostate_only_mr_adc_validation_snapshot(
        *,
        patient_uid,
        pydicom_item,
        all_ref_key,
        mr_adc_ref):
    output_dataframes_dict = pydicom_item[all_ref_key][MULTI_STRUCTURE_PREPROCESSING_OUTPUT_DF_DICT_KEY]
    return {
        "patient_uid": patient_uid,
        "has_mr_adc": mr_adc_ref in pydicom_item,
        "mr_adc_summary_dataframe": copy.deepcopy(output_dataframes_dict.get(MR_ADC_SUMMARY_BY_STRUCTURE_DF_KEY)),
        "prostate_only_temp_dataframe_present": PROSTATE_ONLY_MR_ADC_TEMP_DF_KEY in output_dataframes_dict,
        "prostate_only_temp_dataframe": copy.deepcopy(output_dataframes_dict.get(PROSTATE_ONLY_MR_ADC_TEMP_DF_KEY)),
    }


def compare_patient_prostate_only_mr_adc_validation_snapshots(modular_snapshot, legacy_snapshot):
    mismatches = []
    for snapshot_field in (
            "patient_uid",
            "has_mr_adc",
            "prostate_only_temp_dataframe_present"):
        if modular_snapshot[snapshot_field] != legacy_snapshot[snapshot_field]:
            mismatches.append(
                f"{snapshot_field}: modular={modular_snapshot[snapshot_field]} legacy={legacy_snapshot[snapshot_field]}"
            )

    for dataframe_field in (
            "mr_adc_summary_dataframe",
            "prostate_only_temp_dataframe"):
        modular_dataframe = modular_snapshot[dataframe_field]
        legacy_dataframe = legacy_snapshot[dataframe_field]
        if modular_dataframe is None or legacy_dataframe is None:
            if modular_dataframe is not legacy_dataframe:
                mismatches.append(f"{dataframe_field}: dataframe presence mismatch")
            continue
        try:
            assert_frame_equal(
                modular_dataframe,
                legacy_dataframe,
                check_dtype=True,
                check_like=False,
            )
        except AssertionError as exc:
            mismatches.append(f"{dataframe_field}: {str(exc).splitlines()[0]}")

    return {
        "patient_uid": legacy_snapshot["patient_uid"],
        "overall_match_bool": len(mismatches) == 0,
        "mismatch_count": len(mismatches),
        "mismatch_fields": [mismatch.split(":", 1)[0] for mismatch in mismatches],
        "mismatch_summary": " | ".join(mismatches[:10]),
    }


def append_patient_prostate_only_mr_adc_validation_result(
        *,
        pydicom_item,
        all_ref_key,
        validation_result):
    output_dataframes_dict = pydicom_item[all_ref_key][MULTI_STRUCTURE_PREPROCESSING_OUTPUT_DF_DICT_KEY]
    validation_result_row = pandas.DataFrame(
        {
            "Patient ID": [validation_result["patient_uid"]],
            "Overall match bool": [validation_result["overall_match_bool"]],
            "Mismatch count": [validation_result["mismatch_count"]],
            "Mismatch fields": ["; ".join(validation_result["mismatch_fields"])],
            "Mismatch summary": [validation_result["mismatch_summary"]],
        }
    )
    existing_validation_dataframe = output_dataframes_dict.get(PROSTATE_ONLY_MR_ADC_VALIDATION_DF_KEY)
    if existing_validation_dataframe is None:
        output_dataframes_dict[PROSTATE_ONLY_MR_ADC_VALIDATION_DF_KEY] = validation_result_row
        return

    output_dataframes_dict[PROSTATE_ONLY_MR_ADC_VALIDATION_DF_KEY] = pandas.concat(
        [existing_validation_dataframe, validation_result_row],
        ignore_index=True,
    )


def validate_patient_prostate_only_mr_adc_against_legacy(
        *,
        patient_uid,
        pydicom_item,
        all_ref_key,
        dil_ref,
        mr_adc_ref,
        demonstrate_mr_adc_pcd_containment_correctness_prostate_only_all_other_structures_removed_bool,
        indeterminate_progress_sub,
        modular_pre_stage_state,
        modular_live_state,
        modular_validation_snapshot,
        important_info=None,
        live_display=None,
        runtime_logger=None):
    restore_patient_prostate_only_mr_adc_live_state(
        pydicom_item=pydicom_item,
        all_ref_key=all_ref_key,
        saved_state=modular_pre_stage_state,
    )
    process_patient_prostate_only_mr_adc_legacy(
        patient_uid=patient_uid,
        pydicom_item=pydicom_item,
        all_ref_key=all_ref_key,
        dil_ref=dil_ref,
        mr_adc_ref=mr_adc_ref,
        demonstrate_mr_adc_pcd_containment_correctness_prostate_only_all_other_structures_removed_bool=(
            demonstrate_mr_adc_pcd_containment_correctness_prostate_only_all_other_structures_removed_bool
        ),
        indeterminate_progress_sub=indeterminate_progress_sub,
    )
    legacy_validation_snapshot = capture_patient_prostate_only_mr_adc_validation_snapshot(
        patient_uid=patient_uid,
        pydicom_item=pydicom_item,
        all_ref_key=all_ref_key,
        mr_adc_ref=mr_adc_ref,
    )
    validation_result = compare_patient_prostate_only_mr_adc_validation_snapshots(
        modular_validation_snapshot,
        legacy_validation_snapshot,
    )
    restore_patient_prostate_only_mr_adc_live_state(
        pydicom_item=pydicom_item,
        all_ref_key=all_ref_key,
        saved_state=modular_live_state,
    )
    append_patient_prostate_only_mr_adc_validation_result(
        pydicom_item=pydicom_item,
        all_ref_key=all_ref_key,
        validation_result=validation_result,
    )
    if runtime_logger is not None:
        runtime_logger.checkpoint(
            "preprocessing.prostate_only_mr_adc.validation",
            "Validated modular prostate-only MR ADC summary against legacy inline path.",
            details={
                "patient_uid": validation_result["patient_uid"],
                "overall_match_bool": validation_result["overall_match_bool"],
                "mismatch_count": validation_result["mismatch_count"],
                "mismatch_fields": validation_result["mismatch_fields"][:10],
            },
        )
    if validation_result["overall_match_bool"] == False and important_info is not None:
        mismatch_summary = validation_result["mismatch_summary"]
        if len(mismatch_summary) > 400:
            mismatch_summary = mismatch_summary[:400] + "..."
        important_info.add_text_line(
            f"WARNING! Modular prostate-only MR ADC mismatch for patient {patient_uid}. {mismatch_summary}",
            live_display,
        )


def prostate_only_mr_adc_processer(master_structure_reference_dict,
                                   master_structure_info_dict,
                                   all_ref_key,
                                   dil_ref,
                                   mr_adc_ref,
                                   demonstrate_mr_adc_pcd_containment_correctness_prostate_only_all_other_structures_removed_bool,
                                   patients_progress,
                                   completed_progress,
                                   indeterminate_progress_sub,
                                   live_display,
                                   validate_against_legacy=False,
                                   important_info=None,
                                   runtime_logger=None):
    patientUID_default = "Initializing"
    processing_patients_task_main_description = "[red]Computing Prostate Only MR Values [{}]...".format(patientUID_default)
    processing_patients_task_completed_main_description = "[green]Computing Prostate Only MR Values "
    processing_patients_task = patients_progress.add_task(processing_patients_task_main_description, total=master_structure_info_dict["Global"]["Num cases"])
    processing_patients_task_completed = completed_progress.add_task(processing_patients_task_completed_main_description, total=master_structure_info_dict["Global"]["Num cases"], visible=False)

    for patientUID, pydicom_item in master_structure_reference_dict.items():
        processing_patients_task_main_description = "[red]Computing Prostate Only MR Values  [{}]...".format(patientUID)
        patients_progress.update(processing_patients_task, description=processing_patients_task_main_description)

        modular_pre_stage_state = None
        if validate_against_legacy == True:
            modular_pre_stage_state = capture_patient_prostate_only_mr_adc_live_state(
                pydicom_item=pydicom_item,
                all_ref_key=all_ref_key,
            )

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

        if validate_against_legacy == True:
            modular_validation_snapshot = capture_patient_prostate_only_mr_adc_validation_snapshot(
                patient_uid=patientUID,
                pydicom_item=pydicom_item,
                all_ref_key=all_ref_key,
                mr_adc_ref=mr_adc_ref,
            )
            modular_live_state = capture_patient_prostate_only_mr_adc_live_state(
                pydicom_item=pydicom_item,
                all_ref_key=all_ref_key,
            )
            validate_patient_prostate_only_mr_adc_against_legacy(
                patient_uid=patientUID,
                pydicom_item=pydicom_item,
                all_ref_key=all_ref_key,
                dil_ref=dil_ref,
                mr_adc_ref=mr_adc_ref,
                demonstrate_mr_adc_pcd_containment_correctness_prostate_only_all_other_structures_removed_bool=(
                    demonstrate_mr_adc_pcd_containment_correctness_prostate_only_all_other_structures_removed_bool
                ),
                indeterminate_progress_sub=indeterminate_progress_sub,
                modular_pre_stage_state=modular_pre_stage_state,
                modular_live_state=modular_live_state,
                modular_validation_snapshot=modular_validation_snapshot,
                important_info=important_info,
                live_display=live_display,
                runtime_logger=runtime_logger,
            )

        patients_progress.update(processing_patients_task, advance=1)
        completed_progress.update(processing_patients_task_completed, advance=1)
    patients_progress.update(processing_patients_task, visible=False)
    completed_progress.update(processing_patients_task_completed, visible=True)

    return live_display