from datetime import datetime
import sys
import tkinter as tk
from tkinter import filedialog as fd

import numpy as np
import pandas

import uncertainty_file_writer


class uncertainty_data:
    def __init__(self, patientUID, struct_type, structure_roi, struct_ref_num, master_ref_dict_specific_structure_index, frame_of_reference):
        self.patientUID = patientUID
        self.struct_type = struct_type
        self.structure_roi = structure_roi
        self.struct_ref_num = struct_ref_num
        self.master_ref_dict_specific_structure_index = master_ref_dict_specific_structure_index
        self.uncertainty_data_mean_arr = None
        self.uncertainty_data_sigma_arr = None
        self.uncertainty_data_info_dict = {"Frame of reference": frame_of_reference, "Distribution": 'Normal'}

    def fill_means_and_sigmas(self, means_arr, sigmas_arr, means_arr_dilations, sigmas_arr_dilations, means_arr_rotations, sigmas_arr_rotations):
        self.uncertainty_data_mean_arr = means_arr
        self.uncertainty_data_sigma_arr = sigmas_arr
        self.uncertainty_data_dilations_mean_arr = means_arr_dilations
        self.uncertainty_data_dilations_sigma_arr = sigmas_arr_dilations
        self.uncertainty_data_rotations_mean_arr = means_arr_rotations
        self.uncertainty_data_rotations_sigma_arr = sigmas_arr_rotations


def attach_patient_uncertainty_data_from_dataframe(*,
                                                   patient_uid,
                                                   pydicom_item,
                                                   read_uncertainties_dataframe,
                                                   uncertainty_data_cls):
    patient_uncertainties_dataframe = read_uncertainties_dataframe[
        read_uncertainties_dataframe["Patient UID"] == patient_uid
    ]

    attached_uncertainty_count = 0

    for _, row in patient_uncertainties_dataframe.iterrows():
        structure_type = row["Structure type"]
        structure_ROI = row["Structure ID"]
        structure_ref_num = row["Structure dicom ref num"]
        master_ref_dict_specific_structure_index = row["Structure index"]
        frame_of_reference = row["Frame of reference"]

        means_arr = np.array([row["mu (X)"],
                              row["mu (Y)"],
                              row["mu (Z)"]], dtype=float)
        sigmas_arr = np.array([row["sigma (X)"],
                               row["sigma (Y)"],
                               row["sigma (Z)"]], dtype=float)
        means_arr_dilations = np.array([row["Dilations mu (XY)"],
                                        row["Dilations mu (Z)"]], dtype=float)
        sigmas_arr_dilations = np.array([row["Dilations sigma (XY)"],
                                         row["Dilations sigma (Z)"]], dtype=float)
        means_arr_rotations = np.array([row["Rotations mu (X)"],
                                        row["Rotations mu (Y)"],
                                        row["Rotations mu (Z)"]], dtype=float)
        sigmas_arr_rotations = np.array([row["Rotations sigma (X)"],
                                         row["Rotations sigma (Y)"],
                                         row["Rotations sigma (Z)"]], dtype=float)

        uncertainty_data_obj = uncertainty_data_cls(patient_uid,
                                                    structure_type,
                                                    structure_ROI,
                                                    structure_ref_num,
                                                    master_ref_dict_specific_structure_index,
                                                    frame_of_reference)

        uncertainty_data_obj.fill_means_and_sigmas(means_arr,
                                                   sigmas_arr,
                                                   means_arr_dilations,
                                                   sigmas_arr_dilations,
                                                   means_arr_rotations,
                                                   sigmas_arr_rotations)

        pydicom_item[structure_type][master_ref_dict_specific_structure_index]["Uncertainty data"] = uncertainty_data_obj
        attached_uncertainty_count += 1

    return attached_uncertainty_count


def attach_uncertainty_data_from_dataframe(master_structure_reference_dict,
                                           read_uncertainties_dataframe,
                                           uncertainty_data_cls
                                           ):
    for _, row in read_uncertainties_dataframe.iterrows():
        patientUID = row["Patient UID"]
        structure_type = row["Structure type"]
        structure_ROI = row["Structure ID"]
        structure_ref_num = row["Structure dicom ref num"]
        master_ref_dict_specific_structure_index = row["Structure index"]
        frame_of_reference = row["Frame of reference"]

        means_arr = np.array([row["mu (X)"],
                              row["mu (Y)"],
                              row["mu (Z)"]], dtype=float)
        sigmas_arr = np.array([row["sigma (X)"],
                               row["sigma (Y)"],
                               row["sigma (Z)"]], dtype=float)
        means_arr_dilations = np.array([row["Dilations mu (XY)"],
                                        row["Dilations mu (Z)"]], dtype=float)
        sigmas_arr_dilations = np.array([row["Dilations sigma (XY)"],
                                         row["Dilations sigma (Z)"]], dtype=float)
        means_arr_rotations = np.array([row["Rotations mu (X)"],
                                        row["Rotations mu (Y)"],
                                        row["Rotations mu (Z)"]], dtype=float)
        sigmas_arr_rotations = np.array([row["Rotations sigma (X)"],
                                         row["Rotations sigma (Y)"],
                                         row["Rotations sigma (Z)"]], dtype=float)

        uncertainty_data_obj = uncertainty_data_cls(patientUID,
                                                    structure_type,
                                                    structure_ROI,
                                                    structure_ref_num,
                                                    master_ref_dict_specific_structure_index,
                                                    frame_of_reference)

        uncertainty_data_obj.fill_means_and_sigmas(means_arr,
                                                   sigmas_arr,
                                                   means_arr_dilations,
                                                   sigmas_arr_dilations,
                                                   means_arr_rotations,
                                                   sigmas_arr_rotations)

        master_structure_reference_dict[patientUID][structure_type][master_ref_dict_specific_structure_index]["Uncertainty data"] = uncertainty_data_obj


def prepare_and_attach_uncertainty_data(master_structure_reference_dict,
                                        master_structure_info_dict,
                                        master_cohort_patient_data_and_dataframes,
                                        structs_referenced_list,
                                        structs_referenced_dict,
                                        biopsy_variation_uncertainty_setting,
                                        non_biopsy_variation_uncertainty_setting,
                                        use_added_in_quad_errors_as,
                                        uncertainty_dir,
                                        uncertainty_file_name,
                                        uncertainty_file_extension,
                                        modify_generated_uncertainty_template,
                                        data_dir,
                                        ques_funcs,
                                        stopwatch,
                                        live_display,
                                        uncertainty_data_cls
                                        ):
    date_time_now = datetime.now()
    date_time_now_file_name_format = date_time_now.strftime(" Date-%b-%d-%Y Time-%H,%M,%S")
    uncertainties_file = uncertainty_dir.joinpath(uncertainty_file_name + date_time_now_file_name_format + uncertainty_file_extension)

    uncertainties_dataframe = uncertainty_file_writer.uncertainty_file_preper_by_struct_type_dataframe_NEW(
        master_structure_reference_dict,
        structs_referenced_list,
        structs_referenced_dict,
        biopsy_variation_uncertainty_setting,
        non_biopsy_variation_uncertainty_setting,
        use_added_in_quad_errors_as,
        master_structure_info_dict,
    )

    uncertainties_dataframe.to_csv(uncertainties_file)
    master_cohort_patient_data_and_dataframes["Dataframes"]["Uncertainties dataframe (unedited)"] = uncertainties_dataframe

    if modify_generated_uncertainty_template == True:
        live_display.stop()
        live_display.console.print("[bold red]User input required:")
        uncertainty_file_ready = False
        while uncertainty_file_ready == False:
            stopwatch.stop()
            uncertainty_file_ready = ques_funcs.ask_ok('>You indicated in launch params that you would like to modify the uncertainty file. Is the uncertainty file prepared/filled out?')
            stopwatch.start()
            if uncertainty_file_ready == True:
                print('>Please select the file with the dialog box')
                root = tk.Tk()
                root.withdraw()
                uncertainties_file_filled = fd.askopenfilename(title='Open the uncertainties data file', initialdir=data_dir, filetypes=[("Excel files", ".xlsx .xls .csv")])
                read_uncertainties_dataframe = pandas.read_csv(uncertainties_file_filled)
                print(read_uncertainties_dataframe)

            else:
                print('>Please fill out the generated uncertainties file generated at ', uncertainties_file)
                stopwatch.stop()
                ask_to_quit = ques_funcs.ask_ok('>Would you like to quit the programme instead?')
                stopwatch.start()
                if ask_to_quit == True:
                    sys.exit(">You have quit the programme.")

    else:
        uncertainties_file_filled = uncertainties_file
        read_uncertainties_dataframe = pandas.read_csv(uncertainties_file_filled)

    master_cohort_patient_data_and_dataframes["Dataframes"]["Uncertainties dataframe (final)"] = read_uncertainties_dataframe
    attach_uncertainty_data_from_dataframe(
        master_structure_reference_dict,
        read_uncertainties_dataframe,
        uncertainty_data_cls,
    )
    live_display.start()

    return uncertainties_file, uncertainties_file_filled, read_uncertainties_dataframe, live_display