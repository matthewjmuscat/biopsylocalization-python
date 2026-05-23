from __future__ import annotations

"""Raw-contour pulling adapters for patient preprocessing.

This module moves the main-body RTSTRUCT contour-read block behind a named
boundary. It preserves the current pydicom read path, simulated-biopsy skip,
centroid writes for non-biopsy structures, and existing dictionary keys.
"""

import numpy as np
import pydicom

import centroid_finder


def pull_raw_structure_contour_for_structure(
    pydicom_item,
    rtstruct_dicom_path,
    structure_type,
    specific_structure_index,
    bx_ref,
):
    """Pull raw contour points for one structure entry."""
    specific_structure = pydicom_item[structure_type][specific_structure_index]
    if structure_type == bx_ref:
        simulated_bool = specific_structure["Simulated bool"]
    else:
        simulated_bool = None

    # create points for simulated biopsies to create
    if simulated_bool == True:
        return
        # USED TO CREATE THE SIMULATED BIOPSIES HERE, BUT i CANT BECAUSE I WANT THEIR LENGTHS TO DEPEND ON THE MEAN LENGTH OF THE REAL BIOPSIES!
        #threeDdata_zslice_list = biopsy_creator.biopsy_points_creater_by_transport_for_sim_bxs(centroid_line_vec_list,centroid_first_pos_list,num_centroids_for_sim_bxs,centroid_sep_dist,simulated_bx_rad,plot_simulated_cores_immediately)
    # otherwise just read the data from dicoms
    else:
        threeDdata_zslice_list = []
        with pydicom.dcmread(rtstruct_dicom_path, defer_size = '2 MB') as py_dicom_item:
            for roi_contour_seq_item in py_dicom_item.ROIContourSequence:
                if int(roi_contour_seq_item["ReferencedROINumber"].value) == int(specific_structure["Ref #"]):
                    structure_contour_points_raw_sequence = roi_contour_seq_item.ContourSequence[0:]
                    break
                else:
                    pass
        for index, slice_object in enumerate(structure_contour_points_raw_sequence):
            contour_slice_points = slice_object.ContourData
            threeDdata_zslice = np.fromiter([contour_slice_points[i:i + 3] for i in range(0, len(contour_slice_points), 3)], dtype=np.dtype((np.float64, (3,))))
            threeDdata_zslice_list.append(threeDdata_zslice)

    total_structure_points = sum([np.shape(x)[0] for x in threeDdata_zslice_list])
    if isinstance(total_structure_points, int):
        pass
    elif isinstance(total_structure_points, float) & total_structure_points.is_integer():
        total_structure_points = int(total_structure_points)
    elif isinstance(total_structure_points, float) & total_structure_points.is_integer() == False:
        raise Exception("Seems the cumulative number of spatial components of contour points is not a whole number!")
    else:
        raise Exception("Something went wrong when calculating total number of points in structure!")

    # for non-biopsy only
    if structure_type != bx_ref:
    ## THIS WAS INDENTED UNDER THE IF STATEMENT BEFORE
        structure_centroids_array = np.empty([len(threeDdata_zslice_list),3])
        # find zslice-wise centroids
        for index, threeDdata_zslice in enumerate(threeDdata_zslice_list):
            structure_zslice_centroid = np.mean(threeDdata_zslice,axis=0)
            structure_centroids_array[index] = structure_zslice_centroid
        structure_global_centroid = centroid_finder.centeroidfinder_numpy_3D(structure_centroids_array)
        pydicom_item[structure_type][specific_structure_index]["Structure centroid pts"] = structure_centroids_array
        pydicom_item[structure_type][specific_structure_index]["Structure global centroid"] = structure_global_centroid
    ## THIS WAS INDENTED UNDER THE IF STATEMENT BEFORE

    pydicom_item[structure_type][specific_structure_index]["Raw contour pts zslice list"] = threeDdata_zslice_list


def pull_raw_structure_contours_for_patient(
    *,
    patient_uid,
    pydicom_item,
    rtstruct_dicom_path,
    structs_referenced_list_generalized,
    bx_ref,
    structures_progress=None,
    pulling_structures_task=None,
):
    """Pull raw contour points for every eligible structure in one patient."""
    for structs in structs_referenced_list_generalized:
        for specific_structure_index, specific_structure in enumerate(pydicom_item[structs]):
            structureID = specific_structure["ROI"]
            structure_reference_number = specific_structure["Ref #"]
            if structs == bx_ref:
                simulated_bool = specific_structure["Simulated bool"]
            else:
                simulated_bool = None
            pulling_structures_task_main_description = "[cyan]Pulling structures [{},{}]...".format(patient_uid,structureID)
            if structures_progress is not None and pulling_structures_task is not None:
                structures_progress.update(pulling_structures_task, description = pulling_structures_task_main_description)

            if simulated_bool == True:
                if structures_progress is not None and pulling_structures_task is not None:
                    structures_progress.update(pulling_structures_task, advance=1)
                continue # dont do anything if its a simulated biopsy!

            pull_raw_structure_contour_for_structure(
                pydicom_item,
                rtstruct_dicom_path,
                structs,
                specific_structure_index,
                bx_ref,
            )

            if structures_progress is not None and pulling_structures_task is not None:
                structures_progress.update(pulling_structures_task, advance=1)


def pull_raw_structure_contours_for_cohort(
    master_structure_reference_dict,
    master_structure_info_dict,
    rtstruct_dicom_paths_by_patient_uid,
    structs_referenced_list_generalized,
    bx_ref,
    all_ref_key,
    patients_progress,
    structures_progress,
    completed_progress,
):
    """Run the main-facing raw-contour pulling block for every patient."""
    patientUID_default = "Initializing"
    pulling_patients_task_main_description = "[red]Pulling patient structure data [{}]...".format(patientUID_default)
    pulling_patients_task_completed_main_description = "[green]Pulling patient structure data"
    pulling_patients_task = patients_progress.add_task(pulling_patients_task_main_description, total=master_structure_info_dict["Global"]["Num cases"])
    pulling_patients_task_completed = completed_progress.add_task(pulling_patients_task_completed_main_description, total=master_structure_info_dict["Global"]["Num cases"], visible = False)

    for patientUID,pydicom_item in master_structure_reference_dict.items():
        pulling_patients_task_main_description = "[red]Pulling patient structure data [{}]...".format(patientUID)
        patients_progress.update(pulling_patients_task, description = pulling_patients_task_main_description)

        structureID_default = "Initializing"
        num_general_structs_patient_specific = master_structure_info_dict["By patient"][patientUID][all_ref_key]["Total num structs"]
        pulling_structures_task_main_description = "[cyan]Pulling structures [{},{}]...".format(patientUID,structureID_default)
        pulling_structures_task = structures_progress.add_task(pulling_structures_task_main_description, total=num_general_structs_patient_specific)
        pull_raw_structure_contours_for_patient(
            patient_uid=patientUID,
            pydicom_item=pydicom_item,
            rtstruct_dicom_path=rtstruct_dicom_paths_by_patient_uid[patientUID],
            structs_referenced_list_generalized=structs_referenced_list_generalized,
            bx_ref=bx_ref,
            structures_progress=structures_progress,
            pulling_structures_task=pulling_structures_task,
        )
        structures_progress.remove_task(pulling_structures_task)
        patients_progress.update(pulling_patients_task, advance=1)
        completed_progress.update(pulling_patients_task_completed, advance=1)
    patients_progress.update(pulling_patients_task, visible=False)
    completed_progress.update(pulling_patients_task_completed,  visible=True)
