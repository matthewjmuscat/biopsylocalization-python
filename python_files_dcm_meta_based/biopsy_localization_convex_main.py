import pydicom # imported for reading dicom files
import pathlib # imported for navigating file system
import glob
import plotting_funcs
import matplotlib.pyplot as plt
import centroid_finder
import pca
import scipy
import scipy.spatial # for kdtree creation and NN search
import numpy as np
import biopsy_creator
import sys # imported for loading bar
from decimal import Decimal # for use in the loading bar
import loading_tools # imported for more sophisticated loading bar
import time # allows function to tell programme to wait, this was for testing the loading bar 
import ques_funcs
import timeit
from random import random
from shapely.geometry import Point, Polygon, MultiPoint # for point in polygon test
import open3d as o3d # for data visualization and meshing
import MC_simulator_convex
import uncertainty_processor
import alphashape
import uncertainty_file_writer
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
import csv
from prettytable import from_csv
import pandas
import anatomy_reconstructor_tools
import open3d.visualization.gui as gui
import os
import alphashape
import pymeshfix
import pyvista as pv
import point_containment_tools
import multiprocess
#import pathos, multiprocess
#from pathos.multiprocessing import ProcessingPool
import dill
import math
from datetime import date, datetime
import rich
from rich.progress import Progress, track
from rich.console import Group
from rich.panel import Panel
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
from rich.table import Table
from rich.layout import Layout
from rich.console import Console
import rich_preambles
from stopwatch import Stopwatch
import copy
import math_funcs as mf

def main():
    
    """
    A programme designed to receive dicom data consisting of prostate 
    ultrasound containing contouring and dosimetry information. The programme is then 
    designed to analyse the contour information to localize the biopsy 
    contours relative to the DIL and prostate contours. This version 
    of the programme does not rely on the structure of the data folder,
    all data may simply be dumped into the data folder, in whatever structure
    the analyzer would like. The programme relies solely on the dicom
    meta-data to identify patients, treatments and dicom type. At present, 
    it only requires that there exist a folder called Data, located two levels 
    above this file.
    """

    algo_global_start = time.time()
    stopwatch = Stopwatch(1)

    global loader

    # The following could be user input, for now they are defined here, and used throughout 
    # the programme for generality
    data_folder_name = 'Data'
    modality_list = ['RTSTRUCT','RTDOSE','RTPLAN']
    oaroi_contour_names = ['Prostate','Urethra','Rectum','random']
    biopsy_contour_names = ['Bx']
    dil_contour_names = ['DIL']
    uncertainty_folder_name = 'Uncertainty data'
    uncertainty_file_name = "uncertainties_prepared_unfilled"
    uncertainty_file_extension = ".csv"
    spinner_type = 'line'
    output_folder_name = 'Output data'
    biopsy_radius = 0.6
    num_sample_pts_per_bx_input = 10
    num_MC_simulations_input = 9
    biopsy_z_voxel_length = 1 #voxelize biopsy core every 1 mm along core

    cpu_count = os.cpu_count()
    with multiprocess.Pool(cpu_count) as parallel_pool:

        #st = time.time()

        progress_group_info_list = rich_preambles.get_progress_all(spinner_type)
        completed_progress, patients_progress, structures_progress, biopsies_progress, MC_trial_progress, indeterminate_progress_main, indeterminate_progress_sub, progress_group = progress_group_info_list

        rich_layout = rich_preambles.make_layout()

        important_info = rich_preambles.info_output()
        app_header = rich_preambles.Header()
        app_footer = rich_preambles.Footer(algo_global_start, stopwatch)

        layout_groups = (app_header,progress_group_info_list,important_info,app_footer)
        
               
        with Live(rich_layout, refresh_per_second = 8, screen = True) as live_display:
            rich_layout["header"].update(app_header)
            rich_layout["main-left"].update(progress_group)
            #rich_layout["box2"].update(Panel(make_syntax(), border_style="green"))
            rich_layout["main-right"].update(important_info)
            rich_layout["footer"].update(app_footer)

            # The figure dictionary to be plotted, this needs to be requested of the user later in the programme, after the  dicoms are read
            # First we access the data directory, it must be in a location 
            # two levels up from this file
            data_dir = pathlib.Path(__file__).parents[2].joinpath(data_folder_name)
            uncertainty_dir = data_dir.joinpath(uncertainty_folder_name)
            output_dir = data_dir.joinpath(output_folder_name)
            dicom_paths_list = list(pathlib.Path(data_dir).glob("**/*.dcm")) # list all file paths found in the data folder that have the .dcm extension
            important_info.add_text_line("Reading dicom data from: "+ str(data_dir), live_display)
            important_info.add_text_line("Reading uncertainty data from: "+ str(uncertainty_dir), live_display)

            num_dicoms = len(dicom_paths_list)
            important_info.add_text_line("Found "+str(num_dicoms)+" dicom files.", live_display)
            reading_dicoms_task_indeterminate = indeterminate_progress_main.add_task('[red]Reading dicom data from file...', total=None)
            reading_dicoms_task_indeterminate_completed = completed_progress.add_task('[green]Reading dicom data from file', total=num_dicoms, visible = False)
            dicom_elems_list = list(map(pydicom.dcmread,dicom_paths_list)) # read all the found dicom file paths using pydicom to create a list of FileDataset instances 
            indeterminate_progress_main.update(reading_dicoms_task_indeterminate, visible = False)
            completed_progress.update(reading_dicoms_task_indeterminate_completed, advance = num_dicoms,visible = True)
            live_display.refresh()

            # The 0x0008,0x0060 dcm tag specifies the 'Modality', here it is used to identify the type
            # of dicom file 
            #RTst_dcms = [x for x in dicom_elems_list if x[0x0008,0x0060].value == modality_list[0]]
            #RTdose_dcms = [x for x in dicom_elems_list if x[0x0008,0x0060].value == modality_list[1]]
            #RTplan_dcms = [x for x in dicom_elems_list if x[0x0008,0x0060].value == modality_list[2]]
            
            # the below is the first use of the UID_generator(pydicom_obj) function, which is used for the
            # creation of the PatientUID, that is generally created from or referenced from here 
            # throughout the programme, it is formed as "patientname (patientID)"
            RTst_dcms_dict = {UID_generator(x): x for x in dicom_elems_list if x[0x0008,0x0060].value == modality_list[0]}
            RTdose_dcms_dict = {UID_generator(x): x for x in dicom_elems_list if x[0x0008,0x0060].value == modality_list[1]}
            
            num_RTst_dcms_entries = len(RTst_dcms_dict)
            num_RTdose_dcms_entries = len(RTdose_dcms_dict)
            important_info.add_text_line("Found "+str(num_RTst_dcms_entries)+" unique patients with RT structure files.", live_display)
            important_info.add_text_line("Found "+str(num_RTdose_dcms_entries)+" unique patients with RT dose files.", live_display)

            if num_RTst_dcms_entries != num_RTdose_dcms_entries:
                live_display.stop()
                stopwatch.stop()
                exit_programme = ques_funcs.ask_ok('>Unequal number of structure files vs dose files, will encounter error later in the programme. Continue anyway?' )
                stopwatch.start()
                if exit_programme == True:
                    sys.exit('>Programme exited.')
                else:
                    pass

            if RTst_dcms_dict.keys() != RTdose_dcms_dict.keys():
                live_display.stop()
                stopwatch.stop()
                exit_programme = ques_funcs.ask_ok('>Same number of structure files vs dose files but there is an incongruency between them (file pairs do not match patients), will encounter error later in the programme. Continue anyway?' )
                stopwatch.start()
                if exit_programme == True:
                    sys.exit('>Programme exited.')
                else:
                    pass
            important_info.add_text_line("Each patient contains a structure and dose file.", live_display)    
            
            
            building_patient_dictionaries_task = indeterminate_progress_main.add_task('[red]Building patient dictionary...', total=None)
            building_patient_dictionaries_task_completed = completed_progress.add_task('[green]Building patient dictionary', total=num_RTst_dcms_entries, visible = False)
            master_structure_reference_dict, master_structure_info_dict, structs_referenced_list, dose_ref = structure_referencer(RTst_dcms_dict, RTdose_dcms_dict, oaroi_contour_names,dil_contour_names,biopsy_contour_names)
            indeterminate_progress_main.update(building_patient_dictionaries_task, visible = False)
            completed_progress.update(building_patient_dictionaries_task_completed, advance = num_RTst_dcms_entries,visible = True)
            important_info.add_text_line("Patient master dictionary built for "+str(master_structure_info_dict["Global"]["Num patients"])+" patients.", live_display)  
            live_display.refresh()


            # Now, we dont want to add the contour points to the structure list above,
            # because the contour data is already stored in a data tree, which will allow
            # for faster processing when accessed and iterated. update: I lied..... I ended up
            # doing exactly this. I will implement a data tree for the purpose of a search
            # algorithm when I do a nearest neighbour search
            

            # this dictionary determines which organs of which patient are to be plotted, in theory this could be user input
            # update: fig_dict ended up being deprecated, put data directly into master_dict instead
            # fig_dict = {UID: {specific_structure["ROI"]: True for structs in structs_referenced_list for specific_structure in pydicom_item[structs]} for UID, pydicom_item in master_structure_reference_dict.items()}
            
            # build a data dictionary to store the data we extract and build about the patient
            # update: data_dict never ended up being used, put data directly into master_dict
            # data_dict = {UID: None for UID, pydicom_item in master_structure_reference_dict.items()}

            # instantiate the variables used for the loading bar
            num_patients = master_structure_info_dict["Global"]["Num patients"]
            num_general_structs = master_structure_info_dict["Global"]["Num structures"]


            #important_info.add_text_line("important info will appear here1", live_display)
            #rich_layout["main-right"].update(important_info_Text)
           

            patientUID_default = "Initializing"
            processing_patients_dose_task_main_description = "[red]Building dose grids [{}]...".format(patientUID_default)
            processing_patients_dose_task_completed_main_description = "[green]Building dose grids"

            processing_patients_dose_task = patients_progress.add_task(processing_patients_dose_task_main_description, total=num_patients)
            processing_patients_dose_task_completed = completed_progress.add_task(processing_patients_dose_task_completed_main_description, total=num_patients, visible=False)

            for patientUID,pydicom_item in master_structure_reference_dict.items():
                processing_patients_dose_task_main_description = "[red]Building dose grids [{}]...".format(patientUID)
                patients_progress.update(processing_patients_dose_task, description=processing_patients_dose_task_main_description)

                dose_ref_dict = master_structure_reference_dict[patientUID][dose_ref]
                dose_pixel_slice_list = dose_ref_dict["Dose pixel arr"]
                grid_frame_offset_vec_list = dose_ref_dict["Grid frame offset vector"]
                num_dose_grid_slices = len(grid_frame_offset_vec_list)

                image_orientation_patient = dose_ref_dict["Image orientation patient"]
                Xx, Xy, Xz = image_orientation_patient[0:3]
                Yx, Yy, Yz = image_orientation_patient[3:]

                image_position_patient = dose_ref_dict["Image position patient"]
                Sx, Sy, Sz = image_position_patient

                dose_grid_scaling = dose_ref_dict["Dose grid scaling"]

                pixel_spacing = dose_ref_dict["Pixel spacing"]
                row_spacing_del_j = pixel_spacing[0]
                column_spacing_del_i = pixel_spacing[1]

                dose_first_slice_2darr = dose_pixel_slice_list[0]

                conversion_matrix_pixel_2_physical = np.array([[Xx*column_spacing_del_i, Yx*row_spacing_del_j,0,Sx],
                [Xy*column_spacing_del_i, Yy*row_spacing_del_j,0,Sy],
                [Xz*column_spacing_del_i, Yz*row_spacing_del_j,0,Sz],
                [0,0,0,1]])

                pixel_ds_grid_num_rows = dose_first_slice_2darr.shape[0]
                pixel_ds_grid_num_cols = dose_first_slice_2darr.shape[1]

                phys_space_pixel_mapping_arr_ji_XY = np.empty((pixel_ds_grid_num_rows*pixel_ds_grid_num_cols,7), dtype=float)
                # only need to do pixel to location conversion for one slice
                for row_index_j in range(pixel_ds_grid_num_rows):
                    for col_index_i in range(pixel_ds_grid_num_cols):
                        pixel_vec = np.array([[col_index_i], [row_index_j], [0], [1]])
                        phys_vec = np.matmul(conversion_matrix_pixel_2_physical,pixel_vec)
                        current_index = row_index_j*pixel_ds_grid_num_cols + col_index_i
                        # the entries below are as follows: slice, row index, col index, xval, yval, zval, dose, the first and last entry are populated later
                        # and so are set to 0 for now
                        current_pixel_map_entry = [0, row_index_j, col_index_i, phys_vec[0][0], phys_vec[1][0], phys_vec[2][0], 0]
                        phys_space_pixel_mapping_arr_ji_XY[current_index] = current_pixel_map_entry
                        

                phys_space_dose_map_2d_arr_slice_wise_list = [np.empty((pixel_ds_grid_num_rows*pixel_ds_grid_num_cols,4), dtype=float)]*num_dose_grid_slices
                for grid_frame_offset_cur_slice_index, grid_frame_offset_cur_slice in enumerate(grid_frame_offset_vec_list):
                    phys_space_dose_map_current_slice_2d_arr = phys_space_pixel_mapping_arr_ji_XY.copy()
                    phys_space_dose_map_current_slice_2d_arr[:,0] = grid_frame_offset_cur_slice_index
                    phys_space_dose_map_current_slice_2d_arr[:,5] = phys_space_dose_map_current_slice_2d_arr[:,5] + grid_frame_offset_cur_slice
                    current_slice_properly_scaled_dose_2d_arr = dose_pixel_slice_list[grid_frame_offset_cur_slice_index]*dose_grid_scaling
                    current_slice_properly_scaled_dose_2d_arr_flattened = current_slice_properly_scaled_dose_2d_arr.flatten(order='C')
                    phys_space_dose_map_current_slice_2d_arr[:,6] = current_slice_properly_scaled_dose_2d_arr_flattened
                    phys_space_dose_map_2d_arr_slice_wise_list[grid_frame_offset_cur_slice_index] = phys_space_dose_map_current_slice_2d_arr
                    
                phys_space_dose_map_3d_arr = np.asarray(phys_space_dose_map_2d_arr_slice_wise_list)
                #phys_space_dose_map_3d_arr_flattened = np.reshape(phys_space_dose_map_3d_arr, (-1,7) , order = 'C')
                dose_ref_dict["Dose phys space and pixel 3d arr"] = phys_space_dose_map_3d_arr

                dose_point_cloud = plotting_funcs.create_dose_point_cloud(phys_space_dose_map_3d_arr, paint_dose_color = True)
                dose_ref_dict["Dose grid point cloud"] = dose_point_cloud

                patients_progress.stop_task(processing_patients_dose_task)
                completed_progress.stop_task(processing_patients_dose_task_completed)
                stopwatch.stop()
                plotting_funcs.plot_geometries(dose_point_cloud)
                stopwatch.start()
                patients_progress.start_task(processing_patients_dose_task)
                completed_progress.start_task(processing_patients_dose_task_completed)

                # user defined quantity!
                lower_bound_dose_percent = 5
                thresholded_dose_point_cloud = plotting_funcs.create_thresholded_dose_point_cloud(phys_space_dose_map_3d_arr, paint_dose_color = True, lower_bound_percent = lower_bound_dose_percent)
                dose_ref_dict["Dose grid point cloud thresholded"] = thresholded_dose_point_cloud

                patients_progress.stop_task(processing_patients_dose_task)
                completed_progress.stop_task(processing_patients_dose_task_completed)
                stopwatch.stop()
                plotting_funcs.plot_geometries(thresholded_dose_point_cloud)
                stopwatch.start()
                patients_progress.start_task(processing_patients_dose_task)
                completed_progress.start_task(processing_patients_dose_task_completed)

                patients_progress.update(processing_patients_dose_task, advance=1)
                completed_progress.update(processing_patients_dose_task_completed, advance=1)
            
            patients_progress.update(processing_patients_dose_task, visible=False)
            completed_progress.update(processing_patients_dose_task_completed, visible=True)
                    
        

            patientUID_default = "Initializing"
            processing_patients_task_main_description = "[red]Processing patient structure data [{}]...".format(patientUID_default)
            processing_patients_task_completed_main_description = "[green]Processing patient structure data"
            processing_patients_task = patients_progress.add_task(processing_patients_task_main_description, total=num_patients)
            processing_patients_task_completed = completed_progress.add_task(processing_patients_task_completed_main_description, total=num_patients, visible = False)


            for patientUID,pydicom_item in master_structure_reference_dict.items():
                processing_patients_task_main_description = "[red]Processing patient structure data [{}]...".format(patientUID)
                patients_progress.update(processing_patients_task, description = processing_patients_task_main_description)
                
                structureID_default = "Initializing"
                num_general_structs_patient_specific = master_structure_info_dict["By patient"][patientUID]["All ref"]["Total num structs"]
                processing_structures_task_main_description = "[cyan]Processing structures [{},{}]...".format(patientUID,structureID_default)
                processing_structures_task = structures_progress.add_task(processing_structures_task_main_description, total=num_general_structs_patient_specific)
                for structs in structs_referenced_list:
                    for specific_structure_index, specific_structure in enumerate(pydicom_item[structs]):
                        structureID = specific_structure["ROI"]
                        processing_structures_task_main_description = "[cyan]Processing structures [{},{}]...".format(patientUID,structureID)
                        structures_progress.update(processing_structures_task, description = processing_structures_task_main_description)

                        # The below print lines were just for my own understanding of how to access the data structure
                        #print(specific_structure["ROI"])
                        #print(specific_structure["Ref #"])
                        #print(RTst_dcms[dcm_index].ROIContourSequence[int(specific_structure["Ref #"])].ContourSequence[0].ContourData)
                        #print(RTst_dcms[dcm_index].ROIContourSequence[int(specific_structure["Ref #"])].ContourSequence[1].ContourData)

                        # can uncomment surrounding lines to time this particular process
                        #st = time.time()
                        threeDdata_zslice_list = []
                        structure_contour_points_raw_sequence = RTst_dcms_dict[patientUID].ROIContourSequence[int(specific_structure["Ref #"])].ContourSequence[0:]
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

                        threeDdata_array = np.empty([total_structure_points,3])
                        # for biopsy only
                        if structs == structs_referenced_list[0]:
                            structure_centroids_array = np.empty([len(threeDdata_zslice_list),3])
                        lower_bound_index = 0  
                        # build raw threeDdata
                        for index, threeDdata_zslice in enumerate(threeDdata_zslice_list):
                            current_zslice_num_points = np.size(threeDdata_zslice,0)
                            threeDdata_array[lower_bound_index:lower_bound_index + current_zslice_num_points] = threeDdata_zslice
                            lower_bound_index = lower_bound_index + current_zslice_num_points 
                            
                            if structs == structs_referenced_list[0]:
                                structure_zslice_centroid = np.mean(threeDdata_zslice,axis=0)
                                structure_centroids_array[index] = structure_zslice_centroid


                        # conduct INTER-slice interpolation
                        interp_dist_z_slice = 0.5
                        interslice_interpolation_information, threeDdata_equal_pt_zslice_list = anatomy_reconstructor_tools.inter_zslice_interpolator(threeDdata_zslice_list, interp_dist_z_slice)
                        
                        # conduct INTRA-slice interpolation
                        # do you want to interpolate the zslice interpolated data or the raw data? comment out the appropriate line below..
                        threeDdata_to_intra_zslice_interpolate_zslice_list = interslice_interpolation_information.interpolated_pts_list
                        # threeDdata_to_intra_zslice_interpolate_zslice_list = threeDdata_zslice_list

                        num_z_slices_data_to_intra_slice_interpolate = len(threeDdata_to_intra_zslice_interpolate_zslice_list)
                        interp_dist = 2 # this is/should be a user defined length scale! It is used in the interpolation_information_obj class
                        interpolation_information = interpolation_information_obj(num_z_slices_data_to_intra_slice_interpolate)
                        
                        interpolation_information.parallel_analyze(parallel_pool, threeDdata_to_intra_zslice_interpolate_zslice_list,interp_dist)

                        #for index, threeDdata_zslice in enumerate(threeDdata_to_intra_zslice_interpolate_zslice_list):
                        #    interpolation_information.analyze_structure_slice(threeDdata_zslice,interp_dist)

                        # fill in the end caps
                        interp_dist_caps = 2
                        first_zslice = threeDdata_to_intra_zslice_interpolate_zslice_list[0]
                        last_zslice = threeDdata_to_intra_zslice_interpolate_zslice_list[-1]
                        interpolation_information.create_fill(first_zslice, interp_dist_caps)
                        interpolation_information.create_fill(last_zslice, interp_dist_caps)

                        #et = time.time()
                        #elapsed_time = et - st
                        #print('\n Execution time:', elapsed_time, 'seconds')

                        # generate point cloud of raw threeDdata
                        threeDdata_pcd_color = np.random.uniform(0, 0.7, size=3)
                        threeDdata_point_cloud = point_containment_tools.create_point_cloud(threeDdata_array, threeDdata_pcd_color)
                        
                        

                        # generate delaunay triangulations 
                        deulaunay_objs_zslice_wise_list = point_containment_tools.adjacent_slice_delaunay_parallel(parallel_pool, threeDdata_zslice_list)

                        zslice1 = threeDdata_array[0,2]
                        zslice2 = threeDdata_array[-1,2]
                        delaunay_global_convex_structure_obj = point_containment_tools.delaunay_obj(threeDdata_array, threeDdata_pcd_color, zslice1, zslice2)
                        delaunay_global_convex_structure_obj.generate_lineset()

                        
                        # Below is a sample simulation to test the containment algorithm:
                        """
                        print(specific_structure["ROI"])
                        # zslice wise convex structure box simulation
                        num_simulations = 500
                        
                        test_points_results_zslice_delaunay, test_pts_point_cloud_zslice_delaunay = MC_simulator_convex.box_simulator_delaunay_zslice_wise_parallel(parallel_pool, num_simulations, deulaunay_objs_zslice_wise_list, threeDdata_point_cloud)
                        test_points_results_fully_concave, test_pts_point_cloud_concave_zslice_updated = point_containment_tools.plane_point_in_polygon_concave(test_points_results_zslice_delaunay,interslice_interpolation_information, test_pts_point_cloud_zslice_delaunay)
                        # plot zslice wise delaunay in open3d ?
                        #plotting_funcs.plot_tri_immediately_efficient_multilineset(threeDdata_array, test_pts_point_cloud_zslice_delaunay, deulaunay_objs_zslice_wise_list, label = specific_structure["ROI"])
                        #plotting_funcs.plot_tri_immediately_efficient_multilineset(threeDdata_array, test_pts_point_cloud_concave_zslice_updated, deulaunay_objs_zslice_wise_list, label = specific_structure["ROI"])
                        
                        line_sets_of_threeDdata_equal_pt_zslices_list = anatomy_reconstructor_tools.create_lineset_all_zslices(interslice_interpolation_information.interpolated_pts_list)
                        inter_zslice_interpolated_point_cloud = point_containment_tools.create_point_cloud(interslice_interpolation_information.interpolated_pts_np_arr)

                        
                        #plotting_funcs.plot_geometries(inter_zslice_interpolated_point_cloud, test_pts_point_cloud_concave_zslice_updated, *line_sets_of_threeDdata_equal_pt_zslices_list, label='Unknown')
                        # global convex structure box simulation
                        num_simulations = 50
                        test_points_results_global_delaunay, test_pts_point_cloud_global_delaunay = MC_simulator_convex.box_simulator_delaunay_global_convex_structure_parallel(parallel_pool, num_simulations, delaunay_global_convex_structure_obj, threeDdata_point_cloud)
                        # plot delaunay global convex structure in open3d ?
                        #plotting_funcs.plot_tri_immediately_efficient(threeDdata_array, delaunay_global_convex_structure_obj.delaunay_line_set, test_pts_point_cloud_global_delaunay, label = specific_structure["ROI"])

                        """

                        threeDdata_array_fully_interpolated = interpolation_information.interpolated_pts_np_arr
                        threeDdata_array_fully_interpolated_with_end_caps = interpolation_information.interpolated_pts_with_end_caps_np_arr
                        threeDdata_array_interslice_interpolation = np.vstack(interslice_interpolation_information.interpolated_pts_list)

                        
                        # plot raw points ?
                        #plotting_funcs.plot_point_clouds(threeDdata_array, label='Unknown')

                        # WARNING : The function (plotting_funcs.point_cloud_with_order_labels) has an error, when called the second time after .run it outputs a GLFW not initialized error!
                        # plot points with order labels of interpolated intraslice ?
                        #plotting_funcs.point_cloud_with_order_labels(threeDdata_array_fully_interpolated)

                        # plot points with order labels of raw data ?
                        #if test_ind > 1:
                        #   plotting_funcs.point_cloud_with_order_labels(threeDdata_array)
                        #test_ind = test_ind + 1

                        
                        
                        # plot fully interpolated points of z data ?
                        #plotting_funcs.point_cloud_with_order_labels(threeDdata_array_interslice_interpolation)
                        #plotting_funcs.plot_point_clouds(threeDdata_array_interslice_interpolation,threeDdata_array,threeDdata_array_fully_interpolated, label='Unknown')
                        #plotting_funcs.plot_point_clouds(threeDdata_array_interslice_interpolation, label='Unknown')
                        #plotting_funcs.plot_point_clouds(threeDdata_array_fully_interpolated, label='Unknown')


                        # plot two point clouds side by side ? 
                        #plotting_funcs.plot_two_point_clouds_side_by_side(threeDdata_array, threeDdata_array_fully_interpolated)
                        #plotting_funcs.plot_two_point_clouds_side_by_side(threeDdata_array, threeDdata_array_fully_interpolated_with_end_caps)
                        


                        # for biopsies only
                        if structs == structs_referenced_list[0]:
                            centroid_line = pca.linear_fitter(structure_centroids_array.T)
                            centroid_line_length = np.linalg.norm(centroid_line[0,:] - centroid_line[1,:])
                            slice_reconstruction_max_distance = 0.1
                            num_centroid_samples_of_centroid_line = int(math.ceil(centroid_line_length/slice_reconstruction_max_distance))
                            centroid_line_sample = np.empty((num_centroid_samples_of_centroid_line,3), dtype=float)
                            centroid_line_sample[0,:] = centroid_line[0,:]
                            travel_vec = np.array([centroid_line[1]-centroid_line[0]])*1/num_centroid_samples_of_centroid_line
                            for i in range(1,num_centroid_samples_of_centroid_line):
                                init_point = centroid_line_sample[-1]
                                new_point = init_point + travel_vec
                                centroid_line_sample[i] = new_point
    
                            # conduct a nearest neighbour search of biopsy centroids

                            #treescipy = scipy.spatial.KDTree(threeDdata_array)
                            #nn = treescipy.query(centroid_line_sample[0])
                            #nearest_neighbour = treescipy.data[nn[1]]

                            list_travel_vec = np.squeeze(travel_vec).tolist()
                            list_centroid_line_first_point = np.squeeze(centroid_line_sample[0]).tolist()
                            biopsy_reconstructed_cyl_z_length_from_contour_data = centroid_line_length
                            drawn_biopsy_array_transpose = biopsy_creator.biopsy_points_creater_by_transport(list_travel_vec,list_centroid_line_first_point,num_centroid_samples_of_centroid_line,np.linalg.norm(travel_vec), biopsy_radius, False)
                            drawn_biopsy_array = drawn_biopsy_array_transpose.T
                            reconstructed_bx_pcd_color = np.random.uniform(0, 0.7, size=3)
                            reconstructed_biopsy_point_cloud = point_containment_tools.create_point_cloud(drawn_biopsy_array, reconstructed_bx_pcd_color)
                            reconstructed_bx_delaunay_global_convex_structure_obj = point_containment_tools.delaunay_obj(drawn_biopsy_array, reconstructed_bx_pcd_color)
                            reconstructed_bx_delaunay_global_convex_structure_obj.generate_lineset()
                            #plot reconstructions?
                            #plotting_funcs.plot_geometries(reconstructed_biopsy_point_cloud, threeDdata_point_cloud)
                            #plotting_funcs.plot_tri_immediately_efficient(drawn_biopsy_array, reconstructed_bx_delaunay_global_convex_structure_obj.delaunay_line_set, label = specific_structure["ROI"])


                        # plot only the biopsies
                        if structs == structs_referenced_list[0]:
                            specific_structure["Plot attributes"].plot_bool = True



                        # store all calculated quantities
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Raw contour pts"] = threeDdata_array
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Equal num zslice contour pts"] = threeDdata_equal_pt_zslice_list
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Inter-slice interpolation information"] = interslice_interpolation_information                        
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Intra-slice interpolation information"] = interpolation_information
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Delaunay triangulation zslice-wise list"] = deulaunay_objs_zslice_wise_list
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Delaunay triangulation global structure"] = delaunay_global_convex_structure_obj
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Point cloud raw"] = threeDdata_point_cloud
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Structure centroid pts"] = structure_centroids_array
                        if structs == structs_referenced_list[0]:
                            master_structure_reference_dict[patientUID][structs][specific_structure_index]["Reconstructed biopsy cylinder length (from contour data)"] = biopsy_reconstructed_cyl_z_length_from_contour_data
                            master_structure_reference_dict[patientUID][structs][specific_structure_index]["Best fit line of centroid pts"] = centroid_line
                            master_structure_reference_dict[patientUID][structs][specific_structure_index]["Centroid line sample pts"] = centroid_line_sample
                            master_structure_reference_dict[patientUID][structs][specific_structure_index]["Reconstructed structure pts arr"] = drawn_biopsy_array
                            master_structure_reference_dict[patientUID][structs][specific_structure_index]["Reconstructed structure point cloud"] = reconstructed_biopsy_point_cloud
                            master_structure_reference_dict[patientUID][structs][specific_structure_index]["Reconstructed structure delaunay global"] = reconstructed_bx_delaunay_global_convex_structure_obj


                        structures_progress.update(processing_structures_task, advance=1)
                structures_progress.remove_task(processing_structures_task)
                patients_progress.update(processing_patients_task, advance=1)
                completed_progress.update(processing_patients_task_completed, advance=1)
            patients_progress.update(processing_patients_task, visible=False)
            completed_progress.update(processing_patients_task_completed,  visible=True)                

            live_display.refresh()

            print('test')

            for patientUID,pydicom_item in master_structure_reference_dict.items():
                dose_ref_dict = pydicom_item[dose_ref]
                dose_grid_pcd = dose_ref_dict["Dose grid point cloud thresholded"]
                pcd_list = [dose_grid_pcd]
                for structs in structs_referenced_list:
                    for specific_structure_index, specific_structure in enumerate(pydicom_item[structs]):
                        if structs == structs_referenced_list[0]: 
                            structure_pcd = specific_structure["Reconstructed structure point cloud"]
                        else: 
                            structure_pcd = specific_structure["Point cloud raw"]
                        pcd_list.append(structure_pcd)
                        
                plotting_funcs.plot_geometries(*pcd_list)

                
            #et = time.time()
            #elapsed_time = et - st
            #print('\n Execution time:', elapsed_time, 'seconds')

            """

            
            with loading_tools.Loader(num_patients,"Generating KD trees and conducting nearest neighbour searches...") as loader:
                for patientUID,pydicom_item in master_structure_reference_dict.items():
                    Bx_structs = structs_referenced_list[0]
                    for specific_BX_structure_index, specific_BX_structure in enumerate(pydicom_item[Bx_structs]): 
                        BX_centroid_line_sample = specific_BX_structure["Centroid line sample pts"]
                        for non_BX_structs in structs_referenced_list[1:]:
                            for specific_non_BX_structs_index, specific_non_BX_structs in enumerate(pydicom_item[non_BX_structs]):
                                
                                # create a KDtree for all non BX structures
                                non_BX_struct_threeDdata_array = specific_non_BX_structs["Raw contour pts"]
                                non_BX_struct_KDtree = scipy.spatial.KDTree(non_BX_struct_threeDdata_array)
                                master_structure_reference_dict[patientUID][non_BX_structs][specific_non_BX_structs_index]["KDtree"] = non_BX_struct_KDtree
                                
                                # conduct NN search
                                nearest_neighbours = non_BX_struct_KDtree.query(BX_centroid_line_sample)
                                
                                master_structure_reference_dict[patientUID][Bx_structs][specific_BX_structure_index]["Nearest neighbours objects"].append(nearest_neighbour_parent(specific_BX_structure["ROI"],specific_non_BX_structs["ROI"],non_BX_structs,non_BX_struct_threeDdata_array,BX_centroid_line_sample,nearest_neighbours))
                                
                    loader.iterator = loader.iterator + 1

            
            global_data_list = []
            disp_figs = ques_funcs.ask_ok('Do you want to open any figures now?')
            
            if disp_figs == True:
                for patientUID,pydicom_item in master_structure_reference_dict.items():
                    global_data_list_per_patient = []
                    global_patient_info = {}
                    for structs in structs_referenced_list:
                        for specific_structure in pydicom_item[structs]:
                            plot_fig_bool = specific_structure["Plot attributes"].plot_bool
                            if plot_fig_bool == True:
                                threeDdata_array = specific_structure["Raw contour pts"]
                                threeDdata_array_transpose = threeDdata_array.T
                                threeDdata_list = threeDdata_array_transpose.tolist()
                                threeDdata_list_and_color = threeDdata_list
                                threeDdata_list_and_color.append((random(), random(), random()))
                                threeDdata_list_and_color.append('o')
                                global_data_list_per_patient.append(threeDdata_list_and_color)

                                if structs == structs_referenced_list[0]:
                                    centroid_line_sample = specific_structure["Centroid line sample pts"] 
                                    structure_centroids_array = specific_structure["Structure centroid pts"]
                                    centroid_line = specific_structure["Best fit line of centroid pts"] 
                                    drawn_biopsy_array = specific_structure["Reconstructed structure pts"] 

                                    centroid_line_sample_transpose = centroid_line_sample.T
                                    centroid_line_sample_list = centroid_line_sample_transpose.tolist()
                                    centroid_line_sample_list_and_color = centroid_line_sample_list
                                    centroid_line_sample_list_and_color.append('y')
                                    centroid_line_sample_list_and_color.append('x')
                                    
                                    drawn_biopsy_array_transpose = drawn_biopsy_array.T
                                    drawn_biopsy_list = drawn_biopsy_array_transpose.tolist()
                                    drawn_biopsy_list_and_color = drawn_biopsy_list
                                    drawn_biopsy_list_and_color.append('m')
                                    drawn_biopsy_list_and_color.append('+')
                                    
                                    structure_centroids_array_transpose = structure_centroids_array.T
                                    structure_centroids_list = structure_centroids_array_transpose.tolist()
                                    structure_centroids_list_and_color = structure_centroids_list
                                    structure_centroids_list_and_color.append('b')
                                    structure_centroids_list_and_color.append('o')

                                    global_data_list_per_patient.append(centroid_line_sample_list_and_color)
                                    global_data_list_per_patient.append(drawn_biopsy_list_and_color)
                                    global_data_list_per_patient.append(structure_centroids_list_and_color)


                                info = specific_structure
                                info["Patient Name"] = pydicom_item["Patient Name"]
                                info["Patient ID"] = pydicom_item["Patient ID"]
                                #specific_structure_fig = plotting_funcs.arb_threeD_scatter_plotter(*global_data_list_per_patient,**info)
                                specific_structure_fig = plotting_funcs.arb_threeD_scatter_plotter_list(*global_data_list_per_patient, **info)
                                specific_structure_fig = plotting_funcs.add_line(specific_structure_fig,centroid_line)
                            
                                #specific_structure_fig.show()


                                
                    global_data_list.append(global_data_list_per_patient)

                close_figs = ques_funcs.ask_to_continue("Press carriage return when you wish to close all figures")
                plt.close('all')
            else:
                pass
            

            if disp_figs == True:
                disp_figs_global = ques_funcs.ask_ok('Do you want to open the global data figure too?')
                if disp_figs_global == True:
                    figure_global = plotting_funcs.arb_threeD_scatter_plotter_global(global_data_list[0])
                    figure_global.show()
                    close_figs = ques_funcs.ask_to_continue("Press carriage return when you wish to close the figure")
                    plt.close('all')

            
            if disp_figs == True:
                disp_figs_DIL_NN = ques_funcs.ask_ok('Do you want to show the NN plots?')
                for patientUID,pydicom_item in master_structure_reference_dict.items():
                    info = {}
                    info["Patient Name"] = pydicom_item["Patient Name"]
                    info["Patient ID"] = pydicom_item["Patient ID"]
                    figure_global_per_patient = plotting_funcs.plot_general_per_patient(pydicom_item, structs_referenced_list, OAR_plot_attr=[], DIL_plot_attr=['raw'], **info)
                    figure_global_per_patient.show()
                    close_figs = ques_funcs.ask_to_continue("Press carriage return when you wish to close the figure")
                    plt.close('all')
            

            """


            ## uniformly sample points from biopsies
            #st = time.time()
            args_list = []
            num_biopsies = master_structure_info_dict["Global"]["Num biopsies"]
            num_sample_pts_per_bx = num_sample_pts_per_bx_input
            master_structure_info_dict["Global"]["MC info"]["Num sample pts per BX core"] = num_sample_pts_per_bx

            patientUID_default = "Initializing"
            processing_patient_parallel_computing_main_description = "Preparing patient for parallel processing [{}]...".format(patientUID_default)
            processing_patients_task = patients_progress.add_task("[red]"+processing_patient_parallel_computing_main_description, total = num_patients)
            processing_patient_parallel_computing_main_description_completed = "Preparing patient for parallel processing"
            processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_parallel_computing_main_description_completed, total=num_patients, visible=False)

            for patientUID,pydicom_item in master_structure_reference_dict.items():
                bx_structs = structs_referenced_list[0]

                processing_patient_parallel_computing_main_description = "Preparing patient for parallel processing [{}]...".format(patientUID)
                patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_parallel_computing_main_description)
                
                num_biopsies_per_patient = master_structure_info_dict["By patient"][patientUID][bx_structs]["Num structs"]
                biopsyID_default = "Initializing"
                processing_biopsies_main_description = "[cyan]Preparing biopsy data for parallel processing [{},{}]...".format(patientUID,biopsyID_default)
                processing_biopsies_task = biopsies_progress.add_task(processing_biopsies_main_description, total=num_biopsies_per_patient)

                for specific_structure_index, specific_structure in enumerate(pydicom_item[bx_structs]):
                    specific_bx_structure_roi = specific_structure["ROI"]
                    processing_biopsies_main_description = "[cyan]Preparing biopsy data for parallel processing [{},{}]...".format(patientUID,specific_bx_structure_roi)
                    biopsies_progress.update(processing_biopsies_task, description = processing_biopsies_main_description)
                    reconstructed_biopsy_point_cloud = master_structure_reference_dict[patientUID][bx_structs][specific_structure_index]["Reconstructed structure point cloud"]
                    reconstructed_biopsy_arr = master_structure_reference_dict[patientUID][bx_structs][specific_structure_index]["Reconstructed structure pts arr"]
                    reconstructed_delaunay_global_convex_structure_obj = master_structure_reference_dict[patientUID][bx_structs][specific_structure_index]["Reconstructed structure delaunay global"]
                    args_list.append((num_sample_pts_per_bx, reconstructed_delaunay_global_convex_structure_obj.delaunay_triangulation, reconstructed_biopsy_arr, patientUID, bx_structs, specific_structure_index))
                    biopsies_progress.update(processing_biopsies_task, advance=1)
                
                biopsies_progress.update(processing_biopsies_task, visible = False)
                patients_progress.update(processing_patients_task, advance = 1)
                completed_progress.update(processing_patients_completed_task, advance = 1)

            
            patients_progress.update(processing_patients_task, visible = False)
            completed_progress.update(processing_patients_completed_task, visible = True)


            #et = time.time()
            #elapsed_time = et - st
            #print('\n Execution time (NON PARALLEL):', elapsed_time, 'seconds')
            
            
            #st = time.time()
        
        
            sampling_points_task_indeterminate = indeterminate_progress_main.add_task("[red]Sampling points from all patient biopsies (parallel)...", total=None)
            sampling_points_task_indeterminate_completed = completed_progress.add_task("[green]Sampling points from all patient biopsies (parallel)", visible = False, total = num_patients)
            parallel_results_sampled_bx_points_from_global_delaunay_arr_and_bounding_box_arr = parallel_pool.starmap(MC_simulator_convex.point_sampler_from_global_delaunay_convex_structure_parallel, args_list)

            indeterminate_progress_main.update(sampling_points_task_indeterminate, visible = False, refresh = True)
            completed_progress.update(sampling_points_task_indeterminate_completed, advance = num_patients, visible = True, refresh = True)
            live_display.refresh()

            


            #et = time.time()
            #elapsed_time = et - st
            #print('\n Execution time (PARALLEL):', elapsed_time, 'seconds')

            global_num_biopsies = master_structure_info_dict["Global"]["Num biopsies"]
            patientUID_default = "Initializing"
            bx_ID_default = "Initializing"
            parsing_sampled_biopsy_data_task_main_description = "Parsing sampled biopsy information [{},{}]".format(patientUID_default,bx_ID_default)
            parsing_sampled_biopsy_data_task_main_description_completed = "Parsing sampled biopsy information"
            parsing_sampled_biopsy_data_task = biopsies_progress.add_task("[red]"+parsing_sampled_biopsy_data_task_main_description, total = global_num_biopsies)
            parsing_sampled_biopsy_data_task_completed = completed_progress.add_task("[green]"+parsing_sampled_biopsy_data_task_main_description_completed, total = global_num_biopsies, visible = False)

            for sampled_bx_pts_arr, axis_aligned_bounding_box_arr, structure_info_dict in parallel_results_sampled_bx_points_from_global_delaunay_arr_and_bounding_box_arr:
                temp_patient_UID = structure_info_dict["Patient UID"]
                temp_structure_type = structure_info_dict["Structure type"]
                temp_specific_structure_index = structure_info_dict["Specific structure index"]
                temp_structure_ID = master_structure_reference_dict[temp_patient_UID][temp_structure_type][temp_specific_structure_index]["ROI"]
                
                parsing_sampled_biopsy_data_task_main_description = "Parsing sampled biopsy information [{},{}]".format(temp_patient_UID,temp_structure_ID)
                biopsies_progress.update(parsing_sampled_biopsy_data_task, description="[red]"+parsing_sampled_biopsy_data_task_main_description, refresh=True)
                live_display.refresh()
                
                
                sampled_bx_points_from_global_delaunay_point_cloud_color = np.random.uniform(0, 0.7, size=3)
                sampled_bx_points_from_global_delaunay_point_cloud = point_containment_tools.create_point_cloud(sampled_bx_pts_arr, sampled_bx_points_from_global_delaunay_point_cloud_color)
                axis_aligned_bounding_box = o3d.geometry.AxisAlignedBoundingBox()
                axis_aligned_bounding_box_o3d3dvector_points = o3d.utility.Vector3dVector(axis_aligned_bounding_box_arr)
                axis_aligned_bounding_box = axis_aligned_bounding_box.create_from_points(axis_aligned_bounding_box_o3d3dvector_points)
                #axis_aligned_bounding_box_points_arr = np.asarray(axis_aligned_bounding_box.get_box_points())
                bounding_box_color_arr = np.array([0,0,0], dtype=float)
                axis_aligned_bounding_box.color = bounding_box_color_arr

                # update master dict 
                
                master_structure_reference_dict[temp_patient_UID][temp_structure_type][temp_specific_structure_index]["Random uniformly sampled volume pts arr"] = sampled_bx_pts_arr
                master_structure_reference_dict[temp_patient_UID][temp_structure_type][temp_specific_structure_index]["Random uniformly sampled volume pts pcd"] = sampled_bx_points_from_global_delaunay_point_cloud
                master_structure_reference_dict[temp_patient_UID][temp_structure_type][temp_specific_structure_index]["Bounding box for random uniformly sampled volume pts"] = axis_aligned_bounding_box
                reconstructed_bx_pcd = master_structure_reference_dict[temp_patient_UID][temp_structure_type][temp_specific_structure_index]["Reconstructed structure point cloud"] 

                biopsies_progress.stop_task(parsing_sampled_biopsy_data_task)
                completed_progress.stop_task(parsing_sampled_biopsy_data_task_completed)
                stopwatch.stop()
                plotting_funcs.plot_geometries(sampled_bx_points_from_global_delaunay_point_cloud, reconstructed_bx_pcd, axis_aligned_bounding_box)
                stopwatch.start()
                biopsies_progress.start_task(parsing_sampled_biopsy_data_task)
                completed_progress.start_task(parsing_sampled_biopsy_data_task_completed)

                biopsies_progress.update(parsing_sampled_biopsy_data_task, advance = 1, refresh = True)
                completed_progress.update(parsing_sampled_biopsy_data_task_completed, advance = 1, refresh = True)
                live_display.refresh()
                
            biopsies_progress.update(parsing_sampled_biopsy_data_task, visible = False, refresh = True)
            completed_progress.update(parsing_sampled_biopsy_data_task_completed, visible = True, refresh = True)
            live_display.refresh()


            patientUID_default = "Initializing"
            processing_patient_rotating_bx_main_description = "Creating biopsy oriented coordinate system [{}]...".format(patientUID_default)
            processing_patients_task = patients_progress.add_task("[red]"+processing_patient_rotating_bx_main_description, total = num_patients)
            processing_patient_rotating_bx_main_description_completed = "Creating biopsy oriented coordinate system"
            processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_rotating_bx_main_description_completed, total=num_patients, visible=False)

            # rotating pointclouds to create bx oriented frame of reference
            for patientUID,pydicom_item in master_structure_reference_dict.items():
                bx_structs = structs_referenced_list[0]

                processing_patient_rotating_bx_main_description = "Creating biopsy oriented coordinate system [{}]...".format(patientUID)
                patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_rotating_bx_main_description)
                
                num_biopsies_per_patient = master_structure_info_dict["By patient"][patientUID][bx_structs]["Num structs"]
                biopsyID_default = "Initializing"
                

                for specific_structure_index, specific_structure in enumerate(pydicom_item[bx_structs]):
                    specific_bx_structure_roi = specific_structure["ROI"]

                    bx_best_fit_line_of_reconstructed_centroids = specific_structure['Best fit line of centroid pts']
                    vec_with_largest_z_val_index = bx_best_fit_line_of_reconstructed_centroids[:,2].argmax()
                    vec_with_largest_z_val = bx_best_fit_line_of_reconstructed_centroids[vec_with_largest_z_val_index,:]
                    base_sup_vec_bx_centroid_arr = vec_with_largest_z_val

                    vec_with_smallest_z_val_index = bx_best_fit_line_of_reconstructed_centroids[:,2].argmin()
                    vec_with_smallest_z_val = bx_best_fit_line_of_reconstructed_centroids[vec_with_smallest_z_val_index,:]
                    apex_inf_vec_bx_centroid_arr = vec_with_smallest_z_val

                    translation_vec_bx_coord_sys_origin = -apex_inf_vec_bx_centroid_arr
                    apex_to_base_bx_best_fit_vec = base_sup_vec_bx_centroid_arr - apex_inf_vec_bx_centroid_arr

                    reconstructed_biopsy_point_cloud = specific_structure["Reconstructed structure point cloud"]
                    reconstructed_biopsy_arr = specific_structure["Reconstructed structure pts arr"]
                    sampled_bx_points_pcd = specific_structure["Random uniformly sampled volume pts pcd"]
                    sampled_bx_points_arr = specific_structure["Random uniformly sampled volume pts arr"]
                    axis_aligned_bounding_box = specific_structure["Bounding box for random uniformly sampled volume pts"]
                    
                    reconstructed_biopsy_bx_coord_sys_tr_arr = reconstructed_biopsy_arr + translation_vec_bx_coord_sys_origin
                    sampled_bx_points_bx_coord_sys_tr_arr = sampled_bx_points_arr + translation_vec_bx_coord_sys_origin
                    reconstructed_biopsy_bx_coord_sys_tr_point_cloud = copy.copy(reconstructed_biopsy_point_cloud)
                    reconstructed_biopsy_bx_coord_sys_tr_from_arr_point_cloud = point_containment_tools.create_point_cloud(reconstructed_biopsy_bx_coord_sys_tr_arr)
                    sampled_bx_points_bx_coord_sys_tr_pcd = copy.copy(sampled_bx_points_pcd)
                    
                    reconstructed_biopsy_bx_coord_sys_tr_point_cloud.translate(translation_vec_bx_coord_sys_origin)
                    sampled_bx_points_bx_coord_sys_tr_pcd.translate(translation_vec_bx_coord_sys_origin)

                    patients_progress.stop_task(processing_patients_task)
                    completed_progress.stop_task(processing_patients_completed_task)
                    stopwatch.stop()
                    plotting_funcs.plot_geometries(sampled_bx_points_pcd, reconstructed_biopsy_point_cloud, axis_aligned_bounding_box, reconstructed_biopsy_bx_coord_sys_tr_point_cloud, sampled_bx_points_bx_coord_sys_tr_pcd)
                    stopwatch.start()
                    patients_progress.start_task(processing_patients_task)
                    completed_progress.start_task(processing_patients_completed_task)

                    z_axis_np_vec = np.array([0,0,1],dtype=float)

                    z_rot_angle = mf.angle_between(z_axis_np_vec, apex_to_base_bx_best_fit_vec)
                    xyz_rotation_arr = np.array([0,0,z_rot_angle], dtype=float)
                    centroid_line_to_z_axis_rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz(xyz_rotation_arr)

                    reconstructed_biopsy_bx_coord_sys_tr_and_rot_point_cloud = copy.copy(reconstructed_biopsy_bx_coord_sys_tr_point_cloud)
                    sampled_bx_points_bx_coord_sys_tr_and_rot_pcd = copy.copy(sampled_bx_points_bx_coord_sys_tr_pcd)
                    
                    #reconstructed_biopsy_bx_coord_sys_tr_and_rot_point_cloud.rotate(centroid_line_to_z_axis_rotation_matrix, center=(0, 0, 0))
                    #sampled_bx_points_bx_coord_sys_tr_and_rot_pcd.rotate(centroid_line_to_z_axis_rotation_matrix, center=(0, 0, 0))
                    
                    centroid_line_to_z_axis_rotation_matrix_other = mf.rotation_matrix_from_vectors(apex_to_base_bx_best_fit_vec, z_axis_np_vec)
                    reconstructed_biopsy_bx_coord_sys_tr_and_rot_point_cloud.rotate(centroid_line_to_z_axis_rotation_matrix_other, center=(0, 0, 0))
                    sampled_bx_points_bx_coord_sys_tr_and_rot_pcd.rotate(centroid_line_to_z_axis_rotation_matrix_other, center=(0, 0, 0))

                    reconstructed_biopsy_bx_coord_sys_tr_and_rot_arr = (centroid_line_to_z_axis_rotation_matrix_other @ reconstructed_biopsy_bx_coord_sys_tr_arr.T).T
                    sampled_bx_points_bx_coord_sys_tr_and_rot_arr = (centroid_line_to_z_axis_rotation_matrix_other @ sampled_bx_points_bx_coord_sys_tr_arr.T).T
                    reconstructed_biopsy_bx_coord_sys_tr_and_rot_arr_point_cloud = point_containment_tools.create_point_cloud(reconstructed_biopsy_bx_coord_sys_tr_and_rot_arr)
                    sampled_bx_points_bx_coord_sys_tr_and_rot_arr_point_cloud = point_containment_tools.create_point_cloud(sampled_bx_points_bx_coord_sys_tr_and_rot_arr)

                    # plot shifted and rotated biopsies to ensure transformations are correct for biopsy coord system?
                    #coord_frame = o3d.geometry.create_mesh_coordinate_frame()
                    patients_progress.stop_task(processing_patients_task)
                    completed_progress.stop_task(processing_patients_completed_task)
                    stopwatch.stop()
                    #plotting_funcs.plot_geometries(sampled_bx_points_pcd, reconstructed_biopsy_point_cloud, axis_aligned_bounding_box, reconstructed_biopsy_bx_coord_sys_tr_point_cloud, sampled_bx_points_bx_coord_sys_tr_pcd, reconstructed_biopsy_bx_coord_sys_tr_and_rot_point_cloud, sampled_bx_points_bx_coord_sys_tr_and_rot_pcd)
                    #plotting_funcs.plot_geometries(sampled_bx_points_pcd, reconstructed_biopsy_point_cloud, axis_aligned_bounding_box, reconstructed_biopsy_bx_coord_sys_tr_and_rot_point_cloud, sampled_bx_points_bx_coord_sys_tr_and_rot_pcd)                    
                    #plotting_funcs.plot_geometries_with_axes(sampled_bx_points_pcd, reconstructed_biopsy_point_cloud, axis_aligned_bounding_box, reconstructed_biopsy_bx_coord_sys_tr_point_cloud, sampled_bx_points_bx_coord_sys_tr_pcd, reconstructed_biopsy_bx_coord_sys_tr_and_rot_point_cloud, sampled_bx_points_bx_coord_sys_tr_and_rot_pcd)
                    #plotting_funcs.plot_geometries_with_axes(sampled_bx_points_pcd, reconstructed_biopsy_point_cloud, axis_aligned_bounding_box, reconstructed_biopsy_bx_coord_sys_tr_and_rot_point_cloud, sampled_bx_points_bx_coord_sys_tr_and_rot_pcd)
                    #plotting_funcs.plot_geometries_with_axes(sampled_bx_points_pcd, reconstructed_biopsy_point_cloud, axis_aligned_bounding_box, reconstructed_biopsy_bx_coord_sys_tr_from_arr_point_cloud, sampled_bx_points_bx_coord_sys_tr_pcd, reconstructed_biopsy_bx_coord_sys_tr_and_rot_arr_point_cloud, sampled_bx_points_bx_coord_sys_tr_and_rot_pcd)
                    #plotting_funcs.plot_geometries_with_axes(sampled_bx_points_pcd, reconstructed_biopsy_point_cloud, axis_aligned_bounding_box, reconstructed_biopsy_bx_coord_sys_tr_and_rot_arr_point_cloud, sampled_bx_points_bx_coord_sys_tr_and_rot_pcd)
                    stopwatch.start()
                    
                    # check if the arrays are equal? using the two different methods
                    sampled_bx_points_bx_coord_sys_tr_and_rot_arr_from_pcd_transform = np.asarray(sampled_bx_points_bx_coord_sys_tr_and_rot_pcd.points)
                    
                    
                    specific_structure["Random uniformly sampled volume pts bx coord sys arr"] = sampled_bx_points_bx_coord_sys_tr_and_rot_arr
                    specific_structure["Random uniformly sampled volume pts bx coord sys pcd"] = sampled_bx_points_bx_coord_sys_tr_and_rot_arr_point_cloud

                    patients_progress.start_task(processing_patients_task)
                    completed_progress.start_task(processing_patients_completed_task)


                    

                patients_progress.update(processing_patients_task, advance = 1)
                completed_progress.update(processing_patients_completed_task, advance = 1)

            
            patients_progress.update(processing_patients_task, visible = False)
            completed_progress.update(processing_patients_completed_task, visible = True)



            live_display.stop()
            
            ## begin simulation section
            created_dir = False
            while created_dir == False:
                live_display.console.print("[bold red]User input required:")
                print('>Must create an uncertainties folder at ', uncertainty_dir, '. If the folder already exists it will NOT be overwritten.')
                stopwatch.stop()
                uncertainty_dir_generate = ques_funcs.ask_ok('>Continue?')
                stopwatch.start()

                if uncertainty_dir_generate == True:
                    if os.path.isdir(uncertainty_dir) == True:
                        print('>Directory already exists')
                        created_dir = True
                    else:
                        os.mkdir(uncertainty_dir)
                        print('>Directory: ', uncertainty_dir, ' created.')
                        created_dir = True
                else:
                    stopwatch.stop()
                    exit_programme = ques_funcs.ask_ok('>This directory must be created. Do you want to exit the programme?' )
                    stopwatch.start()
                    if exit_programme == True:
                        sys.exit('>Programme exited.')
                    else: 
                        pass
            
            stopwatch.stop()
            uncertainty_template_generate = ques_funcs.ask_ok('>Do you want to generate an uncertainty file template for this patient data repo?')
            stopwatch.start()
            if uncertainty_template_generate == True:
                # create a blank uncertainties file filled with the proper patient data, it is uniquely IDd by including the date and time in the file name
                #today = date.today()
                #date_file_name_format = today.strftime("%b-%d-%Y")

                date_time_now = datetime.now()
                date_time_now_file_name_format = date_time_now.strftime(" Date-%b-%d-%Y Time-%H,%M,%S")
                uncertainties_file = uncertainty_dir.joinpath(uncertainty_file_name+date_time_now_file_name_format+uncertainty_file_extension)

                uncertainty_file_writer.uncertainty_file_preper(uncertainties_file, master_structure_reference_dict, structs_referenced_list, num_general_structs)
            else:
                pass

            uncertainty_file_ready = False
            while uncertainty_file_ready == False:
                stopwatch.stop()
                uncertainty_file_ready = ques_funcs.ask_ok('>Is the uncertainty file prepared/filled out?') 
                stopwatch.start()
                if uncertainty_file_ready == True:
                    print('>Please select the file with the dialog box')
                    root = tk.Tk() # these two lines are to get rid of errant tkinter window
                    root.withdraw() # these two lines are to get rid of errant tkinter window
                    # this is a user defined quantity, should be a tab delimited csv file in the future, mu sigma for each uncertainty direction
                    uncertainties_file_filled = fd.askopenfilename(title='Open the uncertainties data file', initialdir=data_dir, filetypes=[("Excel files", ".xlsx .xls .csv")])
                    with open(uncertainties_file_filled, "r", newline='\n') as uncertainties_file_filled_csv:
                        uncertainties_filled = uncertainties_file_filled_csv
                    pandas_read_uncertainties = pandas.read_csv(uncertainties_file_filled, names = [0, 1, 2, 3, 4, 5])  
                    print(pandas_read_uncertainties)
                else:
                    print('>Please fill out the generated uncertainties file generated at ', uncertainties_file)
                    stopwatch.stop()
                    ask_to_quit = ques_funcs.ask_ok('>Would you like to quit the programme instead?')
                    stopwatch.start()
                    if ask_to_quit == True:
                        sys.exit(">You have quit the programme.")
                    else:
                        pass

            # Transfer read uncertainty data to master_reference
            num_general_structs = int(pandas_read_uncertainties.values[1][0])
            for specific_structure_index in range(num_general_structs):
                structure_row_num_start = specific_structure_index*5+3
                patientUID = pandas_read_uncertainties.values[structure_row_num_start+1][0]
                structure_type = pandas_read_uncertainties.values[structure_row_num_start+1][1]
                structure_ROI = pandas_read_uncertainties.values[structure_row_num_start+1][2]
                structure_ref_num = pandas_read_uncertainties.values[structure_row_num_start+1][3]
                master_ref_dict_specific_structure_index = int(pandas_read_uncertainties.values[structure_row_num_start+1][4])
                frame_of_reference = pandas_read_uncertainties.values[structure_row_num_start+1][5]
                means_arr = np.empty([3], dtype=float)
                sigmas_arr = np.empty([3], dtype=float)

                means_arr[0] = pandas_read_uncertainties.values[structure_row_num_start+3][0] # X
                means_arr[1] = pandas_read_uncertainties.values[structure_row_num_start+3][2] # Y
                means_arr[2] = pandas_read_uncertainties.values[structure_row_num_start+3][4] # Z

                sigmas_arr[0] = pandas_read_uncertainties.values[structure_row_num_start+3][1] # X
                sigmas_arr[1] = pandas_read_uncertainties.values[structure_row_num_start+3][3] # Y
                sigmas_arr[2] = pandas_read_uncertainties.values[structure_row_num_start+3][5] # Z


                uncertainty_data_obj = uncertainty_data(patientUID, structure_type, structure_ROI, structure_ref_num, master_ref_dict_specific_structure_index, frame_of_reference)
                uncertainty_data_obj.fill_means_and_sigmas(means_arr, sigmas_arr)
                master_structure_reference_dict[patientUID][structure_type][master_ref_dict_specific_structure_index]["Uncertainty data"] = uncertainty_data_obj

            stopwatch.stop()
            simulation_ans = ques_funcs.ask_ok('>Uncertainty data collected. Begin Monte Carlo simulation?')
            stopwatch.start()

            num_simulations = num_MC_simulations_input
            master_structure_info_dict["Global"]["MC info"]["Num MC simulations"] = num_simulations
            if simulation_ans ==  True:
                print('>Beginning simulation')
                master_structure_reference_dict = MC_simulator_convex.simulator_parallel(parallel_pool, live_display, layout_groups, master_structure_reference_dict, structs_referenced_list, dose_ref, master_structure_info_dict, biopsy_z_voxel_length, spinner_type)
                #master_structure_reference_dict_simulated = MC_simulator_convex.simulator(master_structure_reference_dict, structs_referenced_list,num_simulations, pandas_read_uncertainties)
                print('test')
            else: 
                pass

            live_display.stop()
            live_display.console.print("[bold green]Simulation complete.")
            live_display.console.print("[bold red]User input required:")

            stopwatch.stop()
            write_containment_to_file_ans = ques_funcs.ask_ok('>Save containment output to file?')
            stopwatch.start()
            created_output_dir = False
            specific_output_dir_exists = False
            if write_containment_to_file_ans ==  True:
                while created_output_dir == False:
                    
                    print('>Must create an output folder at ', output_dir, '. If the folder already exists it will NOT be overwritten.')
                    stopwatch.stop()
                    output_dir_generate = ques_funcs.ask_ok('>Continue?')
                    stopwatch.start()

                    if output_dir_generate == True:
                        if os.path.isdir(output_dir) == True:
                            print('>Directory already exists')
                            created_output_dir = True
                        else:
                            os.mkdir(output_dir)
                            print('>Directory: ', output_dir, ' created.')
                            created_output_dir = True
                    else:
                        stopwatch.stop()
                        exit_programme = ques_funcs.ask_ok('>This directory must be created. Do you want to exit the programme?' )
                        stopwatch.start()
                        if exit_programme == True:
                            sys.exit('>Programme exited.')
                        else: 
                            pass

                date_time_now = datetime.now()
                date_time_now_file_name_format = date_time_now.strftime(" Date-%b-%d-%Y Time-%H,%M,%S")
                specific_output_dir_name = 'MC_sim_out-'+date_time_now_file_name_format
                specific_output_dir = output_dir.joinpath(specific_output_dir_name)

                print('>Creating specific output directory.')
                if os.path.isdir(specific_output_dir) == True:
                    print('>Directory already exists.')
                    specific_output_dir_exists = True
                else:
                    os.mkdir(specific_output_dir)
                    print('>Directory: ', specific_output_dir, ' created.')
                    specific_output_dir_exists = True

                
                for patientUID,pydicom_item in master_structure_reference_dict.items():
                    bx_structs = structs_referenced_list[0]
                    for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structs]):
                        bx_points_bx_coords_sys_arr = specific_bx_structure["Random uniformly sampled volume pts bx coord sys arr"]
                        bx_points_bx_coords_sys_arr_list = list(bx_points_bx_coords_sys_arr)
                        bx_points_bx_coords_sys_arr_list.insert(0,'')
                        containment_output_file_name = patientUID+','+specific_bx_structure['ROI']+',n_MC='+str(num_simulations)+',n_bx='+str(num_sample_pts_per_bx)+'-containment_out.csv'
                        containment_output_csv_file_path = specific_output_dir.joinpath(containment_output_file_name)
                        with open(containment_output_csv_file_path, 'w', newline='') as f:
                            write = csv.writer(f)
                            write.writerow(['Patient ID ->',patientUID])
                            write.writerow(['BX ID ->',specific_bx_structure['ROI']])
                            write.writerow(['BX length (from contour data) (mm)', specific_bx_structure['Reconstructed biopsy cylinder length (from contour data)']])
                            write.writerow(['Num MC sims ->',num_simulations])
                            write.writerow(['Num bx pt samples ->',num_sample_pts_per_bx])
                            write.writerow(['Row ->','Fixed containment structure'])
                            write.writerow(['Col ->','Fixed bx point'])
                            write.writerow(bx_points_bx_coords_sys_arr_list)
                            for containment_structure_key_tuple, containment_structure_dict in specific_bx_structure['MC data: compiled sim results'].items():
                                containment_structure_list = containment_structure_dict['Total successes (containment) list']
                                containment_structure_ROI = containment_structure_key_tuple[0]
                                containment_structure_list_with_cont_anat_ROI = [containment_structure_ROI]+containment_structure_list
                                write.writerow(containment_structure_list_with_cont_anat_ROI)


                for patientUID,pydicom_item in master_structure_reference_dict.items():
                    bx_structs = structs_referenced_list[0]
                    for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structs]):
                        containment_voxelized_output_file_name = patientUID+','+specific_bx_structure['ROI']+',n_MC='+str(num_simulations)+',n_bx='+str(num_sample_pts_per_bx)+'-containment_voxelized_out.csv'
                        containment_voxelized_output_csv_file_path = specific_output_dir.joinpath(containment_voxelized_output_file_name)
                        with open(containment_voxelized_output_csv_file_path, 'w', newline='') as f:
                            write = csv.writer(f)
                            write.writerow(['Patient ID ->',patientUID])
                            write.writerow(['BX ID ->',specific_bx_structure['ROI']])
                            write.writerow(['BX length (from contour data) (mm)', specific_bx_structure['Reconstructed biopsy cylinder length (from contour data)']])
                            write.writerow(['Num MC sims ->',num_simulations])
                            write.writerow(['Num bx pt samples ->',num_sample_pts_per_bx])
                            write.writerow(['Row ->','Fixed containment structure'])
                            write.writerow(['Col ->','Fixed voxel'])
                                                        
                            for containment_structure_key_tuple, voxelized_containment_structure_dict in specific_bx_structure["MC data: voxelized containment results dict (dict of lists)"].items():
                                containment_structure_ROI = containment_structure_key_tuple[0]
                                num_voxels = voxelized_containment_structure_dict["Num voxels"]
                                voxel_index_row = list(range(num_voxels))
                                voxel_index_row.insert(0,'Voxel index')
                                biopsy_z_voxel_range_row = voxelized_containment_structure_dict["Voxel z range"].copy()
                                rounded_biopsy_z_voxel_range_row = [[round(sub_list[0],2),round(sub_list[1],2)] for sub_list in biopsy_z_voxel_range_row]
                                biopsy_z_voxel_range_row.insert(0,'Voxel z range (mm)')
                                rounded_biopsy_z_voxel_range_row.insert(0,'Voxel z range (mm)')
                                num_sample_pts_in_voxel_row = voxelized_containment_structure_dict["Num sample pts in voxel"].copy()
                                num_sample_pts_in_voxel_row.insert(0, 'Num sample pts in vxl')
                                arth_mean_binomial_estimator_row = voxelized_containment_structure_dict["Arithmetic mean of binomial estimators in voxel"].copy()
                                arth_mean_binomial_estimator_row.insert(0, 'Arth mean (binomial estimator)')
                                std_dev_binomial_estimator_row = voxelized_containment_structure_dict["Std dev of binomial estimators in voxel"].copy()
                                std_dev_binomial_estimator_row.insert(0, 'Std dev (binomial estimator)')

                                write.writerow([containment_structure_ROI])
                                write.writerow(voxel_index_row)
                                write.writerow(biopsy_z_voxel_range_row)
                                write.writerow(rounded_biopsy_z_voxel_range_row)
                                write.writerow(num_sample_pts_in_voxel_row)
                                write.writerow(arth_mean_binomial_estimator_row)
                                write.writerow(std_dev_binomial_estimator_row)
                                write.writerow([''])

            else:
                pass


            stopwatch.stop()
            write_dose_to_file_ans = ques_funcs.ask_ok('>Save dose output to file?')
            stopwatch.start()

            if write_dose_to_file_ans ==  True:
                if created_output_dir == False:
                    while created_output_dir == False:
                        
                        print('>Must create an output folder at ', output_dir, '. If the folder already exists it will NOT be overwritten.')
                        stopwatch.stop()
                        output_dir_generate = ques_funcs.ask_ok('>Continue?')
                        stopwatch.start()

                        if output_dir_generate == True:
                            if os.path.isdir(output_dir) == True:
                                print('>Directory already exists')
                                created_output_dir = True
                            else:
                                os.mkdir(output_dir)
                                print('>Directory: ', output_dir, ' created.')
                                created_output_dir = True
                        else:
                            stopwatch.stop()
                            exit_programme = ques_funcs.ask_ok('>This directory must be created. Do you want to exit the programme?' )
                            stopwatch.start()
                            if exit_programme == True:
                                sys.exit('>Programme exited.')
                            else: 
                                pass
                else:
                    pass
                if specific_output_dir_exists == False:
                    date_time_now = datetime.now()
                    date_time_now_file_name_format = date_time_now.strftime(" Date-%b-%d-%Y Time-%H,%M,%S")
                    specific_output_dir_name = 'MC_sim_out-'+date_time_now_file_name_format
                    specific_output_dir = output_dir.joinpath(specific_output_dir_name)

                    print('>Creating specific output directory.')
                    if os.path.isdir(specific_output_dir) == True:
                        print('>Directory already exists.')
                        specific_output_dir_exists = True
                    else:
                        os.mkdir(specific_output_dir)
                        print('>Directory: ', specific_output_dir, ' created.')
                        specific_output_dir_exists = True
                else:
                    pass

                
                for patientUID,pydicom_item in master_structure_reference_dict.items():
                    bx_structs = structs_referenced_list[0]
                    for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structs]):
                        bx_points_bx_coords_sys_arr = specific_bx_structure["Random uniformly sampled volume pts bx coord sys arr"]
                        bx_points_bx_coords_sys_arr_list = list(bx_points_bx_coords_sys_arr)
                        dose_output_file_name = patientUID+','+specific_bx_structure['ROI']+',n_MC='+str(num_simulations)+',n_bx='+str(num_sample_pts_per_bx)+'-dose_out.csv'
                        dose_output_csv_file_path = specific_output_dir.joinpath(dose_output_file_name)
                        with open(dose_output_csv_file_path, 'w', newline='') as f:
                            write = csv.writer(f)
                            write.writerow(['Patient ID ->',patientUID])
                            write.writerow(['BX ID ->',specific_bx_structure['ROI']])
                            write.writerow(['BX length (from contour data) (mm)', specific_bx_structure['Reconstructed biopsy cylinder length (from contour data)']])
                            write.writerow(['Num MC sims ->',num_simulations])
                            write.writerow(['Num bx pt samples ->',num_sample_pts_per_bx])
                            write.writerow(['Row ->','Fixed bx pt'])
                            write.writerow(['Col ->','Fixed MC trial'])
                            for pt_index, dose_vals_row in enumerate(specific_bx_structure['MC data: Dose vals for each sampled bx pt list']):
                                dose_vals_row_with_point = dose_vals_row.copy()
                                dose_vals_row_with_point.insert(0,bx_points_bx_coords_sys_arr_list[pt_index])
                                write.writerow(dose_vals_row_with_point)


                for patientUID,pydicom_item in master_structure_reference_dict.items():
                    bx_structs = structs_referenced_list[0]
                    for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structs]):
                        dose_voxelized_output_file_name = patientUID+','+specific_bx_structure['ROI']+',n_MC='+str(num_simulations)+',n_bx='+str(num_sample_pts_per_bx)+'-dose_voxelized_out.csv'
                        dose_voxelized_output_csv_file_path = specific_output_dir.joinpath(dose_voxelized_output_file_name)
                        with open(dose_voxelized_output_csv_file_path, 'w', newline='') as f:
                            write = csv.writer(f)
                            write.writerow(['Patient ID ->',patientUID])
                            write.writerow(['BX ID ->',specific_bx_structure['ROI']])
                            write.writerow(['BX length (from contour data) (mm)', specific_bx_structure['Reconstructed biopsy cylinder length (from contour data)']])
                            write.writerow(['Num MC sims ->',num_simulations])
                            write.writerow(['Num bx pt samples ->',num_sample_pts_per_bx])
                            write.writerow(['Row ->','Info'])
                            write.writerow(['Col ->','Fixed voxel'])
                                                        
                            voxelized_dose_dict = specific_bx_structure['MC data: voxelized dose results dict (dict of lists)']
                            num_voxels = voxelized_dose_dict["Num voxels"]
                            voxel_index_row = list(range(num_voxels))
                            voxel_index_row.insert(0,'Voxel index')
                            biopsy_z_voxel_range_row = voxelized_dose_dict["Voxel z range"].copy()
                            rounded_biopsy_z_voxel_range_row = [[round(sub_list[0],2),round(sub_list[1],2)] for sub_list in biopsy_z_voxel_range_row]
                            biopsy_z_voxel_range_row.insert(0,'Voxel z range (mm)')
                            rounded_biopsy_z_voxel_range_row.insert(0,'Voxel z range (mm)')
                            num_sample_pts_in_voxel_row = voxelized_dose_dict["Num sample pts in voxel"].copy()
                            num_sample_pts_in_voxel_row.insert(0, 'Num sample pts in vxl')
                            num_MC_trials_in_voxel_row = voxelized_dose_dict['Total num MC trials in voxel'].copy()
                            num_MC_trials_in_voxel_row.insert(0, 'Num MC trials in vxl')
                            arth_mean_dose_row = voxelized_dose_dict["Arithmetic mean of dose in voxel"].copy()
                            arth_mean_dose_row.insert(0, 'Arth mean (dose)')
                            std_dev_dose_row = voxelized_dose_dict["Std dev of dose in voxel"].copy()
                            std_dev_dose_row.insert(0, 'Std dev (binomial estimator)')

                            write.writerow(voxel_index_row)
                            write.writerow(biopsy_z_voxel_range_row)
                            write.writerow(rounded_biopsy_z_voxel_range_row)
                            write.writerow(num_sample_pts_in_voxel_row)
                            write.writerow(num_MC_trials_in_voxel_row)
                            write.writerow(arth_mean_dose_row)
                            write.writerow(std_dev_dose_row)
                            for i in range(5):
                                write.writerow([''])
                            
                            write.writerow(['Row ->','Fixed voxel'])
                            write.writerow(['Col ->','Dose values'])
                            
                            voxelized_dose_list = specific_bx_structure['MC data: voxelized dose results list']

                            for voxel_index, voxel_dict in enumerate(voxelized_dose_list):
                                dose_vals_in_voxel_row = voxel_dict['All dose vals in voxel list'].copy()
                                dose_vals_in_voxel_row.insert(0,voxel_index)
                                write.writerow(dose_vals_in_voxel_row)
            else:
                pass
            print('>Programme has ended.')

def UID_generator(pydicom_obj):
    UID_def = f"{str(pydicom_obj[0x0010,0x0010].value)} ({str(pydicom_obj[0x0010,0x0020].value)})"
    return UID_def


def structure_referencer(structure_dcm_dict, dose_dcm_dict, OAR_list,DIL_list,Bx_list):
    """
    A function that builds a reference library of the dicom elements passed to it so that 
    we can match the ROI name to the contour information, since the contour
    information is referenced to the name by a number.
    """
    master_st_ds_ref_dict = {}
    master_st_ds_info_dict = {}
    master_st_ds_info_global_dict = {"Global": None, "By patient": None}
    st_ref_list = ["Bx ref","OAR ref","DIL ref"] # note that Bx ref has to be the first entry for other parts of the code to work!
    ds_ref = "Dose ref"
    global_num_biopsies = 0
    global_num_OAR = 0
    global_num_DIL = 0
    global_total_num_structs = 0
    global_num_patients = 0
    for UID, structure_item in structure_dcm_dict.items():
        bpsy_ref = [{"ROI":x.ROIName, "Ref #":x.ROINumber, "Reconstructed biopsy cylinder length (from contour data)": None, "Raw contour pts": None, "Equal num zslice contour pts": None, "Intra-slice interpolation information": None, "Inter-slice interpolation information": None, "Point cloud raw": None, "Delaunay triangulation global structure": None, "Delaunay triangulation zslice-wise list": None, "Structure centroid pts": None, "Best fit line of centroid pts": None, "Centroid line sample pts": None, "Reconstructed structure pts arr": None, "Reconstructed structure point cloud": None, "Reconstructed structure delaunay global": None, "Random uniformly sampled volume pts arr": None, "Random uniformly sampled volume pts pcd": None, "Random uniformly sampled volume pts bx coord sys arr": None, "Random uniformly sampled volume pts bx coord sys pcd": None, "Bounding box for random uniformly sampled volume pts": None, "Uncertainty data": None, "MC data: Generated normal dist random samples arr": None, "MC data: bx only shifted 3darr": None, "MC data: bx and structure shifted dict": None, "MC data: MC sim translation results dict": None, "MC data: compiled sim results": None, "MC data: voxelized containment results dict": None, "MC data: voxelized containment results dict (dict of lists)": None, "MC data: bx to dose NN search objects list": None, "MC data: Dose NN child obj for each sampled bx pt list": None, "MC data: Dose vals for each sampled bx pt list": None, "MC data: Dose statistics (MLE) for each sampled bx pt list (mean, std)": None, "MC data: voxelized dose results list": None, "MC data: voxelized dose results dict (dict of lists)": None, "KDtree": None, "Nearest neighbours objects": [], "Plot attributes": plot_attributes()} for x in structure_item.StructureSetROISequence if any(i in x.ROIName for i in Bx_list)]    
        OAR_ref = [{"ROI":x.ROIName, "Ref #":x.ROINumber, "Raw contour pts": None, "Equal num zslice contour pts": None, "Intra-slice interpolation information": None, "Inter-slice interpolation information": None, "Point cloud raw": None, "Delaunay triangulation global structure": None, "Delaunay triangulation zslice-wise list": None, "Structure centroid pts": None, "Best fit line of centroid pts": None, "Centroid line sample pts": None, "Reconstructed structure pts arr": None, "Reconstructed structure point cloud": None, "Reconstructed structure delaunay global": None, "Uncertainty data": None, "MC data: Generated normal dist random samples arr": None, "KDtree": None, "Nearest neighbours objects": [], "Plot attributes": plot_attributes()} for x in structure_item.StructureSetROISequence if any(i in x.ROIName for i in OAR_list)]
        DIL_ref = [{"ROI":x.ROIName, "Ref #":x.ROINumber, "Raw contour pts": None, "Equal num zslice contour pts": None, "Intra-slice interpolation information": None, "Inter-slice interpolation information": None, "Point cloud raw": None, "Delaunay triangulation global structure": None, "Delaunay triangulation zslice-wise list": None, "Structure centroid pts": None, "Best fit line of centroid pts": None, "Centroid line sample pts": None, "Reconstructed structure pts arr": None, "Reconstructed structure point cloud": None, "Reconstructed structure delaunay global": None, "Uncertainty data": None, "MC data: Generated normal dist random samples arr": None, "KDtree": None, "Nearest neighbours objects": [], "Plot attributes": plot_attributes()} for x in structure_item.StructureSetROISequence if any(i in x.ROIName for i in DIL_list)] 

        bpsy_info = {"Num structs": len(bpsy_ref)}
        OAR_info = {"Num structs": len(OAR_ref)}
        DIL_info = {"Num structs": len(DIL_ref)}
        patient_total_num_structs = bpsy_info["Num structs"] + OAR_info["Num structs"] + DIL_info["Num structs"]
        all_structs_info = {"Total num structs": patient_total_num_structs}
        
        global_num_OAR = global_num_OAR + OAR_info["Num structs"]
        global_num_DIL = global_num_DIL + DIL_info["Num structs"] 
        global_num_biopsies = global_num_biopsies + bpsy_info["Num structs"]
        global_total_num_structs = global_total_num_structs + patient_total_num_structs
        global_num_patients = global_num_patients + 1

        master_st_ds_ref_dict[UID] = {"Patient ID":str(structure_item[0x0010,0x0020].value),"Patient Name":str(structure_item[0x0010,0x0010].value),st_ref_list[0]:bpsy_ref, st_ref_list[1]:OAR_ref, st_ref_list[2]:DIL_ref,"Ready to plot data list": None}
        master_st_ds_info_dict[UID] = {"Patient ID":str(structure_item[0x0010,0x0020].value),"Patient Name":str(structure_item[0x0010,0x0010].value),st_ref_list[0]:bpsy_info, st_ref_list[1]:OAR_info, st_ref_list[2]:DIL_info, "All ref":all_structs_info}
    
    for UID, dose_item in dose_dcm_dict.items():
        dose_ID = UID + dose_item.StudyDate
        dose_ref_dict = {"Dose ID": dose_ID, "Study date": dose_item.StudyDate, "Dose pixel data": dose_item.PixelData, "Dose pixel arr": dose_item.pixel_array, "Pixel spacing": [float(item) for item in dose_item.PixelSpacing], "Dose grid scaling": float(dose_item.DoseGridScaling), "Dose units": dose_item.DoseUnits, "Dose type": dose_item.DoseType, "Grid frame offset vector": [float(item) for item in dose_item.GridFrameOffsetVector], "Image orientation patient": [float(item) for item in dose_item.ImageOrientationPatient], "Image position patient": [float(item) for item in dose_item.ImagePositionPatient], "Dose phys space and pixel 3d arr": None, "Dose grid point cloud": None, "Dose grid point cloud thresholded": None, "KDtree": None}
        master_st_ds_ref_dict[UID][ds_ref] = dose_ref_dict

    mc_info = {"Num MC simulations": None, "Num sample pts per BX core": None}
    master_st_ds_info_global_dict["Global"] = {"Num patients": global_num_patients, "Num structures": global_total_num_structs, "Num biopsies": global_num_biopsies, "Num DILs": global_num_DIL, "Num OARs": global_num_OAR, "MC info": mc_info}
    master_st_ds_info_global_dict["By patient"] = master_st_ds_info_dict
    return master_st_ds_ref_dict, master_st_ds_info_global_dict, st_ref_list, ds_ref

class uncertainty_data:
    def __init__(self, patientUID, struct_type, structure_roi, struct_ref_num, master_ref_dict_specific_structure_index, frame_of_reference):
        self.patientUID = patientUID
        self.struct_type = struct_type
        self.structure_roi = structure_roi
        self.struct_ref_num = struct_ref_num
        self.master_ref_dict_specific_structure_index = master_ref_dict_specific_structure_index
        self.uncertainty_data_mean_arr = None
        self.uncertainty_data_sigma_arr = None
        self.uncertainty_data_mean_dict = {"Frame of reference": frame_of_reference, "Distribution": 'Normal', "Means": None, "Sigmas": None} 
    def fill_means_and_sigmas(self, means_arr, sigmas_arr):
        self.uncertainty_data_mean_arr = means_arr
        self.uncertainty_data_sigma_arr = sigmas_arr



class plot_attributes:
    def __init__(self,plot_bool_init = True):
        self.plot_bool = plot_bool_init
        self.color_raw = 'r'
        self.color_best_fit = 'g' 


class nearest_neighbour_parent:
    def __init__(self,BX_struct_name,comparison_struct_name,comparison_struct_type,comparison_structure_points_that_made_KDtree,queried_BX_points,NN_search_output):
        self.BX_structure_name = BX_struct_name
        self.comparison_structure_name = comparison_struct_name
        self.comparison_structure_type = comparison_struct_type
        #self.comparison_structure_points = comparison_structure_points_that_made_KDtree
        self.queried_Bx_points = queried_BX_points
        self.NN_search_output = NN_search_output
        self.NN_data_list = self.NN_list_builder(comparison_structure_points_that_made_KDtree)

    def NN_list_builder(self,comparison_structure_points_that_made_KDtree):
        comparison_structure_NN_distances = self.NN_search_output[0]
        comparison_structure_NN_indices = self.NN_search_output[1]
        nearest_points_on_comparison_struct = comparison_structure_points_that_made_KDtree[comparison_structure_NN_indices]
        
        NN_data_list = [nearest_neighbour_child(self.queried_Bx_points[index], nearest_points_on_comparison_struct[index], comparison_structure_NN_distances[index]) for index in range(0,len(self.queried_Bx_points))]
        #NN_data_list = [{"Queried BX pt": self.queried_Bx_points[index], "NN pt on comparison struct": nearest_points_on_comparison_struct[index], "Euclidean distance": comparison_structure_NN_distances[index]} for index in range(0,len(self.queried_Bx_points))]
        return NN_data_list


class nearest_neighbour_child:
    def __init__(self, queried_BX_pt, NN_pt_on_comparison_struct, euclidean_dist):
        self.queried_BX_pt = queried_BX_pt
        self.NN_pt_on_comparison_struct = NN_pt_on_comparison_struct
        self.euclidean_dist = euclidean_dist

class delaunay_obj:
    def __init__(self, np_points, delaunay_tri_color):
        self.delaunay_triangulation = self.scipy_delaunay_triangulation(np_points)
        self.delaunay_line_set = self.line_set(np_points, self.delaunay_triangulation, delaunay_tri_color)

    def scipy_delaunay_triangulation(self, numpy_points):
        delaunay_triang = scipy.spatial.Delaunay(numpy_points)
        return delaunay_triang

    def collect_edges(self, tri):
        edges = set()

        def sorted_tuple(a,b):
            return (a,b) if a < b else (b,a)
        # Add edges of tetrahedron (sorted so we don't add an edge twice, even if it comes in reverse order).
        for (i0, i1, i2, i3) in tri.simplices:
            edges.add(sorted_tuple(i0,i1))
            edges.add(sorted_tuple(i0,i2))
            edges.add(sorted_tuple(i0,i3))
            edges.add(sorted_tuple(i1,i2))
            edges.add(sorted_tuple(i1,i3))
            edges.add(sorted_tuple(i2,i3))
        return edges

    def line_set(self, points, tri, color):
        edges = self.collect_edges(tri)
        colors = [[color[0], color[1], color[2]] for i in range(len(edges))]
        x = np.array([])
        y = np.array([])
        z = np.array([])
        for (i,j) in edges:
            x = np.append(x, [points[i, 0], points[j, 0], np.nan])      
            y = np.append(y, [points[i, 1], points[j, 1], np.nan])      
            z = np.append(z, [points[i, 2], points[j, 2], np.nan])

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(edges)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set

class interpolation_information_obj:
    def __init__(self,num_z_slices_raw):
        self.interpolate_distance = None
        self.scipylinesegments_by_zslice_keys_dict = {}
        self.numpoints_after_interpolation_per_zslice_dict = {}
        self.numpoints_raw_per_zslice_dict = {}
        self.interpolated_pts_list = []
        self.interpolated_pts_np_arr = None
        self.num_z_slices_raw = num_z_slices_raw
        #self.z_slice_seg_obj_list_temp = None
        self.endcaps_points = []
        self.interpolated_pts_with_end_caps_list = None
        self.interpolated_pts_with_end_caps_np_arr = None 

    def parallel_analyze(self, parallel_pool, three_Ddata_list,interp_dist):
        pool = parallel_pool
        self.interpolate_distance = interp_dist
        parallel_result = pool.map(self.analyze_structure_slice, three_Ddata_list)
        for result in parallel_result:
            zslice_key = result[0] 
            z_slice_seg_obj_list = result[1]
            numpoints_raw_per_zslice = result[2]
            numpoints_after_interpolation_per_zslice_temp  = result[3]
            threeDdata_zslice_interpolated_list = result[4]

            self.scipylinesegments_by_zslice_keys_dict[zslice_key] = z_slice_seg_obj_list
            self.numpoints_raw_per_zslice_dict[zslice_key] = numpoints_raw_per_zslice
            self.numpoints_after_interpolation_per_zslice_dict[zslice_key] = numpoints_after_interpolation_per_zslice_temp
            for interpolated_point in threeDdata_zslice_interpolated_list:
                self.interpolated_pts_list.append(interpolated_point)
        self.interpolated_pts_np_arr = np.asarray(self.interpolated_pts_list)

                
    
    def analyze_structure_slice(self, threeDdata_zslice):
        interp_dist = self.interpolate_distance
        numpoints_raw_per_zslice_temp = None
        numpoints_after_interpolation_per_zslice_temp = None
        z_val = threeDdata_zslice[0,2] 
        current_zslice_num_points = np.size(threeDdata_zslice,0)
        #z_slice_seg_obj_list_temp = self.create_zslice(z_val, current_zslice_num_points)
        num_segments_in_zslice = current_zslice_num_points
        zslice_key = z_val
        z_slice_seg_obj_list_temp = [None]*num_segments_in_zslice
        #self.numpoints_raw_per_zslice_dict[zslice_key] = num_points_in_zslice_raw
        numpoints_raw_per_zslice_temp = current_zslice_num_points
        

        threeDdata_zslice_interpolated_list = []
        zslice_pt_counter = current_zslice_num_points
        for j in range(0,current_zslice_num_points):
            if j < current_zslice_num_points-1:
                segment_points = threeDdata_zslice[j:j+2,0:3]
            else:
                segment_points = np.empty([2,3], dtype = float)
                segment_points[0,0:3] = threeDdata_zslice[j,0:3]
                segment_points[1,0:3] = threeDdata_zslice[0,0:3]
            
            segment_vec = segment_points[1,:] - segment_points[0,:]
            segment_length = np.linalg.norm(segment_vec)
            segment_obj = anatomy_reconstructor_tools.line_segment_obj(segment_vec,segment_length,segment_points)
            z_slice_seg_obj_list_temp[j] = segment_obj
            num_interpolations_on_seg = int(np.floor(segment_length/interp_dist))
            t_vals_with_end_points = np.linspace(0, 1, num=num_interpolations_on_seg+2) # generate the t values to evaluate along the longest segment 
            t_vals_without_end_points = t_vals_with_end_points[1:-1]
            interpolated_segment_list = []
            for t_val in t_vals_without_end_points:
                new_point = np.empty([1,3],dtype=float)
                new_point = segment_obj.new_xyz_via_vector_travel(t_val)
                interpolated_segment_list.append(new_point)
            
            first_point = segment_points[0,:]
            threeDdata_zslice_interpolated_list.append(first_point)
            for interpolated_point in interpolated_segment_list:
                threeDdata_zslice_interpolated_list.append(interpolated_point)
            zslice_pt_counter = zslice_pt_counter + num_interpolations_on_seg

        #self.numpoints_after_interpolation_per_zslice_dict[z_val] = zslice_pt_counter
        numpoints_after_interpolation_per_zslice_temp = zslice_pt_counter
        #self.insert_zslice(z_val, z_slice_seg_obj_list_temp)
        #for interpolated_point in threeDdata_zslice_interpolated_list:
        #    interpolated_pts_list.append(interpolated_point)
        #self.interpolated_pts_np_arr = np.asarray(self.interpolated_pts_list)
        # plot slicewise for debugging ?
        #plotting_funcs.plot_point_clouds(self.interpolated_pts_np_arr, label='Unknown')
        return zslice_key, z_slice_seg_obj_list_temp, numpoints_raw_per_zslice_temp, numpoints_after_interpolation_per_zslice_temp, threeDdata_zslice_interpolated_list

    def create_zslice(self, zslice_key, num_points_in_zslice_raw): # call this first
        num_segments_in_zslice = num_points_in_zslice_raw
        z_slice_seg_obj_list_temp = self.prealloc_zslice_list(num_segments_in_zslice)
        self.numpoints_raw_per_zslice_dict[zslice_key] = num_points_in_zslice_raw
        return z_slice_seg_obj_list_temp

    def prealloc_zslice_list(self,num_segments_in_zslice): # this is automatically used by the class
        zslice_segments_list = [None]*num_segments_in_zslice
        return zslice_segments_list      
        
    def insert_zslice(self, zslice_key,z_slice_seg_obj_list): # then use this after all iterations are complete
        self.scipylinesegments_by_zslice_keys_dict[zslice_key] = z_slice_seg_obj_list


    def create_fill(self, threeDdata_zslice, maximum_point_distance):
        if self.interpolated_pts_with_end_caps_list == None:
            self.interpolated_pts_with_end_caps_list = self.interpolated_pts_list.copy()
        else:
            pass

        z_val = threeDdata_zslice[0,2]
        min_x, min_y = np.amin(threeDdata_zslice[:,0:2], axis=0)
        max_x, max_y = np.amax(threeDdata_zslice[:,0:2], axis=0)
        grid_spacing = maximum_point_distance/np.sqrt(2)
        fill_points_xy_grid_arr = np.mgrid[min_x-grid_spacing:max_x+grid_spacing:grid_spacing, min_y-grid_spacing:max_y+grid_spacing:grid_spacing].reshape(2, -1).T
        fill_points_xyz_grid_arr = np.empty((len(fill_points_xy_grid_arr),3), dtype = float)
        fill_points_xyz_grid_arr[:,0:2] = fill_points_xy_grid_arr
        fill_points_xyz_grid_arr[:,2] = z_val
        fill_points_xyz_grid_list = fill_points_xyz_grid_arr.tolist()
        twoD_zslice_data_arr = threeDdata_zslice[:,0:2]
        twoD_zslice_data_list = twoD_zslice_data_arr.tolist()
        zslice_polygon_shapely = Polygon(twoD_zslice_data_list)
        threeDdata_zslice_fill_list = []
        for index, test_point in enumerate(fill_points_xyz_grid_list):
            test_point_shapely = Point(test_point)
            if test_point_shapely.within(zslice_polygon_shapely):
                threeDdata_zslice_fill_list.append(test_point)
        for fill_point in threeDdata_zslice_fill_list:
            fill_point_as_arr = np.asarray(fill_point)
            self.endcaps_points.append(fill_point_as_arr)
            self.interpolated_pts_with_end_caps_list.append(fill_point_as_arr)
        self.interpolated_pts_with_end_caps_np_arr = np.asarray(self.interpolated_pts_with_end_caps_list)

 
def trimesh_reconstruction_ball_pivot(threeD_data_arr, ball_radii):
    num_points = threeD_data_arr.shape[0]
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(threeD_data_arr)
    pcd_color = np.random.uniform(0, 0.7, size=3)
    point_cloud.paint_uniform_color(pcd_color) 
    point_cloud.estimate_normals()
    #point_cloud.orient_normals_consistent_tangent_plane(num_points)
    point_cloud.orient_normals_to_align_with_direction(np.array([0.0,0.0,0.0],dtype=float))
    point_cloud.normalize_normals()
    struct_tri_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                            point_cloud,  o3d.utility.DoubleVector(ball_radii))
    return struct_tri_mesh

def trimesh_reconstruction_poisson(threeD_data_arr):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(threeD_data_arr)
    pcd_color = np.random.uniform(0, 0.7, size=3)
    point_cloud.paint_uniform_color(pcd_color) 
    point_cloud.estimate_normals()
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        struct_tri_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=9)
    return struct_tri_mesh

def trimesh_reconstruction_alphashape(threeD_data_arr):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(threeD_data_arr)
    pcd_color = np.random.uniform(0, 0.7, size=3)
    point_cloud.paint_uniform_color(pcd_color) 
    point_cloud.estimate_normals()
    alpha = 0.03
    tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(point_cloud)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(point_cloud, alpha, tetra_mesh, pt_map)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)




if __name__ == '__main__':    
    main()
    