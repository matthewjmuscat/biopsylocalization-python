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
import plotly.express as px
import shutil
import statsmodels.api as sm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from statsmodels.nonparametric import kernel_regression
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.regression.quantile_regression import QuantRegResults
import misc_tools
import matplotlib.colors as mcolors
import production_plots


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
    input_data_folder_name = "Input data"
    modality_list = ['RTSTRUCT','RTDOSE','RTPLAN']
    #oaroi_contour_names = ['Prostate','Urethra','Rectum','Normal', 'CTV','random'] 
    oaroi_contour_names = ['Prostate'] # consider prostate only for OARs! If the first position is the prostate, the simulated biopsies will be generated relative to this structure
    biopsy_contour_names = ['Bx']
    dil_contour_names = ['DIL']
    uncertainty_folder_name = 'Uncertainty data'
    uncertainty_file_name = "uncertainties_file_auto_generated"
    uncertainty_file_extension = ".csv"
    spinner_type = 'line'
    output_folder_name = 'Output data'
    lower_bound_dose_percent = 5
    color_flattening_deg = 3 
    interp_inter_slice_dist = 0.5
    interp_intra_slice_dist = 1 # user defined length scale for intraslice interpolation min distance between points. It is used in the interpolation_information_obj class
    interp_dist_caps = 2
    biopsy_radius = 0.275
    biopsy_needle_compartment_length = 19 # length in millimeters of the biopsy needle core compartment
    simulate_uniform_bx_shifts_due_to_bx_needle_compartment = True
    #num_sample_pts_per_bx_input = 250 # uncommenting this line will do nothing, this line is deprecated in favour of constant cubic lattice spacing
    bx_sample_pts_lattice_spacing = 0.2
    num_MC_containment_simulations_input = 10
    num_MC_dose_simulations_input = 100
    biopsy_z_voxel_length = 0.5 #voxelize biopsy core every 0.5 mm along core
    num_dose_calc_NN = 8
    num_dose_NN_to_show_for_animation_plotting = 100
    num_bootstraps_all_MC_data_input = 15
    pio.templates.default = "plotly_white"
    svg_image_scale = 3
    NPKR_bandwidth = 0.5
    svg_image_height = 1080
    svg_image_width = 1920
    open3d_views_jsons_folder_name = "open3d_views_jsons"
    open3d_views_dose_folder_name = "dose_views"
    open3d_views_containment_folder_name = "containment_views"
    open_3d_screen_views_dose_jsons = ["ScreenCamera_2023-03-15-12-33-41.json", 
                                       "ScreenCamera_2023-03-15-12-33-53.json",
                                       "ScreenCamera_2023-03-15-13-07-02.json", 
                                       "ScreenCamera_2023-03-15-13-08-08.json"
                                       ]
    open_3d_screen_views_containment_jsons = ["ScreenCamera_2023-02-19-15-14-47.json", 
                                              "ScreenCamera_2023-02-19-15-27-46.json",
                                              "ScreenCamera_2023-02-19-15-14-47.json", 
                                              "ScreenCamera_2023-02-19-15-29-43.json"
                                              ]
    
    bx_sim_locations = ["centroid"] # change to empty list if dont want to create any simulated biopsies. Also the code at the moment only supports creating centroid simulated biopsies.
    bx_sim_ref_identifier = "sim"
    simulate_biopsies_relative_to = ['DIL'] # can include elements in the list such as "DIL" or "Prostate"...
    differential_dvh_resolution = 100 # the number of bins
    cumulative_dvh_resolution = 100 # the larger the number the more resolution the cDVH calculations will have
    display_dvh_as = ['counts','percent', 'volume'] # can be 'counts', 'percent', 'volume'
    num_cumulative_dvh_plots_to_show = 50
    num_differential_dvh_plots_to_show = 50
    volume_DVH_percent_dose = [100,125,150,200,300]


    # plots to show:
    show_NN_dose_demonstration_plots = False
    show_containment_demonstration_plots = False
    show_3d_dose_renderings = False
    show_processed_3d_datasets_renderings = True
    show_processed_3d_datasets_renderings_plotly = False
    show_reconstructed_biopsy_in_biopsy_coord_sys_tr_and_rot = False
    plot_uniform_shifts_to_check_plotly = False # if this is true, will produce many plots if num_simulations is high!
    plot_translation_vectors_pointclouds = False

    # Final production plots to create:
    num_z_vals_to_evaluate_for_regression_plots = 1000
    production_plots_input_dictionary = {"Sampled translation vector magnitudes box plots": \
                                            {"Plot bool": True, "Name": "sampled_translations_magnitudes_box_plot"}, 
                                        "Axial spatial dose distribution": 

                                        }
    


    # non-user changeable variables, but need to be initiatied:
    bx_ref = "Bx ref"
    oar_ref = "OAR ref"
    dil_ref = "DIL ref"
    structs_referenced_dict = {bx_ref: biopsy_contour_names, oar_ref: oaroi_contour_names, dil_ref: dil_contour_names} 
    structs_referenced_list = list(structs_referenced_dict.keys()) # note that Bx ref has to be the first entry for other parts of the code to work!
    dose_ref = "Dose ref"
    plan_ref = "Plan ref"
    num_simulated_bxs_to_create = len(bx_sim_locations)

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


            # set the paths for the JSON views for the NN dose demonstration
            open3d_views_jsons_dir = pathlib.Path(__file__).parents[1].joinpath(open3d_views_jsons_folder_name)
            dose_views_jsons_dir = open3d_views_jsons_dir.joinpath(open3d_views_dose_folder_name)
            dose_views_jsons_paths_list = [dose_views_jsons_dir.joinpath(name) for name in open_3d_screen_views_dose_jsons]

            # set the paths for the JSON views for the containment demonstration
            containment_views_jsons_dir = open3d_views_jsons_dir.joinpath(open3d_views_containment_folder_name)
            containment_views_jsons_paths_list = [containment_views_jsons_dir.joinpath(name) for name in open_3d_screen_views_containment_jsons]

            # The figure dictionary to be plotted, this needs to be requested of the user later in the programme, after the  dicoms are read
            # First we access the data directory, it must be in a location 
            # two levels up from this file
            data_dir = pathlib.Path(__file__).parents[2].joinpath(data_folder_name)
            uncertainty_dir = data_dir.joinpath(uncertainty_folder_name)
            output_dir = data_dir.joinpath(output_folder_name)
            input_dir = data_dir.joinpath(input_data_folder_name)

            misc_tools.checkdirs(live_display, important_info, data_dir,uncertainty_dir,output_dir,input_dir)
            #data_dir.mkdir(parents=True, exist_ok=True)
            #uncertainty_dir.mkdir(parents=True, exist_ok=True)
            #output_dir.mkdir(parents=True, exist_ok=True)
            #input_dir.mkdir(parents=True, exist_ok=True)
            dicom_paths_list = list(pathlib.Path(input_dir).glob("**/*.dcm")) # list all file paths found in the data folder that have the .dcm extension
            important_info.add_text_line("Reading dicom data from: "+ str(input_dir), live_display)
            important_info.add_text_line("Reading uncertainty data from: "+ str(uncertainty_dir), live_display)
            
            #live_display.stop()
            num_dicoms = len(dicom_paths_list)
            if num_dicoms == 0:
                live_display.stop()
                while num_dicoms == 0:
                    print("The input folder is empty!")
                    print("Reading dicom data from: "+ str(input_dir))
                    print("Fill input folder with data then continue.")
                    continue_programme = ques_funcs.ask_ok('> Continue?' )
                    if continue_programme == False:
                        sys.exit('> Programme exited.')
                    else:
                        dicom_paths_list = list(pathlib.Path(input_dir).glob("**/*.dcm"))
                        num_dicoms = len(dicom_paths_list)
                live_display.start()

            important_info.add_text_line("Found "+str(num_dicoms)+" dicom files.", live_display)
            reading_dicoms_task_indeterminate = indeterminate_progress_main.add_task('[red]Reading dicom data from file...', total=None)
            reading_dicoms_task_indeterminate_completed = completed_progress.add_task('[green]Reading dicom data from file', total=num_dicoms, visible = False)
            dicom_elems_modality_list = []
            for dicom_path in dicom_paths_list:
                with pydicom.dcmread(dicom_path, defer_size = '2 MB') as py_dicom_item:
                    dicom_elems_modality_list.append(copy.deepcopy(py_dicom_item[0x0008,0x0060].value))
            #dicom_elems_list = list(map(pydicom.dcmread,dicom_paths_list)) # read all the found dicom file paths using pydicom to create a list of FileDataset instances 
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
            RTst_dcms_dict = {}
            RTdose_dcms_dict = {}
            RTplan_dcms_dict = {}
            for dicom_path_index, dicom_path in enumerate(dicom_paths_list):
                if dicom_elems_modality_list[dicom_path_index] == modality_list[0]:
                    with pydicom.dcmread(dicom_path, defer_size = '2 MB') as py_dicom_item: 
                        RTst_dcms_dict[UID_generator(py_dicom_item)] = dicom_path
                elif dicom_elems_modality_list[dicom_path_index] == modality_list[1]:
                    with pydicom.dcmread(dicom_path, defer_size = '2 MB') as py_dicom_item: 
                        RTdose_dcms_dict[UID_generator(py_dicom_item)] = dicom_path
                elif dicom_elems_modality_list[dicom_path_index] == modality_list[2]:
                    with pydicom.dcmread(dicom_path, defer_size = '2 MB') as py_dicom_item: 
                        RTplan_dcms_dict[UID_generator(py_dicom_item)] = dicom_path

            #RTst_dcms_dict = {UID_generator(pydicom.dcmread(dicom_paths_list[j])): pydicom.dcmread(dicom_paths_list[j]) for j in range(num_dicoms) if dicom_elems_modality_list[j] == modality_list[0]}
            #RTdose_dcms_dict = {UID_generator(pydicom.dcmread(dicom_paths_list[j])): pydicom.dcmread(dicom_paths_list[j]) for j in range(num_dicoms) if dicom_elems_modality_list[j] == modality_list[1]}
            #live_display.stop()
            num_RTst_dcms_entries = len(RTst_dcms_dict)
            num_RTdose_dcms_entries = len(RTdose_dcms_dict)
            num_RTplan_dcms_entries = len(RTplan_dcms_dict)
            important_info.add_text_line("Found "+str(num_RTst_dcms_entries)+" unique patients with RT structure files.", live_display)
            important_info.add_text_line("Found "+str(num_RTdose_dcms_entries)+" unique patients with RT dose files.", live_display)
            important_info.add_text_line("Found "+str(num_RTplan_dcms_entries)+" unique patients with RT plan files.", live_display)


            # check if the found files make sense
            num_RTst_neq_RTdose = False
            num_RTst_neq_RTplan = False
            num_RTdose_neq_RTplan = False
            if num_RTst_dcms_entries != num_RTdose_dcms_entries:
                num_RTst_neq_RTdose = True
            if num_RTst_dcms_entries != num_RTplan_dcms_entries:
                num_RTst_neq_RTplan = True
            if num_RTdose_dcms_entries != num_RTplan_dcms_entries:
                num_RTdose_neq_RTplan = True

            if num_RTdose_neq_RTplan or num_RTst_neq_RTplan or num_RTst_neq_RTdose:
                live_display.stop()
                stopwatch.stop()
                continue_programme = ques_funcs.ask_ok('>Unequal number of structure files('+str(num_RTst_dcms_entries)+ \
                                                    ') dose files ('+str(num_RTdose_dcms_entries)+\
                                                    '), to plan files ('+str(num_RTplan_dcms_entries)+\
                                                    ') will encounter error later in the programme. Continue anyway?')
                stopwatch.start()
                if continue_programme == False:
                    sys.exit('>Programme exited.')
                else:
                    important_info.add_text_line("There are NOT the same number of structure, dose and plan files.", live_display)
            else: 
                important_info.add_text_line("There are the same number of structure, dose and plan files.", live_display)   

            
            

            # check if each patient has the correct files
            num_RTst_neq_RTdose_keys = False
            num_RTst_neq_RTplan_keys = False
            num_RTdose_neq_RTplan_keys = False
            if RTst_dcms_dict.keys() != RTdose_dcms_dict.keys():
                num_RTst_neq_RTdose_keys = True
            if RTst_dcms_dict.keys() != RTplan_dcms_dict.keys():
                num_RTst_neq_RTplan_keys = True            
            if RTdose_dcms_dict.keys() != RTplan_dcms_dict.keys():
                num_RTdose_neq_RTplan_keys = True

            if num_RTst_neq_RTdose_keys or num_RTst_neq_RTplan_keys or num_RTdose_neq_RTplan_keys:
                live_display.stop()
                stopwatch.stop()
                exit_programme = ques_funcs.ask_ok('>Same number of structure files, dose files and plan files but there is an incongruency between them (file pairs do not match patients), will encounter error later in the programme. Continue anyway?' )
                stopwatch.start()
                if exit_programme == True:
                    sys.exit('>Programme exited.')
                else:
                    important_info.add_text_line("Each patient does NOT contain a structure, dose and plan file.", live_display) 
            else: 
                important_info.add_text_line("Each patient contains a structure, dose and plan file.", live_display)    
            
            
            # setting some variables for use in simulating biopsies
            if len(bx_sim_locations) >= 1:
                simulate_biopsies_relative_to_struct_type_list = [None]*len(simulate_biopsies_relative_to)
                for bx_sim_relative_structure_index, bx_sim_relative_structure in enumerate(simulate_biopsies_relative_to):
                    keyfound = False
                    for struct_type_key in structs_referenced_dict.keys():
                        if bx_sim_relative_structure in structs_referenced_dict[struct_type_key]:
                            if keyfound == True:
                                raise Exception("Structure specified to simulate biopsies to found in more than one structure type.")
                            simulate_biopsies_relative_to_struct_type_list[bx_sim_relative_structure_index] = struct_type_key
                            keyfound = True
                    if keyfound == False:
                        raise Exception("Structure specified to simulate biopsies to was not found in specified structures to analyse.")
                important_info.add_text_line("Simulating "+ ", ".join(bx_sim_locations)+" biopsies relative to "+", ".join(simulate_biopsies_relative_to)+" (Found under "+ ", ".join(simulate_biopsies_relative_to_struct_type_list)+").", live_display)          
                live_display.refresh()
            else: 
                important_info.add_text_line("Not creating any simulated biopsies.")          
                live_display.refresh() 
            
            
            
            # patient dictionary creation
            building_patient_dictionaries_task = indeterminate_progress_main.add_task('[red]Building patient dictionary...', total=None)
            building_patient_dictionaries_task_completed = completed_progress.add_task('[green]Building patient dictionary', total=num_RTst_dcms_entries, visible = False)
            master_structure_reference_dict, master_structure_info_dict = structure_referencer(RTst_dcms_dict, 
                                                                                               RTdose_dcms_dict,
                                                                                               RTplan_dcms_dict, 
                                                                                               oaroi_contour_names,
                                                                                               dil_contour_names,
                                                                                               biopsy_contour_names,
                                                                                               structs_referenced_list,
                                                                                               dose_ref,
                                                                                               plan_ref,
                                                                                               bx_sim_locations,
                                                                                               bx_sim_ref_identifier,
                                                                                               simulate_biopsies_relative_to,
                                                                                               simulate_biopsies_relative_to_struct_type_list,
                                                                                               bx_sample_pts_lattice_spacing)
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

                dose_point_cloud = plotting_funcs.create_dose_point_cloud(phys_space_dose_map_3d_arr, color_flattening_deg, paint_dose_color = True)
                dose_ref_dict["Dose grid point cloud"] = dose_point_cloud
                
                # plot labelled dose point cloud (note due to the number of labels, this is very buggy and doesnt display properly as of open3d 0.16.1)
                """
                patients_progress.stop_task(processing_patients_dose_task)
                completed_progress.stop_task(processing_patients_dose_task_completed)
                stopwatch.stop()
                plotting_funcs.dose_point_cloud_with_dose_labels(phys_space_dose_map_3d_arr, paint_dose_with_color = True)
                stopwatch.start()
                patients_progress.start_task(processing_patients_dose_task)
                completed_progress.start_task(processing_patients_dose_task_completed)
                """
                # plot dose point cloud cubic lattice (color only)
                if show_3d_dose_renderings == True:
                    patients_progress.stop_task(processing_patients_dose_task)
                    completed_progress.stop_task(processing_patients_dose_task_completed)
                    stopwatch.stop()
                    plotting_funcs.plot_geometries(dose_point_cloud)
                    stopwatch.start()
                    patients_progress.start_task(processing_patients_dose_task)
                    completed_progress.start_task(processing_patients_dose_task_completed)

                # user defined quantity moved to beginning of programme
                thresholded_dose_point_cloud = plotting_funcs.create_thresholded_dose_point_cloud(phys_space_dose_map_3d_arr, color_flattening_deg, paint_dose_color = True, lower_bound_percent = lower_bound_dose_percent)
                dose_ref_dict["Dose grid point cloud thresholded"] = thresholded_dose_point_cloud
                
                # plot dose point cloud thresholded cubic lattice (color only)
                if show_3d_dose_renderings == True:
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


            # create info for simulated biopsies
            if num_simulated_bxs_to_create >= 1:
                centroid_line_vec_list = [0,0,1]
                centroid_first_pos_list = [0,0,0]
                num_centroids_for_sim_bxs = 10
                centroid_sep_dist = biopsy_needle_compartment_length/(num_centroids_for_sim_bxs-1) # the minus 1 ensures that the legnth of the biopsy is actually correct!
                simulated_bx_rad = 2
                plot_simulated_cores_immediately = False


            patientUID_default = "Initializing"
            pulling_patients_task_main_description = "[red]Pulling patient structure data [{}]...".format(patientUID_default)
            pulling_patients_task_completed_main_description = "[green]Pulling patient structure data"
            pulling_patients_task = patients_progress.add_task(pulling_patients_task_main_description, total=num_patients)
            pulling_patients_task_completed = completed_progress.add_task(pulling_patients_task_completed_main_description, total=num_patients, visible = False) 
                    
            
            for patientUID,pydicom_item in master_structure_reference_dict.items():
                pulling_patients_task_main_description = "[red]Processing patient structure data [{}]...".format(patientUID)
                patients_progress.update(pulling_patients_task, description = pulling_patients_task_main_description)

                structureID_default = "Initializing"
                num_general_structs_patient_specific = master_structure_info_dict["By patient"][patientUID]["All ref"]["Total num structs"]
                pulling_structures_task_main_description = "[cyan]Processing structures [{},{}]...".format(patientUID,structureID_default)
                pulling_structures_task = structures_progress.add_task(pulling_structures_task_main_description, total=num_general_structs_patient_specific)
                for structs in structs_referenced_list:
                    for specific_structure_index, specific_structure in enumerate(pydicom_item[structs]):
                        structureID = specific_structure["ROI"]
                        structure_reference_number = specific_structure["Ref #"]
                        pulling_structures_task_main_description = "[cyan]Processing structures [{},{}]...".format(patientUID,structureID)
                        structures_progress.update(pulling_structures_task, description = pulling_structures_task_main_description)
                        
                        # create points for simulated biopsies to create
                        if bx_sim_ref_identifier in str(structure_reference_number):
                            threeDdata_zslice_list = biopsy_creator.biopsy_points_creater_by_transport_for_sim_bxs(centroid_line_vec_list,centroid_first_pos_list,num_centroids_for_sim_bxs,centroid_sep_dist,simulated_bx_rad,plot_simulated_cores_immediately)
                        # otherwise just read the data from dicoms
                        else:
                            threeDdata_zslice_list = []
                            with pydicom.dcmread(RTst_dcms_dict[patientUID], defer_size = '2 MB') as py_dicom_item: 
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
                        if structs != bx_ref:
                            structure_centroids_array = np.empty([len(threeDdata_zslice_list),3])
                            # find zslice-wise centroids
                            for index, threeDdata_zslice in enumerate(threeDdata_zslice_list):                           
                                structure_zslice_centroid = np.mean(threeDdata_zslice,axis=0)
                                structure_centroids_array[index] = structure_zslice_centroid
                            structure_global_centroid = centroid_finder.centeroidfinder_numpy_3D(structure_centroids_array)
                            master_structure_reference_dict[patientUID][structs][specific_structure_index]["Structure centroid pts"] = structure_centroids_array
                            master_structure_reference_dict[patientUID][structs][specific_structure_index]["Structure global centroid"] = structure_global_centroid

                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Raw contour pts zslice list"] = threeDdata_zslice_list

                        structures_progress.update(pulling_structures_task, advance=1)
                structures_progress.remove_task(pulling_structures_task)
                patients_progress.update(pulling_patients_task, advance=1)
                completed_progress.update(pulling_patients_task_completed, advance=1)
            patients_progress.update(pulling_patients_task, visible=False)
            completed_progress.update(pulling_patients_task_completed,  visible=True)            


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
                        structure_reference_number = specific_structure["Ref #"]
                        processing_structures_task_main_description = "[cyan]Processing structures [{},{}]...".format(patientUID,structureID)
                        structures_progress.update(processing_structures_task, description = processing_structures_task_main_description)

                        # The below print lines were just for my own understanding of how to access the data structure
                        #print(specific_structure["ROI"])
                        #print(specific_structure["Ref #"])
                        #print(RTst_dcms[dcm_index].ROIContourSequence[int(specific_structure["Ref #"])].ContourSequence[0].ContourData)
                        #print(RTst_dcms[dcm_index].ROIContourSequence[int(specific_structure["Ref #"])].ContourSequence[1].ContourData)

                        # can uncomment surrounding lines to time this particular process
                        #st = time.time()

                        # create vectors and info for simulated biopsies
                        

                        threeDdata_zslice_list = master_structure_reference_dict[patientUID][structs][specific_structure_index]["Raw contour pts zslice list"].copy()
                        
                        # transform simulated biopsies to location
                        if structs == bx_ref and bx_sim_ref_identifier in str(structure_reference_number):
                            # first extract the appropriate relative structure to transform biopsies to
                            relative_structure_ref_num_from_bx_info = master_structure_reference_dict[patientUID][structs][specific_structure_index]["Relative structure ref #"]
                            for relative_struct_type in simulate_biopsies_relative_to_struct_type_list:
                                for relative_specific_structure_index, relative_specific_structure in enumerate(master_structure_reference_dict[patientUID][relative_struct_type]):
                                    if relative_structure_ref_num_from_bx_info == relative_specific_structure["Ref #"]:
                                        simulated_bx_relative_to_specific_structure_index = relative_specific_structure_index
                                        simulate_biopsies_relative_to_specific_structure_struct_type = relative_struct_type
                                        break
                                    else:
                                        pass
                            relative_structure_for_sim_bx_global_centroid = master_structure_reference_dict[patientUID][simulate_biopsies_relative_to_specific_structure_struct_type][simulated_bx_relative_to_specific_structure_index]["Structure global centroid"].copy()
                            threeDdata_arr_temp = np.concatenate(threeDdata_zslice_list, axis=0)
                            simulated_bx_global_centroid_before_translation = centroid_finder.centeroidfinder_numpy_3D(threeDdata_arr_temp)
                            translation_vector_to_relative_structure_centroid = relative_structure_for_sim_bx_global_centroid - simulated_bx_global_centroid_before_translation
                            threeDdata_zslice_list_temp = threeDdata_zslice_list.copy()
                            for bx_zslice_arr_index, bx_zslice_arr in enumerate(threeDdata_zslice_list_temp):
                                temp_bx_zslice_arr = bx_zslice_arr.copy()
                                translated_bx_zslice_arr = temp_bx_zslice_arr + translation_vector_to_relative_structure_centroid
                                threeDdata_zslice_list_temp[bx_zslice_arr_index] = translated_bx_zslice_arr
                            threeDdata_zslice_list = threeDdata_zslice_list_temp
                            # at this point the created biopsy centroid has been shifted to the global centroid of the relative structure
                            # now we want to move each created biopsy to the appropriate sub position within the relative structure
                            
                            # CODE REMOVED: because after discussion with Nathan, simulating this 12 core pattern to compare to targeted biopsy is not as interesting to him 
                            #If this code needs to be reinstated, it needs to be reworked, variables have changed.
                            """
                            relative_structure_for_sim_bx_global_centroid = master_structure_reference_dict[patientUID][simulate_biopsies_relative_to_specific_structure_struct_type][simulated_bx_relative_to_specific_structure_index]["Structure global centroid"].copy()
                            threeD_data_zslice_list_relative_structure = master_structure_reference_dict[patientUID][simulate_biopsies_relative_to_specific_structure_struct_type][simulated_bx_relative_to_specific_structure_index]["Raw contour pts zslice list"].copy()
                            closest_zslice_index_of_relative_structure_to_its_global_centroid = misc_tools.find_closest_z_slice(threeD_data_zslice_list_relative_structure, relative_structure_for_sim_bx_global_centroid)
                            closest_zslice_of_relative_structure_arr = threeD_data_zslice_list_relative_structure[closest_zslice_index_of_relative_structure_to_its_global_centroid]
                            centroid_of_zslice_of_relative_structure = centroid_finder.centeroidfinder_numpy_3D(closest_zslice_of_relative_structure_arr)
                            """
                        
                        
                        total_structure_points = sum([np.shape(x)[0] for x in threeDdata_zslice_list])
                        threeDdata_array = np.empty([total_structure_points,3])
                        # for biopsy only
                        if structs == bx_ref:
                            structure_centroids_array = np.empty([len(threeDdata_zslice_list),3])
                        lower_bound_index = 0  
                        # build raw threeDdata
                        for index, threeDdata_zslice in enumerate(threeDdata_zslice_list):
                            current_zslice_num_points = np.size(threeDdata_zslice,0)
                            threeDdata_array[lower_bound_index:lower_bound_index + current_zslice_num_points] = threeDdata_zslice
                            lower_bound_index = lower_bound_index + current_zslice_num_points 
                            
                            if structs == bx_ref:
                                structure_zslice_centroid = np.mean(threeDdata_zslice,axis=0)
                                structure_centroids_array[index] = structure_zslice_centroid


                           
                        # conduct INTER-slice interpolation
                        interslice_interpolation_information, threeDdata_equal_pt_zslice_list = anatomy_reconstructor_tools.inter_zslice_interpolator(parallel_pool, threeDdata_zslice_list, interp_inter_slice_dist)
                        
                        # conduct INTRA-slice interpolation
                        # do you want to interpolate the zslice interpolated data or the raw data? comment out the appropriate line below..
                        threeDdata_to_intra_zslice_interpolate_zslice_list = interslice_interpolation_information.interpolated_pts_list
                        # threeDdata_to_intra_zslice_interpolate_zslice_list = threeDdata_zslice_list

                        num_z_slices_data_to_intra_slice_interpolate = len(threeDdata_to_intra_zslice_interpolate_zslice_list)
                        
                        # SLOWER TO ANALYZE PARALLEL
                        #interpolation_information = interpolation_information_obj(num_z_slices_data_to_intra_slice_interpolate)
                        #interpolation_information.parallel_analyze(parallel_pool, threeDdata_to_intra_zslice_interpolate_zslice_list,interp_intra_slice_dist)
                        

                        # FASTER TO ANALYZE SERIALLY
                        interpolation_information = interpolation_information_obj(num_z_slices_data_to_intra_slice_interpolate)
                        interpolation_information.serial_analyze(threeDdata_to_intra_zslice_interpolate_zslice_list,interp_intra_slice_dist)
                        

                        #for index, threeDdata_zslice in enumerate(threeDdata_to_intra_zslice_interpolate_zslice_list):
                        #    interpolation_information.analyze_structure_slice(threeDdata_zslice,interp_intra_slice_dist)

                        # fill in the end caps
                        first_zslice = threeDdata_to_intra_zslice_interpolate_zslice_list[0]
                        last_zslice = threeDdata_to_intra_zslice_interpolate_zslice_list[-1]
                        interpolation_information.create_fill(first_zslice, interp_dist_caps)
                        interpolation_information.create_fill(last_zslice, interp_dist_caps)

                        

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
                        pcd_struct_rand_color = np.random.uniform(0, 0.9, size=3)
                        interslice_interp_pcd = point_containment_tools.create_point_cloud(threeDdata_array_interslice_interpolation, pcd_struct_rand_color)
                        inter_and_intra_interp_pcd = point_containment_tools.create_point_cloud(threeDdata_array_fully_interpolated, pcd_struct_rand_color)
                        inter_and_intra_and_end_caps_interp_pcd = point_containment_tools.create_point_cloud(threeDdata_array_fully_interpolated_with_end_caps, pcd_struct_rand_color)
                        interpolated_pcd_dict = {"Interslice": interslice_interp_pcd, "Full": inter_and_intra_interp_pcd, "Full with end caps": inter_and_intra_and_end_caps_interp_pcd}
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
                        if structs == bx_ref:
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
                        if structs == bx_ref:
                            specific_structure["Plot attributes"].plot_bool = True



                        # store all calculated quantities
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Raw contour pts"] = threeDdata_array
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Equal num zslice contour pts"] = threeDdata_equal_pt_zslice_list
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Inter-slice interpolation information"] = interslice_interpolation_information                        
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Intra-slice interpolation information"] = interpolation_information
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Delaunay triangulation zslice-wise list"] = deulaunay_objs_zslice_wise_list
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Delaunay triangulation global structure"] = delaunay_global_convex_structure_obj
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Point cloud raw"] = threeDdata_point_cloud
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Interpolated structure point cloud dict"] = interpolated_pcd_dict
                        if structs == bx_ref:
                            master_structure_reference_dict[patientUID][structs][specific_structure_index]["Structure centroid pts"] = structure_centroids_array
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

            # displays 3d renderings of patient contour data and dose data
            if show_processed_3d_datasets_renderings == True:
                for patientUID,pydicom_item in master_structure_reference_dict.items():
                    dose_ref_dict = pydicom_item[dose_ref]
                    dose_grid_pcd = dose_ref_dict["Dose grid point cloud thresholded"]
                    pcd_list = [dose_grid_pcd]
                    for structs in structs_referenced_list:
                        for specific_structure_index, specific_structure in enumerate(pydicom_item[structs]):
                            if structs == bx_ref: 
                                structure_pcd = specific_structure["Reconstructed structure point cloud"]
                            else: 
                                #structure_pcd = specific_structure["Point cloud raw"]
                                structure_pcd = specific_structure["Interpolated structure point cloud dict"]["Full"]
                            pcd_list.append(structure_pcd)
                            
                    plotting_funcs.plot_geometries(*pcd_list)


            if show_processed_3d_datasets_renderings_plotly == True:
                for patientUID,pydicom_item in master_structure_reference_dict.items():
                    arr_list = []
                    for structs in structs_referenced_list:
                        for specific_structure_index, specific_structure in enumerate(pydicom_item[structs]):
                            if structs == bx_ref: 
                                structure_arr = specific_structure["Reconstructed structure pts arr"]
                            else: 
                                # structure_arr = specific_structure["Raw contour pts"]
                                structure_arr = specific_structure["Intra-slice interpolation information"].interpolated_pts_np_arr
                            arr_list.append(structure_arr)
                    plotting_funcs.plotly_3dscatter_arbitrary_number_of_arrays(arr_list, aspect_mode_input = 'data')

                
            #et = time.time()
            #elapsed_time = et - st
            #print('\n Execution time:', elapsed_time, 'seconds')

            """

            
            with loading_tools.Loader(num_patients,"Generating KD trees and conducting nearest neighbour searches...") as loader:
                for patientUID,pydicom_item in master_structure_reference_dict.items():
                    Bx_structs = bx_ref
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

                                if structs == bx_ref:
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
            #num_sample_pts_per_bx = num_sample_pts_per_bx_input
            #master_structure_info_dict["Global"]["MC info"]["Num sample pts per BX core"] = num_sample_pts_per_bx

            patientUID_default = "Initializing"
            processing_patient_parallel_computing_main_description = "Preparing patient for parallel processing [{}]...".format(patientUID_default)
            processing_patients_task = patients_progress.add_task("[red]"+processing_patient_parallel_computing_main_description, total = num_patients)
            processing_patient_parallel_computing_main_description_completed = "Preparing patient for parallel processing"
            processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_parallel_computing_main_description_completed, total=num_patients, visible=False)

            for patientUID,pydicom_item in master_structure_reference_dict.items():
                bx_structs = bx_ref

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
                    args_list.append((bx_sample_pts_lattice_spacing, reconstructed_delaunay_global_convex_structure_obj.delaunay_triangulation, reconstructed_biopsy_arr, patientUID, bx_structs, specific_structure_index))
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
            parallel_results_sampled_bx_points_from_global_delaunay_arr_and_bounding_box_arr = parallel_pool.starmap(MC_simulator_convex.grid_point_sampler_from_global_delaunay_convex_structure_parallel, args_list)

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

            for sampled_bx_pts_arr, axis_aligned_bounding_box_arr, num_sample_pts_in_specific_bx, structure_info_dict in parallel_results_sampled_bx_points_from_global_delaunay_arr_and_bounding_box_arr:
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
                master_structure_reference_dict[temp_patient_UID][temp_structure_type][temp_specific_structure_index]["Num sampled bx pts"] = num_sample_pts_in_specific_bx
                reconstructed_bx_pcd = master_structure_reference_dict[temp_patient_UID][temp_structure_type][temp_specific_structure_index]["Reconstructed structure point cloud"] 

                biopsies_progress.stop_task(parsing_sampled_biopsy_data_task)
                completed_progress.stop_task(parsing_sampled_biopsy_data_task_completed)
                stopwatch.stop()
                #with or without bounding box?
                #plotting_funcs.plot_geometries(sampled_bx_points_from_global_delaunay_point_cloud, reconstructed_bx_pcd, axis_aligned_bounding_box)
                #plotting_funcs.plot_geometries(sampled_bx_points_from_global_delaunay_point_cloud, reconstructed_bx_pcd)
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
                bx_structs = bx_ref

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
                    apex_to_base_bx_best_fit_unit_vec = apex_to_base_bx_best_fit_vec/np.linalg.norm(apex_to_base_bx_best_fit_vec)

                    specific_structure["Centroid line unit vec (bx needle base to bx needle tip)"] = apex_to_base_bx_best_fit_unit_vec
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
                    
                    # show translated (to biopsy coordinate system) reconstructed biopsies?
                    patients_progress.stop_task(processing_patients_task)
                    completed_progress.stop_task(processing_patients_completed_task)
                    stopwatch.stop()
                    #plotting_funcs.plot_geometries(sampled_bx_points_pcd, reconstructed_biopsy_point_cloud, axis_aligned_bounding_box, reconstructed_biopsy_bx_coord_sys_tr_point_cloud, sampled_bx_points_bx_coord_sys_tr_pcd)
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
                    if show_reconstructed_biopsy_in_biopsy_coord_sys_tr_and_rot == True:
                        # create axis aligned bounding box for the translated and rotated reconstructed biopsies
                        reconstructed_biopsy_bx_coord_sys_tr_and_rot_axis_aligned_bounding_box = reconstructed_biopsy_bx_coord_sys_tr_and_rot_point_cloud.get_axis_aligned_bounding_box()
                        reconstructed_biopsy_bx_coord_sys_tr_and_rot_axis_aligned_bounding_box_arr = np.asarray(reconstructed_biopsy_bx_coord_sys_tr_and_rot_axis_aligned_bounding_box.get_box_points())
                        reconstructed_biopsy_bx_coord_sys_tr_and_rot_axis_aligned_bounding_box.color = np.array([0,0,0], dtype=float)
                        #coord_frame = o3d.geometry.create_mesh_coordinate_frame()
                        patients_progress.stop_task(processing_patients_task)
                        completed_progress.stop_task(processing_patients_completed_task)
                        stopwatch.stop()
                        # other options...
                        #plotting_funcs.plot_geometries(sampled_bx_points_pcd, reconstructed_biopsy_point_cloud, axis_aligned_bounding_box, reconstructed_biopsy_bx_coord_sys_tr_point_cloud, sampled_bx_points_bx_coord_sys_tr_pcd, reconstructed_biopsy_bx_coord_sys_tr_and_rot_point_cloud, sampled_bx_points_bx_coord_sys_tr_and_rot_pcd)
                        #plotting_funcs.plot_geometries(sampled_bx_points_pcd, reconstructed_biopsy_point_cloud, axis_aligned_bounding_box, reconstructed_biopsy_bx_coord_sys_tr_and_rot_point_cloud, sampled_bx_points_bx_coord_sys_tr_and_rot_pcd)   
                        #plotting_funcs.plot_geometries(sampled_bx_points_pcd, reconstructed_biopsy_point_cloud, axis_aligned_bounding_box, reconstructed_biopsy_bx_coord_sys_tr_and_rot_point_cloud, sampled_bx_points_bx_coord_sys_tr_and_rot_pcd, reconstructed_biopsy_bx_coord_sys_tr_and_rot_axis_aligned_bounding_box)         
                        #plotting_funcs.plot_geometries_with_axes(sampled_bx_points_pcd, reconstructed_biopsy_point_cloud, axis_aligned_bounding_box, reconstructed_biopsy_bx_coord_sys_tr_point_cloud, sampled_bx_points_bx_coord_sys_tr_pcd, reconstructed_biopsy_bx_coord_sys_tr_and_rot_point_cloud, sampled_bx_points_bx_coord_sys_tr_and_rot_pcd)
                        #plotting_funcs.plot_geometries_with_axes(sampled_bx_points_pcd, reconstructed_biopsy_point_cloud, axis_aligned_bounding_box, reconstructed_biopsy_bx_coord_sys_tr_and_rot_point_cloud, sampled_bx_points_bx_coord_sys_tr_and_rot_pcd)
                        #plotting_funcs.plot_geometries_with_axes(sampled_bx_points_pcd, reconstructed_biopsy_point_cloud, axis_aligned_bounding_box, reconstructed_biopsy_bx_coord_sys_tr_from_arr_point_cloud, sampled_bx_points_bx_coord_sys_tr_pcd, reconstructed_biopsy_bx_coord_sys_tr_and_rot_arr_point_cloud, sampled_bx_points_bx_coord_sys_tr_and_rot_pcd)
                        #plotting_funcs.plot_geometries_with_axes(sampled_bx_points_pcd, reconstructed_biopsy_point_cloud, axis_aligned_bounding_box, reconstructed_biopsy_bx_coord_sys_tr_and_rot_arr_point_cloud, sampled_bx_points_bx_coord_sys_tr_and_rot_pcd)
                        
                        # just the reconstructed biopsy core and sampled points with its axis aligned bounding box, to show that the transformation to biopsy coordinate system was successful
                        plotting_funcs.plot_geometries(reconstructed_biopsy_bx_coord_sys_tr_and_rot_point_cloud, sampled_bx_points_bx_coord_sys_tr_and_rot_pcd, reconstructed_biopsy_bx_coord_sys_tr_and_rot_axis_aligned_bounding_box)                        
                        #plotting_funcs.plotly_3dscatter_arbitrary_number_of_arrays(arrays_to_plot_list = [reconstructed_biopsy_bx_coord_sys_tr_and_rot_arr, sampled_bx_points_bx_coord_sys_tr_and_rot_arr], colors_for_arrays_list = ['red','black'])
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
            live_display.console.print("[bold red]User input required:")
            ## begin simulation section
            """
            created_dir = False
            while created_dir == False:
                
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
            """
            
            stopwatch.stop()
            uncertainty_template_generate = ques_funcs.ask_ok('>Do you want to generate an uncertainty file template for this patient data repo?')
            stopwatch.start()
            if uncertainty_template_generate == True:
                # create a blank uncertainties file filled with the proper patient data, it is uniquely IDd by including the date and time in the file name
                stopwatch.stop()
                default_sigma = ques_funcs.ask_for_float_question('> Enter the default sigma value to generate for all structures:')
                stopwatch.start()

                date_time_now = datetime.now()
                date_time_now_file_name_format = date_time_now.strftime(" Date-%b-%d-%Y Time-%H,%M,%S")
                uncertainties_file = uncertainty_dir.joinpath(uncertainty_file_name+date_time_now_file_name_format+uncertainty_file_extension)

                uncertainty_file_writer.uncertainty_file_preper(uncertainties_file, master_structure_reference_dict, structs_referenced_list, num_general_structs, default_sigma)
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

            
            master_structure_info_dict["Global"]["MC info"]["Num MC containment simulations"] = num_MC_containment_simulations_input
            master_structure_info_dict["Global"]["MC info"]["Num MC dose simulations"] = num_MC_dose_simulations_input
            if simulation_ans ==  True:
                print('>Beginning simulation')
                master_structure_reference_dict = MC_simulator_convex.simulator_parallel(parallel_pool, 
                                                                                         live_display,
                                                                                         stopwatch, 
                                                                                         layout_groups, 
                                                                                         master_structure_reference_dict, 
                                                                                         structs_referenced_list,
                                                                                         bx_ref,
                                                                                         oar_ref,
                                                                                         dil_ref, 
                                                                                         dose_ref,
                                                                                         plan_ref, 
                                                                                         master_structure_info_dict, 
                                                                                         biopsy_z_voxel_length, 
                                                                                         num_dose_calc_NN, 
                                                                                         num_dose_NN_to_show_for_animation_plotting,
                                                                                         dose_views_jsons_paths_list,
                                                                                         containment_views_jsons_paths_list,
                                                                                         show_NN_dose_demonstration_plots,
                                                                                         show_containment_demonstration_plots,
                                                                                         biopsy_needle_compartment_length,
                                                                                         simulate_uniform_bx_shifts_due_to_bx_needle_compartment,
                                                                                         plot_uniform_shifts_to_check_plotly,
                                                                                         differential_dvh_resolution,
                                                                                         cumulative_dvh_resolution,
                                                                                         volume_DVH_percent_dose,
                                                                                         plot_translation_vectors_pointclouds,
                                                                                         spinner_type)
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


                #create global csv output folder
                csv_output_folder_name = 'Output CSVs'
                csv_output_dir = specific_output_dir.joinpath(csv_output_folder_name)
                csv_output_dir.mkdir(parents=True, exist_ok=True)
                
                # create patient specific output directories for csv files
                patient_sp_output_csv_dir_dict = {}
                for patientUID in master_structure_reference_dict.keys():
                    patient_sp_output_csv_dir = csv_output_dir.joinpath(patientUID)
                    patient_sp_output_csv_dir.mkdir(parents=True, exist_ok=True)
                    patient_sp_output_csv_dir_dict[patientUID] = patient_sp_output_csv_dir

                
                for patientUID,pydicom_item in master_structure_reference_dict.items():
                    bx_structs = bx_ref
                    patient_sp_output_csv_dir = patient_sp_output_csv_dir_dict[patientUID]
                    for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structs]):
                        num_sample_pts_per_bx = specific_bx_structure["Num sampled bx pts"]
                        bx_points_bx_coords_sys_arr = specific_bx_structure["Random uniformly sampled volume pts bx coord sys arr"]
                        bx_points_bx_coords_sys_arr_list = list(bx_points_bx_coords_sys_arr)
                        bx_points_bx_coords_sys_arr_row = bx_points_bx_coords_sys_arr_list.copy()
                        bx_points_bx_coords_sys_arr_row.insert(0,'')
                        containment_output_file_name = patientUID+','+specific_bx_structure['ROI']+',n_MC_c='+str(num_MC_containment_simulations_input)+',n_bx='+str(num_sample_pts_per_bx)+'-containment_out.csv'
                        containment_output_csv_file_path = patient_sp_output_csv_dir.joinpath(containment_output_file_name)
                        with open(containment_output_csv_file_path, 'w', newline='') as f:
                            write = csv.writer(f)
                            write.writerow(['Patient ID ->',patientUID])
                            write.writerow(['BX ID ->',specific_bx_structure['ROI']])
                            write.writerow(['BX length (from contour data) (mm)', specific_bx_structure['Reconstructed biopsy cylinder length (from contour data)']])
                            write.writerow(['Num MC containment sims ->',num_MC_containment_simulations_input])
                            write.writerow(['Num bx pt samples ->',num_sample_pts_per_bx])
                            write.writerow(['Row ->','Fixed containment structure'])
                            write.writerow(['Col ->','Fixed bx point'])
                            write.writerow(bx_points_bx_coords_sys_arr_row)
                            x_vals_row = [point_vec[0] for point_vec in bx_points_bx_coords_sys_arr_list]
                            x_vals_row.insert(0,'')
                            y_vals_row = [point_vec[1] for point_vec in bx_points_bx_coords_sys_arr_list]
                            y_vals_row.insert(0,'')
                            z_vals_row = [point_vec[2] for point_vec in bx_points_bx_coords_sys_arr_list]
                            z_vals_row.insert(0,'')
                            pt_radius_bx_coord_sys_row = [np.linalg.norm(point_vec[0:2]) for point_vec in bx_points_bx_coords_sys_arr_list]
                            pt_radius_bx_coord_sys_row.insert(0,'')
                            write.writerow(x_vals_row)
                            write.writerow(y_vals_row)
                            write.writerow(z_vals_row)
                            write.writerow(pt_radius_bx_coord_sys_row)
                            
                            for containment_structure_key_tuple, containment_structure_dict in specific_bx_structure['MC data: compiled sim results'].items():
                                containment_structure_ROI = containment_structure_key_tuple[0]
                                
                                containment_structure_successes_list = containment_structure_dict['Total successes (containment) list']
                                containment_structure_successes_with_cont_anat_ROI_row = [containment_structure_ROI + ' Total successes']+containment_structure_successes_list

                                containment_structure_binom_est_list = containment_structure_dict["Binomial estimator list"]
                                containment_structure_binom_est_with_cont_anat_ROI_row = [containment_structure_ROI + ' Mean probability']+containment_structure_binom_est_list
                                
                                containment_structure_stand_err_list = containment_structure_dict["Standard error (containment) list"]
                                containment_structure_stand_err_with_cont_anat_ROI_row = [containment_structure_ROI + ' STD']+containment_structure_stand_err_list

                                containment_structure_conf_int_list = containment_structure_dict["Confidence interval 95 (containment) list"]
                                containment_structure_conf_int_with_cont_anat_ROI_row = [containment_structure_ROI + ' 95% CI']+containment_structure_conf_int_list
                                
                                write.writerow(containment_structure_successes_with_cont_anat_ROI_row)
                                write.writerow(containment_structure_binom_est_with_cont_anat_ROI_row)
                                write.writerow(containment_structure_stand_err_with_cont_anat_ROI_row)
                                write.writerow(containment_structure_conf_int_with_cont_anat_ROI_row)

                                


                for patientUID,pydicom_item in master_structure_reference_dict.items():
                    bx_structs = bx_ref
                    patient_sp_output_csv_dir = patient_sp_output_csv_dir_dict[patientUID]
                    for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structs]):
                        num_sample_pts_per_bx = specific_bx_structure["Num sampled bx pts"]
                        containment_voxelized_output_file_name = patientUID+','+specific_bx_structure['ROI']+',n_MC_c='+str(num_MC_containment_simulations_input)+',n_bx='+str(num_sample_pts_per_bx)+'-containment_voxelized_out.csv'
                        containment_voxelized_output_csv_file_path = patient_sp_output_csv_dir.joinpath(containment_voxelized_output_file_name)
                        with open(containment_voxelized_output_csv_file_path, 'w', newline='') as f:
                            write = csv.writer(f)
                            write.writerow(['Patient ID ->',patientUID])
                            write.writerow(['BX ID ->',specific_bx_structure['ROI']])
                            write.writerow(['BX length (from contour data) (mm)', specific_bx_structure['Reconstructed biopsy cylinder length (from contour data)']])
                            write.writerow(['Num MC containment sims ->',num_MC_containment_simulations_input])
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
                
                # copy uncertainty file used for simulation to output folder 
                shutil.copy(uncertainties_file_filled, specific_output_dir)
                

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
                
                
                
                # write csv files
                for patientUID,pydicom_item in master_structure_reference_dict.items():
                    bx_structs = bx_ref
                    patient_sp_output_csv_dir = patient_sp_output_csv_dir_dict[patientUID]
                    for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structs]):
                        num_sample_pts_per_bx = specific_bx_structure["Num sampled bx pts"]
                        bx_points_bx_coords_sys_arr = specific_bx_structure["Random uniformly sampled volume pts bx coord sys arr"]
                        bx_points_bx_coords_sys_arr_list = list(bx_points_bx_coords_sys_arr)
                        
                        differential_dvh_dict = specific_bx_structure["MC data: Differential DVH dict"]
                        cumulative_dvh_dict = specific_bx_structure["MC data: Cumulative DVH dict"]

                        dvh_metric_vol_dose_percent_dict = specific_bx_structure["MC data: dose volume metrics dict"]
                        
                        dose_output_file_name = patientUID+','+specific_bx_structure['ROI']+',n_MC_d='+str(num_MC_dose_simulations_input)+',n_bx='+str(num_sample_pts_per_bx)+'-dose_out.csv'
                        dose_output_csv_file_path = patient_sp_output_csv_dir.joinpath(dose_output_file_name)
                        specific_bx_structure["Output csv file paths dict"]["Dose output point-wise csv"] = dose_output_csv_file_path
                        
                        with open(dose_output_csv_file_path, 'w', newline='') as f:
                            write = csv.writer(f)
                            write.writerow(['Patient ID ->',patientUID])
                            write.writerow(['BX ID ->',specific_bx_structure['ROI']])
                            write.writerow(['BX length (from contour data) (mm)', specific_bx_structure['Reconstructed biopsy cylinder length (from contour data)']])
                            write.writerow(['Num MC dose sims ->',num_MC_dose_simulations_input])
                            write.writerow(['Num bx pt samples ->',num_sample_pts_per_bx])
                            write.writerow(['Row ->','Fixed bx pt'])
                            write.writerow(['Col ->','Fixed MC trial'])
                            write.writerow(['Vector (mm)','X (mm)', 'Y (mm)', 'Z (mm)', 'r (mm)', 'Mean (Gy)', 'STD (Gy)', 'All MC trials doses (Gy) -->'])
                            stats_dose_val_all_MC_trials_by_bx_pt_list = specific_bx_structure["MC data: Dose statistics for each sampled bx pt list (mean, std, quantiles)"]
                            for pt_index, dose_vals_row in enumerate(specific_bx_structure['MC data: Dose vals for each sampled bx pt list']):
                                #dose_vals_row_with_point = dose_vals_row.copy()
                                pt_radius_bx_coord_sys = np.linalg.norm(bx_points_bx_coords_sys_arr_list[pt_index][0:2])
                                mean_dose_val_specific_bx_pt = stats_dose_val_all_MC_trials_by_bx_pt_list["Mean dose by bx pt"][pt_index]
                                std_dose_val_specific_bx_pt = stats_dose_val_all_MC_trials_by_bx_pt_list["STD by bx pt"][pt_index]
                                info_row_part = [bx_points_bx_coords_sys_arr_list[pt_index], bx_points_bx_coords_sys_arr_list[pt_index][0], bx_points_bx_coords_sys_arr_list[pt_index][1], bx_points_bx_coords_sys_arr_list[pt_index][2], pt_radius_bx_coord_sys, mean_dose_val_specific_bx_pt, std_dose_val_specific_bx_pt]
                                complete_dose_vals_row = info_row_part + dose_vals_row
                                write.writerow(complete_dose_vals_row)


                            for dvh_display_as_str in display_dvh_as:
                                if dvh_display_as_str == 'counts':
                                    differential_dvh_histogram_counts_by_MC_trial_arr = differential_dvh_dict["Counts arr"]
                                    cumulative_dvh_counts_by_MC_trial_arr = cumulative_dvh_dict["Counts arr"]
                                elif dvh_display_as_str == 'percent':
                                    differential_dvh_histogram_counts_by_MC_trial_arr = differential_dvh_dict["Percent arr"]
                                    cumulative_dvh_counts_by_MC_trial_arr = cumulative_dvh_dict["Percent arr"]
                                elif dvh_display_as_str == 'volume':
                                    differential_dvh_histogram_counts_by_MC_trial_arr = differential_dvh_dict["Volume arr (cubic mm)"]
                                    cumulative_dvh_counts_by_MC_trial_arr = cumulative_dvh_dict["Volume arr (cubic mm)"]
                                else:
                                    continue
                                
                                differential_dvh_dose_bin_edges_1darr = differential_dvh_dict["Dose bins (edges) arr (Gy)"][0]
                                                            
                                write.writerow(['___'])
                                write.writerow(['Differential DVH info'])
                                write.writerow(['Each row is a fixed MC trial'])
                                write.writerow(['Lower bin edge']+differential_dvh_dose_bin_edges_1darr.tolist()[0:-1])
                                write.writerow(['Upper bin edge']+differential_dvh_dose_bin_edges_1darr.tolist()[1:])
                                for mc_trial in range(num_MC_dose_simulations_input):
                                    write.writerow(['']+differential_dvh_histogram_counts_by_MC_trial_arr[mc_trial,:].tolist())


                                cumulative_dvh_dose_vals_by_MC_trial_1darr = cumulative_dvh_dict["Dose vals arr (Gy)"]
                                
                                write.writerow(['___'])
                                write.writerow(['Cumulative DVH info'])
                                write.writerow(['Each row is a fixed MC trial'])
                                write.writerow(['Dose value']+cumulative_dvh_dose_vals_by_MC_trial_1darr.tolist())
                                for mc_trial in range(num_MC_dose_simulations_input):
                                    write.writerow(['']+cumulative_dvh_counts_by_MC_trial_arr[mc_trial,:].tolist())

                            write.writerow(['___'])
                            write.writerow(['DVH metrics, percentages are relative to CTV target dose'])
                            write.writerow(['Each row is a fixed DVH metric, each column is a fixed MC trial'])
                            for vol_DVH_percent in volume_DVH_percent_dose:
                                dvh_metric_all_MC_trials = dvh_metric_vol_dose_percent_dict[str(vol_DVH_percent)]["All trials list"]
                                dvh_metric_mean = dvh_metric_vol_dose_percent_dict[str(vol_DVH_percent)]["Mean"]
                                dvh_metric_std = dvh_metric_vol_dose_percent_dict[str(vol_DVH_percent)]["STD"]
                                write.writerow(['V'+str(vol_DVH_percent)+'%']+dvh_metric_all_MC_trials)
                                write.writerow(['V'+str(vol_DVH_percent)+'% mean', dvh_metric_mean, 'V'+str(vol_DVH_percent)+'% STD', dvh_metric_std])


                for patientUID,pydicom_item in master_structure_reference_dict.items():
                    bx_structs = bx_ref
                    patient_sp_output_csv_dir = patient_sp_output_csv_dir_dict[patientUID]
                    for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structs]):
                        num_sample_pts_per_bx = specific_bx_structure["Num sampled bx pts"]
                        dose_voxelized_output_file_name = patientUID+','+specific_bx_structure['ROI']+',n_MC_d='+str(num_MC_dose_simulations_input)+',n_bx='+str(num_sample_pts_per_bx)+'-dose_voxelized_out.csv'
                        dose_voxelized_output_csv_file_path = patient_sp_output_csv_dir.joinpath(dose_voxelized_output_file_name)
                        with open(dose_voxelized_output_csv_file_path, 'w', newline='') as f:
                            write = csv.writer(f)
                            write.writerow(['Patient ID ->',patientUID])
                            write.writerow(['BX ID ->',specific_bx_structure['ROI']])
                            write.writerow(['BX length (from contour data) (mm)', specific_bx_structure['Reconstructed biopsy cylinder length (from contour data)']])
                            write.writerow(['Num MC dose sims ->',num_MC_dose_simulations_input])
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
            
            
            stopwatch.stop()
            create_dose_probability_plots_ans = ques_funcs.ask_ok('>Generate dose and containment plots?')
            stopwatch.start()

            if create_dose_probability_plots_ans ==  True:
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

                # make output figures directory
                figures_output_dir_name = 'Output figures'
                output_figures_dir = specific_output_dir.joinpath(figures_output_dir_name)
                os.mkdir(output_figures_dir)
                print('>Directory: ', output_figures_dir, ' created.')


                stopwatch.stop()
                regression_type_ans = ques_funcs.multi_choice_question('> Type of regression (LOWESS = 1, NPKR = 0)?')
                stopwatch.start()
                stopwatch.stop()
                global_regression_ans = ques_funcs.ask_ok('> Perform regression on global data?')
                stopwatch.start()

                # generate and store patient directory folders for saving
                patient_sp_output_figures_dir_dict = {}
                for patientUID in master_structure_reference_dict.keys():
                    bx_structs = bx_ref
                    patient_sp_output_figures_dir = output_figures_dir.joinpath(patientUID)
                    patient_sp_output_figures_dir.mkdir(parents=True, exist_ok=True)
                    patient_sp_output_figures_dir_dict[patientUID] = patient_sp_output_figures_dir

                
                # plot boxplots of sampled rigid shift vectors
                for patientUID,pydicom_item in master_structure_reference_dict.items():
                    production_plots.production_plot_sampled_shift_vector_box_plots_by_patient(patientUID,
                                              patient_sp_output_figures_dir_dict,
                                              structs_referenced_list,
                                              bx_structs,
                                              pydicom_item,
                                              svg_image_scale,
                                              svg_image_width,
                                              svg_image_height)


                
                # all MC trials spatial axial dose distribution with global regression 
                for patientUID,pydicom_item in master_structure_reference_dict.items():
                    production_plots.production_plot_axial_dose_distribution_all_trials_and_regression_by_patient(patient_sp_output_figures_dir_dict,
                                                                   patientUID,
                                                                   pydicom_item,
                                                                   bx_structs,
                                                                   global_regression_ans,
                                                                   regression_type_ans,
                                                                   parallel_pool,
                                                                   num_bootstraps_all_MC_data_input,
                                                                   NPKR_bandwidth,
                                                                   svg_image_scale,
                                                                   svg_image_width,
                                                                   svg_image_height,
                                                                   num_z_vals_to_evaluate_for_regression_plots
                                                                   )
                
                # 3d scatter and 2d color axial and radial dose distribution map
                for patientUID,pydicom_item in master_structure_reference_dict.items():
                    production_plots.production_3d_scatter_dose_axial_radial_distribution_by_patient(patient_sp_output_figures_dir_dict,
                                                                   patientUID,
                                                                   pydicom_item,
                                                                   bx_structs,
                                                                   parallel_pool,
                                                                   svg_image_scale,
                                                                   svg_image_width,
                                                                   svg_image_height
                                                                   )
                
                # quantile regression of axial dose distribution
                for patientUID,pydicom_item in master_structure_reference_dict.items():
                    production_plots.production_plot_axial_dose_distribution_quantile_regressions_by_patient(patient_sp_output_figures_dir_dict,
                                                                patientUID,
                                                                pydicom_item,
                                                                bx_structs,
                                                                regression_type_ans,
                                                                parallel_pool,
                                                                NPKR_bandwidth,
                                                                svg_image_scale,
                                                                svg_image_width,
                                                                svg_image_height,
                                                                num_z_vals_to_evaluate_for_regression_plots
                                                                )   



                # voxelized box and violin axial dose distribution  
                for patientUID,pydicom_item in master_structure_reference_dict.items():
                    production_plots.production_plot_voxelized_axial_dose_distribution_box_violin_by_patient(patient_sp_output_figures_dir_dict,
                                                                            patientUID,
                                                                            pydicom_item,
                                                                            bx_structs,
                                                                            svg_image_scale,
                                                                            svg_image_width,
                                                                            svg_image_height,
                                                                            )



                        


                        # create differential dvh plots
                        #differential_dvh_dict = specific_bx_structure["MC data: Differential DVH dict"]                    
                        #differential_dvh_histogram_percent_by_MC_trial_arr = differential_dvh_dict["Percent arr"]
                        #dose_output_by_MC_trial_pandas_data_frame = pandas.DataFrame.from_dict(data=dose_output_dict_by_MC_trial_for_pandas_data_frame)
                        differential_dvh_dict = specific_bx_structure["MC data: Differential DVH dict"]
                        differential_dvh_histogram_percent_by_MC_trial_arr = differential_dvh_dict["Percent arr"]
                        differential_dvh_dose_vals_by_MC_trial_1darr = differential_dvh_dict["Dose bins (edges) arr (Gy)"][0]
                        differential_dvh_dose_vals_list = differential_dvh_dose_vals_by_MC_trial_1darr.tolist()
                        differential_dvh_dose_bins_categorical_list = ['['+str(round(differential_dvh_dose_vals_list[i],1))+','+str(round(differential_dvh_dose_vals_list[i+1],1))+']' for i in range(len(differential_dvh_dose_vals_by_MC_trial_1darr)-1)]
                        differential_dvh_histogram_percent_by_MC_trial_list_of_lists = differential_dvh_histogram_percent_by_MC_trial_arr.tolist()
                        
                        percent_vals_list = []
                        dose_bins_list = differential_dvh_dose_bins_categorical_list*num_differential_dvh_plots_to_show 
                        mc_trial_index_list = []
                        for mc_trial_index in range(num_differential_dvh_plots_to_show):
                            percent_vals_list = percent_vals_list + differential_dvh_histogram_percent_by_MC_trial_list_of_lists[mc_trial_index]
                            mc_trial_index_list = mc_trial_index_list + [mc_trial_index]*len(differential_dvh_histogram_percent_by_MC_trial_list_of_lists[mc_trial_index])
                        differential_dvh_dict_for_pandas_dataframe = {"Percent volume": percent_vals_list, 
                                                                    "Dose (Gy)": dose_bins_list,
                                                                    "MC trial": mc_trial_index_list}
                        differential_dvh_pandas_dataframe = pandas.DataFrame.from_dict(differential_dvh_dict_for_pandas_dataframe)

                        fig_global = px.line(differential_dvh_pandas_dataframe, x="Dose (Gy)", y="Percent volume", color = "MC trial", width  = svg_image_width, height = svg_image_height)
                        fig_global.update_layout(
                            title = 'Differential DVH of biopsy core (' + patientUID +', '+ bx_struct_roi+'), (Displaying '+str(num_differential_dvh_plots_to_show)+' trials)',
                            hovermode = "x unified"
                        )
                        fig_global = plotting_funcs.fix_plotly_grid_lines(fig_global, y_axis = True, x_axis = True)
                        
                        svg_differential_dvh_fig_name = bx_struct_roi + ' - differential_dvh_showing_'+str(num_differential_dvh_plots_to_show)+'_trials.svg'
                        svg_differential_dvh_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_differential_dvh_fig_name)
                        fig_global.write_image(svg_differential_dvh_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

                        html_differential_dvh_fig_name = bx_struct_roi + ' - differential_dvh_showing_'+str(num_differential_dvh_plots_to_show)+'_trials.html'
                        html_differential_dvh_fig_file_path = patient_sp_output_figures_dir.joinpath(html_differential_dvh_fig_name) 
                        fig_global.write_html(html_differential_dvh_fig_file_path)



                        # create box plots of differential DVH quantile data                       
                        differential_dvh_dict = specific_bx_structure["MC data: Differential DVH dict"]
                        differential_dvh_histogram_percent_by_MC_trial_arr = differential_dvh_dict["Percent arr"]
                        differential_dvh_histogram_percent_by_dose_bin_arr = differential_dvh_histogram_percent_by_MC_trial_arr.T
                        differential_dvh_dose_vals_by_MC_trial_1darr = differential_dvh_dict["Dose bins (edges) arr (Gy)"][0]
                        differential_dvh_dose_vals_list = differential_dvh_dose_vals_by_MC_trial_1darr.tolist()
                        differential_dvh_dose_bins_list = [[round(differential_dvh_dose_vals_list[i],1),round(differential_dvh_dose_vals_list[i+1],1)] for i in range(len(differential_dvh_dose_vals_by_MC_trial_1darr)-1)]

                        percent_volume_binned_dict_for_pandas_data_frame = {str(differential_dvh_dose_bins_list[i]): differential_dvh_histogram_percent_by_dose_bin_arr[i,:] for i in range(len(differential_dvh_dose_bins_list))}
                        percent_volume_binned_dict_pandas_data_frame = pandas.DataFrame(data=percent_volume_binned_dict_for_pandas_data_frame)
                        
                        # box plot
                        fig = px.box(percent_volume_binned_dict_pandas_data_frame, points = False)
                        fig = plotting_funcs.fix_plotly_grid_lines(fig, y_axis = True, x_axis = False)
                        fig.update_layout(
                            yaxis_title='Percent volume',
                            xaxis_title='Dose (Gy)',
                            title='Differential DVH of biopsy core (' + patientUID +', '+ bx_struct_roi+')',
                            hovermode="x unified"
                        )

                        svg_dose_fig_name = bx_struct_roi + ' - differential_DVH_binned_box_plot.svg'
                        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
                        fig.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

                        html_dose_fig_name = bx_struct_roi + ' - differential_DVH_binned_box_plot.html'
                        html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
                        fig.write_html(html_dose_fig_file_path)




                        """
                        # perform non parametric kernel regression through conditional quantiles and conditional mean differntial DVH plot
                        dose_vals_to_evaluate = np.linspace(0, len(differential_dvh_dose_bins_categorical_list), num=10000)
                        quantiles_differential_dvh_dict = differential_dvh_dict["Quantiles percent dict"]
                        differential_dvh_output_dict_for_regression = {"Dose pts (Gy)": differential_dvh_dose_vals_by_MC_trial_1darr}
                        differential_dvh_output_dict_for_regression.update(quantiles_differential_dvh_dict)
                        non_parametric_kernel_regressions_dict = {}
                        data_for_non_parametric_kernel_regressions_dict = {}
                        data_keys_to_regress = ["Q95","Q5","Q50","Q75","Q25"]
                        num_bootstraps_mean_and_quantile_data = 15
                        for data_key in data_keys_to_regress:
                            data_for_non_parametric_kernel_regressions_dict[data_key]=differential_dvh_output_dict_for_regression[data_key].copy()

                        

                        for data_key, data_to_regress in data_for_non_parametric_kernel_regressions_dict.items():
                            dummy_xvals = np.linspace(0,1,num=np.shape(data_to_regress)[0])
                            if regression_type_ans == True:
                                non_parametric_regression_fit, \
                                non_parametric_regression_lower, \
                                non_parametric_regression_upper = mf.non_param_LOWESS_regression_with_confidence_bounds_bootstrap_parallel(
                                    parallel_pool,
                                    x = dummy_xvals, 
                                    y = data_to_regress, 
                                    eval_x = dose_vals_to_evaluate, 
                                    N = num_bootstraps_mean_and_quantile_data, 
                                    conf_interval = 0.95
                                )
                            elif regression_type_ans == False:
                                non_parametric_regression_fit, \
                                non_parametric_regression_lower, \
                                non_parametric_regression_upper = mf.non_param_kernel_regression_with_confidence_bounds_bootstrap_parallel(
                                    parallel_pool,
                                    x = dummy_xvals, 
                                    y = data_to_regress, 
                                    eval_x = dose_vals_to_evaluate, 
                                    N = num_bootstraps_mean_and_quantile_data, 
                                    conf_interval = 0.95, 
                                    bandwidth = NPKR_bandwidth
                                )
                            
                            non_parametric_kernel_regressions_dict[data_key] = (
                                non_parametric_regression_fit, 
                                non_parametric_regression_lower, 
                                non_parametric_regression_upper
                            )
                            
                        # create regression figure
                        fig_regressions_only_quantiles_and_mean = go.Figure()
                        for data_key,regression_tuple in non_parametric_kernel_regressions_dict.items(): 
                            fig_regressions_only_quantiles_and_mean.add_trace(
                                go.Scatter(
                                    name = data_key + ' regression',
                                    x = dose_vals_to_evaluate,
                                    y = regression_tuple[0],
                                    mode = "lines",
                                    line = dict(color = regression_colors_dict[data_key], 
                                    dash = regression_line_styles_dict[data_key]),
                                    showlegend = True
                                    )
                            )
                            fig_regressions_only_quantiles_and_mean.add_trace(
                                go.Scatter(
                                    name = data_key+': Upper 95% CI',
                                    x = dose_vals_to_evaluate,
                                    y = regression_tuple[2],
                                    mode = 'lines',
                                    marker = dict(color="#444"),
                                    line = dict(width=0),
                                    showlegend = False
                                )
                            )
                            fig_regressions_only_quantiles_and_mean.add_trace(
                                go.Scatter(
                                    name = data_key+': Lower 95% CI',
                                    x = dose_vals_to_evaluate,
                                    y = regression_tuple[1],
                                    marker = dict(color="#444"),
                                    line = dict(width=0),
                                    mode = 'lines',
                                    fillcolor = 'rgba(0, 100, 20, 0.3)',
                                    fill = 'tonexty',
                                    showlegend = False
                                )
                            )
                            
                        
                        fig_regressions_only_quantiles_and_mean.update_layout(
                            yaxis_title = 'Percent volume',
                            xaxis_title = 'Dose (Gy)',
                            title = 'Quantile regression of differential DVH (' + patientUID +', '+ bx_struct_roi+')',
                            hovermode = "x unified"
                        )
                        fig_regressions_only_quantiles_and_mean = plotting_funcs.fix_plotly_grid_lines(fig_regressions_only_quantiles_and_mean, y_axis = True, x_axis = True)

                        svg_dose_fig_name = bx_struct_roi + ' - differential_DVH_regressions_quantiles.svg'
                        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
                        fig_regressions_only_quantiles_and_mean.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

                        html_dose_fig_name = bx_struct_roi + ' - differential_DVH_regressions_quantiles.html'
                        html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
                        fig_regressions_only_quantiles_and_mean.write_html(html_dose_fig_file_path)


                        # create simplified regression figure
                        quantile_pairs_list = [("Q5","Q95", 'rgba(0, 255, 0, 0.3)'), ("Q25","Q75", 'rgba(0, 0, 255, 0.3)')] # must be organized where first element is lower and second element is upper bound
                        fig_regressions_dose_quantiles_simple = go.Figure()
                        for quantile_pair_tuple in quantile_pairs_list: 
                            lower_regression_key = quantile_pair_tuple[0]
                            upper_regression_key = quantile_pair_tuple[1]
                            fill_color = quantile_pair_tuple[2]
                            lower_regression_tuple = non_parametric_kernel_regressions_dict[lower_regression_key]
                            upper_regression_tuple = non_parametric_kernel_regressions_dict[upper_regression_key]
                            fig_regressions_dose_quantiles_simple.add_trace(
                                go.Scatter(
                                    name=upper_regression_key+' regression',
                                    x=dose_vals_to_evaluate,
                                    y=upper_regression_tuple[0],
                                    mode='lines',
                                    marker=dict(color="#444"),
                                    line=dict(color=regression_colors_dict[upper_regression_key], dash = regression_line_styles_dict[upper_regression_key]),
                                    showlegend=True
                                )
                            )
                            fig_regressions_dose_quantiles_simple.add_trace(
                                go.Scatter(
                                    name=lower_regression_key+' regression',
                                    x=dose_vals_to_evaluate,
                                    y=lower_regression_tuple[0],
                                    marker=dict(color="#444"),
                                    line=dict(color=regression_colors_dict[lower_regression_key], dash = regression_line_styles_dict[lower_regression_key]),
                                    mode='lines',
                                    fillcolor=fill_color,
                                    fill='tonexty',
                                    showlegend=True
                                )
                            )
                            
                        median_key = "Q50"
                        median_dose_regression_tuple = non_parametric_kernel_regressions_dict[median_key]
                        fig_regressions_dose_quantiles_simple.add_trace(
                            go.Scatter(
                                name=median_key+' regression',
                                x=dose_vals_to_evaluate,
                                y=median_dose_regression_tuple[0],
                                mode="lines",
                                line=dict(color=regression_colors_dict[median_key], dash = regression_line_styles_dict[median_key]),
                                showlegend=True
                            )
                        )
                        
                        fig_regressions_dose_quantiles_simple.update_layout(
                            yaxis_title = 'Percent volume',
                            xaxis_title = 'Dose (Gy)',
                            title = 'Quantile regression of differential DVH (' + patientUID +', '+ bx_struct_roi+')',
                            hovermode = "x unified"
                        )
                        fig_regressions_dose_quantiles_simple = plotting_funcs.fix_plotly_grid_lines(fig_regressions_dose_quantiles_simple, y_axis = True, x_axis = True)

                        svg_dose_fig_name = bx_struct_roi + ' - differential_DVH_regressions_quantiles_simplified.svg'
                        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
                        fig_regressions_dose_quantiles_simple.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

                        html_dose_fig_name = bx_struct_roi + ' - differential_DVH_regressions_quantiles_simplified.html'
                        html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
                        fig_regressions_dose_quantiles_simple.write_html(html_dose_fig_file_path)
                        """





                        # create cumulative DVH plots
                        cumulative_dvh_dict = specific_bx_structure["MC data: Cumulative DVH dict"]
                        cumulative_dvh_histogram_percent_by_MC_trial_arr = cumulative_dvh_dict["Percent arr"]
                        cumulative_dvh_dose_vals_by_MC_trial_1darr = cumulative_dvh_dict["Dose vals arr (Gy)"]
                        cumulative_dvh_histogram_percent_by_MC_trial_list_of_lists = cumulative_dvh_histogram_percent_by_MC_trial_arr.tolist()
                        cumulative_dvh_dose_vals_by_MC_trial_list = cumulative_dvh_dose_vals_by_MC_trial_1darr.tolist()
                        percent_vals_list = []
                        dose_vals_list = cumulative_dvh_dose_vals_by_MC_trial_list*num_cumulative_dvh_plots_to_show 
                        mc_trial_index_list = []
                        for mc_trial_index in range(num_cumulative_dvh_plots_to_show):
                            percent_vals_list = percent_vals_list + cumulative_dvh_histogram_percent_by_MC_trial_list_of_lists[mc_trial_index]
                            mc_trial_index_list = mc_trial_index_list + [mc_trial_index]*len(cumulative_dvh_histogram_percent_by_MC_trial_list_of_lists[mc_trial_index])
                        cumulative_dvh_dict_for_pandas_dataframe = {"Percent volume": percent_vals_list, 
                                                                    "Dose (Gy)": dose_vals_list,
                                                                    "MC trial": mc_trial_index_list}
                        cumulative_dvh_pandas_dataframe = pandas.DataFrame.from_dict(cumulative_dvh_dict_for_pandas_dataframe)

                        fig_global = px.line(cumulative_dvh_pandas_dataframe, x="Dose (Gy)", y="Percent volume", color = "MC trial", width  = svg_image_width, height = svg_image_height)
                        fig_global.update_layout(
                            title='Cumulative DVH of biopsy core (' + patientUID +', '+ bx_struct_roi+'), (Displaying '+str(num_cumulative_dvh_plots_to_show)+' trials)',
                            hovermode="x unified"
                        )
                        fig_global = plotting_funcs.fix_plotly_grid_lines(fig_global, y_axis = True, x_axis = True)
                        
                        svg_cumulative_dvh_fig_name = bx_struct_roi + ' - cumulative_dvh_showing_'+str(num_cumulative_dvh_plots_to_show)+'_trials.svg'
                        svg_cumulative_dvh_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_cumulative_dvh_fig_name)
                        fig_global.write_image(svg_cumulative_dvh_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

                        html_cumulative_dvh_fig_name = bx_struct_roi + ' - cumulative_dvh_showing_'+str(num_cumulative_dvh_plots_to_show)+'_trials.html'
                        html_cumulative_dvh_fig_file_path = patient_sp_output_figures_dir.joinpath(html_cumulative_dvh_fig_name) 
                        fig_global.write_html(html_cumulative_dvh_fig_file_path)



                        # perform non parametric kernel regression through conditional quantiles and conditional mean cumulative DVH plot
                        dose_vals_to_evaluate = np.linspace(min(cumulative_dvh_dose_vals_by_MC_trial_1darr), max(cumulative_dvh_dose_vals_by_MC_trial_1darr), num=10000)
                        quantiles_cumulative_dvh_dict = cumulative_dvh_dict["Quantiles percent dict"]
                        cumulative_dvh_output_dict_for_regression = {"Dose pts (Gy)": cumulative_dvh_dose_vals_by_MC_trial_1darr}
                        cumulative_dvh_output_dict_for_regression.update(quantiles_cumulative_dvh_dict)
                        non_parametric_kernel_regressions_dict = {}
                        data_for_non_parametric_kernel_regressions_dict = {}
                        data_keys_to_regress = ["Q95","Q5","Q50","Q75","Q25"]
                        num_bootstraps_mean_and_quantile_data = 15
                        for data_key in data_keys_to_regress:
                            data_for_non_parametric_kernel_regressions_dict[data_key]=cumulative_dvh_output_dict_for_regression[data_key].copy()
                            
                        for data_key, data_to_regress in data_for_non_parametric_kernel_regressions_dict.items():
                            if regression_type_ans == True:
                                non_parametric_regression_fit, \
                                non_parametric_regression_lower, \
                                non_parametric_regression_upper = mf.non_param_LOWESS_regression_with_confidence_bounds_bootstrap_parallel(
                                    parallel_pool,
                                    x = cumulative_dvh_output_dict_for_regression["Dose pts (Gy)"], 
                                    y = data_to_regress, 
                                    eval_x = dose_vals_to_evaluate, 
                                    N=num_bootstraps_mean_and_quantile_data, 
                                    conf_interval=0.95
                                )
                            elif regression_type_ans == False:
                                non_parametric_regression_fit, \
                                non_parametric_regression_lower, \
                                non_parametric_regression_upper = mf.non_param_kernel_regression_with_confidence_bounds_bootstrap_parallel(
                                    parallel_pool,
                                    x = cumulative_dvh_output_dict_for_regression["Dose pts (Gy)"], 
                                    y = data_to_regress, 
                                    eval_x = dose_vals_to_evaluate, 
                                    N=num_bootstraps_mean_and_quantile_data, 
                                    conf_interval=0.95, bandwidth = NPKR_bandwidth
                                )
                            
                            non_parametric_kernel_regressions_dict[data_key] = (
                                non_parametric_regression_fit, 
                                non_parametric_regression_lower, 
                                non_parametric_regression_upper
                            )
                            
                        # create regression figure
                        fig_regressions_only_quantiles_and_mean = go.Figure()
                        for data_key,regression_tuple in non_parametric_kernel_regressions_dict.items(): 
                            fig_regressions_only_quantiles_and_mean.add_trace(
                                go.Scatter(
                                    name=data_key+' regression',
                                    x=dose_vals_to_evaluate,
                                    y=regression_tuple[0],
                                    mode="lines",
                                    line=dict(color = regression_colors_dict[data_key], dash = regression_line_styles_dict[data_key]),
                                    showlegend=True
                                    )
                            )
                            fig_regressions_only_quantiles_and_mean.add_trace(
                                go.Scatter(
                                    name=data_key+': Upper 95% CI',
                                    x=dose_vals_to_evaluate,
                                    y=regression_tuple[2],
                                    mode='lines',
                                    marker=dict(color="#444"),
                                    line=dict(width=0),
                                    showlegend=False
                                )
                            )
                            fig_regressions_only_quantiles_and_mean.add_trace(
                                go.Scatter(
                                    name=data_key+': Lower 95% CI',
                                    x=dose_vals_to_evaluate,
                                    y=regression_tuple[1],
                                    marker=dict(color="#444"),
                                    line=dict(width=0),
                                    mode='lines',
                                    fillcolor='rgba(0, 100, 20, 0.3)',
                                    fill='tonexty',
                                    showlegend=False
                                )
                            )
                            
                        
                        fig_regressions_only_quantiles_and_mean.update_layout(
                            yaxis_title='Percent volume',
                            xaxis_title='Dose (Gy)',
                            title='Quantile regression of cumulative DVH (' + patientUID +', '+ bx_struct_roi+')',
                            hovermode="x unified"
                        )
                        fig_regressions_only_quantiles_and_mean = plotting_funcs.fix_plotly_grid_lines(fig_regressions_only_quantiles_and_mean, y_axis = True, x_axis = True)

                        svg_dose_fig_name = bx_struct_roi + ' - cumulative_DVH_regressions_quantiles.svg'
                        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
                        fig_regressions_only_quantiles_and_mean.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

                        html_dose_fig_name = bx_struct_roi + ' - cumulative_DVH_regressions_quantiles.html'
                        html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
                        fig_regressions_only_quantiles_and_mean.write_html(html_dose_fig_file_path)


                        # create simplified regression figure
                        quantile_pairs_list = [("Q5","Q95", 'rgba(0, 255, 0, 0.3)'), ("Q25","Q75", 'rgba(0, 0, 255, 0.3)')] # must be organized where first element is lower and second element is upper bound
                        fig_regressions_dose_quantiles_simple = go.Figure()
                        for quantile_pair_tuple in quantile_pairs_list: 
                            lower_regression_key = quantile_pair_tuple[0]
                            upper_regression_key = quantile_pair_tuple[1]
                            fill_color = quantile_pair_tuple[2]
                            lower_regression_tuple = non_parametric_kernel_regressions_dict[lower_regression_key]
                            upper_regression_tuple = non_parametric_kernel_regressions_dict[upper_regression_key]
                            fig_regressions_dose_quantiles_simple.add_trace(
                                go.Scatter(
                                    name=upper_regression_key+' regression',
                                    x=dose_vals_to_evaluate,
                                    y=upper_regression_tuple[0],
                                    mode='lines',
                                    marker=dict(color="#444"),
                                    line=dict(color=regression_colors_dict[upper_regression_key], dash = regression_line_styles_dict[upper_regression_key]),
                                    showlegend=True
                                )
                            )
                            fig_regressions_dose_quantiles_simple.add_trace(
                                go.Scatter(
                                    name=lower_regression_key+' regression',
                                    x=dose_vals_to_evaluate,
                                    y=lower_regression_tuple[0],
                                    marker=dict(color="#444"),
                                    line=dict(color=regression_colors_dict[lower_regression_key], dash = regression_line_styles_dict[lower_regression_key]),
                                    mode='lines',
                                    fillcolor=fill_color,
                                    fill='tonexty',
                                    showlegend=True
                                )
                            )
                            
                        median_key = "Q50"
                        median_dose_regression_tuple = non_parametric_kernel_regressions_dict[median_key]
                        fig_regressions_dose_quantiles_simple.add_trace(
                            go.Scatter(
                                name=median_key+' regression',
                                x=dose_vals_to_evaluate,
                                y=median_dose_regression_tuple[0],
                                mode="lines",
                                line=dict(color=regression_colors_dict[median_key], dash = regression_line_styles_dict[median_key]),
                                showlegend=True
                            )
                        )
                        
                        fig_regressions_dose_quantiles_simple.update_layout(
                            yaxis_title='Percent volume',
                            xaxis_title='Dose (Gy)',
                            title='Quantile regression of cumulative DVH (' + patientUID +', '+ bx_struct_roi+')',
                            hovermode="x unified"
                        )
                        fig_regressions_dose_quantiles_simple = plotting_funcs.fix_plotly_grid_lines(fig_regressions_dose_quantiles_simple, y_axis = True, x_axis = True)

                        svg_dose_fig_name = bx_struct_roi + ' - cumulative_DVH_regressions_quantiles_simplified.svg'
                        svg_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_dose_fig_name)
                        fig_regressions_dose_quantiles_simple.write_image(svg_dose_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

                        html_dose_fig_name = bx_struct_roi + ' - cumulative_DVH_regressions_quantiles_simplified.html'
                        html_dose_fig_file_path = patient_sp_output_figures_dir.joinpath(html_dose_fig_name)
                        fig_regressions_dose_quantiles_simple.write_html(html_dose_fig_file_path)






                # perform containment probabilities plots and regressions
                for patientUID,pydicom_item in master_structure_reference_dict.items():
                    bx_structs = bx_ref
                    patient_sp_output_figures_dir = patient_sp_output_figures_dir_dict[patientUID]
                    for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structs]):
                        bx_struct_roi = specific_bx_structure["ROI"]
                        bx_points_bx_coords_sys_arr = specific_bx_structure["Random uniformly sampled volume pts bx coord sys arr"]
                        bx_points_bx_coords_sys_arr_list = list(bx_points_bx_coords_sys_arr)
                        bx_points_XY_bx_coords_sys_arr_list = list(bx_points_bx_coords_sys_arr[:,0:2])
                        pt_radius_bx_coord_sys = np.linalg.norm(bx_points_XY_bx_coords_sys_arr_list, axis = 1)

                        pt_radius_point_wise_for_pd_data_frame_list = []
                        axial_Z_point_wise_for_pd_data_frame_list = []
                        binom_est_point_wise_for_pd_data_frame_list = []
                        total_successes_point_wise_for_pd_data_frame_list = []
                        std_err_point_wise_for_pd_data_frame_list = []
                        MC_trial_index_point_wise_for_pd_data_frame_list = []
                        ROI_name_point_wise_for_pd_data_frame_list = []
                             
                        for containment_structure_key_tuple, containment_structure_dict in specific_bx_structure['MC data: compiled sim results'].items():
                            containment_structure_ROI = containment_structure_key_tuple[0]
                            ROI_name_point_wise_for_pd_data_frame_list = ROI_name_point_wise_for_pd_data_frame_list + [containment_structure_ROI]*len(bx_points_bx_coords_sys_arr_list)
                            containment_structure_successes_list = containment_structure_dict['Total successes (containment) list']
                            containment_structure_binom_est_list = containment_structure_dict["Binomial estimator list"]
                            containment_structure_stand_err_list = containment_structure_dict["Standard error (containment) list"]
                            total_successes_point_wise_for_pd_data_frame_list = total_successes_point_wise_for_pd_data_frame_list + containment_structure_successes_list
                            binom_est_point_wise_for_pd_data_frame_list = binom_est_point_wise_for_pd_data_frame_list + containment_structure_binom_est_list
                            std_err_point_wise_for_pd_data_frame_list = std_err_point_wise_for_pd_data_frame_list + containment_structure_stand_err_list
                            
                            pt_radius_point_wise_for_pd_data_frame_list = pt_radius_point_wise_for_pd_data_frame_list + pt_radius_bx_coord_sys.tolist()
                            axial_Z_point_wise_for_pd_data_frame_list = axial_Z_point_wise_for_pd_data_frame_list + bx_points_bx_coords_sys_arr[:,2].tolist()     
                            
                        containment_output_dict_by_MC_trial_for_pandas_data_frame = {"Structure ROI": ROI_name_point_wise_for_pd_data_frame_list, "Radial pos (mm)": pt_radius_point_wise_for_pd_data_frame_list, "Axial pos Z (mm)": axial_Z_point_wise_for_pd_data_frame_list, "Mean probability (binom est)": binom_est_point_wise_for_pd_data_frame_list, "Total successes": total_successes_point_wise_for_pd_data_frame_list, "STD err": std_err_point_wise_for_pd_data_frame_list}
                        containment_output_by_MC_trial_pandas_data_frame = pandas.DataFrame.from_dict(data=containment_output_dict_by_MC_trial_for_pandas_data_frame)
   
                        # do non parametric kernel regression (local linear)
                        z_vals_to_evaluate = np.linspace(min(bx_points_bx_coords_sys_arr[:,2]), max(bx_points_bx_coords_sys_arr[:,2]), num=10000)
                        all_MC_trials_containment_vs_axial_Z_non_parametric_regression_fit_lower_upper_dict = {}
                        for containment_structure_key_tuple, containment_structure_dict in specific_bx_structure['MC data: compiled sim results'].items():
                            containment_structure_ROI = containment_structure_key_tuple[0]
                            if regression_type_ans == True:
                                all_MC_trials_containment_vs_axial_Z_non_parametric_regression_fit, \
                                all_MC_trials_containment_vs_axial_Z_non_parametric_regression_lower, \
                                all_MC_trials_containment_vs_axial_Z_non_parametric_regression_upper = mf.non_param_LOWESS_regression_with_confidence_bounds_bootstrap_parallel(
                                    parallel_pool,
                                    x = containment_output_by_MC_trial_pandas_data_frame.loc[containment_output_by_MC_trial_pandas_data_frame["Structure ROI"] == containment_structure_ROI, "Axial pos Z (mm)"], 
                                    y = containment_output_by_MC_trial_pandas_data_frame.loc[containment_output_by_MC_trial_pandas_data_frame["Structure ROI"] == containment_structure_ROI, "Mean probability (binom est)"], 
                                    eval_x = z_vals_to_evaluate, N=num_bootstraps_all_MC_data_input, conf_interval=0.95
                                )
                            elif regression_type_ans == False:
                                all_MC_trials_containment_vs_axial_Z_non_parametric_regression_fit, \
                                all_MC_trials_containment_vs_axial_Z_non_parametric_regression_lower, \
                                all_MC_trials_containment_vs_axial_Z_non_parametric_regression_upper = mf.non_param_kernel_regression_with_confidence_bounds_bootstrap_parallel(
                                    parallel_pool,
                                    x = containment_output_by_MC_trial_pandas_data_frame.loc[containment_output_by_MC_trial_pandas_data_frame["Structure ROI"] == containment_structure_ROI, "Axial pos Z (mm)"], 
                                    y = containment_output_by_MC_trial_pandas_data_frame.loc[containment_output_by_MC_trial_pandas_data_frame["Structure ROI"] == containment_structure_ROI, "Mean probability (binom est)"], 
                                    eval_x = z_vals_to_evaluate, N=num_bootstraps_all_MC_data_input, conf_interval=0.95, bandwidth = NPKR_bandwidth
                                )
                            containment_regressions_dict = {"Mean regression": all_MC_trials_containment_vs_axial_Z_non_parametric_regression_fit, 
                                "Lower 95 regression": all_MC_trials_containment_vs_axial_Z_non_parametric_regression_lower, 
                                "Upper 95 regression": all_MC_trials_containment_vs_axial_Z_non_parametric_regression_upper}

                            all_MC_trials_containment_vs_axial_Z_non_parametric_regression_fit_lower_upper_dict[containment_structure_key_tuple] = containment_regressions_dict
                        
                        # create 2d scatter dose plot axial (z) vs all containment probabilities from all MC trials with regressions
                        plot_type_list = ['with_errors','']
                        done_regression_only = False                        
                        for plot_type in plot_type_list:
                            # one with error bars on binom est, one without error bars
                            if plot_type == 'with_errors':
                                fig_global = px.scatter(containment_output_by_MC_trial_pandas_data_frame, x="Axial pos Z (mm)", y="Mean probability (binom est)", color = "Structure ROI", error_y = "STD err", width  = svg_image_width, height = svg_image_height)
                            if plot_type == '':
                                fig_global = px.scatter(containment_output_by_MC_trial_pandas_data_frame, x="Axial pos Z (mm)", y="Mean probability (binom est)", color = "Structure ROI", width  = svg_image_width, height = svg_image_height)
                            
                            fig_regression_only = go.Figure()
                            for containment_structure_key_tuple, containment_structure_regressions_dict in all_MC_trials_containment_vs_axial_Z_non_parametric_regression_fit_lower_upper_dict.items():
                                regression_color = 'rgb'+str(tuple(np.random.randint(low=0,high=225,size=3)))
                                containment_structure_ROI = containment_structure_key_tuple[0]
                                mean_regression = containment_structure_regressions_dict["Mean regression"]
                                lower95_regression = containment_structure_regressions_dict["Lower 95 regression"]
                                upper95_regression = containment_structure_regressions_dict["Upper 95 regression"]
                                fig_global.add_trace(
                                    go.Scatter(
                                        name=containment_structure_ROI+' regression',
                                        x=z_vals_to_evaluate,
                                        y=mean_regression,
                                        mode="lines",
                                        line=dict(color=regression_color),
                                        showlegend=True
                                        )
                                )
                                fig_global.add_trace(
                                    go.Scatter(
                                        name=containment_structure_ROI+' upper 95% CI',
                                        x=z_vals_to_evaluate,
                                        y=upper95_regression,
                                        mode='lines',
                                        marker=dict(color="#444"),
                                        line=dict(width=0),
                                        showlegend=False
                                    )
                                )
                                fig_global.add_trace(
                                    go.Scatter(
                                        name=containment_structure_ROI+' lower 95% CI',
                                        x=z_vals_to_evaluate,
                                        y=lower95_regression,
                                        marker=dict(color="#444"),
                                        line=dict(width=0),
                                        mode='lines',
                                        fillcolor='rgba(0, 100, 20, 0.3)',
                                        fill='tonexty',
                                        showlegend=False
                                    )
                                )

                                # regressions only figure
                                fig_regression_only.add_trace(
                                    go.Scatter(
                                        name=containment_structure_ROI + ' regression',
                                        x=z_vals_to_evaluate,
                                        y=mean_regression,
                                        mode="lines",
                                        line=dict(color=regression_color),
                                        showlegend=True
                                    )
                                )
                                fig_regression_only.add_trace(
                                    go.Scatter(
                                        name=containment_structure_ROI + ' upper 95% CI',
                                        x=z_vals_to_evaluate,
                                        y=upper95_regression,
                                        mode='lines',
                                        marker=dict(color="#444"),
                                        line=dict(width=0),
                                        showlegend=False
                                    )
                                )
                                fig_regression_only.add_trace(
                                    go.Scatter(
                                        name=containment_structure_ROI + ' lower 95% CI',
                                        x=z_vals_to_evaluate,
                                        y=lower95_regression,
                                        marker=dict(color="#444"),
                                        line=dict(width=0),
                                        mode='lines',
                                        fillcolor='rgba(0, 100, 20, 0.3)',
                                        fill='tonexty',
                                        showlegend=False
                                    )
                                )

                            fig_global.update_layout(
                                title='Containment probability (axial) of biopsy core (' + patientUID +', '+ bx_struct_roi+')',
                                hovermode="x unified"
                            )
                            fig_global = plotting_funcs.fix_plotly_grid_lines(fig_global, y_axis = True, x_axis = True)

                            fig_regression_only.update_layout(
                                yaxis_title='Conditional mean probability',
                                xaxis_title='Axial pos Z (mm)',
                                title='Containment probability (axial) of biopsy core (' + patientUID +', '+ bx_struct_roi+')',
                                hovermode="x unified"
                            )
                            fig_regression_only = plotting_funcs.fix_plotly_grid_lines(fig_regression_only, y_axis = True, x_axis = True)
                            
                            if plot_type == 'with_errors':
                                svg_all_MC_trials_containment_fig_name = bx_struct_roi + ' - 2d_scatter_and_regression_all_MC_trials_containment_with_errors.svg'
                            else:                       
                                svg_all_MC_trials_containment_fig_name = bx_struct_roi + ' - 2d_scatter_and_regression_all_MC_trials_containment.svg'
                            svg_all_MC_trials_containment_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_all_MC_trials_containment_fig_name)
                            fig_global.write_image(svg_all_MC_trials_containment_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

                            if plot_type == 'with_errors':
                                html_all_MC_trials_containment_fig_name = bx_struct_roi + ' - 2d_scatter_and_regression_all_MC_trials_containment_with_errors.html'
                            else:
                                html_all_MC_trials_containment_fig_name = bx_struct_roi + ' - 2d_scatter_and_regression_all_MC_trials_containment.html'
                            html_all_MC_trials_containment_fig_file_path = patient_sp_output_figures_dir.joinpath(html_all_MC_trials_containment_fig_name)
                            fig_global.write_html(html_all_MC_trials_containment_fig_file_path)
                            
                            if done_regression_only == False:
                                svg_all_MC_trials_containment_fig_name = bx_struct_roi + ' - 2d_regression_all_MC_trials_containment.svg'
                                svg_all_MC_trials_containment_fig_file_path = patient_sp_output_figures_dir.joinpath(svg_all_MC_trials_containment_fig_name)
                                fig_regression_only.write_image(svg_all_MC_trials_containment_fig_file_path, scale = svg_image_scale, width = svg_image_width, height = svg_image_height)

                                html_all_MC_trials_containment_fig_name = bx_struct_roi + ' - 2d_regression_all_MC_trials_containment.html'
                                html_all_MC_trials_containment_fig_file_path = patient_sp_output_figures_dir.joinpath(html_all_MC_trials_containment_fig_name)
                                fig_regression_only.write_html(html_all_MC_trials_containment_fig_file_path)
                            
                                done_regression_only = True
                            else: 
                                pass

                            
                       
                        
            
            
            print('>Programme has ended.')


def UID_generator(pydicom_obj):
    UID_def = f"{str(pydicom_obj[0x0010,0x0010].value)} ({str(pydicom_obj[0x0010,0x0020].value)})"
    return UID_def


def structure_referencer(structure_dcm_dict, dose_dcm_dict, plan_dcm_dict, OAR_list,DIL_list,Bx_list,st_ref_list,ds_ref,pln_ref,bx_sim_locations_list,bx_sim_ref_identifier_str,sim_bx_relative_to_list,sim_bx_relative_to_struct_type, bx_sample_pt_lattice_spacing):
    """
    A function that builds a reference library of the dicom elements passed to it so that 
    we can match the ROI name to the contour information, since the contour
    information is referenced to the name by a number.
    """
    master_st_ds_ref_dict = {}
    master_st_ds_info_dict = {}
    master_st_ds_info_global_dict = {"Global": None, "By patient": None}
    
    global_num_biopsies = 0
    global_num_OAR = 0
    global_num_DIL = 0
    global_total_num_structs = 0
    global_num_patients = 0
    for UID, structure_item_path in structure_dcm_dict.items():
        with pydicom.dcmread(structure_item_path, defer_size = '2 MB') as structure_item: 
            


            OAR_ref = [{"ROI":x.ROIName, 
                        "Ref #":x.ROINumber,
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
                        "Uncertainty data": None, 
                        "MC data: Generated normal dist random samples arr": None, 
                        "KDtree": None, 
                        "Nearest neighbours objects": [], 
                        "Plot attributes": plot_attributes()
                        } for x in structure_item.StructureSetROISequence if any(i in x.ROIName for i in OAR_list)]
            
            DIL_ref = [{"ROI":x.ROIName, 
                        "Ref #":x.ROINumber,
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
                        "Uncertainty data": None, 
                        "MC data: Generated normal dist random samples arr": None, 
                        "KDtree": None, 
                        "Nearest neighbours objects": [], 
                        "Plot attributes": plot_attributes()
                        } for x in structure_item.StructureSetROISequence if any(i in x.ROIName for i in DIL_list)] 

            
            bpsy_ref = [{"ROI":x.ROIName, 
                         "Ref #":x.ROINumber, 
                         "Reconstructed biopsy cylinder length (from contour data)": None, 
                         "Raw contour pts zslice list": None,
                         "Raw contour pts": None, 
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
                         "MC data: compiled sim results": None, 
                         "MC data: voxelized containment results dict": None, 
                         "MC data: voxelized containment results dict (dict of lists)": None, 
                         "MC data: bx to dose NN search objects list": None, 
                         "MC data: Dose NN child obj for each sampled bx pt list": None, 
                         "MC data: Dose vals for each sampled bx pt list": None,
                         "MC data: Differential DVH dict": None,
                         "MC data: Cumulative DVH dict": None,
                         "MC data: dose volume metrics dict": None, 
                         "MC data: Dose statistics for each sampled bx pt list (mean, std, quantiles)": None, 
                         "MC data: Dose statistics (MLE) for each sampled bx pt list (mean, std)": None, 
                         "MC data: voxelized dose results list": None, 
                         "MC data: voxelized dose results dict (dict of lists)": None, 
                         "Output csv file paths dict": {}, 
                         "Output data frames": {}, 
                         "KDtree": None, 
                         "Nearest neighbours objects": [], 
                         "Plot attributes": plot_attributes()
                         } for x in structure_item.StructureSetROISequence if any(i in x.ROIName for i in Bx_list)]    
            
            bpsy_ref_simulated = [{"ROI": "Bx_Tr_"+bx_sim_ref_identifier_str+" " + x.ROIName, 
                         "Ref #": bx_sim_ref_identifier_str +" "+ x.ROIName, 
                         "Relative structure name": x.ROIName,
                         "Relative structure ref #": x.ROINumber, 
                         "Reconstructed biopsy cylinder length (from contour data)": None, 
                         "Raw contour pts zslice list": None,
                         "Raw contour pts": None, 
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
                         "MC data: compiled sim results": None, 
                         "MC data: voxelized containment results dict": None, 
                         "MC data: voxelized containment results dict (dict of lists)": None, 
                         "MC data: bx to dose NN search objects list": None, 
                         "MC data: Dose NN child obj for each sampled bx pt list": None, 
                         "MC data: Dose vals for each sampled bx pt list": None,
                         "MC data: Differential DVH dict": None,
                         "MC data: Cumulative DVH dict": None,
                         "MC data: dose volume metrics dict": None,
                         "MC data: Dose statistics for each sampled bx pt list (mean, std, quantiles)": None, 
                         "MC data: Dose statistics (MLE) for each sampled bx pt list (mean, std)": None, 
                         "MC data: voxelized dose results list": None, 
                         "MC data: voxelized dose results dict (dict of lists)": None, 
                         "Output csv file paths dict": {}, 
                         "Output data frames": {}, 
                         "KDtree": None, 
                         "Nearest neighbours objects": [], 
                         "Plot attributes": plot_attributes()
                         } for x in structure_item.StructureSetROISequence if any(i in x.ROIName for i in sim_bx_relative_to_list)]
            
            bpsy_ref = bpsy_ref + bpsy_ref_simulated 
            
            
            bpsy_info = {"Num structs": len(bpsy_ref), 
                         "Num sim structs": len(bpsy_ref_simulated), 
                         "Num real structs": len(bpsy_ref) - len(bpsy_ref_simulated)}
            OAR_info = {"Num structs": len(OAR_ref)}
            DIL_info = {"Num structs": len(DIL_ref)}
            patient_total_num_structs = bpsy_info["Num structs"] + OAR_info["Num structs"] + DIL_info["Num structs"]
            all_structs_info = {"Total num structs": patient_total_num_structs}
            
            global_num_OAR = global_num_OAR + OAR_info["Num structs"]
            global_num_DIL = global_num_DIL + DIL_info["Num structs"] 
            global_num_biopsies = global_num_biopsies + bpsy_info["Num structs"]
            global_total_num_structs = global_total_num_structs + patient_total_num_structs
            global_num_patients = global_num_patients + 1

            master_st_ds_ref_dict[UID] = {"Patient ID": str(structure_item[0x0010,0x0020].value),
                                          "Patient Name": str(structure_item[0x0010,0x0010].value),
                                          st_ref_list[0]: bpsy_ref, 
                                          st_ref_list[1]:OAR_ref, 
                                          st_ref_list[2]: DIL_ref,
                                          "Ready to plot data list": None
                                          }
            
            master_st_ds_info_dict[UID] = {"Patient ID": str(structure_item[0x0010,0x0020].value),
                                           "Patient Name": str(structure_item[0x0010,0x0010].value),
                                           st_ref_list[0]: bpsy_info, 
                                           st_ref_list[1]: OAR_info, 
                                           st_ref_list[2]: DIL_info, 
                                           "All ref": all_structs_info
                                           }
    
    for UID, dose_item_path in dose_dcm_dict.items():
        with pydicom.dcmread(dose_item_path, defer_size = '2 MB') as dose_item: 
            dose_ID = UID + dose_item.StudyDate
            dose_ref_dict = {"Dose ID": dose_ID, 
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
                             "Dose phys space and pixel 3d arr": None, 
                             "Dose grid point cloud": None, 
                             "Dose grid point cloud thresholded": None, 
                             "KDtree": None
                             }
            master_st_ds_ref_dict[UID][ds_ref] = dose_ref_dict

    for UID, plan_item_path in plan_dcm_dict.items():
        with pydicom.dcmread(plan_item_path, defer_size = '2 MB') as plan_item: 
            plan_ID = UID + plan_item.StudyDate
            plan_ref_dict = {"Plan ID": plan_ID, 
                             "Study date": plan_item.StudyDate,
                             "Dose units": 'Gy', # this is by default for this dicom tag: (300A,0026)
                             "Prescription doses dict": {}
                             }
            
            for dose_ref_seq_ind in range(len(plan_item.DoseReferenceSequence)):
                plan_ref_dict["Prescription doses dict"][plan_item.DoseReferenceSequence[dose_ref_seq_ind]["DoseReferenceType"].value] = plan_item.DoseReferenceSequence[dose_ref_seq_ind]["TargetPrescriptionDose"].value
                
            master_st_ds_ref_dict[UID][pln_ref] = plan_ref_dict

    mc_info = {"Num MC containment simulations": None, 
               "Num MC dose simulations": None, 
               "Num sample pts per BX core": None, 
               "BX sample pt lattice spacing": bx_sample_pt_lattice_spacing}
    
    master_st_ds_info_global_dict["Global"] = {"Num patients": global_num_patients, 
                                               "Num structures": global_total_num_structs, 
                                               "Num biopsies": global_num_biopsies, 
                                               "Num DILs": global_num_DIL, 
                                               "Num OARs": global_num_OAR, 
                                               "MC info": mc_info}
    
    master_st_ds_info_global_dict["By patient"] = master_st_ds_info_dict
    return master_st_ds_ref_dict, master_st_ds_info_global_dict

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

    def serial_analyze(self,three_Ddata_list,interp_dist):
        self.interpolate_distance = interp_dist
        for threeDdata_zslice in three_Ddata_list:
            result = self.analyze_structure_slice(threeDdata_zslice)
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
    