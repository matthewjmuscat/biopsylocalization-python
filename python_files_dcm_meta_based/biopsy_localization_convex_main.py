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
import pickle
import fanova
import dataframe_builders


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
    """
    Consider prostate only for OARs! If the first position is the prostate, the simulated biopsies 
    will be generated relative to this structure.

    -- Also the first structure in the below list is the structure specified to plot probability of missing this structure!
    """
    oaroi_contour_names = ['Prostate']
    structure_miss_probability_roi = oaroi_contour_names[0]
    biopsy_contour_names = ['Bx']
    dil_contour_names = ['DIL']
    oar_default_sigma = 1 # default sigma in mm
    biopsy_default_sigma = 2 # default sigma in mm
    dil_default_sigma = 3 # default sigma in mm
    uncertainty_folder_name = 'Uncertainty data'
    uncertainty_file_name = "uncertainties_file_auto_generated"
    uncertainty_file_extension = ".csv"
    spinner_type = 'line'
    output_folder_name = 'Output data'
    preprocessed_data_folder_name = 'Preprocessed data'
    preprocessed_master_structure_ref_dict_for_export_name = 'master_structure_reference_dict'
    preprocessed_master_structure_info_dict_for_export_name = 'master_structure_info_dict'
    lower_bound_dose_percent = 5
    color_flattening_deg = 3 
    interp_inter_slice_dist = 0.5
    interp_intra_slice_dist = 1 # user defined length scale for intraslice interpolation min distance between points. It is used in the interpolation_information_obj class
    interp_dist_caps = 2
    biopsy_radius = 0.275
    biopsy_needle_compartment_length = 19 # length in millimeters of the biopsy needle core compartment
    
    # MC parameters
    simulate_uniform_bx_shifts_due_to_bx_needle_compartment = True
    #num_sample_pts_per_bx_input = 250 # uncommenting this line will do nothing, this line is deprecated in favour of constant cubic lattice spacing
    bx_sample_pts_lattice_spacing = 0.25
    num_MC_containment_simulations_input = 1000
    num_MC_dose_simulations_input = 1000
    biopsy_z_voxel_length = 0.5 #voxelize biopsy core every 0.5 mm along core
    num_dose_calc_NN = 4
    perform_MC_sim = True
    
    
    num_dose_NN_to_show_for_animation_plotting = 100
    num_bootstraps_for_regression_plots_input = 15
    pio.templates.default = "plotly_white"
    svg_image_scale = 1 # setting this value to something not equal to 1 produces misaligned plots with multiple traces!
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
    
    bx_sim_locations = ['centroid'] # change to empty list if dont want to create any simulated biopsies. Also the code at the moment only supports creating centroid simulated biopsies, ie. change to list containing string 'centroid'.
    bx_sim_ref_identifier = "sim"
    simulate_biopsies_relative_to = ['DIL'] # can include elements in the list such as "DIL" or "Prostate"...
    differential_dvh_resolution = 100 # the number of bins
    cumulative_dvh_resolution = 100 # the larger the number the more resolution the cDVH calculations will have
    display_dvh_as = ['counts','percent', 'volume'] # can be 'counts', 'percent', 'volume'
    num_cumulative_dvh_plots_to_show = 25
    num_differential_dvh_plots_to_show = 25
    volume_DVH_percent_dose = [100,125,150,200,300]
    volume_DVH_quantiles_to_calculate = [5,25,50,75,95]
    
    #fanova
    num_FANOVA_containment_simulations_input = 2**11 # must be a power of two for the scipy function to work, 2^10 is good
    num_FANOVA_dose_simulations_input = 2**11
    perform_fanova = False 
    perform_dose_fanova = False
    perform_containment_fanova = False
    show_fanova_containment_demonstration_plots = False
    plot_cupy_fanova_containment_distribution_results = False
    fanova_plot_uniform_shifts_to_check_plotly = False
    num_sobol_bootstraps = 100
    sobol_indices_bootstrap_conf_interval = 0.95
    show_NN_FANOVA_dose_demonstration_plots = False

    # patient sample cohort analyzer
    only_perform_patient_analyser = True
    perform_patient_sample_analyser_at_end = True
    box_plot_points_option = 'outliers'
    notch_option = False
    boxmean_option = 'sd'

    # plots to show:
    show_NN_dose_demonstration_plots = False
    show_containment_demonstration_plots = False
    show_3d_dose_renderings = False
    show_processed_3d_datasets_renderings = False
    show_processed_3d_datasets_renderings_plotly = False
    show_reconstructed_biopsy_in_biopsy_coord_sys_tr_and_rot = False
    plot_uniform_shifts_to_check_plotly = False # if this is true, will produce many plots if num_simulations is high!
    plot_translation_vectors_pointclouds = False
    plot_cupy_containment_distribution_results = False # nice because it shows all trials at once
    plot_shifted_biopsies = False

    # Final production plots to create:
    regression_type_input = 0 # LOWESS = 1 or True, NPKR = 0 or False, this concerns the type of non parametric kernel regression that is performed
    global_regression_input = False # True or False bool type, this concerns whether a regression is performed on the axial dose distribution scatter plot containing all the data points of dose from all trials for each point 

    num_z_vals_to_evaluate_for_regression_plots = 1000
    tissue_class_probability_plot_type_list = ['with_errors','']
    production_plots_input_dictionary = {"Sampled translation vector magnitudes box plots": \
                                            {"Plot bool": True, 
                                             "Plot name": " - sampling-box_plot-sampled_translations_magnitudes_all_trials",
                                             "Plot color": 'rgba(0, 92, 171, 1)'
                                             }, 
                                        "Axial dose distribution all trials and global regression": \
                                            {"Plot bool": True, 
                                             "Plot name": " - dose-scatter-all_trials_axial_dose_distribution"
                                             },
                                        "Axial and radial (3D, surface) dose distribution": \
                                            {"Plot bool": False, 
                                             "Plot name": " - dose-scatter-axial_and_radial_3D_surface_dose_distribution"
                                             },
                                        "Axial and radial (2D, color) dose distribution": \
                                            {"Plot bool": False, 
                                             "Plot name": " - dose-scatter-axial_and_radial_2D_color_dose_distribution"
                                             },
                                        "Axial dose distribution quantiles scatter plot": \
                                            {"Plot bool": False, 
                                             "Plot name": " - dose-scatter-quantiles_axial_dose_distribution"
                                             },
                                        "Axial dose distribution quantiles regression plot": \
                                            {"Plot bool": True, 
                                             "Plot name": " - dose-regression-quantiles_axial_dose_distribution"
                                             },
                                        "Axial dose distribution voxelized box plot": \
                                            {"Plot bool": True, 
                                             "Plot name": " - dose-box_plot-voxelized_axial_dose_distribution",
                                             "Plot color": 'rgba(0, 92, 171, 1)'
                                             },
                                        "Axial dose distribution voxelized violin plot": \
                                            {"Plot bool": True, 
                                             "Plot name": " - dose-violin_plot-voxelized_axial_dose_distribution",
                                             "Plot color": 'rgba(0, 92, 171, 1)'
                                             },
                                        "Differential DVH showing N trials plot": \
                                            {"Plot bool": False, 
                                             "Plot name": ' - dose-DVH-differential_dvh_showing_'+str(num_differential_dvh_plots_to_show)+'_trials'
                                             },
                                        "Differential DVH dose binned all trials box plot": \
                                            {"Plot bool": True, 
                                             "Plot name": ' - dose-DVH-differential_DVH_binned_box_plot',
                                             "Box plot color": 'rgba(0, 92, 171, 1)',
                                             "Nominal point color": 'rgba(227, 27, 35, 1)'
                                             },
                                        "Cumulative DVH showing N trials plot": \
                                            {"Plot bool": False, 
                                             "Plot name": ' - dose-DVH-cumulative_dvh_showing_'+str(num_cumulative_dvh_plots_to_show)+'_trials'
                                             },
                                        "Cumulative DVH quantile regression all trials plot regression only": \
                                            {"Plot bool": False, 
                                             "Plot name": ' - dose-DVH-cumulative_DVH_regressions_quantiles_regression_only'
                                             },
                                        "Cumulative DVH quantile regression all trials plot colorwash": \
                                            {"Plot bool": True, 
                                             "Plot name": ' - dose-DVH-cumulative_DVH_regressions_quantiles_colorwash'
                                             },
                                        "Tissue classification scatter and regression probabilities all trials plot": \
                                            {"Plot bool": True, 
                                             "Plot name": ' - tissue_class-regression-probabilities'
                                             },
                                        "Tissue classification mutual probabilities plot": \
                                            {"Plot bool": True, 
                                             "Plot name": ' - tissue_class_mutual-regression-probabilities',
                                             "Structure miss ROI": structure_miss_probability_roi
                                             },
                                        "Tissue classification Sobol indices global plot": \
                                            {"Plot bool dict": {"Global FO": True,
                                                                "Global TO": True,
                                                                "Global FO, sim vs non sim": True
                                                                }, 
                                             "Plot name dict": {"Global FO": 'FANOVA_global_tissue_class_first_order_sobol',
                                                                "Global TO": 'FANOVA_global_tissue_class_total_order_sobol',
                                                                "Global FO, sim vs non sim": 'FANOVA_global_tissue_class_first_order_sim_v_non_sim_sobol'
                                                                },
                                             "Structure miss ROI": structure_miss_probability_roi,
                                             "Box plot points option": 'all' # can be 'all', 'outliers' or False
                                             },
                                        "Dosimetry Sobol indices global plot": \
                                            {"Plot bool dict": {"Global FO": True,
                                                                "Global FO by function output": True,
                                                                "Global TO": True,
                                                                "Global TO by function output": True,
                                                                "Global FO, sim only": True,
                                                                "Global FO, sim only by function output": True,
                                                                "Global FO, non-sim only": True,
                                                                "Global FO, non-sim only by function output": True
                                                                }, 
                                             "Plot name dict": {"Global FO": 'FANOVA_global_dose_first_order_sobol',
                                                                "Global FO by function output": 'FANOVA_global_dose_first_order_sobol_by_function_output',
                                                                "Global TO": 'FANOVA_global_dose_total_order_sobol',
                                                                "Global TO by function output": 'FANOVA_global_dose_total_order_sobol_by_function_output',
                                                                "Global FO, sim only": 'FANOVA_global_dose_first_order_sim_only_sobol',
                                                                "Global FO, sim only by function output": 'FANOVA_global_dose_first_order_sim_only_by_function_output_sobol',
                                                                "Global FO, non-sim only": 'FANOVA_global_dose_first_order_non-sim_only_sobol',
                                                                "Global FO, non-sim only by function output": 'FANOVA_global_dose_first_order_non-sim_only_by_function_output_sobol'
                                                                },
                                             "Box plot points option": 'all' # can be 'all', 'outliers' or False
                                             }
                                        }
    
    

    # other parameters
    modify_generated_uncertainty_template = False # if True, the algorithm wont be able to run from start to finish without an interupt, allowing one to modify the uncertainty file
    write_containment_to_file_ans = True # If True, this generates and saves to file a csv file of the containment simulation
    write_dose_to_file_ans = True # If True, this generates and saves to file a csv file of the dose simulation
    export_pickled_preprocessed_data = False # If True, this exports a pickled version of master_structure_reference_dict and master_structure_info_dict
    skip_preprocessing = False # If True, you will be asked to specify the locations of master_structure_info_dict and master_structure_reference_dict
    
    # for dataframe builder
    cancer_tissue_label = 'DIL'
    miss_structure_complement_label = structure_miss_probability_roi + ' complement'

    # non-user changeable variables, but need to be initiatied:
    all_ref_key = "All ref"
    bx_ref = "Bx ref"
    oar_ref = "OAR ref"
    dil_ref = "DIL ref"
    structs_referenced_dict = { bx_ref: {"Contour names": biopsy_contour_names, 
                                        "Default sigma": biopsy_default_sigma
                                        }, 
                                oar_ref: {"Contour names": oaroi_contour_names,
                                          "Default sigma": oar_default_sigma
                                          }, 
                                dil_ref: {"Contour names": dil_contour_names,
                                          "Default sigma": dil_default_sigma
                                          } 
                                }
    structs_referenced_list = list(structs_referenced_dict.keys()) # note that Bx ref has to be the first entry for other parts of the code to work! In fact the ordering of all entries must be maintained. 1. BX, 2. OAR, 3. DIL
    dose_ref = "Dose ref"
    plan_ref = "Plan ref"
    num_simulated_bxs_to_create = len(bx_sim_locations)
    if len(bx_sim_locations) == 0:
        simulate_biopsies_relative_to = []
    # note below two lines are necessary since we have plot bool dict instead of plot bool for this entry, this will need to be done for any entries that have a plot bool dict 
    production_plots_input_dictionary["Tissue classification Sobol indices global plot"]["Plot bool"] = any(production_plots_input_dictionary["Tissue classification Sobol indices global plot"]["Plot bool dict"].values())
    production_plots_input_dictionary["Dosimetry Sobol indices global plot"]["Plot bool"] = any(production_plots_input_dictionary["Dosimetry Sobol indices global plot"]["Plot bool dict"].values())
    create_at_least_one_production_plot = any([x["Plot bool"] for x in production_plots_input_dictionary.values()]) # will produce True if at least one plot bool in the production_plots_input_dictionary is true, otherwise will be false if all are false 
    if simulate_uniform_bx_shifts_due_to_bx_needle_compartment == True:
        fanova_sobol_indices_names_by_index = ['X', 'Y', 'Z', 'T'] # the order is important!
    else:
        fanova_sobol_indices_names_by_index = ['X', 'Y', 'Z'] # the order is important!


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

            # Initial check and recalibration of inputs
            if num_cumulative_dvh_plots_to_show > num_MC_dose_simulations_input:
                num_cumulative_dvh_plots_to_show = num_MC_dose_simulations_input
                important_info.add_text_line("Altered number of cumulative DVH plots to show input to maximum set by number of dose simulations input, since exceeded maxmimum allowable. New value is: "+str(num_cumulative_dvh_plots_to_show), live_display)
            else:
                pass

            if num_differential_dvh_plots_to_show > num_MC_dose_simulations_input:
                num_differential_dvh_plots_to_show = num_MC_dose_simulations_input
                important_info.add_text_line("Altered number of differential DVH plots to show input to maximum set by number of dose simulations input, since exceeded maxmimum allowable. New value is: "+str(num_differential_dvh_plots_to_show), live_display)
            else:
                pass


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
            preprocessed_data_dir = data_dir.joinpath(preprocessed_data_folder_name)

            misc_tools.checkdirs(live_display, important_info, data_dir,uncertainty_dir,output_dir,input_dir, preprocessed_data_dir)




            # only perform patient sample analyzer

            if only_perform_patient_analyser == True:
                #data_frame_list = []
                #sample_dict = {}
                #num_actual_biopsies = 0
                #num_sim_biopsies = 0
                live_display.stop()
                output_csvs_folder = pathlib.Path(fd.askdirectory(title='Open output CSVs folder', initialdir=output_dir))
                #live_display.start()
                all_patient_sub_dirs = [x for x in output_csvs_folder.iterdir() if x.is_dir()]
                
                
                # cohort tissue class 
                num_actual_biopsies, num_sim_biopsies, cohort_containment_dataframe = dataframe_builders.containment_global_scores_all_patients_dataframe_builder(all_patient_sub_dirs)
                """
                for directory in all_patient_sub_dirs:
                    csv_files_in_directory_list = list(directory.glob('*.csv'))
                    containment_csvs_list = [csv_file for csv_file in csv_files_in_directory_list if "containment_out" in csv_file.name]
                    for contianment_csv in containment_csvs_list:
                        with open(contianment_csv, "r", newline='\n') as contianment_csv_open:
                            reader_obj_list = list(csv.reader(contianment_csv_open))
                            info = reader_obj_list[0:3]
                            patient_id = info[0][1]
                            bx_id = info[1][1]
                            simulated_string = info[2][1]
                            if simulated_string.lower() == 'false':
                                simulated_bool = False 
                                num_actual_biopsies = num_actual_biopsies + 1
                            else: 
                                simulated_bool = True
                                num_sim_biopsies = num_sim_biopsies + 1

                            for row_index,row in enumerate(reader_obj_list):
                                if "Global by class" in row:
                                    starting_index = row_index + 2
                                    break
                                pass
                            
                            tissue_iteration = 1
                            sample_dict["Patient ID"] = []
                            sample_dict["Bx ID"] = []
                            sample_dict["Simulated bool"] = []
                            for row_index, row in enumerate(reader_obj_list[starting_index:]):
                                if "+++" in row:
                                    tissue_iteration = tissue_iteration + 1
                                    sample_dict["Patient ID"].append(patient_id)
                                    sample_dict["Bx ID"].append(bx_id)
                                    sample_dict["Simulated bool"].append(simulated_bool)
                                    continue
                                if "---" in row:
                                    break
                                
                                if tissue_iteration == 1:
                                    if row[0] == 'Tissue type':
                                        sample_dict[row[0]] = [row[1]]
                                    else:
                                        sample_dict[row[0]] = [float(row[1])]
                                    
                                else:
                                    if row[0] == 'Tissue type':
                                        sample_dict[row[0]].append(row[1])
                                    else:
                                        sample_dict[row[0]].append(float(row[1]))

                            bx_sp_dataframe = pandas.DataFrame(data=sample_dict)
                            data_frame_list.append(bx_sp_dataframe)

                cohort_containment_dataframe = pandas.concat(data_frame_list,ignore_index = True)  
                """ 

                # Make cohort output directories
                cohort_figures_output_dir_name = 'Cohort figures'
                tissue_class_output_dir_name = 'Tissue classification'
                cohort_output_figures_dir = output_csvs_folder.parents[0].joinpath(cohort_figures_output_dir_name)
                cohort_output_figures_dir.mkdir(parents=False, exist_ok=True)
                tissue_class_cohort_output_figures_dir = cohort_output_figures_dir.joinpath(tissue_class_output_dir_name)
                tissue_class_cohort_output_figures_dir.mkdir(parents=False, exist_ok=True)
                tissue_class_general_plot_name_string = 'Patient_cohort_tissue_classification_box_plot'
                
                production_plots.production_plot_tissue_patient_cohort(cohort_containment_dataframe,
                                                                num_actual_biopsies,
                                                                num_sim_biopsies,
                                                                svg_image_scale,
                                                                svg_image_width,
                                                                svg_image_height,
                                                                tissue_class_general_plot_name_string,
                                                                tissue_class_cohort_output_figures_dir,
                                                                box_plot_points_option,
                                                                notch_option,
                                                                boxmean_option
                                                                )


                # cohort dosimetry
                num_actual_biopsies, num_sim_biopsies, cohort_dose_dataframe = dataframe_builders.dose_global_scores_all_patients_dataframe_builder(all_patient_sub_dirs)
                
                dose_output_dir_name = 'Dosimetry'
                dose_cohort_output_figures_dir = cohort_output_figures_dir.joinpath(dose_output_dir_name)
                dose_cohort_output_figures_dir.mkdir(parents=False, exist_ok=True)
                dose_general_plot_name_string = 'Patient_cohort_dose_box_plot'

                production_plots.production_plot_dose_patient_cohort(cohort_dose_dataframe,
                                    num_actual_biopsies,
                                    num_sim_biopsies,
                                    svg_image_scale,
                                    svg_image_width,
                                    svg_image_height,
                                    dose_general_plot_name_string,
                                    dose_cohort_output_figures_dir,
                                    box_plot_points_option,
                                    notch_option,
                                    boxmean_option
                                    )

                dose_distribution_plot_name_string = 'Patient_cohort_dose_distribution_histogram_plot'

                fit_parameters_sim_dict, fit_parameters_actual_dict = production_plots.production_plot_dose_distribution_patient_cohort(cohort_dose_dataframe,
                                    num_actual_biopsies,
                                    num_sim_biopsies,
                                    svg_image_scale,
                                    svg_image_width,
                                    svg_image_height,
                                    dose_distribution_plot_name_string,
                                    dose_cohort_output_figures_dir
                                    )

                dose_difference_box_plot_name_string = 'Patient_cohort_global_nominal_dose_difference_box_plot'

                production_plots.production_plot_dose_nominal_global_difference_box_patient_cohort(cohort_dose_dataframe,
                                    num_actual_biopsies,
                                    num_sim_biopsies,
                                    svg_image_scale,
                                    svg_image_width,
                                    svg_image_height,
                                    dose_difference_box_plot_name_string,
                                    dose_cohort_output_figures_dir,
                                    box_plot_points_option,
                                    notch_option,
                                    boxmean_option
                                    )

                sys.exit('>Programme exited.')



            if skip_preprocessing == False:
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
                            if bx_sim_relative_structure in structs_referenced_dict[struct_type_key]["Contour names"]:
                                if keyfound == True:
                                    raise Exception("Structure specified to simulate biopsies to found in more than one structure type.")
                                simulate_biopsies_relative_to_struct_type_list[bx_sim_relative_structure_index] = struct_type_key
                                keyfound = True
                        if keyfound == False:
                            raise Exception("Structure specified to simulate biopsies to was not found in specified structures to analyse.")
                    important_info.add_text_line("Simulating "+ ", ".join(bx_sim_locations)+" biopsies relative to "+", ".join(simulate_biopsies_relative_to)+" (Found under "+ ", ".join(simulate_biopsies_relative_to_struct_type_list)+").", live_display)          
                    live_display.refresh()
                else: 
                    simulate_biopsies_relative_to_struct_type_list = []
                    important_info.add_text_line("Not creating any simulated biopsies.", live_display)          
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
                                                                                                all_ref_key,
                                                                                                bx_sim_locations,
                                                                                                bx_sim_ref_identifier,
                                                                                                simulate_biopsies_relative_to,
                                                                                                simulate_biopsies_relative_to_struct_type_list,
                                                                                                fanova_sobol_indices_names_by_index
                                                                                                )
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
                #num_patients = master_structure_info_dict["Global"]["Num patients"]
                #num_general_structs = master_structure_info_dict["Global"]["Num structures"]


                #important_info.add_text_line("important info will appear here1", live_display)
                #rich_layout["main-right"].update(important_info_Text)
            

                patientUID_default = "Initializing"
                processing_patients_dose_task_main_description = "[red]Building dose grids [{}]...".format(patientUID_default)
                processing_patients_dose_task_completed_main_description = "[green]Building dose grids"

                processing_patients_dose_task = patients_progress.add_task(processing_patients_dose_task_main_description, total=master_structure_info_dict["Global"]["Num patients"])
                processing_patients_dose_task_completed = completed_progress.add_task(processing_patients_dose_task_completed_main_description, total=master_structure_info_dict["Global"]["Num patients"], visible=False)

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
                    
                    # saving dose grid point cloud to master reference dictionary has been shifted down since it is not pickleable
                    #dose_ref_dict["Dose grid point cloud"] = dose_point_cloud
                    
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
                    
                    # saving dose grid point cloud to master reference dictionary has been shifted down since it is not pickleable
                    #dose_ref_dict["Dose grid point cloud thresholded"] = thresholded_dose_point_cloud
                    
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
                pulling_patients_task = patients_progress.add_task(pulling_patients_task_main_description, total=master_structure_info_dict["Global"]["Num patients"])
                pulling_patients_task_completed = completed_progress.add_task(pulling_patients_task_completed_main_description, total=master_structure_info_dict["Global"]["Num patients"], visible = False) 
                        
                
                for patientUID,pydicom_item in master_structure_reference_dict.items():
                    pulling_patients_task_main_description = "[red]Pulling patient structure data [{}]...".format(patientUID)
                    patients_progress.update(pulling_patients_task, description = pulling_patients_task_main_description)

                    structureID_default = "Initializing"
                    num_general_structs_patient_specific = master_structure_info_dict["By patient"][patientUID]["All ref"]["Total num structs"]
                    pulling_structures_task_main_description = "[cyan]Pulling structures [{},{}]...".format(patientUID,structureID_default)
                    pulling_structures_task = structures_progress.add_task(pulling_structures_task_main_description, total=num_general_structs_patient_specific)
                    for structs in structs_referenced_list:
                        for specific_structure_index, specific_structure in enumerate(pydicom_item[structs]):
                            structureID = specific_structure["ROI"]
                            structure_reference_number = specific_structure["Ref #"]
                            pulling_structures_task_main_description = "[cyan]Pulling structures [{},{}]...".format(patientUID,structureID)
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
                processing_patients_task = patients_progress.add_task(processing_patients_task_main_description, total=master_structure_info_dict["Global"]["Num patients"])
                processing_patients_task_completed = completed_progress.add_task(processing_patients_task_completed_main_description, total=master_structure_info_dict["Global"]["Num patients"], visible = False)

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
                            #delaunay_global_convex_structure_obj.generate_lineset()

                            
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
                                #reconstructed_bx_delaunay_global_convex_structure_obj.generate_lineset()
                                #plot reconstructions?
                                #plotting_funcs.plot_geometries(reconstructed_biopsy_point_cloud, threeDdata_point_cloud)
                                #plotting_funcs.plot_tri_immediately_efficient(drawn_biopsy_array, reconstructed_bx_delaunay_global_convex_structure_obj.delaunay_line_set, label = specific_structure["ROI"])

                                # calculate biopsy vectors
                                vec_with_largest_z_val_index = centroid_line[:,2].argmax()
                                vec_with_largest_z_val = centroid_line[vec_with_largest_z_val_index,:]
                                base_sup_vec_bx_centroid_arr = vec_with_largest_z_val

                                vec_with_smallest_z_val_index = centroid_line[:,2].argmin()
                                vec_with_smallest_z_val = centroid_line[vec_with_smallest_z_val_index,:]
                                apex_inf_vec_bx_centroid_arr = vec_with_smallest_z_val

                                translation_vec_bx_coord_sys_origin = -apex_inf_vec_bx_centroid_arr
                                apex_to_base_bx_best_fit_vec = base_sup_vec_bx_centroid_arr - apex_inf_vec_bx_centroid_arr
                                apex_to_base_bx_best_fit_vec_length = np.linalg.norm(apex_to_base_bx_best_fit_vec)
                                apex_to_base_bx_best_fit_unit_vec = apex_to_base_bx_best_fit_vec/apex_to_base_bx_best_fit_vec_length


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
                            #master_structure_reference_dict[patientUID][structs][specific_structure_index]["Point cloud raw"] = threeDdata_point_cloud
                            #master_structure_reference_dict[patientUID][structs][specific_structure_index]["Interpolated structure point cloud dict"] = interpolated_pcd_dict
                            if structs == bx_ref:
                                master_structure_reference_dict[patientUID][structs][specific_structure_index]["Structure centroid pts"] = structure_centroids_array
                                master_structure_reference_dict[patientUID][structs][specific_structure_index]["Reconstructed biopsy cylinder length (from contour data)"] = biopsy_reconstructed_cyl_z_length_from_contour_data
                                master_structure_reference_dict[patientUID][structs][specific_structure_index]["Best fit line of centroid pts"] = centroid_line
                                master_structure_reference_dict[patientUID][structs][specific_structure_index]["Centroid line unit vec (bx needle base to bx needle tip)"] = apex_to_base_bx_best_fit_unit_vec
                                master_structure_reference_dict[patientUID][structs][specific_structure_index]["Centroid line vec (bx needle base to bx needle tip)"] = apex_to_base_bx_best_fit_vec
                                master_structure_reference_dict[patientUID][structs][specific_structure_index]["Centroid line vec length (bx needle base to bx needle tip)"] = apex_to_base_bx_best_fit_vec_length
                                master_structure_reference_dict[patientUID][structs][specific_structure_index]["Centroid line sample pts"] = centroid_line_sample
                                master_structure_reference_dict[patientUID][structs][specific_structure_index]["Reconstructed structure pts arr"] = drawn_biopsy_array
                                #master_structure_reference_dict[patientUID][structs][specific_structure_index]["Reconstructed structure point cloud"] = reconstructed_biopsy_point_cloud
                                master_structure_reference_dict[patientUID][structs][specific_structure_index]["Reconstructed structure delaunay global"] = reconstructed_bx_delaunay_global_convex_structure_obj


                            structures_progress.update(processing_structures_task, advance=1)
                    structures_progress.remove_task(processing_structures_task)
                    patients_progress.update(processing_patients_task, advance=1)
                    completed_progress.update(processing_patients_task_completed, advance=1)
                patients_progress.update(processing_patients_task, visible=False)
                completed_progress.update(processing_patients_task_completed,  visible=True)                

                
            
                ## Up until this point, the master structure reference dictionary contains only pickleable objects!

                # Now can export master structure dict to file!
                if export_pickled_preprocessed_data == True:
                    export_preprocessed_data_task_indeterminate = indeterminate_progress_main.add_task("[red]Exporting preprocessed data...", total=None)
                    export_preprocessed_data_task_indeterminate_completed = completed_progress.add_task("[green]Exporting preprocessed data", visible = False, total=master_structure_info_dict["Global"]["Num patients"])
                    date_time_now = datetime.now()
                    date_time_now_file_name_format = date_time_now.strftime(" Date-%b-%d-%Y Time-%H,%M,%S")
                    global_num_structures = master_structure_info_dict["Global"]["Num structures"]
                    specific_preprocessed_data_dir_name = str(master_structure_info_dict["Global"]["Num patients"])+' patients - '+str(global_num_structures)+' structures - '+date_time_now_file_name_format
                    specific_preprocessed_data_dir = preprocessed_data_dir.joinpath(specific_preprocessed_data_dir_name)
                    specific_preprocessed_data_dir.mkdir(parents=False, exist_ok=False)

                    preprocessed_master_structure_ref_dict_path = specific_preprocessed_data_dir.joinpath(preprocessed_master_structure_ref_dict_for_export_name)
                    with open(preprocessed_master_structure_ref_dict_path, 'wb') as master_structure_reference_dict_file:
                        pickle.dump(master_structure_reference_dict, master_structure_reference_dict_file)

                    preprocessed_master_structure_ref_info_path = specific_preprocessed_data_dir.joinpath(preprocessed_master_structure_info_dict_for_export_name)
                    with open(preprocessed_master_structure_ref_info_path, 'wb') as master_structure_info_dict_file:
                        pickle.dump(master_structure_info_dict, master_structure_info_dict_file)

                    preprocessed_info_file_name = str(master_structure_info_dict["Global"]["Num patients"])+' patients - '+str(global_num_structures)+' structures.csv'
                    preprocessed_info_file_path = specific_preprocessed_data_dir.joinpath(preprocessed_info_file_name)
                    #master_structure_info_dict["By patient"]
                    with open(preprocessed_info_file_path, 'w', newline='') as f:
                        write = csv.writer(f)
                        for patientUID,patient_info_dict in master_structure_info_dict["By patient"].items():
                            write.writerow(['Patient UID (generated)',
                                            'Patient name',
                                            'Num biopsies', 
                                            'Num OARs',
                                            'Num DILs'
                                            ])
                            write.writerow([patient_info_dict["Patient UID (generated)"],
                                            patient_info_dict["Patient Name"],
                                            patient_info_dict[structs_referenced_list[0]]["Num structs"],
                                            patient_info_dict[structs_referenced_list[1]]["Num structs"],
                                            patient_info_dict[structs_referenced_list[2]]["Num structs"]
                                            ])
                            write.writerow([' '])
                            write.writerow(["Biopsy names:"]+[x["ROI"] for x in master_structure_reference_dict[patientUID][structs_referenced_list[0]]])
                            write.writerow(["OAR names:"]+[x["ROI"] for x in master_structure_reference_dict[patientUID][structs_referenced_list[1]]])
                            write.writerow(["DIL names:"]+[x["ROI"] for x in master_structure_reference_dict[patientUID][structs_referenced_list[2]]])
                            write.writerow(['___','___','___'])


                    indeterminate_progress_main.update(export_preprocessed_data_task_indeterminate, visible = False, refresh = True)
                    completed_progress.update(export_preprocessed_data_task_indeterminate_completed, visible = True, refresh = True, advance=master_structure_info_dict["Global"]["Num patients"])
                    live_display.refresh()
                else:
                    export_preprocessed_data_task_indeterminate_skipped_completed = completed_progress.add_task("[green]Exporting preprocessed data [SKIPPED]", visible = False, total=None)
                    completed_progress.stop_task(export_preprocessed_data_task_indeterminate_skipped_completed)
                    completed_progress.update(export_preprocessed_data_task_indeterminate_skipped_completed, visible = True, refresh = True)
                    live_display.refresh()

            
            elif skip_preprocessing == True:
                live_display.stop()
                live_display.console.print("[bold red]User input required:")
                preprocessed_file_ready = False
                while preprocessed_file_ready == False:
                    stopwatch.stop()
                    preprocessed_file_ready = ques_funcs.ask_ok('> You indicated to skip data preprocessing. Would you like to select the preprocessed dataset?') 
                    stopwatch.start()
                    if preprocessed_file_ready == True:
                        print('> Please indicate the location of master_structure_reference_dict.')
                        root = tk.Tk() # these two lines are to get rid of errant tkinter window
                        root.withdraw() # these two lines are to get rid of errant tkinter window
                        # this is a user defined quantity
                        preprocessed_master_structure_reference_dict_path_str = fd.askopenfilename(title='Open the master_structure_reference_dict file', initialdir=preprocessed_data_dir)
                        with open(preprocessed_master_structure_reference_dict_path_str, "rb") as preprocessed_master_structure_reference_dict_file:
                            master_structure_reference_dict = pickle.load(preprocessed_master_structure_reference_dict_file)
                        
                        print('> Please indicate the location of master_structure_info_dict.')
                        preprocessed_master_structure_reference_dict_path = pathlib.Path(preprocessed_master_structure_reference_dict_path_str)
                        preprocessed_master_structure_reference_dict_path_parent = preprocessed_master_structure_reference_dict_path.parents[0]
                        preprocessed_master_structure_info_dict_path_str = fd.askopenfilename(title='Open the master_structure_info_dict file', initialdir=preprocessed_master_structure_reference_dict_path_parent)
                        with open(preprocessed_master_structure_info_dict_path_str, "rb") as preprocessed_master_structure_info_dict_file:
                            master_structure_info_dict = pickle.load(preprocessed_master_structure_info_dict_file)
                    else:
                        print('> Please run the algorithm without skipping preprocessing, in order to process a dataset. You may store the preprocessed dataset to use this feature.')
                        stopwatch.stop()
                        ask_to_quit = ques_funcs.ask_ok('> Would you like to quit the programme?')
                        stopwatch.start()
                        if ask_to_quit == True:
                            sys.exit("> You have quit the programme.")
                        else:
                            preprocessed_file_ready = True
                                
                live_display.start()
                important_info.add_text_line("Loaded master_structure_reference_dict from: "+ preprocessed_master_structure_reference_dict_path_str, live_display)
                important_info.add_text_line("Loaded master_structure_info_dict from: "+ preprocessed_master_structure_info_dict_path_str, live_display)

                

            # create non-pickleable objects concerning the background dose data
            patientUID_default = "Initializing"
            pickling_dose_patients_task_main_description = "[red]Pickling patient dose data [{}]...".format(patientUID_default)
            pickling_dose_patients_task_completed_main_description = "[green]Pickling patient dose data"
            pickling_dose_patients_task = patients_progress.add_task(pickling_dose_patients_task_main_description, total=master_structure_info_dict["Global"]["Num patients"])
            pickling_dose_patients_task_completed = completed_progress.add_task(pickling_dose_patients_task_completed_main_description, total=master_structure_info_dict["Global"]["Num patients"], visible = False)

            for patientUID,pydicom_item in master_structure_reference_dict.items():
                pickling_dose_patients_task_main_description = "[red]Pickling patient dose data [{}]...".format(patientUID)
                patients_progress.update(pickling_dose_patients_task, description = pickling_dose_patients_task_main_description)
                
                dose_ref_dict = pydicom_item[dose_ref]
                phys_space_dose_map_3d_arr = dose_ref_dict["Dose phys space and pixel 3d arr"]

                # create dose point cloud and thresholded dose point cloud
                dose_point_cloud = plotting_funcs.create_dose_point_cloud(phys_space_dose_map_3d_arr, color_flattening_deg, paint_dose_color = True)
                thresholded_dose_point_cloud = plotting_funcs.create_thresholded_dose_point_cloud(phys_space_dose_map_3d_arr, color_flattening_deg, paint_dose_color = True, lower_bound_percent = lower_bound_dose_percent)
                
                dose_ref_dict["Dose grid point cloud"] = dose_point_cloud
                dose_ref_dict["Dose grid point cloud thresholded"] = thresholded_dose_point_cloud

                master_structure_reference_dict[patientUID][dose_ref] = dose_ref_dict

                

                patients_progress.update(pickling_dose_patients_task, advance=1)
                completed_progress.update(pickling_dose_patients_task_completed, advance=1)
            patients_progress.update(pickling_dose_patients_task, visible=False)
            completed_progress.update(pickling_dose_patients_task_completed,  visible=True)        
            
            live_display.refresh()
            
            # plot dose point cloud cubic lattice (color only)
            if show_3d_dose_renderings == True:
                dose_point_cloud_list = []
                for patientUID,pydicom_item in master_structure_reference_dict.keys():
                    dose_point_cloud = master_structure_reference_dict[patientUID][dose_ref]["Dose grid point cloud"]
                    dose_point_cloud_list.append(dose_point_cloud)
                
                stopwatch.stop()
                plotting_funcs.plot_geometries(*dose_point_cloud_list)
                stopwatch.start()
                
                del dose_point_cloud_list
            

            # plot dose point cloud thresholded cubic lattice (color only)
            if show_3d_dose_renderings == True:
                dose_point_cloud_thresholded_list = []
                for patientUID,pydicom_item in master_structure_reference_dict.keys():
                    dose_point_cloud_thresholded = master_structure_reference_dict[patientUID][dose_ref]["Dose grid point cloud thresholded"]
                    dose_point_cloud_thresholded_list.append(dose_point_cloud_thresholded)
                
                stopwatch.stop()
                plotting_funcs.plot_geometries(*dose_point_cloud_thresholded_list)
                stopwatch.start()
                
                del dose_point_cloud_thresholded_list
            

            
            patientUID_default = "Initializing"
            pickling_structure_patients_task_main_description = "[red]Pickling patient structure data [{}]...".format(patientUID_default)
            pickling_structure_patients_task_completed_main_description = "[green]Pickling patient structure data"
            pickling_structure_patients_task = patients_progress.add_task(pickling_structure_patients_task_main_description, total=master_structure_info_dict["Global"]["Num patients"])
            pickling_structure_patients_task_completed = completed_progress.add_task(pickling_structure_patients_task_completed_main_description, total=master_structure_info_dict["Global"]["Num patients"], visible = False)

            for patientUID,pydicom_item in master_structure_reference_dict.items():
                pickling_structure_patients_task_main_description = "[red]Pickling patient structure data [{}]...".format(patientUID)
                patients_progress.update(pickling_structure_patients_task, description = pickling_structure_patients_task_main_description)
                for structs in structs_referenced_list:
                    for specific_structure_index, specific_structure in enumerate(pydicom_item[structs]):
                        # Creating pointcloud dictionary of the interpolation done
                        interslice_interpolation_information = pydicom_item[structs][specific_structure_index]["Inter-slice interpolation information"]
                        interpolation_information = pydicom_item[structs][specific_structure_index]["Intra-slice interpolation information"]
                        threeDdata_array_fully_interpolated = interpolation_information.interpolated_pts_np_arr
                        threeDdata_array_fully_interpolated_with_end_caps = interpolation_information.interpolated_pts_with_end_caps_np_arr
                        threeDdata_array_interslice_interpolation = np.vstack(interslice_interpolation_information.interpolated_pts_list)
                        pcd_struct_rand_color = np.random.uniform(0, 0.9, size=3)
                        interslice_interp_pcd = point_containment_tools.create_point_cloud(threeDdata_array_interslice_interpolation, pcd_struct_rand_color)
                        inter_and_intra_interp_pcd = point_containment_tools.create_point_cloud(threeDdata_array_fully_interpolated, pcd_struct_rand_color)
                        inter_and_intra_and_end_caps_interp_pcd = point_containment_tools.create_point_cloud(threeDdata_array_fully_interpolated_with_end_caps, pcd_struct_rand_color)
                        interpolated_pcd_dict = {"Interslice": interslice_interp_pcd, "Full": inter_and_intra_interp_pcd, "Full with end caps": inter_and_intra_and_end_caps_interp_pcd}
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Interpolated structure point cloud dict"] = interpolated_pcd_dict

                        # creating pointcloud of the raw contour points
                        threeDdata_array = pydicom_item[structs][specific_structure_index]["Raw contour pts"]
                        threeDdata_pcd_color = np.random.uniform(0, 0.7, size=3)
                        threeDdata_point_cloud = point_containment_tools.create_point_cloud(threeDdata_array, threeDdata_pcd_color)
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Point cloud raw"] = threeDdata_point_cloud

                        # creating the lineset of the delaunay global convex structure
                        delaunay_global_convex_structure_obj = pydicom_item[structs][specific_structure_index]["Delaunay triangulation global structure"]
                        delaunay_global_convex_structure_obj.generate_lineset()
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Delaunay triangulation global structure"] = delaunay_global_convex_structure_obj

                        # creating lineset of the zslice wise delaunay convex structure
                        delaunay_triangulation_obj_zslicewise_list = pydicom_item[structs][specific_structure_index]["Delaunay triangulation zslice-wise list"]
                        for delaunay_obj in delaunay_triangulation_obj_zslicewise_list:
                            delaunay_obj.generate_lineset()
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Delaunay triangulation zslice-wise list"] = delaunay_triangulation_obj_zslicewise_list

                        # For biopsies only
                        if structs == bx_ref:
                            # creating pointcloud of the reconstructed biopsy
                            drawn_biopsy_array = pydicom_item[structs][specific_structure_index]["Reconstructed structure pts arr"] 
                            reconstructed_bx_pcd_color = np.random.uniform(0, 0.7, size=3)
                            reconstructed_biopsy_point_cloud = point_containment_tools.create_point_cloud(drawn_biopsy_array, reconstructed_bx_pcd_color)
                            master_structure_reference_dict[patientUID][structs][specific_structure_index]["Reconstructed structure point cloud"] = reconstructed_biopsy_point_cloud

                            # creating lineset of the reconstructed biopsy global delaunay object
                            reconstructed_bx_delaunay_global_convex_structure_obj = pydicom_item[structs][specific_structure_index]["Reconstructed structure delaunay global"]
                            reconstructed_bx_delaunay_global_convex_structure_obj.generate_lineset()
                            master_structure_reference_dict[patientUID][structs][specific_structure_index]["Reconstructed structure delaunay global"] = reconstructed_bx_delaunay_global_convex_structure_obj

                patients_progress.update(pickling_structure_patients_task, advance=1)
                completed_progress.update(pickling_structure_patients_task_completed, advance=1)
            patients_progress.update(pickling_structure_patients_task, visible=False)
            completed_progress.update(pickling_structure_patients_task_completed,  visible=True)  

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

                
            


            ## uniformly sample points from biopsies
            #st = time.time()
            args_list = []
            master_structure_info_dict["Global"]["MC info"]["BX sample pt lattice spacing"] = bx_sample_pts_lattice_spacing

            patientUID_default = "Initializing"
            processing_patient_parallel_computing_main_description = "Preparing patient for parallel processing [{}]...".format(patientUID_default)
            processing_patients_task = patients_progress.add_task("[red]"+processing_patient_parallel_computing_main_description, total = master_structure_info_dict["Global"]["Num patients"])
            processing_patient_parallel_computing_main_description_completed = "Preparing patient for parallel processing"
            processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_parallel_computing_main_description_completed, total=master_structure_info_dict["Global"]["Num patients"], visible=False)

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
                    
                    z_axis_np_vec = np.array([0,0,1],dtype=float)
                    apex_to_base_bx_best_fit_vec = specific_structure["Centroid line vec (bx needle base to bx needle tip)"]
                    z_axis_to_centroid_vec_rotation_matrix = mf.rotation_matrix_from_vectors(z_axis_np_vec,apex_to_base_bx_best_fit_vec)
                    
                    args_list.append((bx_sample_pts_lattice_spacing, reconstructed_delaunay_global_convex_structure_obj.delaunay_triangulation, reconstructed_biopsy_arr, patientUID, bx_structs, specific_structure_index,z_axis_to_centroid_vec_rotation_matrix))
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
            sampling_points_task_indeterminate_completed = completed_progress.add_task("[green]Sampling points from all patient biopsies (parallel)", visible = False, total = master_structure_info_dict["Global"]["Num patients"])
            #parallel_results_sampled_bx_points_from_global_delaunay_arr_and_bounding_box_arr = parallel_pool.starmap(MC_simulator_convex.grid_point_sampler_from_global_delaunay_convex_structure_parallel, args_list)
            parallel_results_sampled_bx_points_from_global_delaunay_arr_and_bounding_box_arr = parallel_pool.starmap(MC_simulator_convex.grid_point_sampler_rotated_from_global_delaunay_convex_structure_parallel, args_list)

            indeterminate_progress_main.update(sampling_points_task_indeterminate, visible = False, refresh = True)
            completed_progress.update(sampling_points_task_indeterminate_completed, advance = master_structure_info_dict["Global"]["Num patients"], visible = True, refresh = True)
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
            processing_patients_task = patients_progress.add_task("[red]"+processing_patient_rotating_bx_main_description, total = master_structure_info_dict["Global"]["Num patients"])
            processing_patient_rotating_bx_main_description_completed = "Creating biopsy oriented coordinate system"
            processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_rotating_bx_main_description_completed, total=master_structure_info_dict["Global"]["Num patients"], visible=False)

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
                    apex_to_base_bx_best_fit_vec = specific_structure["Centroid line vec (bx needle base to bx needle tip)"]
                    
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
                        plotting_funcs.plot_geometries(sampled_bx_points_bx_coord_sys_tr_and_rot_pcd, reconstructed_biopsy_bx_coord_sys_tr_and_rot_axis_aligned_bounding_box)   
                        plotting_funcs.plotly_3dscatter_arbitrary_number_of_arrays(arrays_to_plot_list = [reconstructed_biopsy_bx_coord_sys_tr_and_rot_arr, sampled_bx_points_bx_coord_sys_tr_and_rot_arr], colors_for_arrays_list = ['red','black'])
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



            #live_display.stop()
            #live_display.console.print("[bold red]User input required:")
            ## begin simulation section
            
            # first question
            #stopwatch.stop()
            #uncertainty_template_generate = ques_funcs.ask_ok('>Do you want to generate an uncertainty file template for this patient data repo?')
            #stopwatch.start()
            
            # create a blank uncertainties file filled with the proper patient data, it is uniquely IDd by including the date and time in the file name
            #stopwatch.stop()
            #default_sigma = ques_funcs.ask_for_float_question('> Enter the default sigma value to generate for all structures:')
            #stopwatch.start()


            # generate uncertainty file
            date_time_now = datetime.now()
            date_time_now_file_name_format = date_time_now.strftime(" Date-%b-%d-%Y Time-%H,%M,%S")
            uncertainties_file = uncertainty_dir.joinpath(uncertainty_file_name+date_time_now_file_name_format+uncertainty_file_extension)
            
            num_general_structs = master_structure_info_dict["Global"]["Num structures"]

            uncertainty_file_writer.uncertainty_file_preper_sigma_by_struct_type(uncertainties_file, 
                                                                                 master_structure_reference_dict, 
                                                                                 structs_referenced_list, 
                                                                                 num_general_structs, 
                                                                                 structs_referenced_dict
                                                                                 )
            
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
            else:
                # this is run if the uncertainty file is not to be modified by the user before running the simulation
                uncertainties_file_filled = uncertainties_file
                pandas_read_uncertainties = pandas.read_csv(uncertainties_file_filled, names = [0, 1, 2, 3, 4, 5])  
                
            

           


            # Transfer read uncertainty data to master_reference
            num_general_structs_from_uncertainty_file = int(pandas_read_uncertainties.values[1][0])
            for specific_structure_index in range(num_general_structs_from_uncertainty_file):
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

            

            
            master_structure_info_dict["Global"]["MC info"]["Num MC containment simulations"] = num_MC_containment_simulations_input
            master_structure_info_dict["Global"]["MC info"]["Num MC dose simulations"] = num_MC_dose_simulations_input
           
            #live_display.stop()
            # Run MC simulation
            if perform_MC_sim == True:
                master_structure_reference_dict, live_display = MC_simulator_convex.simulator_parallel(parallel_pool, 
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
                                                                                        volume_DVH_quantiles_to_calculate,
                                                                                        plot_translation_vectors_pointclouds,
                                                                                        plot_cupy_containment_distribution_results,
                                                                                        plot_shifted_biopsies,
                                                                                        structure_miss_probability_roi,
                                                                                        spinner_type
                                                                                        )
            
            if perform_fanova == True:
                fanova.fanova_analysis(
                    parallel_pool, 
                    live_display,
                    stopwatch, 
                    layout_groups, 
                    master_structure_reference_dict, 
                    master_structure_info_dict,
                    structs_referenced_list,
                    bx_ref,
                    dose_ref,
                    biopsy_needle_compartment_length,
                    simulate_uniform_bx_shifts_due_to_bx_needle_compartment,
                    num_FANOVA_containment_simulations_input,
                    num_FANOVA_dose_simulations_input,
                    fanova_plot_uniform_shifts_to_check_plotly,
                    show_fanova_containment_demonstration_plots,
                    plot_cupy_fanova_containment_distribution_results,
                    num_sobol_bootstraps,
                    sobol_indices_bootstrap_conf_interval,
                    show_NN_FANOVA_dose_demonstration_plots,
                    num_dose_calc_NN,
                    dose_views_jsons_paths_list,
                    perform_dose_fanova,
                    perform_containment_fanova
                    )

            live_display.start(refresh=True)
            #live_display.stop()

            # Create the specific output directory folder
            date_time_now = datetime.now()
            date_time_now_file_name_format = date_time_now.strftime(" Date-%b-%d-%Y Time-%H,%M,%S")
            specific_output_dir_name = 'MC_sim_out-'+date_time_now_file_name_format
            specific_output_dir = output_dir.joinpath(specific_output_dir_name)
            specific_output_dir.mkdir(parents=False, exist_ok=True)

            # copy uncertainty file used for simulation to output folder 
            shutil.copy(uncertainties_file_filled, specific_output_dir)

            # If writing containment or dose csvs to file or producing at least one production plot, create the directory
            if any([write_containment_to_file_ans, write_dose_to_file_ans, create_at_least_one_production_plot]):
                if any([write_containment_to_file_ans, write_dose_to_file_ans]):
                    # create global csv output folder
                    csv_output_folder_name = 'Output CSVs'
                    csv_output_dir = specific_output_dir.joinpath(csv_output_folder_name)
                    csv_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # create patient specific output directories for csv files
                    patient_sp_output_csv_dir_dict = {}
                    for patientUID in master_structure_reference_dict.keys():
                        patient_sp_output_csv_dir = csv_output_dir.joinpath(patientUID)
                        patient_sp_output_csv_dir.mkdir(parents=True, exist_ok=True)
                        patient_sp_output_csv_dir_dict[patientUID] = patient_sp_output_csv_dir

                if create_at_least_one_production_plot == True:
                    # make output figures directory
                    figures_output_dir_name = 'Output figures'
                    output_figures_dir = specific_output_dir.joinpath(figures_output_dir_name)
                    output_figures_dir.mkdir(parents=True, exist_ok=True)

                    # generate and store patient directory folders for saving
                    patient_sp_output_figures_dir_dict = {}
                    for patientUID in master_structure_reference_dict.keys():
                        patient_sp_output_figures_dir = output_figures_dir.joinpath(patientUID)
                        patient_sp_output_figures_dir.mkdir(parents=True, exist_ok=True)
                        patient_sp_output_figures_dir_dict[patientUID] = patient_sp_output_figures_dir

                    # create a global folder
                    patient_sp_output_figures_dir = output_figures_dir.joinpath('Global')
                    patient_sp_output_figures_dir.mkdir(parents=True, exist_ok=True)
                    patient_sp_output_figures_dir_dict["Global"] = patient_sp_output_figures_dir
                
            if perform_MC_sim == True:
                # Build dataframes
                for patientUID,pydicom_item in master_structure_reference_dict.items():

                    for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structs]):

                        containment_output_dict_by_MC_trial_for_pandas_data_frame, containment_output_by_MC_trial_pandas_data_frame = dataframe_builders.tissue_probability_dataframe_builder_by_bx_pt(specific_bx_structure, 
                                                                                                                                                                                                        structure_miss_probability_roi,
                                                                                                                                                                                                        cancer_tissue_label,
                                                                                                                                                                                                        miss_structure_complement_label
                                                                                                                                                                                                        )
                        
                        specific_bx_structure["Output data frames"]["Mutual containment ouput by bx point"] = containment_output_by_MC_trial_pandas_data_frame
                        specific_bx_structure["Output dicts for data frames"]["Mutual containment ouput by bx point"] = containment_output_dict_by_MC_trial_for_pandas_data_frame




            if write_containment_to_file_ans ==  True and perform_MC_sim == True:
                important_info.add_text_line("Writing containment CSVs to file.", live_display)
                
                patientUID_default = "Initializing"
                processing_patient_csv_writing_description = "Writing containment CSVs to file [{}]...".format(patientUID_default)
                processing_patients_task = patients_progress.add_task("[red]"+processing_patient_csv_writing_description, total = master_structure_info_dict["Global"]["Num patients"])
                processing_patient_csv_writing_description_completed = "Writing containment CSVs to file"
                processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_csv_writing_description_completed, total=master_structure_info_dict["Global"]["Num patients"], visible=False)
                
                for patientUID,pydicom_item in master_structure_reference_dict.items():

                    processing_patient_csv_writing_description = "Writing containment CSVs to file [{}]...".format(patientUID)
                    patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_csv_writing_description)

                    bx_structs = bx_ref
                    patient_sp_output_csv_dir = patient_sp_output_csv_dir_dict[patientUID]
                    for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structs]):
                        simulated_bool = specific_bx_structure["Simulated bool"]
                        num_sample_pts_per_bx = specific_bx_structure["Num sampled bx pts"]
                        bx_points_bx_coords_sys_arr = specific_bx_structure["Random uniformly sampled volume pts bx coord sys arr"]
                        bx_points_bx_coords_sys_arr_list = list(bx_points_bx_coords_sys_arr)
                        bx_points_bx_coords_sys_arr_row = bx_points_bx_coords_sys_arr_list.copy()
                        bx_points_bx_coords_sys_arr_row.insert(0,'Sampled point vector (Bx coord sys) (mm)')
                        containment_output_file_name = patientUID+','+specific_bx_structure['ROI']+',n_MC_c='+str(num_MC_containment_simulations_input)+',n_bx='+str(num_sample_pts_per_bx)+'-containment_out.csv'
                        containment_output_csv_file_path = patient_sp_output_csv_dir.joinpath(containment_output_file_name)
                        with open(containment_output_csv_file_path, 'w', newline='') as f:
                            write = csv.writer(f)
                            write.writerow(['Patient ID',patientUID])
                            write.writerow(['BX ID',specific_bx_structure['ROI']])
                            write.writerow(['Simulated', simulated_bool])
                            write.writerow(['BX length (from contour data) (mm)', specific_bx_structure['Reconstructed biopsy cylinder length (from contour data)']])
                            write.writerow(['Num MC containment sims',num_MC_containment_simulations_input])
                            write.writerow(['Num bx pt samples',num_sample_pts_per_bx])
                            
                            # global tissue class
                            write.writerow(['---'])
                            write.writerow(['Global by class'])
                            write.writerow(['---'])
                            rows_to_write_list = []

                            containment_output_by_MC_trial_pandas_data_frame = specific_bx_structure["Output data frames"]["Mutual containment ouput by bx point"]
                            tissue_classes_list = [cancer_tissue_label,structure_miss_probability_roi,miss_structure_complement_label]
                            for tissue_class in tissue_classes_list:
                                tissue_class_row = ['Tissue type', tissue_class]

                                mean_prob = containment_output_by_MC_trial_pandas_data_frame[containment_output_by_MC_trial_pandas_data_frame["Structure ROI"] == tissue_class]["Mean probability (binom est)"].mean()
                                mean_prob_row = ['Mean probability', mean_prob]

                                mean_std = containment_output_by_MC_trial_pandas_data_frame[containment_output_by_MC_trial_pandas_data_frame["Structure ROI"] == tissue_class]["Mean probability (binom est)"].std()
                                mean_std_row = ['STD', mean_std]

                                mean_stderr = containment_output_by_MC_trial_pandas_data_frame[containment_output_by_MC_trial_pandas_data_frame["Structure ROI"] == tissue_class]["STD err"].mean()
                                std_err_row = ['STD err', mean_stderr]
                            
                                tissue_class_CI_tuple = mf.normal_CI_estimator(mean_prob, mean_stderr)
                                tissue_class_CI_lower_row = ['CI lower', tissue_class_CI_tuple[0]]
                                tissue_class_CI_upper_row = ['CI upper', tissue_class_CI_tuple[1]]

                                rows_to_write_list.append(tissue_class_row)
                                rows_to_write_list.append(mean_prob_row)
                                rows_to_write_list.append(mean_std_row)
                                rows_to_write_list.append(std_err_row)
                                rows_to_write_list.append(tissue_class_CI_lower_row)
                                rows_to_write_list.append(tissue_class_CI_upper_row)
                                rows_to_write_list.append(['+++'])

                            for row_to_write in rows_to_write_list:
                                write.writerow(row_to_write)
                            
                            del rows_to_write_list

                            # global
                            write.writerow(['---'])
                            write.writerow(['Global by structure'])
                            write.writerow(['---'])
                            rows_to_write_list = []
                            for containment_structure_key_tuple, containment_structure_dict in specific_bx_structure['MC data: compiled sim results'].items():
                                containment_structure_ROI = containment_structure_key_tuple[0]
                                containment_structure_nominal_list = containment_structure_dict['Nominal containment list']
                                containment_structure_nominal_num_contained = sum(containment_structure_nominal_list)
                                containment_structure_nominal_percent_contained = (containment_structure_nominal_num_contained/len(containment_structure_nominal_list))*100
                                containment_structure_nominal_percent_contained_with_cont_anat_ROI_row = [containment_structure_ROI + ' Nominal percent volume contained',containment_structure_nominal_percent_contained]
                                rows_to_write_list.append(containment_structure_nominal_percent_contained_with_cont_anat_ROI_row)
                                

                                containment_structure_binom_est_arr = np.array(containment_structure_dict["Binomial estimator list"])
                                containment_structure_binom_est_global_mean = np.mean(containment_structure_binom_est_arr)
                                containment_structure_binom_est_std = np.std(containment_structure_binom_est_arr)
                                containment_structure_binom_est_std_err = containment_structure_binom_est_std / np.sqrt(np.size(containment_structure_binom_est_arr))
                                containment_structure_binom_est_CI = mf.normal_CI_estimator(containment_structure_binom_est_global_mean, containment_structure_binom_est_std_err)
                                containment_structure_binom_est_global_mean_with_cont_anat_ROI_row = [containment_structure_ROI + ' Mean probability', containment_structure_binom_est_global_mean]
                                containment_structure_binom_est_global_std_with_cont_anat_ROI_row = [containment_structure_ROI + ' STD', containment_structure_binom_est_std]
                                containment_structure_binom_est_global_std_err_with_cont_anat_ROI_row = [containment_structure_ROI + ' STD err in mean', containment_structure_binom_est_std_err]
                                containment_structure_binom_est_global_mean_CI_with_cont_anat_ROI_row = [containment_structure_ROI + ' Mean CI', containment_structure_binom_est_CI]
                                
                                rows_to_write_list.append(containment_structure_binom_est_global_mean_with_cont_anat_ROI_row)
                                rows_to_write_list.append(containment_structure_binom_est_global_std_with_cont_anat_ROI_row)
                                rows_to_write_list.append(containment_structure_binom_est_global_std_err_with_cont_anat_ROI_row)
                                rows_to_write_list.append(containment_structure_binom_est_global_mean_CI_with_cont_anat_ROI_row)
                                

                            for row_to_write in rows_to_write_list:
                                write.writerow(row_to_write)
                            
                            del rows_to_write_list

                            
                            # Point wise
                            write.writerow(['---'])
                            write.writerow(['Point-wise'])
                            write.writerow(['---'])
                            
                            write.writerow(['Row ->','Fixed containment structure'])
                            write.writerow(['Col ->','Fixed bx point'])
                            write.writerow(bx_points_bx_coords_sys_arr_row)
                            x_vals_row = [point_vec[0] for point_vec in bx_points_bx_coords_sys_arr_list]
                            x_vals_row.insert(0,'X coord (mm)')
                            y_vals_row = [point_vec[1] for point_vec in bx_points_bx_coords_sys_arr_list]
                            y_vals_row.insert(0,'Y coord (mm)')
                            z_vals_row = [point_vec[2] for point_vec in bx_points_bx_coords_sys_arr_list]
                            z_vals_row.insert(0,'Z coord (mm)')
                            pt_radius_bx_coord_sys_row = [np.linalg.norm(point_vec[0:2]) for point_vec in bx_points_bx_coords_sys_arr_list]
                            pt_radius_bx_coord_sys_row.insert(0,'Cyl coord radius (mm)')
                            write.writerow(x_vals_row)
                            write.writerow(y_vals_row)
                            write.writerow(z_vals_row)
                            write.writerow(pt_radius_bx_coord_sys_row)
                            
                            rows_to_write_list = []
                            for containment_structure_key_tuple, containment_structure_dict in specific_bx_structure['MC data: compiled sim results'].items():
                                containment_structure_ROI = containment_structure_key_tuple[0]

                                containment_structure_nominal_list = containment_structure_dict['Nominal containment list']
                                containment_structure_nominal_with_cont_anat_ROI_row = [containment_structure_ROI + ' Nominal containment (0 or 1)']+containment_structure_nominal_list
                                rows_to_write_list.append(containment_structure_nominal_with_cont_anat_ROI_row)
                                
                                containment_structure_successes_list = containment_structure_dict['Total successes (containment) list']
                                containment_structure_successes_with_cont_anat_ROI_row = [containment_structure_ROI + ' Total successes']+containment_structure_successes_list
                                rows_to_write_list.append(containment_structure_successes_with_cont_anat_ROI_row)

                                containment_structure_binom_est_list = containment_structure_dict["Binomial estimator list"]
                                containment_structure_binom_est_with_cont_anat_ROI_row = [containment_structure_ROI + ' Mean probability']+containment_structure_binom_est_list
                                rows_to_write_list.append(containment_structure_binom_est_with_cont_anat_ROI_row)

                                containment_structure_stand_err_list = containment_structure_dict["Standard error (containment) list"]
                                containment_structure_stand_err_with_cont_anat_ROI_row = [containment_structure_ROI + ' STD']+containment_structure_stand_err_list
                                rows_to_write_list.append(containment_structure_stand_err_with_cont_anat_ROI_row)

                                containment_structure_conf_int_list = containment_structure_dict["Confidence interval 95 (containment) list"]
                                containment_structure_conf_int_with_cont_anat_ROI_row = [containment_structure_ROI + ' 95% CI']+containment_structure_conf_int_list
                                rows_to_write_list.append(containment_structure_conf_int_with_cont_anat_ROI_row)

                            for row_to_write in rows_to_write_list:
                                write.writerow(row_to_write)
                            
                            del rows_to_write_list



                                
                    patients_progress.update(processing_patients_task, advance = 1)
                    completed_progress.update(processing_patients_completed_task, advance = 1)

                patients_progress.update(processing_patients_task, visible = False)
                completed_progress.update(processing_patients_completed_task, visible = True)

                                


                patientUID_default = "Initializing"
                processing_patient_csv_writing_voxelized_description = "Writing containment CSVs (voxelized) to file [{}]...".format(patientUID_default)
                processing_patients_task = patients_progress.add_task("[red]"+processing_patient_csv_writing_voxelized_description, total = master_structure_info_dict["Global"]["Num patients"])
                processing_patient_csv_writing_voxelized_description_completed = "Writing containment CSVs (voxelized) to file"
                processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_csv_writing_voxelized_description_completed, total=master_structure_info_dict["Global"]["Num patients"], visible=False)
                
                for patientUID,pydicom_item in master_structure_reference_dict.items():

                    processing_patient_csv_writing_voxelized_description = "Writing containment CSVs (voxelized) to file [{}]...".format(patientUID)
                    patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_csv_writing_voxelized_description)

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


                    patients_progress.update(processing_patients_task, advance = 1)
                    completed_progress.update(processing_patients_completed_task, advance = 1)

                patients_progress.update(processing_patients_task, visible = False)
                completed_progress.update(processing_patients_completed_task, visible = True)
            else:
                pass



            if write_dose_to_file_ans ==  True and perform_MC_sim == True:
                important_info.add_text_line("Writing dosimetry CSVs to file.", live_display)

                patientUID_default = "Initializing"
                processing_patient_csv_writing_description = "Writing dosimetry CSVs to file [{}]...".format(patientUID_default)
                processing_patients_task = patients_progress.add_task("[red]"+processing_patient_csv_writing_description, total = master_structure_info_dict["Global"]["Num patients"])
                processing_patient_csv_writing_description_completed = "Writing dosimetry CSVs to file"
                processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_csv_writing_description_completed, total=master_structure_info_dict["Global"]["Num patients"], visible=False)
                
                for patientUID,pydicom_item in master_structure_reference_dict.items():

                    processing_patient_csv_writing_description = "Writing dosimetry CSVs to file [{}]...".format(patientUID)
                    patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_csv_writing_description)

                    bx_structs = bx_ref
                    patient_sp_output_csv_dir = patient_sp_output_csv_dir_dict[patientUID]
                    for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structs]):
                        simulated_bool = specific_bx_structure["Simulated bool"]
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
                            write.writerow(['Simulated', simulated_bool])
                            write.writerow(['BX length (from contour data) (mm)', specific_bx_structure['Reconstructed biopsy cylinder length (from contour data)']])
                            write.writerow(['Num MC dose sims ->',num_MC_dose_simulations_input])
                            write.writerow(['Num bx pt samples ->',num_sample_pts_per_bx])
                            
                            

                            stats_dose_val_all_MC_trials_by_bx_pt_list = specific_bx_structure["MC data: Dose statistics for each sampled bx pt list (mean, std, quantiles)"]
                            # global
                            
                            write.writerow(['---'])
                            write.writerow(['Global'])
                            write.writerow(['---'])
                            nominal_dose_by_bx_pt_arr = np.array(specific_bx_structure['MC data: Dose vals for each sampled bx pt arr (nominal)'])
                            nominal_mean_dose = np.mean(nominal_dose_by_bx_pt_arr)
                            nominal_std_dose = np.std(nominal_dose_by_bx_pt_arr)
                            nominal_std_err_dose = nominal_std_dose / np.sqrt(np.size(nominal_dose_by_bx_pt_arr))
                            nominal_dose_mean_CI = mf.normal_CI_estimator(nominal_mean_dose, nominal_std_err_dose)

                            global_dose_by_bx_pt_arr = np.array(stats_dose_val_all_MC_trials_by_bx_pt_list["Mean dose by bx pt"])
                            global_mean_dose = np.mean(global_dose_by_bx_pt_arr)
                            global_std_dose = np.std(global_dose_by_bx_pt_arr)
                            global_std_err_dose = global_std_dose / np.sqrt(np.size(global_dose_by_bx_pt_arr))
                            global_dose_mean_CI = mf.normal_CI_estimator(global_mean_dose, global_std_err_dose)
                            
                            write.writerow(['Nominal mean dose',nominal_mean_dose])
                            write.writerow(['Nominal std dose',nominal_std_dose])
                            write.writerow(['Nominal std err dose',nominal_std_err_dose])
                            write.writerow(['Nominal mean CI dose lower',nominal_dose_mean_CI[0]])
                            write.writerow(['Nominal mean CI dose upper',nominal_dose_mean_CI[1]])

                            write.writerow(['Global mean dose',global_mean_dose])
                            write.writerow(['Global std dose',global_std_dose])
                            write.writerow(['Global std err dose ',global_std_err_dose])
                            write.writerow(['Global mean CI dose lower',global_dose_mean_CI[0]])
                            write.writerow(['Global mean CI dose upper',global_dose_mean_CI[1]])


                            
                            
                            
                            # point-wise

                            write.writerow(['---'])
                            write.writerow(['Point-wise'])
                            write.writerow(['---'])
                            
                            write.writerow(['Row ->','Fixed bx pt'])
                            write.writerow(['Col ->','Fixed MC trial'])
                            write.writerow(['Vector (mm)',
                                            'X (mm)', 
                                            'Y (mm)', 
                                            'Z (mm)', 
                                            'r (mm)',
                                            'Nominal (Gy)', 
                                            'Mean (Gy)', 
                                            'STD (Gy)', 
                                            'All MC trials doses (Gy) -->'
                                            ])
                            
                            
                            for pt_index in range(num_sample_pts_per_bx):
                                #dose_vals_row_with_point = dose_vals_row.copy()
                                pt_radius_bx_coord_sys = np.linalg.norm(bx_points_bx_coords_sys_arr_list[pt_index][0:2])
                                mean_dose_val_specific_bx_pt = stats_dose_val_all_MC_trials_by_bx_pt_list["Mean dose by bx pt"][pt_index]
                                std_dose_val_specific_bx_pt = stats_dose_val_all_MC_trials_by_bx_pt_list["STD by bx pt"][pt_index]
                                nominal_dose_val_specific_bx_pt = specific_bx_structure['MC data: Dose vals for each sampled bx pt arr (nominal)'][pt_index]
                                info_row_part = [bx_points_bx_coords_sys_arr_list[pt_index], 
                                                 bx_points_bx_coords_sys_arr_list[pt_index][0], 
                                                 bx_points_bx_coords_sys_arr_list[pt_index][1], 
                                                 bx_points_bx_coords_sys_arr_list[pt_index][2], 
                                                 pt_radius_bx_coord_sys, 
                                                 nominal_dose_val_specific_bx_pt,
                                                 mean_dose_val_specific_bx_pt, 
                                                 std_dose_val_specific_bx_pt
                                                 ]
                                dose_vals_row_arr = specific_bx_structure['MC data: Dose vals for each sampled bx pt arr (all MC trials)'][pt_index]
                                dose_vals_row_list = dose_vals_row_arr.tolist()
                                complete_dose_vals_row = info_row_part + dose_vals_row_list
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
                                write.writerow(['Differential DVH info '+dvh_display_as_str])
                                write.writerow(['Each row is a fixed MC trial'])
                                write.writerow(['Lower dose bin edge (across)']+differential_dvh_dose_bin_edges_1darr.tolist()[0:-1])
                                write.writerow(['Upper dose bin edge (across)']+differential_dvh_dose_bin_edges_1darr.tolist()[1:])
                                write.writerow(['Trial number (down)'])
                                for mc_trial in range(differential_dvh_histogram_counts_by_MC_trial_arr.shape[0]):
                                    if mc_trial == 0:
                                        mc_trial_desc = 'Nominal'
                                    else:
                                        mc_trial_desc = str(mc_trial)
                                    write.writerow([mc_trial_desc]+differential_dvh_histogram_counts_by_MC_trial_arr[mc_trial,:].tolist())


                                cumulative_dvh_dose_vals_by_MC_trial_1darr = cumulative_dvh_dict["Dose vals arr (Gy)"]
                                
                                write.writerow(['___'])
                                write.writerow(['Cumulative DVH info '+dvh_display_as_str])
                                write.writerow(['Each row is a fixed MC trial'])
                                write.writerow(['Dose value (across)']+cumulative_dvh_dose_vals_by_MC_trial_1darr.tolist())
                                write.writerow(['Trial number (down)'])
                                for mc_trial in range(cumulative_dvh_counts_by_MC_trial_arr.shape[0]):
                                    if mc_trial == 0:
                                        mc_trial_desc = 'Nominal'
                                    else:
                                        mc_trial_desc = str(mc_trial)
                                    write.writerow([mc_trial_desc]+cumulative_dvh_counts_by_MC_trial_arr[mc_trial,:].tolist())

                            write.writerow(['___'])
                            write.writerow(['DVH metrics, percentages are relative to CTV target dose'])
                            write.writerow(['Each row is a fixed DVH metric, each column is a fixed MC trial'])
                            for vol_DVH_percent in volume_DVH_percent_dose:
                                dvh_metric_all_MC_trials = dvh_metric_vol_dose_percent_dict[str(vol_DVH_percent)]["All MC trials list"]
                                dvh_metric_nominal = dvh_metric_vol_dose_percent_dict[str(vol_DVH_percent)]["Nominal"]
                                dvh_metric_mean = dvh_metric_vol_dose_percent_dict[str(vol_DVH_percent)]["Mean"]
                                dvh_metric_std = dvh_metric_vol_dose_percent_dict[str(vol_DVH_percent)]["STD"]
                                dvh_metric_quantiles_dict = dvh_metric_vol_dose_percent_dict[str(vol_DVH_percent)]["Quantiles"]
                                all_MC_trial_number_list = np.arange(1,len(dvh_metric_all_MC_trials)).tolist()
                                nominal_and_all_MC_trial_number_list = ["Nominal"]+all_MC_trial_number_list
                                write.writerow([' ']) 
                                write.writerow(['Trial number (across)']+nominal_and_all_MC_trial_number_list)
                                write.writerow(['DVH quantity (down)'])
                                write.writerow(['V'+str(vol_DVH_percent)+'%']+dvh_metric_all_MC_trials)
                                write.writerow(['V'+str(vol_DVH_percent)+'% mean', dvh_metric_mean]) 
                                write.writerow(['V'+str(vol_DVH_percent)+'% STD', dvh_metric_std])
                                for q,q_val in dvh_metric_quantiles_dict.items():
                                    write.writerow(['V'+str(vol_DVH_percent)+'% '+str(q), q_val])

                    patients_progress.update(processing_patients_task, advance = 1)
                    completed_progress.update(processing_patients_completed_task, advance = 1)

                patients_progress.update(processing_patients_task, visible = False)
                completed_progress.update(processing_patients_completed_task, visible = True)


                patientUID_default = "Initializing"
                processing_patient_csv_writing_voxelized_description = "Writing dosimetry CSVs (voxelized) to file [{}]...".format(patientUID_default)
                processing_patients_task = patients_progress.add_task("[red]"+processing_patient_csv_writing_voxelized_description, total = master_structure_info_dict["Global"]["Num patients"])
                processing_patient_csv_writing_voxelized_description_completed = "Writing dosimetry CSVs (voxelized) to file"
                processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_csv_writing_voxelized_description_completed, total=master_structure_info_dict["Global"]["Num patients"], visible=False)
                
                for patientUID,pydicom_item in master_structure_reference_dict.items():

                    processing_patient_csv_writing_voxelized_description = "Writing dosimetry CSVs (voxelized) to file [{}]...".format(patientUID)
                    patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_csv_writing_voxelized_description)

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

                    patients_progress.update(processing_patients_task, advance = 1)
                    completed_progress.update(processing_patients_completed_task, advance = 1)

                patients_progress.update(processing_patients_task, visible = False)
                completed_progress.update(processing_patients_completed_task, visible = True)
            else:
                pass
            


            if create_at_least_one_production_plot == True and perform_MC_sim == True:

                important_info.add_text_line("Creating production plots.", live_display)

                # generate a pandas data frame that is used in numerous production plot functions
                for patientUID,pydicom_item in master_structure_reference_dict.items():
                    for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structs]):                        
                        stats_dose_val_all_MC_trials_by_bx_pt_list = specific_bx_structure["MC data: Dose statistics for each sampled bx pt list (mean, std, quantiles)"]
                        mean_dose_val_specific_bx_pt = stats_dose_val_all_MC_trials_by_bx_pt_list["Mean dose by bx pt"].copy()
                        std_dose_val_specific_bx_pt = stats_dose_val_all_MC_trials_by_bx_pt_list["STD by bx pt"].copy()
                        quantiles_dose_val_specific_bx_pt_dict_of_lists = stats_dose_val_all_MC_trials_by_bx_pt_list["Quantiles dose by bx pt dict"].copy()
                        bx_points_bx_coords_sys_arr = specific_bx_structure["Random uniformly sampled volume pts bx coord sys arr"]
                        bx_points_XY_bx_coords_sys_arr_list = list(bx_points_bx_coords_sys_arr[:,0:2])
                        pt_radius_bx_coord_sys = np.linalg.norm(bx_points_XY_bx_coords_sys_arr_list, axis = 1)

                        dose_output_dict_for_pandas_data_frame = {"Radial pos (mm)": pt_radius_bx_coord_sys, 
                                                                  "Axial pos Z (mm)": bx_points_bx_coords_sys_arr[:,2], 
                                                                  "Mean dose (Gy)": mean_dose_val_specific_bx_pt, 
                                                                  "STD dose": std_dose_val_specific_bx_pt
                                                                  }
                        dose_output_dict_for_pandas_data_frame.update(quantiles_dose_val_specific_bx_pt_dict_of_lists)
                        dose_output_pandas_data_frame = pandas.DataFrame(data=dose_output_dict_for_pandas_data_frame)
                        
                        specific_bx_structure["Output data frames"]["Dose output Z and radius"] = dose_output_pandas_data_frame
                        specific_bx_structure["Output dicts for data frames"]["Dose output Z and radius"] = dose_output_dict_for_pandas_data_frame

                


                #### BEGIN MAKING PLOTS ####


                # Plot boxplots of sampled rigid shift vectors
                if production_plots_input_dictionary["Sampled translation vector magnitudes box plots"]["Plot bool"] == True:
                    
                    general_plot_name_string = production_plots_input_dictionary["Sampled translation vector magnitudes box plots"]["Plot name"]
                    box_plot_color = production_plots_input_dictionary["Sampled translation vector magnitudes box plots"]["Plot color"]

                    patientUID_default = "Initializing"
                    processing_patient_production_plot_description = "Creating box plots of sampled rigid shift vectors [{}]...".format(patientUID_default)
                    processing_patients_task = patients_progress.add_task("[red]"+processing_patient_production_plot_description, total = master_structure_info_dict["Global"]["Num patients"])
                    processing_patient_production_plot_description_completed = "Creating box plots of sampled rigid shift vectors"
                    processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_production_plot_description_completed, total=master_structure_info_dict["Global"]["Num patients"], visible=False)


                    for patientUID,pydicom_item in master_structure_reference_dict.items():
                        
                        processing_patient_production_plot_description = "Creating box plots of sampled rigid shift vectors [{}]...".format(patientUID)
                        patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_production_plot_description)

                        max_simulations = master_structure_info_dict["Global"]["MC info"]["Max of num MC simulations"]
                        production_plots.production_plot_sampled_shift_vector_box_plots_by_patient(patientUID,
                                                patient_sp_output_figures_dir_dict,
                                                structs_referenced_list,
                                                bx_structs,
                                                pydicom_item,
                                                max_simulations,
                                                all_ref_key,
                                                svg_image_scale,
                                                svg_image_width,
                                                svg_image_height,
                                                box_plot_color,
                                                general_plot_name_string
                                                )
                        
                        patients_progress.update(processing_patients_task, advance = 1)
                        completed_progress.update(processing_patients_completed_task, advance = 1)

                    patients_progress.update(processing_patients_task, visible = False)
                    completed_progress.update(processing_patients_completed_task, visible = True)
                else:
                    pass

                
                # all MC trials spatial axial dose distribution with global regression 

                if production_plots_input_dictionary["Axial dose distribution all trials and global regression"]["Plot bool"] == True:
                    
                    general_plot_name_string = production_plots_input_dictionary["Axial dose distribution all trials and global regression"]["Plot name"]
                
                    patientUID_default = "Initializing"
                    processing_patient_production_plot_description = "Creating axial dose distribution scatter plot (all trials) [{}]...".format(patientUID_default)
                    processing_patients_task = patients_progress.add_task("[red]"+processing_patient_production_plot_description, total = master_structure_info_dict["Global"]["Num patients"])
                    processing_patient_production_plot_description_completed = "Creating axial dose distribution scatter plot (all trials)"
                    processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_production_plot_description_completed, total=master_structure_info_dict["Global"]["Num patients"], visible=False)


                    for patientUID,pydicom_item in master_structure_reference_dict.items():
                        
                        processing_patient_production_plot_description = "Creating axial dose distribution scatter plot (all trials) [{}]...".format(patientUID)
                        patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_production_plot_description)

                        production_plots.production_plot_axial_dose_distribution_all_trials_and_regression_by_patient(patient_sp_output_figures_dir_dict,
                                                                    patientUID,
                                                                    pydicom_item,
                                                                    bx_structs,
                                                                    global_regression_input,
                                                                    regression_type_input,
                                                                    parallel_pool,
                                                                    num_bootstraps_for_regression_plots_input,
                                                                    NPKR_bandwidth,
                                                                    svg_image_scale,
                                                                    svg_image_width,
                                                                    svg_image_height,
                                                                    num_z_vals_to_evaluate_for_regression_plots,
                                                                    general_plot_name_string
                                                                    )
                        
                        patients_progress.update(processing_patients_task, advance = 1)
                        completed_progress.update(processing_patients_completed_task, advance = 1)

                    patients_progress.update(processing_patients_task, visible = False)
                    completed_progress.update(processing_patients_completed_task, visible = True)
                else:
                    pass


                
                # 3d scatter axial and radial dose distribution map

                if production_plots_input_dictionary["Axial and radial (3D, surface) dose distribution"]["Plot bool"] == True:
                    
                    general_plot_name_string = production_plots_input_dictionary["Axial and radial (3D, surface) dose distribution"]["Plot name"]

                    patientUID_default = "Initializing"
                    processing_patient_production_plot_description = "Creating (3D) scatter axial/radial dose distribution plot (all trials) [{}]...".format(patientUID_default)
                    processing_patients_task = patients_progress.add_task("[red]"+processing_patient_production_plot_description, total = master_structure_info_dict["Global"]["Num patients"])
                    processing_patient_production_plot_description_completed = "Creating (3D) scatter axial/radial dose distribution plot (all trials)"
                    processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_production_plot_description_completed, total=master_structure_info_dict["Global"]["Num patients"], visible=False)


                    for patientUID,pydicom_item in master_structure_reference_dict.items():
                        
                        processing_patient_production_plot_description = "Creating (3D) scatter axial/radial dose distribution plot (all trials) [{}]...".format(patientUID)
                        patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_production_plot_description)

                        production_plots.production_3d_scatter_dose_axial_radial_distribution_by_patient(patient_sp_output_figures_dir_dict,
                                                                    patientUID,
                                                                    pydicom_item,
                                                                    bx_structs,
                                                                    svg_image_scale,
                                                                    svg_image_width,
                                                                    svg_image_height,
                                                                    general_plot_name_string
                                                                    )
                        
                        patients_progress.update(processing_patients_task, advance = 1)
                        completed_progress.update(processing_patients_completed_task, advance = 1)

                    patients_progress.update(processing_patients_task, visible = False)
                    completed_progress.update(processing_patients_completed_task, visible = True)
                else:
                    pass


                # 2d color axial and radial dose distribution map

                if production_plots_input_dictionary["Axial and radial (2D, color) dose distribution"]["Plot bool"] == True:
                    
                    general_plot_name_string = production_plots_input_dictionary["Axial and radial (2D, color) dose distribution"]["Plot name"]

                    patientUID_default = "Initializing"
                    processing_patient_production_plot_description = "Creating (2D) colored scatter axial/radial dose distribution plot (all trials) [{}]...".format(patientUID_default)
                    processing_patients_task = patients_progress.add_task("[red]"+processing_patient_production_plot_description, total = master_structure_info_dict["Global"]["Num patients"])
                    processing_patient_production_plot_description_completed = "Creating (2D) colored scatter axial/radial dose distribution plot (all trials)"
                    processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_production_plot_description_completed, total=master_structure_info_dict["Global"]["Num patients"], visible=False)


                    for patientUID,pydicom_item in master_structure_reference_dict.items():
                        
                        processing_patient_production_plot_description = "Creating (2D) colored scatter axial/radial dose distribution plot (all trials) [{}]...".format(patientUID)
                        patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_production_plot_description)

                        production_plots.production_2d_scatter_dose_axial_radial_color_distribution_by_patient(patient_sp_output_figures_dir_dict,
                                                                    patientUID,
                                                                    pydicom_item,
                                                                    bx_structs,
                                                                    svg_image_scale,
                                                                    svg_image_width,
                                                                    svg_image_height,
                                                                    general_plot_name_string
                                                                    )
                        
                        patients_progress.update(processing_patients_task, advance = 1)
                        completed_progress.update(processing_patients_completed_task, advance = 1)

                    patients_progress.update(processing_patients_task, visible = False)
                    completed_progress.update(processing_patients_completed_task, visible = True)
                else:
                    pass


                


                # quantiles scatter plot of axial dose distribution

                if production_plots_input_dictionary["Axial dose distribution quantiles scatter plot"]["Plot bool"] == True:
                    
                    general_plot_name_string = production_plots_input_dictionary["Axial dose distribution quantiles scatter plot"]["Plot name"]

                    patientUID_default = "Initializing"
                    processing_patient_production_plot_description = "Creating axial dose distribution quantile scatter plots [{}]...".format(patientUID_default)
                    processing_patients_task = patients_progress.add_task("[red]"+processing_patient_production_plot_description, total = master_structure_info_dict["Global"]["Num patients"])
                    processing_patient_production_plot_description_completed = "Creating axial dose distribution quantile scatter plots"
                    processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_production_plot_description_completed, total=master_structure_info_dict["Global"]["Num patients"], visible=False)


                    for patientUID,pydicom_item in master_structure_reference_dict.items():
                        
                        processing_patient_production_plot_description = "Creating axial dose distribution quantile scatter plots [{}]...".format(patientUID)
                        patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_production_plot_description)


                        production_plots.production_plot_axial_dose_distribution_quantile_scatter_by_patient(patient_sp_output_figures_dir_dict,
                                                                        patientUID,
                                                                        pydicom_item,
                                                                        bx_structs,
                                                                        svg_image_scale,
                                                                        svg_image_width,
                                                                        svg_image_height,
                                                                        general_plot_name_string
                                                                        )


                        patients_progress.update(processing_patients_task, advance = 1)
                        completed_progress.update(processing_patients_completed_task, advance = 1)

                    patients_progress.update(processing_patients_task, visible = False)
                    completed_progress.update(processing_patients_completed_task, visible = True)   
                else:
                    pass

                


                # quantile regression of axial dose distribution

                
                if production_plots_input_dictionary["Axial dose distribution quantiles regression plot"]["Plot bool"] == True:
                    
                    general_plot_name_string = production_plots_input_dictionary["Axial dose distribution quantiles regression plot"]["Plot name"]
                    
                    patientUID_default = "Initializing"
                    processing_patient_production_plot_description = "Creating axial dose distribution quantile regression plots [{}]...".format(patientUID_default)
                    processing_patients_task = patients_progress.add_task("[red]"+processing_patient_production_plot_description, total = master_structure_info_dict["Global"]["Num patients"])
                    processing_patient_production_plot_description_completed = "Creating axial dose distribution quantile regression plots"
                    processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_production_plot_description_completed, total=master_structure_info_dict["Global"]["Num patients"], visible=False)


                    for patientUID,pydicom_item in master_structure_reference_dict.items():
                        
                        processing_patient_production_plot_description = "Creating axial dose distribution quantile regression plots [{}]...".format(patientUID)
                        patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_production_plot_description)

                        production_plots.production_plot_axial_dose_distribution_quantile_regressions_by_patient(patient_sp_output_figures_dir_dict,
                                                                    patientUID,
                                                                    pydicom_item,
                                                                    bx_structs,
                                                                    regression_type_input,
                                                                    parallel_pool,
                                                                    NPKR_bandwidth,
                                                                    num_bootstraps_for_regression_plots_input,
                                                                    svg_image_scale,
                                                                    svg_image_width,
                                                                    svg_image_height,
                                                                    num_z_vals_to_evaluate_for_regression_plots,
                                                                    general_plot_name_string
                                                                    )
                        
                        patients_progress.update(processing_patients_task, advance = 1)
                        completed_progress.update(processing_patients_completed_task, advance = 1)

                    patients_progress.update(processing_patients_task, visible = False)
                    completed_progress.update(processing_patients_completed_task, visible = True)   
                else:
                    pass



                # voxelized box plot axial dose distribution  

                if production_plots_input_dictionary["Axial dose distribution voxelized box plot"]["Plot bool"] == True:
                    
                    general_plot_name_string = production_plots_input_dictionary["Axial dose distribution voxelized box plot"]["Plot name"]
                    box_plot_color = production_plots_input_dictionary["Axial dose distribution voxelized box plot"]["Plot color"]
                    
                    patientUID_default = "Initializing"
                    processing_patient_production_plot_description = "Creating voxelized axial dose distribution box plots [{}]...".format(patientUID_default)
                    processing_patients_task = patients_progress.add_task("[red]"+processing_patient_production_plot_description, total = master_structure_info_dict["Global"]["Num patients"])
                    processing_patient_production_plot_description_completed = "Creating voxelized axial dose distribution box plots"
                    processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_production_plot_description_completed, total=master_structure_info_dict["Global"]["Num patients"], visible=False)


                    for patientUID,pydicom_item in master_structure_reference_dict.items():
                        
                        processing_patient_production_plot_description = "Creating voxelized axial dose distribution box plots [{}]...".format(patientUID)
                        patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_production_plot_description)

                        production_plots.production_plot_voxelized_axial_dose_distribution_box_plot_by_patient(patient_sp_output_figures_dir_dict,
                                                                            patientUID,
                                                                            pydicom_item,
                                                                            bx_structs,
                                                                            svg_image_scale,
                                                                            svg_image_width,
                                                                            svg_image_height,
                                                                            box_plot_color,
                                                                            general_plot_name_string
                                                                            )
                        
                        patients_progress.update(processing_patients_task, advance = 1)
                        completed_progress.update(processing_patients_completed_task, advance = 1)

                    patients_progress.update(processing_patients_task, visible = False)
                    completed_progress.update(processing_patients_completed_task, visible = True)   
                else:
                    pass

                

                # voxelized violin plot axial dose distribution  

                if production_plots_input_dictionary["Axial dose distribution voxelized violin plot"]["Plot bool"] == True:
                    
                    general_plot_name_string = production_plots_input_dictionary["Axial dose distribution voxelized violin plot"]["Plot name"]
                    violin_plot_color = production_plots_input_dictionary["Axial dose distribution voxelized violin plot"]["Plot color"]
                    
                    patientUID_default = "Initializing"
                    processing_patient_production_plot_description = "Creating voxelized axial dose distribution violin plots [{}]...".format(patientUID_default)
                    processing_patients_task = patients_progress.add_task("[red]"+processing_patient_production_plot_description, total = master_structure_info_dict["Global"]["Num patients"])
                    processing_patient_production_plot_description_completed = "Creating voxelized axial dose distribution violin plots"
                    processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_production_plot_description_completed, total=master_structure_info_dict["Global"]["Num patients"], visible=False)


                    for patientUID,pydicom_item in master_structure_reference_dict.items():
                        
                        processing_patient_production_plot_description = "Creating voxelized axial dose distribution violin plots [{}]...".format(patientUID)
                        patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_production_plot_description)

                        production_plots.production_plot_voxelized_axial_dose_distribution_violin_plot_by_patient(patient_sp_output_figures_dir_dict,
                                                                            patientUID,
                                                                            pydicom_item,
                                                                            bx_structs,
                                                                            svg_image_scale,
                                                                            svg_image_width,
                                                                            svg_image_height,
                                                                            violin_plot_color,
                                                                            general_plot_name_string
                                                                            )
                        
                        patients_progress.update(processing_patients_task, advance = 1)
                        completed_progress.update(processing_patients_completed_task, advance = 1)

                    patients_progress.update(processing_patients_task, visible = False)
                    completed_progress.update(processing_patients_completed_task, visible = True)   
                else:
                    pass
                
                
                        
                # Show N trials of differential DVH plots

                if production_plots_input_dictionary["Differential DVH showing N trials plot"]["Plot bool"] == True:
                    
                    general_plot_name_string = production_plots_input_dictionary["Differential DVH showing N trials plot"]["Plot name"]
                    
                    patientUID_default = "Initializing"
                    processing_patient_production_plot_description = "Creating differential DVH plots ("+str(num_differential_dvh_plots_to_show)+" trials) [{}]...".format(patientUID_default)
                    processing_patients_task = patients_progress.add_task("[red]"+processing_patient_production_plot_description, total = master_structure_info_dict["Global"]["Num patients"])
                    processing_patient_production_plot_description_completed = "Creating differential DVH plots ("+str(num_differential_dvh_plots_to_show)+" trials)"
                    processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_production_plot_description_completed, total=master_structure_info_dict["Global"]["Num patients"], visible=False)


                    for patientUID,pydicom_item in master_structure_reference_dict.items():
                        
                        processing_patient_production_plot_description = "Creating differential DVH plots ("+str(num_differential_dvh_plots_to_show)+" trials) [{}]...".format(patientUID)
                        patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_production_plot_description)

                        production_plots.production_plot_differential_DVH_showing_N_trials_by_patient(patient_sp_output_figures_dir_dict,
                                                patientUID,
                                                pydicom_item,
                                                bx_structs,
                                                svg_image_scale,
                                                svg_image_width,
                                                svg_image_height,
                                                num_differential_dvh_plots_to_show,
                                                general_plot_name_string
                                                )
                        
                        patients_progress.update(processing_patients_task, advance = 1)
                        completed_progress.update(processing_patients_completed_task, advance = 1)

                    patients_progress.update(processing_patients_task, visible = False)
                    completed_progress.update(processing_patients_completed_task, visible = True)   
                else:
                    pass


                # Show box plots (quantiles) from all trials of DVH data

                if production_plots_input_dictionary["Differential DVH dose binned all trials box plot"]["Plot bool"] == True:
                    
                    general_plot_name_string = production_plots_input_dictionary["Differential DVH dose binned all trials box plot"]["Plot name"]
                    box_plot_color = production_plots_input_dictionary["Differential DVH dose binned all trials box plot"]["Box plot color"]
                    nominal_pt_color = production_plots_input_dictionary["Differential DVH dose binned all trials box plot"]["Nominal point color"]

                    patientUID_default = "Initializing"
                    processing_patient_production_plot_description = "Creating box plots of differential DVH data (all trials) [{}]...".format(patientUID_default)
                    processing_patients_task = patients_progress.add_task("[red]"+processing_patient_production_plot_description, total = master_structure_info_dict["Global"]["Num patients"])
                    processing_patient_production_plot_description_completed = "Creating box plots of differential DVH data (all trials)"
                    processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_production_plot_description_completed, total=master_structure_info_dict["Global"]["Num patients"], visible=False)


                    for patientUID,pydicom_item in master_structure_reference_dict.items():
                        
                        processing_patient_production_plot_description = "Creating box plots of differential DVH data (all trials) [{}]...".format(patientUID)
                        patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_production_plot_description)

                        production_plots.production_plot_differential_dvh_quantile_box_plot(patient_sp_output_figures_dir_dict,
                                                    patientUID,
                                                    pydicom_item,
                                                    bx_structs,
                                                    svg_image_scale,
                                                    svg_image_width,
                                                    svg_image_height,
                                                    box_plot_color,
                                                    nominal_pt_color,
                                                    general_plot_name_string
                                                    ) 
                        
                        patients_progress.update(processing_patients_task, advance = 1)
                        completed_progress.update(processing_patients_completed_task, advance = 1)

                    patients_progress.update(processing_patients_task, visible = False)
                    completed_progress.update(processing_patients_completed_task, visible = True)  
                else:
                    pass 
                        


                # show N trials of cumulative DVH plots

                if production_plots_input_dictionary["Cumulative DVH showing N trials plot"]["Plot bool"] == True:
                    
                    general_plot_name_string = production_plots_input_dictionary["Cumulative DVH showing N trials plot"]["Plot name"]
                    
                    patientUID_default = "Initializing"
                    processing_patient_production_plot_description = "Creating cumulative DVH plots ("+str(num_cumulative_dvh_plots_to_show)+" trials) [{}]...".format(patientUID_default)
                    processing_patients_task = patients_progress.add_task("[red]"+processing_patient_production_plot_description, total = master_structure_info_dict["Global"]["Num patients"])
                    processing_patient_production_plot_description_completed = "Creating cumulative DVH plots ("+str(num_cumulative_dvh_plots_to_show)+" trials)"
                    processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_production_plot_description_completed, total=master_structure_info_dict["Global"]["Num patients"], visible=False)


                    for patientUID,pydicom_item in master_structure_reference_dict.items():
                        
                        processing_patient_production_plot_description = "Creating cumulative DVH plots ("+str(num_cumulative_dvh_plots_to_show)+" trials) [{}]...".format(patientUID)
                        patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_production_plot_description)

                        production_plots.production_plot_cumulative_DVH_showing_N_trials_by_patient(patient_sp_output_figures_dir_dict,
                                                    patientUID,
                                                    pydicom_item,
                                                    bx_structs,
                                                    svg_image_scale,
                                                    svg_image_width,
                                                    svg_image_height,
                                                    num_cumulative_dvh_plots_to_show,
                                                    general_plot_name_string
                                                    )
                        
                        patients_progress.update(processing_patients_task, advance = 1)
                        completed_progress.update(processing_patients_completed_task, advance = 1)

                    patients_progress.update(processing_patients_task, visible = False)
                    completed_progress.update(processing_patients_completed_task, visible = True)   
                else:
                    pass
                        


                # show quantile regression plots of cumulative DVH data from all trials

                if (production_plots_input_dictionary["Cumulative DVH quantile regression all trials plot regression only"]["Plot bool"] == True) or \
                    (production_plots_input_dictionary["Cumulative DVH quantile regression all trials plot colorwash"]["Plot bool"] == True):
                    
                    general_plot_name_string_regression_only = production_plots_input_dictionary["Cumulative DVH quantile regression all trials plot regression only"]["Plot name"]
                    general_plot_name_string_colorwash = production_plots_input_dictionary["Cumulative DVH quantile regression all trials plot colorwash"]["Plot name"]

                    plot_colorwash_bool = production_plots_input_dictionary["Cumulative DVH quantile regression all trials plot colorwash"]["Plot bool"]
                    plot_regression_only_bool = production_plots_input_dictionary["Cumulative DVH quantile regression all trials plot regression only"]["Plot bool"]

                    patientUID_default = "Initializing"
                    processing_patient_production_plot_description = "Creating cumulative DVH quantile regression plots [{}]...".format(patientUID_default)
                    processing_patients_task = patients_progress.add_task("[red]"+processing_patient_production_plot_description, total = master_structure_info_dict["Global"]["Num patients"])
                    processing_patient_production_plot_description_completed = "Creating cumulative DVH quantile regression plots"
                    processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_production_plot_description_completed, total=master_structure_info_dict["Global"]["Num patients"], visible=False)


                    for patientUID,pydicom_item in master_structure_reference_dict.items():
                        
                        processing_patient_production_plot_description = "Creating cumulative DVH quantile regression plots [{}]...".format(patientUID)
                        patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_production_plot_description)

                        production_plots.production_plot_cumulative_DVH_quantile_regression_by_patient(patient_sp_output_figures_dir_dict,
                                                    patientUID,
                                                    pydicom_item,
                                                    bx_structs,
                                                    regression_type_input,
                                                    num_z_vals_to_evaluate_for_regression_plots,
                                                    parallel_pool,
                                                    NPKR_bandwidth,
                                                    num_bootstraps_for_regression_plots_input,
                                                    svg_image_scale,
                                                    svg_image_width,
                                                    svg_image_height,
                                                    plot_regression_only_bool,
                                                    plot_colorwash_bool,
                                                    general_plot_name_string_regression_only,
                                                    general_plot_name_string_colorwash
                                                    )
                        
                        patients_progress.update(processing_patients_task, advance = 1)
                        completed_progress.update(processing_patients_completed_task, advance = 1)

                    patients_progress.update(processing_patients_task, visible = False)
                    completed_progress.update(processing_patients_completed_task, visible = True)   
                else:
                    pass
                        


                # perform containment probabilities plots and regressions
                #live_display.stop()
                if production_plots_input_dictionary["Tissue classification scatter and regression probabilities all trials plot"]["Plot bool"] == True:
                    
                    general_plot_name_string = production_plots_input_dictionary["Tissue classification scatter and regression probabilities all trials plot"]["Plot name"]


                    patientUID_default = "Initializing"
                    processing_patient_production_plot_description = "Creating containment probability plots [{}]...".format(patientUID_default)
                    processing_patients_task = patients_progress.add_task("[red]"+processing_patient_production_plot_description, total = master_structure_info_dict["Global"]["Num patients"])
                    processing_patient_production_plot_description_completed = "Creating containment probability plots"
                    processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_production_plot_description_completed, total=master_structure_info_dict["Global"]["Num patients"], visible=False)


                    for patientUID,pydicom_item in master_structure_reference_dict.items():
                        
                        processing_patient_production_plot_description = "Creating containment probability plots [{}]...".format(patientUID)
                        patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_production_plot_description)

                        production_plots.production_plot_containment_probabilities_by_patient(patient_sp_output_figures_dir_dict,
                                                    patientUID,
                                                    pydicom_item,
                                                    bx_structs,
                                                    regression_type_input,
                                                    parallel_pool,
                                                    NPKR_bandwidth,
                                                    num_bootstraps_for_regression_plots_input,
                                                    num_z_vals_to_evaluate_for_regression_plots,
                                                    tissue_class_probability_plot_type_list,
                                                    svg_image_scale,
                                                    svg_image_width,
                                                    svg_image_height,
                                                    general_plot_name_string
                                                    )
                        
                        patients_progress.update(processing_patients_task, advance = 1)
                        completed_progress.update(processing_patients_completed_task, advance = 1)

                    patients_progress.update(processing_patients_task, visible = False)
                    completed_progress.update(processing_patients_completed_task, visible = True)  
                else:
                    pass


                # perform containment probabilities plots and regressions
                #live_display.stop()
                if production_plots_input_dictionary["Tissue classification mutual probabilities plot"]["Plot bool"] == True:
                    
                    general_plot_name_string = production_plots_input_dictionary["Tissue classification mutual probabilities plot"]["Plot name"]
                    structure_miss_probability_roi = production_plots_input_dictionary["Tissue classification mutual probabilities plot"]["Structure miss ROI"]


                    patientUID_default = "Initializing"
                    processing_patient_production_plot_description = "Creating mutual containment probability plots [{}]...".format(patientUID_default)
                    processing_patients_task = patients_progress.add_task("[red]"+processing_patient_production_plot_description, total = master_structure_info_dict["Global"]["Num patients"])
                    processing_patient_production_plot_description_completed = "Creating mutual containment probability plots"
                    processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_production_plot_description_completed, total=master_structure_info_dict["Global"]["Num patients"], visible=False)


                    for patientUID,pydicom_item in master_structure_reference_dict.items():
                        
                        processing_patient_production_plot_description = "Creating mutual containment probability plots [{}]...".format(patientUID)
                        patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_production_plot_description)

                        
                        production_plots.production_plot_mutual_containment_probabilities_by_patient(patient_sp_output_figures_dir_dict,
                                                    patientUID,
                                                    pydicom_item,
                                                    bx_structs,
                                                    regression_type_input,
                                                    parallel_pool,
                                                    NPKR_bandwidth,
                                                    num_bootstraps_for_regression_plots_input,
                                                    num_z_vals_to_evaluate_for_regression_plots,
                                                    tissue_class_probability_plot_type_list,
                                                    svg_image_scale,
                                                    svg_image_width,
                                                    svg_image_height,
                                                    general_plot_name_string,
                                                    structure_miss_probability_roi
                                                    )
                        
                        patients_progress.update(processing_patients_task, advance = 1)
                        completed_progress.update(processing_patients_completed_task, advance = 1)

                    patients_progress.update(processing_patients_task, visible = False)
                    completed_progress.update(processing_patients_completed_task, visible = True)  
                else:
                    pass



            if perform_fanova == True:
                if perform_containment_fanova == True:
                    if production_plots_input_dictionary["Tissue classification Sobol indices global plot"]["Plot bool"] == True:
                        
                        tissue_class_sobol_global_plot_bool_dict = production_plots_input_dictionary["Tissue classification Sobol indices global plot"]["Plot bool dict"]
                        general_plot_name_string_dict = production_plots_input_dictionary["Tissue classification Sobol indices global plot"]["Plot name dict"]
                        structure_miss_probability_roi = production_plots_input_dictionary["Tissue classification Sobol indices global plot"]["Structure miss ROI"]
                        box_plot_points_option = production_plots_input_dictionary["Tissue classification Sobol indices global plot"]["Box plot points option"]
                        
                        global_sobol_plots_task_indeterminate = indeterminate_progress_main.add_task('[red]Plotting global Sobol indices (containment)...', total=None)
                        global_sobol_plots_task_indeterminate_completed = completed_progress.add_task('[green]Plotting global Sobol indices (containment)', total=1, visible = False)

                        production_plots.production_plot_sobol_indices_global_containment(patient_sp_output_figures_dir_dict,
                                                master_structure_reference_dict,
                                                master_structure_info_dict,
                                                bx_structs,
                                                dil_ref,
                                                svg_image_scale,
                                                svg_image_width,
                                                svg_image_height,
                                                general_plot_name_string_dict,
                                                structure_miss_probability_roi,
                                                tissue_class_sobol_global_plot_bool_dict,
                                                box_plot_points_option
                                                )

                        indeterminate_progress_main.update(global_sobol_plots_task_indeterminate, visible = False)
                        completed_progress.update(global_sobol_plots_task_indeterminate_completed, advance = 1,visible = True)
                        live_display.refresh()
                    else:
                        pass

                if perform_dose_fanova == True:
                    if production_plots_input_dictionary["Dosimetry Sobol indices global plot"]["Plot bool"] == True:
                        
                        dose_sobol_global_plot_bool_dict = production_plots_input_dictionary["Dosimetry Sobol indices global plot"]["Plot bool dict"]
                        general_plot_name_string_dict = production_plots_input_dictionary["Dosimetry Sobol indices global plot"]["Plot name dict"]
                        box_plot_points_option = production_plots_input_dictionary["Dosimetry Sobol indices global plot"]["Box plot points option"]
                        
                        global_sobol_plots_task_indeterminate = indeterminate_progress_main.add_task('[red]Plotting global Sobol indices (dose)...', total=None)
                        global_sobol_plots_task_indeterminate_completed = completed_progress.add_task('[green]Plotting global Sobol indices (dose)', total=1, visible = False)

                        production_plots.production_plot_sobol_indices_global_dosimetry(patient_sp_output_figures_dir_dict,
                                                master_structure_reference_dict,
                                                master_structure_info_dict,
                                                bx_structs,
                                                svg_image_scale,
                                                svg_image_width,
                                                svg_image_height,
                                                general_plot_name_string_dict,
                                                dose_sobol_global_plot_bool_dict,
                                                box_plot_points_option
                                                )

                        indeterminate_progress_main.update(global_sobol_plots_task_indeterminate, visible = False)
                        completed_progress.update(global_sobol_plots_task_indeterminate_completed, advance = 1,visible = True)
                        live_display.refresh()
                    else:
                        pass
                    

                            
            print('>Programme has ended.')


def UID_generator(pydicom_obj):
    UID_def = f"{str(pydicom_obj[0x0010,0x0010].value)} ({str(pydicom_obj[0x0010,0x0020].value)})"
    return UID_def


def structure_referencer(structure_dcm_dict, 
                         dose_dcm_dict, 
                         plan_dcm_dict, 
                         OAR_list,
                         DIL_list,
                         Bx_list,
                         st_ref_list,
                         ds_ref,
                         pln_ref,
                         all_ref_key,
                         bx_sim_locations_list,
                         bx_sim_ref_identifier_str,
                         sim_bx_relative_to_list,
                         sim_bx_relative_to_struct_type,
                         fanova_sobol_indices_names_by_index
                         ):
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

            
            bpsy_ref = [{"ROI": x.ROIName, 
                         "Ref #": x.ROINumber, 
                         "Simulated bool": False,
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
                         "MC data: MC sim containment raw results dataframe": None, 
                         "MC data: compiled sim results": None, 
                         "MC data: mutual compiled sim results": None,
                         "MC data: tumor tissue probability": None,
                         "MC data: miss structure tissue probability": None,
                         "MC data: voxelized containment results dict": None, 
                         "MC data: voxelized containment results dict (dict of lists)": None, 
                         "MC data: bx to dose NN search objects list": None, 
                         "MC data: Dose NN child obj for each sampled bx pt list (nominal & all MC trials)": None,
                         "MC data: Dose vals for each sampled bx pt arr (nominal & all MC trials)": None, 
                         "MC data: Dose vals for each sampled bx pt arr (all MC trials)": None,
                         "MC data: Dose vals for each sampled bx pt arr (nominal)": None,
                         "MC data: Differential DVH dict": None,
                         "MC data: Cumulative DVH dict": None,
                         "MC data: dose volume metrics dict": None, 
                         "MC data: Dose statistics for each sampled bx pt list (mean, std, quantiles)": None, 
                         "MC data: Dose statistics (MLE) for each sampled bx pt list (mean, std)": None, 
                         "MC data: voxelized dose results list": None, 
                         "MC data: voxelized dose results dict (dict of lists)": None, 
                         "Output csv file paths dict": {}, 
                         "Output data frames": {},
                         "Output dicts for data frames": {},  
                         "KDtree": None, 
                         "Nearest neighbours objects": [], 
                         "Plot attributes": plot_attributes()
                         } for x in structure_item.StructureSetROISequence if any(i in x.ROIName for i in Bx_list)]    
            
            bpsy_ref_simulated = [{"ROI": "Bx_Tr_"+bx_sim_ref_identifier_str+" " + x.ROIName, 
                         "Ref #": bx_sim_ref_identifier_str +" "+ x.ROIName, 
                         "Simulated bool": True,
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
                         "MC data: MC sim containment raw results dataframe": None,
                         "MC data: compiled sim results": None,
                         "MC data: mutual compiled sim results": None, 
                         "MC data: tumor tissue probability": None,
                         "MC data: miss structure tissue probability": None,
                         "MC data: voxelized containment results dict": None, 
                         "MC data: voxelized containment results dict (dict of lists)": None, 
                         "MC data: bx to dose NN search objects list": None, 
                         "MC data: Dose NN child obj for each sampled bx pt list (nominal & all MC trials)": None,
                         "MC data: Dose vals for each sampled bx pt arr (nominal & all MC trials)": None,
                         "MC data: Dose vals for each sampled bx pt arr (all MC trials)": None,
                         "MC data: Dose vals for each sampled bx pt arr (nominal)": None,
                         "MC data: Differential DVH dict": None,
                         "MC data: Cumulative DVH dict": None,
                         "MC data: dose volume metrics dict": None,
                         "MC data: Dose statistics for each sampled bx pt list (mean, std, quantiles)": None, 
                         "MC data: Dose statistics (MLE) for each sampled bx pt list (mean, std)": None, 
                         "MC data: voxelized dose results list": None, 
                         "MC data: voxelized dose results dict (dict of lists)": None, 
                         "Output csv file paths dict": {}, 
                         "Output data frames": {},
                         "Output dicts for data frames": {}, 
                         "KDtree": None, 
                         "Nearest neighbours objects": [], 
                         "Plot attributes": plot_attributes()
                         } for x in structure_item.StructureSetROISequence if any(i in x.ROIName for i in sim_bx_relative_to_list)]
            
            bpsy_ref = bpsy_ref + bpsy_ref_simulated 


            all_ref = {"Multi-structure output data frames dict": {}}
            
            
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

            master_st_ds_ref_dict[UID] = {"Patient UID (generated)": str(UID),
                                            "Patient ID (from dicom)": str(structure_item[0x0010,0x0020].value),
                                            "Patient Name": str(structure_item[0x0010,0x0010].value),
                                            st_ref_list[0]: bpsy_ref, 
                                            st_ref_list[1]: OAR_ref, 
                                            st_ref_list[2]: DIL_ref,
                                            all_ref_key: all_ref,
                                            "Ready to plot data list": None
                                        }
            
            master_st_ds_info_dict[UID] = {"Patient UID (generated)": str(UID),
                                            "Patient ID (from dicom)": str(structure_item[0x0010,0x0020].value),
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
               "Num MC dose simulations and nominal": None, 
               "Num sample pts per BX core": None, 
               "BX sample pt lattice spacing": None,
               "Max of num MC simulations": None}
    
    master_st_ds_info_global_dict["Global"] = {"Num patients": global_num_patients, 
                                               "Num structures": global_total_num_structs, 
                                               "Num biopsies": global_num_biopsies, 
                                               "Num DILs": global_num_DIL, 
                                               "Num OARs": global_num_OAR, 
                                               "MC info": mc_info,
                                               "FANOVA: num variance vars": None,
                                               "FANOVA: sobol var names by index": fanova_sobol_indices_names_by_index
                                               }
    
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
    