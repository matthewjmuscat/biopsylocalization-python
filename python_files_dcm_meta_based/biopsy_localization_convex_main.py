import os 
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
import random
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
import csv_writers
import cuspatial
import geopandas
import biopsy_optimizer
from itertools import combinations
import biopsy_transporter
import machina_learning
import matplotlib.pyplot as plt
from collections import defaultdict
import lattice_reconstruction_tools
import MC_prepper_funcs 
import MC_simulator_MR
import dose_lattice_helper_funcs

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

    # prevents matplotlib plots from opening unless explicitely asked to with plt.show()
    plt.ioff()

    # Data removals dictionary (Specify patient and biopsy ids to remove from the dataset)
    # specify the patient IDs and the list of biopsy names to remove from the dataset
    data_removals_dict = {"189 (F2)": ["Bx_Tr LM1 blood"],
                            "192 (F2)": ["Bx_trk LM blood"]
                            }


    # The following could be user input, for now they are defined here, and used throughout 
    # the programme for generality
    data_folder_name = 'Data'
    input_data_folder_name = "Input data"
    #oaroi_contour_names = ['Prostate','Urethra','Rectum','Normal', 'CTV','random'] 
    """
    Consider prostate only for OARs!

    -- Also the first structure in the below list is the structure specified to plot probability of missing this structure!
    """
    ### Note that in these contour name lists, the first one is seen as the priority string during the selection process 
    # (see misc_tools.specific_structure_selector_dataframe_version for more details)
    # with the exception of biopsies and dils as they are excluded from structs_referenced_list_generalized_unique_structs
    prostate_contour_name = 'Prostate'
    oaroi_contour_names = [prostate_contour_name]
    structure_miss_probability_roi = oaroi_contour_names[0]
    biopsy_contour_names = ['Bx']
    dil_contour_names = ['DIL']
    rectum_contour_names = ['Rectum']
    urethra_contour_names = ['Urethra']
    ### IMPORTANT! I THINK FROM THIS POINT FORWARD I AM GOING TO HAVE EACH STRUCTURE HAVE THEIR OWN REFERENCE! IN FUTURE VERSIONS
    ### I SHOULD MIGRATE OAR_REF TO PROSTATE_REF! SEE THE LINE BELOW THAT DEFINES structs_referenced_list_generalized !!!    
    
    ## Allowable prefixes for recognizing different fractions from the patient id field
    fraction_prefixes = ['f', 'fraction', '']


         

    ### DEFAULT SIGMAS
    # FROM LITERATURE OF INTEROBSERVER VARIABILITY IN PROSTATE
    # "Comparison of prostate volume, shape, and contouring variability determined from preimplant magnetic resonance and transrectal ultrasound images" - Liu et al.
    # Took half of the length width height values from FIG 3.
    oar_default_sigma_X = 1 # default sigma in mm
    oar_default_sigma_Y = 1 # default sigma in mm
    oar_default_sigma_Z = 2 # default sigma in mm
    oar_default_sigma_X_list = [2.5,2.5] # default sigma in mm # 2.5 for contouring variability and 2.5 for MDA registration uncertainty, also consistent with literature
    oar_default_sigma_Y_list = [2.5,2.5] # default sigma in mm
    oar_default_sigma_Z_list = [2.5,2.5] # default sigma in mm

    # THIS SHOULD COME FROM MEAN MDA IN US TO US, THE OTHER COMPONENT COMES FROM MEAN VARIATION IN BIOPSY CENTROIDS AND IS CALCULATED IN PREPROCESSING
    biopsy_default_sigma_X = 1.5 # default sigma in mm
    biopsy_default_sigma_Y = 1.5 # default sigma in mm
    biopsy_default_sigma_Z = 1.5 # default sigma in mm
    biopsy_default_sigma_X_list = [2.5] # default sigma in mm  2.5 for MDA registration uncertainty, also consistent with literature
    biopsy_default_sigma_Y_list = [2.5] # default sigma in mm
    biopsy_default_sigma_Z_list = [2.5] # default sigma in mm

    # CALCULATE FROM MEAN MDA BETWEEN MRI/US
    dil_default_sigma_X = 2.5 # default sigma in mm
    dil_default_sigma_Y = 2.5 # default sigma in mm
    dil_default_sigma_Z = 2.5 # default sigma in mm
    dil_default_sigma_X_list = [2.5,2.5] # default sigma in mm # 2.5 for contouring variability and 2.5 for MDA registration uncertainty, also consistent with literature
    dil_default_sigma_Y_list = [2.5,2.5] # default sigma in mm
    dil_default_sigma_Z_list = [2.5,2.5] # default sigma in mm
    
    use_added_in_quad_errors_as = 'two sigma' # can be 'sigma' or 'two sigma', 'two sigma' will provide tighter uncertainty clouds 
    biopsy_variation_uncertainty_setting = "Per biopsy mean" # Can be "Per biopsy max", "Per biopsy mean", "Global mean" or "Default only" .... See function (uncertainty_file_preper_by_struct_type_dataframe_NEW) defined in uncertainty_file_writer
    # "Global mean" = the mean variation will be used between all patients across all patients
    # "Per biopsy max" = will automatically alter uncertainty file to include the max variation of the biopsy contours for each biopsy seperately in the sigma value for the biopsy uncertainty
    # "Per biopsy mean" = will automatically alter uncertainty file to include the mean variation of the biopsy contours for each biopsy seperately in the sigma value for the biopsy uncertainty
    # "Default only" = will only use the values provided by the biospy_default_list, presumably this would account only for registration uncertainty
    non_biopsy_variation_uncertainty_setting = "Default only" # At the moment, only "Default only" is supported
    
    
    uncertainty_folder_name = 'Uncertainty data'
    uncertainty_file_name = "uncertainties_file_auto_generated"
    uncertainty_file_extension = ".csv"
    spinner_type = 'moon' # other decent ones are 'point' and 'line' or 'line2'
    output_folder_name = 'Output data'
    preprocessed_data_folder_name = 'Preprocessed data'
    preprocessed_master_structure_ref_dict_for_export_name = 'master_structure_reference_dict'
    preprocessed_master_structure_info_dict_for_export_name = 'master_structure_info_dict'
    output_master_structure_ref_dict_for_export_name = 'master_structure_reference_dict_results'
    output_master_structure_info_dict_for_export_name = 'master_structure_info_dict_results'
    lower_bound_dose_value = None # can also set to None and will try to assign by pydicom_item[plan_ref]["Prescription doses dict"]["TARGET"]
    #lower_bound_dose_percent = 10
    lower_bound_dose_gradient_value = 0
    lower_bound_mr_adc_value = 500
    upper_bound_mr_adc_value = 900
    color_flattening_deg = 3
    color_flattening_deg_MR = 1 # 1 means it will not flatten
    interp_inter_slice_dist = 0.5
    interp_intra_slice_dist = 0.5 # user defined length scale for intraslice interpolation min distance between points. It is used in the interpolation_information_obj class
    interp_dist_caps = 0.25
    biopsy_radius = 0.5
    biopsy_needle_compartment_length = 19 # length in millimeters of the biopsy needle core compartment
    biopsy_fire_travel_distances = [15,22] # how far the needle tip travels from unfired to fired position, for the magnum bard there were two penetration depth settings
    biopsy_needle_tip_length = 6 # tip to compartment distance
    voxel_size_for_structure_volume_calc_bx = 0.1 # if set to 0 then it is calculated based on the maximum pairwise distance of the structure
    voxel_size_for_structure_volume_calc_non_bx = 1
    voxel_size_for_structure_dimension_calc = 0.1 # this one is of calculating the length dimension of each structure at the position of the centroid!
    factor_for_voxel_size = 100 # only relevant if one of the above variables (voxel_size_for_structure_volume_calc_XXX) is equal to 0!
    
    # A note on radius vs knn for normal estimation
    """
    Radius or K-neighbors: When using methods like k-nearest neighbors or radius search for normal estimation, 
    the choice of the radius or the number of neighbors can affect accuracy. A smaller radius or fewer neighbors 
    might capture finer details but could also be sensitive to noise, while a larger radius or more neighbors might 
    provide a smoother result but could miss smaller features.
    """
    # It may be best at this point to be consistent at 1mm
    radius_for_normals_estimation = 1 # Making this larger produces better normals on a sphere, but too large may have undesired effects??
    radius_for_curvature_estimation = 1 # This is the radius for determination of tangent plane normal, making this large tends to uniformize the pointwise curvature values
    max_nn_for_normals_estimation = 30


    # MC parameters
    simulate_uniform_bx_shifts_due_to_bx_needle_compartment = True
    #num_sample_pts_per_bx_input = 250 # uncommenting this line will do nothing, this line is deprecated in favour of constant cubic lattice spacing
    bx_sample_pts_lattice_spacing = 0.5
    num_MC_containment_simulations_input = 10000
    num_MC_dose_simulations_input = 10000
    num_MC_MR_simulations_input = num_MC_dose_simulations_input ### IMPORTANT, THIS NUMBER IS ALSO USED FOR MR IMAGING SIMULATIONS since we want to randomly sample from trials for our experiment, so them being the same amount will allow for this more succinctly. Since the way the localization is performed is the same for each (Ie. NN KDTree) these numbers should affect performance similarly
    biopsy_z_voxel_length = 1 #voxelize biopsy core every 1 mm along core
    num_dose_calc_NN = 4 # This determines the number of nearest neighbours to the dosimetric lattice for each biopsy sampled point
    num_mr_calc_NN = 4 # This determines the number of nearest neighbours to the MR lattice for each biopsy sampled point  
    idw_power = 1 # This determines the power of the inverse distance weighting (interpolation) for the NN dose search of the dose lattice!
    tissue_length_above_probability_threshold_list = [0.95,0.75,0.5,0.25]
    n_bootstraps_for_tissue_length_above_threshold = 1000
    raw_data_mc_dosimetry_dump_bool = False # ALSO SLOWS EVERYTHING DOWN! WARNING: MAY TAKE UP HUNDREDS OF GIGS OF DISK SPACE! USE WITH CAUTION! IF WANT TO REDUCE SIZE, REDUCE NUMBER OF DOSE AND CONTAINMENT SIMULATIONS! If True, will output the raw results data of the mc sim for dose tests! 
    raw_data_mc_containment_dump_bool = False  # ALSO SLOWS EVERYTHING DOWN! WARNING: MAY TAKE UP HUNDREDS OF GIGS OF DISK SPACE! USE WITH CAUTION! IF WANT TO REDUCE SIZE, REDUCE NUMBER OF DOSE AND CONTAINMENT SIMULATIONS! If True, will output the raw results data of the mc sim for containment tests!
    raw_data_mc_MR_dump_bool = False # Haven't actually set this one to True yet but likely takes huge amount of space like the two above!

    num_dose_NN_to_show_for_animation_plotting = 100
    num_bootstraps_for_regression_plots_input = 15
    pio.templates.default = "plotly_white"
    svg_image_scale = 1 # setting this value to something not equal to 1 produces misaligned plots with multiple traces!
    NPKR_bandwidth = 0.5
    svg_image_height = 1080
    svg_image_width = 1920
    dpi_for_seaborn_plots = 100
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
    
   
    # for optimal dil sampling location
    voxel_size_for_dil_optimizer_grid = 1
    num_normal_dist_points_for_biopsy_optimizer = 10000
    normal_dist_sigma_factor_biopsy_optimizer = 1/4
    plot_each_normal_dist_containment_result_bool = False
    plot_optimization_point_lattice_bool = False
    show_optimization_point_bool = False
    display_optimization_contour_plots_bool = False
    optimal_normal_dist_option = 'dil dimension driven' # can be 'biopsy_and_dil_sigmas' or 'dil dimension driven', note that the biopsy_and_dil_sigmas option adds all sigmas in quadrature and then uses this value as TWO sigma. Note that the dil deimnsion driven option uses the dimension of the respective dil at the position of the dil centroid in each direction as TWO sigma
    # these multipliers provide a lengthening or stretching of the normal dist to bias a certain dimension as relatively more important 
    bias_LR_multiplier = 1
    bias_AP_multiplier = 1
    bias_SI_multiplier = 1.5 
    


    # for simulated biopsies
    centroid_dil_sim_key = 'Centroid DIL'
    optimal_dil_sim_key = 'Optimal DIL'
    bx_sim_locations_dict = {centroid_dil_sim_key:
                              {"Create": True,
                              "Relative to": 'DIL',
                              "Identifier string": 'sim_centroid_dil'}
                              ,   
                            optimal_dil_sim_key:
                              {"Create": True,
                              "Relative to": 'DIL',
                              "Identifier string": 'sim_optimal_dil'}
                            }
    simulated_biopsy_length_method = 'real normal' # can be 'full' (ie. 19mm), 'real normal' (samples from a normal distribution with mu=real_mean, std = real_std) or 'real mean' (all sim biopsy lengths are equal to real_mean)
    color_discrete_map_by_sim_type = {'Real': 'rgba(0, 92, 171, 1)', centroid_dil_sim_key: 'rgba(227, 27, 35,1)', optimal_dil_sim_key: 'rgba(0, 0, 0,1)'}
    biopsy_pcd_colors_dict = {'Real': np.array([0.5, 0.0, 0.5]), centroid_dil_sim_key: np.array([1.0, 0.55, 0.0]), optimal_dil_sim_key: np.array([0.0, 0.8, 0.6])} # real: purple, centroid: deep orange, optimal: light teal

    #bx_sim_locations = ['centroid'] # change to empty list if dont want to create any simulated biopsies. Also the code at the moment only supports creating centroid simulated biopsies, ie. change to list containing string 'centroid'.
    #bx_sim_ref_identifier = "sim"
    #simulate_biopsies_relative_to = ['DIL'] # can include elements in the list such as "DIL" or "Prostate"...


    differential_dvh_resolution = 100 # the number of bins
    cumulative_dvh_resolution = 100 # the larger the number the more resolution the cDVH calculations will have
    display_dvh_as = ['counts','percent', 'volume'] # can be 'counts', 'percent', 'volume'
    num_cumulative_dvh_plots_to_show = 25
    num_differential_dvh_plots_to_show = 25
    v_percent_DVH_to_calc_list = [100,125,150,200,300] # These are V_x, note that these values should be given as percentages relative to CTV, this is pulled automatically from plan ref, the output is a percent volume 
    d_x_DVH_to_calc_list = [2,50,98] # These are D_x, x values should be given as percentages of the total volume (ie. between 0,100). the output is a dose value
    volume_DVH_quantiles_to_calculate = [5,25,50,75,95]

    #fanova
    num_FANOVA_containment_simulations_input = 0 # must be a power of two for the scipy function to work, 2^10 is good
    num_FANOVA_dose_simulations_input = 0
    show_fanova_containment_demonstration_plots = False
    plot_cupy_fanova_containment_distribution_results = False
    plot_cupy_containment_distribution_fanova_results = False # nice because it shows all trials at once
    fanova_plot_uniform_shifts_to_check_plotly = False
    num_sobol_bootstraps = 100
    sobol_indices_bootstrap_conf_interval = 0.95
    show_NN_FANOVA_dose_demonstration_plots = False

    # patient sample cohort analyzer
    only_perform_patient_analyser = False
    perform_patient_sample_analyser_at_end = True
    box_plot_points_option = 'outliers'
    notch_option = False
    boxmean_option = True # can be 'sd' or True

    
    
    
    
    ### PLOTS TO SHOW:
    
    # Dosimetry
    show_NN_dose_demonstration_plots = False # this shows one trial at a time!!!
    show_3d_dose_renderings = False
    show_3d_dose_renderings_thresholded = False
    show_NN_dose_demonstration_plots_all_trials_at_once = False # nice because shows all trials at once

    # Tissue class
    show_containment_demonstration_plots = False # this shows one trial at a time!!!
    plot_cupy_containment_distribution_results = False # nice because it shows all trials at once
    plot_nearest_neighbour_surface_boundary_demonstration = False # you see one trial at a time
    plot_relative_structure_centroid_demonstration = False # you see one trial at a time

    # MRs
    show_3d_mr_adc_renderings = False
    show_3d_mr_adc_renderings_thresholded = False
    show_NN_mr_adc_demonstration_plots = False # this shows one trial at a time!!
    show_NN_mr_adc_demonstration_plots_all_trials_at_once = False # nice because shows all trials at once
    
    # Combined
    show_processed_3d_datasets_renderings = False
    show_processed_3d_datasets_renderings_plotly = False

    # Misc
    show_reconstructed_biopsy_in_biopsy_coord_sys_tr_and_rot = False
    plot_uniform_shifts_to_check_plotly = False # if this is true, will produce many plots if num_simulations is high!
    plot_translation_vectors_pointclouds = False
    plot_shifted_biopsies = False
    plot_volume_calculation_containment_result_bool = False
    display_curvature_bool = False
    display_structure_surface_mesh_bool = False
    plot_binary_mask_bool = False
    plot_guidance_map_transducer_plane_open3d_structure_set_complete_demonstration_bool = False
    show_equivalent_ellipsoid_from_pca_bool = False
    plot_dimension_calculation_containment_result_bool = False
    display_pca_fit_variation_for_biopsies_bool = False

    ###







    ### Final production plots to create:
    
    plot_immediately_after_simulation = True
    regression_type_input = 0 # LOWESS = 1 or True, NPKR = 0 or False, this concerns the type of non parametric kernel regression that is performed
    global_regression_input = False # True or False bool type, this concerns whether a regression is performed on the axial dose distribution scatter plot containing all the data points of dose from all trials for each point 

    num_z_vals_to_evaluate_for_regression_plots = 1000
    tissue_class_probability_plot_type_list = ['with_errors','']
    production_plots_input_dictionary = {"Sampled translation vector magnitudes box plots": \
                                            {"Plot bool": True, #
                                             "Plot name": " - sampling-box_plot-sampled_translations_magnitudes_all_trials",
                                             "Plot color": 'rgba(0, 92, 171, 1)'
                                             }, 
                                        "Biopsy positions relative to target DILs density plots": \
                                            {"Plot bool": True, # No code behind this method yet
                                             "Plot name": " - biopsy_positions_relative_to_target_DILs_density_plots",
                                             "Plot color": 'rgba(0, 92, 171, 1)'
                                             }, 
                                        "Guidance maps":\
                                            {"Plot bool": False, 
                                             "Plot name": "guidance maps",
                                             "Plot color": 'rgba(0, 92, 171, 1)'
                                             },
                                        "Guidance maps with actual cores":\
                                            {"Plot bool": False, #
                                             "Plot name": "guidance maps with actual cores",
                                             "Plot color": 'rgba(0, 92, 171, 1)'
                                             },      
                                        "Axial dose distribution all trials and global regression": \
                                            {"Plot bool": False, 
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
                                        "Axial dose distribution quantiles regression plot matplotlib": \
                                            {"Plot bool": True, #
                                             "Plot name": " - dose-regression-quantiles_axial_dose_distribution_matplotlib",
                                             "Num rand trials to show": 10
                                             },
                                        "Axial dose GRADIENT distribution quantiles regression plot matplotlib": \
                                            {"Plot bool": True, #
                                             "Plot name": " - dose-regression-quantiles_axial_dose_gradient_distribution_matplotlib",
                                             "Num rand trials to show": 10
                                             },
                                        "Axial dose distribution quantiles regression plot": \
                                            {"Plot bool": False, #
                                             "Plot name": " - dose-regression-quantiles_axial_dose_distribution"
                                             },
                                        "Axial dose distribution voxelized box plot": \
                                            {"Plot bool": False, ### DEPRECATED DO NOT USE!
                                             "Plot name": " - dose-box_plot-voxelized_axial_dose_distribution",
                                             "Plot color": 'rgba(0, 92, 171, 1)'
                                             },
                                        "Axial dose voxelized ridgeline plot": \
                                            {"Plot bool": True, 
                                             "Plot name": " - dose_voxelized_ridgeline",
                                             },
                                        "Axial dose and tissue colored voxelized ridgeline plot": \
                                            {"Plot bool": True, 
                                             "Plot name": " - dose_and_dil_tissue_colored_voxelized_ridgeline",
                                             },
                                        "Axial dose distribution voxelized violin plot": \
                                            {"Plot bool": False, ### DEPRECATED DO NOT USE!
                                             "Plot name": " - dose-violin_plot-voxelized_axial_dose_distribution",
                                             "Plot color": 'rgba(0, 92, 171, 1)'
                                             },
                                        "Differential DVH showing N trials plot": \
                                            {"Plot bool": False, 
                                             "Plot name": f' - dose-DVH-differential_dvh_showing_{int(num_differential_dvh_plots_to_show)}_trials'
                                             },
                                        "Differential DVH dose binned all trials box plot": \
                                            {"Plot bool": False, #
                                             "Plot name": ' - dose-DVH-differential_DVH_binned_box_plot',
                                             "Box plot color": 'rgba(0, 92, 171, 1)',
                                             "Nominal point color": 'rgba(227, 27, 35, 1)'
                                             },
                                        "Differential DVH dose quantiles plot seaborn": \
                                            {"Plot bool": True,  #
                                             "Plot name": ' - dose-differential_DVH_quantiles_plot'
                                             },
                                        "Cumulative DVH quantile regression seaborn": \
                                            {"Plot bool": True, #
                                             "Plot name": ' - dose-cumulative_DVH_quantiles_plot'
                                             },
                                        "Cumulative DVH showing N trials plot": \
                                            {"Plot bool": False, 
                                             "Plot name": f' - dose-DVH-cumulative_dvh_showing_{int(num_cumulative_dvh_plots_to_show)}_trials'
                                             },
                                        "Cumulative DVH quantile regression all trials plot regression only": \
                                            {"Plot bool": False, 
                                             "Plot name": ' - dose-DVH-cumulative_DVH_regressions_quantiles_regression_only'
                                             },
                                        "Cumulative DVH quantile regression all trials plot colorwash": \
                                            {"Plot bool": True, #
                                             "Plot name": ' - dose-DVH-cumulative_DVH_regressions_quantiles_colorwash'
                                             },
                                        "Cohort - Dosimetry boxplots all biopsies": \
                                            {"Plot bool": True, #
                                             "Plot name": 'Cohort all biopsies global dosimetry boxplot'
                                             },
                                        "Tissue classification scatter and regression probabilities all trials plot": \
                                            {"Plot bool": False, #
                                             "Plot name": ' - tissue_class-regression-probabilities'
                                             },
                                        "Tissue classification mutual probabilities plot": \
                                            {"Plot bool": False, #
                                             "Plot name": ' - tissue_class_mutual-regression-probabilities',
                                             "Structure miss ROI": structure_miss_probability_roi
                                             },
                                        "Tissue classification sum-to-one plot": \
                                            {"Plot bool": False, #
                                             "Plot name": ' - tissue_class_sum-to-one_binom_regression_probabilities',
                                            },
                                        "Tissue classification sum-to-one nominal plot": \
                                            {"Plot bool": False, #
                                             "Plot name": ' - tissue_class_sum-to-one_nominal_chart',
                                            },
                                        "Axial tissue class voxelized ridgeline plot": \
                                            {"Plot bool": False, 
                                             "Plot name": ' - tissue_class_ridgeline',
                                             "Structure miss ROI": structure_miss_probability_roi
                                             },
                                        "Axial MR distribution quantiles regression plot matplotlib": \
                                            {"Plot bool": True, # 
                                             "Plot name": " - mr-regression-quantiles_axial_distribution_matplotlib"
                                             },
                                        "Cohort - Sampled translation vector magnitudes box plots": \
                                            {"Plot bool": True, 
                                             "Plot name": " - Cohort - sampled_translations_vecs_and_mags_boxplot"
                                             },
                                        "Cohort - Prostate biopsies heatmap": \
                                            {"Plot bool": True, 
                                             "Plot name": " - Cohort - biopsies_prostate_heatmap"
                                             },
                                        "Cohort - Prostate DILs heatmap": \
                                            {"Plot bool": True, 
                                             "Plot name": " - Cohort - DILs_prostate_heatmap"
                                             },   
                                        "Cohort - Scatter plot matrix targeting accuracy": \
                                            {"Plot bool": True, 
                                             "Plot name": ' - scatter_plot_matrix_targeting_accuracy'
                                             },
                                        "Cohort - Scatter plot 2d gaussian transverse accuracy": \
                                            {"Plot bool": True, 
                                             "Plot name": ' - 2d_gaussian_transverse_accuracy'
                                             },
                                        "Cohort - Scatter plot 2d gaussian sagittal accuracy": \
                                            {"Plot bool": True, 
                                             "Plot name": ' - 2d_gaussian_sagittal_accuracy'
                                             },
                                        "Cohort - Tissue class global score by biopsy type": \
                                            {"Plot bool": True, 
                                             "Plot name": ' cohort - tissue_class_global_by_bx_type',
                                             "Structure miss ROI": structure_miss_probability_roi
                                             },
                                        "Cohort - Tissue class global score sum-to-one box plots": \
                                            {"Plot bool": True, 
                                             "Plot name": ' cohort - tissue_class_sum-to-one_global_scores_by_bx_type'
                                             },
                                        "Cohort - Tissue class sum-to-one spatial ridgeline plots": \
                                            {"Plot bool": True, 
                                             "Plot name": ' cohort - tissue_class_sum-to-one_ridgeline'
                                             }, 
                                        "Cohort - Tissue class sum-to-one spatial nominal heatmap": \
                                            {"Plot bool": True, 
                                             "Plot name": ' cohort - tissue_class_sum-to-one_nominal_heatmap'
                                             },
                                        "Cohort - All biopsy voxels histogram sum-to-one binom est by tissue class": \
                                            {"Plot bool": True, 
                                             "Plot name": ' cohort - tissue_class_sum-to-one_all_biopsy_voxels_histogram_by_tissue_class'
                                             },
                                        "Cohort - Tissue volume by biopsy type": \
                                            {"Plot bool": True, 
                                             "Plot name": ' cohort - tissue_volume_threshold_by_bx_type',
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
                                        "Tissue classification Sobol indices per biopsy plot": \
                                            {"Plot bool dict": {"FO": True,
                                                                "TO": True
                                                                }, 
                                             "Plot name dict": {"FO": 'FANOVA_tissue_class_first_order_sobol',
                                                                "TO": 'FANOVA_tissue_class_total_order_sobol'
                                                                },
                                             "Structure miss ROI": structure_miss_probability_roi,
                                             },
                                        "Dosimetry Sobol indices global plot": \
                                            {"Plot bool dict": {"Global FO": True,
                                                                "Global FO by function output": True,
                                                                "Global TO": True,
                                                                "Global TO by function output": True,
                                                                "Global FO, sim only": True,
                                                                "Global FO, sim only by function output": True,
                                                                "Global FO, non-sim only": True,
                                                                "Global FO, non-sim only by function output": True,
                                                                ###
                                                                "Global FO, sim_only, by function output, separate": True,
                                                                "Global FO, non_sim_only, by function output, separate": True,
                                                                "Global FO, sim_and_non_sim, by function output, separate": True,
                                                                "Global TO, sim_only, by function output, separate": True,
                                                                "Global TO, non_sim_only, by function output, separate": True,
                                                                "Global TO, sim_and_non_sim, by function output, separate": True,
                                                                }, 
                                             "Plot name dict": {"Global FO": 'FANOVA_global_dose_first_order_sobol',
                                                                "Global FO by function output": 'FANOVA_global_dose_first_order_sobol_by_function_output',
                                                                "Global TO": 'FANOVA_global_dose_total_order_sobol',
                                                                "Global TO by function output": 'FANOVA_global_dose_total_order_sobol_by_function_output',
                                                                "Global FO, sim only": 'FANOVA_global_dose_first_order_sim_only_sobol',
                                                                "Global FO, sim only by function output": 'FANOVA_global_dose_first_order_sim_only_by_function_output_sobol',
                                                                "Global FO, non-sim only": 'FANOVA_global_dose_first_order_non-sim_only_sobol',
                                                                "Global FO, non-sim only by function output": 'FANOVA_global_dose_first_order_non-sim_only_by_function_output_sobol',
                                                                ###
                                                                "Global FO, sim_only, by function output, separate": "FANOVA_dose_FO_sim_by_function_output_separated",
                                                                "Global FO, non_sim_only, by function output, separate": "FANOVA_dose_FO_nonsim_by_function_output_separated",
                                                                "Global FO, sim_and_non_sim, by function output, separate": "FANOVA_dose_FO_sim_and_non_sim_by_function_output_separated",
                                                                "Global TO, sim_only, by function output, separate": "FANOVA_dose_TO_sim_by_function_output_separated",
                                                                "Global TO, non_sim_only, by function output, separate": "FANOVA_dose_TO_nonsim_by_function_output_separated",
                                                                "Global TO, sim_and_non_sim, by function output, separate": "FANOVA_dose_TO_sim_and_non_sim_by_function_output_separated",
                                                                },
                                             "Box plot points option": 'all' # can be 'all', 'outliers' or False
                                             },
                                        "Dosimetry Sobol indices per biopsy plot": \
                                            {"Plot bool dict": {"FO": True,
                                                                "TO": True
                                                                }, 
                                             "Plot name dict": {"FO": 'FANOVA_dosimetry_first_order_sobol',
                                                                "TO": 'FANOVA_dosimetry_total_order_sobol'
                                                                },
                                             "Structure miss ROI": structure_miss_probability_roi,
                                             }
                                        }
    
    

    # other parameters
    modify_generated_uncertainty_template = False # if True, the algorithm wont be able to run from start to finish without an interupt, allowing one to modify the uncertainty file
    write_containment_to_file_ans = True # If True, this generates and saves to file a csv file of the containment simulation
    write_dose_to_file_ans = True # If True, this generates and saves to file a csv file of the dose simulation
    export_pickled_preprocessed_data = False # If True, this exports a pickled version of master_structure_reference_dict and master_structure_info_dict
    skip_preprocessing = False # If True, you will be asked to specify the locations of master_structure_info_dict and master_structure_reference_dict
    write_sobol_dose_data_to_file = True
    write_sobol_containment_data_to_file = True
    write_preprocessing_data_to_file = True
    write_cohort_data_to_file = True

    cupy_array_upper_limit_NxN_size_input = 1e9 ### THIS IS A NUMBER THAT IS LIMITED BY YOUR GPU MEMORY! APPROXIMATELY 1e9 IS A GOOD COMPROMISE FOR A 3080 TI WITH 12GB VRAM!
    numpy_array_upper_limit_NxN_size_input = 1e9 ### THIS IS A NUMBER THAT IS LIMITED BY YOUR RAM MEMORY! APPROXIMATELY 1e9 IS A GOOD COMPROMISE FOR 32GB RAM!
    nearest_zslice_vals_and_indices_cupy_generic_max_size = 5e7 # 5e7 was stable
    nearest_zslice_vals_and_indices_numpy_generic_max_size = 1e9

    # for dataframe builder
    cancer_tissue_label = 'DIL'
    miss_structure_complement_label = structure_miss_probability_roi + ' complement'
    default_exterior_tissue = 'Periprostatic' # For tissue class stuff! Basically dictates what to call tissue that doesnt lie in any defined structure!

    # non-user changeable variables, but need to be initiatied:
    all_ref_key = "All ref"
    bx_ref = "Bx ref"
    oar_ref = "OAR ref"
    dil_ref = "DIL ref"
    rectum_ref_key = "Rectum ref"
    urethra_ref_key = "Urethra ref"
    # DO NOT CHANGE THE ORDER OF THE KEYS IN THE BELOW DICTIONARY!!!! 
    structs_referenced_dict = { bx_ref: {"Contour names": biopsy_contour_names, 
                                        "Default sigma X": biopsy_default_sigma_X_list,
                                        "Default sigma Y": biopsy_default_sigma_Y_list,
                                        "Default sigma Z": biopsy_default_sigma_Z_list,
                                        'Test tissue class': None, # should always be None
                                        'Tissue heirarchy': None, # should always be None
                                        'Tissue class name': None, # Not used for anything as of yet..
                                        'PCD color dict': biopsy_pcd_colors_dict
                                        }, 
                                oar_ref: {"Contour names": oaroi_contour_names,
                                          "Default sigma X": oar_default_sigma_X_list,
                                          "Default sigma Y": oar_default_sigma_Y_list,
                                          "Default sigma Z": oar_default_sigma_Z_list,
                                          'Test tissue class': True,
                                          'Tissue heirarchy': 3,
                                          'Tissue class name': 'Prostatic', 
                                          'PCD color': np.array([0.86, 0.08, 0.24]) # crimson
                                          }, 
                                dil_ref: {"Contour names": dil_contour_names,
                                          "Default sigma X": dil_default_sigma_X_list,
                                          "Default sigma Y": dil_default_sigma_Y_list,
                                          "Default sigma Z": dil_default_sigma_Z_list,
                                          'Test tissue class': True,
                                          'Tissue heirarchy': 0,
                                          'Tissue class name': cancer_tissue_label,
                                          'PCD color': np.array([0.13, 0.55, 0.13]) # forest green
                                          },
                                rectum_ref_key: {"Contour names": rectum_contour_names,
                                          "Default sigma X": oar_default_sigma_X_list,
                                          "Default sigma Y": oar_default_sigma_Y_list,
                                          "Default sigma Z": oar_default_sigma_Z_list,
                                          'Test tissue class': True,
                                          'Tissue heirarchy': 2,
                                          'Tissue class name': 'Rectal',
                                          'PCD color': np.array([1.0, 0.84, 0.0]) # gold
                                          },
                                urethra_ref_key: {"Contour names": urethra_contour_names,
                                          "Default sigma X": oar_default_sigma_X_list,
                                          "Default sigma Y": oar_default_sigma_Y_list,
                                          "Default sigma Z": oar_default_sigma_Z_list,
                                          'Test tissue class': True,
                                          'Tissue heirarchy': 1,
                                          'Tissue class name': 'Urethral',
                                          'PCD color': np.array([0.0, 0.75, 1.0]) # sky blue
                                          } 
                                }
    #structs_referenced_list = list(structs_referenced_dict.keys()) # note that Bx ref has to be the first entry for other parts of the code to work! In fact the ordering of all entries must be maintained. 1. BX, 2. OAR, 3. DIL
    structs_referenced_list = [key for key, value in structs_referenced_dict.items() if value.get('Test tissue class', False)]
    structs_referenced_list.insert(0,bx_ref) # this inserts bx_ref to the beginning of the list!
    

    ### IMPORTANT
    # this is a generalized version of structs referenced list, structs referenced list is the list that is referenced 
    # for the main tissue containement testing pipeline. The generalized version contains references that are not 
    # necessarily tested in the containment testing pipeline
    ### THE ORDER OF THE ENTRIES IN STRUCTS REFERENCED LIST IS IMPORTANT, BUT IT SHOULDNT BE! DONT RELY ON ORDERING WHEN POSSIBLE
    # the idea for the future is that you should be able to add entries to structs referenced dict for each structure you want to test
    # tissue class against by changing Test tissue class to True. but this should be done very carefully because the ordering of the references in structs_referenced_list matter,
    # the way I built the code was not thought out in this way, it depends on the ordering but really it shouldnt.
    structs_referenced_list_generalized = list(structs_referenced_dict.keys())
    
    # structs_referenced_list_generalized_unique_structs represents structure types that have a unique structure, ie. there is only one 
    structs_referenced_list_generalized_unique_structs = copy.deepcopy(structs_referenced_list_generalized)
    structs_referenced_list_generalized_unique_structs.remove(bx_ref)
    structs_referenced_list_generalized_unique_structs.remove(dil_ref)


    dose_ref = "Dose ref"
    plan_ref = "Plan ref"
    mr_adc_ref = "MR ADC ref"
    mr_t2_ref = "MR T2 ref"
    us_ref = "US ref"
    num_simulated_bxs_to_create = sum([x["Create"] for x in bx_sim_locations_dict.values()])
    #num_simulated_bxs_to_create = len(bx_sim_locations)
    #if num_simulated_bxs_to_create == 0:
    #    simulate_biopsies_relative_to = []
    # note below two lines are necessary since we have plot bool dict instead of plot bool for this entry, this will need to be done for any entries that have a plot bool dict 
    production_plots_input_dictionary["Tissue classification Sobol indices global plot"]["Plot bool"] = any(production_plots_input_dictionary["Tissue classification Sobol indices global plot"]["Plot bool dict"].values())
    production_plots_input_dictionary["Dosimetry Sobol indices global plot"]["Plot bool"] = any(production_plots_input_dictionary["Dosimetry Sobol indices global plot"]["Plot bool dict"].values())
    production_plots_input_dictionary["Tissue classification Sobol indices per biopsy plot"]["Plot bool"] = any(production_plots_input_dictionary["Tissue classification Sobol indices per biopsy plot"]["Plot bool dict"].values())
    production_plots_input_dictionary["Dosimetry Sobol indices per biopsy plot"]["Plot bool"] = any(production_plots_input_dictionary["Dosimetry Sobol indices per biopsy plot"]["Plot bool dict"].values())
    create_at_least_one_production_plot = any([x["Plot bool"] for x in production_plots_input_dictionary.values()]) # will produce True if at least one plot bool in the production_plots_input_dictionary is true, otherwise will be false if all are false 
    if simulate_uniform_bx_shifts_due_to_bx_needle_compartment == True:
        fanova_sobol_indices_names_by_index = ['X', 'Y', 'Z', 'T'] # the order is important!
    else:
        fanova_sobol_indices_names_by_index = ['X', 'Y', 'Z'] # the order is important!
    
    # initialize perform mc sim based on other parameters
    perform_mc_dose_sim = bool(num_MC_dose_simulations_input)
    perform_mc_containment_sim = bool(num_MC_containment_simulations_input)
    perform_mc_mr_sim = bool(num_MC_MR_simulations_input)
    perform_MC_sim = perform_mc_containment_sim or perform_mc_dose_sim or perform_mc_mr_sim


    # initialize performed_fanova variable based on perform_dose and containment fanovas
    perform_dose_fanova = bool(num_FANOVA_dose_simulations_input)
    perform_containment_fanova = bool(num_FANOVA_containment_simulations_input)
    perform_fanova = perform_containment_fanova or perform_dose_fanova

    # create a dict for cohort data and dataframes
    mr_global_multi_structure_output_dataframe_str = "Global MR ADC statistics"
    mr_global_by_voxel_multi_structure_output_dataframe_str = "Global by voxel MR ADC statistics"

    master_cohort_patient_data_and_dataframes = {"Data": {"Cohort: Random forest tumor morphology results": None
                                                          },
                                                 "Dataframes": {"Uncertainties dataframe (unedited)": None,
                                                                "Uncertainties dataframe (final)": None,
                                                                "Cohort: Nearest DILs to each biopsy": None,
                                                                "Cohort: Biopsy basic spatial features dataframe": None,
                                                                "Cohort: 3D radiomic features all OAR and DIL structures": None,
                                                                "Cohort: All MC structure shift vectors": None,
                                                                "Cohort: structure specific mc results": None,
                                                                "Cohort: sum-to-one mc results": None,
                                                                "Cohort: global sum-to-one mc results": None,
                                                                "Cohort: mutual tissue class mc results": None,
                                                                "Cohort: tissue class global scores (tissue type)": None,
                                                                "Cohort: tissue class global scores (structure)": None,
                                                                "Cohort: tissue volume above threshold": None,
                                                                "Cohort: Entire point-wise binom est distribution": None,
                                                                "Cohort: DIL global tissue scores and DIL features": None,
                                                                "Cohort: Entire point-wise dose distribution": None,
                                                                "Cohort: Bx DVH metrics": None,
                                                                "Cohort: Bx DVH metrics (generalized)": None,
                                                                "Cohort: Bx global info dataframe": None,
                                                                "Cohort: "+ mr_global_multi_structure_output_dataframe_str: None,
                                                                "Cohort: "+ mr_global_by_voxel_multi_structure_output_dataframe_str: None
                                                                }
                                                 }


    cpu_count = os.cpu_count()
    with multiprocess.Pool(cpu_count) as parallel_pool:

        #st = time.time()

        progress_group_info_list = rich_preambles.get_progress_all(spinner_type)
        completed_progress, completed_sections_progress, patients_progress, structures_progress, biopsies_progress, MC_trial_progress, indeterminate_progress_main, indeterminate_progress_sub, progress_group = progress_group_info_list

        rich_layout = rich_preambles.make_layout()

        important_info = rich_preambles.info_output()
        app_header = rich_preambles.Header()
        app_footer = rich_preambles.Footer(algo_global_start, stopwatch)
        completed_sections_manager = rich_preambles.CompletedSectionsManager(completed_sections_progress)

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

                live_display.stop()
                output_csvs_folder = pathlib.Path(fd.askdirectory(title='Open output CSVs folder', initialdir=output_dir))
                #live_display.start()
                all_patient_sub_dirs = [x for x in output_csvs_folder.iterdir() if x.is_dir()]
                
                
                # cohort tissue class 
                num_actual_biopsies, num_sim_biopsies, cohort_containment_dataframe = dataframe_builders.containment_global_scores_all_patients_dataframe_builder(all_patient_sub_dirs)
                # drop complement data
                cohort_containment_dataframe = cohort_containment_dataframe[cohort_containment_dataframe.columns.drop(list(cohort_containment_dataframe.filter(regex='complement')))]
                

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


                # cohort tissue length by threshold probability
                num_actual_biopsies, num_sim_biopsies, cohort_tissue_length_dataframe = dataframe_builders.tissue_length_by_threshold_all_patients_dataframe_builder(all_patient_sub_dirs)

                


                # Make cohort output directories
                cohort_figures_output_dir_name = 'Cohort figures'
                tissue_length_output_dir_name = 'Tissue length'
                cohort_output_figures_dir = output_csvs_folder.parents[0].joinpath(cohort_figures_output_dir_name)
                cohort_output_figures_dir.mkdir(parents=False, exist_ok=True)
                tissue_length_cohort_output_figures_dir = cohort_output_figures_dir.joinpath(tissue_length_output_dir_name)
                tissue_length_cohort_output_figures_dir.mkdir(parents=False, exist_ok=True)


                # box plot tissue length 
                tissue_length_box_general_plot_name_string = 'Patient_cohort_tissue_length_box_plot'
                production_plots.production_plot_tissue_length_box_plots_patient_cohort(cohort_tissue_length_dataframe,
                                    num_actual_biopsies,
                                    num_sim_biopsies,
                                    svg_image_scale,
                                    svg_image_width,
                                    svg_image_height,
                                    tissue_length_box_general_plot_name_string,
                                    tissue_length_cohort_output_figures_dir,
                                    box_plot_points_option,
                                    notch_option,
                                    boxmean_option
                                    )
                

                # calculate cumulative histogram for each threshold
                cdf_by_threshold_dict = {}
                for threshold in cohort_tissue_length_dataframe["Probability threshold"].unique(): 
                    cdf_dict_sp_threshold = dataframe_builders.cumulative_histogram_for_tissue_length_dataframe_builder(cohort_tissue_length_dataframe,
                                                                                                threshold)
                    cdf_by_threshold_dict[threshold] = cdf_dict_sp_threshold

                # distribution tissue length figs
                fit_parameters_by_threshold_dict = {}
                for threshold in cohort_tissue_length_dataframe["Probability threshold"].unique(): 
                    tissue_length_general_plot_name_string = 'Patient_cohort_tissue_length_distribution_plot_'+str(threshold)
                    cdf_sp_threshold_dict = cdf_by_threshold_dict[threshold]
                    fit_parameters_sim_dict, fit_parameters_actual_dict = production_plots.production_plot_tissue_length_distribution_patient_cohort(cohort_tissue_length_dataframe,
                                        num_actual_biopsies,
                                        num_sim_biopsies,
                                        svg_image_scale,
                                        svg_image_width,
                                        svg_image_height,
                                        tissue_length_general_plot_name_string,
                                        tissue_length_cohort_output_figures_dir,
                                        threshold,
                                        cdf_sp_threshold_dict
                                        )
                    fit_parameters_by_threshold_dict[threshold] = {"Fit params for simulated bx": fit_parameters_sim_dict,
                                                                   "Fit params for non-sim bx": fit_parameters_actual_dict
                                                                   }

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


            section_start_time = datetime.now() 
    
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
                US_dcms_dict = defaultdict(list) # defaultdict(list) allows you to append to lists that are created automatically when a new key is created
                MR_T2_dcms_dict = defaultdict(list)
                MR_ADC_dcms_dict = defaultdict(list)
                for dicom_path_index, dicom_path in enumerate(dicom_paths_list):
                    if dicom_elems_modality_list[dicom_path_index] == 'RTSTRUCT':
                        with pydicom.dcmread(dicom_path, defer_size = '2 MB') as py_dicom_item: 
                            RTst_dcms_dict[UID_generator(py_dicom_item)] = dicom_path
                    elif dicom_elems_modality_list[dicom_path_index] == 'RTDOSE':
                        with pydicom.dcmread(dicom_path, defer_size = '2 MB') as py_dicom_item: 
                            RTdose_dcms_dict[UID_generator(py_dicom_item)] = dicom_path
                    elif dicom_elems_modality_list[dicom_path_index] == 'RTPLAN':
                        with pydicom.dcmread(dicom_path, defer_size = '2 MB') as py_dicom_item: 
                            RTplan_dcms_dict[UID_generator(py_dicom_item)] = dicom_path
                    elif dicom_elems_modality_list[dicom_path_index] == 'US': # if the modality is US (likely exported from vitesse with option US or some other software that correctly identified it as US, because recall Vitesse exports US as MR if chosen as such)
                        with pydicom.dcmread(dicom_path, defer_size = '2 MB') as py_dicom_item: 
                            US_dcms_dict[UID_generator(py_dicom_item)].append(dicom_path)  
                    elif dicom_elems_modality_list[dicom_path_index] == 'MR':
                        with pydicom.dcmread(dicom_path, defer_size = '2 MB') as py_dicom_item: 
                            if py_dicom_item[0x0008,0x103E].value == 'T2': # Must label MR T2 images with SeriesDescription as 'T2'
                                MR_T2_dcms_dict[UID_generator(py_dicom_item)].append(dicom_path)
                            elif py_dicom_item[0x0008,0x103E].value == 'ADC': # Must label MR ADC images with SeriesDescription as 'ADC'
                                MR_ADC_dcms_dict[UID_generator(py_dicom_item)].append(dicom_path)
                            elif py_dicom_item[0x0018,0x0023].value == '': # Also identified as US if the modality is MR but the MRAcquisitionType tag is empty
                                US_dcms_dict[UID_generator(py_dicom_item)].append(dicom_path)

                #RTst_dcms_dict = {UID_generator(pydicom.dcmread(dicom_paths_list[j])): pydicom.dcmread(dicom_paths_list[j]) for j in range(num_dicoms) if dicom_elems_modality_list[j] == modality_list[0]}
                #RTdose_dcms_dict = {UID_generator(pydicom.dcmread(dicom_paths_list[j])): pydicom.dcmread(dicom_paths_list[j]) for j in range(num_dicoms) if dicom_elems_modality_list[j] == modality_list[1]}
                #live_display.stop()
                num_RTst_dcms_entries = len(RTst_dcms_dict)
                num_RTdose_dcms_entries = len(RTdose_dcms_dict)
                num_RTplan_dcms_entries = len(RTplan_dcms_dict)
                num_MR_T2_dcms_entries = len(MR_T2_dcms_dict)
                num_MR_ADC_dcms_entries = len(MR_ADC_dcms_dict)
                num_US_dcms_entries = len(US_dcms_dict)
                important_info.add_text_line("Found "+str(num_RTst_dcms_entries)+" unique patients with RT structure files.", live_display)
                important_info.add_text_line("Found "+str(num_RTdose_dcms_entries)+" unique patients with RT dose files.", live_display)
                important_info.add_text_line("Found "+str(num_RTplan_dcms_entries)+" unique patients with RT plan files.", live_display)
                important_info.add_text_line("Found "+str(num_MR_T2_dcms_entries)+" unique patients with MR T2 images.", live_display)
                important_info.add_text_line("Found "+str(num_MR_ADC_dcms_entries)+" unique patients with MR ADC images.", live_display)
                important_info.add_text_line("Found "+str(num_US_dcms_entries)+" unique patients with US images.", live_display)


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
                """
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
                """
                if num_simulated_bxs_to_create >= 1:
                    for sim_bx_type_str,sim_bx_type_dict in bx_sim_locations_dict.items():
                        simulate_biopsies_relative_to = sim_bx_type_dict["Relative to"]
                        keyfound = False
                        for struct_type_key in structs_referenced_dict.keys():
                            if simulate_biopsies_relative_to in structs_referenced_dict[struct_type_key]["Contour names"]:
                                if keyfound == True:
                                    raise Exception("Structure specified to simulate biopsies to was found in more than one structure type.")
                                simulate_biopsies_relative_to_struct_type = struct_type_key
                                keyfound = True
                        if keyfound == False:
                            raise Exception("Structure specified to simulate biopsies to was not found in specified structures to analyse.")
                        sim_bx_type_dict["Relative to struct type"] = simulate_biopsies_relative_to_struct_type
                        important_info.add_text_line("Simulating "+ sim_bx_type_str+" biopsies relative to "+simulate_biopsies_relative_to+" (Found under "+simulate_biopsies_relative_to_struct_type+").", live_display)          
                        live_display.refresh()
                else: 
                    simulate_biopsies_relative_to_struct_type = None
                    important_info.add_text_line("Not creating any simulated biopsies.", live_display)          
                    live_display.refresh() 
                
                #live_display.stop()
                # patient dictionary creation
                building_patient_dictionaries_task = indeterminate_progress_main.add_task('[red]Building patient dictionary...', total=None)
                building_patient_dictionaries_task_completed = completed_progress.add_task('[green]Building patient dictionary', total=num_RTst_dcms_entries, visible = False)
                master_structure_reference_dict, master_structure_info_dict = structure_referencer(data_removals_dict,
                                                                                                RTst_dcms_dict, 
                                                                                                RTdose_dcms_dict,
                                                                                                RTplan_dcms_dict, 
                                                                                                US_dcms_dict,
                                                                                                MR_T2_dcms_dict,
                                                                                                MR_ADC_dcms_dict,
                                                                                                oaroi_contour_names,
                                                                                                dil_contour_names,
                                                                                                biopsy_contour_names,
                                                                                                structs_referenced_list_generalized,
                                                                                                dose_ref,
                                                                                                plan_ref,
                                                                                                mr_adc_ref,
                                                                                                mr_t2_ref,
                                                                                                us_ref,
                                                                                                all_ref_key,
                                                                                                mr_global_multi_structure_output_dataframe_str,
                                                                                                mr_global_by_voxel_multi_structure_output_dataframe_str,
                                                                                                bx_sim_locations_dict,
                                                                                                fanova_sobol_indices_names_by_index,
                                                                                                rectum_contour_names,
                                                                                                urethra_contour_names,
                                                                                                interp_inter_slice_dist,
                                                                                                interp_intra_slice_dist,
                                                                                                fraction_prefixes,
                                                                                                important_info,
                                                                                                live_display
                                                                                                )
                #live_display.stop()
                indeterminate_progress_main.update(building_patient_dictionaries_task, visible = False)
                completed_progress.update(building_patient_dictionaries_task_completed, advance = num_RTst_dcms_entries,visible = True)
                important_info.add_text_line("Patient master dictionary built for "+str(master_structure_info_dict["Global"]["Num cases"])+" patients.", live_display)  
                live_display.refresh()


                ### Check if there are more than one ADC MRs for each patient:
                mr_adc_units = None 
                for patientUID,pydicom_item in master_structure_reference_dict.items():

                    if mr_adc_ref not in pydicom_item:
                        important_info.add_text_line("Notice! no ADC MR for: "+ str(patientUID), live_display)  
                        continue
                     
                    if len(master_structure_reference_dict[patientUID][mr_adc_ref]) > 1: 
                        important_info.add_text_line("Notice! There are "+ str(len(master_structure_reference_dict[patientUID][mr_adc_ref]))+ "ADC MRs for: " +str(patientUID), live_display)                           

                        ###### IMPORTANT! WE REMOVE ALL ENTRIES OF MR ADC IMAGES EXCEPT FOR THE FIRST!

                        important_info.add_text_line("Removing all MR ADCs except the first for: " +str(patientUID), live_display)   

                    # Delete all MR ADCs except the first one 
                    # Get the first key-value pair
                    series_uid, mr_adc_subdict = next(iter(master_structure_reference_dict[patientUID][mr_adc_ref].items()))

                    # Only store the sub dictionary of the first MR series
                    master_structure_reference_dict[patientUID][mr_adc_ref] = mr_adc_subdict

                    if mr_adc_units == None:
                        mr_adc_units = mr_adc_subdict["Units"]
                    elif mr_adc_units != mr_adc_units:
                        important_info.add_text_line("The units of your MRs are not the same between patients! Detected on patient: "+ str(patientUID), live_display)   

                
                
                
                # check if there are any mr adcs to be analysed   
                # This is a flag that can skip certain things that will cause errors if run because there are no mr adc images to analyse
                no_cohort_mr_adc_flag = True
                for patientUID,pydicom_item in master_structure_reference_dict.items():
                    if mr_adc_ref in pydicom_item:
                        no_cohort_mr_adc_flag = False
                        break
                    else:
                        continue




                #live_display.stop()
                #print('test')
                #live_display.start()



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
                #num_patients = master_structure_info_dict["Global"]["Num cases"]
                #num_general_structs = master_structure_info_dict["Global"]["Num structures"]


                #important_info.add_text_line("important info will appear here1", live_display)
                #rich_layout["main-right"].update(important_info_Text)
            
                live_display.stop()
                """
                patientUID_default = "Initializing"
                processing_patients_dose_task_main_description = "[red]Building dose grids [{}]...".format(patientUID_default)
                processing_patients_dose_task_completed_main_description = "[green]Building dose grids"

                processing_patients_dose_task = patients_progress.add_task(processing_patients_dose_task_main_description, total=master_structure_info_dict["Global"]["Num cases"])
                processing_patients_dose_task_completed = completed_progress.add_task(processing_patients_dose_task_completed_main_description, total=master_structure_info_dict["Global"]["Num cases"], visible=False)

                for patientUID,pydicom_item in master_structure_reference_dict.items():
                    processing_patients_dose_task_main_description = "[red]Building dose grids [{}]...".format(patientUID)
                    patients_progress.update(processing_patients_dose_task, description=processing_patients_dose_task_main_description)


                    if dose_ref not in pydicom_item:
                        patients_progress.update(processing_patients_dose_task, advance=1)
                        completed_progress.update(processing_patients_dose_task_completed, advance=1)
                        continue

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
                    
                    
                    # plot labelled dose point cloud (note due to the number of labels, this is very buggy and doesnt display properly as of open3d 0.16.1)
                    """ """
                    patients_progress.stop_task(processing_patients_dose_task)
                    completed_progress.stop_task(processing_patients_dose_task_completed)
                    stopwatch.stop()
                    plotting_funcs.dose_point_cloud_with_dose_labels(phys_space_dose_map_3d_arr, paint_dose_with_color = True)
                    stopwatch.start()
                    patients_progress.start_task(processing_patients_dose_task)
                    completed_progress.start_task(processing_patients_dose_task_completed)
                    """ """

                    # plot dose point cloud cubic lattice (color only)
                    dose_point_cloud = plotting_funcs.create_dose_point_cloud(phys_space_dose_map_3d_arr, color_flattening_deg, paint_dose_color = True)

                    if show_3d_dose_renderings == True:

                        patients_progress.stop_task(processing_patients_dose_task)
                        completed_progress.stop_task(processing_patients_dose_task_completed)
                        stopwatch.stop()
                        plotting_funcs.plot_geometries(dose_point_cloud)
                        stopwatch.start()
                        patients_progress.start_task(processing_patients_dose_task)
                        completed_progress.start_task(processing_patients_dose_task_completed)

                    thresholded_dose_point_cloud = plotting_funcs.create_thresholded_dose_point_cloud(phys_space_dose_map_3d_arr, color_flattening_deg, paint_dose_color = True, lower_bound_percent = lower_bound_dose_percent)
                    
                    # plot dose point cloud thresholded cubic lattice (color only)
                    if show_3d_dose_renderings_thresholded == True:
                        patients_progress.stop_task(processing_patients_dose_task)
                        completed_progress.stop_task(processing_patients_dose_task_completed)
                        stopwatch.stop()
                        plotting_funcs.plot_geometries(thresholded_dose_point_cloud)
                        stopwatch.start()
                        patients_progress.start_task(processing_patients_dose_task)
                        completed_progress.start_task(processing_patients_dose_task_completed)



                    # Store computed objects
                    dose_ref_dict["Dose phys space and pixel 3d arr"] = phys_space_dose_map_3d_arr                   
                    dose_ref_dict["Dose grid point cloud"] = dose_point_cloud
                    dose_ref_dict["Dose grid point cloud thresholded"] = thresholded_dose_point_cloud


                    #gradient_vector_lattice, gradient_norm_lattice = dose_lattice_helper_funcs.calculate_gradient_lattices(dose_pixel_slice_list, pixel_spacing, grid_frame_offset_vec_list)

                    patients_progress.update(processing_patients_dose_task, advance=1)
                    completed_progress.update(processing_patients_dose_task_completed, advance=1)
                
                patients_progress.update(processing_patients_dose_task, visible=False)
                completed_progress.update(processing_patients_dose_task_completed, visible=True)
                """



                



                patientUID_default = "Initializing"
                processing_patients_dose_task_main_description = "[red]Building dose grids [{}]...".format(patientUID_default)
                processing_patients_dose_task_completed_main_description = "[green]Building dose grids"

                processing_patients_dose_task = patients_progress.add_task(processing_patients_dose_task_main_description, total=master_structure_info_dict["Global"]["Num cases"])
                processing_patients_dose_task_completed = completed_progress.add_task(processing_patients_dose_task_completed_main_description, total=master_structure_info_dict["Global"]["Num cases"], visible=False)

                # Main loop for processing patients
                for patientUID, pydicom_item in master_structure_reference_dict.items():
                    processing_patients_dose_task_main_description = "[red]Building dose grids [{}]...".format(patientUID)
                    patients_progress.update(processing_patients_dose_task, description=processing_patients_dose_task_main_description)

                    if dose_ref not in pydicom_item:
                        patients_progress.update(processing_patients_dose_task, advance=1)
                        completed_progress.update(processing_patients_dose_task_completed, advance=1)
                        continue

                    dose_ref_dict = master_structure_reference_dict[patientUID][dose_ref]
                    conversion_matrix = np.array([
                        [dose_ref_dict["Image orientation patient"][0] * dose_ref_dict["Pixel spacing"][1], 
                        dose_ref_dict["Image orientation patient"][3] * dose_ref_dict["Pixel spacing"][0], 
                        0, dose_ref_dict["Image position patient"][0]],
                        [dose_ref_dict["Image orientation patient"][1] * dose_ref_dict["Pixel spacing"][1], 
                        dose_ref_dict["Image orientation patient"][4] * dose_ref_dict["Pixel spacing"][0], 
                        0, dose_ref_dict["Image position patient"][1]],
                        [dose_ref_dict["Image orientation patient"][2] * dose_ref_dict["Pixel spacing"][1], 
                        dose_ref_dict["Image orientation patient"][5] * dose_ref_dict["Pixel spacing"][0], 
                        0, dose_ref_dict["Image position patient"][2]],
                        [0, 0, 0, 1]
                    ])

                    phys_space_dose_map_3d_arr = dose_lattice_helper_funcs.build_dose_grid(
                        dose_pixel_slices=dose_ref_dict["Dose pixel arr"],
                        scaling_factor=dose_ref_dict["Dose grid scaling"],
                        conversion_matrix=conversion_matrix,
                        grid_frame_offset_vec_list=dose_ref_dict["Grid frame offset vector"]
                    )

                    """
                    # plot dose point cloud cubic lattice (color only)
                    dose_point_cloud = plotting_funcs.create_dose_point_cloud(phys_space_dose_map_3d_arr, color_flattening_deg, paint_dose_color = True)

                    if show_3d_dose_renderings == True:

                        patients_progress.stop_task(processing_patients_dose_task)
                        completed_progress.stop_task(processing_patients_dose_task_completed)
                        stopwatch.stop()
                        plotting_funcs.plot_geometries(dose_point_cloud)
                        stopwatch.start()
                        patients_progress.start_task(processing_patients_dose_task)
                        completed_progress.start_task(processing_patients_dose_task_completed)

                    thresholded_dose_point_cloud = plotting_funcs.create_thresholded_dose_point_cloud(phys_space_dose_map_3d_arr, color_flattening_deg, paint_dose_color = True, lower_bound_percent = lower_bound_dose_percent)
                    
                    # plot dose point cloud thresholded cubic lattice (color only)
                    if show_3d_dose_renderings_thresholded == True:
                        patients_progress.stop_task(processing_patients_dose_task)
                        completed_progress.stop_task(processing_patients_dose_task_completed)
                        stopwatch.stop()
                        plotting_funcs.plot_geometries(thresholded_dose_point_cloud)
                        stopwatch.start()
                        patients_progress.start_task(processing_patients_dose_task)
                        completed_progress.start_task(processing_patients_dose_task_completed)
                    """
                    


                    ### DOSE GRADIENT


                    # Scale the dose values before computing gradients
                    scaled_dose_data = dose_ref_dict["Dose pixel arr"] * dose_ref_dict["Dose grid scaling"]

                    gradient_vector_lattice, gradient_norm_lattice, normalized_gradient_vector_lattice = dose_lattice_helper_funcs.calculate_gradient_lattices(scaled_dose_data, dose_ref_dict["Pixel spacing"], dose_ref_dict["Grid frame offset vector"])
                    
                    """
                    Returns:
                        phys_space_gradient_map_3d_arr (numpy.ndarray): Updated slice-wise array with gradients and normalized gradients added.
                            Shape: (num_slices, num_voxels_per_slice, 14).
                            Columns:
                                [0]  - Slice index
                                [1]  - Row index (j)
                                [2]  - Column index (i)
                                [3]  - X-coordinate (physical space)
                                [4]  - Y-coordinate (physical space)
                                [5]  - Z-coordinate (physical space)
                                [6]  - Dose value
                                [7]  - Gradient in X (Gx)
                                [8]  - Gradient in Y (Gy)
                                [9]  - Gradient in Z (Gz)
                                [10] - Gradient norm (|G|)
                                [11] - Normalized Gradient in X (NGx)
                                [12] - Normalized Gradient in Y (NGy)
                                [13] - Normalized Gradient in Z (NGz)
                    """
                    phys_space_dose_map_and_gradient_map_3d_arr = dose_lattice_helper_funcs.map_gradient_to_physical_space(
                        phys_space_dose_map_3d_arr=phys_space_dose_map_3d_arr,
                        gradient_vector_lattice=gradient_vector_lattice,
                        gradient_norm_lattice=gradient_norm_lattice,
                        normalized_gradient_vector_lattice = normalized_gradient_vector_lattice
                    )

                    
                    dose_point_cloud, dose_gradient_arrows_point_cloud = plotting_funcs.create_dose_point_cloud_with_gradients(phys_space_dose_map_and_gradient_map_3d_arr,
                                                                                                                        paint_dose_color=True,
                                                                                                                        arrow_scale=1.0,
                                                                                                                        truncate_below_dose=None,
                                                                                                                        truncate_below_gradient_norm=None
                                                                                                                    )
                    if show_3d_dose_renderings == True:
                        patients_progress.stop_task(processing_patients_dose_task)
                        completed_progress.stop_task(processing_patients_dose_task_completed)
                        stopwatch.stop()
                        plotting_funcs.plot_geometries(dose_point_cloud, dose_gradient_arrows_point_cloud)
                        stopwatch.start()
                        patients_progress.start_task(processing_patients_dose_task)
                        completed_progress.start_task(processing_patients_dose_task_completed)

                    if lower_bound_dose_value == None:
                        try:
                            lower_bound_dose_value = pydicom_item[plan_ref]["Prescription doses dict"]["TARGET"]
                        except Exception as e:
                            lower_bound_dose_value = 0

                    thresholded_dose_point_cloud, thresholded_dose_gradient_arrows_point_cloud = plotting_funcs.create_dose_point_cloud_with_gradients(phys_space_dose_map_and_gradient_map_3d_arr,
                                                                                                                        paint_dose_color=True,
                                                                                                                        arrow_scale=1.0,
                                                                                                                        truncate_below_dose=lower_bound_dose_value,
                                                                                                                        truncate_below_gradient_norm=lower_bound_dose_gradient_value
                                                                                                                    )

                    # plot dose point cloud thresholded cubic lattice (color only)
                    if show_3d_dose_renderings_thresholded == True:
                        patients_progress.stop_task(processing_patients_dose_task)
                        completed_progress.stop_task(processing_patients_dose_task_completed)
                        stopwatch.stop()
                        plotting_funcs.plot_geometries(thresholded_dose_point_cloud, thresholded_dose_gradient_arrows_point_cloud)
                        stopwatch.start()
                        patients_progress.start_task(processing_patients_dose_task)
                        completed_progress.start_task(processing_patients_dose_task_completed)
                    

                    dose_ref_dict["Dose and gradient phys space and pixel 3d arr"] = phys_space_dose_map_and_gradient_map_3d_arr
                    #dose_ref_dict["Dose phys space and pixel 3d arr"] = phys_space_dose_map_3d_arr
                    dose_ref_dict["Dose grid point cloud"] = dose_point_cloud
                    dose_ref_dict["Dose grid point cloud thresholded"] = thresholded_dose_point_cloud
                    dose_ref_dict["Dose grid gradient point cloud"] = dose_gradient_arrows_point_cloud
                    dose_ref_dict["Dose grid gradient point cloud thresholded"] = thresholded_dose_gradient_arrows_point_cloud

                    # Update progress
                    patients_progress.update(processing_patients_dose_task, advance=1)
                    completed_progress.update(processing_patients_dose_task_completed, advance=1)

                # Finalize progress display
                patients_progress.update(processing_patients_dose_task, visible=False)
                completed_progress.update(processing_patients_dose_task_completed, visible=True)














                patientUID_default = "Initializing"
                processing_patients_adc_mr_task_main_description = "[red]Building ADC MR grids [{}]...".format(patientUID_default)
                processing_patients_adc_mr_task_completed_main_description = "[green]Building ADC MR grids"

                processing_patients_adc_mr_task = patients_progress.add_task(processing_patients_adc_mr_task_main_description, total=master_structure_info_dict["Global"]["Num cases"])
                processing_patients_adc_mr_task_completed = completed_progress.add_task(processing_patients_adc_mr_task_completed_main_description, total=master_structure_info_dict["Global"]["Num cases"], visible=False)

                for patientUID,pydicom_item in master_structure_reference_dict.items():
                    processing_patients_adc_mr_task_main_description = "[red]Building ADC MR grids [{}]...".format(patientUID)
                    patients_progress.update(processing_patients_adc_mr_task, description=processing_patients_adc_mr_task_main_description)

                    if mr_adc_ref not in pydicom_item:
                        patients_progress.update(processing_patients_adc_mr_task, advance=1)
                        completed_progress.update(processing_patients_adc_mr_task_completed, advance=1)
                        continue
                    
                    mr_adc_subdict = master_structure_reference_dict[patientUID][mr_adc_ref]

                        
                    filtered_non_negative_adc_mr_phys_space_arr = lattice_reconstruction_tools.reconstruct_mr_lattice_with_coordinates_from_dict_v2(mr_adc_subdict, filter_out_negatives = True)
                    # Don't store this, it is too large, just call the above function if you want to retrieve the MR information lattice
                    #mr_adc_subdict["MR ADC phys space Nx4 arr (filtered, non-negative)"] = filtered_non_negative_adc_mr_phys_space_arr


                    mr_adc_point_cloud = plotting_funcs.create_MR_point_cloud(filtered_non_negative_adc_mr_phys_space_arr, 
                                                                                    color_flattening_deg_MR, 
                                                                                    paint_mr_color = True)


                    thresholded_mr_adc_point_cloud = plotting_funcs.create_thresholded_MR_ADC_point_cloud(filtered_non_negative_adc_mr_phys_space_arr, 
                                                                                                                color_flattening_deg_MR, 
                                                                                                                paint_mr_color = True, 
                                                                                                                lower_bound = lower_bound_mr_adc_value, 
                                                                                                                upper_bound = upper_bound_mr_adc_value, 
                                                                                                                z_val_range_list = None)

                    del filtered_non_negative_adc_mr_phys_space_arr


                    if show_3d_mr_adc_renderings == True:     
                        patients_progress.stop_task(processing_patients_dose_task)
                        completed_progress.stop_task(processing_patients_dose_task_completed)
                        stopwatch.stop()
                        plotting_funcs.plot_geometries(mr_adc_point_cloud)
                        stopwatch.start()
                        patients_progress.start_task(processing_patients_dose_task)
                        completed_progress.start_task(processing_patients_dose_task_completed)

                    
                    
                    # plot dose point cloud thresholded cubic lattice (color only)
                    if show_3d_mr_adc_renderings_thresholded == True:
                        patients_progress.stop_task(processing_patients_dose_task)
                        completed_progress.stop_task(processing_patients_dose_task_completed)
                        stopwatch.stop()
                        plotting_funcs.plot_geometries(thresholded_mr_adc_point_cloud)
                        stopwatch.start()
                        patients_progress.start_task(processing_patients_dose_task)
                        completed_progress.start_task(processing_patients_dose_task_completed)


                    # Store computed objects
                    mr_adc_subdict["MR ADC grid point cloud"] = mr_adc_point_cloud
                    mr_adc_subdict["MR ADC grid point cloud thresholded"] = thresholded_mr_adc_point_cloud


                    patients_progress.update(processing_patients_adc_mr_task, advance=1)
                    completed_progress.update(processing_patients_adc_mr_task_completed, advance=1)
                
                patients_progress.update(processing_patients_adc_mr_task, visible=False)
                completed_progress.update(processing_patients_adc_mr_task_completed, visible=True)


                """
                # create info for simulated biopsies
                if num_simulated_bxs_to_create >= 1:
                    centroid_line_vec_list = [0,0,1]
                    centroid_first_pos_list = [0,0,0]
                    num_centroids_for_sim_bxs = 10
                    centroid_sep_dist = biopsy_needle_compartment_length/(num_centroids_for_sim_bxs-1) # the minus 1 ensures that the legnth of the biopsy is actually correct!
                    simulated_bx_rad = 2
                    plot_simulated_cores_immediately = False
                """



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
                    for structs in structs_referenced_list_generalized:
                        for specific_structure_index, specific_structure in enumerate(pydicom_item[structs]):
                            structureID = specific_structure["ROI"]
                            structure_reference_number = specific_structure["Ref #"]
                            if structs == bx_ref:
                                simulated_bool = specific_structure["Simulated bool"]
                            else: 
                                simulated_bool = None
                            pulling_structures_task_main_description = "[cyan]Pulling structures [{},{}]...".format(patientUID,structureID)
                            structures_progress.update(pulling_structures_task, description = pulling_structures_task_main_description)
                            
                            # create points for simulated biopsies to create
                            if simulated_bool == True:
                                structures_progress.update(pulling_structures_task, advance=1)
                                continue # dont do anything if its a simulated biopsy!
                                # USED TO CREATE THE SIMULATED BIOPSIES HERE, BUT i CANT BECAUSE I WANT THEIR LENGTHS TO DEPEND ON THE MEAN LENGTH OF THE REAL BIOPSIES!
                                #threeDdata_zslice_list = biopsy_creator.biopsy_points_creater_by_transport_for_sim_bxs(centroid_line_vec_list,centroid_first_pos_list,num_centroids_for_sim_bxs,centroid_sep_dist,simulated_bx_rad,plot_simulated_cores_immediately)
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
                            ## THIS WAS INDENTED UNDER THE IF STATEMENT BEFORE
                                structure_centroids_array = np.empty([len(threeDdata_zslice_list),3])
                                # find zslice-wise centroids
                                for index, threeDdata_zslice in enumerate(threeDdata_zslice_list):                           
                                    structure_zslice_centroid = np.mean(threeDdata_zslice,axis=0)
                                    structure_centroids_array[index] = structure_zslice_centroid
                                structure_global_centroid = centroid_finder.centeroidfinder_numpy_3D(structure_centroids_array)
                                master_structure_reference_dict[patientUID][structs][specific_structure_index]["Structure centroid pts"] = structure_centroids_array
                                master_structure_reference_dict[patientUID][structs][specific_structure_index]["Structure global centroid"] = structure_global_centroid
                            ## THIS WAS INDENTED UNDER THE IF STATEMENT BEFORE

                            master_structure_reference_dict[patientUID][structs][specific_structure_index]["Raw contour pts zslice list"] = threeDdata_zslice_list

                            structures_progress.update(pulling_structures_task, advance=1)
                    structures_progress.remove_task(pulling_structures_task)
                    patients_progress.update(pulling_patients_task, advance=1)
                    completed_progress.update(pulling_patients_task_completed, advance=1)
                patients_progress.update(pulling_patients_task, visible=False)
                completed_progress.update(pulling_patients_task_completed,  visible=True)            

                


                live_display.start()

                ### Selecting unqiue structures of each type (except biopsies and dils) for future calculations

                patientUID_default = "Initializing"
                processing_patients_task_main_description = "[red]Selecting unique structures [{}]...".format(patientUID_default)
                processing_patients_task_completed_main_description = "[green]Selecting unique structures"
                processing_patients_task = patients_progress.add_task(processing_patients_task_main_description, total=master_structure_info_dict["Global"]["Num cases"])
                processing_patients_task_completed = completed_progress.add_task(processing_patients_task_completed_main_description, total=master_structure_info_dict["Global"]["Num cases"], visible = False)

                for patientUID,pydicom_item in master_structure_reference_dict.items():
                    processing_patients_task_main_description = "[red]Selecting unique structures [{}]...".format(patientUID)
                    patients_progress.update(processing_patients_task, description = processing_patients_task_main_description)
                    

                    sp_patient_selected_structure_info_dataframe = pandas.DataFrame()

                    for structure_type in structs_referenced_list_generalized_unique_structs:
                        structure_type_contour_names_list =  structs_referenced_dict[structure_type]["Contour names"]

                        selected_structure_info_dataframe, message_string = misc_tools.specific_structure_selector_dataframe_version(pydicom_item,
                                                                                                                                            structure_type,
                                                                                                                                            structure_type_contour_names_list) 
                        

                        important_info.add_text_line(message_string, live_display) 


                        sp_patient_selected_structure_info_dataframe = pandas.concat([sp_patient_selected_structure_info_dataframe,selected_structure_info_dataframe], ignore_index = True)

                    sp_patient_selected_structure_info_dataframe.insert(loc=0, column="Patient ID", value=patientUID)

                    pydicom_item[all_ref_key]["Multi-structure pre-processing output dataframes dict"]["Selected structures"] = sp_patient_selected_structure_info_dataframe



                    ### Now delete all the structures that were not chosen from the master ref dict 
                    ### Note that this was done primarily for the MC simulation section to simplify modifying the code for testing tissue class against 
                    ### individual structures. Instead of modifying that section of code heavily, I am simply removing the structures
                    ### that weren't selected

                    sp_patient_selected_structure_info_dataframe_more_than_one_struct_found_subset_dataframe = sp_patient_selected_structure_info_dataframe[sp_patient_selected_structure_info_dataframe["Total num structs found"] > 1]
                    num_structs_difference = 0
                    for row_index, row in sp_patient_selected_structure_info_dataframe_more_than_one_struct_found_subset_dataframe.iterrows():
                        struct_selected_type = row["Struct ref type"]
                        struct_selected_index = row["Index number"]
                        
                        updated_sp_structure_list = [pydicom_item[struct_selected_type][struct_selected_index]] if 0 <= struct_selected_index < len(pydicom_item[struct_selected_type]) else []

                        pydicom_item[struct_selected_type] = updated_sp_structure_list
                        
        
                        # Update the master patient info record
                        current_num_structs = master_structure_info_dict["By patient"][patientUID][struct_selected_type]["Num structs"]
                        updated_num_structs = len(updated_sp_structure_list)
                        difference = current_num_structs - updated_num_structs
                        num_structs_difference += difference
                    
                        master_structure_info_dict["By patient"][patientUID][struct_selected_type]["Num structs"] = updated_num_structs
                        

                    # Update the master patient info record
                    current_total_num_structs = master_structure_info_dict["By patient"][patientUID][all_ref_key]["Total num structs"]
                    master_structure_info_dict["By patient"][patientUID][all_ref_key]["Total num structs"] = current_total_num_structs - num_structs_difference

                    total_num_structs_updated = 0
                    for patientUID,pydicom_item in master_structure_reference_dict.items():
                        for structure_type in structs_referenced_list_generalized:

                            num_structs = len(pydicom_item[structure_type])
                            total_num_structs_updated += num_structs

                    master_structure_info_dict["Global"]["Num structures"] = total_num_structs_updated
                   
                    patients_progress.update(processing_patients_task, advance=1)
                    completed_progress.update(processing_patients_task_completed, advance=1)
                patients_progress.update(processing_patients_task, visible=False)
                completed_progress.update(processing_patients_task_completed,  visible=True)










                ### PREPROCESSING OARs


                #live_display.stop()

                patientUID_default = "Initializing"
                processing_patients_task_main_description = "[red]Processing patient prostates [{}]...".format(patientUID_default)
                processing_patients_task_completed_main_description = "[green]Processing patient prostates"
                processing_patients_task = patients_progress.add_task(processing_patients_task_main_description, total=master_structure_info_dict["Global"]["Num cases"])
                processing_patients_task_completed = completed_progress.add_task(processing_patients_task_completed_main_description, total=master_structure_info_dict["Global"]["Num cases"], visible = False)

                for patientUID,pydicom_item in master_structure_reference_dict.items():
                    processing_patients_task_main_description = "[red]Processing patient prostates [{}]...".format(patientUID)
                    patients_progress.update(processing_patients_task, description = processing_patients_task_main_description)
                    
                    structureID_default = "Initializing"
                    #num_general_structs_patient_specific = master_structure_info_dict["By patient"][patientUID][all_ref_key]["Total num structs"]
                    num_total_structs_patient_specific = master_structure_info_dict["By patient"][patientUID][all_ref_key]["Total num structs"]
                    num_bx_structs_patient_specific = master_structure_info_dict["By patient"][patientUID][bx_ref]["Num structs"]
                    
                    num_prostates = len(pydicom_item[oar_ref])

                    
                    processing_structures_task_main_description = "[cyan]Processing structures [{},{}]...".format(patientUID,structureID_default)
                    processing_structures_task = structures_progress.add_task(processing_structures_task_main_description, total=num_prostates)

                    structs = oar_ref
                    for specific_structure_index, specific_structure in enumerate(pydicom_item[structs]):
                        structureID = specific_structure["ROI"]
                        structure_reference_number = specific_structure["Ref #"]
                        processing_structures_task_main_description = "[cyan]Processing structures [{},{}]...".format(patientUID,structureID)
                        structures_progress.update(processing_structures_task, description = processing_structures_task_main_description)

                        # The below print lines were just for my own understanding of how to access the data structure
                        #print(RTst_dcms[dcm_index].ROIContourSequence[int(specific_structure["Ref #"])].ContourSequence[0].ContourData)
                        #print(RTst_dcms[dcm_index].ROIContourSequence[int(specific_structure["Ref #"])].ContourSequence[1].ContourData)

                        ### BUILD RAW DATA
                        ###
                        indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Build raw data", total = None)
                        ###

                        threeDdata_zslice_list = master_structure_reference_dict[patientUID][structs][specific_structure_index]["Raw contour pts zslice list"].copy()
                        
                        total_structure_points = sum([np.shape(x)[0] for x in threeDdata_zslice_list])
                        threeDdata_array = np.empty([total_structure_points,3])                                                       

                        # build raw threeDdata for non biopsies
                        lower_bound_index = 0  
                        for index, threeDdata_zslice in enumerate(threeDdata_zslice_list):
                            current_zslice_num_points = np.size(threeDdata_zslice,0)
                            threeDdata_array[lower_bound_index:lower_bound_index + current_zslice_num_points] = threeDdata_zslice
                            lower_bound_index = lower_bound_index + current_zslice_num_points 
                        
                        
                        ###
                        indeterminate_progress_sub.update(indeterminate_task, visible = False)
                        ###
                        ###### END BUILD RAW DATA





                        ### INTERPOLATE STRUCTURE
                        ###
                        indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Interpolate structure", total = None)
                        ###
                        
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
                        pcd_color = structs_referenced_dict[structs]['PCD color']
                        threeDdata_point_cloud = point_containment_tools.create_point_cloud(threeDdata_array, pcd_color)
                        
                        # generate delaunay triangulations (Deprecated, no longer need to use delaunay)
                        """
                        deulaunay_objs_zslice_wise_list = point_containment_tools.adjacent_slice_delaunay_parallel(parallel_pool, threeDdata_zslice_list)

                        zslice1 = threeDdata_array[0,2]
                        zslice2 = threeDdata_array[-1,2]
                        delaunay_global_convex_structure_obj = point_containment_tools.delaunay_obj(threeDdata_array, threeDdata_pcd_color, zslice1, zslice2)
                        #delaunay_global_convex_structure_obj.generate_lineset()
                        """

                        threeDdata_array_fully_interpolated = interpolation_information.interpolated_pts_np_arr
                        threeDdata_array_fully_interpolated_with_end_caps = interpolation_information.interpolated_pts_with_end_caps_np_arr
                        threeDdata_array_interslice_interpolation = np.vstack(interslice_interpolation_information.interpolated_pts_list)
                        interslice_interp_pcd = point_containment_tools.create_point_cloud(threeDdata_array_interslice_interpolation, pcd_color)
                        inter_and_intra_interp_pcd = point_containment_tools.create_point_cloud(threeDdata_array_fully_interpolated, pcd_color)
                        inter_and_intra_and_end_caps_interp_pcd = point_containment_tools.create_point_cloud(threeDdata_array_fully_interpolated_with_end_caps, pcd_color)
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
                        


                        ###
                        indeterminate_progress_sub.update(indeterminate_task, visible = False)
                        ###
                        ###### END INTERPOLATE STRUCTURE








                        ### CALCULATE THE STRUCTURES VOLUME
                        ###
                        indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Calculating structure volume", total = None)
                        ###
                                
                        structure_info = misc_tools.specific_structure_info_dict_creator('given', specific_structure = specific_structure) 

                        interpolated_pts_np_arr = interslice_interpolation_information.interpolated_pts_np_arr
                        interpolated_zvals_list = interslice_interpolation_information.zslice_vals_after_interpolation_list
                        zslices_list = interslice_interpolation_information.interpolated_pts_list

                        structure_volume, maximum_distance, voxel_size_for_structure_volume_calc, binary_mask_arr, live_display = misc_tools.structure_volume_calculator(interpolated_pts_np_arr,
                            interpolated_zvals_list,
                            zslices_list,
                            structure_info,
                            plot_volume_calculation_containment_result_bool,
                            plot_binary_mask_bool,
                            voxel_size_for_structure_volume_calc_non_bx,
                            factor_for_voxel_size,
                            cupy_array_upper_limit_NxN_size_input,
                            layout_groups,
                            nearest_zslice_vals_and_indices_cupy_generic_max_size,
                            structures_progress,
                            live_display
                            )
                        
                        ###
                        indeterminate_progress_sub.update(indeterminate_task, visible = False)
                        ###
                        ###### END STRUCTURE VOLUME CALCULATION



                        ### CALCULATE THE STRUCTURES DIMENSIONS AT THE CENTROID IN X,Y,Z DIRECTIONS
                        ###
                        indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Calculating structure dimensions", total = None)
                        ###
                        
                        interpolated_pts_np_arr = interslice_interpolation_information.interpolated_pts_np_arr
                        interpolated_zvals_list = interslice_interpolation_information.zslice_vals_after_interpolation_list
                        zslices_list = interslice_interpolation_information.interpolated_pts_list
                        non_bx_structure_global_centroid = specific_structure["Structure global centroid"].copy()
                        non_bx_structure_global_centroid = np.reshape(non_bx_structure_global_centroid,(3))

                        structure_dimension_at_centroid_dict, voxel_size_for_structure_dimension_calc, live_display = misc_tools.structure_dimensions_calculator(interpolated_pts_np_arr,
                                                                                                                                                    interpolated_zvals_list,
                                                                                                                                                    zslices_list,
                                                                                                                                                    non_bx_structure_global_centroid,
                                                                                                                                                    structure_info,
                                                                                                                                                    plot_dimension_calculation_containment_result_bool,
                                                                                                                                                    voxel_size_for_structure_dimension_calc,
                                                                                                                                                    factor_for_voxel_size,
                                                                                                                                                    cupy_array_upper_limit_NxN_size_input,
                                                                                                                                                    layout_groups,
                                                                                                                                                    nearest_zslice_vals_and_indices_cupy_generic_max_size,
                                                                                                                                                    structures_progress,
                                                                                                                                                    live_display
                                                                                                                                                    )

                        ###
                        indeterminate_progress_sub.update(indeterminate_task, visible = False)
                        ###


                        """
                        ### COMPUTE POINT-WISE CURVATURE FOR DILS ONLY

                        ###
                        indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Calculating structure curvature", total = None)
                        ###
                        

                        structure_curvature_dictionary = misc_tools.determine_structure_curvature_dictionary_output(threeDdata_array_fully_interpolated_with_end_caps,
                                                                                                                    radius_for_normals_estimation,
                                                                                                                    max_nn_for_normals_estimation,
                                                                                                                    radius_for_curvature_estimation,
                                                                                                                    display_curvature_bool)



                        ###
                        indeterminate_progress_sub.update(indeterminate_task, visible = False)
                        ###
                        """





                        ### COMPUTE TRIANGLE MESH AND STRUCTURE SURFACE AREA


                        ###
                        indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Calculating structure triangle mesh", total = None)
                        live_display.refresh()
                        ###
                        #live_display.stop()
                        #st = time.time()


                        fully_interp_with_end_caps_structure_triangle_mesh, _ = misc_tools.compute_structure_triangle_mesh(interp_inter_slice_dist, 
                            interp_intra_slice_dist,
                            threeDdata_array_fully_interpolated_with_end_caps,
                            radius_for_normals_estimation,
                            max_nn_for_normals_estimation
                            )

                        
                        #et = time.time()
                        #regular_time = et - st
                        
                        #st = time.time()
                        # The non blocking version was created to try to allow the live_display to continue running and not be frozen during this
                        # execution, but the triangle mesh methods of o3d are blocking. Unfortunately, this work-around attempt didnt work. No point in working on this more for now.
                        """
                        fully_interp_with_end_caps_structure_triangle_mesh, _, live_display = misc_tools.compute_structure_triangle_mesh_non_blocking(interp_inter_slice_dist, 
                            interp_intra_slice_dist,
                            threeDdata_array_fully_interpolated_with_end_caps,
                            radius_for_normals_estimation,
                            max_nn_for_normals_estimation,
                            live_display = live_display
                            )
                        """
                        #et = time.time()

                        #non_blocking_time = et - st

                        
                        if display_structure_surface_mesh_bool == True:
                            o3d.visualization.draw_geometries([fully_interp_with_end_caps_structure_triangle_mesh], mesh_show_back_face=True)


                        #live_display.stop()
                        #print('\n Execution time (regular):', regular_time, 'seconds')
                        #print('\n Execution time (non-blocking):', non_blocking_time, 'seconds')


                        ###
                        indeterminate_progress_sub.update(indeterminate_task, visible = False)
                        ###
                        ###
                        indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Calculating surface area", total = None)
                        ###

                        structure_fully_interp_with_end_caps_surface_area = misc_tools.compute_surface_area(fully_interp_with_end_caps_structure_triangle_mesh)
                        """
                        end_caps_points = np.array(interpolation_information.endcaps_points)
                        area_voxel_size = interp_dist_caps**2
                        end_caps_area = misc_tools.compute_end_caps_area(end_caps_points,area_voxel_size)

                        structure_total_surface_area = structure_fully_interp_surface_area + end_caps_area
                        """


                        ###
                        indeterminate_progress_sub.update(indeterminate_task, visible = False)
                        ###


                        ### COMPUTE OTHER 3D SHAPE FEATURES

                        surface_volume_ratio = structure_fully_interp_with_end_caps_surface_area/structure_volume
                        sphericity = misc_tools.calculate_sphericity(structure_volume,structure_fully_interp_with_end_caps_surface_area)
                        compactness_1 = misc_tools.calculate_compactness_1(structure_volume,structure_fully_interp_with_end_caps_surface_area)
                        compactness_2 = misc_tools.calculate_compactness_2(structure_volume,structure_fully_interp_with_end_caps_surface_area)
                        spherical_disproportion = misc_tools.spherical_disproportion(structure_volume,structure_fully_interp_with_end_caps_surface_area)
                        maximum_3D_diameter = maximum_distance 

                        # Note that the eigenvectors are vstacked
                        pca_lengths_of_structure_dict, pca_eigenvectors_of_structure_arr = misc_tools.pca_lengths(binary_mask_arr)

                        # This is the same method as pyradiomics
                        equivalent_ellipse_dimensions = {"Major axis": 4*math.sqrt(pca_lengths_of_structure_dict["Major"]),
                                                            "Minor axis": 4*math.sqrt(pca_lengths_of_structure_dict["Minor"]),
                                                            "Least axis": 4*math.sqrt(pca_lengths_of_structure_dict["Least"])}

                        if show_equivalent_ellipsoid_from_pca_bool == True:
                            axis_diameters = list(equivalent_ellipse_dimensions.values())
                            misc_tools.draw_oriented_ellipse_point_cloud(threeDdata_array_fully_interpolated_with_end_caps, axis_diameters, pca_eigenvectors_of_structure_arr)


                        elongation = math.sqrt(pca_lengths_of_structure_dict["Minor"]/pca_lengths_of_structure_dict["Major"])
                        flatness = math.sqrt(pca_lengths_of_structure_dict["Least"]/pca_lengths_of_structure_dict["Major"])

                        

                        # Create dataframe of the 3d shape features
                        shape_features_3d_dictionary = {"Patient ID": [patientUID],
                                                        "Structure ID": [structureID],
                                                        "Structure type": [structs],
                                                        "Structure refnum": [structure_reference_number],
                                                        "Volume": [structure_volume],
                                                        "Surface area": [structure_fully_interp_with_end_caps_surface_area],
                                                        "Surface area to volume ratio": [surface_volume_ratio],
                                                        "Sphericity": [sphericity],
                                                        "Compactness 1": [compactness_1],
                                                        "Compactness 2": [compactness_2],
                                                        "Spherical disproportion": [spherical_disproportion],
                                                        "Maximum 3D diameter": [maximum_3D_diameter],
                                                        "PCA major": [pca_lengths_of_structure_dict["Major"]],
                                                        "PCA minor": [pca_lengths_of_structure_dict["Minor"]],
                                                        "PCA least": [pca_lengths_of_structure_dict["Least"]],
                                                        "PCA eigenvector major": [tuple(pca_eigenvectors_of_structure_arr[0,:])],
                                                        "PCA eigenvector minor": [tuple(pca_eigenvectors_of_structure_arr[1,:])],
                                                        "PCA eigenvector least": [tuple(pca_eigenvectors_of_structure_arr[2,:])],
                                                        "Major axis (equivalent ellipse)": [equivalent_ellipse_dimensions["Major axis"]],
                                                        "Minor axis (equivalent ellipse)": [equivalent_ellipse_dimensions["Minor axis"]],
                                                        "Least axis (equivalent ellipse)": [equivalent_ellipse_dimensions["Least axis"]],
                                                        "Elongation": [elongation],
                                                        "Flatness": [flatness],
                                                        "L/R dimension at centroid": structure_dimension_at_centroid_dict['X dimension length at centroid'],
                                                        "A/P dimension at centroid": structure_dimension_at_centroid_dict['Y dimension length at centroid'],
                                                        "S/I dimension at centroid": structure_dimension_at_centroid_dict['Z dimension length at centroid']
                                                        }

                        shape_features_dataframe = pandas.DataFrame(shape_features_3d_dictionary)
                        shape_features_dataframe = dataframe_builders.convert_columns_to_categorical_and_downcast(shape_features_dataframe, threshold=0.25)

                        # store all calculated quantities
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Raw contour pts"] = threeDdata_array
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Equal num zslice contour pts"] = threeDdata_equal_pt_zslice_list
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Inter-slice interpolation information"] = interslice_interpolation_information                        
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Intra-slice interpolation information"] = interpolation_information
                        #master_structure_reference_dict[patientUID][structs][specific_structure_index]["Delaunay triangulation zslice-wise list"] = deulaunay_objs_zslice_wise_list # DEPRECATED
                        #master_structure_reference_dict[patientUID][structs][specific_structure_index]["Delaunay triangulation global structure"] = delaunay_global_convex_structure_obj # DEPRECATED
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Maximum pairwise distance"] = maximum_3D_diameter
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Structure volume"] = structure_volume
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Voxel size for structure volume calc"] = voxel_size_for_structure_volume_calc
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Structure dimension at centroid dict"] = structure_dimension_at_centroid_dict
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Voxel size for structure dimension calc"] = voxel_size_for_structure_dimension_calc
                        #master_structure_reference_dict[patientUID][structs][specific_structure_index]["Structure curvature dict"] = structure_curvature_dictionary
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Structure surface area"] = structure_fully_interp_with_end_caps_surface_area
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Structure features dataframe"] = shape_features_dataframe
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Point cloud raw"] = threeDdata_point_cloud
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Interpolated structure point cloud dict"] = interpolated_pcd_dict
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Structure OPEN3D triangle mesh object"] = fully_interp_with_end_caps_structure_triangle_mesh

                        structures_progress.update(processing_structures_task, advance=1)

                    structures_progress.remove_task(processing_structures_task)
                    patients_progress.update(processing_patients_task, advance=1)
                    completed_progress.update(processing_patients_task_completed, advance=1)
                patients_progress.update(processing_patients_task, visible=False)
                completed_progress.update(processing_patients_task_completed,  visible=True)    


















                ### PREPROCESSING RECTUMS


                #live_display.stop()

                patientUID_default = "Initializing"
                processing_patients_task_main_description = "[red]Processing patient rectums [{}]...".format(patientUID_default)
                processing_patients_task_completed_main_description = "[green]Processing patient rectums"
                processing_patients_task = patients_progress.add_task(processing_patients_task_main_description, total=master_structure_info_dict["Global"]["Num cases"])
                processing_patients_task_completed = completed_progress.add_task(processing_patients_task_completed_main_description, total=master_structure_info_dict["Global"]["Num cases"], visible = False)

                for patientUID,pydicom_item in master_structure_reference_dict.items():
                    processing_patients_task_main_description = "[red]Processing patient rectums [{}]...".format(patientUID)
                    patients_progress.update(processing_patients_task, description = processing_patients_task_main_description)
                    
                    structureID_default = "Initializing"          
                    num_rectums = master_structure_info_dict["By patient"][patientUID][rectum_ref_key]["Num structs"]



                    
                    processing_structures_task_main_description = "[cyan]Processing [{},{}]...".format(patientUID,structureID_default)
                    processing_structures_task = structures_progress.add_task(processing_structures_task_main_description, total=num_rectums)

                    structs = rectum_ref_key
                    for specific_structure_index, specific_structure in enumerate(pydicom_item[structs]):
                        structureID = specific_structure["ROI"]
                        structure_reference_number = specific_structure["Ref #"]
                        processing_structures_task_main_description = "[cyan]Processing [{},{}]...".format(patientUID,structureID)
                        structures_progress.update(processing_structures_task, description = processing_structures_task_main_description)

                        # The below print lines were just for my own understanding of how to access the data structure
                        #print(RTst_dcms[dcm_index].ROIContourSequence[int(specific_structure["Ref #"])].ContourSequence[0].ContourData)
                        #print(RTst_dcms[dcm_index].ROIContourSequence[int(specific_structure["Ref #"])].ContourSequence[1].ContourData)

                        threeDdata_zslice_list = master_structure_reference_dict[patientUID][structs][specific_structure_index]["Raw contour pts zslice list"].copy()
                        
                        total_structure_points = sum([np.shape(x)[0] for x in threeDdata_zslice_list])
                        threeDdata_array = np.empty([total_structure_points,3])                                                       

                        # build raw threeDdata for non biopsies
                        lower_bound_index = 0  
                        for index, threeDdata_zslice in enumerate(threeDdata_zslice_list):
                            current_zslice_num_points = np.size(threeDdata_zslice,0)
                            threeDdata_array[lower_bound_index:lower_bound_index + current_zslice_num_points] = threeDdata_zslice
                            lower_bound_index = lower_bound_index + current_zslice_num_points 
                        
                        
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
                        pcd_color = structs_referenced_dict[structs]['PCD color']
                        threeDdata_point_cloud = point_containment_tools.create_point_cloud(threeDdata_array, pcd_color)
                        
                        # generate delaunay triangulations (DEPRECATED)
                        """
                        deulaunay_objs_zslice_wise_list = point_containment_tools.adjacent_slice_delaunay_parallel(parallel_pool, threeDdata_zslice_list)

                        zslice1 = threeDdata_array[0,2]
                        zslice2 = threeDdata_array[-1,2]
                        delaunay_global_convex_structure_obj = point_containment_tools.delaunay_obj(threeDdata_array, threeDdata_pcd_color, zslice1, zslice2)
                        #delaunay_global_convex_structure_obj.generate_lineset()
                        """

                        threeDdata_array_fully_interpolated = interpolation_information.interpolated_pts_np_arr
                        threeDdata_array_fully_interpolated_with_end_caps = interpolation_information.interpolated_pts_with_end_caps_np_arr
                        threeDdata_array_interslice_interpolation = np.vstack(interslice_interpolation_information.interpolated_pts_list)
                        interslice_interp_pcd = point_containment_tools.create_point_cloud(threeDdata_array_interslice_interpolation, pcd_color)
                        inter_and_intra_interp_pcd = point_containment_tools.create_point_cloud(threeDdata_array_fully_interpolated, pcd_color)
                        inter_and_intra_and_end_caps_interp_pcd = point_containment_tools.create_point_cloud(threeDdata_array_fully_interpolated_with_end_caps, pcd_color)
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
                        


                        structure_info = misc_tools.specific_structure_info_dict_creator('given', specific_structure = specific_structure)

                        ### CALCULATE THE STRUCTURES VOLUME
                        ###
                        indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Calculating structure volume", total = None)
                        ###
                                
                        
                        interpolated_pts_np_arr = interslice_interpolation_information.interpolated_pts_np_arr
                        interpolated_zvals_list = interslice_interpolation_information.zslice_vals_after_interpolation_list
                        zslices_list = interslice_interpolation_information.interpolated_pts_list

                        structure_volume, maximum_distance, voxel_size_for_structure_volume_calc, binary_mask_arr, live_display = misc_tools.structure_volume_calculator(interpolated_pts_np_arr,
                            interpolated_zvals_list,
                            zslices_list,
                            structure_info,
                            plot_volume_calculation_containment_result_bool,
                            plot_binary_mask_bool,
                            voxel_size_for_structure_volume_calc_non_bx,
                            factor_for_voxel_size,
                            cupy_array_upper_limit_NxN_size_input,
                            layout_groups,
                            nearest_zslice_vals_and_indices_cupy_generic_max_size,
                            structures_progress,
                            live_display
                            )
                        
                        ###
                        indeterminate_progress_sub.update(indeterminate_task, visible = False)
                        ###
                        ###### END STRUCTURE VOLUME CALCULATION



                        ### CALCULATE THE STRUCTURES DIMENSIONS AT THE CENTROID IN X,Y,Z DIRECTIONS
                        ###
                        indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Calculating structure dimensions", total = None)
                        ###
                        
                        interpolated_pts_np_arr = interslice_interpolation_information.interpolated_pts_np_arr
                        interpolated_zvals_list = interslice_interpolation_information.zslice_vals_after_interpolation_list
                        zslices_list = interslice_interpolation_information.interpolated_pts_list
                        non_bx_structure_global_centroid = specific_structure["Structure global centroid"].copy()
                        non_bx_structure_global_centroid = np.reshape(non_bx_structure_global_centroid,(3))

                        structure_dimension_at_centroid_dict, voxel_size_for_structure_dimension_calc, live_display = misc_tools.structure_dimensions_calculator(interpolated_pts_np_arr,
                                                                                                                                                    interpolated_zvals_list,
                                                                                                                                                    zslices_list,
                                                                                                                                                    non_bx_structure_global_centroid,
                                                                                                                                                    structure_info,
                                                                                                                                                    plot_dimension_calculation_containment_result_bool,
                                                                                                                                                    voxel_size_for_structure_dimension_calc,
                                                                                                                                                    factor_for_voxel_size,
                                                                                                                                                    cupy_array_upper_limit_NxN_size_input,
                                                                                                                                                    layout_groups,
                                                                                                                                                    nearest_zslice_vals_and_indices_cupy_generic_max_size,
                                                                                                                                                    structures_progress,
                                                                                                                                                    live_display
                                                                                                                                                    )

                        ###
                        indeterminate_progress_sub.update(indeterminate_task, visible = False)
                        ###


                        """
                        ### COMPUTE POINT-WISE CURVATURE FOR DILS ONLY

                        ###
                        indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Calculating structure curvature", total = None)
                        ###
                        

                        structure_curvature_dictionary = misc_tools.determine_structure_curvature_dictionary_output(threeDdata_array_fully_interpolated_with_end_caps,
                                                                                                                    radius_for_normals_estimation,
                                                                                                                    max_nn_for_normals_estimation,
                                                                                                                    radius_for_curvature_estimation,
                                                                                                                    display_curvature_bool)



                        ###
                        indeterminate_progress_sub.update(indeterminate_task, visible = False)
                        ###
                        """





                        ### COMPUTE TRIANGLE MESH AND STRUCTURE SURFACE AREA


                        ###
                        indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Calculating structure triangle mesh and surface area", total = None)
                        live_display.refresh()
                        ###
                        #live_display.stop()

                        fully_interp_with_end_caps_structure_triangle_mesh, _ = misc_tools.compute_structure_triangle_mesh(interp_inter_slice_dist, 
                            interp_intra_slice_dist,
                            threeDdata_array_fully_interpolated_with_end_caps,
                            radius_for_normals_estimation,
                            max_nn_for_normals_estimation
                            )
                        
                        if display_structure_surface_mesh_bool == True:
                            o3d.visualization.draw_geometries([fully_interp_with_end_caps_structure_triangle_mesh], mesh_show_back_face=True)

                        structure_fully_interp_with_end_caps_surface_area = misc_tools.compute_surface_area(fully_interp_with_end_caps_structure_triangle_mesh)
                        """
                        end_caps_points = np.array(interpolation_information.endcaps_points)
                        area_voxel_size = interp_dist_caps**2
                        end_caps_area = misc_tools.compute_end_caps_area(end_caps_points,area_voxel_size)

                        structure_total_surface_area = structure_fully_interp_surface_area + end_caps_area
                        """


                        ###
                        indeterminate_progress_sub.update(indeterminate_task, visible = False)
                        ###


                        ### COMPUTE OTHER 3D SHAPE FEATURES

                        surface_volume_ratio = structure_fully_interp_with_end_caps_surface_area/structure_volume
                        sphericity = misc_tools.calculate_sphericity(structure_volume,structure_fully_interp_with_end_caps_surface_area)
                        compactness_1 = misc_tools.calculate_compactness_1(structure_volume,structure_fully_interp_with_end_caps_surface_area)
                        compactness_2 = misc_tools.calculate_compactness_2(structure_volume,structure_fully_interp_with_end_caps_surface_area)
                        spherical_disproportion = misc_tools.spherical_disproportion(structure_volume,structure_fully_interp_with_end_caps_surface_area)
                        maximum_3D_diameter = maximum_distance 

                        # Note that the eigenvectors are vstacked
                        pca_lengths_of_structure_dict, pca_eigenvectors_of_structure_arr = misc_tools.pca_lengths(binary_mask_arr)

                        # This is the same method as pyradiomics
                        equivalent_ellipse_dimensions = {"Major axis": 4*math.sqrt(pca_lengths_of_structure_dict["Major"]),
                                                            "Minor axis": 4*math.sqrt(pca_lengths_of_structure_dict["Minor"]),
                                                            "Least axis": 4*math.sqrt(pca_lengths_of_structure_dict["Least"])}

                        if show_equivalent_ellipsoid_from_pca_bool == True:
                            axis_diameters = list(equivalent_ellipse_dimensions.values())
                            misc_tools.draw_oriented_ellipse_point_cloud(threeDdata_array_fully_interpolated_with_end_caps, axis_diameters, pca_eigenvectors_of_structure_arr)


                        elongation = math.sqrt(pca_lengths_of_structure_dict["Minor"]/pca_lengths_of_structure_dict["Major"])
                        flatness = math.sqrt(pca_lengths_of_structure_dict["Least"]/pca_lengths_of_structure_dict["Major"])

                        

                        # Create dataframe of the 3d shape features
                        shape_features_3d_dictionary = {"Patient ID": [patientUID],
                                                        "Structure ID": [structureID],
                                                        "Structure type": [structs],
                                                        "Structure refnum": [structure_reference_number],
                                                        "Volume": [structure_volume],
                                                        "Surface area": [structure_fully_interp_with_end_caps_surface_area],
                                                        "Surface area to volume ratio": [surface_volume_ratio],
                                                        "Sphericity": [sphericity],
                                                        "Compactness 1": [compactness_1],
                                                        "Compactness 2": [compactness_2],
                                                        "Spherical disproportion": [spherical_disproportion],
                                                        "Maximum 3D diameter": [maximum_3D_diameter],
                                                        "PCA major": [pca_lengths_of_structure_dict["Major"]],
                                                        "PCA minor": [pca_lengths_of_structure_dict["Minor"]],
                                                        "PCA least": [pca_lengths_of_structure_dict["Least"]],
                                                        "PCA eigenvector major": [tuple(pca_eigenvectors_of_structure_arr[0,:])],
                                                        "PCA eigenvector minor": [tuple(pca_eigenvectors_of_structure_arr[1,:])],
                                                        "PCA eigenvector least": [tuple(pca_eigenvectors_of_structure_arr[2,:])],
                                                        "Major axis (equivalent ellipse)": [equivalent_ellipse_dimensions["Major axis"]],
                                                        "Minor axis (equivalent ellipse)": [equivalent_ellipse_dimensions["Minor axis"]],
                                                        "Least axis (equivalent ellipse)": [equivalent_ellipse_dimensions["Least axis"]],
                                                        "Elongation": [elongation],
                                                        "Flatness": [flatness],
                                                        "L/R dimension at centroid": structure_dimension_at_centroid_dict['X dimension length at centroid'],
                                                        "A/P dimension at centroid": structure_dimension_at_centroid_dict['Y dimension length at centroid'],
                                                        "S/I dimension at centroid": structure_dimension_at_centroid_dict['Z dimension length at centroid']
                                                        }



                        

                        shape_features_dataframe = pandas.DataFrame(shape_features_3d_dictionary)
                        shape_features_dataframe = dataframe_builders.convert_columns_to_categorical_and_downcast(shape_features_dataframe, threshold=0.25)


                        # store all calculated quantities
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Raw contour pts"] = threeDdata_array
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Equal num zslice contour pts"] = threeDdata_equal_pt_zslice_list
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Inter-slice interpolation information"] = interslice_interpolation_information                        
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Intra-slice interpolation information"] = interpolation_information
                        #master_structure_reference_dict[patientUID][structs][specific_structure_index]["Delaunay triangulation zslice-wise list"] = deulaunay_objs_zslice_wise_list # DEPRECATED
                        #master_structure_reference_dict[patientUID][structs][specific_structure_index]["Delaunay triangulation global structure"] = delaunay_global_convex_structure_obj # DEPRECATED
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Maximum pairwise distance"] = maximum_3D_diameter
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Structure volume"] = structure_volume
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Voxel size for structure volume calc"] = voxel_size_for_structure_volume_calc
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Structure dimension at centroid dict"] = structure_dimension_at_centroid_dict
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Voxel size for structure dimension calc"] = voxel_size_for_structure_dimension_calc
                        #master_structure_reference_dict[patientUID][structs][specific_structure_index]["Structure curvature dict"] = structure_curvature_dictionary
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Structure surface area"] = structure_fully_interp_with_end_caps_surface_area
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Structure features dataframe"] = shape_features_dataframe
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Point cloud raw"] = threeDdata_point_cloud
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Interpolated structure point cloud dict"] = interpolated_pcd_dict
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Structure OPEN3D triangle mesh object"] = fully_interp_with_end_caps_structure_triangle_mesh


                        structures_progress.update(processing_structures_task, advance=1)

                    structures_progress.remove_task(processing_structures_task)
                    patients_progress.update(processing_patients_task, advance=1)
                    completed_progress.update(processing_patients_task_completed, advance=1)
                patients_progress.update(processing_patients_task, visible=False)
                completed_progress.update(processing_patients_task_completed,  visible=True)    







                ### PREPROCESSING URETHRAS


                #live_display.stop()

                patientUID_default = "Initializing"
                processing_patients_task_main_description = "[red]Processing patient urethras [{}]...".format(patientUID_default)
                processing_patients_task_completed_main_description = "[green]Processing patient urethras"
                processing_patients_task = patients_progress.add_task(processing_patients_task_main_description, total=master_structure_info_dict["Global"]["Num cases"])
                processing_patients_task_completed = completed_progress.add_task(processing_patients_task_completed_main_description, total=master_structure_info_dict["Global"]["Num cases"], visible = False)

                for patientUID,pydicom_item in master_structure_reference_dict.items():
                    processing_patients_task_main_description = "[red]Processing patient urethras [{}]...".format(patientUID)
                    patients_progress.update(processing_patients_task, description = processing_patients_task_main_description)
                    
                    structureID_default = "Initializing"          
                    num_urethras = master_structure_info_dict["By patient"][patientUID][urethra_ref_key]["Num structs"]



                    
                    processing_structures_task_main_description = "[cyan]Processing [{},{}]...".format(patientUID,structureID_default)
                    processing_structures_task = structures_progress.add_task(processing_structures_task_main_description, total=num_urethras)

                    structs = urethra_ref_key
                    for specific_structure_index, specific_structure in enumerate(pydicom_item[structs]):
                        structureID = specific_structure["ROI"]
                        structure_reference_number = specific_structure["Ref #"]
                        processing_structures_task_main_description = "[cyan]Processing [{},{}]...".format(patientUID,structureID)
                        structures_progress.update(processing_structures_task, description = processing_structures_task_main_description)

                        # The below print lines were just for my own understanding of how to access the data structure
                        #print(RTst_dcms[dcm_index].ROIContourSequence[int(specific_structure["Ref #"])].ContourSequence[0].ContourData)
                        #print(RTst_dcms[dcm_index].ROIContourSequence[int(specific_structure["Ref #"])].ContourSequence[1].ContourData)

                        threeDdata_zslice_list = master_structure_reference_dict[patientUID][structs][specific_structure_index]["Raw contour pts zslice list"].copy()
                        
                        total_structure_points = sum([np.shape(x)[0] for x in threeDdata_zslice_list])
                        threeDdata_array = np.empty([total_structure_points,3])                                                       

                        # build raw threeDdata for non biopsies
                        lower_bound_index = 0  
                        for index, threeDdata_zslice in enumerate(threeDdata_zslice_list):
                            current_zslice_num_points = np.size(threeDdata_zslice,0)
                            threeDdata_array[lower_bound_index:lower_bound_index + current_zslice_num_points] = threeDdata_zslice
                            lower_bound_index = lower_bound_index + current_zslice_num_points 
                        
                        
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
                        pcd_color = structs_referenced_dict[structs]['PCD color']
                        threeDdata_point_cloud = point_containment_tools.create_point_cloud(threeDdata_array, pcd_color)
                        
                        # generate delaunay triangulations 
                        """
                        deulaunay_objs_zslice_wise_list = point_containment_tools.adjacent_slice_delaunay_parallel(parallel_pool, threeDdata_zslice_list)

                        zslice1 = threeDdata_array[0,2]
                        zslice2 = threeDdata_array[-1,2]
                        delaunay_global_convex_structure_obj = point_containment_tools.delaunay_obj(threeDdata_array, threeDdata_pcd_color, zslice1, zslice2)
                        #delaunay_global_convex_structure_obj.generate_lineset()
                        """

                        threeDdata_array_fully_interpolated = interpolation_information.interpolated_pts_np_arr
                        threeDdata_array_fully_interpolated_with_end_caps = interpolation_information.interpolated_pts_with_end_caps_np_arr
                        threeDdata_array_interslice_interpolation = np.vstack(interslice_interpolation_information.interpolated_pts_list)
                        interslice_interp_pcd = point_containment_tools.create_point_cloud(threeDdata_array_interslice_interpolation, pcd_color)
                        inter_and_intra_interp_pcd = point_containment_tools.create_point_cloud(threeDdata_array_fully_interpolated, pcd_color)
                        inter_and_intra_and_end_caps_interp_pcd = point_containment_tools.create_point_cloud(threeDdata_array_fully_interpolated_with_end_caps, pcd_color)
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
                        


                        structure_info = misc_tools.specific_structure_info_dict_creator('given', specific_structure = specific_structure)

                        ### CALCULATE THE STRUCTURES VOLUME
                        ###
                        indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Calculating structure volume", total = None)
                        ###
                                
                        
                        interpolated_pts_np_arr = interslice_interpolation_information.interpolated_pts_np_arr
                        interpolated_zvals_list = interslice_interpolation_information.zslice_vals_after_interpolation_list
                        zslices_list = interslice_interpolation_information.interpolated_pts_list

                        structure_volume, maximum_distance, voxel_size_for_structure_volume_calc, binary_mask_arr, live_display = misc_tools.structure_volume_calculator(interpolated_pts_np_arr,
                            interpolated_zvals_list,
                            zslices_list,
                            structure_info,
                            plot_volume_calculation_containment_result_bool,
                            plot_binary_mask_bool,
                            voxel_size_for_structure_volume_calc_non_bx,
                            factor_for_voxel_size,
                            cupy_array_upper_limit_NxN_size_input,
                            layout_groups,
                            nearest_zslice_vals_and_indices_cupy_generic_max_size,
                            structures_progress,
                            live_display
                            )
                        
                        ###
                        indeterminate_progress_sub.update(indeterminate_task, visible = False)
                        ###
                        ###### END STRUCTURE VOLUME CALCULATION



                        ### CALCULATE THE STRUCTURES DIMENSIONS AT THE CENTROID IN X,Y,Z DIRECTIONS
                        ###
                        indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Calculating structure dimensions", total = None)
                        ###
                        
                        interpolated_pts_np_arr = interslice_interpolation_information.interpolated_pts_np_arr
                        interpolated_zvals_list = interslice_interpolation_information.zslice_vals_after_interpolation_list
                        zslices_list = interslice_interpolation_information.interpolated_pts_list
                        non_bx_structure_global_centroid = specific_structure["Structure global centroid"].copy()
                        non_bx_structure_global_centroid = np.reshape(non_bx_structure_global_centroid,(3))

                        structure_dimension_at_centroid_dict, voxel_size_for_structure_dimension_calc, live_display = misc_tools.structure_dimensions_calculator(interpolated_pts_np_arr,
                                                                                                                                                    interpolated_zvals_list,
                                                                                                                                                    zslices_list,
                                                                                                                                                    non_bx_structure_global_centroid,
                                                                                                                                                    structure_info,
                                                                                                                                                    plot_dimension_calculation_containment_result_bool,
                                                                                                                                                    voxel_size_for_structure_dimension_calc,
                                                                                                                                                    factor_for_voxel_size,
                                                                                                                                                    cupy_array_upper_limit_NxN_size_input,
                                                                                                                                                    layout_groups,
                                                                                                                                                    nearest_zslice_vals_and_indices_cupy_generic_max_size,
                                                                                                                                                    structures_progress,
                                                                                                                                                    live_display
                                                                                                                                                    )

                        ###
                        indeterminate_progress_sub.update(indeterminate_task, visible = False)
                        ###


                        """
                        ### COMPUTE POINT-WISE CURVATURE FOR DILS ONLY

                        ###
                        indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Calculating structure curvature", total = None)
                        ###
                        

                        structure_curvature_dictionary = misc_tools.determine_structure_curvature_dictionary_output(threeDdata_array_fully_interpolated_with_end_caps,
                                                                                                                    radius_for_normals_estimation,
                                                                                                                    max_nn_for_normals_estimation,
                                                                                                                    radius_for_curvature_estimation,
                                                                                                                    display_curvature_bool)



                        ###
                        indeterminate_progress_sub.update(indeterminate_task, visible = False)
                        ###
                        """





                        ### COMPUTE TRIANGLE MESH AND STRUCTURE SURFACE AREA


                        ###
                        indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Calculating structure triangle mesh and surface area", total = None)
                        live_display.refresh()
                        ###
                        #live_display.stop()

                        fully_interp_with_end_caps_structure_triangle_mesh, _ = misc_tools.compute_structure_triangle_mesh(interp_inter_slice_dist, 
                            interp_intra_slice_dist,
                            threeDdata_array_fully_interpolated_with_end_caps,
                            radius_for_normals_estimation,
                            max_nn_for_normals_estimation
                            )
                        
                        if display_structure_surface_mesh_bool == True:
                            o3d.visualization.draw_geometries([fully_interp_with_end_caps_structure_triangle_mesh], mesh_show_back_face=True)

                        structure_fully_interp_with_end_caps_surface_area = misc_tools.compute_surface_area(fully_interp_with_end_caps_structure_triangle_mesh)
                        """
                        end_caps_points = np.array(interpolation_information.endcaps_points)
                        area_voxel_size = interp_dist_caps**2
                        end_caps_area = misc_tools.compute_end_caps_area(end_caps_points,area_voxel_size)

                        structure_total_surface_area = structure_fully_interp_surface_area + end_caps_area
                        """


                        ###
                        indeterminate_progress_sub.update(indeterminate_task, visible = False)
                        ###


                        ### COMPUTE OTHER 3D SHAPE FEATURES

                        surface_volume_ratio = structure_fully_interp_with_end_caps_surface_area/structure_volume
                        sphericity = misc_tools.calculate_sphericity(structure_volume,structure_fully_interp_with_end_caps_surface_area)
                        compactness_1 = misc_tools.calculate_compactness_1(structure_volume,structure_fully_interp_with_end_caps_surface_area)
                        compactness_2 = misc_tools.calculate_compactness_2(structure_volume,structure_fully_interp_with_end_caps_surface_area)
                        spherical_disproportion = misc_tools.spherical_disproportion(structure_volume,structure_fully_interp_with_end_caps_surface_area)
                        maximum_3D_diameter = maximum_distance 

                        # Note that the eigenvectors are vstacked
                        pca_lengths_of_structure_dict, pca_eigenvectors_of_structure_arr = misc_tools.pca_lengths(binary_mask_arr)

                        # This is the same method as pyradiomics
                        equivalent_ellipse_dimensions = {"Major axis": 4*math.sqrt(pca_lengths_of_structure_dict["Major"]),
                                                            "Minor axis": 4*math.sqrt(pca_lengths_of_structure_dict["Minor"]),
                                                            "Least axis": 4*math.sqrt(pca_lengths_of_structure_dict["Least"])}

                        if show_equivalent_ellipsoid_from_pca_bool == True:
                            axis_diameters = list(equivalent_ellipse_dimensions.values())
                            misc_tools.draw_oriented_ellipse_point_cloud(threeDdata_array_fully_interpolated_with_end_caps, axis_diameters, pca_eigenvectors_of_structure_arr)


                        elongation = math.sqrt(pca_lengths_of_structure_dict["Minor"]/pca_lengths_of_structure_dict["Major"])
                        flatness = math.sqrt(pca_lengths_of_structure_dict["Least"]/pca_lengths_of_structure_dict["Major"])

                        

                        # Create dataframe of the 3d shape features
                        shape_features_3d_dictionary = {"Patient ID": [patientUID],
                                                        "Structure ID": [structureID],
                                                        "Structure type": [structs],
                                                        "Structure refnum": [structure_reference_number],
                                                        "Volume": [structure_volume],
                                                        "Surface area": [structure_fully_interp_with_end_caps_surface_area],
                                                        "Surface area to volume ratio": [surface_volume_ratio],
                                                        "Sphericity": [sphericity],
                                                        "Compactness 1": [compactness_1],
                                                        "Compactness 2": [compactness_2],
                                                        "Spherical disproportion": [spherical_disproportion],
                                                        "Maximum 3D diameter": [maximum_3D_diameter],
                                                        "PCA major": [pca_lengths_of_structure_dict["Major"]],
                                                        "PCA minor": [pca_lengths_of_structure_dict["Minor"]],
                                                        "PCA least": [pca_lengths_of_structure_dict["Least"]],
                                                        "PCA eigenvector major": [tuple(pca_eigenvectors_of_structure_arr[0,:])],
                                                        "PCA eigenvector minor": [tuple(pca_eigenvectors_of_structure_arr[1,:])],
                                                        "PCA eigenvector least": [tuple(pca_eigenvectors_of_structure_arr[2,:])],
                                                        "Major axis (equivalent ellipse)": [equivalent_ellipse_dimensions["Major axis"]],
                                                        "Minor axis (equivalent ellipse)": [equivalent_ellipse_dimensions["Minor axis"]],
                                                        "Least axis (equivalent ellipse)": [equivalent_ellipse_dimensions["Least axis"]],
                                                        "Elongation": [elongation],
                                                        "Flatness": [flatness],
                                                        "L/R dimension at centroid": structure_dimension_at_centroid_dict['X dimension length at centroid'],
                                                        "A/P dimension at centroid": structure_dimension_at_centroid_dict['Y dimension length at centroid'],
                                                        "S/I dimension at centroid": structure_dimension_at_centroid_dict['Z dimension length at centroid']
                                                        }



                        

                        shape_features_dataframe = pandas.DataFrame(shape_features_3d_dictionary)
                        shape_features_dataframe = dataframe_builders.convert_columns_to_categorical_and_downcast(shape_features_dataframe, threshold=0.25)


                        # store all calculated quantities
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Raw contour pts"] = threeDdata_array
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Equal num zslice contour pts"] = threeDdata_equal_pt_zslice_list
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Inter-slice interpolation information"] = interslice_interpolation_information                        
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Intra-slice interpolation information"] = interpolation_information
                        #master_structure_reference_dict[patientUID][structs][specific_structure_index]["Delaunay triangulation zslice-wise list"] = deulaunay_objs_zslice_wise_list # DEPRECATED
                        #master_structure_reference_dict[patientUID][structs][specific_structure_index]["Delaunay triangulation global structure"] = delaunay_global_convex_structure_obj # DEPRECATED
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Maximum pairwise distance"] = maximum_3D_diameter
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Structure volume"] = structure_volume
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Voxel size for structure volume calc"] = voxel_size_for_structure_volume_calc
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Structure dimension at centroid dict"] = structure_dimension_at_centroid_dict
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Voxel size for structure dimension calc"] = voxel_size_for_structure_dimension_calc
                        #master_structure_reference_dict[patientUID][structs][specific_structure_index]["Structure curvature dict"] = structure_curvature_dictionary
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Structure surface area"] = structure_fully_interp_with_end_caps_surface_area
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Structure features dataframe"] = shape_features_dataframe
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Point cloud raw"] = threeDdata_point_cloud
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Interpolated structure point cloud dict"] = interpolated_pcd_dict
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Structure OPEN3D triangle mesh object"] = fully_interp_with_end_caps_structure_triangle_mesh


                        structures_progress.update(processing_structures_task, advance=1)

                    structures_progress.remove_task(processing_structures_task)
                    patients_progress.update(processing_patients_task, advance=1)
                    completed_progress.update(processing_patients_task_completed, advance=1)
                patients_progress.update(processing_patients_task, visible=False)
                completed_progress.update(processing_patients_task_completed,  visible=True)



















                #live_display.stop()

                #print('test')



                #live_display.start()




















                ##### PREPROCESSING DILs
                #live_display.stop()

                patientUID_default = "Initializing"
                processing_patients_task_main_description = "[red]Processing patient DILs [{}]...".format(patientUID_default)
                processing_patients_task_completed_main_description = "[green]Processing patient DILs"
                processing_patients_task = patients_progress.add_task(processing_patients_task_main_description, total=master_structure_info_dict["Global"]["Num cases"])
                processing_patients_task_completed = completed_progress.add_task(processing_patients_task_completed_main_description, total=master_structure_info_dict["Global"]["Num cases"], visible = False)

                for patientUID,pydicom_item in master_structure_reference_dict.items():
                    processing_patients_task_main_description = "[red]Processing patient DILs [{}]...".format(patientUID)
                    patients_progress.update(processing_patients_task, description = processing_patients_task_main_description)
                    
                    structureID_default = "Initializing"
                    #num_general_structs_patient_specific = master_structure_info_dict["By patient"][patientUID][all_ref_key]["Total num structs"]
                    num_total_structs_patient_specific = master_structure_info_dict["By patient"][patientUID][all_ref_key]["Total num structs"]
                    num_bx_structs_patient_specific = master_structure_info_dict["By patient"][patientUID][bx_ref]["Num structs"]
                    
                    num_dils = len(pydicom_item[dil_ref])


                    ### SELECT PROSTATE, OR DEFAULT TO ORIGIN FOR PROSTATE COM IF NONE FOUND
                    #selected_prostate_info, message_string, prostate_found_bool, num_prostates_found = misc_tools.specific_structure_selector(pydicom_item,
                    #                                                                                                                        oar_ref,
                    #                                                                                                                        prostate_contour_name)   
                    
                    sp_patient_selected_structure_info_dataframe = pydicom_item[all_ref_key]["Multi-structure pre-processing output dataframes dict"]["Selected structures"]

                    
                    
                    processing_structures_task_main_description = "[cyan]Processing structures [{},{}]...".format(patientUID,structureID_default)
                    processing_structures_task = structures_progress.add_task(processing_structures_task_main_description, total=num_dils)


                    structs = dil_ref
                    for specific_structure_index, specific_structure in enumerate(pydicom_item[structs]):
                        structureID = specific_structure["ROI"]
                        structure_reference_number = specific_structure["Ref #"]
                        processing_structures_task_main_description = "[cyan]Processing structures [{},{}]...".format(patientUID,structureID)
                        structures_progress.update(processing_structures_task, description = processing_structures_task_main_description)

                        # The below print lines were just for my own understanding of how to access the data structure
                        #print(RTst_dcms[dcm_index].ROIContourSequence[int(specific_structure["Ref #"])].ContourSequence[0].ContourData)
                        #print(RTst_dcms[dcm_index].ROIContourSequence[int(specific_structure["Ref #"])].ContourSequence[1].ContourData)

                        threeDdata_zslice_list = master_structure_reference_dict[patientUID][structs][specific_structure_index]["Raw contour pts zslice list"].copy()
                        
                        total_structure_points = sum([np.shape(x)[0] for x in threeDdata_zslice_list])
                        threeDdata_array = np.empty([total_structure_points,3])                                                       

                        # build raw threeDdata for non biopsies
                        lower_bound_index = 0  
                        for index, threeDdata_zslice in enumerate(threeDdata_zslice_list):
                            current_zslice_num_points = np.size(threeDdata_zslice,0)
                            threeDdata_array[lower_bound_index:lower_bound_index + current_zslice_num_points] = threeDdata_zslice
                            lower_bound_index = lower_bound_index + current_zslice_num_points 
                        
                        
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
                        pcd_color = structs_referenced_dict[structs]['PCD color']

                        threeDdata_point_cloud = point_containment_tools.create_point_cloud(threeDdata_array, pcd_color)
                        
                        # generate delaunay triangulations 
                        """
                        deulaunay_objs_zslice_wise_list = point_containment_tools.adjacent_slice_delaunay_parallel(parallel_pool, threeDdata_zslice_list)

                        zslice1 = threeDdata_array[0,2]
                        zslice2 = threeDdata_array[-1,2]
                        delaunay_global_convex_structure_obj = point_containment_tools.delaunay_obj(threeDdata_array, threeDdata_pcd_color, zslice1, zslice2)
                        #delaunay_global_convex_structure_obj.generate_lineset()
                        """

                        threeDdata_array_fully_interpolated = interpolation_information.interpolated_pts_np_arr
                        threeDdata_array_fully_interpolated_with_end_caps = interpolation_information.interpolated_pts_with_end_caps_np_arr
                        threeDdata_array_interslice_interpolation = np.vstack(interslice_interpolation_information.interpolated_pts_list)
                        interslice_interp_pcd = point_containment_tools.create_point_cloud(threeDdata_array_interslice_interpolation, pcd_color)
                        inter_and_intra_interp_pcd = point_containment_tools.create_point_cloud(threeDdata_array_fully_interpolated, pcd_color)
                        inter_and_intra_and_end_caps_interp_pcd = point_containment_tools.create_point_cloud(threeDdata_array_fully_interpolated_with_end_caps, pcd_color)
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
                        


                        structure_info = misc_tools.specific_structure_info_dict_creator('given', specific_structure = specific_structure) 


                        ### CALCULATE THE STRUCTURES VOLUME
                        ###
                        indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Calculating structure volume", total = None)
                        ###
                                
                        
                        interpolated_pts_np_arr = interslice_interpolation_information.interpolated_pts_np_arr
                        interpolated_zvals_list = interslice_interpolation_information.zslice_vals_after_interpolation_list
                        zslices_list = interslice_interpolation_information.interpolated_pts_list

                        structure_volume, maximum_distance, voxel_size_for_structure_volume_calc, binary_mask_arr, live_display = misc_tools.structure_volume_calculator(interpolated_pts_np_arr,
                            interpolated_zvals_list,
                            zslices_list,
                            structure_info,
                            plot_volume_calculation_containment_result_bool,
                            plot_binary_mask_bool,
                            voxel_size_for_structure_volume_calc_non_bx,
                            factor_for_voxel_size,
                            cupy_array_upper_limit_NxN_size_input,
                            layout_groups,
                            nearest_zslice_vals_and_indices_cupy_generic_max_size,
                            structures_progress,
                            live_display
                            )
                        
                        ###
                        indeterminate_progress_sub.update(indeterminate_task, visible = False)
                        ###
                        ###### END STRUCTURE VOLUME CALCULATION



                        ### CALCULATE THE STRUCTURES DIMENSIONS AT THE CENTROID IN X,Y,Z DIRECTIONS
                        ###
                        indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Calculating structure dimensions", total = None)
                        ###
                        
                        interpolated_pts_np_arr = interslice_interpolation_information.interpolated_pts_np_arr
                        interpolated_zvals_list = interslice_interpolation_information.zslice_vals_after_interpolation_list
                        zslices_list = interslice_interpolation_information.interpolated_pts_list
                        non_bx_structure_global_centroid = specific_structure["Structure global centroid"].copy()
                        non_bx_structure_global_centroid = np.reshape(non_bx_structure_global_centroid,(3))

                        structure_dimension_at_centroid_dict, voxel_size_for_structure_dimension_calc, live_display = misc_tools.structure_dimensions_calculator(interpolated_pts_np_arr,
                                                                                                                                                    interpolated_zvals_list,
                                                                                                                                                    zslices_list,
                                                                                                                                                    non_bx_structure_global_centroid,
                                                                                                                                                    structure_info,
                                                                                                                                                    plot_dimension_calculation_containment_result_bool,
                                                                                                                                                    voxel_size_for_structure_dimension_calc,
                                                                                                                                                    factor_for_voxel_size,
                                                                                                                                                    cupy_array_upper_limit_NxN_size_input,
                                                                                                                                                    layout_groups,
                                                                                                                                                    nearest_zslice_vals_and_indices_cupy_generic_max_size,
                                                                                                                                                    structures_progress,
                                                                                                                                                    live_display
                                                                                                                                                    )

                        ###
                        indeterminate_progress_sub.update(indeterminate_task, visible = False)
                        ###


                        """
                        ### COMPUTE POINT-WISE CURVATURE FOR DILS ONLY

                        ###
                        indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Calculating structure curvature", total = None)
                        ###
                        

                        structure_curvature_dictionary = misc_tools.determine_structure_curvature_dictionary_output(threeDdata_array_fully_interpolated_with_end_caps,
                                                                                                                    radius_for_normals_estimation,
                                                                                                                    max_nn_for_normals_estimation,
                                                                                                                    radius_for_curvature_estimation,
                                                                                                                    display_curvature_bool)



                        ###
                        indeterminate_progress_sub.update(indeterminate_task, visible = False)
                        ###
                        """





                        ### COMPUTE TRIANGLE MESH AND STRUCTURE SURFACE AREA


                        ###
                        indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Calculating structure triangle mesh and surface area", total = None)
                        live_display.refresh()
                        ###
                        #live_display.stop()

                        fully_interp_with_end_caps_structure_triangle_mesh, _ = misc_tools.compute_structure_triangle_mesh(interp_inter_slice_dist, 
                            interp_intra_slice_dist,
                            threeDdata_array_fully_interpolated_with_end_caps,
                            radius_for_normals_estimation,
                            max_nn_for_normals_estimation
                            )
                        
                        if display_structure_surface_mesh_bool == True:
                            o3d.visualization.draw_geometries([fully_interp_with_end_caps_structure_triangle_mesh], mesh_show_back_face=True)

                        structure_fully_interp_with_end_caps_surface_area = misc_tools.compute_surface_area(fully_interp_with_end_caps_structure_triangle_mesh)
                        """
                        end_caps_points = np.array(interpolation_information.endcaps_points)
                        area_voxel_size = interp_dist_caps**2
                        end_caps_area = misc_tools.compute_end_caps_area(end_caps_points,area_voxel_size)

                        structure_total_surface_area = structure_fully_interp_surface_area + end_caps_area
                        """


                        ###
                        indeterminate_progress_sub.update(indeterminate_task, visible = False)
                        ###


                        ### COMPUTE OTHER 3D SHAPE FEATURES

                        surface_volume_ratio = structure_fully_interp_with_end_caps_surface_area/structure_volume
                        sphericity = misc_tools.calculate_sphericity(structure_volume,structure_fully_interp_with_end_caps_surface_area)
                        compactness_1 = misc_tools.calculate_compactness_1(structure_volume,structure_fully_interp_with_end_caps_surface_area)
                        compactness_2 = misc_tools.calculate_compactness_2(structure_volume,structure_fully_interp_with_end_caps_surface_area)
                        spherical_disproportion = misc_tools.spherical_disproportion(structure_volume,structure_fully_interp_with_end_caps_surface_area)
                        maximum_3D_diameter = maximum_distance 

                        # Note that the eigenvectors are vstacked
                        pca_lengths_of_structure_dict, pca_eigenvectors_of_structure_arr = misc_tools.pca_lengths(binary_mask_arr)

                        # This is the same method as pyradiomics
                        equivalent_ellipse_dimensions = {"Major axis": 4*math.sqrt(pca_lengths_of_structure_dict["Major"]),
                                                            "Minor axis": 4*math.sqrt(pca_lengths_of_structure_dict["Minor"]),
                                                            "Least axis": 4*math.sqrt(pca_lengths_of_structure_dict["Least"])}

                        if show_equivalent_ellipsoid_from_pca_bool == True:
                            axis_diameters = list(equivalent_ellipse_dimensions.values())
                            misc_tools.draw_oriented_ellipse_point_cloud(threeDdata_array_fully_interpolated_with_end_caps, axis_diameters, pca_eigenvectors_of_structure_arr)


                        elongation = math.sqrt(pca_lengths_of_structure_dict["Minor"]/pca_lengths_of_structure_dict["Major"])
                        flatness = math.sqrt(pca_lengths_of_structure_dict["Least"]/pca_lengths_of_structure_dict["Major"])

                        
                        selected_prostate_df = sp_patient_selected_structure_info_dataframe[sp_patient_selected_structure_info_dataframe["Struct ref type"] == oar_ref]
                        selected_prostate_info = selected_prostate_df.to_dict('records')[0]

                        prostate_found_bool = selected_prostate_info["Struct found bool"]
                            

                        live_display.refresh()
                        if prostate_found_bool == True:
                            prostate_structure_index = selected_prostate_info["Index number"]
                            prostate_structure = pydicom_item[oar_ref][prostate_structure_index]
                            prostate_structure_global_centroid = prostate_structure["Structure global centroid"].copy().reshape((3))
                            prostate_dimension_at_centroid_dict = prostate_structure["Structure dimension at centroid dict"]
                            prostate_z_dimension_length_at_centroid = prostate_dimension_at_centroid_dict["Z dimension length at centroid"]
                        
                            # note that distance_to_mid_gland_threshold should be a positive quantity for the position classifier function below!
                            distance_to_mid_gland_threshold = abs(prostate_z_dimension_length_at_centroid/6) 

                            # determine dil location within prostate 
                            # Calculate DIL location in prostate reference frame 
                            specific_structure_global_centroid = specific_structure["Structure global centroid"][0]
                            specific_structure_global_centroid_in_prostate_frame = specific_structure_global_centroid - prostate_structure_global_centroid

                            # despite the function name, it can be used on any structure, not just biopsies
                            dil_prostate_position_dict = misc_tools.bx_position_classifier_in_prostate_frame_sextant(specific_structure_global_centroid_in_prostate_frame,
                                        distance_to_mid_gland_threshold)
                        else: 
                            dil_prostate_position_dict = {"LR": None,"AP": None,"SI": None}

                        # Create dataframe of the 3d shape features
                        shape_features_3d_dictionary = {"Patient ID": [patientUID],
                                                        "Structure ID": [structureID],
                                                        "Structure type": [structs],
                                                        "Structure refnum": [structure_reference_number],
                                                        "Volume": [structure_volume],
                                                        "Surface area": [structure_fully_interp_with_end_caps_surface_area],
                                                        "Surface area to volume ratio": [surface_volume_ratio],
                                                        "Sphericity": [sphericity],
                                                        "Compactness 1": [compactness_1],
                                                        "Compactness 2": [compactness_2],
                                                        "Spherical disproportion": [spherical_disproportion],
                                                        "Maximum 3D diameter": [maximum_3D_diameter],
                                                        "PCA major": [pca_lengths_of_structure_dict["Major"]],
                                                        "PCA minor": [pca_lengths_of_structure_dict["Minor"]],
                                                        "PCA least": [pca_lengths_of_structure_dict["Least"]],
                                                        "PCA eigenvector major": [tuple(pca_eigenvectors_of_structure_arr[0,:])],
                                                        "PCA eigenvector minor": [tuple(pca_eigenvectors_of_structure_arr[1,:])],
                                                        "PCA eigenvector least": [tuple(pca_eigenvectors_of_structure_arr[2,:])],
                                                        "Major axis (equivalent ellipse)": [equivalent_ellipse_dimensions["Major axis"]],
                                                        "Minor axis (equivalent ellipse)": [equivalent_ellipse_dimensions["Minor axis"]],
                                                        "Least axis (equivalent ellipse)": [equivalent_ellipse_dimensions["Least axis"]],
                                                        "Elongation": [elongation],
                                                        "Flatness": [flatness],
                                                        "L/R dimension at centroid": structure_dimension_at_centroid_dict['X dimension length at centroid'],
                                                        "A/P dimension at centroid": structure_dimension_at_centroid_dict['Y dimension length at centroid'],
                                                        "S/I dimension at centroid": structure_dimension_at_centroid_dict['Z dimension length at centroid'],
                                                        "DIL centroid (X, prostate frame)": specific_structure_global_centroid_in_prostate_frame[0],
                                                        "DIL centroid (Y, prostate frame)": specific_structure_global_centroid_in_prostate_frame[1],
                                                        "DIL centroid (Z, prostate frame)": specific_structure_global_centroid_in_prostate_frame[2],
                                                        "DIL centroid distance (prostate frame)": np.linalg.norm(specific_structure_global_centroid_in_prostate_frame),
                                                        "DIL prostate sextant (LR)": dil_prostate_position_dict["LR"],
                                                        "DIL prostate sextant (AP)": dil_prostate_position_dict["AP"],
                                                        "DIL prostate sextant (SI)": dil_prostate_position_dict["SI"]
                                                        }



                        

                        shape_features_dataframe = pandas.DataFrame(shape_features_3d_dictionary)
                        shape_features_dataframe = dataframe_builders.convert_columns_to_categorical_and_downcast(shape_features_dataframe, threshold=0.25)


                        # store all calculated quantities
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Raw contour pts"] = threeDdata_array
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Equal num zslice contour pts"] = threeDdata_equal_pt_zslice_list
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Inter-slice interpolation information"] = interslice_interpolation_information                        
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Intra-slice interpolation information"] = interpolation_information
                        #master_structure_reference_dict[patientUID][structs][specific_structure_index]["Delaunay triangulation zslice-wise list"] = deulaunay_objs_zslice_wise_list # DEPRECATED
                        #master_structure_reference_dict[patientUID][structs][specific_structure_index]["Delaunay triangulation global structure"] = delaunay_global_convex_structure_obj # DEPRECATED
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Maximum pairwise distance"] = maximum_3D_diameter
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Structure volume"] = structure_volume
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Voxel size for structure volume calc"] = voxel_size_for_structure_volume_calc
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Structure dimension at centroid dict"] = structure_dimension_at_centroid_dict
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Voxel size for structure dimension calc"] = voxel_size_for_structure_dimension_calc
                        #master_structure_reference_dict[patientUID][structs][specific_structure_index]["Structure curvature dict"] = structure_curvature_dictionary
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Structure surface area"] = structure_fully_interp_with_end_caps_surface_area
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Structure features dataframe"] = shape_features_dataframe
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Point cloud raw"] = threeDdata_point_cloud
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Interpolated structure point cloud dict"] = interpolated_pcd_dict
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Structure OPEN3D triangle mesh object"] = fully_interp_with_end_caps_structure_triangle_mesh

                        structures_progress.update(processing_structures_task, advance=1)

                    structures_progress.remove_task(processing_structures_task)
                    patients_progress.update(processing_patients_task, advance=1)
                    completed_progress.update(processing_patients_task_completed, advance=1)
                patients_progress.update(processing_patients_task, visible=False)
                completed_progress.update(processing_patients_task_completed,  visible=True)    





                ########## PERFORM BIOPSY DIL OPTIMIZATION


                #live_display.stop()

                patientUID_default = "Initializing"
                processing_patients_task_main_description = "[red]Optimizing Bx location within DILs [{}]...".format(patientUID_default)
                processing_patients_task_completed_main_description = "[green]Optimizing Bx location within DILs"
                processing_patients_task = patients_progress.add_task(processing_patients_task_main_description, total=master_structure_info_dict["Global"]["Num cases"])
                processing_patients_task_completed = completed_progress.add_task(processing_patients_task_completed_main_description, total=master_structure_info_dict["Global"]["Num cases"], visible = False)

                for patientUID,pydicom_item in master_structure_reference_dict.items():
                    processing_patients_task_main_description = "[red]Optimizing Bx location within DILs [{}]...".format(patientUID)
                    patients_progress.update(processing_patients_task, description = processing_patients_task_main_description)
                    #####
                    
                    ### SELECT PROSTATE, OR DEFAULT TO ORIGIN FOR PROSTATE COM IF NONE FOUND
                    # selected_prostate_info, message_string, prostate_found_bool, num_prostates_found = misc_tools.specific_structure_selector(pydicom_item,
                    #                         oar_ref,
                    #                         prostate_contour_name)    

                    sp_patient_selected_structure_info_dataframe = pydicom_item[all_ref_key]["Multi-structure pre-processing output dataframes dict"]["Selected structures"]                 

                    specific_prostate_info_df = sp_patient_selected_structure_info_dataframe[sp_patient_selected_structure_info_dataframe["Struct ref type"] == oar_ref]
                    selected_prostate_info = specific_prostate_info_df.to_dict('records')[0]

                    prostate_ID = selected_prostate_info["Structure ID"]
                    prostate_ref_type = selected_prostate_info["Struct ref type"]
                    prostate_ref_num = selected_prostate_info["Dicom ref num"]
                    prostate_structure_index = selected_prostate_info["Index number"]
                    prostate_found_bool = selected_prostate_info["Struct found bool"]


                    if prostate_found_bool == True:
                        prostate_centroid = pydicom_item[prostate_ref_type][prostate_structure_index]["Structure global centroid"].reshape(3)
                    else: 
                        important_info.add_text_line('Prostate not found! Defaulting prostate centroid to Zero-vector')
                        prostate_centroid = np.array([0,0,0])



                    ## GENERATE LATTICE ENCOMPASSING ALL GEOMETRIES
                    # add the dils!
                    list_of_all_dils_interpolated_pts = []
                    for specific_dil_structure_index, specific_dil_structure in enumerate(pydicom_item[dil_ref]):
                        sp_dil_interslice_interpolation_information = specific_dil_structure["Inter-slice interpolation information"]
                        sp_dil_interpolated_pts_np_arr = sp_dil_interslice_interpolation_information.interpolated_pts_np_arr
                        list_of_all_dils_interpolated_pts.append(sp_dil_interpolated_pts_np_arr)
                    #all_dils_interpolated_pts = np.vstack(list_of_all_dils_interpolated_pts)

                    # add the OARs!
                    list_of_all_oar_interpolated_pts = []
                    for specific_oar_structure in pydicom_item[oar_ref]:
                        oar_interslice_interpolation_information = specific_oar_structure["Inter-slice interpolation information"]
                        oar_interpolated_pts_np_arr = oar_interslice_interpolation_information.interpolated_pts_np_arr
                        list_of_all_oar_interpolated_pts.append(oar_interpolated_pts_np_arr)

                    all_geometries_list_of_interpolated_pts = list_of_all_dils_interpolated_pts + list_of_all_oar_interpolated_pts
                    all_geometries_interpolated_pts = np.vstack(all_geometries_list_of_interpolated_pts)

                    # Before, only added prostate, but better to add ALL OARs
                    """
                    if prostate_found_bool == True:
                        prostate_interslice_interpolation_information = pydicom_item[prostate_ref_type][prostate_structure_index]["Inter-slice interpolation information"]
                        prostate_interpolated_pts_np_arr = prostate_interslice_interpolation_information.interpolated_pts_np_arr
                        all_geometries_interpolated_pts = np.vstack([all_dils_interpolated_pts,prostate_interpolated_pts_np_arr])
                    else: 
                        all_geometries_interpolated_pts = all_dils_interpolated_pts
                    """

                    # all geometries means dils + prostate (if a prostate could be found!)
                    all_geometries_interpolated_pts_point_cloud = point_containment_tools.create_point_cloud(all_geometries_interpolated_pts)
                    interpolated_pts_point_cloud_color = np.array([0,0,1])
                    all_geometries_interpolated_pts_point_cloud.paint_uniform_color(interpolated_pts_point_cloud_color)

                    all_geometries_axis_aligned_bounding_box = all_geometries_interpolated_pts_point_cloud.get_axis_aligned_bounding_box()
                    all_geometries_axis_aligned_bounding_box_points_arr = np.asarray(all_geometries_axis_aligned_bounding_box.get_box_points())
                    all_geometries_bounding_box_color = np.array([0,0,0], dtype=float)
                    all_geometries_axis_aligned_bounding_box.color = all_geometries_bounding_box_color
                    all_geometries_max_bounds = np.amax(all_geometries_axis_aligned_bounding_box_points_arr, axis=0)
                    all_geometries_min_bounds = np.amin(all_geometries_axis_aligned_bounding_box_points_arr, axis=0)

                    lattice_sizex = int(math.ceil(abs(all_geometries_max_bounds[0]-all_geometries_min_bounds[0])/voxel_size_for_dil_optimizer_grid) + 1)
                    lattice_sizey = int(math.ceil(abs(all_geometries_max_bounds[1]-all_geometries_min_bounds[1])/voxel_size_for_dil_optimizer_grid) + 1)
                    lattice_sizez = int(math.ceil(abs(all_geometries_max_bounds[2]-all_geometries_min_bounds[2])/voxel_size_for_dil_optimizer_grid) + 1)
                    origin = all_geometries_min_bounds

                    # generate cubic lattice of points
                    all_geometries_centered_cubic_lattice_arr = MC_simulator_convex.generate_cubic_lattice(voxel_size_for_dil_optimizer_grid, 
                                                                                                        lattice_sizex,
                                                                                                        lattice_sizey,
                                                                                                        lattice_sizez,
                                                                                                        origin)


                    # CREATE A COPY TO REMOVE THE POINTS CONTAINED IN THE DILS!
                    #all_geometries_centered_cubic_lattice_with_dil_points_removed_arr = all_geometries_centered_cubic_lattice_arr.copy()
                    

                    # Create empty dataframe for all contained points in lattice
                    #dil_contained_points_df = pandas.DataFrame()
                    
                    # Make an empty array to keep track of all the points 
                    #centered_cubic_lattice_points_only_contained_in_ANY_dil_arr = np.empty((0, 3))

                    #####
                    structureID_default = "Initializing"
                    num_dil_structs_patient_specific = master_structure_info_dict["By patient"][patientUID][dil_ref]["Num structs"]
                    processing_structures_task_main_description = "[cyan]Processing structures [{},{}]...".format(patientUID,structureID_default)
                    processing_structures_task = structures_progress.add_task(processing_structures_task_main_description, total=num_dil_structs_patient_specific)
                    for specific_dil_structure_index, specific_dil_structure in enumerate(pydicom_item[dil_ref]):
                        structureID_dil = specific_dil_structure["ROI"]
                        structure_reference_number_dil = specific_dil_structure["Ref #"]
                        processing_structures_task_main_description = "[cyan]Processing structures [{},{}]...".format(patientUID,structureID_dil)
                        structures_progress.update(processing_structures_task, description = processing_structures_task_main_description)

                        ### FIND OPTIMAL POSITION FOR BIOPSY SAMPLING (DIL ONLY)
                        structure_info = misc_tools.specific_structure_info_dict_creator('given', specific_structure = specific_dil_structure)

                        interslice_interpolation_information = specific_dil_structure["Inter-slice interpolation information"]
                        interpolated_pts_np_arr = interslice_interpolation_information.interpolated_pts_np_arr
                        interpolated_zvals_list = interslice_interpolation_information.zslice_vals_after_interpolation_list
                        zslices_list = interslice_interpolation_information.interpolated_pts_list

                        # create geoseries of the dil structure for containment tests
                        max_zval = max(interpolated_zvals_list)
                        min_zval = min(interpolated_zvals_list)
                        zslices_polygons_list = [Polygon(polygon[:,0:2]) for polygon in zslices_list]
                        zslices_polygons_cuspatial_geoseries = cuspatial.GeoSeries(geopandas.GeoSeries(zslices_polygons_list))

                        # Extract the dil centroid
                        dil_global_centroid = specific_dil_structure["Structure global centroid"]


                        ### CONSTRUCT THE LATTICE POINTS TO PASS TO THE OPTIMIZER FUNCTION
                        all_geometries_centered_cubic_lattice_arr_XY = all_geometries_centered_cubic_lattice_arr[:,0:2]
                        all_geometries_centered_cubic_lattice_arr_Z = all_geometries_centered_cubic_lattice_arr[:,2]

                        
                        #nearest_interpolated_zslice_for_test_lattice_index_array, nearest_interpolated_zslice_for_test_lattice_vals_array = point_containment_tools.take_closest_cupy(interpolated_zvals_list, all_geometries_centered_cubic_lattice_arr_Z)

                        nearest_interpolated_zslice_for_test_lattice_index_array, nearest_interpolated_zslice_for_test_lattice_vals_array = point_containment_tools.nearest_zslice_vals_and_indices_cupy_generic(interpolated_zvals_list, 
                                                                                                                                                            all_geometries_centered_cubic_lattice_arr_Z,
                                                                                                                                                            nearest_zslice_vals_and_indices_cupy_generic_max_size,
                                                                                                                                                            structures_progress
                                                                                                                                                            )
                        
                        
                        all_geometries_centered_cubic_lattice_XY_interleaved_1darr = all_geometries_centered_cubic_lattice_arr_XY.flatten()
                        all_geometries_centered_cubic_lattice_XY_cuspatial_geoseries_points = cuspatial.GeoSeries.from_points_xy(all_geometries_centered_cubic_lattice_XY_interleaved_1darr)

                        # Test point containment to remove points from the potential optimization testing point lattice that are not inside the DIL
                        containment_info_for_all_lattice_points_grand_pandas_dataframe, live_display = point_containment_tools.cuspatial_points_contained_generic_numpy_pandas(zslices_polygons_cuspatial_geoseries,
                            all_geometries_centered_cubic_lattice_XY_cuspatial_geoseries_points, 
                            all_geometries_centered_cubic_lattice_arr, 
                            nearest_interpolated_zslice_for_test_lattice_index_array,
                            nearest_interpolated_zslice_for_test_lattice_vals_array,
                            max_zval,
                            min_zval,
                            structure_info,
                            layout_groups,
                            live_display,
                            structures_progress,
                            upper_limit_size_input = cupy_array_upper_limit_NxN_size_input,
                            )
                        del nearest_interpolated_zslice_for_test_lattice_index_array
                        del nearest_interpolated_zslice_for_test_lattice_vals_array
                        live_display.refresh()

                        #containment_info_for_all_lattice_points_grand_pandas_dataframe = containment_info_for_all_lattice_points_grand_cudf_dataframe.to_pandas()
                        
                        containment_info_for_lattice_points_in_sp_dil_grand_pandas_dataframe = containment_info_for_all_lattice_points_grand_pandas_dataframe.drop(containment_info_for_all_lattice_points_grand_pandas_dataframe[containment_info_for_all_lattice_points_grand_pandas_dataframe["Pt contained bool"] == False].index).reset_index()
                        containment_info_for_lattice_points_NOT_in_sp_dil_grand_pandas_dataframe = containment_info_for_all_lattice_points_grand_pandas_dataframe.drop(containment_info_for_all_lattice_points_grand_pandas_dataframe[containment_info_for_all_lattice_points_grand_pandas_dataframe["Pt contained bool"] == True].index).reset_index()
                        del containment_info_for_all_lattice_points_grand_pandas_dataframe

                        #dil_contained_points_df = dil_contained_points_df.append(containment_info_for_lattice_points_in_sp_dil_grand_pandas_dataframe)

                        centered_cubic_lattice_points_contained_only_in_sp_dil_arr = all_geometries_centered_cubic_lattice_arr[containment_info_for_lattice_points_in_sp_dil_grand_pandas_dataframe["index"].to_numpy()]
                        centered_cubic_lattice_points_NOT_contained_only_in_sp_dil_arr = all_geometries_centered_cubic_lattice_arr[containment_info_for_lattice_points_NOT_in_sp_dil_grand_pandas_dataframe["index"].to_numpy()]
                        #centered_cubic_lattice_points_NOT_contained_in_ANY_dil_arr = centered_cubic_lattice_points_NOT_contained_in_ANY_dil_arr[containment_info_for_lattice_points_NOT_in_sp_dil_grand_pandas_dataframe["index"].to_numpy()]
                        del containment_info_for_lattice_points_in_sp_dil_grand_pandas_dataframe

                        optimal_locations_dataframe, potential_optimal_locations_dataframe, zero_locations_dataframe, live_display = biopsy_optimizer.find_dil_optimal_sampling_position(specific_dil_structure,
                                                                                                        optimal_normal_dist_option,
                                                                                                        bias_LR_multiplier,
                                                                                                        bias_AP_multiplier,
                                                                                                        bias_SI_multiplier,
                                                                                                        patientUID,
                                                                                                        structs_referenced_dict,
                                                                                                        bx_ref,
                                                                                                        dil_ref,
                                                                                                        interpolated_pts_np_arr,
                                                                                                        interpolated_zvals_list,
                                                                                                        zslices_list,
                                                                                                        structure_info,
                                                                                                        dil_global_centroid,
                                                                                                        voxel_size_for_dil_optimizer_grid,
                                                                                                        num_normal_dist_points_for_biopsy_optimizer,
                                                                                                        normal_dist_sigma_factor_biopsy_optimizer,
                                                                                                        prostate_centroid,
                                                                                                        selected_prostate_info,
                                                                                                        plot_each_normal_dist_containment_result_bool,
                                                                                                        plot_optimization_point_lattice_bool,
                                                                                                        show_optimization_point_bool,
                                                                                                        layout_groups,
                                                                                                        live_display,
                                                                                                        cupy_array_upper_limit_NxN_size_input,
                                                                                                        numpy_array_upper_limit_NxN_size_input,
                                                                                                        nearest_zslice_vals_and_indices_cupy_generic_max_size,
                                                                                                        nearest_zslice_vals_and_indices_numpy_generic_max_size,
                                                                                                        structures_progress,
                                                                                                        test_lattice_arr = centered_cubic_lattice_points_contained_only_in_sp_dil_arr,
                                                                                                        all_points_to_set_to_zero_arr = centered_cubic_lattice_points_NOT_contained_only_in_sp_dil_arr # This was added to make the "search" volume to include the entire volume
                                                                                                        )


                        # add constant plane indices 

                        potential_optimal_locations_dataframe = misc_tools.assign_plane_indices(potential_optimal_locations_dataframe, 
                                                voxel_size_for_dil_optimizer_grid, 
                                                'Test location (Prostate centroid origin) (X)',
                                                'Test location (Prostate centroid origin) (Y)',
                                                'Test location (Prostate centroid origin) (Z)')
                        
                        zero_locations_dataframe = misc_tools.assign_plane_indices(zero_locations_dataframe, 
                                                voxel_size_for_dil_optimizer_grid, 
                                                'Test location (Prostate centroid origin) (X)',
                                                'Test location (Prostate centroid origin) (Y)',
                                                'Test location (Prostate centroid origin) (Z)')

                        optimal_locations_dataframe = misc_tools.assign_plane_indices(optimal_locations_dataframe, 
                                                voxel_size_for_dil_optimizer_grid, 
                                                'Test location (Prostate centroid origin) (X)',
                                                'Test location (Prostate centroid origin) (Y)',
                                                'Test location (Prostate centroid origin) (Z)')



                        #del centered_cubic_lattice_points_contained_only_in_sp_dil_arr
                        live_display.refresh()
                        
                        # Save the dil centroid optimization result in a seperate dataframe 
                        dil_centroids_optimization_locations_dataframe = pandas.DataFrame(potential_optimal_locations_dataframe.loc[[0],:])

                        dil_centroids_optimization_locations_dataframe = misc_tools.assign_plane_indices(dil_centroids_optimization_locations_dataframe, 
                                                voxel_size_for_dil_optimizer_grid, 
                                                'Test location (Prostate centroid origin) (X)',
                                                'Test location (Prostate centroid origin) (Y)',
                                                'Test location (Prostate centroid origin) (Z)')
                        
                        dil_centroids_optimization_locations_dataframe = dataframe_builders.convert_columns_to_categorical_and_downcast(dil_centroids_optimization_locations_dataframe, threshold=0.25, ignore_types=(np.floating,))
                        optimal_locations_dataframe = dataframe_builders.convert_columns_to_categorical_and_downcast(optimal_locations_dataframe, threshold=0.25, ignore_types=(np.floating,))
                        potential_optimal_locations_dataframe = dataframe_builders.convert_columns_to_categorical_and_downcast(potential_optimal_locations_dataframe, threshold=0.25, ignore_types=(np.floating,))
                        zero_locations_dataframe = dataframe_builders.convert_columns_to_categorical_and_downcast(zero_locations_dataframe, threshold=0.25, ignore_types=(np.floating,))

                        #potential_optimal_locations_dataframe_centroid_dropped = potential_optimal_locations_dataframe.drop([0])
                        #centered_cubic_lattice_points_only_contained_in_ANY_dil_arr = np.vstack([centered_cubic_lattice_points_only_contained_in_ANY_dil_arr,potential_optimal_locations_dataframe_centroid_dropped])


                        specific_dil_structure["Biopsy optimization: DIL centroid optimal biopsy location dataframe"] = dil_centroids_optimization_locations_dataframe
                        specific_dil_structure["Biopsy optimization: Optimal biopsy location dataframe"] = optimal_locations_dataframe
                        specific_dil_structure["Biopsy optimization: Optimal biopsy location (all tested lattice points) dataframe"] = potential_optimal_locations_dataframe
                        specific_dil_structure["Biopsy optimization: Optimal biopsy location (zero lattice) dataframe"] = zero_locations_dataframe
                        specific_dil_structure["Biopsy optimization: cubic lattice of optimization points only in dil"] = centered_cubic_lattice_points_contained_only_in_sp_dil_arr

                        structures_progress.update(processing_structures_task, advance=1)
                    structures_progress.update(processing_structures_task, visible = False)
                    
                    ###
                    indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Tying results in a bow", total = None)
                    ###
                    
                    """
                    centered_cubic_lattice_points_NOT_contained_in_dils_arr = np.delete(all_geometries_centered_cubic_lattice_arr,dil_contained_points_df["index"].to_numpy(),axis = 0)
                    #centered_cubic_lattice_points_NOT_contained_in_dils_arr = all_geometries_centered_cubic_lattice_arr[dil_contained_points_df["index"].to_numpy()]
                    del dil_contained_points_df


                    num_lattice_points_not_in_dils = centered_cubic_lattice_points_NOT_contained_in_dils_arr.shape[0]

                    # Calculate test lattice in prostate coordinates
                    prostate_centroid_to_test_location_arr = centered_cubic_lattice_points_NOT_contained_in_dils_arr - prostate_centroid
                    distance_to_prostate_centroid_arr = np.linalg.norm(prostate_centroid_to_test_location_arr, axis=1) 

                    centered_cubic_lattice_points_NOT_contained_in_dils_dict_for_dataframe = {"Patient ID": [patientUID]*num_lattice_points_not_in_dils,
                                                        'Test location vector': list(centered_cubic_lattice_points_NOT_contained_in_dils_arr),
                                                        'Test location (X)': centered_cubic_lattice_points_NOT_contained_in_dils_arr[:,0],
                                                        'Test location (Y)': centered_cubic_lattice_points_NOT_contained_in_dils_arr[:,1],
                                                        'Test location (Z)': centered_cubic_lattice_points_NOT_contained_in_dils_arr[:,2],
                                                        'Selected prostate ROI': [selected_prostate_info["Structure ID"]]*num_lattice_points_not_in_dils,
                                                        'Selected prostate type': [selected_prostate_info["Struct ref type"]]*num_lattice_points_not_in_dils,
                                                        'Selected prostate ref num': [selected_prostate_info["Dicom ref num"]]*num_lattice_points_not_in_dils,
                                                        'Selected prostate index': [selected_prostate_info["Index number"]]*num_lattice_points_not_in_dils,
                                                        'Test location vector (Prostate centroid origin)': list(prostate_centroid_to_test_location_arr),
                                                        'Test location (Prostate centroid origin) (X)': prostate_centroid_to_test_location_arr[:,0],
                                                        'Test location (Prostate centroid origin) (Y)': prostate_centroid_to_test_location_arr[:,1],
                                                        'Test location (Prostate centroid origin) (Z)': prostate_centroid_to_test_location_arr[:,2],
                                                        'Dist to Prostate centroid': distance_to_prostate_centroid_arr,
                                                        'Number of normal dist points contained': [0]*num_lattice_points_not_in_dils,
                                                        'Number of normal dist points tested': [num_normal_dist_points_for_biopsy_optimizer]*num_lattice_points_not_in_dils,
                                                        'Proportion of normal dist points contained': [0]*num_lattice_points_not_in_dils
                                                        }
                    # all points not in the dil, these values were set to zero, this is for the contour plot production!
                    centered_cubic_lattice_non_dil_locations_dataframe = pandas.DataFrame(centered_cubic_lattice_points_NOT_contained_in_dils_dict_for_dataframe)
                    """

                    #live_display.stop()
                    # extract the results of the optimization lattice testing for each dil
                    #all_dils_optimization_lattices_result_dataframe_list = []
                    for specific_dil_structure_index, specific_dil_structure in enumerate(pydicom_item[dil_ref]):
                        structureID_dil = specific_dil_structure["ROI"]
                        potential_optimal_locations_dataframe = specific_dil_structure["Biopsy optimization: Optimal biopsy location (all tested lattice points) dataframe"]
                        zero_locations_dataframe = specific_dil_structure["Biopsy optimization: Optimal biopsy location (zero lattice) dataframe"]

                        # Drop the centroid from the lattice! It messes up the contour plot! The centroid was inserted into the first position!
                        potential_optimal_locations_dataframe_centroid_dropped = potential_optimal_locations_dataframe.drop([0])

                        # Extract tested optimization points only in the DIL
                        #centered_cubic_lattice_points_contained_only_in_sp_dil_arr = specific_dil_structure["Biopsy optimization: cubic lattice of optimization points only in dil"]

                        # Calculate the optimized planes dataframe
                        sp_dil_optimal_locations_dataframe = specific_dil_structure["Biopsy optimization: Optimal biopsy location dataframe"]
                        guidance_map_max_planes_dataframe = biopsy_optimizer.guidance_map_max_planes_dataframe(potential_optimal_locations_dataframe_centroid_dropped,
                                                                            sp_dil_optimal_locations_dataframe,
                                                                            voxel_size_for_dil_optimizer_grid,
                                                                            zero_locations_dataframe,
                                                                            structureID_dil,
                                                                            patientUID,
                                                                            important_info,
                                                                            live_display)

                        guidance_map_max_planes_dataframe = dataframe_builders.convert_columns_to_categorical_and_downcast(guidance_map_max_planes_dataframe, threshold=0.25, ignore_types=(np.floating,))
                        specific_dil_structure["Biopsy optimization: guidance map max-planes dataframe"] = guidance_map_max_planes_dataframe
                        

                        #all_dils_optimization_lattices_result_dataframe_list.append(potential_optimal_locations_dataframe_centroid_dropped)

                    #all_dils_optimization_lattices_result_dataframe = pandas.concat(all_dils_optimization_lattices_result_dataframe_list)
                    #del all_dils_optimization_lattices_result_dataframe_list

                    # IMPORTANT: Need to investigate this dataframe... The columns should be the same between these two???
                    #all_dils_and_non_dil_optimization_lattices_result_dataframe = pandas.concat([all_dils_optimization_lattices_result_dataframe,centered_cubic_lattice_non_dil_locations_dataframe])
                    
                    #misc_tools.point_remover_from_numpy_arr_v2(all_points_2d_arr, points_to_remove_2d_arr)
                    all_zero_locations_dataframe_list = []
                    all_potential_optimal_locations_dataframe_centroid_dropped_list = [] # these are the points actually tested (ie the points wihtin the dil!)
                    for specific_dil_structure_index, specific_dil_structure in enumerate(pydicom_item[dil_ref]):
                        zero_locations_dataframe = specific_dil_structure["Biopsy optimization: Optimal biopsy location (zero lattice) dataframe"]
                        all_zero_locations_dataframe_list.append(zero_locations_dataframe)
                            
                        potential_optimal_locations_dataframe = specific_dil_structure["Biopsy optimization: Optimal biopsy location (all tested lattice points) dataframe"]
                        potential_optimal_locations_dataframe_centroid_dropped = potential_optimal_locations_dataframe.drop([0])
                        all_potential_optimal_locations_dataframe_centroid_dropped_list.append(potential_optimal_locations_dataframe_centroid_dropped)
                    
                    all_potential_optimal_locations_dataframe_centroid_dropped_all_dils_dataframe = pandas.concat(all_potential_optimal_locations_dataframe_centroid_dropped_list, ignore_index = True)
                    
                    all_zero_locations_dataframe = misc_tools.intersect_dataframes(all_zero_locations_dataframe_list)
                    
                    entire_overlapped_lattice_dataframe = pandas.concat([all_potential_optimal_locations_dataframe_centroid_dropped_all_dils_dataframe, all_zero_locations_dataframe], ignore_index = True)
                    
                    #entire_overlapped_lattice_dataframe = biopsy_optimizer.specific_dil_to_all_dils_optimization_lattice_dataframe_combiner(pydicom_item,
                                                                     #dil_ref)

                    # Calculate the cumulative_projection dataframe
                    cumulative_projection_optimization_scores_dataframe = biopsy_optimizer.guidance_map_cumulative_projection_dataframe_creator(entire_overlapped_lattice_dataframe)

                    # save the selected prostate to all DILs (they will all contain the same value but its the best way to save this)!
                    #for specific_dil_structure_index, specific_dil_structure in enumerate(pydicom_item[dil_ref]):
                    #    specific_dil_structure["Biopsy optimization: selected relative prostate dict"] = {"Info": selected_prostate_info, "Centroid vector array": prostate_centroid}
                    # changed my mind!


                    # save the full intersection zero points
                    all_zero_locations_dataframe = dataframe_builders.convert_columns_to_categorical_and_downcast(all_zero_locations_dataframe, threshold=0.25, ignore_types=(np.floating,))
                    pydicom_item[all_ref_key]["Multi-structure information dict (not for csv output)"]["Biopsy optimization: All points outside of DILs (zero points) dataframe"] = all_zero_locations_dataframe
                    del all_zero_locations_dataframe

                    # save the full intersection tested points (ie the points that were actually in the dils)
                    all_potential_optimal_locations_dataframe_centroid_dropped_all_dils_dataframe = dataframe_builders.convert_columns_to_categorical_and_downcast(all_potential_optimal_locations_dataframe_centroid_dropped_all_dils_dataframe, threshold=0.25, ignore_types=(np.floating,))
                    pydicom_item[all_ref_key]["Multi-structure information dict (not for csv output)"]["Biopsy optimization: All points within DILs (tested points) dataframe"] = all_potential_optimal_locations_dataframe_centroid_dropped_all_dils_dataframe
                    del all_potential_optimal_locations_dataframe_centroid_dropped_all_dils_dataframe

                    # save the full lattice, this will only be useful (i think) for creating the contour plots at the end, ie. doesnt need to be CSVd!!!
                    entire_overlapped_lattice_dataframe = dataframe_builders.convert_columns_to_categorical_and_downcast(entire_overlapped_lattice_dataframe, threshold=0.25, ignore_types=(np.floating,))
                    pydicom_item[all_ref_key]["Multi-structure information dict (not for csv output)"]["Biopsy optimization: Optimal biopsy location (entire cubic lattice) dataframe"] = entire_overlapped_lattice_dataframe
                    del entire_overlapped_lattice_dataframe

                    # save the cumulative_projection
                    cumulative_projection_optimization_scores_dataframe = dataframe_builders.convert_columns_to_categorical_and_downcast(cumulative_projection_optimization_scores_dataframe, threshold=0.25, ignore_types=(np.floating,))
                    pydicom_item[all_ref_key]["Multi-structure pre-processing output dataframes dict"]["Biopsy optimization - Cumulative projection (all points within prostate) dataframe"] = cumulative_projection_optimization_scores_dataframe
                    del cumulative_projection_optimization_scores_dataframe

                    

                    ###
                    indeterminate_progress_sub.update(indeterminate_task, visible = False)
                    ###

                    #live_display.stop()

                    
                    patients_progress.update(processing_patients_task, advance=1)
                    completed_progress.update(processing_patients_task_completed, advance=1)
                patients_progress.update(processing_patients_task, visible=False)
                completed_progress.update(processing_patients_task_completed,  visible=True)



                #live_display.start()

                if display_optimization_contour_plots_bool == True:
                    for patientUID,pydicom_item in master_structure_reference_dict.items():

                        
                        sp_patient_selected_structure_info_dataframe = pydicom_item[all_ref_key]["Multi-structure pre-processing output dataframes dict"]["Selected structures"]                 

                        specific_prostate_info_df = sp_patient_selected_structure_info_dataframe[sp_patient_selected_structure_info_dataframe["Struct ref type"] == oar_ref]
                        selected_prostate_info = specific_prostate_info_df.to_dict('records')[0]

                        prostate_ID = selected_prostate_info["Structure ID"]
                        prostate_ref_type = selected_prostate_info["Struct ref type"]
                        prostate_ref_num = selected_prostate_info["Dicom ref num"]
                        prostate_structure_index = selected_prostate_info["Index number"]
                        prostate_found_bool = selected_prostate_info["Struct found bool"]

                        """
                        if prostate_found_bool == True:
                            selected_prostate_centroid = pydicom_item[prostate_ref_type][prostate_structure_index]["Structure global centroid"].reshape(3)
                        else: 
                            important_info.add_text_line('Prostate not found! Defaulting prostate centroid to Zero-vector')
                            selected_prostate_centroid = np.array([0,0,0])
                        """
                        #selected_prostate_centroid = pydicom_item["Biopsy optimization: selected relative prostate dict"]["Centroid vector array"]

                        optimal_locations_dataframe_list = []
                        #dil_centroids_list = []
                        dil_centroids_optimization_locations_list = []
                        for specific_dil_structure_index, specific_dil_structure in enumerate(pydicom_item[dil_ref]):
                            dil_centroids_optimization_locations_dataframe = specific_dil_structure["Biopsy optimization: DIL centroid optimal biopsy location dataframe"] 
                            optimal_locations_dataframe = specific_dil_structure["Biopsy optimization: Optimal biopsy location dataframe"]
                            dil_global_centroid = specific_dil_structure["Structure global centroid"]
                            
                            dil_centroids_optimization_locations_list.append(dil_centroids_optimization_locations_dataframe)
                            optimal_locations_dataframe_list.append(optimal_locations_dataframe)
                            #dil_centroids_list.append(dil_global_centroid)

                        sp_patient_centroid_optimal_dataframe = pandas.concat(dil_centroids_optimization_locations_list)
                        sp_patient_optimal_dataframe = pandas.concat(optimal_locations_dataframe_list)

                        num_dil_structs_patient_specific = master_structure_info_dict["By patient"][patientUID][dil_ref]["Num structs"]
                        
                        # changed this in favor of selecting unique structures at the beginning of pipeline and saving to dataframe! See above in the patient loop!
                        # can pick a random one, since each dil saved the same information of which prostate was selected for the biopsy optimization
                        #random_dil_structure = pydicom_item[dil_ref][random.randint(0,num_dil_structs_patient_specific-1)]
                        #selected_prostate_info = random_dil_structure["Biopsy optimization: selected relative prostate dict"]["Info"]
                        #selected_prostate_centroid = random_dil_structure["Biopsy optimization: selected relative prostate dict"]["Centroid vector array"]

                        entire_overlapped_lattice_dataframe = pydicom_item[all_ref_key]["Multi-structure information dict (not for csv output)"]["Biopsy optimization: Optimal biopsy location (entire cubic lattice) dataframe"]

                        df_simple = entire_overlapped_lattice_dataframe[['Test location (Prostate centroid origin) (X)','Test location (Prostate centroid origin) (Y)','Test location (Prostate centroid origin) (Z)','Proportion of normal dist points contained']]

                        plane_combinations = [(0,1),(0,2),(2,1)] # This defines Transverse (X,Y), Coronal (X,Z) and Saggital (Z,Y)
                        for combination in plane_combinations:
                            index_to_column_dict = {0: 'Test location (Prostate centroid origin) (X)', 1: 'Test location (Prostate centroid origin) (Y)', 2: 'Test location (Prostate centroid origin) (Z)'}
                            dfcumulative = df_simple.groupby([index_to_column_dict[combination[0]],index_to_column_dict[combination[1]]])['Proportion of normal dist points contained'].sum().reset_index()
                            max_val = (dfcumulative['Proportion of normal dist points contained']).max()
                            dfcumulative['Proportion of normal dist points contained'] = dfcumulative['Proportion of normal dist points contained']/max_val
                            
                            fig = go.Figure()

                            

                            for index, row in sp_patient_optimal_dataframe.iterrows():
                                fig.add_scatter(x=[row[dfcumulative.columns[0]]],
                                        y=[row[dfcumulative.columns[1]]],
                                        marker=dict(
                                            color='orange',
                                            size=10,
                                            symbol = 'circle'
                                        ),
                                        text=[row["Relative DIL ID"]],
                                        mode = "markers+text",
                                        name=row["Relative DIL ID"]+' optimal',
                                        textposition="bottom center",
                                        textfont=dict(
                                            family="sans serif",
                                            size=12,
                                            color="white"
                                        )
                                    )
                                
                            for index, row in sp_patient_centroid_optimal_dataframe.iterrows():
                                fig.add_scatter(x=[row[dfcumulative.columns[0]]],
                                        y=[row[dfcumulative.columns[1]]],
                                        marker=dict(
                                            color='yellow',
                                            size=10,
                                            symbol = 'circle'
                                        ),
                                        text=[row["Relative DIL ID"]],
                                        mode = "markers+text",
                                        name=row["Relative DIL ID"]+' centroid',
                                        textposition="bottom center",
                                        textfont=dict(
                                            family="sans serif",
                                            size=12,
                                            color="white"
                                        )
                                    )
                            fig.add_scatter(x=[0],
                                        y=[0],
                                        marker=dict(
                                            color='black',
                                            size=10,
                                            symbol = 'circle'
                                        ),
                                    name='Prostate centroid')  

                            fig.add_trace(
                                go.Contour(
                                    z=dfcumulative['Proportion of normal dist points contained'],
                                    x=dfcumulative.iloc[:,0],
                                    y=dfcumulative.iloc[:,1],
                                    colorscale=[[0, 'rgb(0,0,255)'], [0.9, 'rgb(255,0,0)'],[1, 'rgb(0,255,0)']],
                                    zmax = 1,
                                    zmin = 0,
                                    autocontour = False,
                                    contours = go.contour.Contours(type = 'levels', showlines = True, coloring = 'heatmap', showlabels = True, size = 0.1),
                                    connectgaps = False, 
                                    colorbar = go.contour.ColorBar(len = 0.5)
                                ))

                            x_axis_name = dfcumulative.columns[0][-2]
                            y_axis_name = dfcumulative.columns[1][-2]
                            patient_pos_dict = {'X': ' (L/R)', "Y":' (A/P)', "Z": '(S/I)'}
                            fig['layout']['xaxis'].update(title=x_axis_name+patient_pos_dict[x_axis_name])
                            fig['layout']['yaxis'].update(title=y_axis_name+patient_pos_dict[y_axis_name])
                            
                            patient_plane_dict = {'XY': ' Transverse (XY)', "YZ": ' Sagittal (YZ)', "XZ": ' Coronal (XZ)',
                                                  'YX': ' Transverse (YX)', "ZY": ' Sagittal (ZY)', "ZX": ' Coronal (ZX)'}
                            patient_plane_determiner_str = x_axis_name+y_axis_name
                            
                            fig.add_annotation(text="Cumulative, "+patient_plane_dict[patient_plane_determiner_str]+' plane',
                                xref="paper", yref="paper",
                                x=0.95, y=0.9, showarrow=False,
                                    font=dict(family="Courier New, monospace", size=16, color="#ffffff")
                                )  
                            fig.add_annotation(text="Patient: "+patientUID,
                                                    xref="paper", yref="paper",
                                                    x=0.95, y=0.95, showarrow=False,
                                                        font=dict(family="Courier New, monospace", size=16, color="#ffffff")
                                                    )  
                            

                            fig.show()   





                ### THIS DATAFRAME IS LARGE, TAKES UP TOO MUCH MEMORY, DELETING AFTER THIS POINT!
                #for patientUID,pydicom_item in master_structure_reference_dict.items():
                #    del pydicom_item[all_ref_key]["Multi-structure information dict (not for csv output)"]["Biopsy optimization: Optimal biopsy location (entire cubic lattice) dataframe"]

                #####DONE##### PERFORM BIOPSY DIL OPTIMIZATION


                #live_display.stop()


                ############################ BIOPSY ONLY (non sim)

                patientUID_default = "Initializing"
                processing_patients_task_main_description = "[red]Processing patient non-sim bx data [{}]...".format(patientUID_default)
                processing_patients_task_completed_main_description = "[green]Processing patient non-sim bx data"
                processing_patients_task = patients_progress.add_task(processing_patients_task_main_description, total=master_structure_info_dict["Global"]["Num cases"])
                processing_patients_task_completed = completed_progress.add_task(processing_patients_task_completed_main_description, total=master_structure_info_dict["Global"]["Num cases"], visible = False)

                for patientUID,pydicom_item in master_structure_reference_dict.items():
                    processing_patients_task_main_description = "[red]Processing patient non-sim bx data [{}]...".format(patientUID)
                    patients_progress.update(processing_patients_task, description = processing_patients_task_main_description)
                    
                    structureID_default = "Initializing"
                    num_nonsim_bx_structs_patient_specific = master_structure_info_dict["By patient"][patientUID][bx_ref]["Num real structs"]
                    processing_structures_task_main_description = "[cyan]Processing structures [{},{}]...".format(patientUID,structureID_default)
                    processing_structures_task = structures_progress.add_task(processing_structures_task_main_description, total=num_nonsim_bx_structs_patient_specific)
                    
                    for specific_structure_index, specific_structure in enumerate(pydicom_item[bx_ref]):
                        structureID = specific_structure["ROI"]
                        structure_reference_number = specific_structure["Ref #"]
                        simulated_bool = specific_structure["Simulated bool"]
                        sim_type = specific_structure["Simulated type"]

                        # IF THIS IS A SIMULATED BIOPSY, THEN DO NOTHING!
                        if simulated_bool == True:
                            continue

                        processing_structures_task_main_description = "[cyan]Processing structures [{},{}]...".format(patientUID,structureID)
                        structures_progress.update(processing_structures_task, description = processing_structures_task_main_description)

                        
                        threeDdata_zslice_list = specific_structure["Raw contour pts zslice list"].copy()

                        
                        
                        total_structure_points = sum([np.shape(x)[0] for x in threeDdata_zslice_list])
                        threeDdata_array = np.empty([total_structure_points,3])
                        
                        # build raw threeDdata for biopsies
                        structure_centroids_array = np.empty([len(threeDdata_zslice_list),3])

                        lower_bound_index = 0  
                        for index, threeDdata_zslice in enumerate(threeDdata_zslice_list):
                            current_zslice_num_points = np.size(threeDdata_zslice,0)
                            threeDdata_array[lower_bound_index:lower_bound_index + current_zslice_num_points] = threeDdata_zslice
                            lower_bound_index = lower_bound_index + current_zslice_num_points 
                            
                            # find zslice centroid
                            structure_zslice_centroid = np.mean(threeDdata_zslice,axis=0)
                            structure_centroids_array[index] = structure_zslice_centroid

                        structure_global_centroid = centroid_finder.centeroidfinder_numpy_3D(structure_centroids_array)
                        
                        
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
                        pcd_color = structs_referenced_dict[bx_ref]['PCD color dict'][sim_type]
                        threeDdata_point_cloud = point_containment_tools.create_point_cloud(threeDdata_array, pcd_color)
                        
                        

                        # generate delaunay triangulations 
                        """
                        deulaunay_objs_zslice_wise_list = point_containment_tools.adjacent_slice_delaunay_parallel(parallel_pool, threeDdata_zslice_list)

                        zslice1 = threeDdata_array[0,2]
                        zslice2 = threeDdata_array[-1,2]
                        delaunay_global_convex_structure_obj = point_containment_tools.delaunay_obj(threeDdata_array, threeDdata_pcd_color, zslice1, zslice2)
                        #delaunay_global_convex_structure_obj.generate_lineset()
                        """

                        


                        threeDdata_array_fully_interpolated = interpolation_information.interpolated_pts_np_arr
                        threeDdata_array_fully_interpolated_with_end_caps = interpolation_information.interpolated_pts_with_end_caps_np_arr
                        threeDdata_array_interslice_interpolation = np.vstack(interslice_interpolation_information.interpolated_pts_list)
                        interslice_interp_pcd = point_containment_tools.create_point_cloud(threeDdata_array_interslice_interpolation, pcd_color)
                        inter_and_intra_interp_pcd = point_containment_tools.create_point_cloud(threeDdata_array_fully_interpolated, pcd_color)
                        inter_and_intra_and_end_caps_interp_pcd = point_containment_tools.create_point_cloud(threeDdata_array_fully_interpolated_with_end_caps, pcd_color)
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



                        # calculate mean variation of biopsy centroids with the pca
                        line_start = centroid_line[0,:]
                        line_end = centroid_line[1,:]
                        variation_distance_arr = np.empty(structure_centroids_array.shape[0])
                        closest_point_on_line_arr = np.empty_like(structure_centroids_array)
                        for index,point in enumerate(structure_centroids_array):
                            distance, closest_point_on_line = biopsy_creator.point_to_line_segment_distance(point, line_start, line_end)
                            variation_distance_arr[index] = distance
                            closest_point_on_line_arr[index] = closest_point_on_line
                        mean_variation = np.mean(variation_distance_arr)




                        # calculate the maximum distance between the original biopsy centroids, where all points have been projected onto the plan given by the 
                        # normal vector of the pca line
                        maximum_2d_distance_between_centroids = biopsy_creator.distance_of_most_distant_points_2d_projection(structure_centroids_array, travel_vec)

                        
                        if display_pca_fit_variation_for_biopsies_bool == True:
                            live_display.stop()
                            
                            # Create the point cloud
                            biopsy_centroids_pcd = point_containment_tools.create_point_cloud(structure_centroids_array, np.array([0, 0, 1]))
                            
                            # Create the line set for pca biopsy
                            line_color = np.array([0, 1, 1])
                            centroid_line_set = o3d.geometry.LineSet()
                            centroid_line_set.points = o3d.utility.Vector3dVector(centroid_line)
                            centroid_line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
                            centroid_line_set.paint_uniform_color(line_color)
                            
                            # Create the line set for points between centroids and pca biopsy line 
                            nearest_points_line_color = np.array([1, 0, 1])
                            nearest_points_pca_to_centroids_line_set = o3d.geometry.LineSet()
                            nearest_points_pca_to_centroids_line_set.points = o3d.utility.Vector3dVector(np.vstack((structure_centroids_array,closest_point_on_line_arr)))
                            num_centroids_temp = structure_centroids_array.shape[0]
                            lines_connections = [[i,i+num_centroids_temp] for i in range(0,num_centroids_temp)]
                            nearest_points_pca_to_centroids_line_set.lines = o3d.utility.Vector2iVector(lines_connections)
                            nearest_points_pca_to_centroids_line_set.paint_uniform_color(nearest_points_line_color)
                            
                            # Create the text annotation at the top-right corner of the bounding box
                            print(f"Pt: {patientUID}, Bx: {structureID}, Mean variation: {mean_variation}, Max dinstance between centroids: {maximum_2d_distance_between_centroids}")

                            # Plot the geometries with the text annotation
                            plotting_funcs.plot_geometries(biopsy_centroids_pcd, centroid_line_set, nearest_points_pca_to_centroids_line_set)

                            # Clean up
                            del biopsy_centroids_pcd, centroid_line_set, nearest_points_pca_to_centroids_line_set, nearest_points_line_color, line_color, lines_connections, num_centroids_temp
                            live_display.start()





                        # conduct a nearest neighbour search of biopsy centroids

                        #treescipy = scipy.spatial.KDTree(threeDdata_array)
                        #nn = treescipy.query(centroid_line_sample[0])
                        #nearest_neighbour = treescipy.data[nn[1]]

                        list_travel_vec = np.squeeze(travel_vec).tolist()
                        list_centroid_line_first_point = np.squeeze(centroid_line_sample[0]).tolist()
                        biopsy_reconstructed_cyl_z_length_from_contour_data = centroid_line_length
                        drawn_biopsy_array_transpose = biopsy_creator.biopsy_points_creater_by_transport(list_travel_vec,list_centroid_line_first_point,num_centroid_samples_of_centroid_line,np.linalg.norm(travel_vec), biopsy_radius, False)
                        drawn_biopsy_array = drawn_biopsy_array_transpose.T
                        reconstructed_biopsy_point_cloud = point_containment_tools.create_point_cloud(drawn_biopsy_array, pcd_color)
                        reconstructed_bx_delaunay_global_convex_structure_obj = point_containment_tools.delaunay_obj(drawn_biopsy_array, pcd_color)
                        reconstructed_bx_delaunay_global_convex_structure_obj.generate_lineset()
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



                            


                        ### CALCULATE THE STRUCTURES VOLUME
                        ###
                        indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Calculating structure volume", total = None)
                        ###

                        structure_info = misc_tools.specific_structure_info_dict_creator('given', specific_structure = specific_structure) 
                        
                        
                        z_axis_np_vec = np.array([0,0,1],dtype=float)
                        #apex_to_base_bx_best_fit_vec = specific_structure["Centroid line vec (bx needle base to bx needle tip)"]
                        centroid_line_to_z_axis_rotation_matrix_other = mf.rotation_matrix_from_vectors(apex_to_base_bx_best_fit_vec, z_axis_np_vec)
                        rotated_reconstructed_bx_arr = (centroid_line_to_z_axis_rotation_matrix_other @ drawn_biopsy_array.T).T
                        rotated_reconstructed_bx_arr_rounded = np.copy(rotated_reconstructed_bx_arr)
                        
                        # Using the biopsy creator transport function, the maximum distance between rings is 0.1! So each constant zslice should be at most 
                        # 0.1mm apart, this is important for the rounding below because we are rounding the constant zslices to travel_vec_dist + 1 decimal place
                        distance_between_rings = np.linalg.norm(travel_vec)
                        sci_not_dist_bet_rings = '%e' % distance_between_rings
                        num_zeros_before_first_dig_after_decimal = int(sci_not_dist_bet_rings.partition('-')[2]) - 1
                        num_decimals_for_rounding = num_zeros_before_first_dig_after_decimal + 2
                        rotated_reconstructed_bx_arr_rounded[:,2] = np.round(rotated_reconstructed_bx_arr[:,2], decimals = num_decimals_for_rounding)
                        
                        zvals_list = np.unique(rotated_reconstructed_bx_arr_rounded[:,2]).tolist()
                        zslices_list = [rotated_reconstructed_bx_arr_rounded[rotated_reconstructed_bx_arr_rounded[:,2] == z_val] for z_val in zvals_list]

                        structure_volume, maximum_distance, voxel_size_for_structure_volume_calc, binary_mask_arr, live_display = misc_tools.structure_volume_calculator(rotated_reconstructed_bx_arr_rounded,
                            zvals_list,
                            zslices_list,
                            structure_info,
                            plot_volume_calculation_containment_result_bool,
                            plot_binary_mask_bool,
                            voxel_size_for_structure_volume_calc_bx,
                            factor_for_voxel_size,
                            cupy_array_upper_limit_NxN_size_input,
                            layout_groups,
                            nearest_zslice_vals_and_indices_cupy_generic_max_size,
                            structures_progress,
                            live_display
                            )
                        

                        ###
                        indeterminate_progress_sub.update(indeterminate_task, visible = False)
                        ###
                        ###### END STRUCTURE VOLUME CALCULATION



                        # store all calculated quantities
                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Raw contour pts"] = threeDdata_array
                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Equal num zslice contour pts"] = threeDdata_equal_pt_zslice_list
                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Inter-slice interpolation information"] = interslice_interpolation_information                        
                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Intra-slice interpolation information"] = interpolation_information
                        #master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Delaunay triangulation zslice-wise list"] = deulaunay_objs_zslice_wise_list # DEPRECATED
                        #master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Delaunay triangulation global structure"] = delaunay_global_convex_structure_obj # DEPRECATED
                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Maximum pairwise distance"] = maximum_distance
                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Structure volume"] = structure_volume
                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Voxel size for structure volume calc"] = voxel_size_for_structure_volume_calc
                        
                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Point cloud raw"] = threeDdata_point_cloud
                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Interpolated structure point cloud dict"] = interpolated_pcd_dict

                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Centroid variation arr"] = variation_distance_arr
                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Mean centroid variation"] = mean_variation
                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Maximum projected distance between original centroids"] = maximum_2d_distance_between_centroids
                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Structure centroid pts"] = structure_centroids_array
                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Reconstructed biopsy cylinder length (from contour data)"] = biopsy_reconstructed_cyl_z_length_from_contour_data
                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Best fit line of centroid pts"] = centroid_line
                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Centroid line unit vec (bx needle base to bx needle tip)"] = apex_to_base_bx_best_fit_unit_vec
                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Centroid line vec (bx needle base to bx needle tip)"] = apex_to_base_bx_best_fit_vec
                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Centroid line vec length (bx needle base to bx needle tip)"] = apex_to_base_bx_best_fit_vec_length
                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Centroid line sample pts"] = centroid_line_sample
                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Reconstructed structure pts arr"] = drawn_biopsy_array
                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Reconstructed structure point cloud"] = reconstructed_biopsy_point_cloud
                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Reconstructed structure delaunay global"] = reconstructed_bx_delaunay_global_convex_structure_obj
                        #master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Structure centroid pts"] = structure_centroids_array
                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Structure global centroid"] = structure_global_centroid

                        structures_progress.update(processing_structures_task, advance=1)

           
                    structures_progress.remove_task(processing_structures_task)
                    patients_progress.update(processing_patients_task, advance=1)
                    completed_progress.update(processing_patients_task_completed, advance=1)
                patients_progress.update(processing_patients_task, visible=False)
                completed_progress.update(processing_patients_task_completed,  visible=True)  

                ############################ BIOPSY ONLY
                
                


                ###  DETERMINE LENGTH FOR SIMULATED CORES

                patientUID_default = "Initializing"
                processing_patients_task_main_description = "[red]Determining simulated biopsy lengths [{}]...".format(patientUID_default)
                processing_patients_task_completed_main_description = "[green]Determining simulated biopsy lengths"
                processing_patients_task = patients_progress.add_task(processing_patients_task_main_description, total=master_structure_info_dict["Global"]["Num cases"])
                processing_patients_task_completed = completed_progress.add_task(processing_patients_task_completed_main_description, total=master_structure_info_dict["Global"]["Num cases"], visible = False)

                real_biopsy_lengths_list = []
                for patientUID,pydicom_item in master_structure_reference_dict.items():
                    processing_patients_task_main_description = "[red]Determining simulated biopsy lengths [{}]...".format(patientUID)
                    patients_progress.update(processing_patients_task, description = processing_patients_task_main_description)
                    
                    structureID_default = "Initializing"
                    num_bx_structs_patient_specific = master_structure_info_dict["By patient"][patientUID][bx_ref]["Num structs"]
                    processing_structures_task_main_description = "[cyan]Processing structures [{},{}]...".format(patientUID,structureID_default)
                    processing_structures_task = structures_progress.add_task(processing_structures_task_main_description, total=num_bx_structs_patient_specific)
                    
                    for specific_structure_index, specific_structure in enumerate(pydicom_item[bx_ref]):
                        simulated_bool = specific_structure["Simulated bool"]
                        if simulated_bool == True:
                            pass
                        else:
                            real_biopsy_lengths_list.append(specific_structure["Reconstructed biopsy cylinder length (from contour data)"])
                        
                        structures_progress.update(processing_structures_task, advance=1)

       
                    structures_progress.remove_task(processing_structures_task)
                    patients_progress.update(processing_patients_task, advance=1)
                    completed_progress.update(processing_patients_task_completed, advance=1)
                patients_progress.update(processing_patients_task, visible=False)
                completed_progress.update(processing_patients_task_completed,  visible=True)

               
                ###################    SET SOME PRELIMS FOR THE SIMULATED BIOPSIES  
               
               
                # create info for simulated biopsies
                real_biopsy_lengths_arr = np.array(real_biopsy_lengths_list)
                mean_of_real_biopsy_lengths = np.mean(real_biopsy_lengths_arr)
                std_of_real_biopsy_lengths = np.std(real_biopsy_lengths_arr)


                # initialize the basics for drawing the simulated biopsies
                centroid_line_vec_sim_list = [0,0,1]
                centroid_first_pos_sim_list = [0,0,0]
                num_centroids_for_sim_bxs = 10
                simulated_bx_rad = 2
                plot_simulated_cores_immediately = False
                # note that the length of the simulated biopsy is determined on a per biopsy basis in the below code!
                    

                ############################ SIMULATED BIOPSY ONLY

                patientUID_default = "Initializing"
                processing_patients_task_main_description = "[red]Processing patient sim-bx data [{}]...".format(patientUID_default)
                processing_patients_task_completed_main_description = "[green]Processing patient sim-bx data"
                processing_patients_task = patients_progress.add_task(processing_patients_task_main_description, total=master_structure_info_dict["Global"]["Num cases"])
                processing_patients_task_completed = completed_progress.add_task(processing_patients_task_completed_main_description, total=master_structure_info_dict["Global"]["Num cases"], visible = False)

                for patientUID,pydicom_item in master_structure_reference_dict.items():
                    processing_patients_task_main_description = "[red]Processing patient sim-bx data [{}]...".format(patientUID)
                    patients_progress.update(processing_patients_task, description = processing_patients_task_main_description)
                    
                    structureID_default = "Initializing"
                    num_sim_bx_structs_patient_specific = master_structure_info_dict["By patient"][patientUID][bx_ref]["Num sim structs"]
                    processing_structures_task_main_description = "[cyan]Processing structures [{},{}]...".format(patientUID,structureID_default)
                    processing_structures_task = structures_progress.add_task(processing_structures_task_main_description, total=num_sim_bx_structs_patient_specific)

                    for specific_structure_index, specific_structure in enumerate(pydicom_item[bx_ref]):
                        structureID = specific_structure["ROI"]
                        structure_reference_number = specific_structure["Ref #"]
                        simulated_bool = specific_structure["Simulated bool"]
                        sim_type = specific_structure["Simulated type"]
                        # IF THIS IS NOT A SIMULATED BIOPSY, THEN DO NOTHING!
                        if simulated_bool == False:
                            continue
                        simulated_type = specific_structure["Simulated type"]
                        processing_structures_task_main_description = "[cyan]Processing structures [{},{}]...".format(patientUID,structureID)
                        structures_progress.update(processing_structures_task, description = processing_structures_task_main_description)

                        # determine the length of this particular core
                        if simulated_biopsy_length_method == 'full':
                            centroid_sep_sim_dist = biopsy_needle_compartment_length/(num_centroids_for_sim_bxs-1) # the minus 1 ensures that the legnth of the biopsy is actually correct!
                        elif simulated_biopsy_length_method == 'real normal':
                            within_bounds = False
                            while within_bounds == False:
                                biopsy_needle_sim_sampled_length = np.random.normal(loc = mean_of_real_biopsy_lengths, scale = std_of_real_biopsy_lengths)
                                if (biopsy_needle_sim_sampled_length >= mean_of_real_biopsy_lengths - 2*std_of_real_biopsy_lengths) and (biopsy_needle_sim_sampled_length <= mean_of_real_biopsy_lengths + 2*std_of_real_biopsy_lengths):
                                    within_bounds = True
                                else:
                                    pass
                            centroid_sep_sim_dist = biopsy_needle_sim_sampled_length/(num_centroids_for_sim_bxs-1) # the minus 1 ensures that the legnth of the biopsy is actually correct!
                        elif simulated_biopsy_length_method == 'real mean':
                            centroid_sep_sim_dist = mean_of_real_biopsy_lengths/(num_centroids_for_sim_bxs-1)
                        
                        threeDdata_zslice_list = biopsy_creator.biopsy_points_creater_by_transport_for_sim_bxs(centroid_line_vec_sim_list,
                                                                                                                centroid_first_pos_sim_list,
                                                                                                                num_centroids_for_sim_bxs,
                                                                                                                centroid_sep_sim_dist,
                                                                                                                simulated_bx_rad,
                                                                                                                plot_simulated_cores_immediately)
                        
                        
                        ### Transform simulated biopsies to location 
                        
                        if simulated_type == centroid_dil_sim_key:
                            threeDdata_zslice_list = biopsy_transporter.biopsy_transporter_centroid(pydicom_item,
                                                                                specific_structure,
                                                                                threeDdata_zslice_list
                                                                                )
                            
                        elif simulated_type == optimal_dil_sim_key:
                            threeDdata_zslice_list = biopsy_transporter.biopsy_transporter_optimal(pydicom_item,
                                                                                specific_structure,
                                                                                threeDdata_zslice_list
                                                                                )
                        
                    
                        
                        total_structure_points = sum([np.shape(x)[0] for x in threeDdata_zslice_list])
                        threeDdata_array = np.empty([total_structure_points,3])
                        
                        # build raw threeDdata for biopsies
                        structure_centroids_array = np.empty([len(threeDdata_zslice_list),3])

                        lower_bound_index = 0  
                        for index, threeDdata_zslice in enumerate(threeDdata_zslice_list):
                            current_zslice_num_points = np.size(threeDdata_zslice,0)
                            threeDdata_array[lower_bound_index:lower_bound_index + current_zslice_num_points] = threeDdata_zslice
                            lower_bound_index = lower_bound_index + current_zslice_num_points 
                            
                            # find zslice centroid
                            structure_zslice_centroid = np.mean(threeDdata_zslice,axis=0)
                            structure_centroids_array[index] = structure_zslice_centroid

                        structure_global_centroid = centroid_finder.centeroidfinder_numpy_3D(structure_centroids_array)
                        
                        
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
                        pcd_color = structs_referenced_dict[bx_ref]['PCD color dict'][sim_type]
                        threeDdata_point_cloud = point_containment_tools.create_point_cloud(threeDdata_array, pcd_color)
                        
                        

                        # generate delaunay triangulations (DEPRECATED)
                        """
                        deulaunay_objs_zslice_wise_list = point_containment_tools.adjacent_slice_delaunay_parallel(parallel_pool, threeDdata_zslice_list)

                        zslice1 = threeDdata_array[0,2]
                        zslice2 = threeDdata_array[-1,2]
                        delaunay_global_convex_structure_obj = point_containment_tools.delaunay_obj(threeDdata_array, threeDdata_pcd_color, zslice1, zslice2)
                        #delaunay_global_convex_structure_obj.generate_lineset()
                        """
                        
                        

                        threeDdata_array_fully_interpolated = interpolation_information.interpolated_pts_np_arr
                        threeDdata_array_fully_interpolated_with_end_caps = interpolation_information.interpolated_pts_with_end_caps_np_arr
                        threeDdata_array_interslice_interpolation = np.vstack(interslice_interpolation_information.interpolated_pts_list)
                        interslice_interp_pcd = point_containment_tools.create_point_cloud(threeDdata_array_interslice_interpolation, pcd_color)
                        inter_and_intra_interp_pcd = point_containment_tools.create_point_cloud(threeDdata_array_fully_interpolated, pcd_color)
                        inter_and_intra_and_end_caps_interp_pcd = point_containment_tools.create_point_cloud(threeDdata_array_fully_interpolated_with_end_caps, pcd_color)
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



                        # calculate mean variation of biopsy centroids with the pca
                        line_start = centroid_line[0,:]
                        line_end = centroid_line[1,:]
                        variation_distance_arr = np.empty(structure_centroids_array.shape[0])
                        for index,point in enumerate(structure_centroids_array):
                            distance, _ = biopsy_creator.point_to_line_segment_distance(point, line_start, line_end)
                            variation_distance_arr[index] = distance
                        mean_variation = np.mean(variation_distance_arr)


                        # calculate the maximum distance between the original biopsy centroids, where all points have been projected onto the plan given by the 
                        # normal vector of the pca line
                        maximum_2d_distance_between_centroids = biopsy_creator.distance_of_most_distant_points_2d_projection(structure_centroids_array, travel_vec)




                        # conduct a nearest neighbour search of biopsy centroids

                        #treescipy = scipy.spatial.KDTree(threeDdata_array)
                        #nn = treescipy.query(centroid_line_sample[0])
                        #nearest_neighbour = treescipy.data[nn[1]]

                        list_travel_vec = np.squeeze(travel_vec).tolist()
                        list_centroid_line_first_point = np.squeeze(centroid_line_sample[0]).tolist()
                        biopsy_reconstructed_cyl_z_length_from_contour_data = centroid_line_length
                        drawn_biopsy_array_transpose = biopsy_creator.biopsy_points_creater_by_transport(list_travel_vec,list_centroid_line_first_point,num_centroid_samples_of_centroid_line,np.linalg.norm(travel_vec), biopsy_radius, False)
                        drawn_biopsy_array = drawn_biopsy_array_transpose.T
                        reconstructed_biopsy_point_cloud = point_containment_tools.create_point_cloud(drawn_biopsy_array, pcd_color)
                        reconstructed_bx_delaunay_global_convex_structure_obj = point_containment_tools.delaunay_obj(drawn_biopsy_array, pcd_color)
                        reconstructed_bx_delaunay_global_convex_structure_obj.generate_lineset()
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


                            


                        
                        structure_info = misc_tools.specific_structure_info_dict_creator('given', specific_structure = specific_structure) 
                        ### CALCULATE THE STRUCTURES VOLUME
                        ###
                        indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Calculating structure volume", total = None)
                        ###
                        
                        z_axis_np_vec = np.array([0,0,1],dtype=float)
                        #apex_to_base_bx_best_fit_vec = specific_structure["Centroid line vec (bx needle base to bx needle tip)"]
                        centroid_line_to_z_axis_rotation_matrix_other = mf.rotation_matrix_from_vectors(apex_to_base_bx_best_fit_vec, z_axis_np_vec)
                        rotated_reconstructed_bx_arr = (centroid_line_to_z_axis_rotation_matrix_other @ drawn_biopsy_array.T).T
                        rotated_reconstructed_bx_arr_rounded = np.copy(rotated_reconstructed_bx_arr)
                        
                        # Using the biopsy creator transport function, the maximum distance between rings is 0.1! So each constant zslice should be at most 
                        # 0.1mm apart, this is important for the rounding below because we are rounding the constant zslices to travel_vec_dist + 1 decimal place
                        distance_between_rings = np.linalg.norm(travel_vec)
                        sci_not_dist_bet_rings = '%e' % distance_between_rings
                        num_zeros_before_first_dig_after_decimal = int(sci_not_dist_bet_rings.partition('-')[2]) - 1
                        num_decimals_for_rounding = num_zeros_before_first_dig_after_decimal + 2
                        rotated_reconstructed_bx_arr_rounded[:,2] = np.round(rotated_reconstructed_bx_arr[:,2], decimals = num_decimals_for_rounding)
                        
                        zvals_list = np.unique(rotated_reconstructed_bx_arr_rounded[:,2]).tolist()
                        zslices_list = [rotated_reconstructed_bx_arr_rounded[rotated_reconstructed_bx_arr_rounded[:,2] == z_val] for z_val in zvals_list]

                        structure_volume, maximum_distance, voxel_size_for_structure_volume_calc, binary_mask_arr, live_display = misc_tools.structure_volume_calculator(rotated_reconstructed_bx_arr_rounded,
                            zvals_list,
                            zslices_list,
                            structure_info,
                            plot_volume_calculation_containment_result_bool,
                            plot_binary_mask_bool,
                            voxel_size_for_structure_volume_calc_bx,
                            factor_for_voxel_size,
                            cupy_array_upper_limit_NxN_size_input,
                            layout_groups,
                            nearest_zslice_vals_and_indices_cupy_generic_max_size,
                            structures_progress,
                            live_display
                            )
                        
                        ###
                        indeterminate_progress_sub.update(indeterminate_task, visible = False)
                        ###
                        ###### END STRUCTURE VOLUME CALCULATION



                        # store all calculated quantities
                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Raw contour pts zslice list"] = threeDdata_zslice_list
                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Raw contour pts"] = threeDdata_array
                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Equal num zslice contour pts"] = threeDdata_equal_pt_zslice_list
                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Inter-slice interpolation information"] = interslice_interpolation_information                        
                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Intra-slice interpolation information"] = interpolation_information
                        #master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Delaunay triangulation zslice-wise list"] = deulaunay_objs_zslice_wise_list # DEPRECATED
                        #master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Delaunay triangulation global structure"] = delaunay_global_convex_structure_obj # DEPRECATED
                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Maximum pairwise distance"] = maximum_distance
                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Structure volume"] = structure_volume
                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Voxel size for structure volume calc"] = voxel_size_for_structure_volume_calc
                        
                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Point cloud raw"] = threeDdata_point_cloud
                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Interpolated structure point cloud dict"] = interpolated_pcd_dict

                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Centroid variation arr"] = variation_distance_arr
                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Mean centroid variation"] = mean_variation
                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Maximum projected distance between original centroids"] = maximum_2d_distance_between_centroids
                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Structure centroid pts"] = structure_centroids_array
                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Reconstructed biopsy cylinder length (from contour data)"] = biopsy_reconstructed_cyl_z_length_from_contour_data
                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Best fit line of centroid pts"] = centroid_line
                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Centroid line unit vec (bx needle base to bx needle tip)"] = apex_to_base_bx_best_fit_unit_vec
                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Centroid line vec (bx needle base to bx needle tip)"] = apex_to_base_bx_best_fit_vec
                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Centroid line vec length (bx needle base to bx needle tip)"] = apex_to_base_bx_best_fit_vec_length
                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Centroid line sample pts"] = centroid_line_sample
                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Reconstructed structure pts arr"] = drawn_biopsy_array
                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Reconstructed structure point cloud"] = reconstructed_biopsy_point_cloud
                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Reconstructed structure delaunay global"] = reconstructed_bx_delaunay_global_convex_structure_obj
                        #master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Structure centroid pts"] = structure_centroids_array
                        master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Structure global centroid"] = structure_global_centroid

                        structures_progress.update(processing_structures_task, advance=1)

           
                    structures_progress.remove_task(processing_structures_task)
                    patients_progress.update(processing_patients_task, advance=1)
                    completed_progress.update(processing_patients_task_completed, advance=1)
                patients_progress.update(processing_patients_task, visible=False)
                completed_progress.update(processing_patients_task_completed,  visible=True)  

                ############################ SIMULATED BIOPSY ONLY



                ################## ALL BIOPSIES


                #live_display.stop()
                #print('test')

                patientUID_default = "Initializing"
                processing_patients_task_main_description = "[red]Performing other calculations on patient structure data [{}]...".format(patientUID_default)
                processing_patients_task_completed_main_description = "[green]Performing other calculations on patient structure data"
                processing_patients_task = patients_progress.add_task(processing_patients_task_main_description, total=master_structure_info_dict["Global"]["Num cases"])
                processing_patients_task_completed = completed_progress.add_task(processing_patients_task_completed_main_description, total=master_structure_info_dict["Global"]["Num cases"], visible = False)

                for patientUID,pydicom_item in master_structure_reference_dict.items():
                    processing_patients_task_main_description = "[red]Performing other calculations on patient structure data [{}]...".format(patientUID)
                    patients_progress.update(processing_patients_task, description = processing_patients_task_main_description)
                    


                    sp_patient_selected_structure_info_dataframe = pydicom_item[all_ref_key]["Multi-structure pre-processing output dataframes dict"]["Selected structures"]                 

                    specific_prostate_info_df = sp_patient_selected_structure_info_dataframe[sp_patient_selected_structure_info_dataframe["Struct ref type"] == oar_ref]
                    selected_prostate_info = specific_prostate_info_df.to_dict('records')[0]

                    prostate_found_bool = selected_prostate_info["Struct found bool"]




                    structureID_default = "Initializing"
                    num_general_structs_patient_specific = master_structure_info_dict["By patient"][patientUID][all_ref_key]["Total num structs"]
                    processing_structures_task_main_description = "[cyan]Processing structures [{},{}]...".format(patientUID,structureID_default)
                    processing_structures_task = structures_progress.add_task(processing_structures_task_main_description, total=num_general_structs_patient_specific)
                    for structs in structs_referenced_list:
                        for specific_structure_index, specific_structure in enumerate(pydicom_item[structs]):
                            structureID = specific_structure["ROI"]
                            structure_reference_number = specific_structure["Ref #"]
                            processing_structures_task_main_description = "[cyan]Processing structures [{},{}]...".format(patientUID,structureID)
                            structures_progress.update(processing_structures_task, description = processing_structures_task_main_description)

                            
                            ### DETERMINE BIOPSY POSITION WITHIN THE PROSTATE
                            if structs == bx_ref:
                                bx_structure_global_centroid = specific_structure["Structure global centroid"].copy().reshape((3))
                                


                                # selected_prostate_info, message_string, prostate_found_bool, num_prostates_found = misc_tools.specific_structure_selector(pydicom_item,
                                #                             oar_ref,
                                #                             prostate_contour_name
                                #                             )

                                # if num_prostates_found > 1:
                                #     important_info.add_text_line(message_string,live_display) 
                                # elif prostate_found_bool == False:
                                #     important_info.add_text_line(message_string,live_display) 

                                live_display.refresh()
                                if prostate_found_bool == True:
                                    prostate_structure_index = selected_prostate_info["Index number"]
                                    prostate_structure = pydicom_item[oar_ref][prostate_structure_index]
                                    prostate_structure_global_centroid = prostate_structure["Structure global centroid"].copy().reshape((3))
                                    prostate_dimension_at_centroid_dict = prostate_structure["Structure dimension at centroid dict"]
                                    prostate_z_dimension_length_at_centroid = prostate_dimension_at_centroid_dict["Z dimension length at centroid"]
                                
                                    # note that distance_to_mid_gland_threshold should be a positive quantity for the position classifier function below!
                                    distance_to_mid_gland_threshold = abs(prostate_z_dimension_length_at_centroid/6) 

                                    # determine biopsy location within prostate 
                                    bx_centroid_vec_rel_to_prostate_centroid = bx_structure_global_centroid - prostate_structure_global_centroid
                                
                                    bx_prostate_position_dict = misc_tools.bx_position_classifier_in_prostate_frame_sextant(bx_centroid_vec_rel_to_prostate_centroid,
                                                distance_to_mid_gland_threshold)
                                else: 
                                    bx_prostate_position_dict = {"LR": None,"AP": None,"SI": None}

                                bx_location_in_prostate_ref_frame_dict = {"Relative prostate info": selected_prostate_info,
                                                                        "Bx position in prostate": bx_prostate_position_dict
                                                                        }


                            ### DETERMINE TARGET DILS OF EACH BIOPSY    
                            if structs == bx_ref:
                                bx_structure_global_centroid = specific_structure["Structure global centroid"].copy().reshape((3))
                                bx_structure_reconstructed_pts = specific_structure["Reconstructed structure pts arr"].copy()

                                dil_distance_dict = {}
                                closest_dil_centroid_info = [None,None]
                                closest_dil_surface_info = [None,None]
                                for specific_dil_structure_index, specific_dil_structure in enumerate(pydicom_item[dil_ref]):

                                    dil_structureID = specific_dil_structure["ROI"]
                                    dil_structure_reference_number = specific_dil_structure["Ref #"]

                                    dil_structure_info = (dil_structureID,
                                                dil_ref,
                                                dil_structure_reference_number,
                                                specific_dil_structure_index
                                                )
                                    
                                    # define target as smallest distance between bx centroid and dil centroid
                                    dil_structure_global_centroid = specific_dil_structure["Structure global centroid"].copy().reshape((3))

                                    vector_between_bx_centroid_and_dil_centroid = dil_structure_global_centroid - bx_structure_global_centroid
                                    distance_between_bx_centroid_and_dil_centroid = np.linalg.norm(vector_between_bx_centroid_and_dil_centroid)


                                    if closest_dil_centroid_info[1] == None:
                                        closest_dil_centroid_info[0] = dil_structure_info
                                        closest_dil_centroid_info[1] = distance_between_bx_centroid_and_dil_centroid
                                    elif distance_between_bx_centroid_and_dil_centroid < closest_dil_centroid_info[1]:
                                        closest_dil_centroid_info[0] = dil_structure_info
                                        closest_dil_centroid_info[1] = distance_between_bx_centroid_and_dil_centroid
                                    else:
                                        pass


                                    # define target as smallest distance between bx centroid and dil surface

                                    dil_interslice_interpolation_information = specific_dil_structure["Inter-slice interpolation information"]       
                                    dil_structure_interpolated_points = dil_interslice_interpolation_information.interpolated_pts_np_arr
                                    dil_kd_tree_scipy = scipy.spatial.KDTree(dil_structure_interpolated_points)
                                    nn_distances, nn_indices = dil_kd_tree_scipy.query(bx_structure_reconstructed_pts, k=1)
                                    closest_surface_to_surface_distance_bx_to_dil = np.amin(nn_distances)

                                    if closest_dil_surface_info[1] == None:
                                        closest_dil_surface_info[0] = dil_structure_info
                                        closest_dil_surface_info[1] = closest_surface_to_surface_distance_bx_to_dil
                                    elif closest_surface_to_surface_distance_bx_to_dil < closest_dil_surface_info[1]:
                                        closest_dil_surface_info[0] = dil_structure_info
                                        closest_dil_surface_info[1] = closest_surface_to_surface_distance_bx_to_dil
                                    else:
                                        pass


                                    dil_distance_dict[dil_structure_info] = {"DIL centroid vector": dil_structure_global_centroid,
                                                                             "Bx centroid vector": bx_structure_global_centroid,
                                                                             "Vector DIL centroid - BX centroid": vector_between_bx_centroid_and_dil_centroid,
                                                                             "X to DIL centroid": vector_between_bx_centroid_and_dil_centroid[0],
                                                                             "Y to DIL centroid": vector_between_bx_centroid_and_dil_centroid[1],
                                                                             "Z to DIL centroid": vector_between_bx_centroid_and_dil_centroid[2],
                                                                             "Distance DIL centroid - BX centroid": distance_between_bx_centroid_and_dil_centroid,
                                                                             "Shortest distance from BX surface to DIL surface": closest_surface_to_surface_distance_bx_to_dil
                                                                            }


                                target_dil_by_centroids_dict = {closest_dil_centroid_info[0]: dil_distance_dict[closest_dil_centroid_info[0]]}
                                target_dil_by_surfaces_dict = {closest_dil_surface_info[0]: dil_distance_dict[closest_dil_surface_info[0]]}

                                


                            if structs == bx_ref:
                                # BX POSITION RELATIVE TO PROSTATE 
                                master_structure_reference_dict[patientUID][structs][specific_structure_index]["Bx location in prostate dict"] = bx_location_in_prostate_ref_frame_dict
                                # TARGET DILS 
                                master_structure_reference_dict[patientUID][structs][specific_structure_index]["Target DIL by centroid dict"] = target_dil_by_centroids_dict
                                master_structure_reference_dict[patientUID][structs][specific_structure_index]["Target DIL by surfaces dict"] = target_dil_by_surfaces_dict
                                master_structure_reference_dict[patientUID][structs][specific_structure_index]["Nearest DILs info dict"] = dil_distance_dict

                            structures_progress.update(processing_structures_task, advance=1)
                    structures_progress.remove_task(processing_structures_task)
                    patients_progress.update(processing_patients_task, advance=1)
                    completed_progress.update(processing_patients_task_completed, advance=1)
                patients_progress.update(processing_patients_task, visible=False)
                completed_progress.update(processing_patients_task_completed,  visible=True)    




                # CALCULATE MEAN BIOPSY UNCERTAINTY FROM MEAN VARIATION OF CENTROIDS
                patientUID_default = "Initializing"
                processing_patient_description = "Determining biopsy uncertainty [{}]...".format(patientUID_default)
                processing_patients_task = patients_progress.add_task("[red]"+processing_patient_description, total = master_structure_info_dict["Global"]["Num cases"])
                processing_patient_description_completed = "Determining biopsy uncertainty"
                processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_description_completed, total=master_structure_info_dict["Global"]["Num cases"], visible=False)

                mean_variation_list = []
                for patientUID,pydicom_item in master_structure_reference_dict.items():

                    processing_patient_description = "Determining biopsy uncertainty  [{}]...".format(patientUID)
                    patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_description)
                                        
                    
                    for specific_structure_index, specific_structure in enumerate(pydicom_item[bx_ref]):
                        # only consider non-simulated biopsies
                        if specific_structure["Simulated bool"] == True:
                            continue
                        
                        mean_variation = specific_structure["Mean centroid variation"]
                        mean_variation_list.append(mean_variation)


                    patients_progress.update(processing_patients_task, advance = 1)
                    completed_progress.update(processing_patients_completed_task, advance = 1)
                patients_progress.update(processing_patients_task, visible = False)
                completed_progress.update(processing_patients_completed_task, visible = True)


                # mean variation 
                mean_variation_arr = np.array(mean_variation_list)
                mean_variation_of_biopsy_centroids_cohort = np.mean(mean_variation_arr)
                #master_cohort_patient_data_and_dataframes["Data"]["Mean biopsy centroid variation"] = mean_variation_of_biopsy_centroids_cohort
                master_structure_info_dict["Global"]["Mean biopsy centroid variation"] = mean_variation_of_biopsy_centroids_cohort


                master_structure_info_dict["Global"]['Preprocessing info']["Preprocessing performed"] = True
                ## END PREPROCESSING             

                
                #live_display.stop()

                ## Up until this point, the master structure reference dictionary contains only pickleable objects!

                # Now can export master structure dict to file!
                if export_pickled_preprocessed_data == True:
                    export_preprocessed_data_task_indeterminate = indeterminate_progress_main.add_task("[red]Exporting preprocessed data...", total=None)
                    export_preprocessed_data_task_indeterminate_completed = completed_progress.add_task("[green]Exporting preprocessed data", visible = False, total=master_structure_info_dict["Global"]["Num cases"])
                    
                    date_time_now = datetime.now()
                    date_time_now_file_name_format = date_time_now.strftime(" Date-%b-%d-%Y Time-%H,%M,%S")
                    global_num_structures = master_structure_info_dict["Global"]["Num structures"]
                    specific_preprocessed_data_dir_name = str(master_structure_info_dict["Global"]["Num cases"])+' patients - '+str(global_num_structures)+' structures - '+date_time_now_file_name_format
                    specific_preprocessed_data_dir = preprocessed_data_dir.joinpath(specific_preprocessed_data_dir_name)
                    specific_preprocessed_data_dir.mkdir(parents=False, exist_ok=False)

                    # Prepare to dump the master_structure_reference_dict to file by first making a copy
                    master_structure_reference_dict_only_picklable = copy.deepcopy(master_structure_reference_dict)

                    # Delete all non-picklable objects from pre-processing
                    for patientUID,pydicom_item in master_structure_reference_dict_only_picklable.items():
                        for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
                            del specific_bx_structure['Point cloud raw']
                            #specific_bx_structure['Delaunay triangulation global structure'].delaunay_line_set = None
                            #for delaunay_obj in specific_bx_structure['Delaunay triangulation zslice-wise list']:
                            #    delaunay_obj.delaunay_line_set = None
                            del specific_bx_structure['Interpolated structure point cloud dict']
                            del specific_bx_structure['Structure OPEN3D triangle mesh object']
                            del specific_bx_structure['Reconstructed structure point cloud']
                            specific_bx_structure['Reconstructed structure delaunay global'].delaunay_line_set = None
                            
                        for specific_oar_structure_index, specific_oar_structure in enumerate(pydicom_item[oar_ref]):
                            del specific_oar_structure['Point cloud raw']
                            #specific_oar_structure['Delaunay triangulation global structure'].delaunay_line_set = None
                            #for delaunay_obj in specific_oar_structure['Delaunay triangulation zslice-wise list']:
                            #    delaunay_obj.delaunay_line_set = None
                            del specific_oar_structure['Interpolated structure point cloud dict']
                            del specific_oar_structure['Structure OPEN3D triangle mesh object']

                        for specific_dil_structure_index, specific_dil_structure in enumerate(pydicom_item[dil_ref]):
                            del specific_dil_structure['Point cloud raw']
                            #del specific_dil_structure['Delaunay triangulation global structure']
                            #for delaunay_obj in specific_dil_structure['Delaunay triangulation zslice-wise list']:
                            #    delaunay_obj.delaunay_line_set = None
                            del specific_dil_structure['Interpolated structure point cloud dict']
                            del specific_dil_structure['Structure OPEN3D triangle mesh object']

                        for specific_rectum_structure_index, specific_rectum_structure in enumerate(pydicom_item[rectum_ref_key]):
                            del specific_rectum_structure['Point cloud raw']
                            #del specific_rectum_structure['Delaunay triangulation global structure']
                            #for delaunay_obj in specific_rectum_structure['Delaunay triangulation zslice-wise list']:
                            #    delaunay_obj.delaunay_line_set = None
                            del specific_rectum_structure['Interpolated structure point cloud dict']
                            del specific_rectum_structure['Structure OPEN3D triangle mesh object']

                        for specific_urethra_structure_index, specific_urethra_structure in enumerate(pydicom_item[urethra_ref_key]):
                            del specific_urethra_structure['Point cloud raw']
                            #del specific_urethra_structure['Delaunay triangulation global structure']
                            #for delaunay_obj in specific_urethra_structure['Delaunay triangulation zslice-wise list']:
                            #    delaunay_obj.delaunay_line_set = None
                            del specific_urethra_structure['Interpolated structure point cloud dict']
                            del specific_urethra_structure['Structure OPEN3D triangle mesh object']

                        if dose_ref in pydicom_item:
                            del pydicom_item[dose_ref]['Dose grid point cloud']
                            del pydicom_item[dose_ref]['Dose grid point cloud thresholded']
                        if mr_adc_ref in pydicom_item:
                            del pydicom_item[mr_adc_ref]["MR ADC grid point cloud"]
                            del pydicom_item[mr_adc_ref]["MR ADC grid point cloud thresholded"]

                    preprocessed_master_structure_ref_dict_path = specific_preprocessed_data_dir.joinpath(preprocessed_master_structure_ref_dict_for_export_name)
                    with open(preprocessed_master_structure_ref_dict_path, 'wb') as master_structure_reference_dict_file:
                        pickle.dump(master_structure_reference_dict_only_picklable, master_structure_reference_dict_file)
                    
                    del master_structure_reference_dict_only_picklable

                    preprocessed_master_structure_ref_info_path = specific_preprocessed_data_dir.joinpath(preprocessed_master_structure_info_dict_for_export_name)
                    with open(preprocessed_master_structure_ref_info_path, 'wb') as master_structure_info_dict_file:
                        pickle.dump(master_structure_info_dict, master_structure_info_dict_file)

                    preprocessed_info_file_name = str(master_structure_info_dict["Global"]["Num cases"])+' patients - '+str(global_num_structures)+' structures.csv'
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
                    completed_progress.update(export_preprocessed_data_task_indeterminate_completed, visible = True, refresh = True, advance=master_structure_info_dict["Global"]["Num cases"])
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



                #### REBUILD NON-PICKLABLE OBJECTS

                # create non-pickleable objects concerning the background dose data
                patientUID_default = "Initializing"
                pickling_dose_patients_task_main_description = "[red]Rebuilding non-picklable dose data [{}]...".format(patientUID_default)
                pickling_dose_patients_task_completed_main_description = "[green]Rebuilding non-picklable dose data"
                pickling_dose_patients_task = patients_progress.add_task(pickling_dose_patients_task_main_description, total=master_structure_info_dict["Global"]["Num cases"])
                pickling_dose_patients_task_completed = completed_progress.add_task(pickling_dose_patients_task_completed_main_description, total=master_structure_info_dict["Global"]["Num cases"], visible = False)

                for patientUID,pydicom_item in master_structure_reference_dict.items():
                    pickling_dose_patients_task_main_description = "[red]Rebuilding non-picklable dose data [{}]...".format(patientUID)
                    patients_progress.update(pickling_dose_patients_task, description = pickling_dose_patients_task_main_description)
                    
                    if dose_ref not in pydicom_item:
                        patients_progress.update(pickling_dose_patients_task, advance=1)
                        completed_progress.update(pickling_dose_patients_task_completed, advance=1)
                        continue

                    dose_ref_dict = pydicom_item[dose_ref]
                    phys_space_dose_map_and_gradient_map_3d_arr = dose_ref_dict["Dose and gradient phys space and pixel 3d arr"]
                    #phys_space_dose_map_3d_arr = phys_space_dose_map_and_gradient_map_3d_arr[:, :, :7]

                    # create dose point cloud and thresholded dose point cloud
                    #dose_point_cloud = plotting_funcs.create_dose_point_cloud(phys_space_dose_map_3d_arr, color_flattening_deg, paint_dose_color = True)
                    #thresholded_dose_point_cloud = plotting_funcs.create_thresholded_dose_point_cloud(phys_space_dose_map_3d_arr, color_flattening_deg, paint_dose_color = True, lower_bound_percent = lower_bound_dose_percent)
                    
                    dose_point_cloud, dose_gradient_arrows_point_cloud = plotting_funcs.create_dose_point_cloud_with_gradients(phys_space_dose_map_and_gradient_map_3d_arr,
                                                                                                                        paint_dose_color=True,
                                                                                                                        arrow_scale=1.0,
                                                                                                                        truncate_below_dose=None,
                                                                                                                        truncate_below_gradient_norm=None
                                                                                                                    )
                    thresholded_dose_point_cloud, thresholded_dose_gradient_arrows_point_cloud = plotting_funcs.create_dose_point_cloud_with_gradients(phys_space_dose_map_and_gradient_map_3d_arr,
                                                                                                                        paint_dose_color=True,
                                                                                                                        arrow_scale=1.0,
                                                                                                                        truncate_below_dose=lower_bound_dose_value,
                                                                                                                        truncate_below_gradient_norm=lower_bound_dose_gradient_value
                                                                                                                    )

                    dose_ref_dict["Dose grid point cloud"] = dose_point_cloud
                    dose_ref_dict["Dose grid point cloud thresholded"] = thresholded_dose_point_cloud
                    dose_ref_dict["Dose grid gradient point cloud"] = dose_gradient_arrows_point_cloud
                    dose_ref_dict["Dose grid gradient point cloud thresholded"] = thresholded_dose_gradient_arrows_point_cloud

                    master_structure_reference_dict[patientUID][dose_ref] = dose_ref_dict

                    

                    patients_progress.update(pickling_dose_patients_task, advance=1)
                    completed_progress.update(pickling_dose_patients_task_completed, advance=1)
                patients_progress.update(pickling_dose_patients_task, visible=False)
                completed_progress.update(pickling_dose_patients_task_completed,  visible=True)        
                
                # Create non-pickleable objkects concerning MR data
                patientUID_default = "Initializing"
                pickling_MR_patients_task_main_description = "[red]Rebuilding non-picklable MR data [{}]...".format(patientUID_default)
                pickling_MR_patients_task_completed_main_description = "[green]Rebuilding non-picklable MR data"
                pickling_MR_patients_task = patients_progress.add_task(pickling_MR_patients_task_main_description, total=master_structure_info_dict["Global"]["Num cases"])
                pickling_MR_patients_task_completed = completed_progress.add_task(pickling_MR_patients_task_completed_main_description, total=master_structure_info_dict["Global"]["Num cases"], visible = False)

                for patientUID,pydicom_item in master_structure_reference_dict.items():
                    pickling_MR_patients_task_main_description = "[red]Rebuilding non-picklable MR data [{}]...".format(patientUID)
                    patients_progress.update(pickling_MR_patients_task, description = pickling_MR_patients_task_main_description)


                    if mr_adc_ref not in pydicom_item:
                        patients_progress.update(pickling_MR_patients_task, advance=1)
                        completed_progress.update(pickling_MR_patients_task_completed, advance=1)
                        continue
                        

                    mr_adc_subdict = pydicom_item[mr_adc_ref]

                    filtered_non_negative_adc_mr_phys_space_arr = lattice_reconstruction_tools.reconstruct_mr_lattice_with_coordinates_from_dict_v2(mr_adc_subdict, filter_out_negatives = True)
                    # Don't store this, it is too large, just call the above function if you want to retrieve the MR information lattice
                    #mr_adc_subdict["MR ADC phys space Nx4 arr (filtered, non-negative)"] = filtered_non_negative_adc_mr_phys_space_arr


                    mr_adc_point_cloud = plotting_funcs.create_MR_point_cloud(filtered_non_negative_adc_mr_phys_space_arr, 
                                                                                        color_flattening_deg_MR, 
                                                                                        paint_mr_color = True)
                    
                    thresholded_mr_adc_point_cloud = plotting_funcs.create_thresholded_MR_ADC_point_cloud(filtered_non_negative_adc_mr_phys_space_arr, 
                                                                                                                    color_flattening_deg_MR, 
                                                                                                                    paint_mr_color = True, 
                                                                                                                    lower_bound = lower_bound_mr_adc_value, 
                                                                                                                    upper_bound = upper_bound_mr_adc_value 
                                                                                                                    )


                    mr_adc_subdict["MR ADC grid point cloud"] = mr_adc_point_cloud
                    mr_adc_subdict["MR ADC grid point cloud thresholded"] = thresholded_mr_adc_point_cloud

                    del filtered_non_negative_adc_mr_phys_space_arr
                    

                    patients_progress.update(pickling_MR_patients_task, advance=1)
                    completed_progress.update(pickling_MR_patients_task_completed, advance=1)
                patients_progress.update(pickling_MR_patients_task, visible=False)
                completed_progress.update(pickling_MR_patients_task_completed,  visible=True)



                
                

                
                patientUID_default = "Initializing"
                pickling_structure_patients_task_main_description = "[red]Rebuilding non-picklable structure data [{}]...".format(patientUID_default)
                pickling_structure_patients_task_completed_main_description = "[green]Rebuilding non-picklable structure data"
                pickling_structure_patients_task = patients_progress.add_task(pickling_structure_patients_task_main_description, total=master_structure_info_dict["Global"]["Num cases"])
                pickling_structure_patients_task_completed = completed_progress.add_task(pickling_structure_patients_task_completed_main_description, total=master_structure_info_dict["Global"]["Num cases"], visible = False)

                for patientUID,pydicom_item in master_structure_reference_dict.items():
                    pickling_structure_patients_task_main_description = "[red]Rebuilding non-picklable structure data [{}]...".format(patientUID)
                    patients_progress.update(pickling_structure_patients_task, description = pickling_structure_patients_task_main_description)
                    for structs in structs_referenced_list_generalized:
                        for specific_structure_index, specific_structure in enumerate(pydicom_item[structs]):
                            specific_structure_roi = specific_structure["ROI"]
                            ###
                            # Creating pointcloud dictionary of the interpolation done
                            indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Creating pcds of interp structures [{}]".format(specific_structure_roi), total = None)

                            interslice_interpolation_information = pydicom_item[structs][specific_structure_index]["Inter-slice interpolation information"]
                            interpolation_information = pydicom_item[structs][specific_structure_index]["Intra-slice interpolation information"]
                            threeDdata_array_fully_interpolated = interpolation_information.interpolated_pts_np_arr
                            threeDdata_array_fully_interpolated_with_end_caps = interpolation_information.interpolated_pts_with_end_caps_np_arr
                            threeDdata_array_interslice_interpolation = np.vstack(interslice_interpolation_information.interpolated_pts_list)
                            #pcd_struct_rand_color = np.random.uniform(0, 0.9, size=3)
                            if structs == bx_ref:
                                sim_type = specific_structure["Simulated type"]
                                pcd_struct_color = structs_referenced_dict[structs]['PCD color dict'][sim_type]
                            else:
                                pcd_struct_color = structs_referenced_dict[structs]['PCD color']
                            interslice_interp_pcd = point_containment_tools.create_point_cloud(threeDdata_array_interslice_interpolation, pcd_struct_color)
                            inter_and_intra_interp_pcd = point_containment_tools.create_point_cloud(threeDdata_array_fully_interpolated, pcd_struct_color)
                            inter_and_intra_and_end_caps_interp_pcd = point_containment_tools.create_point_cloud(threeDdata_array_fully_interpolated_with_end_caps, pcd_struct_color)
                            interpolated_pcd_dict = {"Interslice": interslice_interp_pcd, "Full": inter_and_intra_interp_pcd, "Full with end caps": inter_and_intra_and_end_caps_interp_pcd}
                            master_structure_reference_dict[patientUID][structs][specific_structure_index]["Interpolated structure point cloud dict"] = interpolated_pcd_dict

                            indeterminate_progress_sub.update(indeterminate_task, visible = False)
                            ###

                            ###
                            # creating pointcloud of the raw contour points
                            indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Creating pcds of raw structures [{}]".format(specific_structure_roi), total = None)

                            threeDdata_array = pydicom_item[structs][specific_structure_index]["Raw contour pts"]
                            #threeDdata_pcd_color = np.random.uniform(0, 0.7, size=3)
                            threeDdata_point_cloud = point_containment_tools.create_point_cloud(threeDdata_array, pcd_struct_color)
                            master_structure_reference_dict[patientUID][structs][specific_structure_index]["Point cloud raw"] = threeDdata_point_cloud

                            indeterminate_progress_sub.update(indeterminate_task, visible = False)
                            ###

                            """ DEPRECATED! No longer need delaunays except for reconstructed biopsies, which even then could be recoded to not need it.
                            ###
                            # creating the lineset of the delaunay global convex structure
                            indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Creating delaunay linesets 1 [{}]".format(specific_structure_roi), total = None)

                            delaunay_global_convex_structure_obj = pydicom_item[structs][specific_structure_index]["Delaunay triangulation global structure"]
                            delaunay_global_convex_structure_obj.generate_lineset()
                            master_structure_reference_dict[patientUID][structs][specific_structure_index]["Delaunay triangulation global structure"] = delaunay_global_convex_structure_obj
                            
                            indeterminate_progress_sub.update(indeterminate_task, visible = False)
                            ###

                            ###
                            # creating lineset of the zslice wise delaunay convex structure
                            indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Creating delaunay linesets 2 [{}]".format(specific_structure_roi), total = None)

                            delaunay_triangulation_obj_zslicewise_list = pydicom_item[structs][specific_structure_index]["Delaunay triangulation zslice-wise list"]
                            for delaunay_obj in delaunay_triangulation_obj_zslicewise_list:
                                delaunay_obj.generate_lineset()
                            master_structure_reference_dict[patientUID][structs][specific_structure_index]["Delaunay triangulation zslice-wise list"] = delaunay_triangulation_obj_zslicewise_list
                            
                            indeterminate_progress_sub.update(indeterminate_task, visible = False)
                            ###
                            """
                            
                            ###
                            # trimesh
                            indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Creating trimesh [{}]".format(specific_structure_roi), total = None)
                            live_display.refresh()

                            fully_interp_with_end_caps_structure_triangle_mesh, _ = misc_tools.compute_structure_triangle_mesh(interp_inter_slice_dist, 
                                interp_intra_slice_dist,
                                threeDdata_array_fully_interpolated_with_end_caps,
                                radius_for_normals_estimation,
                                max_nn_for_normals_estimation
                                )
                            master_structure_reference_dict[patientUID][structs][specific_structure_index]["Structure OPEN3D triangle mesh object"] = fully_interp_with_end_caps_structure_triangle_mesh
                            
                            indeterminate_progress_sub.update(indeterminate_task, visible = False)
                            ###

                            # For biopsies only
                            if structs == bx_ref:
                                ###
                                # creating pointcloud of the reconstructed biopsy
                                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Creating pcd of rcn bpsy [{}]".format(specific_structure_roi), total = None)

                                drawn_biopsy_array = pydicom_item[structs][specific_structure_index]["Reconstructed structure pts arr"] 
                                
                            
                                reconstructed_biopsy_point_cloud = point_containment_tools.create_point_cloud(drawn_biopsy_array, pcd_struct_color)
                                master_structure_reference_dict[patientUID][structs][specific_structure_index]["Reconstructed structure point cloud"] = reconstructed_biopsy_point_cloud
                                
                                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                                ###

                                ###
                                # creating lineset of the reconstructed biopsy global delaunay object
                                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Creating delaunay lineset of rcn bpsy [{}]".format(specific_structure_roi), total = None)

                                reconstructed_bx_delaunay_global_convex_structure_obj = pydicom_item[structs][specific_structure_index]["Reconstructed structure delaunay global"]
                                reconstructed_bx_delaunay_global_convex_structure_obj.generate_lineset()
                                master_structure_reference_dict[patientUID][structs][specific_structure_index]["Reconstructed structure delaunay global"] = reconstructed_bx_delaunay_global_convex_structure_obj
                                
                                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                                ###

                    patients_progress.update(pickling_structure_patients_task, advance=1)
                    completed_progress.update(pickling_structure_patients_task_completed, advance=1)
                patients_progress.update(pickling_structure_patients_task, visible=False)
                completed_progress.update(pickling_structure_patients_task_completed,  visible=True)  







            rich_preambles.section_completed("Preprocessing", section_start_time, completed_progress, completed_sections_manager)


            section_start_time = datetime.now() 

            ### IMPORTANT THAT THIS IS PLACED PRECISELY HERE!!
            # Create the specific output directory folder
            date_time_now = datetime.now()
            date_time_now_file_name_format = date_time_now.strftime(" Date-%b-%d-%Y Time-%H,%M,%S")
            specific_output_dir_name = 'MC_sim_out-'+date_time_now_file_name_format
            specific_output_dir = output_dir.joinpath(specific_output_dir_name)
            specific_output_dir.mkdir(parents=False, exist_ok=True)

            master_structure_info_dict["Global"]["Specific output dir"] = specific_output_dir
            ###
            

            ### MAKE ALL DIRECTORIES!
            raw_mc_output_folder_name = 'Raw MC output'
            raw_mc_output_dir = specific_output_dir.joinpath(raw_mc_output_folder_name)
            raw_mc_output_dir.mkdir(parents=True, exist_ok=True)

            master_structure_info_dict["Global"]["Raw MC output dir"] = raw_mc_output_dir
        

            

            live_display.refresh()
        
            """
            # plot dose point cloud cubic lattice (color only)
            if show_3d_dose_renderings == True:
                for patientUID,pydicom_item in master_structure_reference_dict.items():
                    if dose_ref not in pydicom_item:
                        continue
                    
                    dose_point_cloud = pydicom_item[dose_ref]["Dose grid point cloud"]
                
                    stopwatch.stop()
                    plotting_funcs.plot_geometries(dose_point_cloud)
                    stopwatch.start()
                            

            # plot dose point cloud thresholded cubic lattice (color only)
            if show_3d_dose_renderings == True:
                for patientUID,pydicom_item in master_structure_reference_dict.items():
                    if dose_ref not in pydicom_item:
                        continue

                    dose_point_cloud_thresholded = pydicom_item[dose_ref]["Dose grid point cloud thresholded"]
                
                    stopwatch.stop()
                    plotting_funcs.plot_geometries(dose_point_cloud_thresholded)
                    stopwatch.start()

            """

            # displays 3d renderings of patient contour data, dose data and MR data
            if show_processed_3d_datasets_renderings == True:
                live_display.stop()
                for patientUID,pydicom_item in master_structure_reference_dict.items():
                    print(patientUID)
                    pcd_list = list()

                    # Include dose pcd
                    if dose_ref in pydicom_item:
                        dose_ref_dict = pydicom_item[dose_ref]
                        dose_grid_pcd = dose_ref_dict["Dose grid point cloud thresholded"]
                        dose_grid_gradient_pcd = dose_ref_dict["Dose grid gradient point cloud thresholded"]

                    # Include MR pcd if present
                    if mr_adc_ref in pydicom_item:
                        mr_adc_subdict = pydicom_item[mr_adc_ref]
                        thresholded_mr_adc_point_cloud = mr_adc_subdict["MR ADC grid point cloud thresholded"]

                    for structs in structs_referenced_list:
                        for specific_structure_index, specific_structure in enumerate(pydicom_item[structs]):
                            if structs == bx_ref: 
                                structure_pcd = specific_structure["Reconstructed structure point cloud"]
                            else: 
                                #structure_pcd = specific_structure["Point cloud raw"]
                                structure_pcd = specific_structure["Interpolated structure point cloud dict"]["Full"]
                            pcd_list.append(structure_pcd)

                    # Only plot the structures
                    plotting_funcs.plot_geometries(*pcd_list)

                    # Include the dosimetry
                    if dose_ref in pydicom_item:
                        pcd_list_dose = pcd_list + [dose_grid_pcd, dose_grid_gradient_pcd]
                        plotting_funcs.plot_geometries(*pcd_list_dose)
                        
                        del pcd_list_dose


                    # Include the MR ADC data
                    if mr_adc_ref in pydicom_item:

                        pcd_list_MR_ADC = pcd_list + [thresholded_mr_adc_point_cloud]
                        plotting_funcs.plot_geometries(*pcd_list_MR_ADC)
                        
                        del pcd_list_MR_ADC

                    
                    if (mr_adc_ref in pydicom_item) and (dose_ref in pydicom_item):

                        # Include the MR ADC data and dose
                        pcd_list_MR_ADC_and_dose = pcd_list + [thresholded_mr_adc_point_cloud, dose_grid_pcd, dose_grid_gradient_pcd]
                        plotting_funcs.plot_geometries(*pcd_list_MR_ADC_and_dose)

                        del pcd_list_MR_ADC_and_dose

                    del pcd_list
                    
                live_display.start()


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
            master_structure_info_dict["Global"]["MC info"]["BX sample pt lattice spacing (mm)"] = bx_sample_pts_lattice_spacing
            master_structure_info_dict["Global"]["MC info"]["BX sample pt volume element (mm^3)"] = bx_sample_pts_lattice_spacing**3

            patientUID_default = "Initializing"
            processing_patient_parallel_computing_main_description = "Preparing patient for parallel processing [{}]...".format(patientUID_default)
            processing_patients_task = patients_progress.add_task("[red]"+processing_patient_parallel_computing_main_description, total = master_structure_info_dict["Global"]["Num cases"])
            processing_patient_parallel_computing_main_description_completed = "Preparing patient for parallel processing"
            processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_parallel_computing_main_description_completed, total=master_structure_info_dict["Global"]["Num cases"], visible=False)

            for patientUID,pydicom_item in master_structure_reference_dict.items():

                processing_patient_parallel_computing_main_description = "Preparing patient for parallel processing [{}]...".format(patientUID)
                patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_parallel_computing_main_description)
                
                num_biopsies_per_patient = master_structure_info_dict["By patient"][patientUID][bx_ref]["Num structs"]
                biopsyID_default = "Initializing"
                processing_biopsies_main_description = "[cyan]Preparing biopsy data for parallel processing [{},{}]...".format(patientUID,biopsyID_default)
                processing_biopsies_task = biopsies_progress.add_task(processing_biopsies_main_description, total=num_biopsies_per_patient)

                for specific_structure_index, specific_structure in enumerate(pydicom_item[bx_ref]):
                    specific_bx_structure_roi = specific_structure["ROI"]
                    processing_biopsies_main_description = "[cyan]Preparing biopsy data for parallel processing [{},{}]...".format(patientUID,specific_bx_structure_roi)
                    biopsies_progress.update(processing_biopsies_task, description = processing_biopsies_main_description)
                    reconstructed_biopsy_point_cloud = master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Reconstructed structure point cloud"]
                    reconstructed_biopsy_arr = master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Reconstructed structure pts arr"]
                    reconstructed_delaunay_global_convex_structure_obj = master_structure_reference_dict[patientUID][bx_ref][specific_structure_index]["Reconstructed structure delaunay global"]
                    
                    z_axis_np_vec = np.array([0,0,1],dtype=float)
                    apex_to_base_bx_best_fit_vec = specific_structure["Centroid line vec (bx needle base to bx needle tip)"]
                    z_axis_to_centroid_vec_rotation_matrix = mf.rotation_matrix_from_vectors(z_axis_np_vec,apex_to_base_bx_best_fit_vec)
                    
                    args_list.append((bx_sample_pts_lattice_spacing, reconstructed_delaunay_global_convex_structure_obj.delaunay_triangulation, reconstructed_biopsy_arr, patientUID, bx_ref, specific_structure_index,z_axis_to_centroid_vec_rotation_matrix))
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
            sampling_points_task_indeterminate_completed = completed_progress.add_task("[green]Sampling points from all patient biopsies (parallel)", visible = False, total = master_structure_info_dict["Global"]["Num cases"])
            #parallel_results_sampled_bx_points_from_global_delaunay_arr_and_bounding_box_arr = parallel_pool.starmap(MC_simulator_convex.grid_point_sampler_from_global_delaunay_convex_structure_parallel, args_list)
            parallel_results_sampled_bx_points_from_global_delaunay_arr_and_bounding_box_arr = parallel_pool.starmap(MC_simulator_convex.grid_point_sampler_rotated_from_global_delaunay_convex_structure_parallel, args_list)

            indeterminate_progress_main.update(sampling_points_task_indeterminate, visible = False, refresh = True)
            completed_progress.update(sampling_points_task_indeterminate_completed, advance = master_structure_info_dict["Global"]["Num cases"], visible = True, refresh = True)
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
            processing_patients_task = patients_progress.add_task("[red]"+processing_patient_rotating_bx_main_description, total = master_structure_info_dict["Global"]["Num cases"])
            processing_patient_rotating_bx_main_description_completed = "Creating biopsy oriented coordinate system"
            processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_rotating_bx_main_description_completed, total=master_structure_info_dict["Global"]["Num cases"], visible=False)

            # rotating pointclouds to create bx oriented frame of reference
            for patientUID,pydicom_item in master_structure_reference_dict.items():

                processing_patient_rotating_bx_main_description = "Creating biopsy oriented coordinate system [{}]...".format(patientUID)
                patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_rotating_bx_main_description)
                
                num_biopsies_per_patient = master_structure_info_dict["By patient"][patientUID][bx_ref]["Num structs"]
                biopsyID_default = "Initializing"
                

                for specific_structure_index, specific_structure in enumerate(pydicom_item[bx_ref]):
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





            ### UNCERTAINTIES 



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
            
            #num_general_structs = master_structure_info_dict["Global"]["Num structures"]

            """
            uncertainty_file_writer.uncertainty_file_preper_sigma_by_struct_type(uncertainties_file, 
                                                                                master_structure_reference_dict, 
                                                                                structs_referenced_list, 
                                                                                num_general_structs, 
                                                                                structs_referenced_dict,
                                                                                biopsy_variation_uncertainty_setting,
                                                                                master_structure_info_dict
                                                                                )
            """

            uncertainties_dataframe = uncertainty_file_writer.uncertainty_file_preper_by_struct_type_dataframe_NEW(master_structure_reference_dict, 
                                                 structs_referenced_list, 
                                                 structs_referenced_dict,
                                                 biopsy_variation_uncertainty_setting,
                                                 non_biopsy_variation_uncertainty_setting,
                                                 use_added_in_quad_errors_as,
                                                 master_structure_info_dict
                                                 )

            uncertainties_dataframe.to_csv(uncertainties_file)
            master_cohort_patient_data_and_dataframes["Dataframes"]["Uncertainties dataframe (unedited)"] = uncertainties_dataframe
            
            """
            f = open(uncertainties_file, "w", newline='\n')
            writer = csv.writer(f)
            """       

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
                            pass
            else:
                # this is run if the uncertainty file is not to be modified by the user before running the simulation
                uncertainties_file_filled = uncertainties_file
                read_uncertainties_dataframe = pandas.read_csv(uncertainties_file_filled)  


            ### Save the potentially edited uncertainties file
            master_cohort_patient_data_and_dataframes["Dataframes"]["Uncertainties dataframe (final)"] = read_uncertainties_dataframe     

            # Transfer read uncertainty data to master_reference
            for index, row in read_uncertainties_dataframe.iterrows():
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


                uncertainty_data_obj = uncertainty_data(patientUID, 
                                                        structure_type, 
                                                        structure_ROI, 
                                                        structure_ref_num, 
                                                        master_ref_dict_specific_structure_index, 
                                                        frame_of_reference
                                                        )
                
                uncertainty_data_obj.fill_means_and_sigmas(means_arr, sigmas_arr)
                
                master_structure_reference_dict[patientUID][structure_type][master_ref_dict_specific_structure_index]["Uncertainty data"] = uncertainty_data_obj


            """
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

            """


            rich_preambles.section_completed("Preparing for simulations", section_start_time, completed_progress, completed_sections_manager)


            section_start_time = datetime.now() 



            ######
            ###### THIS IS WHERE THE IF STATEMENT OF THE FANOVA AND MC SIM WAS WAS MOVED TO! IF CODE BREAKS JUST DELETE THE IF BELOW!
            ######                
            if (perform_MC_sim == True or perform_fanova == True):
                
                master_structure_info_dict["Global"]["MC info"]["Num MC containment simulations"] = num_MC_containment_simulations_input
                master_structure_info_dict["Global"]["MC info"]["Num MC dose simulations"] = num_MC_dose_simulations_input
                master_structure_info_dict["Global"]["MC info"]["Num MC MR simulations"] = num_MC_MR_simulations_input

                # Determine and store max simulations
                max_simulations = max(num_MC_dose_simulations_input,
                                      num_MC_containment_simulations_input, 
                                      num_MC_MR_simulations_input)
                master_structure_info_dict["Global"]["MC info"]["Max of num MC simulations"] = max_simulations

                num_biopsies_global = master_structure_info_dict["Global"]["Num biopsies"]
                #num_OARs_global = master_structure_info_dict["Global"]["Num OARs"]
                #num_DILs_global = master_structure_info_dict["Global"]["Num DILs"]
                bx_sample_pt_lattice_spacing = master_structure_info_dict["Global"]["MC info"]["BX sample pt lattice spacing (mm)"]
                num_global_structures = master_structure_info_dict["Global"]["Num structures"]
                num_cases = master_structure_info_dict["Global"]["Num cases"]
                num_unique_patients = master_structure_info_dict['Global']['Num unique patient names']

                # Output simulation information
                simulation_info_important_line_str = f"Simulation data: # MC tissue class sims = {num_MC_containment_simulations_input} | # MC dose sims = {num_MC_dose_simulations_input} | # MC MR sims = {num_MC_MR_simulations_input} | Lattice spacing for BX cores (mm) = {bx_sample_pt_lattice_spacing} | # biopsies = {num_biopsies_global} | # anatomical structures = {num_global_structures-num_biopsies_global} | # cases = {num_cases} | # unique patients {num_unique_patients}."
                important_info.add_text_line(simulation_info_important_line_str, live_display)

                
                ### Shift all structures and biopsies the same way for all simulations 
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Shifting biopsies & structs ", total = None)

                #live_display.stop()
                # Generate shifts
                MC_prepper_funcs.generate_shifts(master_structure_reference_dict,
                                                simulate_uniform_bx_shifts_due_to_bx_needle_compartment,
                                                bx_ref,
                                                biopsy_needle_compartment_length,
                                                max_simulations,
                                                structs_referenced_list)
                
                
                
                # Shift anatomy
                MC_prepper_funcs.biopsy_and_structure_shifter(master_structure_reference_dict,
                                 bx_ref,
                                 structs_referenced_list,
                                 simulate_uniform_bx_shifts_due_to_bx_needle_compartment,
                                 max_simulations,
                                 plot_uniform_shifts_to_check_plotly,
                                 plot_shifted_biopsies = plot_shifted_biopsies,
                                 plot_translation_vectors_pointclouds = plot_translation_vectors_pointclouds
                                 )


                #live_display.start()          

                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                ### End shifting anatomy


            
                #live_display.stop()
                # Run MC simulation
                if perform_MC_sim == True:
                    master_structure_reference_dict, master_structure_info_dict, live_display = MC_simulator_convex.simulator_parallel(parallel_pool, 
                                                                                            live_display,
                                                                                            stopwatch, 
                                                                                            layout_groups, 
                                                                                            master_structure_reference_dict, 
                                                                                            structs_referenced_list,
                                                                                            structs_referenced_dict,
                                                                                            bx_ref,
                                                                                            oar_ref,
                                                                                            dil_ref, 
                                                                                            dose_ref,
                                                                                            plan_ref,
                                                                                            all_ref_key, 
                                                                                            master_structure_info_dict, 
                                                                                            biopsy_z_voxel_length, 
                                                                                            num_dose_calc_NN, 
                                                                                            num_dose_NN_to_show_for_animation_plotting,
                                                                                            dose_views_jsons_paths_list,
                                                                                            containment_views_jsons_paths_list,
                                                                                            show_NN_dose_demonstration_plots,
                                                                                            show_NN_dose_demonstration_plots_all_trials_at_once,
                                                                                            show_containment_demonstration_plots,
                                                                                            plot_nearest_neighbour_surface_boundary_demonstration,
                                                                                            plot_relative_structure_centroid_demonstration,
                                                                                            biopsy_needle_compartment_length,
                                                                                            simulate_uniform_bx_shifts_due_to_bx_needle_compartment,
                                                                                            plot_uniform_shifts_to_check_plotly,
                                                                                            differential_dvh_resolution,
                                                                                            cumulative_dvh_resolution,
                                                                                            v_percent_DVH_to_calc_list,
                                                                                            volume_DVH_quantiles_to_calculate,
                                                                                            plot_translation_vectors_pointclouds,
                                                                                            plot_cupy_containment_distribution_results,
                                                                                            plot_shifted_biopsies,
                                                                                            structure_miss_probability_roi,
                                                                                            cancer_tissue_label,
                                                                                            default_exterior_tissue,
                                                                                            miss_structure_complement_label,
                                                                                            tissue_length_above_probability_threshold_list,
                                                                                            n_bootstraps_for_tissue_length_above_threshold,
                                                                                            perform_mc_containment_sim,
                                                                                            perform_mc_dose_sim,
                                                                                            spinner_type,
                                                                                            cupy_array_upper_limit_NxN_size_input,
                                                                                            nearest_zslice_vals_and_indices_cupy_generic_max_size,
                                                                                            idw_power,
                                                                                            raw_data_mc_dosimetry_dump_bool, 
                                                                                            raw_data_mc_containment_dump_bool
                                                                                            )

                    if no_cohort_mr_adc_flag == False:
                        master_structure_reference_dict, master_structure_info_dict, live_display = MC_simulator_MR.simulator_parallel(live_display,
                                                                                                        layout_groups,
                                                                                                        master_structure_reference_dict,
                                                                                                        master_structure_info_dict,
                                                                                                        structs_referenced_list,
                                                                                                        mr_adc_ref,
                                                                                                        bx_ref,
                                                                                                        num_mr_calc_NN,
                                                                                                        idw_power,
                                                                                                        raw_data_mc_MR_dump_bool,
                                                                                                        show_NN_mr_adc_demonstration_plots,
                                                                                                        stopwatch,
                                                                                                        dose_views_jsons_paths_list,
                                                                                                        perform_mc_mr_sim,
                                                                                                        show_NN_mr_adc_demonstration_plots_all_trials_at_once)


                    mc_containment_sim_complete = master_structure_info_dict['Global']["MC info"]['MC containment sim performed']  
                    mc_dose_sim_complete = master_structure_info_dict['Global']["MC info"]['MC dose sim performed']
                    mc_mr_sim_complete = master_structure_info_dict['Global']["MC info"]['MC MR sim performed']

                    list_of_mc_sim_types = [mc_dose_sim_complete,mc_containment_sim_complete,mc_mr_sim_complete]

                    master_structure_info_dict['Global']["MC info"]['MC sim performed'] = any(list_of_mc_sim_types)
                
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
                        dil_ref,
                        oar_ref,
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
                        perform_containment_fanova,
                        structure_miss_probability_roi,
                        cancer_tissue_label,
                        nearest_zslice_vals_and_indices_cupy_generic_max_size,
                        cupy_array_upper_limit_NxN_size_input,
                        plot_cupy_containment_distribution_fanova_results
                        )

                live_display.start(refresh=True)
                #live_display.stop()

                

                # copy uncertainty file used for simulation to output folder 
                shutil.copy(uncertainties_file_filled, specific_output_dir)

                
                ## PREPARE TO PICKLE MASTER STRUCTURE REFERENCE DICT, DELETE ALL UNPICKLEABLE ITEMS
                for patientUID,pydicom_item in master_structure_reference_dict.items():
                    for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
                        #del specific_bx_structure['Intra-slice interpolation information']
                        #del specific_bx_structure['Inter-slice interpolation information']
                        del specific_bx_structure['Point cloud raw']
                        #del specific_bx_structure['Delaunay triangulation global structure']
                        #del specific_bx_structure['Delaunay triangulation zslice-wise list']
                        del specific_bx_structure['Interpolated structure point cloud dict']
                        del specific_bx_structure['Reconstructed structure point cloud']
                        del specific_bx_structure['Reconstructed structure delaunay global']
                        del specific_bx_structure['Random uniformly sampled volume pts pcd']
                        del specific_bx_structure['Random uniformly sampled volume pts bx coord sys pcd']
                        del specific_bx_structure['Bounding box for random uniformly sampled volume pts']
                        del specific_bx_structure['Uncertainty data']
                        del specific_bx_structure['MC data: bx and structure shifted dict']
                        #del specific_bx_structure['MC data: compiled sim results']
                        del specific_bx_structure['MC data: bx to dose NN search objects list']
                        #del specific_bx_structure['MC data: Dose NN child obj for each sampled bx pt list (nominal & all MC trials)']
                        del specific_bx_structure['FANOVA: sobol indices (containment)']
                        del specific_bx_structure['FANOVA: sobol indices (dose)']
                        del specific_bx_structure['FANOVA: sobol indices (DIL tissue)']
                        #del specific_bx_structure['Structure OPEN3D triangle mesh object']
                        
                    for specific_oar_structure_index, specific_oar_structure in enumerate(pydicom_item[oar_ref]):
                        #del specific_oar_structure['Intra-slice interpolation information']
                        #del specific_oar_structure['Inter-slice interpolation information']
                        del specific_oar_structure['Point cloud raw']
                        #del specific_oar_structure['Delaunay triangulation global structure']
                        #del specific_oar_structure['Delaunay triangulation zslice-wise list']
                        del specific_oar_structure['Interpolated structure point cloud dict']
                        del specific_oar_structure['Uncertainty data']
                        del specific_oar_structure['Structure OPEN3D triangle mesh object']

                    for specific_dil_structure_index, specific_dil_structure in enumerate(pydicom_item[dil_ref]):
                        #del specific_dil_structure['Intra-slice interpolation information']
                        #del specific_dil_structure['Inter-slice interpolation information']
                        del specific_dil_structure['Point cloud raw']
                        #del specific_dil_structure['Delaunay triangulation global structure']
                        #del specific_dil_structure['Delaunay triangulation zslice-wise list']
                        del specific_dil_structure['Interpolated structure point cloud dict']
                        del specific_dil_structure['Uncertainty data']
                        del specific_dil_structure['Structure OPEN3D triangle mesh object']

                    for specific_rectum_structure_index, specific_rectum_structure in enumerate(pydicom_item[rectum_ref_key]):
                        #del specific_rectum_structure['Intra-slice interpolation information']
                        #del specific_rectum_structure['Inter-slice interpolation information']
                        del specific_rectum_structure['Point cloud raw']
                        #del specific_rectum_structure['Delaunay triangulation global structure']
                        #del specific_rectum_structure['Delaunay triangulation zslice-wise list']
                        del specific_rectum_structure['Interpolated structure point cloud dict']
                        del specific_rectum_structure['Uncertainty data']
                        del specific_rectum_structure['Structure OPEN3D triangle mesh object']

                    for specific_urethra_structure_index, specific_urethra_structure in enumerate(pydicom_item[urethra_ref_key]):
                        #del specific_urethra_structure['Intra-slice interpolation information']
                        #del specific_urethra_structure['Inter-slice interpolation information']
                        del specific_urethra_structure['Point cloud raw']
                        #del specific_urethra_structure['Delaunay triangulation global structure']
                        #del specific_urethra_structure['Delaunay triangulation zslice-wise list']
                        del specific_urethra_structure['Interpolated structure point cloud dict']
                        del specific_urethra_structure['Uncertainty data']
                        del specific_urethra_structure['Structure OPEN3D triangle mesh object']

                    if dose_ref in pydicom_item:
                        del pydicom_item[dose_ref]['Dose grid point cloud']
                        del pydicom_item[dose_ref]['Dose grid point cloud thresholded']
                        del pydicom_item[dose_ref]["Dose grid gradient point cloud"]
                        del pydicom_item[dose_ref]["Dose grid gradient point cloud thresholded"]
                        del pydicom_item[dose_ref]['KDtree']
                    if mr_adc_ref in pydicom_item:
                        del pydicom_item[mr_adc_ref]["MR ADC grid point cloud"]
                        del pydicom_item[mr_adc_ref]["MR ADC grid point cloud thresholded"]
                        del pydicom_item[mr_adc_ref]["KDtree"]



                
                
                date_time_now = datetime.now()
                date_time_now_file_name_format = date_time_now.strftime(" Date-%b-%d-%Y Time-%H,%M,%S")
                global_num_structures = master_structure_info_dict["Global"]["Num structures"]
                specific_output_pickle_data_dir_name = str(master_structure_info_dict["Global"]["Num cases"])+' patients - '+str(global_num_structures)+' structures - '+date_time_now_file_name_format+' pickled data'
                specific_output_pickle_data_dir = specific_output_dir.joinpath(specific_output_pickle_data_dir_name)
                specific_output_pickle_data_dir.mkdir(parents=False, exist_ok=False)

                pickled_output_master_structure_ref_dict_path = specific_output_pickle_data_dir.joinpath(output_master_structure_ref_dict_for_export_name)
                with open(pickled_output_master_structure_ref_dict_path, 'wb') as master_structure_reference_dict_file:
                    pickle.dump(master_structure_reference_dict, master_structure_reference_dict_file)
                
                pickled_output_master_structure_ref_info_path = specific_output_pickle_data_dir.joinpath(output_master_structure_info_dict_for_export_name)
                with open(pickled_output_master_structure_ref_info_path, 'wb') as master_structure_info_dict_file:
                    pickle.dump(master_structure_info_dict, master_structure_info_dict_file)

                if plot_immediately_after_simulation == False:
                    sys.exit('> Programme exited.')
            
            

            elif (perform_MC_sim == False and perform_fanova == False):

                live_display.stop()
                live_display.console.print("[bold red]User input required:")
                results_file_ready = False
                while results_file_ready == False:
                    stopwatch.stop()
                    results_file_ready = ques_funcs.ask_ok('> You indicated to skip fanova and MC sim. Would you like to select the results dataset?') 
                    stopwatch.start()
                    if results_file_ready == True:
                        print('> Please indicate the location of master_structure_reference_dict_results.')
                        root = tk.Tk() # these two lines are to get rid of errant tkinter window
                        root.withdraw() # these two lines are to get rid of errant tkinter window
                        # this is a user defined quantity
                        results_master_structure_reference_dict_path_str = fd.askopenfilename(title='Open the master_structure_reference_dict_results file', initialdir=output_dir)
                        with open(results_master_structure_reference_dict_path_str, "rb") as preprocessed_master_structure_reference_dict_file:
                            master_structure_reference_dict = pickle.load(preprocessed_master_structure_reference_dict_file)
                        
                        print('> Please indicate the location of master_structure_info_dict_results.')
                        results_master_structure_reference_dict_path = pathlib.Path(results_master_structure_reference_dict_path_str)
                        results_master_structure_reference_dict_path_parent = results_master_structure_reference_dict_path.parents[0]
                        results_master_structure_info_dict_path_str = fd.askopenfilename(title='Open the master_structure_info_dict file', initialdir=results_master_structure_reference_dict_path_parent)
                        with open(results_master_structure_info_dict_path_str, "rb") as results_master_structure_info_dict_file:
                            master_structure_info_dict = pickle.load(results_master_structure_info_dict_file)

                        live_display.start()
                        important_info.add_text_line("Loaded master_structure_reference_dict_results from: "+ results_master_structure_reference_dict_path_str, live_display)
                        important_info.add_text_line("Loaded master_structure_info_dict_results from: "+ results_master_structure_info_dict_path_str, live_display)
                    
                    else:
                        print('> If you dont, no results will be analysed. Please run the algorithm without skipping the MC simulation or fanova simulation, in order to produce a results dataset.')
                        stopwatch.stop()
                        ask_to_continue = ques_funcs.ask_ok('> Would you like to continue without results anyways?')
                        stopwatch.start()
                        if ask_to_continue == True:
                            live_display.start()
                            break
                        else:
                            pass
                                
    

            rich_preambles.section_completed("Simulations", section_start_time, completed_progress, completed_sections_manager)    
            

            section_start_time = datetime.now() 
            
            # BEGIN SECTION TO DO AFTER READING MASTER STRUCTURE AND INFO FILES


            preprocessing_complete_bool = master_structure_info_dict["Global"]["Preprocessing info"]["Preprocessing performed"]
            mc_sim_complete_bool = master_structure_info_dict['Global']["MC info"]['MC sim performed']
            mc_containment_sim_complete_bool = master_structure_info_dict['Global']["MC info"]['MC containment sim performed']
            mc_dose_sim_complete_bool = master_structure_info_dict['Global']["MC info"]['MC dose sim performed']
            mc_mr_sim_complete = master_structure_info_dict['Global']["MC info"]['MC MR sim performed']

            fanova_sim_complete_bool = master_structure_info_dict['Global']["FANOVA info"]['FANOVA sim performed']
            fanova_containment_sim_complete_bool = master_structure_info_dict['Global']["FANOVA info"]['FANOVA containment sim performed']
            fanova_dose_sim_complete_bool = master_structure_info_dict['Global']["FANOVA info"]['FANOVA dose sim performed']

            num_patients = master_structure_info_dict["Global"]["Num cases"]
            interp_inter_slice_dist = master_structure_info_dict["Global"]["Preprocessing info"]["Interslice interp dist"]
            interp_intra_slice_dist = master_structure_info_dict["Global"]["Preprocessing info"]["Intraslice interp dist"]

            specific_output_dir = master_structure_info_dict["Global"]["Specific output dir"]

            #live_display.stop()

            # CREATE DATAFRAMES ---------------------------

            # FOR CSVs
            if preprocessing_complete_bool == True:
                csv_dataframe_building_indeterminate = indeterminate_progress_main.add_task('[red]Generating dataframes (preprocessing)...', total=None)
                csv_dataframe_building_indeterminate_completed = completed_progress.add_task('[green]Generating dataframes (preprocessing)', total=1, visible = False)

                # nearest dils to biopsies dataframe builder
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~DF 1", total = None)
                cohort_nearest_dils_dataframe = dataframe_builders.bx_nearest_dils_dataframe_builder(master_structure_reference_dict,
                                       structs_referenced_list,
                                       all_ref_key,
                                       bx_ref
                                       )
                master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: Nearest DILs to each biopsy"] = cohort_nearest_dils_dataframe
                indeterminate_progress_sub.update(indeterminate_task, visible = False)

                # structure volume dataframe builder (note that this function references the dataframe produced by bx_nearest_dils_dataframe_builder func)
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~DF 2", total = None)
                cohort_biopsy_basic_spatial_features_dataframe = dataframe_builders.biopsy_basic_spatial_features_information_dataframe_builder(master_structure_reference_dict,
                                       all_ref_key,
                                       bx_ref)
                master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: Biopsy basic spatial features dataframe"] = cohort_biopsy_basic_spatial_features_dataframe
                indeterminate_progress_sub.update(indeterminate_task, visible = False)

                
                # structure dimension dataframe builder (useless)
                """
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~DF 2", total = None)
                dataframe_builders.structure_dimension_dataframe_builder(master_structure_reference_dict,
                                       structs_referenced_list,
                                       all_ref_key,
                                       bx_ref)
                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                """

                

                
                # results of the biopsy optimization algorithm
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~DF 3", total = None)
                dataframe_builders.dil_optimization_results_dataframe_builder(master_structure_reference_dict,
                                       all_ref_key,
                                       dil_ref
                                       )
                indeterminate_progress_sub.update(indeterminate_task, visible = False)

                # structure radiomic 3D segmentation features dataframe
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~DF 4", total = None)
                structure_cohort_3d_radiomic_features_dataframe = dataframe_builders.cohort_structure_features_dataframe_builder(master_structure_reference_dict,
                                                structs_referenced_list,
                                                bx_ref)
                master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: 3D radiomic features all OAR and DIL structures"] = structure_cohort_3d_radiomic_features_dataframe
                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                      
                
                indeterminate_progress_main.update(csv_dataframe_building_indeterminate, visible = False)
                completed_progress.update(csv_dataframe_building_indeterminate_completed, advance = 1,visible = True)
                live_display.refresh()
                
            #live_display.stop()
            if mc_sim_complete_bool == True:
                
                ### MC General
                general_mc_sims_dataframe_building_indeterminate = indeterminate_progress_main.add_task('[red]Generating dataframes (MC, general)...', total=None)
                general_mc_sims_dataframe_building_indeterminate_completed = completed_progress.add_task('[green]Generating dataframes (MC, general)', total=1, visible = False)
       
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~DF 1", total = None)
                cohort_all_structure_shifts_pandas_data_frame = dataframe_builders.all_structure_shift_vectors_dataframe_builder(master_structure_reference_dict,
                                  structs_referenced_list, 
                                  bx_ref, 
                                  max_simulations,
                                  all_ref_key,
                                  important_info,
                                  live_display)

                master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: All MC structure shift vectors"] = cohort_all_structure_shifts_pandas_data_frame
                indeterminate_progress_sub.update(indeterminate_task, visible = False)

                indeterminate_progress_main.update(general_mc_sims_dataframe_building_indeterminate, visible = False)
                completed_progress.update(general_mc_sims_dataframe_building_indeterminate_completed, advance = 1,visible = True)
                live_display.refresh()



                ### MC Tissue
                csv_dataframe_building_indeterminate = indeterminate_progress_main.add_task('[red]Generating dataframes (MC, tissue)...', total=None)
                csv_dataframe_building_indeterminate_completed = completed_progress.add_task('[green]Generating dataframes (MC, tissue)', total=1, visible = False)
                
                # Concaetante the tissue class and structure specific results
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~DF 1", total = None)
                cohort_mc_structure_specific_pt_wise_results_dataframe = dataframe_builders.cohort_and_multi_biopsy_mc_structure_specific_pt_wise_results_dataframe_builder(master_structure_reference_dict,
                                                                                    bx_ref,
                                                                                    all_ref_key)
                master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: structure specific mc results"] = cohort_mc_structure_specific_pt_wise_results_dataframe
                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                

                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~DF 2", total = None)
                cohort_mc_sum_to_one_pt_wise_results_dataframe = dataframe_builders.cohort_and_multi_biopsy_mc_sum_to_one_pt_wise_results_dataframe_builder(master_structure_reference_dict,
                                                                                    bx_ref,
                                                                                    all_ref_key)
                master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: sum-to-one mc results"] = cohort_mc_sum_to_one_pt_wise_results_dataframe
                indeterminate_progress_sub.update(indeterminate_task, visible = False)


                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~DF 2.1", total = None)
                cohort_mc_sum_to_one_global_results_dataframe = dataframe_builders.cohort_mc_sum_to_one_global_scores_dataframe_builder(cohort_mc_sum_to_one_pt_wise_results_dataframe)
                master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: global sum-to-one mc results"] = cohort_mc_sum_to_one_global_results_dataframe
                indeterminate_progress_sub.update(indeterminate_task, visible = False)

                
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~DF 3", total = None)
                cohort_mc_tissue_class_pt_wise_results_dataframe = dataframe_builders.cohort_and_multi_biopsy_mc_tissue_class_pt_wise_results_dataframe_builder(master_structure_reference_dict,
                                                                                    bx_ref,
                                                                                    all_ref_key)
                master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: mutual tissue class mc results"] = cohort_mc_tissue_class_pt_wise_results_dataframe
                indeterminate_progress_sub.update(indeterminate_task, visible = False)


                # create global tissue class scores dataframes for each biopsy
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~DF 4", total = None)
                cohort_global_tissue_class_by_tissue_type_dataframe = dataframe_builders.global_scores_by_tissue_class_dataframe_builder(master_structure_reference_dict,
                                                                                    bx_ref,
                                                                                    all_ref_key)
                master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: tissue class global scores (tissue type)"] = cohort_global_tissue_class_by_tissue_type_dataframe
                indeterminate_progress_sub.update(indeterminate_task, visible = False)

                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~DF 5", total = None)
                cohort_global_tissue_class_by_structure_dataframe = dataframe_builders.global_scores_by_specific_structure_dataframe_builder(master_structure_reference_dict,
                                                    bx_ref,
                                                    all_ref_key)
                master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: tissue class global scores (structure)"] = cohort_global_tissue_class_by_structure_dataframe
                indeterminate_progress_sub.update(indeterminate_task, visible = False)

                # create dataframe for tissue volume above threshold
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~DF 6", total = None)
                cohort_tissue_volume_above_threshold_dataframe = dataframe_builders.tissue_volume_threshold_dataframe_builder_NEW(master_structure_reference_dict,
                                                                                    bx_ref)
                master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: tissue volume above threshold"] = cohort_tissue_volume_above_threshold_dataframe
                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                
                # All binom est info for cohort
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~DF 7", total = None) 
                cohort_all_binom_est_data_by_pt_and_voxel = dataframe_builders.cohort_creator_binom_est_by_pt_and_voxel_dataframe(master_structure_reference_dict,
                                                       bx_ref)
                
                master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: Entire point-wise binom est distribution"] = cohort_all_binom_est_data_by_pt_and_voxel
                indeterminate_progress_sub.update(indeterminate_task, visible = False)

                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~DF 8", total = None) 
                cohort_global_tissue_scores_with_target_dil_radiomic_features_df = dataframe_builders.bx_global_score_to_target_dil_3d_radiomic_features_dataframe_builder(structure_cohort_3d_radiomic_features_dataframe,
                                                                         cohort_global_tissue_class_by_structure_dataframe,
                                                                         master_structure_reference_dict,
                                                                         bx_ref)
                master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: DIL global tissue scores and DIL features"] = cohort_global_tissue_scores_with_target_dil_radiomic_features_df
                indeterminate_progress_sub.update(indeterminate_task, visible = False)

                indeterminate_progress_main.update(csv_dataframe_building_indeterminate, visible = False)
                completed_progress.update(csv_dataframe_building_indeterminate_completed, advance = 1,visible = True)
                live_display.refresh()


                
                ####### MACHINE LEARNING RANDOM FOREST ON TUMOR MORPHOLOGY AND TISSUE CLASS SCORE
                #live_display.stop()
                if num_patients > 1:
                    csv_dataframe_building_indeterminate = indeterminate_progress_main.add_task('[red]Random forest training (MC, tissue)...', total=None)
                    csv_dataframe_building_indeterminate_completed = completed_progress.add_task('[green]Random forest training (MC, tissue)', total=1, visible = False)
                    
                    tumor_morphology_rf_metrics_df, tumor_morphology_rf_regressor, tumor_morphology_rf_results, tumor_morphology_feature_columns = machina_learning.random_forest_global_tissue_class_score_predicted_by_tumor_morphology(cohort_global_tissue_scores_with_target_dil_radiomic_features_df)

                    master_cohort_patient_data_and_dataframes["Data"]["Cohort: Random forest tumor morphology results"] = {"Metrics":tumor_morphology_rf_metrics_df, 
                                                                                                                            "Regressor": tumor_morphology_rf_regressor,
                                                                                                                            "Results": tumor_morphology_rf_results,
                                                                                                                            "Feature columns": tumor_morphology_feature_columns}
                
                    indeterminate_progress_main.update(csv_dataframe_building_indeterminate, visible = False)
                    completed_progress.update(csv_dataframe_building_indeterminate_completed, advance = 1,visible = True)
                else:
                    pass

                
                #print('test')
                #live_display.start()

                
                
                
                
                csv_dataframe_building_indeterminate = indeterminate_progress_main.add_task('[red]Generating dataframes (MC, dosimetry)...', total=None)
                csv_dataframe_building_indeterminate_completed = completed_progress.add_task('[green]Generating dataframes (MC, dosimetry)', total=1, visible = False)

                # More primitive point-dose dataframe
                # I AM ATTEMPTING TO REMOVE THIS LINE NOW!
                #dataframe_builders.pointwise_mean_dose_and_standard_deviation_dataframe_builder(master_structure_reference_dict, bx_ref)

                ### DO NOT USE DOSE_OUTPUT_VOXELIED_DATAFRAME_BUILDER! RELIES ON DEPRECATED CODE!
                # Voxelized dose dataframe
                #dataframe_builders.dose_output_voxelized_dataframe_builder(master_structure_reference_dict, bx_ref)

                # create grand dose data dataframe for each biopsy by MC trial and bx pt
                """
                cohort_all_dose_data_by_trial_and_pt = dataframe_builders.all_dose_data_by_trial_and_pt_from_dataframe_builder_and_voxelizer_NEW(master_structure_reference_dict,
                                                                  bx_ref,
                                                                  biopsy_z_voxel_length
                                                                  )
                master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: Entire point-wise dose distribution"] = cohort_all_dose_data_by_trial_and_pt                
                """
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~DF 1", total = None)
                live_display.stop()

                st = time.time()
                dataframe_builders.all_dose_data_by_trial_and_pt_from_dataframe_builder_and_voxelizer_v4(master_structure_reference_dict,
                                                                  bx_ref,
                                                                  biopsy_z_voxel_length,
                                                                  dose_ref
                                                                  )
                et = time.time()
                duration = et-st
                print(f"Dose DF1: {duration}")
                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                



                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~DF 2", total = None) 
                """
                cohort_global_dosimetry_by_voxel_dataframe = dataframe_builders.global_dosimetry_by_voxel_values_dataframe_builder(master_structure_reference_dict,
                                                    bx_ref,
                                                    all_ref_key,
                                                    dose_ref)
                """
                
                # for some reason it seems like the livedisplay can cause these functions to hang?
                """
                st = time.time()
                # I made the below to try to make the execution quicker, however it turned out to not take almost exactly the same amount of time
                cohort_global_dosimetry_by_voxel_dataframe = dataframe_builders.global_dosimetry_by_voxel_values_dataframe_builder_ALTERNATE(master_structure_reference_dict,
                                                    bx_ref,
                                                    all_ref_key,
                                                    dose_ref)
                et = time.time()
                duration = et-st
                print(f"Dose DF2: {duration}")
                
                master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: Global dosimetry by voxel"] = cohort_global_dosimetry_by_voxel_dataframe
                """

                """
                The below function replaced the above to account for gradients, and actually is generalizable to more information. 
                The list of dose_value_columns can be generalized! Also the output is now a multiindex dataframe.
                """
                st = time.time()
                dose_value_columns = ['Dose (Gy)', 'Dose grad (Gy/mm)']
                cohort_global_dosimetry_by_voxel_dataframe = dataframe_builders.global_dosimetry_by_voxel_values_dataframe_builder_v3_generalized(master_structure_reference_dict, 
                                                                            bx_ref, 
                                                                            all_ref_key, 
                                                                            dose_ref, 
                                                                            dose_value_columns)

                master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: Global dosimetry by voxel"] = cohort_global_dosimetry_by_voxel_dataframe
                et = time.time()
                duration = et-st
                print(f"Dose DF2: {duration}")

                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                
                st = time.time()
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~DF 3", total = None) 
                cohort_global_dosimetry_dataframe = dataframe_builders.global_dosimetry_values_dataframe_builder(master_structure_reference_dict,
                                                    bx_ref,
                                                    all_ref_key,
                                                    dose_ref)
                et = time.time()
                duration = et-st
                print(f"Dose DF3: {duration}")
                
                master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: Global dosimetry"] = cohort_global_dosimetry_dataframe
                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                

                st = time.time()
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~DF 3.1", total = None) 
                cohort_global_dosimetry_dataframe = dataframe_builders.global_dosimetry_by_biopsy_dataframe_builder_NEW_multiindex_df(master_structure_reference_dict,
                                                    bx_ref,
                                                    all_ref_key,
                                                    dose_ref,
                                                    dose_value_columns)
                et = time.time()
                duration = et-st
                print(f"Dose DF3: {duration}")
                
                master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: Global dosimetry (NEW)"] = cohort_global_dosimetry_dataframe
                indeterminate_progress_sub.update(indeterminate_task, visible = False)

                
                
                st = time.time()
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~DF 4", total = None)
                dataframe_builders.differential_dvh_dataframe_all_mc_trials_dataframe_builder_v2(master_structure_reference_dict,
                                                                                            master_structure_info_dict,
                                                                                            bx_ref,
                                                                                            dose_ref)
                et = time.time()
                duration = et-st
                print(f"Dose DF4: {duration}")
                indeterminate_progress_sub.update(indeterminate_task, visible = False)

                st = time.time()
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~DF 5", total = None)
                dataframe_builders.cumulative_dvh_dataframe_all_mc_trials_dataframe_builder_v2(master_structure_reference_dict,
                                                            master_structure_info_dict,
                                                            bx_ref,
                                                            dose_ref)
                et = time.time()
                duration = et-st
                print(f"Dose DF5: {duration}")
                indeterminate_progress_sub.update(indeterminate_task, visible = False)



                st = time.time()
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~DF 6", total = None)
                cohort_all_bx_dvh_metric_dataframe = dataframe_builders.dvh_metrics_dataframe_builder_sp_biopsy(master_structure_reference_dict,
                                                                            bx_ref,
                                                                            all_ref_key,
                                                                            dose_ref)

                master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: Bx DVH metrics"] = cohort_all_bx_dvh_metric_dataframe

                et = time.time()
                duration = et-st
                print(f"Dose DF6: {duration}")
                indeterminate_progress_sub.update(indeterminate_task, visible = False)


                
                # DVH metrics new and improved!
                st = time.time()
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~DF 7", total = None)
                cohort_all_bx_dvh_metric_generalized_dataframe = dataframe_builders.dvh_metrics_calculator_and_dataframe_builder_cohort(master_structure_reference_dict,
                                            bx_ref,
                                            all_ref_key,
                                            dose_ref,
                                            plan_ref,
                                            d_x_DVH_to_calc_list,
                                            v_percent_DVH_to_calc_list,
                                            default_ctv_dose = 13.5)

                master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: Bx DVH metrics (generalized)"] = cohort_all_bx_dvh_metric_generalized_dataframe

                et = time.time()
                duration = et-st
                print(f"Dose DF7: {duration}")
                indeterminate_progress_sub.update(indeterminate_task, visible = False)


                ### THIS NEEDS WORK, THIS MAY BE A BAD IDEA, TOO MANY COLUMNS THAT MIGHT MATCH
                ### This has to be the last one because it depends on the creation of previous dataframes
                #bx_info_cohort_dataframe = dataframe_builders.bx_info_dataframe_builder(cohort_nearest_dils_dataframe,
                #            cohort_global_tissue_class_by_tissue_type_dataframe,
                #            cohort_all_bx_dvh_metric_dataframe)


                #master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: Bx global info dataframe"] = bx_info_cohort_dataframe
                ### THIS NEEDS WORK, THIS MAY BE A BAD IDEA, TOO MANY COLUMNS THAT MIGHT MATCH
                
                
                
                indeterminate_progress_main.update(csv_dataframe_building_indeterminate, visible = False)
                completed_progress.update(csv_dataframe_building_indeterminate_completed, advance = 1,visible = True)

                live_display.refresh()




                ####
                if mc_mr_sim_complete == True:
                    csv_dataframe_building_indeterminate = indeterminate_progress_main.add_task('[red]Generating dataframes (MC, MR)...', total=None)
                    csv_dataframe_building_indeterminate_completed = completed_progress.add_task('[green]Generating dataframes (MC, MR)', total=1, visible = False)

                    mc_sim_mr_adc_arr_str = "MC data: MR ADC vals for each sampled bx pt arr (nominal & all MC trials)"
                    output_mr_adc_dataframe_str = "Point-wise MR ADC output by MC trial number"
                    mr_adc_col_name_str_prefix = "MR ADC"

                    indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~DF 1", total = None)
                    dataframe_builders.all_mr_data_by_trial_and_pt_from_dataframe_builder_and_voxelizer_v4(master_structure_reference_dict, 
                                                                            bx_ref, 
                                                                            biopsy_z_voxel_length, 
                                                                            mr_adc_ref,
                                                                            mc_sim_mr_adc_arr_str,
                                                                            mr_adc_col_name_str_prefix,
                                                                            output_mr_adc_dataframe_str)
                    indeterminate_progress_sub.update(indeterminate_task, visible = False)

                    st = time.time()
                    indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~DF 2", total = None)
                    cohort_global_mr_dataframe = dataframe_builders.global_mr_values_dataframe_builder(master_structure_reference_dict,
                                                        bx_ref,
                                                        all_ref_key,
                                                        mr_adc_ref,
                                                        mr_adc_col_name_str_prefix,
                                                        output_mr_adc_dataframe_str,
                                                        mr_global_multi_structure_output_dataframe_str)
                    et = time.time()
                    duration = et-st
                    print(f"MR DF2: {duration}")

                    master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: " + mr_global_multi_structure_output_dataframe_str] = cohort_global_mr_dataframe
                    indeterminate_progress_sub.update(indeterminate_task, visible = False)

                    indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~DF 3", total = None)
                    """
                    cohort_global_by_voxel_mr_dataframe = dataframe_builders.global_mr_by_voxel_values_dataframe_builder(master_structure_reference_dict,
                                                        bx_ref,
                                                        all_ref_key,
                                                        mr_adc_ref,
                                                        mr_adc_col_name_str_prefix,
                                                        output_mr_adc_dataframe_str,
                                                        mr_global_by_voxel_multi_structure_output_dataframe_str)
                    """
                    # for some reason it seems that the livedisplay can cause this function to hang?
                    st = time.time()
                    cohort_global_by_voxel_mr_dataframe = dataframe_builders.global_mr_by_voxel_values_dataframe_builder_ALTERNATE(master_structure_reference_dict,
                                                            bx_ref,
                                                            all_ref_key,
                                                            mr_adc_ref,
                                                            mr_adc_col_name_str_prefix,
                                                            output_mr_adc_dataframe_str,
                                                            mr_global_by_voxel_multi_structure_output_dataframe_str)
                    et = time.time()
                    duration = et-st
                    print(f"MR DF3: {duration}")
                    master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: " + mr_global_by_voxel_multi_structure_output_dataframe_str] = cohort_global_by_voxel_mr_dataframe
                    indeterminate_progress_sub.update(indeterminate_task, visible = False)


                    indeterminate_progress_main.update(csv_dataframe_building_indeterminate, visible = False)
                    completed_progress.update(csv_dataframe_building_indeterminate_completed, advance = 1,visible = True)
                live_display.refresh()

                live_display.start()
                live_display.refresh()


                



            # CREATE CSV DIRECTORIES ---------------------------


            #live_display.stop()
            # create global csv output folder
            if any([write_preprocessing_data_to_file, write_containment_to_file_ans, write_dose_to_file_ans, write_sobol_containment_data_to_file, write_sobol_dose_data_to_file, write_cohort_data_to_file]):
                csv_output_folder_name = 'Output CSVs'
                csv_output_dir = specific_output_dir.joinpath(csv_output_folder_name)
                csv_output_dir.mkdir(parents=True, exist_ok=True)

            # create preprocessing csv output folder
            if write_preprocessing_data_to_file == True:
                preprocessing_output_folder_name = 'Preprocessing'
                preprocessing_csv_output_dir = csv_output_dir.joinpath(preprocessing_output_folder_name)
                preprocessing_csv_output_dir.mkdir(parents=True, exist_ok=True)

                # create patient specific output directories for csv files
                preprocessing_patient_sp_output_csv_dir_dict = {}
                for patientUID in master_structure_reference_dict.keys():
                    patient_sp_output_csv_dir = preprocessing_csv_output_dir.joinpath(patientUID)
                    patient_sp_output_csv_dir.mkdir(parents=True, exist_ok=True)
                    preprocessing_patient_sp_output_csv_dir_dict[patientUID] = patient_sp_output_csv_dir
                global_preprocessing_output_csv_dir = preprocessing_csv_output_dir.joinpath('Global')
                global_preprocessing_output_csv_dir.mkdir(parents=True, exist_ok=True)
                preprocessing_patient_sp_output_csv_dir_dict["Global"] = global_preprocessing_output_csv_dir
            
            # create mc csv output folder
            if any([write_containment_to_file_ans, write_dose_to_file_ans]):      
                mc_output_folder_name = 'MC simulation'
                mc_csv_output_dir = csv_output_dir.joinpath(mc_output_folder_name)
                mc_csv_output_dir.mkdir(parents=True, exist_ok=True)

                # create patient specific output directories for csv files
                patient_sp_output_csv_dir_dict = {}
                patient_sp_bx_sp_output_csv_dir_dict = {}
                for patientUID in master_structure_reference_dict.keys():
                    patient_sp_output_csv_dir = mc_csv_output_dir.joinpath(patientUID)
                    patient_sp_output_csv_dir.mkdir(parents=True, exist_ok=True)
                    patient_sp_output_csv_dir_dict[patientUID] = patient_sp_output_csv_dir
                    bx_sp_output_csv_dir_dict = {}
                    for sp_bx in master_structure_reference_dict[patientUID][bx_ref]:
                        sp_bx_name = sp_bx["ROI"]
                        bx_index_num = sp_bx["Index number"]
                        sp_bx_dir_str = str(bx_index_num) +"-"+ sp_bx_name
                        
                        bx_sp_output_csv_dir = patient_sp_output_csv_dir.joinpath(sp_bx_dir_str)
                        bx_sp_output_csv_dir.mkdir(parents=True, exist_ok=True)

                        bx_sp_output_csv_dir_dict[bx_index_num] = bx_sp_output_csv_dir
                    patient_sp_bx_sp_output_csv_dir_dict[patientUID] = bx_sp_output_csv_dir_dict

                global_mc_output_csv_dir = mc_csv_output_dir.joinpath('Global')
                global_mc_output_csv_dir.mkdir(parents=True, exist_ok=True)
                patient_sp_output_csv_dir_dict["Global"] = global_mc_output_csv_dir

            # create fanova csv output folder
            if any([write_sobol_containment_data_to_file, write_sobol_dose_data_to_file]):  
                fanova_output_folder_name = 'FANOVA simulation'
                fanova_csv_output_dir = csv_output_dir.joinpath(fanova_output_folder_name)
                fanova_csv_output_dir.mkdir(parents=True, exist_ok=True)

                fanova_patient_sp_output_csv_dir_dict = {}
                for patientUID in master_structure_reference_dict.keys():
                    patient_sp_output_csv_dir = fanova_csv_output_dir.joinpath(patientUID)
                    patient_sp_output_csv_dir.mkdir(parents=True, exist_ok=True)
                    fanova_patient_sp_output_csv_dir_dict[patientUID] = patient_sp_output_csv_dir
                global_fanova_output_csv_dir = fanova_csv_output_dir.joinpath('Global')
                global_fanova_output_csv_dir.mkdir(parents=True, exist_ok=True)
                fanova_patient_sp_output_csv_dir_dict["Global"] = global_fanova_output_csv_dir


            # create cohort csv folder
            if write_cohort_data_to_file == True:  
                cohort_output_folder_name = 'Cohort'
                cohort_csv_output_dir = csv_output_dir.joinpath(cohort_output_folder_name)
                cohort_csv_output_dir.mkdir(parents=True, exist_ok=True)


            # CREATE CSVs -------------------------------
                
            # Preprocessing   
            if write_preprocessing_data_to_file == True and preprocessing_complete_bool == True:
                important_info.add_text_line("Writing preprocessing CSVs to file.", live_display)

                #dict_of_patient_specific_dataframes = {}
                for patientUID,pydicom_item in master_structure_reference_dict.items():
                    patient_sp_csv_dir = preprocessing_patient_sp_output_csv_dir_dict[patientUID]
                    
                    for dataframe_name, dataframe in pydicom_item[all_ref_key]["Multi-structure pre-processing output dataframes dict"].items():
                        if isinstance(dataframe, pandas.DataFrame):

                            dataframe_file_name = str(patientUID) +'-'+ str(dataframe_name)+ '.csv'
                            dataframe_file_path = patient_sp_csv_dir.joinpath(dataframe_file_name)
                            dataframe.to_csv(dataframe_file_path)

                        # also append to create global dataframe
                #         if dataframe_name in dict_of_patient_specific_dataframes:
                #             dict_of_patient_specific_dataframes[dataframe_name].append(dataframe)
                #         else:
                #             dict_of_patient_specific_dataframes[dataframe_name] = [dataframe]
                
                # global_preprocessing_output_csv_dir = preprocessing_patient_sp_output_csv_dir_dict["Global"]
                # for dataframe_name, dataframe_list in dict_of_patient_specific_dataframes.items():
                #     global_df = pandas.concat(dataframe_list)
                #     dataframe_file_name = 'Global' +'-'+ str(dataframe_name)+ '.csv'
                #     dataframe_file_path = global_preprocessing_output_csv_dir.joinpath(dataframe_file_name)
                #     global_df.to_csv(dataframe_file_path)



            # MC containment 
            if write_containment_to_file_ans == True and mc_sim_complete_bool == True:
                important_info.add_text_line("Writing containment CSVs to file.", live_display)
                """
                csv_writers.csv_writer_containment(live_display,
                        layout_groups,
                        master_structure_reference_dict,
                        master_structure_info_dict,
                        patient_sp_output_csv_dir_dict,
                        bx_ref,
                        cancer_tissue_label,
                        structure_miss_probability_roi,
                        miss_structure_complement_label
                        )
                """
                live_display.start(refresh=True)
                
            else:
                pass  


            # MC dose
            if write_dose_to_file_ans == True and mc_sim_complete_bool == True:
                important_info.add_text_line("Writing dosimetry CSVs to file.", live_display)
                """
                csv_writers.csv_writer_dosimetry(live_display,
                        layout_groups,
                        master_structure_reference_dict,
                        master_structure_info_dict,
                        patient_sp_output_csv_dir_dict,
                        bx_ref,
                        display_dvh_as,
                        v_percent_DVH_to_calc_list
                        )
                """
                live_display.start(refresh=True)
                
            else:
                pass
            
            # Write the rest of the dataframes that we've stored
            """
            if mc_csv_output_dir.is_dir():
                important_info.add_text_line("Writing remainder of stored dataframes to file.", live_display)
                csv_dataframe_building_indeterminate = indeterminate_progress_main.add_task('[red]Generating dataframes (MC, dosimetry)...', total=None)
                csv_dataframe_building_indeterminate_completed = completed_progress.add_task('[green]Generating dataframes (MC, dosimetry)', total=1, visible = False)

                for patientUID in master_structure_reference_dict.keys():
                    patient_sp_csv_dir = patient_sp_output_csv_dir_dict[patientUID]
                    for specific_bx_structure in pydicom_item[bx_ref]:
                        bx_roi = specific_bx_structure["ROI"]
                        for dataframe_name, dataframe in specific_bx_structure["Output data frames"].items():
                            dataframe_file_name = str(bx_roi) +'-'+ str(dataframe_name)+ '.csv'
                            dataframe_file_path = patient_sp_csv_dir.joinpath(dataframe_file_name)
                            dataframe.to_csv(dataframe_file_path)

                indeterminate_progress_main.update(csv_dataframe_building_indeterminate, visible = False)
                completed_progress.update(csv_dataframe_building_indeterminate_completed, advance = 1,visible = True)
                live_display.refresh()
            """
            if mc_csv_output_dir.is_dir():
                important_info.add_text_line("Writing MC sim stored dataframes to file.", live_display)
                csv_dataframe_building_indeterminate = indeterminate_progress_main.add_task('[red]Writing MC sim stored dataframes to file...', total=None)
                csv_dataframe_building_indeterminate_completed = completed_progress.add_task('[green]Writing MC sim stored dataframes to file', total=1, visible = False)

                for patientUID, pydicom_item in master_structure_reference_dict.items():
                    patient_sp_csv_dir = patient_sp_output_csv_dir_dict[patientUID]
                    for dataframe_name, dataframe in pydicom_item[all_ref_key]['Multi-structure MC simulation output dataframes dict'].items():
                        if isinstance(dataframe, pandas.DataFrame):
                            dataframe_file_name = str(patientUID) +'-'+ str(dataframe_name)+ '.csv'
                            dataframe_file_path = patient_sp_csv_dir.joinpath(dataframe_file_name)
                            dataframe.to_csv(dataframe_file_path)

                    for sp_bx in pydicom_item[bx_ref]:
                        bx_name = sp_bx["ROI"]
                        bx_type = sp_bx["Simulated type"]
                        bx_index_number = sp_bx["Index number"] 
                        bx_sp_csv_dir = patient_sp_bx_sp_output_csv_dir_dict[patientUID][bx_index_number]
                        for dataframe_name, dataframe in sp_bx['Output data frames'].items():
                            if isinstance(dataframe, pandas.DataFrame): 
                                dataframe_file_name = str(bx_type) +'-'+ str(bx_name)+'-'+ str(dataframe_name)+ '.csv'
                                dataframe_file_path = bx_sp_csv_dir.joinpath(dataframe_file_name)
                                dataframe.to_csv(dataframe_file_path)

                indeterminate_progress_main.update(csv_dataframe_building_indeterminate, visible = False)
                completed_progress.update(csv_dataframe_building_indeterminate_completed, advance = 1,visible = True)
                live_display.refresh()
                

            # fanova containment
            if write_sobol_containment_data_to_file == True and fanova_containment_sim_complete_bool == True:
                important_info.add_text_line("Writing fanova containment CSVs to file.", live_display)

                global_fanova_output_csv_dir = fanova_patient_sp_output_csv_dir_dict["Global"]

                # DIL tissue Sobol dataframe
                fanova_csv_dataframe_filename = 'sobol_dataframe_DIL_tissue_for_all_patients.csv'
                sobol_dil_tissue_dataframe_filepath_all_patients = global_fanova_output_csv_dir.joinpath(fanova_csv_dataframe_filename)

                dataframes_list = []
                for patientUID,pydicom_item in master_structure_reference_dict.items():  
                    for specific_bx_structure in pydicom_item[bx_ref]:
                        sp_bx_sobol_containment_dataframe = specific_bx_structure["FANOVA: sobol containment (DIL tissue) dataframe"]
                        dataframes_list.append(sp_bx_sobol_containment_dataframe)

                grand_sobol_dataframe = pandas.concat(dataframes_list, ignore_index=True) 
                grand_sobol_dataframe.to_csv(sobol_dil_tissue_dataframe_filepath_all_patients)
                del grand_sobol_dataframe


                # All tissues Sobol dataframe
                fanova_csv_dataframe_filename = 'sobol_dataframe_by_tissue_for_all_patients.csv'
                sobol_tissue_dataframe_filepath_all_patients = global_fanova_output_csv_dir.joinpath(fanova_csv_dataframe_filename)

                dataframes_list = []
                for patientUID,pydicom_item in master_structure_reference_dict.items():  
                    for specific_bx_structure in pydicom_item[bx_ref]:
                        sp_bx_sobol_containment_dataframe = specific_bx_structure["FANOVA: sobol containment dataframe"]
                        dataframes_list.append(sp_bx_sobol_containment_dataframe)

                grand_sobol_dataframe = pandas.concat(dataframes_list, ignore_index=True) 
                grand_sobol_dataframe.to_csv(sobol_tissue_dataframe_filepath_all_patients)
                del grand_sobol_dataframe



            else:
                pass
            
            # fanova dose
            if write_sobol_dose_data_to_file ==  True and fanova_dose_sim_complete_bool == True:
                important_info.add_text_line("Writing fanova dosimetry CSVs to file.", live_display)

                global_fanova_output_csv_dir = fanova_patient_sp_output_csv_dir_dict["Global"]

                # dose Sobol dataframe
                fanova_csv_dataframe_filename = 'sobol_dataframe_dose_for_all_patients.csv'
                sobol_dose_dataframe_filepath_all_patients = global_fanova_output_csv_dir.joinpath(fanova_csv_dataframe_filename)

                dataframes_list = []
                for patientUID,pydicom_item in master_structure_reference_dict.items():  
                    for specific_bx_structure in pydicom_item[bx_ref]:
                        sp_bx_sobol_dose_dataframe = specific_bx_structure["FANOVA: sobol dose dataframe"]
                        dataframes_list.append(sp_bx_sobol_dose_dataframe)

                grand_sobol_dataframe = pandas.concat(dataframes_list, ignore_index=True) 
                grand_sobol_dataframe.to_csv(sobol_dose_dataframe_filepath_all_patients)
                del grand_sobol_dataframe

            else:
                pass


            # cohort 
            if write_cohort_data_to_file == True:
                important_info.add_text_line("Writing cohort CSVs to file.", live_display)

                for dataframe_name, dataframe in master_cohort_patient_data_and_dataframes['Dataframes'].items():
                    if isinstance(dataframe, pandas.DataFrame):

                        dataframe_file_name = str(dataframe_name)+ '.csv'
                        dataframe_file_path = cohort_csv_output_dir.joinpath(dataframe_file_name)
                        dataframe.to_csv(dataframe_file_path)

    
            
            # CREATE PRODUCTION PLOT DIRECTORIES ----------------------------------------

            if create_at_least_one_production_plot == True:
            
                # make output figures directory
                figures_output_dir_name = 'Output figures'
                output_figures_dir = specific_output_dir.joinpath(figures_output_dir_name)
                output_figures_dir.mkdir(parents=True, exist_ok=True)

                # PREPROCESSING
                preprocessing_output_folder_name = 'Preprocessing'
                preprocessing_fig_output_dir = output_figures_dir.joinpath(preprocessing_output_folder_name)
                preprocessing_fig_output_dir.mkdir(parents=True, exist_ok=True)

                # generate and store patient directory folders for saving
                patient_sp_preprocessing_output_figures_dir_dict = {}
                for patientUID in master_structure_reference_dict.keys():
                    patient_sp_preprocessing_output_figures_dir = preprocessing_fig_output_dir.joinpath(patientUID)
                    patient_sp_preprocessing_output_figures_dir.mkdir(parents=True, exist_ok=True)
                    patient_sp_preprocessing_output_figures_dir_dict[patientUID] = patient_sp_preprocessing_output_figures_dir

                # create a global folder
                patient_sp_preprocessing_output_figures_dir = preprocessing_fig_output_dir.joinpath('Global')
                patient_sp_preprocessing_output_figures_dir.mkdir(parents=True, exist_ok=True)
                patient_sp_preprocessing_output_figures_dir_dict["Global"] = patient_sp_preprocessing_output_figures_dir

                master_structure_info_dict["Global"]['Patient specific output figures directory (pre-processing) dict'] = patient_sp_preprocessing_output_figures_dir_dict

                # MC SPECIFIC
                mc_output_folder_name = 'MC simulation'
                mc_fig_output_dir = output_figures_dir.joinpath(mc_output_folder_name)
                mc_fig_output_dir.mkdir(parents=True, exist_ok=True)

                # generate and store patient directory folders for saving
                patient_sp_output_figures_dir_dict = {}
                for patientUID in master_structure_reference_dict.keys():
                    patient_sp_output_figures_dir = mc_fig_output_dir.joinpath(patientUID)
                    patient_sp_output_figures_dir.mkdir(parents=True, exist_ok=True)
                    patient_sp_output_figures_dir_dict[patientUID] = patient_sp_output_figures_dir

                # create a global folder
                patient_sp_output_figures_dir = mc_fig_output_dir.joinpath('Global')
                patient_sp_output_figures_dir.mkdir(parents=True, exist_ok=True)
                patient_sp_output_figures_dir_dict["Global"] = patient_sp_output_figures_dir

                master_structure_info_dict["Global"]['Patient specific output figures directory (MC sim) dict'] = patient_sp_output_figures_dir_dict

                # FANOVA SPECIFIC
                fanova_output_folder_name = 'FANOVA simulation'
                fanova_fig_output_dir = output_figures_dir.joinpath(fanova_output_folder_name)
                fanova_csv_output_dir.mkdir(parents=True, exist_ok=True)

                # generate and store patient directory folders for saving
                patient_sp_fanova_output_figures_dir_dict = {}
                for patientUID in master_structure_reference_dict.keys():
                    patient_sp_output_figures_dir = fanova_fig_output_dir.joinpath(patientUID)
                    patient_sp_output_figures_dir.mkdir(parents=True, exist_ok=True)
                    patient_sp_fanova_output_figures_dir_dict[patientUID] = patient_sp_output_figures_dir

                # create a global folder
                patient_sp_output_figures_dir = fanova_fig_output_dir.joinpath('Global')
                patient_sp_output_figures_dir.mkdir(parents=True, exist_ok=True)
                patient_sp_fanova_output_figures_dir_dict["Global"] = patient_sp_output_figures_dir

                master_structure_info_dict["Global"]['Patient specific output figures directory (fanova) dict'] = patient_sp_fanova_output_figures_dir_dict



                # COHORT figures

                # Make cohort output directories
                cohort_figures_output_dir_name = 'Cohort figures'
                cohort_output_figures_dir = output_figures_dir.joinpath(cohort_figures_output_dir_name)
                cohort_output_figures_dir.mkdir(parents=False, exist_ok=True)
                
                master_structure_info_dict["Global"]['Cohort figures dir'] = cohort_output_figures_dir





            rich_preambles.section_completed("Dataframes and directories", section_start_time, completed_progress, completed_sections_manager)    

            section_start_time = datetime.now() 

            # CREATE PRODUCTION PLOTS ---------------------------------------------


            live_display.stop()

            ### PREPROCESSING FIGS AND OPTIMIZATION
            if create_at_least_one_production_plot == True and preprocessing_complete_bool == True:
                patient_sp_preprocessing_output_figures_dir_dict = master_structure_info_dict["Global"]['Patient specific output figures directory (pre-processing) dict']

                if production_plots_input_dictionary["Guidance maps"]["Plot bool"] == True:

                    general_plot_name_string = production_plots_input_dictionary["Guidance maps"]["Plot name"]

                    patientUID_default = "Initializing"
                    processing_patient_production_plot_description = "Creating guidance maps [{}]...".format(patientUID_default)
                    processing_patients_task = patients_progress.add_task("[red]"+processing_patient_production_plot_description, total = master_structure_info_dict["Global"]["Num cases"])
                    processing_patient_production_plot_description_completed = "Creating guidance maps"
                    processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_production_plot_description_completed, total=master_structure_info_dict["Global"]["Num cases"], visible=False)


                    for patientUID,pydicom_item in master_structure_reference_dict.items():
                        
                        processing_patient_production_plot_description = "Creating guidance maps [{}]...".format(patientUID)
                        patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_production_plot_description)

                        patient_sp_preprocessing_output_figures_dir = patient_sp_preprocessing_output_figures_dir_dict[patientUID]
                        
                        # Cumulative projection guidance map
                        production_plots.production_plot_guidance_maps_cumulative_projection(patientUID,
                                                patient_sp_preprocessing_output_figures_dir,
                                                pydicom_item,
                                                all_ref_key,
                                                svg_image_scale,
                                                svg_image_width,
                                                svg_image_height,
                                                general_plot_name_string
                                                )
                        # Max planes guidance maps
                        production_plots.production_plot_guidance_maps_max_planes(patientUID,
                                            patient_sp_preprocessing_output_figures_dir,
                                            pydicom_item,
                                            dil_ref,
                                            oar_ref,
                                            all_ref_key,
                                            svg_image_scale,
                                            svg_image_width,
                                            svg_image_height,
                                            general_plot_name_string
                                            )
                        """
                        production_plots.guidance_map_transducer_angle_sagittal(patientUID,
                                            patient_sp_preprocessing_output_figures_dir,
                                            pydicom_item,
                                            dil_ref,
                                            oar_ref,
                                            rectum_ref_key,
                                            all_ref_key,
                                            structs_referenced_dict,
                                            plot_guidance_map_transducer_plane_open3d_structure_set_complete_demonstration_bool,
                                            biopsy_fire_travel_distances,
                                            biopsy_needle_compartment_length,
                                            important_info,
                                            live_display,
                                            svg_image_scale,
                                            svg_image_width,
                                            svg_image_height,
                                            general_plot_name_string
                                            )
                        """
                        
                        production_plots.guidance_map_transducer_angle_sagittal_and_max_plane_transverse(patientUID,
                                            patient_sp_preprocessing_output_figures_dir,
                                            pydicom_item,
                                            dil_ref,
                                            oar_ref,
                                            rectum_ref_key,
                                            all_ref_key,
                                            structs_referenced_dict,
                                            plot_guidance_map_transducer_plane_open3d_structure_set_complete_demonstration_bool,
                                            biopsy_fire_travel_distances,
                                            biopsy_needle_compartment_length,
                                            interp_inter_slice_dist,
                                            interp_intra_slice_dist,
                                            radius_for_normals_estimation,
                                            max_nn_for_normals_estimation,
                                            important_info,
                                            live_display,
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

                if production_plots_input_dictionary["Cohort - Prostate biopsies heatmap"]["Plot bool"] == True:
                    main_indeterminate_task = indeterminate_progress_main.add_task('[red]Cohort - Prostate biopsies heatmap...', total=None)
                    main_indeterminate_task_completed = completed_progress.add_task('[green]Cohort - Prostate biopsies heatmap', total=1, visible = False)

                    cohort_output_figures_dir = master_structure_info_dict["Global"]['Cohort figures dir']

                    general_plot_name_string = production_plots_input_dictionary["Cohort - Prostate biopsies heatmap"]["Plot name"]

                    cohort_biopsy_basic_spatial_features_dataframe = master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: Biopsy basic spatial features dataframe"]

                    production_plots.production_plot_cohort_double_sextant_biopsy_distribution(cohort_biopsy_basic_spatial_features_dataframe,
                                                                              general_plot_name_string,
                                                                              cohort_output_figures_dir)
                    
                    indeterminate_progress_main.update(main_indeterminate_task, visible = False)
                    completed_progress.update(main_indeterminate_task_completed, advance = 1,visible = True)
                    live_display.refresh() 
                else:
                    pass


                if production_plots_input_dictionary["Cohort - Prostate DILs heatmap"]["Plot bool"] == True:
                    main_indeterminate_task = indeterminate_progress_main.add_task('[red]Cohort - Prostate DILs heatmap...', total=None)
                    main_indeterminate_task_completed = completed_progress.add_task('[green]Cohort - Prostate DILs heatmap', total=1, visible = False)

                    cohort_output_figures_dir = master_structure_info_dict["Global"]['Cohort figures dir']

                    general_plot_name_string = production_plots_input_dictionary["Cohort - Prostate DILs heatmap"]["Plot name"]

                    structure_cohort_3d_radiomic_features_dataframe = master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: 3D radiomic features all OAR and DIL structures"]

                    production_plots.production_plot_cohort_double_sextant_dil_distribution(structure_cohort_3d_radiomic_features_dataframe,
                                                           general_plot_name_string,
                                                           cohort_output_figures_dir)
                    
                    indeterminate_progress_main.update(main_indeterminate_task, visible = False)
                    completed_progress.update(main_indeterminate_task_completed, advance = 1,visible = True)
                    live_display.refresh() 
                else:
                    pass

                




                if production_plots_input_dictionary["Cohort - Scatter plot matrix targeting accuracy"]["Plot bool"] == True:

                    main_indeterminate_task = indeterminate_progress_main.add_task('[red]Cohort - scatter plot matrix target accuracy...', total=None)
                    main_indeterminate_task_completed = completed_progress.add_task('[green]Cohort - scatter plot matrix target accuracy', total=1, visible = False)

                    cohort_output_figures_dir = master_structure_info_dict["Global"]['Cohort figures dir']

                    general_plot_name_string = production_plots_input_dictionary["Cohort - Scatter plot matrix targeting accuracy"]["Plot name"]

                    cohort_nearest_dils_dataframe = master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: Nearest DILs to each biopsy"]

                    production_plots.production_plot_cohort_scatter_plot_matrix_bx_centroids_real_in_dil_frame_v2(cohort_nearest_dils_dataframe,
                                                                              general_plot_name_string,
                                                                              cohort_output_figures_dir)
                    
                    indeterminate_progress_main.update(main_indeterminate_task, visible = False)
                    completed_progress.update(main_indeterminate_task_completed, advance = 1,visible = True)
                    live_display.refresh()    

                else: 
                    pass
                
                if production_plots_input_dictionary["Cohort - Scatter plot 2d gaussian transverse accuracy"]["Plot bool"] == True:

                    main_indeterminate_task = indeterminate_progress_main.add_task('[red]Cohort - scatter plot Gauss transverse accuracy...', total=None)
                    main_indeterminate_task_completed = completed_progress.add_task('[green]Cohort - scatter plot Gauss transverse accuracy', total=1, visible = False)

                    cohort_output_figures_dir = master_structure_info_dict["Global"]['Cohort figures dir']

                    general_plot_name_string = production_plots_input_dictionary["Cohort - Scatter plot 2d gaussian transverse accuracy"]["Plot name"]

                    cohort_nearest_dils_dataframe = master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: Nearest DILs to each biopsy"]
                    cohort_global_tissue_score_by_structure_dataframe = master_cohort_patient_data_and_dataframes['Dataframes']['Cohort: tissue class global scores (structure)']

                    production_plots.production_plot_transverse_accuracy_with_marginals_and_gaussian_fit_global_tissue_score_coloring_with_table(cohort_nearest_dils_dataframe,
                                                                                                                                    cohort_global_tissue_score_by_structure_dataframe,
                                                                                                                                    general_plot_name_string,
                                                                                                                                    cohort_output_figures_dir)
                    
                    indeterminate_progress_main.update(main_indeterminate_task, visible = False)
                    completed_progress.update(main_indeterminate_task_completed, advance = 1,visible = True)
                    live_display.refresh()    

                else: 
                    pass

                if production_plots_input_dictionary["Cohort - Scatter plot 2d gaussian sagittal accuracy"]["Plot bool"] == True:

                    main_indeterminate_task = indeterminate_progress_main.add_task('[red]Cohort - scatter plot Gauss sagittal accuracy...', total=None)
                    main_indeterminate_task_completed = completed_progress.add_task('[green]Cohort - scatter plot Gauss sagittal accuracy', total=1, visible = False)

                    cohort_output_figures_dir = master_structure_info_dict["Global"]['Cohort figures dir']

                    general_plot_name_string = production_plots_input_dictionary["Cohort - Scatter plot 2d gaussian sagittal accuracy"]["Plot name"]

                    cohort_nearest_dils_dataframe = master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: Nearest DILs to each biopsy"]
                    cohort_global_tissue_score_by_structure_dataframe = master_cohort_patient_data_and_dataframes['Dataframes']['Cohort: tissue class global scores (structure)']

                    production_plots.production_plot_sagittal_accuracy_with_marginals_and_gaussian_fit_global_tissue_score_coloring_with_table(cohort_nearest_dils_dataframe,
                                                                                                                                    cohort_global_tissue_score_by_structure_dataframe,
                                                                                                                                    general_plot_name_string,
                                                                                                                                    cohort_output_figures_dir)
                    
                    indeterminate_progress_main.update(main_indeterminate_task, visible = False)
                    completed_progress.update(main_indeterminate_task_completed, advance = 1,visible = True)
                    live_display.refresh()    

                else: 
                    pass


                


                


            ### MC SIM FIGS
            if create_at_least_one_production_plot == True and mc_sim_complete_bool == True:
                important_info.add_text_line("Creating production plots.", live_display)


                patient_sp_output_figures_dir_dict = master_structure_info_dict["Global"]['Patient specific output figures directory (MC sim) dict']
                
                #live_display.stop()

                # Plot boxplots of sampled rigid shift vectors
                if production_plots_input_dictionary["Sampled translation vector magnitudes box plots"]["Plot bool"] == True:
                    
                    general_plot_name_string = production_plots_input_dictionary["Sampled translation vector magnitudes box plots"]["Plot name"]
                    box_plot_color = production_plots_input_dictionary["Sampled translation vector magnitudes box plots"]["Plot color"]

                    patientUID_default = "Initializing"
                    processing_patient_production_plot_description = "Creating box plots of sampled rigid shift vectors [{}]...".format(patientUID_default)
                    processing_patients_task = patients_progress.add_task("[red]"+processing_patient_production_plot_description, total = master_structure_info_dict["Global"]["Num cases"])
                    processing_patient_production_plot_description_completed = "Creating box plots of sampled rigid shift vectors"
                    processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_production_plot_description_completed, total=master_structure_info_dict["Global"]["Num cases"], visible=False)

                    bx_types_list = master_structure_info_dict["Global"]["Bx types list"]
                    mc_containment_sim_complete_bool = master_structure_info_dict['Global']["MC info"]['MC containment sim performed']
                    mc_dose_sim_complete_bool = master_structure_info_dict['Global']["MC info"]['MC dose sim performed']
                    mc_mr_sim_complete = master_structure_info_dict['Global']["MC info"]['MC MR sim performed']

                    for patientUID,pydicom_item in master_structure_reference_dict.items():
                        
                        processing_patient_production_plot_description = "Creating box plots of sampled rigid shift vectors [{}]...".format(patientUID)
                        patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_production_plot_description)
                        """
                        max_simulations = master_structure_info_dict["Global"]["MC info"]["Max of num MC simulations"]
                        production_plots.production_plot_sampled_shift_vector_box_plots_by_patient(patientUID,
                                                patient_sp_output_figures_dir_dict,
                                                structs_referenced_list,
                                                bx_ref,
                                                pydicom_item,
                                                max_simulations,
                                                all_ref_key,
                                                svg_image_scale,
                                                svg_image_width,
                                                svg_image_height,
                                                box_plot_color,
                                                general_plot_name_string
                                                )
                        """

                        
                        # Tissue class 
                        if mc_containment_sim_complete_bool == True:
                            struct_refs_to_include_list = structs_referenced_list
                            bx_sim_types_to_include_list = ['Real']
                            general_plot_name_string_tissue_only_real_bxs = general_plot_name_string + '_tissue_only_real_bxs'
                            num_tissue_sims = master_structure_info_dict["Global"]["MC info"]["Num MC containment simulations"]

                            
                            production_plots.production_plot_sampled_shift_vector_box_plots_by_patient_NEW(patientUID,
                                                patient_sp_output_figures_dir_dict,
                                                pydicom_item,
                                                svg_image_scale,
                                                svg_image_width,
                                                svg_image_height,
                                                general_plot_name_string_tissue_only_real_bxs,
                                                struct_refs_to_include_list,
                                                bx_ref,
                                                bx_sim_types_to_include_list,
                                                all_ref_key,
                                                num_tissue_sims)


                            bx_sim_types_to_include_list = bx_types_list
                            general_plot_name_string_tissue_all_bxs = general_plot_name_string + '_tissue_all_bxs'
                            production_plots.production_plot_sampled_shift_vector_box_plots_by_patient_NEW(patientUID,
                                                patient_sp_output_figures_dir_dict,
                                                pydicom_item,
                                                svg_image_scale,
                                                svg_image_width,
                                                svg_image_height,
                                                general_plot_name_string_tissue_all_bxs,
                                                struct_refs_to_include_list,
                                                bx_ref,
                                                bx_sim_types_to_include_list,
                                                all_ref_key,
                                                num_tissue_sims)

                        # Dosimetry 
                        if mc_dose_sim_complete_bool == True:
                            struct_refs_to_include_list = [bx_ref]
                            bx_sim_types_to_include_list = ['Real']
                            general_plot_name_string_dosim_only_real_bxs = general_plot_name_string + '_dosimetry_only_real_bxs'
                            num_dose_sims = master_structure_info_dict["Global"]["MC info"]["Num MC dose simulations"]

                            production_plots.production_plot_sampled_shift_vector_box_plots_by_patient_NEW(patientUID,
                                                patient_sp_output_figures_dir_dict,
                                                pydicom_item,
                                                svg_image_scale,
                                                svg_image_width,
                                                svg_image_height,
                                                general_plot_name_string_dosim_only_real_bxs,
                                                struct_refs_to_include_list,
                                                bx_ref,
                                                bx_sim_types_to_include_list,
                                                all_ref_key,
                                                num_dose_sims)


                            bx_sim_types_to_include_list = bx_types_list
                            general_plot_name_string_dosim_all_bxs = general_plot_name_string + '_dosimetry_all_bxs'
                            production_plots.production_plot_sampled_shift_vector_box_plots_by_patient_NEW(patientUID,
                                                patient_sp_output_figures_dir_dict,
                                                pydicom_item,
                                                svg_image_scale,
                                                svg_image_width,
                                                svg_image_height,
                                                general_plot_name_string_dosim_all_bxs,
                                                struct_refs_to_include_list,
                                                bx_ref,
                                                bx_sim_types_to_include_list,
                                                all_ref_key,
                                                num_dose_sims)

                        # MR 
                        if mc_mr_sim_complete == True:
                            struct_refs_to_include_list = [bx_ref]
                            bx_sim_types_to_include_list = ['Real']
                            general_plot_name_string_MR_only_real_bxs = general_plot_name_string + '_MR_only_real_bxs'
                            num_MR_sims = master_structure_info_dict["Global"]["MC info"]["Num MC MR simulations"]

                            production_plots.production_plot_sampled_shift_vector_box_plots_by_patient_NEW(patientUID,
                                                patient_sp_output_figures_dir_dict,
                                                pydicom_item,
                                                svg_image_scale,
                                                svg_image_width,
                                                svg_image_height,
                                                general_plot_name_string_MR_only_real_bxs,
                                                struct_refs_to_include_list,
                                                bx_ref,
                                                bx_sim_types_to_include_list,
                                                all_ref_key,
                                                num_MR_sims)


                            bx_sim_types_to_include_list = bx_types_list
                            general_plot_name_string_MR_all_bxs = general_plot_name_string + '_MR_all_bxs'
                            production_plots.production_plot_sampled_shift_vector_box_plots_by_patient_NEW(patientUID,
                                                patient_sp_output_figures_dir_dict,
                                                pydicom_item,
                                                svg_image_scale,
                                                svg_image_width,
                                                svg_image_height,
                                                general_plot_name_string_MR_all_bxs,
                                                struct_refs_to_include_list,
                                                bx_ref,
                                                bx_sim_types_to_include_list,
                                                all_ref_key,
                                                num_MR_sims)
                        
                        patients_progress.update(processing_patients_task, advance = 1)
                        completed_progress.update(processing_patients_completed_task, advance = 1)

                    patients_progress.update(processing_patients_task, visible = False)
                    completed_progress.update(processing_patients_completed_task, visible = True)
                else:
                    pass

                
                if production_plots_input_dictionary["Cohort - Sampled translation vector magnitudes box plots"]["Plot bool"] == True:

                    main_indeterminate_task = indeterminate_progress_main.add_task('[red]Cohort - Sampled shift vectors boxplots...', total=None)
                    main_indeterminate_task_completed = completed_progress.add_task('[green]Cohort - Sampled shift vectors boxplots', total=1, visible = False)

                    cohort_output_figures_dir = master_structure_info_dict["Global"]['Cohort figures dir']

                    general_plot_name_string = production_plots_input_dictionary["Cohort - Sampled translation vector magnitudes box plots"]["Plot name"]


                    bx_types_list = master_structure_info_dict["Global"]["Bx types list"]
                    mc_containment_sim_complete_bool = master_structure_info_dict['Global']["MC info"]['MC containment sim performed']
                    mc_dose_sim_complete_bool = master_structure_info_dict['Global']["MC info"]['MC dose sim performed']
                    mc_mr_sim_complete = master_structure_info_dict['Global']["MC info"]['MC MR sim performed']

                    # Tissue class 
                    if mc_containment_sim_complete_bool == True:
                        struct_refs_to_include_list = structs_referenced_list
                        bx_sim_types_to_include_list = ['Real']
                        general_plot_name_string_tissue_only_real_bxs = general_plot_name_string + '_tissue_only_real_bxs'
                        custom_str_to_append_to_plot_title = 'Tissue only real biopsies'
                        num_tissue_sims = master_structure_info_dict["Global"]["MC info"]["Num MC containment simulations"]
                        
                        production_plots.production_plot_sampled_shift_vector_box_plots_cohort(cohort_output_figures_dir,
                                                            master_cohort_patient_data_and_dataframes,
                                                            svg_image_scale,
                                                            svg_image_width,
                                                            svg_image_height,
                                                            general_plot_name_string_tissue_only_real_bxs,
                                                            struct_refs_to_include_list,
                                                            bx_ref,
                                                            bx_sim_types_to_include_list,
                                                            num_tissue_sims,
                                                            custom_str_to_append_to_plot_title
                                                            )


                        bx_sim_types_to_include_list = bx_types_list
                        general_plot_name_string_tissue_all_bxs = general_plot_name_string + '_tissue_all_bxs'
                        custom_str_to_append_to_plot_title = 'Tissue all biopsies'
                        production_plots.production_plot_sampled_shift_vector_box_plots_cohort(cohort_output_figures_dir,
                                                            master_cohort_patient_data_and_dataframes,
                                                            svg_image_scale,
                                                            svg_image_width,
                                                            svg_image_height,
                                                            general_plot_name_string_tissue_all_bxs,
                                                            struct_refs_to_include_list,
                                                            bx_ref,
                                                            bx_sim_types_to_include_list,
                                                            num_tissue_sims,
                                                            custom_str_to_append_to_plot_title
                                                            )

                    # Dosimetry 
                    if mc_dose_sim_complete_bool == True:
                        struct_refs_to_include_list = [bx_ref]
                        bx_sim_types_to_include_list = ['Real']
                        general_plot_name_string_dosim_only_real_bxs = general_plot_name_string + '_dosimetry_only_real_bxs'
                        custom_str_to_append_to_plot_title = 'Dosimetry only real biopsies'
                        num_dose_sims = master_structure_info_dict["Global"]["MC info"]["Num MC dose simulations"]


                        production_plots.production_plot_sampled_shift_vector_box_plots_cohort(cohort_output_figures_dir,
                                                            master_cohort_patient_data_and_dataframes,
                                                            svg_image_scale,
                                                            svg_image_width,
                                                            svg_image_height,
                                                            general_plot_name_string_dosim_only_real_bxs,
                                                            struct_refs_to_include_list,
                                                            bx_ref,
                                                            bx_sim_types_to_include_list,
                                                            num_dose_sims,
                                                            custom_str_to_append_to_plot_title)


                        bx_sim_types_to_include_list = bx_types_list
                        general_plot_name_string_dosim_all_bxs = general_plot_name_string + '_dosimetry_all_bxs'
                        custom_str_to_append_to_plot_title = 'Dosimetry all biopsies'
                        production_plots.production_plot_sampled_shift_vector_box_plots_cohort(cohort_output_figures_dir,
                                                            master_cohort_patient_data_and_dataframes,
                                                            svg_image_scale,
                                                            svg_image_width,
                                                            svg_image_height,
                                                            general_plot_name_string_dosim_all_bxs,
                                                            struct_refs_to_include_list,
                                                            bx_ref,
                                                            bx_sim_types_to_include_list,
                                                            num_dose_sims,
                                                            custom_str_to_append_to_plot_title)

                    # MR 
                    if mc_mr_sim_complete == True:
                        struct_refs_to_include_list = [bx_ref]
                        bx_sim_types_to_include_list = ['Real']
                        general_plot_name_string_MR_only_real_bxs = general_plot_name_string + '_MR_only_real_bxs'
                        custom_str_to_append_to_plot_title = 'MR only real biopsies'
                        num_MR_sims = master_structure_info_dict["Global"]["MC info"]["Num MC MR simulations"]

                        production_plots.production_plot_sampled_shift_vector_box_plots_cohort(cohort_output_figures_dir,
                                                            master_cohort_patient_data_and_dataframes,
                                                            svg_image_scale,
                                                            svg_image_width,
                                                            svg_image_height,
                                                            general_plot_name_string_MR_only_real_bxs,
                                                            struct_refs_to_include_list,
                                                            bx_ref,
                                                            bx_sim_types_to_include_list,
                                                            num_MR_sims,
                                                            custom_str_to_append_to_plot_title)


                        bx_sim_types_to_include_list = bx_types_list
                        general_plot_name_string_MR_all_bxs = general_plot_name_string + '_MR_all_bxs'
                        custom_str_to_append_to_plot_title = 'MR all biopsies'
                        production_plots.production_plot_sampled_shift_vector_box_plots_cohort(cohort_output_figures_dir,
                                                            master_cohort_patient_data_and_dataframes,
                                                            svg_image_scale,
                                                            svg_image_width,
                                                            svg_image_height,
                                                            general_plot_name_string_MR_all_bxs,
                                                            struct_refs_to_include_list,
                                                            bx_ref,
                                                            bx_sim_types_to_include_list,
                                                            num_MR_sims,
                                                            custom_str_to_append_to_plot_title)
                    
                    indeterminate_progress_main.update(main_indeterminate_task, visible = False)
                    completed_progress.update(main_indeterminate_task_completed, advance = 1,visible = True)
                    live_display.refresh()    

                else: 
                    pass















                # all MC trials spatial axial dose distribution with global regression 

                if production_plots_input_dictionary["Axial dose distribution all trials and global regression"]["Plot bool"] == True:
                    
                    general_plot_name_string = production_plots_input_dictionary["Axial dose distribution all trials and global regression"]["Plot name"]
                
                    patientUID_default = "Initializing"
                    processing_patient_production_plot_description = "Creating axial dose distribution scatter plot (all trials) [{}]...".format(patientUID_default)
                    processing_patients_task = patients_progress.add_task("[red]"+processing_patient_production_plot_description, total = master_structure_info_dict["Global"]["Num cases"])
                    processing_patient_production_plot_description_completed = "Creating axial dose distribution scatter plot (all trials)"
                    processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_production_plot_description_completed, total=master_structure_info_dict["Global"]["Num cases"], visible=False)


                    for patientUID,pydicom_item in master_structure_reference_dict.items():
                        
                        processing_patient_production_plot_description = "Creating axial dose distribution scatter plot (all trials) [{}]...".format(patientUID)
                        patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_production_plot_description)

                        if dose_ref in pydicom_item:
                            production_plots.production_plot_axial_dose_distribution_all_trials_and_regression_by_patient(patient_sp_output_figures_dir_dict,
                                                                        patientUID,
                                                                        pydicom_item,
                                                                        bx_ref,
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
                    processing_patients_task = patients_progress.add_task("[red]"+processing_patient_production_plot_description, total = master_structure_info_dict["Global"]["Num cases"])
                    processing_patient_production_plot_description_completed = "Creating (3D) scatter axial/radial dose distribution plot (all trials)"
                    processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_production_plot_description_completed, total=master_structure_info_dict["Global"]["Num cases"], visible=False)


                    for patientUID,pydicom_item in master_structure_reference_dict.items():
                        
                        processing_patient_production_plot_description = "Creating (3D) scatter axial/radial dose distribution plot (all trials) [{}]...".format(patientUID)
                        patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_production_plot_description)

                        if dose_ref in pydicom_item:
                            production_plots.production_3d_scatter_dose_axial_radial_distribution_by_patient(patient_sp_output_figures_dir_dict,
                                                                        patientUID,
                                                                        pydicom_item,
                                                                        bx_ref,
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
                    processing_patients_task = patients_progress.add_task("[red]"+processing_patient_production_plot_description, total = master_structure_info_dict["Global"]["Num cases"])
                    processing_patient_production_plot_description_completed = "Creating (2D) colored scatter axial/radial dose distribution plot (all trials)"
                    processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_production_plot_description_completed, total=master_structure_info_dict["Global"]["Num cases"], visible=False)


                    for patientUID,pydicom_item in master_structure_reference_dict.items():
                        
                        processing_patient_production_plot_description = "Creating (2D) colored scatter axial/radial dose distribution plot (all trials) [{}]...".format(patientUID)
                        patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_production_plot_description)

                        if dose_ref in pydicom_item:
                            production_plots.production_2d_scatter_dose_axial_radial_color_distribution_by_patient(patient_sp_output_figures_dir_dict,
                                                                        patientUID,
                                                                        pydicom_item,
                                                                        bx_ref,
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
                    processing_patients_task = patients_progress.add_task("[red]"+processing_patient_production_plot_description, total = master_structure_info_dict["Global"]["Num cases"])
                    processing_patient_production_plot_description_completed = "Creating axial dose distribution quantile scatter plots"
                    processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_production_plot_description_completed, total=master_structure_info_dict["Global"]["Num cases"], visible=False)


                    for patientUID,pydicom_item in master_structure_reference_dict.items():
                        
                        processing_patient_production_plot_description = "Creating axial dose distribution quantile scatter plots [{}]...".format(patientUID)
                        patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_production_plot_description)

                        if dose_ref in pydicom_item:
                            production_plots.production_plot_axial_dose_distribution_quantile_scatter_by_patient(patient_sp_output_figures_dir_dict,
                                                                            patientUID,
                                                                            pydicom_item,
                                                                            bx_ref,
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




                # quantile regression of axial dose distribution NEW
                live_display.stop()
                if production_plots_input_dictionary["Axial dose distribution quantiles regression plot matplotlib"]["Plot bool"] == True:
                    
                    general_plot_name_string = production_plots_input_dictionary["Axial dose distribution quantiles regression plot matplotlib"]["Plot name"]
                    num_rand_trials_to_show = production_plots_input_dictionary["Axial dose distribution quantiles regression plot matplotlib"]["Num rand trials to show"]
                    
                    patientUID_default = "Initializing"
                    processing_patient_production_plot_description = "Creating axial dose distribution quantile regression plots [{}]...".format(patientUID_default)
                    processing_patients_task = patients_progress.add_task("[red]"+processing_patient_production_plot_description, total = master_structure_info_dict["Global"]["Num cases"])
                    processing_patient_production_plot_description_completed = "Creating axial dose distribution quantile regression plots"
                    processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_production_plot_description_completed, total=master_structure_info_dict["Global"]["Num cases"], visible=False)

                    # select dose column
                    value_col_key = 'Dose (Gy)'

                    for patientUID,pydicom_item in master_structure_reference_dict.items():
                        
                        processing_patient_production_plot_description = "Creating axial dose distribution quantile regression plots [{}]...".format(patientUID)
                        patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_production_plot_description)

                        if dose_ref in pydicom_item:
                            production_plots.production_plot_axial_dose_distribution_quantile_regression_by_patient_matplotlib(pydicom_item,
                                                                                    patientUID,
                                                                                    bx_ref,
                                                                                    all_ref_key,
                                                                                    value_col_key,
                                                                                    patient_sp_output_figures_dir_dict,
                                                                                    general_plot_name_string,
                                                                                    num_rand_trials_to_show)
                        
                        patients_progress.update(processing_patients_task, advance = 1)
                        completed_progress.update(processing_patients_completed_task, advance = 1)

                    patients_progress.update(processing_patients_task, visible = False)
                    completed_progress.update(processing_patients_completed_task, visible = True)   
                else:
                    pass

                if production_plots_input_dictionary["Axial dose GRADIENT distribution quantiles regression plot matplotlib"]["Plot bool"] == True:
                    
                    general_plot_name_string = production_plots_input_dictionary["Axial dose GRADIENT distribution quantiles regression plot matplotlib"]["Plot name"]
                    num_rand_trials_to_show = production_plots_input_dictionary["Axial dose GRADIENT distribution quantiles regression plot matplotlib"]["Num rand trials to show"]
                    
                    patientUID_default = "Initializing"
                    processing_patient_production_plot_description = "Creating axial dose distribution quantile regression plots [{}]...".format(patientUID_default)
                    processing_patients_task = patients_progress.add_task("[red]"+processing_patient_production_plot_description, total = master_structure_info_dict["Global"]["Num cases"])
                    processing_patient_production_plot_description_completed = "Creating axial dose distribution quantile regression plots"
                    processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_production_plot_description_completed, total=master_structure_info_dict["Global"]["Num cases"], visible=False)

                    # select dose gradient column
                    value_col_key = 'Dose grad (Gy/mm)'

                    for patientUID,pydicom_item in master_structure_reference_dict.items():
                        
                        processing_patient_production_plot_description = "Creating axial dose distribution quantile regression plots [{}]...".format(patientUID)
                        patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_production_plot_description)



                        if dose_ref in pydicom_item:
                            production_plots.production_plot_axial_dose_distribution_quantile_regression_by_patient_matplotlib(pydicom_item,
                                                                                    patientUID,
                                                                                    bx_ref,
                                                                                    all_ref_key,
                                                                                    value_col_key,
                                                                                    patient_sp_output_figures_dir_dict,
                                                                                    general_plot_name_string,
                                                                                    num_rand_trials_to_show)
                        
                        patients_progress.update(processing_patients_task, advance = 1)
                        completed_progress.update(processing_patients_completed_task, advance = 1)

                    patients_progress.update(processing_patients_task, visible = False)
                    completed_progress.update(processing_patients_completed_task, visible = True)   
                else:
                    pass

                
                #live_display.start()

                # quantile regression of axial dose distribution

                
                if production_plots_input_dictionary["Axial dose distribution quantiles regression plot"]["Plot bool"] == True:
                    
                    general_plot_name_string = production_plots_input_dictionary["Axial dose distribution quantiles regression plot"]["Plot name"]
                    
                    patientUID_default = "Initializing"
                    processing_patient_production_plot_description = "Creating axial dose distribution quantile regression plots [{}]...".format(patientUID_default)
                    processing_patients_task = patients_progress.add_task("[red]"+processing_patient_production_plot_description, total = master_structure_info_dict["Global"]["Num cases"])
                    processing_patient_production_plot_description_completed = "Creating axial dose distribution quantile regression plots"
                    processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_production_plot_description_completed, total=master_structure_info_dict["Global"]["Num cases"], visible=False)


                    for patientUID,pydicom_item in master_structure_reference_dict.items():
                        
                        processing_patient_production_plot_description = "Creating axial dose distribution quantile regression plots [{}]...".format(patientUID)
                        patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_production_plot_description)

                        if dose_ref in pydicom_item:
                            production_plots.production_plot_axial_dose_distribution_quantile_regressions_by_patient(patient_sp_output_figures_dir_dict,
                                                                        patientUID,
                                                                        pydicom_item,
                                                                        bx_ref,
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
                    processing_patients_task = patients_progress.add_task("[red]"+processing_patient_production_plot_description, total = master_structure_info_dict["Global"]["Num cases"])
                    processing_patient_production_plot_description_completed = "Creating voxelized axial dose distribution box plots"
                    processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_production_plot_description_completed, total=master_structure_info_dict["Global"]["Num cases"], visible=False)


                    for patientUID,pydicom_item in master_structure_reference_dict.items():
                        
                        processing_patient_production_plot_description = "Creating voxelized axial dose distribution box plots [{}]...".format(patientUID)
                        patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_production_plot_description)

                        if dose_ref in pydicom_item:
                            production_plots.production_plot_voxelized_axial_dose_distribution_box_plot_by_patient(patient_sp_output_figures_dir_dict,
                                                                                patientUID,
                                                                                pydicom_item,
                                                                                bx_ref,
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
                """
                if production_plots_input_dictionary["Axial dose voxelized ridgeline plot"]["Plot bool"] == True:

                    main_indeterminate_task = indeterminate_progress_main.add_task('[red]Dose voxelized ridge plots...', total=None)
                    main_indeterminate_task_completed = completed_progress.add_task('[green]Dose voxelized ridge plots', total=1, visible = False)


                    ridge_line_dose_general_plot_name_string = production_plots_input_dictionary["Axial dose voxelized ridgeline plot"]["Plot name"]
                    #structure_miss_probability_roi = production_plots_input_dictionary["Cohort - Tissue volume by biopsy type"]["Structure miss ROI"]


                    production_plots.production_plot_dose_ridge_plot_by_voxel_v2(master_cohort_patient_data_and_dataframes,
                                                                            svg_image_width,
                                                                            svg_image_height,
                                                                            dpi_for_seaborn_plots,
                                                                            ridge_line_dose_general_plot_name_string,
                                                                            patient_sp_output_figures_dir_dict
                                                                            )



                    indeterminate_progress_main.update(main_indeterminate_task, visible = False)
                    completed_progress.update(main_indeterminate_task_completed, advance = 1,visible = True)
                    live_display.refresh()    

                else: 
                    pass
                """
                if production_plots_input_dictionary["Axial dose voxelized ridgeline plot"]["Plot bool"] == True:

                    main_indeterminate_task = indeterminate_progress_main.add_task('[red]Dose voxelized ridge plots...', total=None)
                    main_indeterminate_task_completed = completed_progress.add_task('[green]Dose voxelized ridge plots', total=1, visible = False)


                    ridge_line_dose_general_plot_name_string = production_plots_input_dictionary["Axial dose voxelized ridgeline plot"]["Plot name"]

                    for patientUID,pydicom_item in master_structure_reference_dict.items():
                        if dose_ref in pydicom_item:
                            for specific_bx_structure in pydicom_item[bx_ref]:
                                
                                sp_bx_dose_distribution_all_trials_df = specific_bx_structure["Output data frames"]["Point-wise dose output by MC trial number"]
                                
                                production_plots.production_plot_dose_ridge_plot_by_voxel_v3(sp_bx_dose_distribution_all_trials_df,
                                                                                        svg_image_width,
                                                                                        svg_image_height,
                                                                                        dpi_for_seaborn_plots,
                                                                                        ridge_line_dose_general_plot_name_string,
                                                                                        patient_sp_output_figures_dir_dict,
                                                                                        all_ref_key
                                                                                        )



                    indeterminate_progress_main.update(main_indeterminate_task, visible = False)
                    completed_progress.update(main_indeterminate_task_completed, advance = 1,visible = True)
                    live_display.refresh()    

                else: 
                    pass
                

                """
                # Dose ridgeline plot with the densities colored according to tissue class
                if production_plots_input_dictionary["Axial dose and tissue colored voxelized ridgeline plot"]["Plot bool"] == True:

                    main_indeterminate_task = indeterminate_progress_main.add_task('[red]Dose and tissue colored voxelized ridge plots...', total=None)
                    main_indeterminate_task_completed = completed_progress.add_task('[green]Dose and tissue colored voxelized ridge plots', total=1, visible = False)


                    ridge_line_dose_and_binom_general_plot_name_string = production_plots_input_dictionary["Axial dose and tissue colored voxelized ridgeline plot"]["Plot name"]


                    production_plots.production_plot_dose_ridge_plot_by_voxel_with_tissue_class_coloring(master_cohort_patient_data_and_dataframes,
                                                                                                        svg_image_width,
                                                                                                        svg_image_height,
                                                                                                        dpi_for_seaborn_plots,
                                                                                                        ridge_line_dose_and_binom_general_plot_name_string,
                                                                                                        patient_sp_output_figures_dir_dict,
                                                                                                        cancer_tissue_label
                                                                                                        )


                    indeterminate_progress_main.update(main_indeterminate_task, visible = False)
                    completed_progress.update(main_indeterminate_task_completed, advance = 1,visible = True)
                    live_display.refresh()    

                else: 
                    pass
                """

                live_display.stop()
                # Dose ridgeline plot with the densities colored according to tissue class
                if production_plots_input_dictionary["Axial dose and tissue colored voxelized ridgeline plot"]["Plot bool"] == True:

                    main_indeterminate_task = indeterminate_progress_main.add_task('[red]Dose and tissue colored voxelized ridge plots...', total=None)
                    main_indeterminate_task_completed = completed_progress.add_task('[green]Dose and tissue colored voxelized ridge plots', total=1, visible = False)


                    ridge_line_dose_and_binom_general_plot_name_string = production_plots_input_dictionary["Axial dose and tissue colored voxelized ridgeline plot"]["Plot name"]

                    cohort_all_binom_data_by_trial_and_pt = master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: Entire point-wise binom est distribution"]

                    cohort_voxel_dose_dataframe_by_voxel = master_cohort_patient_data_and_dataframes['Dataframes']['Cohort: Global dosimetry by voxel']

                    for patientUID,pydicom_item in master_structure_reference_dict.items():
                        if dose_ref in pydicom_item:
                        
                            #sp_patient_all_dose_data_by_trial_and_pt = pydicom_item[all_ref_key]["Dosimetry - All points and trials"]
                            sp_patient_all_binom_data_by_trial_and_pt = cohort_all_binom_data_by_trial_and_pt[cohort_all_binom_data_by_trial_and_pt["Patient ID"] == patientUID]

                            for specific_bx_structure in pydicom_item[bx_ref]:
                                bx_id = specific_bx_structure['ROI']
                                sp_bx_dose_distribution_all_trials_df = specific_bx_structure["Output data frames"]["Point-wise dose output by MC trial number"]

                                sp_patient_and_sp_bx_dose_dataframe_by_voxel = cohort_voxel_dose_dataframe_by_voxel[(cohort_voxel_dose_dataframe_by_voxel['Patient ID'] == patientUID) & 
                                                                                                                    (cohort_voxel_dose_dataframe_by_voxel['Bx ID'] == bx_id)]
                                """
                                production_plots.production_plot_dose_ridge_plot_by_voxel_with_tissue_class_coloring_no_dose_cohort(sp_bx_dose_distribution_all_trials_df,
                                                                                                                                    sp_patient_all_binom_data_by_trial_and_pt,
                                                                                                                    svg_image_width,
                                                                                                                    svg_image_height,
                                                                                                                    dpi_for_seaborn_plots,
                                                                                                                    ridge_line_dose_and_binom_general_plot_name_string,
                                                                                                                    patient_sp_output_figures_dir_dict,
                                                                                                                    cancer_tissue_label
                                                                                                                    )
                                """
                                ### This new function version uses the dataframe that was created describing all dose statistics globally by voxel
                                production_plots.production_plot_dose_ridge_plot_by_voxel_with_tissue_class_coloring_no_dose_cohort_v2(sp_bx_dose_distribution_all_trials_df,
                                                                                                                                    sp_patient_and_sp_bx_dose_dataframe_by_voxel,
                                                                                                                                    sp_patient_all_binom_data_by_trial_and_pt,
                                                                                                                                    svg_image_width,
                                                                                                                                    svg_image_height,
                                                                                                                                    dpi_for_seaborn_plots,
                                                                                                                                    ridge_line_dose_and_binom_general_plot_name_string,
                                                                                                                                    patient_sp_output_figures_dir_dict,
                                                                                                                                    cancer_tissue_label)

                    indeterminate_progress_main.update(main_indeterminate_task, visible = False)
                    completed_progress.update(main_indeterminate_task_completed, advance = 1,visible = True)
                    live_display.refresh()    

                else: 
                    pass

                
                #live_display.start()

                # voxelized violin plot axial dose distribution  

                if production_plots_input_dictionary["Axial dose distribution voxelized violin plot"]["Plot bool"] == True:
                    
                    general_plot_name_string = production_plots_input_dictionary["Axial dose distribution voxelized violin plot"]["Plot name"]
                    violin_plot_color = production_plots_input_dictionary["Axial dose distribution voxelized violin plot"]["Plot color"]
                    
                    patientUID_default = "Initializing"
                    processing_patient_production_plot_description = "Creating voxelized axial dose distribution violin plots [{}]...".format(patientUID_default)
                    processing_patients_task = patients_progress.add_task("[red]"+processing_patient_production_plot_description, total = master_structure_info_dict["Global"]["Num cases"])
                    processing_patient_production_plot_description_completed = "Creating voxelized axial dose distribution violin plots"
                    processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_production_plot_description_completed, total=master_structure_info_dict["Global"]["Num cases"], visible=False)


                    for patientUID,pydicom_item in master_structure_reference_dict.items():
                        
                        processing_patient_production_plot_description = "Creating voxelized axial dose distribution violin plots [{}]...".format(patientUID)
                        patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_production_plot_description)

                        if dose_ref in pydicom_item:
                            production_plots.production_plot_voxelized_axial_dose_distribution_violin_plot_by_patient(patient_sp_output_figures_dir_dict,
                                                                                patientUID,
                                                                                pydicom_item,
                                                                                bx_ref,
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
                    processing_patients_task = patients_progress.add_task("[red]"+processing_patient_production_plot_description, total = master_structure_info_dict["Global"]["Num cases"])
                    processing_patient_production_plot_description_completed = "Creating differential DVH plots ("+str(num_differential_dvh_plots_to_show)+" trials)"
                    processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_production_plot_description_completed, total=master_structure_info_dict["Global"]["Num cases"], visible=False)


                    for patientUID,pydicom_item in master_structure_reference_dict.items():
                        
                        processing_patient_production_plot_description = "Creating differential DVH plots ("+str(num_differential_dvh_plots_to_show)+" trials) [{}]...".format(patientUID)
                        patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_production_plot_description)

                        if dose_ref in pydicom_item:
                            production_plots.production_plot_differential_DVH_showing_N_trials_by_patient(patient_sp_output_figures_dir_dict,
                                                    patientUID,
                                                    pydicom_item,
                                                    bx_ref,
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
                    processing_patients_task = patients_progress.add_task("[red]"+processing_patient_production_plot_description, total = master_structure_info_dict["Global"]["Num cases"])
                    processing_patient_production_plot_description_completed = "Creating box plots of differential DVH data (all trials)"
                    processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_production_plot_description_completed, total=master_structure_info_dict["Global"]["Num cases"], visible=False)


                    for patientUID,pydicom_item in master_structure_reference_dict.items():
                        
                        processing_patient_production_plot_description = "Creating box plots of differential DVH data (all trials) [{}]...".format(patientUID)
                        patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_production_plot_description)
                        
                        if dose_ref in pydicom_item:
                            production_plots.production_plot_differential_dvh_quantile_box_plot(patient_sp_output_figures_dir_dict,
                                                        patientUID,
                                                        pydicom_item,
                                                        bx_ref,
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
                        
                    
                # Show quantile regression from all trials of differential DVH data
                if production_plots_input_dictionary["Differential DVH dose quantiles plot seaborn"]["Plot bool"] == True:
                    
                    general_plot_name_string = production_plots_input_dictionary["Differential DVH dose quantiles plot seaborn"]["Plot name"]

                    patientUID_default = "Initializing"
                    processing_patient_production_plot_description = "Creating quantile plot differential DVH data [{}]...".format(patientUID_default)
                    processing_patients_task = patients_progress.add_task("[red]"+processing_patient_production_plot_description, total = master_structure_info_dict["Global"]["Num cases"])
                    processing_patient_production_plot_description_completed = "Creating quantile plot differential DVH data"
                    processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_production_plot_description_completed, total=master_structure_info_dict["Global"]["Num cases"], visible=False)


                    for patientUID,pydicom_item in master_structure_reference_dict.items():
                        
                        processing_patient_production_plot_description = "Creating quantile plot differential DVH data [{}]...".format(patientUID)
                        patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_production_plot_description)

                        if dose_ref in pydicom_item:
                            production_plots.production_plot_differential_dvh_quantile_plot_NEW(patient_sp_output_figures_dir_dict,
                                                    patientUID,
                                                    pydicom_item,
                                                    bx_ref,
                                                    general_plot_name_string,
                                                    important_info,
                                                    live_display
                                                    )
                        
                        patients_progress.update(processing_patients_task, advance = 1)
                        completed_progress.update(processing_patients_completed_task, advance = 1)

                    patients_progress.update(processing_patients_task, visible = False)
                    completed_progress.update(processing_patients_completed_task, visible = True)  
                else:
                    pass 


                # quantile regression of cumulative DVH plots NEW
                if production_plots_input_dictionary["Cumulative DVH quantile regression seaborn"]["Plot bool"] == True:
                    
                    general_plot_name_string = production_plots_input_dictionary["Cumulative DVH quantile regression seaborn"]["Plot name"]
                    
                    patientUID_default = "Initializing"
                    processing_patient_production_plot_description = "Creating quantile regression cumulative DVH plots [{}]...".format(patientUID_default)
                    processing_patients_task = patients_progress.add_task("[red]"+processing_patient_production_plot_description, total = master_structure_info_dict["Global"]["Num cases"])
                    processing_patient_production_plot_description_completed = "Creating quantile regression cumulative DVH plots"
                    processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_production_plot_description_completed, total=master_structure_info_dict["Global"]["Num cases"], visible=False)


                    for patientUID,pydicom_item in master_structure_reference_dict.items():
                        
                        processing_patient_production_plot_description = "Creating quantile regression cumulative DVH plots [{}]...".format(patientUID)
                        patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_production_plot_description)

                        if dose_ref in pydicom_item:
                            production_plots.production_plot_cumulative_DVH_kernel_quantile_regression_NEW_v2(patient_sp_output_figures_dir_dict,
                                                    patientUID,
                                                    pydicom_item,
                                                    bx_ref,
                                                    general_plot_name_string,
                                                    important_info,
                                                    live_display)
                                                 
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
                    processing_patients_task = patients_progress.add_task("[red]"+processing_patient_production_plot_description, total = master_structure_info_dict["Global"]["Num cases"])
                    processing_patient_production_plot_description_completed = "Creating cumulative DVH plots ("+str(num_cumulative_dvh_plots_to_show)+" trials)"
                    processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_production_plot_description_completed, total=master_structure_info_dict["Global"]["Num cases"], visible=False)


                    for patientUID,pydicom_item in master_structure_reference_dict.items():
                        
                        processing_patient_production_plot_description = "Creating cumulative DVH plots ("+str(num_cumulative_dvh_plots_to_show)+" trials) [{}]...".format(patientUID)
                        patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_production_plot_description)

                        if dose_ref in pydicom_item:
                            production_plots.production_plot_cumulative_DVH_showing_N_trials_by_patient(patient_sp_output_figures_dir_dict,
                                                        patientUID,
                                                        pydicom_item,
                                                        bx_ref,
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
                    processing_patients_task = patients_progress.add_task("[red]"+processing_patient_production_plot_description, total = master_structure_info_dict["Global"]["Num cases"])
                    processing_patient_production_plot_description_completed = "Creating cumulative DVH quantile regression plots"
                    processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_production_plot_description_completed, total=master_structure_info_dict["Global"]["Num cases"], visible=False)


                    for patientUID,pydicom_item in master_structure_reference_dict.items():
                        
                        processing_patient_production_plot_description = "Creating cumulative DVH quantile regression plots [{}]...".format(patientUID)
                        patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_production_plot_description)

                        if dose_ref in pydicom_item:
                            production_plots.production_plot_cumulative_DVH_quantile_regression_by_patient(patient_sp_output_figures_dir_dict,
                                                        patientUID,
                                                        pydicom_item,
                                                        bx_ref,
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
                        
                


                if production_plots_input_dictionary["Cohort - Dosimetry boxplots all biopsies"]["Plot bool"] == True:

                    main_indeterminate_task = indeterminate_progress_main.add_task('[red]Cohort - Dosimetry boxplots all biopsies...', total=None)
                    main_indeterminate_task_completed = completed_progress.add_task('[green]Cohort - Dosimetry boxplots all biopsies', total=1, visible = False)

                    cohort_output_figures_dir = master_structure_info_dict["Global"]['Cohort figures dir']

                    general_plot_name_string = production_plots_input_dictionary["Cohort - Dosimetry boxplots all biopsies"]["Plot name"]

                    cohort_global_dosimetry_dataframe = master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: Global dosimetry"]

                    production_plots.cohort_all_biopsies_dosimtery_boxplot_grouped_by_patient(cohort_global_dosimetry_dataframe,
                                   general_plot_name_string,
                                   cohort_output_figures_dir)
                    
                    indeterminate_progress_main.update(main_indeterminate_task, visible = False)
                    completed_progress.update(main_indeterminate_task_completed, advance = 1,visible = True)
                    live_display.refresh()    

                else: 
                    pass

                    


















                # perform containment probabilities plots and regressions
                #live_display.stop()



                if production_plots_input_dictionary["Tissue classification sum-to-one plot"]["Plot bool"] == True:
                    
                    general_plot_name_string = production_plots_input_dictionary["Tissue classification sum-to-one plot"]["Plot name"]


                    patientUID_default = "Initializing"
                    processing_patient_production_plot_description = "Creating tissue class sum-to-one plots [{}]...".format(patientUID_default)
                    processing_patients_task = patients_progress.add_task("[red]"+processing_patient_production_plot_description, total = master_structure_info_dict["Global"]["Num cases"])
                    processing_patient_production_plot_description_completed = "Creating tissue class sum-to-one plots"
                    processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_production_plot_description_completed, total=master_structure_info_dict["Global"]["Num cases"], visible=False)



                    for patientUID,pydicom_item in master_structure_reference_dict.items():
                        
                        processing_patient_production_plot_description = "Creating tissue class sum-to-one plots [{}]...".format(patientUID)
                        patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_production_plot_description)

                        production_plots.production_plot_sum_to_one_tissue_class_binom_regression_matplotlib(pydicom_item,
                                                                                 patientUID,
                                                                                 bx_ref,
                                                                                 all_ref_key,
                                                                                 structs_referenced_dict,
                                                                                 default_exterior_tissue,
                                                                                 patient_sp_output_figures_dir_dict,
                                                                                 general_plot_name_string)
                        
                        patients_progress.update(processing_patients_task, advance = 1)
                        completed_progress.update(processing_patients_completed_task, advance = 1)

                    patients_progress.update(processing_patients_task, visible = False)
                    completed_progress.update(processing_patients_completed_task, visible = True)  
                else:
                    pass


                # sum to one nominal charts
                if production_plots_input_dictionary["Tissue classification sum-to-one nominal plot"]["Plot bool"] == True:
                    
                    general_plot_name_string = production_plots_input_dictionary["Tissue classification sum-to-one nominal plot"]["Plot name"]


                    patientUID_default = "Initializing"
                    processing_patient_production_plot_description = "Creating tissue class sum-to-one nominal plots [{}]...".format(patientUID_default)
                    processing_patients_task = patients_progress.add_task("[red]"+processing_patient_production_plot_description, total = master_structure_info_dict["Global"]["Num cases"])
                    processing_patient_production_plot_description_completed = "Creating tissue class sum-to-one nominal plots"
                    processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_production_plot_description_completed, total=master_structure_info_dict["Global"]["Num cases"], visible=False)


                    for patientUID,pydicom_item in master_structure_reference_dict.items():
                        
                        processing_patient_production_plot_description = "Creating tissue class sum-to-one nominal plots [{}]...".format(patientUID)
                        patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_production_plot_description)

                        production_plots.production_plot_sum_to_one_tissue_class_nominal_plotly(patient_sp_output_figures_dir_dict,
                                                patientUID,
                                                pydicom_item,
                                                bx_ref,
                                                all_ref_key,
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

                if production_plots_input_dictionary["Cohort - Tissue class global score sum-to-one box plots"]["Plot bool"] == True:

                    main_indeterminate_task = indeterminate_progress_main.add_task('[red]Cohort - tissue class global scores sum-to-one boxplots...', total=None)
                    main_indeterminate_task_completed = completed_progress.add_task('[green]Cohort - tissue class global scores sum-to-one boxplots', total=1, visible = False)

                    cohort_output_figures_dir = master_structure_info_dict["Global"]['Cohort figures dir']

                    general_plot_name_string = production_plots_input_dictionary["Cohort - Tissue class global score sum-to-one box plots"]["Plot name"]

                    cohort_mc_sum_to_one_global_results_dataframe = master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: global sum-to-one mc results"]

                    production_plots.cohort_global_scores_boxplot_by_bx_type(cohort_mc_sum_to_one_global_results_dataframe,
                                 general_plot_name_string,
                                 cohort_output_figures_dir)
                    
                    indeterminate_progress_main.update(main_indeterminate_task, visible = False)
                    completed_progress.update(main_indeterminate_task_completed, advance = 1,visible = True)
                    live_display.refresh()    

                else: 
                    pass

                
                if production_plots_input_dictionary["Cohort - Tissue class sum-to-one spatial ridgeline plots"]["Plot bool"] == True:

                    main_indeterminate_task = indeterminate_progress_main.add_task('[red]Cohort - Tissue class sum-to-one spatial ridgeline plots...', total=None)
                    main_indeterminate_task_completed = completed_progress.add_task('[green]Cohort - Tissue class sum-to-one spatial ridgeline plots', total=1, visible = False)

                    cohort_output_figures_dir = master_structure_info_dict["Global"]['Cohort figures dir']

                    general_plot_name_string = production_plots_input_dictionary["Cohort - Tissue class sum-to-one spatial ridgeline plots"]["Plot name"]

                    cohort_mc_sum_to_one_pt_wise_results_dataframe = master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: sum-to-one mc results"]

                    production_plots.production_plot_cohort_sum_to_one_binom_est_ridge_plot_by_voxel(cohort_mc_sum_to_one_pt_wise_results_dataframe,
                                   svg_image_height/2.25,
                                   svg_image_height,
                                   dpi_for_seaborn_plots,
                                   general_plot_name_string,
                                   cohort_output_figures_dir)

                    
                    indeterminate_progress_main.update(main_indeterminate_task, visible = False)
                    completed_progress.update(main_indeterminate_task_completed, advance = 1,visible = True)
                    live_display.refresh()    

                else: 
                    pass

                if production_plots_input_dictionary["Cohort - Tissue class sum-to-one spatial nominal heatmap"]["Plot bool"] == True:

                    main_indeterminate_task = indeterminate_progress_main.add_task('[red]Cohort - Tissue class sum-to-one spatial nominal heatmap...', total=None)
                    main_indeterminate_task_completed = completed_progress.add_task('[green]Cohort - Tissue class sum-to-one spatial nominal heatmap', total=1, visible = False)

                    cohort_output_figures_dir = master_structure_info_dict["Global"]['Cohort figures dir']

                    general_plot_name_string = production_plots_input_dictionary["Cohort - Tissue class sum-to-one spatial nominal heatmap"]["Plot name"]

                    cohort_mc_sum_to_one_pt_wise_results_dataframe = master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: sum-to-one mc results"]

                    production_plots.production_plot_cohort_sum_to_one_nominal_counts_voxel_vs_tissue_class_heatmap(cohort_mc_sum_to_one_pt_wise_results_dataframe,
                                   svg_image_width,
                                   svg_image_height,
                                   dpi_for_seaborn_plots,
                                   general_plot_name_string,
                                   cohort_output_figures_dir)

                    
                    indeterminate_progress_main.update(main_indeterminate_task, visible = False)
                    completed_progress.update(main_indeterminate_task_completed, advance = 1,visible = True)
                    live_display.refresh()    

                else: 
                    pass


                if production_plots_input_dictionary["Cohort - All biopsy voxels histogram sum-to-one binom est by tissue class"]["Plot bool"] == True:

                    main_indeterminate_task = indeterminate_progress_main.add_task('[red]Cohort - All biopsy voxels histogram sum-to-one binom est by tissue class...', total=None)
                    main_indeterminate_task_completed = completed_progress.add_task('[green]Cohort - All biopsy voxels histogram sum-to-one binom est by tissue class', total=1, visible = False)

                    cohort_output_figures_dir = master_structure_info_dict["Global"]['Cohort figures dir']

                    general_plot_name_string = production_plots_input_dictionary["Cohort - All biopsy voxels histogram sum-to-one binom est by tissue class"]["Plot name"]

                    cohort_mc_sum_to_one_pt_wise_results_dataframe = master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: sum-to-one mc results"]
                    
                    bx_sample_pts_vol_element = master_structure_info_dict["Global"]["MC info"]["BX sample pt volume element (mm^3)"]

                    production_plots.production_plot_cohort_sum_to_one_all_biopsy_voxels_binom_est_histogram_by_tissue_class(cohort_mc_sum_to_one_pt_wise_results_dataframe,
                                                                                    svg_image_width,
                                                                                    svg_image_height,
                                                                                    dpi_for_seaborn_plots,
                                                                                    general_plot_name_string,
                                                                                    cohort_output_figures_dir,
                                                                                    bx_sample_pts_vol_element,
                                                                                    bin_width=0.05,
                                                                                    bandwidth=0.1)

                    
                    indeterminate_progress_main.update(main_indeterminate_task, visible = False)
                    completed_progress.update(main_indeterminate_task_completed, advance = 1,visible = True)
                    live_display.refresh()    

                else: 
                    pass


                    


                if production_plots_input_dictionary["Tissue classification scatter and regression probabilities all trials plot"]["Plot bool"] == True:
                    
                    general_plot_name_string = production_plots_input_dictionary["Tissue classification scatter and regression probabilities all trials plot"]["Plot name"]


                    patientUID_default = "Initializing"
                    processing_patient_production_plot_description = "Creating containment probability plots [{}]...".format(patientUID_default)
                    processing_patients_task = patients_progress.add_task("[red]"+processing_patient_production_plot_description, total = master_structure_info_dict["Global"]["Num cases"])
                    processing_patient_production_plot_description_completed = "Creating containment probability plots"
                    processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_production_plot_description_completed, total=master_structure_info_dict["Global"]["Num cases"], visible=False)


                    for patientUID,pydicom_item in master_structure_reference_dict.items():
                        
                        processing_patient_production_plot_description = "Creating containment probability plots [{}]...".format(patientUID)
                        patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_production_plot_description)

                        production_plots.production_plot_containment_probabilities_by_patient(patient_sp_output_figures_dir_dict,
                                                    patientUID,
                                                    pydicom_item,
                                                    bx_ref,
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
                    processing_patients_task = patients_progress.add_task("[red]"+processing_patient_production_plot_description, total = master_structure_info_dict["Global"]["Num cases"])
                    processing_patient_production_plot_description_completed = "Creating mutual containment probability plots"
                    processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_production_plot_description_completed, total=master_structure_info_dict["Global"]["Num cases"], visible=False)


                    for patientUID,pydicom_item in master_structure_reference_dict.items():
                        
                        processing_patient_production_plot_description = "Creating mutual containment probability plots [{}]...".format(patientUID)
                        patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_production_plot_description)

                        
                        production_plots.production_plot_mutual_containment_probabilities_by_patient(patient_sp_output_figures_dir_dict,
                                                    patientUID,
                                                    pydicom_item,
                                                    bx_ref,
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







                if production_plots_input_dictionary["Axial tissue class voxelized ridgeline plot"]["Plot bool"] == True:

                    main_indeterminate_task = indeterminate_progress_main.add_task('[red]Tissue class voxelized ridge plots...', total=None)
                    main_indeterminate_task_completed = completed_progress.add_task('[green]Tissue class voxelized ridge plots', total=1, visible = False)


                    ridge_line_tissue_class_general_plot_name_string = production_plots_input_dictionary["Axial tissue class voxelized ridgeline plot"]["Plot name"]


                    production_plots.production_plot_binom_est_ridge_plot_by_voxel_v2(master_cohort_patient_data_and_dataframes,
                                             svg_image_width,
                                             svg_image_height,
                                             dpi_for_seaborn_plots,
                                             ridge_line_tissue_class_general_plot_name_string,
                                            patient_sp_output_figures_dir_dict,
                                            cancer_tissue_label)



                    indeterminate_progress_main.update(main_indeterminate_task, visible = False)
                    completed_progress.update(main_indeterminate_task_completed, advance = 1,visible = True)
                    live_display.refresh()    

                else: 
                    pass

                print('Cohort - Tissue class global score by biopsy type')
                if production_plots_input_dictionary["Cohort - Tissue class global score by biopsy type"]["Plot bool"] == True:

                    main_indeterminate_task = indeterminate_progress_main.add_task('[red]Cohort - tissue class global by bx group...', total=None)
                    main_indeterminate_task_completed = completed_progress.add_task('[green]Cohort - tissue class global by bx group', total=1, visible = False)

                    cohort_output_figures_dir = master_structure_info_dict["Global"]['Cohort figures dir']

                    tissue_class_general_plot_name_string = production_plots_input_dictionary["Cohort - Tissue class global score by biopsy type"]["Plot name"]
                    structure_miss_probability_roi = production_plots_input_dictionary["Cohort - Tissue class global score by biopsy type"]["Structure miss ROI"]


                    production_plots.production_plot_tissue_patient_cohort_NEW_v2(master_cohort_patient_data_and_dataframes,
                                              master_structure_info_dict,
                                              miss_structure_complement_label,
                                              all_ref_key,
                                              color_discrete_map_by_sim_type,
                                            svg_image_scale,
                                            svg_image_width,
                                            svg_image_height,
                                            tissue_class_general_plot_name_string,
                                            cohort_output_figures_dir,
                                            box_plot_points_option,
                                            notch_option,
                                            boxmean_option
                                            )
                    
                    indeterminate_progress_main.update(main_indeterminate_task, visible = False)
                    completed_progress.update(main_indeterminate_task_completed, advance = 1,visible = True)
                    live_display.refresh()    

                else: 
                    pass
                
                print('Cohort - Tissue volume by biopsy type')
                if production_plots_input_dictionary["Cohort - Tissue volume by biopsy type"]["Plot bool"] == True:

                    main_indeterminate_task = indeterminate_progress_main.add_task('[red]Cohort - tissue volume by threshold...', total=None)
                    main_indeterminate_task_completed = completed_progress.add_task('[green]Cohort - tissue volume by threshold', total=1, visible = False)

                    cohort_output_figures_dir = master_structure_info_dict["Global"]['Cohort figures dir']

                    tissue_class_general_plot_name_string = production_plots_input_dictionary["Cohort - Tissue volume by biopsy type"]["Plot name"]
                    #structure_miss_probability_roi = production_plots_input_dictionary["Cohort - Tissue volume by biopsy type"]["Structure miss ROI"]


                    production_plots.production_plot_tissue_volume_box_plots_patient_cohort_NEW(master_cohort_patient_data_and_dataframes,
                                                master_structure_info_dict,
                                                color_discrete_map_by_sim_type,
                                                svg_image_scale,
                                                svg_image_width,
                                                svg_image_height,
                                                tissue_class_general_plot_name_string,
                                                cohort_output_figures_dir,
                                                box_plot_points_option,
                                                notch_option,
                                                boxmean_option
                                            )
                    
                    indeterminate_progress_main.update(main_indeterminate_task, visible = False)
                    completed_progress.update(main_indeterminate_task_completed, advance = 1,visible = True)
                    live_display.refresh()    

                else: 
                    pass


                

                # quantile regression of axial dose distribution NEW

                print("Axial MR distribution quantiles regression plot matplotlib")
                if production_plots_input_dictionary["Axial MR distribution quantiles regression plot matplotlib"]["Plot bool"] == True:
                    
                    general_plot_name_string = production_plots_input_dictionary["Axial MR distribution quantiles regression plot matplotlib"]["Plot name"]
                    
                    patientUID_default = "Initializing"
                    processing_patient_production_plot_description = "Creating axial MR distribution quantile regression plots [{}]...".format(patientUID_default)
                    processing_patients_task = patients_progress.add_task("[red]"+processing_patient_production_plot_description, total = master_structure_info_dict["Global"]["Num cases"])
                    processing_patient_production_plot_description_completed = "Creating axial MR distribution quantile regression plots"
                    processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_production_plot_description_completed, total=master_structure_info_dict["Global"]["Num cases"], visible=False)

                    mr_adc_col_name_str_prefix = "MR ADC"
                    output_mr_adc_dataframe_str = "Point-wise MR ADC output by MC trial number"
                    
                    for patientUID,pydicom_item in master_structure_reference_dict.items():
                        
                        processing_patient_production_plot_description = "Creating axial MR distribution quantile regression plots [{}]...".format(patientUID)
                        patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_production_plot_description)

                        if mr_adc_ref in pydicom_item:
                            production_plots.production_plot_axial_mr_distribution_quantile_regression_by_patient_matplotlib(pydicom_item,
                                                                                    patientUID,
                                                                                    bx_ref,
                                                                                    mr_adc_ref,
                                                                                    patient_sp_output_figures_dir_dict,
                                                                                    general_plot_name_string,
                                                                                    mr_adc_col_name_str_prefix,
                                                                                    output_mr_adc_dataframe_str)

                        
                        
                        patients_progress.update(processing_patients_task, advance = 1)
                        completed_progress.update(processing_patients_completed_task, advance = 1)

                    patients_progress.update(processing_patients_task, visible = False)
                    completed_progress.update(processing_patients_completed_task, visible = True)   
                else:
                    pass
                    









            ############# FANOVA FIGURES






            #live_display.stop()
            if create_at_least_one_production_plot == True and fanova_sim_complete_bool == True:
                patient_sp_fanova_output_figures_dir_dict = master_structure_info_dict["Global"]['Patient specific output figures directory (fanova) dict']
                if fanova_containment_sim_complete_bool == True:
                    if production_plots_input_dictionary["Tissue classification Sobol indices global plot"]["Plot bool"] == True:
                        
                        tissue_class_sobol_global_plot_bool_dict = production_plots_input_dictionary["Tissue classification Sobol indices global plot"]["Plot bool dict"]
                        general_plot_name_string_dict = production_plots_input_dictionary["Tissue classification Sobol indices global plot"]["Plot name dict"]
                        structure_miss_probability_roi = production_plots_input_dictionary["Tissue classification Sobol indices global plot"]["Structure miss ROI"]
                        box_plot_points_option = production_plots_input_dictionary["Tissue classification Sobol indices global plot"]["Box plot points option"]
                        
                        global_sobol_plots_task_indeterminate = indeterminate_progress_main.add_task('[red]Plotting global Sobol indices (containment)...', total=None)
                        global_sobol_plots_task_indeterminate_completed = completed_progress.add_task('[green]Plotting global Sobol indices (containment)', total=1, visible = False)

                        production_plots.production_plot_sobol_indices_global_containment(patient_sp_fanova_output_figures_dir_dict,
                                                master_structure_reference_dict,
                                                master_structure_info_dict,
                                                bx_ref,
                                                dil_ref,
                                                cancer_tissue_label,
                                                svg_image_scale,
                                                svg_image_width,
                                                svg_image_height,
                                                general_plot_name_string_dict,
                                                structure_miss_probability_roi,
                                                tissue_class_sobol_global_plot_bool_dict,
                                                box_plot_points_option,
                                                notch_option,
                                                boxmean_option
                                                )

                        indeterminate_progress_main.update(global_sobol_plots_task_indeterminate, visible = False)
                        completed_progress.update(global_sobol_plots_task_indeterminate_completed, advance = 1,visible = True)
                        live_display.refresh()
                    else:
                        pass
                    
                    
                    if production_plots_input_dictionary["Tissue classification Sobol indices per biopsy plot"]["Plot bool"] == True:


                        tissue_class_sobol_per_biopsy_plot_bool_dict = production_plots_input_dictionary["Tissue classification Sobol indices per biopsy plot"]["Plot bool dict"]
                        general_plot_name_string_dict = production_plots_input_dictionary["Tissue classification Sobol indices per biopsy plot"]["Plot name dict"]

                        global_sobol_plots_task_indeterminate = indeterminate_progress_main.add_task('[red]Plotting per biopsy Sobol indices (containment)...', total=None)
                        global_sobol_plots_task_indeterminate_completed = completed_progress.add_task('[green]Plotting per biopsy Sobol indices (containment)', total=1, visible = False)
                        
                        for patientUID,pydicom_item in master_structure_reference_dict.items():
                            production_plots.production_plot_sobol_indices_each_biopsy_containment(patient_sp_fanova_output_figures_dir_dict,
                                                        patientUID,
                                                        pydicom_item,
                                                        master_structure_info_dict,
                                                        bx_ref,
                                                        svg_image_scale,
                                                        svg_image_width,
                                                        svg_image_height,
                                                        general_plot_name_string_dict,
                                                        tissue_class_sobol_per_biopsy_plot_bool_dict
                                                        )
                        
                        indeterminate_progress_main.update(global_sobol_plots_task_indeterminate, visible = False)
                        completed_progress.update(global_sobol_plots_task_indeterminate_completed, advance = 1,visible = True)
                        live_display.refresh()
                        
                    else:
                        pass

                if fanova_dose_sim_complete_bool == True:
                    if production_plots_input_dictionary["Dosimetry Sobol indices global plot"]["Plot bool"] == True:
                        
                        dose_sobol_global_plot_bool_dict = production_plots_input_dictionary["Dosimetry Sobol indices global plot"]["Plot bool dict"]
                        general_plot_name_string_dict = production_plots_input_dictionary["Dosimetry Sobol indices global plot"]["Plot name dict"]
                        box_plot_points_option = production_plots_input_dictionary["Dosimetry Sobol indices global plot"]["Box plot points option"]
                        
                        global_sobol_plots_task_indeterminate = indeterminate_progress_main.add_task('[red]Plotting global Sobol indices (dose)...', total=None)
                        global_sobol_plots_task_indeterminate_completed = completed_progress.add_task('[green]Plotting global Sobol indices (dose)', total=1, visible = False)

                        production_plots.production_plot_sobol_indices_global_dosimetry(patient_sp_fanova_output_figures_dir_dict,
                                                master_structure_reference_dict,
                                                master_structure_info_dict,
                                                bx_ref,
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


                    if production_plots_input_dictionary["Dosimetry Sobol indices per biopsy plot"]["Plot bool"] == True:

                        dosimetry_sobol_per_biopsy_plot_bool_dict = production_plots_input_dictionary["Dosimetry Sobol indices per biopsy plot"]["Plot bool dict"]
                        general_plot_name_string_dict = production_plots_input_dictionary["Dosimetry Sobol indices per biopsy plot"]["Plot name dict"]

                        global_sobol_plots_task_indeterminate = indeterminate_progress_main.add_task('[red]Plotting per biopsy Sobol indices (dose)...', total=None)
                        global_sobol_plots_task_indeterminate_completed = completed_progress.add_task('[green]Plotting per biopsy Sobol indices (dose)', total=1, visible = False)

                        for patientUID,pydicom_item in master_structure_reference_dict.items():
                            production_plots.production_plot_sobol_indices_each_biopsy_dosimetry(patient_sp_fanova_output_figures_dir_dict,
                                                    patientUID,
                                                    pydicom_item,
                                                    master_structure_info_dict,
                                                    bx_ref,
                                                    svg_image_scale,
                                                    svg_image_width,
                                                    svg_image_height,
                                                    general_plot_name_string_dict,
                                                    dosimetry_sobol_per_biopsy_plot_bool_dict
                                                    )
                    
                        indeterminate_progress_main.update(global_sobol_plots_task_indeterminate, visible = False)
                        completed_progress.update(global_sobol_plots_task_indeterminate_completed, advance = 1,visible = True)
                        live_display.refresh()
                    else:
                        pass



            rich_preambles.section_completed("Production plots", section_start_time, completed_progress, completed_sections_manager)

            live_display.stop()
    sys.exit("> Programme complete.")


def UID_generator(pydicom_obj):
    UID_def = f"{str(pydicom_obj[0x0010,0x0010].value)} ({str(pydicom_obj[0x0010,0x0020].value)})"
    return UID_def


def structure_referencer(data_removals_dict,
                         structure_dcm_dict, 
                         dose_dcm_dict, 
                         plan_dcm_dict,
                         US_dcms_dict,
                         MR_T2_dcms_dict,
                         MR_ADC_dcms_dict,
                         OAR_list,
                         DIL_list,
                         Bx_list,
                         st_ref_list,
                         ds_ref,
                         pln_ref,
                         mr_adc_ref,
                         mr_t2_ref,
                         us_ref,
                         all_ref_key,
                         mr_global_multi_structure_output_dataframe_str,
                         mr_global_by_voxel_multi_structure_output_dataframe_str,
                         bx_sim_locations_dict,
                         fanova_sobol_indices_names_by_index,
                         rectum_list,
                         urethra_list,
                         interp_inter_slice_dist,
                         interp_intra_slice_dist,
                         fraction_prefixes,
                         important_info,
                         live_display
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
    global_num_cases = 0
    global_unique_patient_names_list = []

    for UID, structure_item_path in structure_dcm_dict.items():
        with pydicom.dcmread(structure_item_path, defer_size = '2 MB') as structure_item:      
            
            filtered_OARs = [x for x in structure_item.StructureSetROISequence if any(i.lower() in x.ROIName.lower() for i in OAR_list)]
            OAR_ref = [{"ROI":x.ROIName, 
                        "Ref #":x.ROINumber,
                        "Index number": idx,
                        "Struct type": st_ref_list[1],
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
                        "Maximum pairwise distance": None,
                        "Structure volume": None,
                        "Structure OPEN3D triangle mesh object": None,
                        "Voxel size for structure volume calc": None,
                        "Uncertainty data": None, 
                        "MC data: Generated normal dist random samples arr": None, 
                        "KDtree": None, 
                        "Nearest neighbours objects": []
                        } for idx, x in enumerate(filtered_OARs)]
            
            filtered_DILs = [x for x in structure_item.StructureSetROISequence if any(i.lower() in x.ROIName.lower() for i in DIL_list)]
            DIL_ref = [{"ROI":x.ROIName, 
                        "Ref #":x.ROINumber,
                        "Index number": idx,
                        "Struct type": st_ref_list[2],
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
                        "Maximum pairwise distance": None,
                        "Structure OPEN3D triangle mesh object": None,
                        "Structure volume": None, 
                        "Voxel size for structure volume calc": None,
                        "Uncertainty data": None, 
                        "MC data: Generated normal dist random samples arr": None, 
                        "KDtree": None, 
                        "Nearest neighbours objects": []
                        } for idx, x in enumerate(filtered_DILs)] 

            filtered_rectums = [x for x in structure_item.StructureSetROISequence if any(i.lower() in x.ROIName.lower() for i in rectum_list)]
            rectum_ref = [{"ROI":x.ROIName, 
                        "Ref #":x.ROINumber,
                        "Index number": idx,
                        "Struct type": st_ref_list[3],
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
                        "Maximum pairwise distance": None,
                        "Structure OPEN3D triangle mesh object": None,
                        "Structure volume": None,
                        "Voxel size for structure volume calc": None,
                        "Uncertainty data": None, 
                        "MC data: Generated normal dist random samples arr": None, 
                        "KDtree": None, 
                        "Nearest neighbours objects": []
                        } for idx, x in enumerate(filtered_rectums)]

            filtered_urethras = [x for x in structure_item.StructureSetROISequence if any(i.lower() in x.ROIName.lower() for i in urethra_list)]
            urethra_ref = [{"ROI":x.ROIName, 
                        "Ref #":x.ROINumber,
                        "Index number": idx,
                        "Struct type": st_ref_list[4],
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
                        "Maximum pairwise distance": None,
                        "Structure OPEN3D triangle mesh object": None,
                        "Structure volume": None,
                        "Voxel size for structure volume calc": None,
                        "Uncertainty data": None, 
                        "MC data: Generated normal dist random samples arr": None, 
                        "KDtree": None, 
                        "Nearest neighbours objects": []
                        } for idx, x in enumerate(filtered_urethras)]
            
            filtered_BXs = [x for x in structure_item.StructureSetROISequence if any(i.lower() in x.ROIName.lower() for i in Bx_list)]

            ### Remove unwanted data
            if UID in data_removals_dict.keys():
                for bx_id_to_remove in data_removals_dict[UID]:
                    for bpsy in filtered_BXs:
                        if bpsy.ROIName == bx_id_to_remove:
                            filtered_BXs.remove(bpsy)
                            important_info.add_text_line(f"Removed data-point (Pt: {UID}, Bx: {bx_id_to_remove})) ", live_display)


            bpsy_ref = [{"ROI": x.ROIName, 
                         "Ref #": x.ROINumber,
                         "Index number": idx, 
                         "Struct type": st_ref_list[0],
                         "Simulated bool": False,
                         "Simulated type": 'Real',
                         "Reconstructed biopsy cylinder length (from contour data)": None, 
                         "Raw contour pts zslice list": None,
                         "Raw contour pts": None, 
                         "Centroid variation arr": None,
                         "Mean centroid variation": None,
                         "Maximum projected distance between original centroids": None,
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
                         "Maximum pairwise distance": None,
                         "Structure volume": None, 
                         "Voxel size for structure volume calc": None,
                         "Target DIL dict": None,
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
                         "MC data: compiled sim results dataframe": None,
                         "MC data: compiled sim sum-to-one results dataframe": None,
                         "MC data: mutual compiled sim results dataframe": None,
                         "MC data: compiled sim results": None, 
                         "MC data: mutual compiled sim results": None,
                         "MC data: tumor tissue probability": None,
                         "MC data: miss structure tissue probability": None,
                         "MC data: tissue length above threshold dict": None,
                         "MC data: voxelized containment results dict": None, 
                         "MC data: voxelized containment results dict (dict of lists)": None, 
                         "MC data: bx to dose NN search objects list": None, 
                         #"MC data: Dose NN child obj for each sampled bx pt list (nominal & all MC trials)": None,
                         "MC data: Dose vals for each sampled bx pt arr (nominal & all MC trials)": None,
                         #"MC data: Dose vals for each sampled bx pt arr (all MC trials)": None,
                         #"MC data: Dose vals for each sampled bx pt arr (nominal)": None,
                         "MC data: Dose gradient vals for each sampled bx pt arr (nominal & all MC trials)": None,
                         "MC data: Differential DVH dict": None,
                         "MC data: Cumulative DVH dict": None,
                         "MC data: dose volume metrics dict": None, 
                         "MC data: Dose statistics for each sampled bx pt list (mean, std, quantiles)": None, 
                         "MC data: Dose statistics (MLE) for each sampled bx pt list (mean, std)": None, 
                         "MC data: voxelized dose results list": None, 
                         "MC data: voxelized dose results dict (dict of lists)": None, 
                         "FANOVA: sobol indices (containment)": None,
                         "FANOVA: sobol indices (dose)": None,
                         'FANOVA: sobol indices (DIL tissue)': None,
                         "Output csv file paths dict": {}, 
                         "Output data frames": {"Dose output Z and radius": None,
                                                "Dose output voxelized": None,
                                                "Point-wise dose output by MC trial number": None,
                                                "Voxel-wise dose output by MC trial number": None,
                                                "Mutual containment output by bx point": None,
                                                "Tissue volume above threshold": None,
                                                "DVH metrics": None,
                                                "Differential DVH by MC trial": None,
                                                "Cumulative DVH by MC trial": None},
                         "Output dicts for data frames": {},  
                         "KDtree": None, 
                         "Nearest neighbours objects": []
                         } for idx, x in enumerate(filtered_BXs)]


            


            bpsy_ref_index_start = len(bpsy_ref)
            sim_bpsy_ref_index_start = 0
            bpsy_ref_simulated_total = []
            for bx_sim_type_str, bx_sim_type_dict in bx_sim_locations_dict.items():
                if bx_sim_type_dict["Create"] == True and str(structure_item[0x0010,0x0020].value) == 'F2':
                    sim_bx_relative_to = bx_sim_type_dict["Relative to"]
                    bx_sim_ref_identifier_str = bx_sim_type_dict["Identifier string"]
                    
                    filtered_sim_BXs = [x for x in structure_item.StructureSetROISequence if sim_bx_relative_to.lower() in x.ROIName.lower()]
                    bpsy_ref_simulated = [{"ROI": "Bx_Tr_"+bx_sim_ref_identifier_str+" " + x.ROIName, 
                                "Ref #": bx_sim_ref_identifier_str +" "+ x.ROIName,
                                "Index number": bpsy_ref_index_start + sim_bpsy_ref_index_start + idx,
                                "Struct type": st_ref_list[0],
                                "Simulated bool": True,
                                "Simulated type": bx_sim_type_str,
                                "Relative structure type": bx_sim_type_dict["Relative to struct type"],
                                "Relative structure name": x.ROIName,
                                "Relative structure ref #": x.ROINumber, 
                                "Reconstructed biopsy cylinder length (from contour data)": None, 
                                "Raw contour pts zslice list": None,
                                "Raw contour pts": None, 
                                "Centroid variation arr": None,
                                "Mean centroid variation": None,
                                "Maximum projected distance between original centroids": None,
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
                                "Maximum pairwise distance": None,
                                "Structure volume": None, 
                                "Voxel size for structure volume calc": None, 
                                "Target DIL dict": None,
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
                                "MC data: compiled sim results dataframe": None,
                                "MC data: compiled sim sum-to-one results dataframe": None,
                                "MC data: mutual compiled sim results dataframe": None,
                                "MC data: compiled sim results": None,
                                "MC data: mutual compiled sim results": None, 
                                "MC data: tumor tissue probability": None,
                                "MC data: miss structure tissue probability": None,
                                "MC data: tissue length above threshold dict": None,
                                "MC data: voxelized containment results dict": None, 
                                "MC data: voxelized containment results dict (dict of lists)": None, 
                                "MC data: bx to dose NN search objects list": None, 
                                #"MC data: Dose NN child obj for each sampled bx pt list (nominal & all MC trials)": None,
                                "MC data: Dose vals for each sampled bx pt arr (nominal & all MC trials)": None,
                                #"MC data: Dose vals for each sampled bx pt arr (all MC trials)": None,
                                #"MC data: Dose vals for each sampled bx pt arr (nominal)": None,
                                "MC data: Dose gradient vals for each sampled bx pt arr (nominal & all MC trials)": None,
                                "MC data: Differential DVH dict": None,
                                "MC data: Cumulative DVH dict": None,
                                "MC data: dose volume metrics dict": None,
                                "MC data: Dose statistics for each sampled bx pt list (mean, std, quantiles)": None, 
                                "MC data: Dose statistics (MLE) for each sampled bx pt list (mean, std)": None, 
                                "MC data: voxelized dose results list": None, 
                                "MC data: voxelized dose results dict (dict of lists)": None, 
                                "FANOVA: sobol indices (containment)": None,
                                "FANOVA: sobol indices (dose)": None,
                                'FANOVA: sobol indices (DIL tissue)': None,
                                "Output csv file paths dict": {}, 
                                "Output data frames": {"Dose output Z and radius": None,
                                                       "Dose output voxelized": None,
                                                       "Point-wise dose output by MC trial number": None,
                                                       "Voxel-wise dose output by MC trial number": None,
                                                       "Mutual containment output by bx point": None,
                                                       "Tissue volume above threshold": None,
                                                       "DVH metrics": None,
                                                       "Differential DVH by MC trial": None},
                                "Output dicts for data frames": {}, 
                                "KDtree": None, 
                                "Nearest neighbours objects": []
                                } for idx, x in enumerate(filtered_sim_BXs)]

                    sim_bpsy_ref_index_start = len(bpsy_ref_simulated)

                    bpsy_ref_simulated_total = bpsy_ref_simulated_total + bpsy_ref_simulated
                else:
                    pass
                
                
            
            bpsy_ref = bpsy_ref + bpsy_ref_simulated_total 

            """
            ## for each reference type, store each item's index number
            for index, item in enumerate(bpsy_ref):
                item["Index number"] = index
            for index, item in enumerate(DIL_ref):
                item["Index number"] = index
            for index, item in enumerate(OAR_ref):
                item["Index number"] = index
            """

            # Note that all of the dataframes in the below "Multi-structure output data frames dict" are output as csvs in the final report
            all_ref = {"Multi-structure information dict (not for csv output)": {"Biopsy optimization: Optimal biopsy location (entire cubic lattice) dataframe": None, # THIS DATAFRAME WAS DELETED AFTER OPTIMIZATION TO SAVE MEMORY!
                                                                                  },
                        "Multi-structure pre-processing output dataframes dict": {"Selected structures": None,
                                                                                  "Biopsy basic spatial features dataframe": None,
                                                                                  #"Structure information dimension": pandas.DataFrame(),
                                                                                  #"Structure information (Non-BX)": pandas.DataFrame(),
                                                                                  "Nearest DILs info dataframe": None,
                                                                                  "Biopsy optimization - Cumulative projection (all points within prostate) dataframe": None,
                                                                                  "Biopsy optimization - DIL centroids optimal targeting dataframe": None,
                                                                                  "Biopsy optimization - Optimal DIL targeting dataframe": None,
                                                                                  "Biopsy optimization - Optimal DIL targeting entire lattice dataframe": None},    
                        "Multi-structure MC simulation output dataframes dict": {"Tissue class - Global tissue class statistics": None,
                                                                                 "Tissue class - Global tissue by structure statistics": None,
                                                                                 "Tissue class - Tissue length above threshold": None,
                                                                                 "Tissue class - sum-to-one mc results": None,
                                                                                 #"Dosimetry - All points and trials": pandas.DataFrame(),
                                                                                 "DVH metrics": None,
                                                                                 "DVH metrics (Dx, Vx) statistics": None,
                                                                                 "MR - " + str(mr_global_multi_structure_output_dataframe_str): None,
                                                                                 "MR - " + str(mr_global_by_voxel_multi_structure_output_dataframe_str): None},
                        "Multi-structure Fanova output dataframes dict": {}                                             
                        }
            
            

            # Dictionary comprehension to count occurrences of each type of biopsy
            bpsy_type_counts_dict = {item["Simulated type"]: sum(1 for d in bpsy_ref if d["Simulated type"] == item["Simulated type"]) for item in bpsy_ref}

            bpsy_info = {"Num structs": len(bpsy_ref), 
                         "Num sim structs": len(bpsy_ref_simulated_total), 
                         "Num real structs": len(bpsy_ref) - len(bpsy_ref_simulated_total),
                         "Biopsy type counts": bpsy_type_counts_dict}
            OAR_info = {"Num structs": len(OAR_ref)}
            DIL_info = {"Num structs": len(DIL_ref)}
            rectum_info = {"Num structs": len(rectum_ref)}
            urethra_info = {"Num structs": len(urethra_ref)}
            patient_total_num_structs = bpsy_info["Num structs"] + OAR_info["Num structs"] + DIL_info["Num structs"] + rectum_info["Num structs"] + urethra_info["Num structs"]
            all_structs_info = {"Total num structs": patient_total_num_structs}
            
            global_num_OAR = global_num_OAR + OAR_info["Num structs"]
            global_num_DIL = global_num_DIL + DIL_info["Num structs"] 
            global_num_biopsies = global_num_biopsies + bpsy_info["Num structs"]
            global_total_num_structs = global_total_num_structs + patient_total_num_structs
            global_num_cases = global_num_cases + 1
            if str(structure_item[0x0010,0x0010].value) not in global_unique_patient_names_list:
                global_unique_patient_names_list.append(str(structure_item[0x0010,0x0010].value))

            master_st_ds_ref_dict[UID] = {"Patient UID (generated)": str(UID),
                                            "Patient ID (from dicom)": str(structure_item[0x0010,0x0020].value),
                                            "Patient Name": str(structure_item[0x0010,0x0010].value),
                                            "Fraction number": misc_tools.extract_number_from_string(str(structure_item[0x0010,0x0020].value), fraction_prefixes),
                                            st_ref_list[0]: bpsy_ref, 
                                            st_ref_list[1]: OAR_ref, 
                                            st_ref_list[2]: DIL_ref,
                                            st_ref_list[3]: rectum_ref,
                                            st_ref_list[4]: urethra_ref,
                                            all_ref_key: all_ref,
                                            "Ready to plot data list": None
                                        }
            
            master_st_ds_info_dict[UID] = {"Patient UID (generated)": str(UID),
                                            "Patient ID (from dicom)": str(structure_item[0x0010,0x0020].value),
                                            "Patient Name": str(structure_item[0x0010,0x0010].value),
                                            "Fraction number": misc_tools.extract_number_from_string(str(structure_item[0x0010,0x0020].value), fraction_prefixes),
                                            st_ref_list[0]: bpsy_info, 
                                            st_ref_list[1]: OAR_info, 
                                            st_ref_list[2]: DIL_info, 
                                            st_ref_list[3]: rectum_info,
                                            st_ref_list[4]: urethra_info,
                                            all_ref_key: all_structs_info
                                        }
    
    ### Dosimetry
    for UID, dose_item_path in dose_dcm_dict.items():
        if master_st_ds_ref_dict[UID]["Fraction number"] == 1:
            continue
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
                             #"Dose phys space and pixel 3d arr": None,
                             "Dose and gradient phys space and pixel 3d arr": None, 
                             "Dose grid point cloud": None, 
                             "Dose grid point cloud thresholded": None,
                             "Dose grid gradient point cloud": None,
                             "Dose grid gradient point cloud thresholded": None,
                             "KDtree": None,
                             "KDtree gradient": None
                             }
            master_st_ds_ref_dict[UID][ds_ref] = dose_ref_dict
    

    ### MR ADC
    for UID, mr_adc_item_paths_list in MR_ADC_dcms_dict.items():
        mr_adc_ref_dict = {}
        for mr_adc_item_path in mr_adc_item_paths_list:
            with pydicom.dcmread(mr_adc_item_path, defer_size = '2 MB') as mr_adc_item:
                seriesinstanceUID = mr_adc_item.SeriesInstanceUID
                mr_adc_ID = UID + mr_adc_item.StudyDate
                if len(mr_adc_item.RealWorldValueMappingSequence) > 1:
                    important_info.add_text_line(f"Multiple real world value mappings detected for ({UID}, {mr_adc_ID})) ", live_display)
                if seriesinstanceUID not in mr_adc_ref_dict: 
                    mr_adc_ref_subdict = {"MR ADC ID": mr_adc_ID,
                                    "Series instance UID": seriesinstanceUID, 
                                    "Study date": mr_adc_item.StudyDate, 
                                    "Pixel arr (all slices)": mr_adc_item.pixel_array, 
                                    "Pixel spacing": np.array(mr_adc_item.PixelSpacing),
                                    "Units": str(mr_adc_item.RealWorldValueMappingSequence[0].MeasurementUnitsCodeSequence[0].CodeMeaning),
                                    "RWVSlope (all slices)": np.array(mr_adc_item.RealWorldValueMappingSequence[0].RealWorldValueSlope),
                                    "RWVIntercept (all slices)": np.array(mr_adc_item.RealWorldValueMappingSequence[0].RealWorldValueIntercept),
                                    "RWV Units": mr_adc_item.RealWorldValueMappingSequence[0].LUTLabel,
                                    "Slice thickness":  mr_adc_item.SliceThickness,
                                    "Image orientation patient": np.array(mr_adc_item.ImageOrientationPatient), 
                                    "Image position patient (all slices)": np.array(mr_adc_item.ImagePositionPatient), 
                                    "MR ADC phys space Nx4 arr": None,
                                    "MR ADC phys space Nx4 arr (filtered, non-negative)": None,
                                    "MR ADC grid point cloud": None,
                                    "MR ADC grid point cloud thresholded": None, 
                                    "KDtree": None
                                    }
                    mr_adc_ref_dict[seriesinstanceUID] = mr_adc_ref_subdict
                else:
                    mr_adc_ref_subdict = mr_adc_ref_dict[seriesinstanceUID]
                    mr_adc_ref_subdict["Pixel arr (all slices)"] = np.dstack((mr_adc_ref_subdict["Pixel arr (all slices)"], mr_adc_item.pixel_array))
                    mr_adc_ref_subdict["RWVSlope (all slices)"] = np.hstack((mr_adc_ref_subdict["RWVSlope (all slices)"], np.array(mr_adc_item.RealWorldValueMappingSequence[0].RealWorldValueSlope)))
                    mr_adc_ref_subdict["RWVIntercept (all slices)"] = np.hstack((mr_adc_ref_subdict["RWVIntercept (all slices)"], np.array(mr_adc_item.RealWorldValueMappingSequence[0].RealWorldValueIntercept)))
                    mr_adc_ref_subdict["Image position patient (all slices)"] = np.vstack((mr_adc_ref_subdict["Image position patient (all slices)"], np.array(mr_adc_item.ImagePositionPatient)))
                    mr_adc_ref_dict[seriesinstanceUID] = mr_adc_ref_subdict

        master_st_ds_ref_dict[UID][mr_adc_ref] = mr_adc_ref_dict


    
    

    ### Plan file
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

    preprocessing_info = {"Interslice interp dist": interp_inter_slice_dist,
                          "Intraslice interp dist": interp_intra_slice_dist,
                          "Preprocessing performed": False,}

    mc_info = {"Num MC containment simulations": None, 
               "Num MC dose simulations": None,
               "Num MC MR simulations": None,
               "Num sample pts per BX core": None, 
               "BX sample pt lattice spacing (mm)": None,
               "BX sample pt volume element (mm^3)": None,
               "Max of num MC simulations": None,
               'MC sim performed': False,
               'MC containment sim performed': False,
               'MC dose sim performed': False,
               'MC MR sim performed': False,
               }
    
    fanova_info = {"Num fanova containment simulations": None, 
                   "Num fanova dose simulations": None,
                   "FANOVA: num variance vars": None,
                   "FANOVA: sobol var names by index": fanova_sobol_indices_names_by_index,
                   'FANOVA sim performed': False,
                   'FANOVA containment sim performed': False,
                   'FANOVA dose sim performed': False,  
                    }

    # count number of biopsies of each type for the entire cohort
    list_of_bpsy_nums_by_bpsy_type_all_patients = [pt_sp_dict[st_ref_list[0]]["Biopsy type counts"] for pt_sp_dict in master_st_ds_info_dict.values()]
    global_num_biopsies_by_type = {key: sum(d[key] for d in list_of_bpsy_nums_by_bpsy_type_all_patients if key in d) for d in list_of_bpsy_nums_by_bpsy_type_all_patients for key in d}              
    
    # Determine the types of biopsies made
    bx_types_list = ['Real'] + [key for key, value in bx_sim_locations_dict.items() if value.get("Create", False)] 

    master_st_ds_info_global_dict["Global"] = {"Num cases": global_num_cases, # This is a mixture of patients and fractions, ie. 181 F1 and 181 F2 are two different cases
                                               "Num unique patient names": len(global_unique_patient_names_list),
                                               "Num structures": global_total_num_structs, 
                                               "Num biopsies": global_num_biopsies,
                                               "Num biopsies by bx type dict": global_num_biopsies_by_type, 
                                               "Num DILs": global_num_DIL,
                                               "Bx types list": bx_types_list, 
                                               "Preprocessing info": preprocessing_info,
                                               "MC info": mc_info, 
                                               "FANOVA info": fanova_info,
                                               'Patient specific output figures directory (pre-processing) dict': None,                        
                                               'Patient specific output figures directory (MC sim) dict': None,
                                               'Patient specific output figures directory (fanova) dict': None,
                                               "Cohort figures dir": None,
                                               "Specific output dir": None,
                                               "Mean biopsy centroid variation": None
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
        self.uncertainty_data_info_dict = {"Frame of reference": frame_of_reference, "Distribution": 'Normal'} 
    def fill_means_and_sigmas(self, means_arr, sigmas_arr):
        self.uncertainty_data_mean_arr = means_arr
        self.uncertainty_data_sigma_arr = sigmas_arr


"""
class plot_attributes:
    def __init__(self,plot_bool_init = True):
        self.plot_bool = plot_bool_init
        self.color_raw = 'r'
        self.color_best_fit = 'g' 
"""

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
            
            threeDdata_zslice_interpolated_arr = np.asarray(threeDdata_zslice_interpolated_list)
            self.interpolated_pts_list.append(threeDdata_zslice_interpolated_arr)
        self.interpolated_pts_np_arr = np.vstack(self.interpolated_pts_list)


    
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
            
            threeDdata_zslice_interpolated_arr = np.asarray(threeDdata_zslice_interpolated_list)
            self.interpolated_pts_list.append(threeDdata_zslice_interpolated_arr)
        self.interpolated_pts_np_arr = np.vstack(self.interpolated_pts_list)

                
    
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
        self.interpolated_pts_with_end_caps_np_arr = np.vstack(self.interpolated_pts_with_end_caps_list)



 





if __name__ == '__main__':    
    main()
    